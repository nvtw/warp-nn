# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from tests.runtime.test_qwen3_encoder import _tiny_checkpoint
from warp_nn.runtime.flux2 import flux2_klein_schedule
from warp_nn.runtime.flux2.transformer import Flux2KleinTransformerPlan
from warp_nn.runtime.qwen.encoder import QwenEncoder
from warp_nn.runtime.operators import (
    FusedSwiGLUPlan,
    SpatialGroupNormPlan,
    warp_to_numpy_float32,
)


def test_klein_schedule_matches_official_exponential_shift():
    values = flux2_klein_schedule(4, 4096)
    assert values.shape == (5,)
    assert values[0] == 1.0 and values[-1] == 0.0
    assert np.all(np.diff(values) < 0.0)
    mu = (0.00016927 * 4096 + 0.45666666) + (4 - 200) * (
        (0.00016927 * 4096 + 0.45666666) - (8.73809524e-05 * 4096 + 1.89833333)
    ) / 190.0
    expected = 3.0 * np.exp(mu) / (3.0 * np.exp(mu) + 1.0)
    np.testing.assert_allclose(values[1], expected, rtol=1e-6)


def test_shared_spatial_group_norm_matches_numpy():
    rng = np.random.default_rng(391)
    source = rng.normal(size=(2, 16, 16, 8)).astype(np.float32)
    weight = rng.normal(size=8).astype(np.float32)
    bias = rng.normal(size=8).astype(np.float32)
    plan = SpatialGroupNormPlan(
        wp.array(source, device="cpu"),
        wp.array(weight, device="cpu"),
        wp.array(bias, device="cpu"),
        groups=4,
        silu=True,
    )
    grouped = source.reshape(2, 16, 16, 4, 2)
    mean = grouped.mean(axis=(1, 2, 4), keepdims=True)
    variance = grouped.var(axis=(1, 2, 4), keepdims=True)
    normalized = ((grouped - mean) / np.sqrt(variance + 1e-6)).reshape(source.shape)
    affine = normalized * weight + bias
    expected = affine / (1.0 + np.exp(-affine))
    np.testing.assert_allclose(plan.execute().numpy(), expected, atol=2e-5)


def test_shared_fused_swiglu_matches_numpy():
    rng = np.random.default_rng(392)
    values = rng.normal(size=(1, 5, 12)).astype(np.float32)
    gate, up = np.split(values, 2, axis=-1)
    expected = gate / (1.0 + np.exp(-gate)) * up
    plan = FusedSwiGLUPlan(wp.array(values, device="cpu"))
    np.testing.assert_allclose(plan.execute().numpy(), expected, atol=2e-6)


def test_shared_bfloat16_numpy_conversion():
    source = wp.array([1.0, -2.125, 0.5], dtype=wp.bfloat16, device="cpu")
    values = warp_to_numpy_float32(source)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values, [1.0, -2.125, 0.5])


def test_qwen_intermediate_states_match_pre_norm_hidden(tmp_path):
    model = tmp_path / "qwen"
    _tiny_checkpoint(model)
    encoder = QwenEncoder(model, dtype=wp.float16, device="cpu", use_cublas=False)
    ids = [3, 9, 4]
    expected = encoder.encode_ids(ids, final_normalize=False).numpy().copy()
    actual = encoder.encode_intermediate_ids(ids, (1,))[0].numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-3)


def test_qwen_intermediate_states_mask_right_padding(tmp_path):
    model = tmp_path / "qwen"
    _tiny_checkpoint(model)
    encoder = QwenEncoder(model, dtype=wp.float16, device="cpu", use_cublas=False)
    ids = [3, 9, 0, 0]
    masked = (
        encoder.encode_intermediate_ids(ids, (1,), valid_tokens=2)[0].numpy().copy()
    )
    unmasked = encoder.encode_intermediate_ids(ids, (1,))[0].numpy().copy()
    np.testing.assert_allclose(masked[:, :2], unmasked[:, :2], atol=2e-3)
    assert not np.allclose(masked[:, 2:], unmasked[:, 2:])


def test_small_flux_transformer_plan_executes_and_replays():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    device = "cuda:0"
    dtype = wp.bfloat16
    rng = np.random.default_rng(393)
    weights = {}

    def put(name, shape, *, norm=False):
        values = (
            np.ones(shape, dtype=np.float32)
            if norm
            else rng.normal(0.0, 0.01, shape).astype(np.float32)
        )
        weights[name] = wp.array(values, dtype=dtype, device=device)

    put("time_guidance_embed.timestep_embedder.linear_1.weight", (32, 256))
    put("time_guidance_embed.timestep_embedder.linear_2.weight", (32, 32))
    put("double_stream_modulation_img.linear.weight", (192, 32))
    put("double_stream_modulation_txt.linear.weight", (192, 32))
    put("single_stream_modulation.linear.weight", (96, 32))
    put("x_embedder.weight", (32, 128))
    put("context_embedder.weight", (32, 7680))
    put("norm_out.linear.weight", (64, 32))
    put("proj_out.weight", (128, 32))
    prefix = "transformer_blocks.0"
    for name in (
        "to_q",
        "to_k",
        "to_v",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_out.0",
        "to_add_out",
    ):
        put(f"{prefix}.attn.{name}.weight", (32, 32))
    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        put(f"{prefix}.attn.{name}.weight", (32,), norm=True)
    for branch in ("ff", "ff_context"):
        put(f"{prefix}.{branch}.linear_in.weight", (192, 32))
        put(f"{prefix}.{branch}.linear_out.weight", (32, 96))
    for index in range(2):
        prefix = f"single_transformer_blocks.{index}.attn"
        put(f"{prefix}.to_qkv_mlp_proj.weight", (288, 32))
        put(f"{prefix}.to_out.weight", (32, 128))
        for name in ("norm_q", "norm_k"):
            put(f"{prefix}.{name}.weight", (32,), norm=True)
    image = wp.array(
        rng.normal(0.0, 0.1, (1, 1, 128)).astype(np.float32), dtype=dtype, device=device
    )
    text = wp.zeros((1, 512, 7680), dtype=dtype, device=device)
    sigma = wp.array(np.array([1.0], dtype=np.float32), device=device)
    config = {
        "num_attention_heads": 1,
        "attention_head_dim": 32,
        "axes_dims_rope": [8, 8, 8, 8],
        "rope_theta": 2000,
        "num_layers": 1,
        "num_single_layers": 2,
    }
    plan = Flux2KleinTransformerPlan(image, text, sigma, weights, config, 1, 1)
    assert plan.single[0].proj.output.ptr == plan.single[1].proj.output.ptr
    assert plan.single[0].output.ptr != plan.single[1].output.ptr
    first = plan.replay().numpy()
    second = plan.replay().numpy()
    assert first.shape == (1, 1, 128)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, atol=1e-4)
