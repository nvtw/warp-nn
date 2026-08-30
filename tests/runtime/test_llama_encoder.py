# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.llama_encoder import _LlamaEncoderPlan
from warp_nn.runtime.rope import rotary_cache_values


def _runner(device="cpu", dtype=wp.float16):
    hidden, intermediate, layers = 8, 16, 1
    rng = np.random.default_rng(17)
    weights = {
        "model.embed_tokens.weight": wp.array(
            rng.normal(0, 0.1, (32, hidden)).astype(np.float16),
            dtype=dtype,
            device=device,
        ),
        "model.norm.weight": wp.ones((hidden,), dtype=dtype, device=device),
    }
    for index in range(layers):
        prefix = f"model.layers.{index}"
        weights[f"{prefix}.input_layernorm.weight"] = wp.ones(
            (hidden,), dtype=dtype, device=device
        )
        weights[f"{prefix}.post_attention_layernorm.weight"] = wp.ones(
            (hidden,), dtype=dtype, device=device
        )
        for name, out_width, in_width in (
            ("self_attn.q_proj", hidden, hidden),
            ("self_attn.k_proj", hidden // 2, hidden),
            ("self_attn.v_proj", hidden // 2, hidden),
            ("self_attn.o_proj", hidden, hidden),
            ("mlp.gate_proj", intermediate, hidden),
            ("mlp.up_proj", intermediate, hidden),
            ("mlp.down_proj", hidden, intermediate),
        ):
            weights[f"{prefix}.{name}.weight"] = wp.array(
                rng.normal(0, 0.1, (out_width, in_width)).astype(np.float16),
                dtype=dtype,
                device=device,
            )
    cos, sin = rotary_cache_values(16, 4, {"rope_theta": 10000.0})
    return SimpleNamespace(
        device=wp.get_device(device),
        dtype=dtype,
        hidden_size=hidden,
        layers=layers,
        query_heads=2,
        kv_heads=1,
        head_size=4,
        epsilon=1.0e-5,
        cublas=None,
        weights=weights,
        cos_cache=wp.array(cos, dtype=dtype, device=device),
        sin_cache=wp.array(sin, dtype=dtype, device=device),
    )


def test_tiny_bidirectional_llama_encoder_is_finite_and_reusable():
    plan = _LlamaEncoderPlan(_runner(), 4)
    plan.input_ids.assign(np.array([[1, 2, 3, 4]], dtype=np.int64))
    plan.valid.assign(np.array([[True, True, True, True]]))
    plan.embed_mask.assign(np.array([False, True, True, True]))
    pointers = (plan.output.ptr, plan.embedding.ptr)
    first = plan.execute().numpy().copy()
    second = plan.execute().numpy().copy()
    assert first.shape == (1, 8) and np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert pointers == (plan.output.ptr, plan.embedding.ptr)


@pytest.mark.skipif(not wp.get_cuda_devices(), reason="CUDA is unavailable")
def test_tiny_bidirectional_llama_cuda_graph_replay():
    device = wp.get_cuda_devices()[0]
    plan = _LlamaEncoderPlan(_runner(device, wp.bfloat16), 4)
    plan.valid.assign(np.ones((1, 4), dtype=bool))
    plan.embed_mask.assign(np.array([False, True, True, True]))
    plan.input_ids.assign(np.array([[1, 2, 3, 4]], dtype=np.int64))
    first = plan.run().numpy().copy()
    plan.input_ids.assign(np.array([[4, 3, 2, 1]], dtype=np.int64))
    plan.run()
    second = plan.run().numpy().copy()
    assert plan._graph is not None
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert not np.array_equal(first, second)
