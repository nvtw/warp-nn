# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import warp as wp

from tests.runtime.test_qwen3_encoder import _write_safetensors
from tests.utilities import is_device_available, local_model_root
from warp_nn.runtime.ace_step.conditioning import (
    AceStepConditionEncoder,
    condition_weight_names,
)
from warp_nn.runtime.ace_step.dit import AceStepDiTConfig
from warp_nn.runtime.ace_step.runner import load_silence_latent


def _config():
    return AceStepDiTConfig.from_dict(
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "hidden_act": "silu",
            "in_channels": 12,
            "audio_acoustic_hidden_dim": 4,
            "patch_size": 2,
            "encoder_hidden_size": 16,
            "layer_types": ["sliding_attention", "full_attention"] * 4,
            "use_sliding_window": True,
            "sliding_window": 2,
            "rope_theta": 1_000_000,
            "rms_norm_eps": 1.0e-6,
            "attention_bias": False,
            "model_version": "turbo",
            "is_turbo": True,
            "text_hidden_dim": 1024,
            "timbre_hidden_dim": 4,
            "num_lyric_encoder_hidden_layers": 2,
            "num_timbre_encoder_hidden_layers": 2,
            "timbre_fix_frame": 4,
        }
    )


def _layer_weights(weights, base, count):
    for index in range(count):
        prefix = f"{base}.layers.{index}."
        weights.update(
            {
                prefix + "input_layernorm.weight": np.ones(16, dtype=np.float32),
                prefix + "post_attention_layernorm.weight": np.ones(
                    16, dtype=np.float32
                ),
                prefix + "self_attn.q_proj.weight": np.zeros(
                    (16, 16), dtype=np.float32
                ),
                prefix + "self_attn.k_proj.weight": np.zeros((8, 16), dtype=np.float32),
                prefix + "self_attn.v_proj.weight": np.zeros((8, 16), dtype=np.float32),
                prefix + "self_attn.q_norm.weight": np.ones(4, dtype=np.float32),
                prefix + "self_attn.k_norm.weight": np.ones(4, dtype=np.float32),
                prefix + "self_attn.o_proj.weight": np.zeros(
                    (16, 16), dtype=np.float32
                ),
                prefix + "mlp.gate_proj.weight": np.zeros((32, 16), dtype=np.float32),
                prefix + "mlp.up_proj.weight": np.zeros((32, 16), dtype=np.float32),
                prefix + "mlp.down_proj.weight": np.zeros((16, 32), dtype=np.float32),
            }
        )


def _weights(config, rng):
    weights = {
        "encoder.text_projector.weight": rng.normal(0, 0.1, (16, 1024)).astype(
            np.float32
        ),
        "encoder.lyric_encoder.embed_tokens.weight": rng.normal(
            0, 0.1, (16, 1024)
        ).astype(np.float32),
        "encoder.lyric_encoder.embed_tokens.bias": rng.normal(0, 0.1, 16).astype(
            np.float32
        ),
        "encoder.lyric_encoder.norm.weight": np.ones(16, dtype=np.float32),
        "encoder.timbre_encoder.embed_tokens.weight": rng.normal(
            0, 0.1, (16, 4)
        ).astype(np.float32),
        "encoder.timbre_encoder.embed_tokens.bias": rng.normal(0, 0.1, 16).astype(
            np.float32
        ),
        "encoder.timbre_encoder.norm.weight": np.ones(16, dtype=np.float32),
        "encoder.timbre_encoder.special_token": rng.normal(0, 0.1, (1, 1, 16)).astype(
            np.float32
        ),
        "null_condition_emb": rng.normal(0, 0.1, (1, 1, 16)).astype(np.float32),
    }
    _layer_weights(
        weights, "encoder.lyric_encoder", config.num_lyric_encoder_hidden_layers
    )
    _layer_weights(
        weights, "encoder.timbre_encoder", config.num_timbre_encoder_hidden_layers
    )
    assert set(weights) == set(condition_weight_names(config))
    return weights


def test_condition_encoder_exact_timbre_pool_and_stable_pack(tmp_path):
    rng = np.random.default_rng(211)
    config = _config()
    weights = _weights(config, rng)
    path = tmp_path / "dit"
    path.mkdir()
    _write_safetensors(path / "model.safetensors", weights)
    encoder = AceStepConditionEncoder(
        path, config, dtype=wp.float16, device="cpu", use_cublas=False
    )
    text = rng.normal(0, 0.2, (1, 2, 1024)).astype(np.float32)
    lyric = rng.normal(0, 0.2, (1, 3, 1024)).astype(np.float32)
    reference = rng.normal(0, 0.2, (1, 4, 4)).astype(np.float32)
    plan = encoder.plan(
        wp.array(text, dtype=wp.float16, device="cpu"),
        wp.array([[True, True]], device="cpu"),
        wp.array(lyric, dtype=wp.float16, device="cpu"),
        wp.array([[True, False, True]], device="cpu"),
        wp.array(reference, dtype=wp.float16, device="cpu"),
    )
    actual, valid = plan.execute()
    projected_lyric = lyric @ weights["encoder.lyric_encoder.embed_tokens.weight"].T
    projected_lyric += weights["encoder.lyric_encoder.embed_tokens.bias"]
    projected_lyric /= np.sqrt(
        np.mean(projected_lyric**2, axis=-1, keepdims=True) + 1.0e-6
    )
    projected_text = text @ weights["encoder.text_projector.weight"].T
    projected_timbre = (
        reference @ weights["encoder.timbre_encoder.embed_tokens.weight"].T
    )
    projected_timbre += weights["encoder.timbre_encoder.embed_tokens.bias"]
    projected_timbre /= np.sqrt(
        np.mean(projected_timbre**2, axis=-1, keepdims=True) + 1.0e-6
    )
    expected = np.concatenate(
        (
            projected_lyric[:, [0, 2]],
            projected_timbre[:, :1],
            projected_text,
            projected_lyric[:, [1]],
        ),
        axis=1,
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0.02, atol=0.02)
    np.testing.assert_array_equal(
        valid.numpy(), [[True, True, True, True, True, False]]
    )
    null_buffer = plan.null_condition()
    null = null_buffer.numpy()
    np.testing.assert_allclose(
        null,
        np.broadcast_to(weights["null_condition_emb"], null.shape),
        rtol=5e-4,
        atol=5e-4,
    )
    assert plan.null_condition().ptr == null_buffer.ptr


@pytest.mark.skipif(not is_device_available("cuda:0"), reason="CUDA unavailable")
def test_official_condition_encoder_bf16_finite_cuda():
    root = local_model_root() / "ACE-Step" / "Ace-Step1.5"
    path = root / "acestep-v15-turbo"
    silence_path = path / "silence_latent.pt"
    if not (path / "model.safetensors").is_file() or not silence_path.is_file():
        pytest.skip("official ACE-Step DiT bundle is not downloaded")
    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config = AceStepDiTConfig.from_dict(raw)
    assert (
        config.num_lyric_encoder_hidden_layers,
        config.num_timbre_encoder_hidden_layers,
    ) == (8, 4)
    assert len(condition_weight_names(config)) == 141
    encoder = AceStepConditionEncoder(
        path, config, dtype=wp.bfloat16, device="cuda:0", use_cublas=False
    )
    rng = np.random.default_rng(223)
    text = wp.array(
        rng.normal(0, 0.1, (1, 2, config.text_hidden_dim)),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    lyric = wp.array(
        rng.normal(0, 0.1, (1, 3, config.text_hidden_dim)),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    silence = wp.array(
        load_silence_latent(silence_path), dtype=wp.bfloat16, device="cuda:0"
    )
    plan = encoder.plan(
        text,
        wp.ones((1, 2), dtype=wp.bool, device="cuda:0"),
        lyric,
        wp.array([[True, False, True]], device="cuda:0"),
        silence,
    )
    condition, valid = plan.execute()
    values = condition.numpy()
    assert values.shape == (1, 6, 2048)
    assert np.isfinite(values).all()
    assert valid.numpy().tolist() == [[True, True, True, True, True, False]]
    assert np.isfinite(plan.null_condition().numpy()).all()


def test_condition_encoder_executes_live_attention_and_mlp(tmp_path):
    rng = np.random.default_rng(227)
    config = _config()
    weights = _weights(config, rng)
    prefix = "encoder.lyric_encoder.layers.0."
    for suffix in (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ):
        weights[prefix + suffix] = rng.normal(
            0, 0.03, weights[prefix + suffix].shape
        ).astype(np.float32)
    path = tmp_path / "dit"
    path.mkdir()
    _write_safetensors(path / "model.safetensors", weights)
    encoder = AceStepConditionEncoder(
        path, config, dtype=wp.float16, device="cpu", use_cublas=False
    )
    plan = encoder.plan(
        wp.array(rng.normal(size=(1, 1, 1024)), dtype=wp.float16, device="cpu"),
        wp.ones((1, 1), dtype=wp.bool, device="cpu"),
        wp.array(rng.normal(size=(1, 2, 1024)), dtype=wp.float16, device="cpu"),
        wp.ones((1, 2), dtype=wp.bool, device="cpu"),
        wp.array(rng.normal(size=(1, 4, 4)), dtype=wp.float16, device="cpu"),
    )
    condition, _ = plan.execute()
    layer = plan.lyric_stack.layers[0]
    assert np.any(layer.attention.output.numpy() != 0.0)
    assert np.any(layer._mlp_tensors["down"].numpy() != 0.0)
    assert np.isfinite(condition.numpy()).all()
