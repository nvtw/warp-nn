# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from tests.utilities import is_device_available, write_safetensors
from warp_nn.runtime.muse_glimmer import MuseGlimmerRunner, _validate_config, _weight_names


def _bfloat16_bytes(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16).tobytes()


def _write_tiny_muse(path):
    config = {
        "model_type": "muse_glimmer_text",
        "hidden_size": 8,
        "intermediate_size": 12,
        "vocab_size": 16,
        "num_hidden_layers": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "layer_rope_theta": [10000.0, 0.0],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 16,
        "sliding_window": 3,
        "qk_scale_factor": 3.87,
        "rms_norm_eps": 1.0e-5,
        "post_norm_eps": 1.0e-8,
        "output_multiplier": 0.19611613513818404,
        "final_logit_softcapping": 20.0,
        "hidden_activation": "silu",
        "attention_bias": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    }
    shapes = {
        "model.language_model.embed_tokens.weight": (16, 8),
        "model.language_model.norm.weight": (8,),
        "lm_head.weight": (16, 8),
    }
    for index in range(2):
        prefix = f"model.language_model.layers.{index}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (8,),
                prefix + "post_attention_layernorm.weight": (8,),
                prefix + "pre_feedforward_layernorm.weight": (8,),
                prefix + "post_feedforward_layernorm.weight": (8,),
                prefix + "self_attn.q_proj.weight": (8, 8),
                prefix + "self_attn.k_proj.weight": (4, 8),
                prefix + "self_attn.v_proj.weight": (4, 8),
                prefix + "self_attn.gate_proj.weight": (8, 8),
                prefix + "self_attn.o_proj.weight": (8, 8),
                prefix + "mlp.gate_proj.weight": (12, 8),
                prefix + "mlp.up_proj.weight": (12, 8),
                prefix + "mlp.down_proj.weight": (8, 12),
            }
        )

    rng = np.random.default_rng(91)
    tensors = {}
    for name in _weight_names(config):
        shape = shapes[name]
        if name.endswith("layernorm.weight"):
            values = rng.normal(0.0, 0.02, shape).astype(np.float32)
        elif name == "model.language_model.norm.weight":
            values = np.ones(shape, dtype=np.float32)
        else:
            values = rng.normal(0.0, 0.08, shape).astype(np.float32)
        dtype = "F32" if name.endswith("norm.weight") else "BF16"
        data = values.tobytes() if dtype == "F32" else _bfloat16_bytes(values)
        tensors[name] = (dtype, shape, data)

    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "muse_glimmer", "text_config": config}))
    write_safetensors(path / "model.safetensors", tensors)


def test_muse_glimmer_30b_metadata_compatibility():
    config = {
        "hidden_size": 6656,
        "intermediate_size": 19968,
        "vocab_size": 202048,
        "num_hidden_layers": 52,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 13,
        "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0.0] * 13,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "max_position_embeddings": 131072,
        "sliding_window": 2048,
        "qk_scale_factor": 3.87,
        "rms_norm_eps": 1.0e-5,
        "post_norm_eps": 1.0e-8,
        "output_multiplier": 0.19611613513818404,
        "final_logit_softcapping": 20.0,
    }

    _validate_config(config)
    names = _weight_names(config)
    assert len(names) == 627
    assert "model.language_model.layers.51.self_attn.gate_proj.weight" in names


@pytest.mark.parametrize("use_cublas", [False, True])
def test_muse_glimmer_prefill_decode_ring_cache_and_graph_replay(tmp_path, use_cublas):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    model_path = tmp_path / "tiny-muse"
    _write_tiny_muse(model_path)
    runner = MuseGlimmerRunner(
        model_path, device="cuda:0", cache_capacity=8, prefill_chunk_size=2, use_cublas=use_cublas
    )

    assert runner.local_cache_capacity == 4
    first = runner.prefill([1, 2, 3]).numpy()
    assert first.shape == (1, 1, 16)
    assert np.isfinite(first).all()
    assert np.isfinite(runner.decode(4).numpy()).all()
    assert np.isfinite(runner.decode(5).numpy()).all()
    replayed = runner.prefill([1, 2, 3])
    np.testing.assert_allclose(replayed.numpy(), first, atol=2.0e-2, rtol=2.0e-2)
    assert 0 <= runner.sample_greedy(replayed) < 16
