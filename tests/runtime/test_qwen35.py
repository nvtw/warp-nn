# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import json

import numpy as np

from tests.utilities import is_device_available, write_safetensors
from warp_nn.runtime.qwen35 import Qwen35Runner, _validate_config, _weight_names


def _bfloat16_bytes(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16).tobytes()


def _write_tiny_qwen35(path):
    config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 8,
        "intermediate_size": 12,
        "vocab_size": 16,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1.0e-6,
        "attention_bias": False,
        "hidden_act": "silu",
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    }
    rng = np.random.default_rng(97)
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
                prefix + "mlp.gate_proj.weight": (12, 8),
                prefix + "mlp.up_proj.weight": (12, 8),
                prefix + "mlp.down_proj.weight": (8, 12),
            }
        )
    linear = "model.language_model.layers.0.linear_attn."
    shapes.update(
        {
            linear + "in_proj_qkv.weight": (16, 8),
            linear + "in_proj_z.weight": (8, 8),
            linear + "in_proj_a.weight": (2, 8),
            linear + "in_proj_b.weight": (2, 8),
            linear + "conv1d.weight": (16, 1, 3),
            linear + "A_log": (2,),
            linear + "dt_bias": (2,),
            linear + "norm.weight": (4,),
            linear + "out_proj.weight": (8, 8),
        }
    )
    attention = "model.language_model.layers.1.self_attn."
    shapes.update(
        {
            attention + "q_proj.weight": (24, 8),
            attention + "k_proj.weight": (4, 8),
            attention + "v_proj.weight": (4, 8),
            attention + "q_norm.weight": (4,),
            attention + "k_norm.weight": (4,),
            attention + "o_proj.weight": (8, 12),
        }
    )
    tensors = {}
    for name in _weight_names(config):
        shape = shapes[name]
        if name.endswith("layernorm.weight") or name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
            values = np.zeros(shape, dtype=np.float32)
        elif name.endswith("linear_attn.norm.weight"):
            values = np.ones(shape, dtype=np.float32)
        elif name.endswith("A_log"):
            values = np.zeros(shape, dtype=np.float32)
        else:
            values = rng.normal(0.0, 0.08, shape).astype(np.float32)
        tensors[name] = ("BF16", shape, _bfloat16_bytes(values))
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"text_config": config}), encoding="utf-8")
    write_safetensors(path / "model.safetensors", tensors)


def test_qwen38_text_metadata_compatibility():
    config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "num_hidden_layers": 64,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1.0e-6,
        "attention_bias": False,
        "hidden_act": "silu",
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
        },
    }

    _validate_config(config)
    names = _weight_names(config)
    assert len(names) == 851
    assert "model.language_model.layers.62.linear_attn.in_proj_qkv.weight" in names
    assert "model.language_model.layers.63.self_attn.q_proj.weight" in names

    config["rope_parameters"] = {**config["rope_parameters"], "rope_type": "yarn", "factor": 4.0}
    with pytest.raises(ValueError, match="default Qwen rotary"):
        _validate_config(config)


@pytest.mark.parametrize("use_cublas", [False, True])
def test_qwen35_native_prefill_decode_and_graph_replay(tmp_path, use_cublas):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    model_path = tmp_path / "tiny-qwen35"
    _write_tiny_qwen35(model_path)
    runner = Qwen35Runner(model_path, device="cuda:0", cache_capacity=8, prefill_chunk_size=4, use_cublas=use_cublas)

    first = runner.prefill([1, 2, 3]).numpy()
    assert set(runner._chunk_plans) == {2, 4}
    assert first.shape == (1, 1, 16)
    assert np.isfinite(first).all()
    decoded = runner.decode(4).numpy()
    assert decoded.shape == (1, 1, 16)
    assert np.isfinite(decoded).all()
    replayed = runner.prefill([1, 2, 3])
    np.testing.assert_allclose(replayed.numpy(), first, atol=2.0e-2, rtol=2.0e-2)
    assert 0 <= runner.sample_greedy(replayed) < 16
