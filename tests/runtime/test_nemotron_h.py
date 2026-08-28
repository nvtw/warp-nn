# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from tests.utilities import is_device_available, write_safetensors
from warp_nn.runtime.nemotron_h import NemotronHRunner, _validate_config, _weight_names


def _bfloat16_bytes(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16).tobytes()


def _write_tiny_nemotron(path):
    config = {
        "model_type": "nemotron_h",
        "hidden_size": 8,
        "intermediate_size": 12,
        "vocab_size": 16,
        "num_hidden_layers": 3,
        "hybrid_override_pattern": "M-*",
        "mamba_num_heads": 2,
        "mamba_head_dim": 4,
        "n_groups": 2,
        "ssm_state_size": 3,
        "conv_kernel": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "attention_head_dim": 4,
        "max_position_embeddings": 32,
        "layer_norm_epsilon": 1.0e-5,
        "mamba_hidden_act": "silu",
        "mlp_hidden_act": "relu2",
        "attention_bias": False,
        "mamba_proj_bias": False,
        "mlp_bias": False,
        "use_bias": False,
    }
    shapes = {
        "backbone.embeddings.weight": (16, 8),
        "backbone.norm_f.weight": (8,),
        "lm_head.weight": (16, 8),
        "backbone.layers.0.norm.weight": (8,),
        "backbone.layers.0.mixer.norm.weight": (2, 4),
        "backbone.layers.0.mixer.A_log": (2,),
        "backbone.layers.0.mixer.D": (2,),
        "backbone.layers.0.mixer.dt_bias": (2,),
        "backbone.layers.0.mixer.conv1d.weight": (20, 1, 3),
        "backbone.layers.0.mixer.conv1d.bias": (20,),
        "backbone.layers.0.mixer.in_proj.weight": (30, 8),
        "backbone.layers.0.mixer.out_proj.weight": (8, 8),
        "backbone.layers.1.norm.weight": (8,),
        "backbone.layers.1.mixer.up_proj.weight": (12, 8),
        "backbone.layers.1.mixer.down_proj.weight": (8, 12),
        "backbone.layers.2.norm.weight": (8,),
        "backbone.layers.2.mixer.q_proj.weight": (8, 8),
        "backbone.layers.2.mixer.k_proj.weight": (4, 8),
        "backbone.layers.2.mixer.v_proj.weight": (4, 8),
        "backbone.layers.2.mixer.o_proj.weight": (8, 8),
    }
    rng = np.random.default_rng(73)
    tensors = {}
    fp8_name = "backbone.layers.1.mixer.up_proj.weight"
    for name in _weight_names(config):
        shape = shapes[name]
        if name.endswith("norm.weight"):
            values = np.ones(shape, dtype=np.float32)
        elif name.endswith("A_log"):
            values = np.zeros(shape, dtype=np.float32)
        elif name.endswith("dt_bias"):
            values = np.full(shape, -1.0, dtype=np.float32)
        elif name.endswith(".D"):
            values = np.ones(shape, dtype=np.float32)
        else:
            values = rng.normal(0.0, 0.08, shape).astype(np.float32)
        if name == fp8_name:
            tensors[name] = ("F8_E4M3", shape, bytes(np.prod(shape)))
            tensors[name + "_scale"] = ("F32", (1,), np.float32(0.5).tobytes())
        elif name.endswith("norm.weight") or name.endswith("A_log") or name.endswith("dt_bias") or name.endswith(".D"):
            tensors[name] = ("F32", shape, values.tobytes())
        else:
            tensors[name] = ("BF16", shape, _bfloat16_bytes(values))
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_safetensors(path / "model.safetensors", tensors)


def test_nemotron_h_4b_metadata_compatibility():
    config = {
        "hidden_size": 3136,
        "intermediate_size": 12544,
        "vocab_size": 131072,
        "num_hidden_layers": 42,
        "hybrid_override_pattern": "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-",
        "mamba_num_heads": 96,
        "mamba_head_dim": 80,
        "n_groups": 8,
        "ssm_state_size": 128,
        "conv_kernel": 4,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "attention_head_dim": 128,
        "max_position_embeddings": 262144,
        "mamba_hidden_act": "silu",
        "mlp_hidden_act": "relu2",
    }

    _validate_config(config)
    names = _weight_names(config)
    assert len(names) == 263
    assert "backbone.layers.24.mixer.q_proj.weight" in names
    assert "backbone.layers.41.mixer.down_proj.weight" in names


@pytest.mark.parametrize("use_cublas", [False, True])
def test_nemotron_h_fp8_prefill_decode_and_graph_replay(tmp_path, use_cublas):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    model_path = tmp_path / "tiny-nemotron"
    _write_tiny_nemotron(model_path)
    runner = NemotronHRunner(
        model_path, device="cuda:0", cache_capacity=8, prefill_chunk_size=4, use_cublas=use_cublas
    )

    assert runner.weights["backbone.layers.1.mixer.up_proj.weight"].dtype.__name__ == "bfloat16"
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
