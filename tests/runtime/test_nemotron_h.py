# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available, local_model_root, write_safetensors
from warp_nn.runtime.nemotron.runner import (
    NemotronHRunner,
    _language_config,
    _validate_config,
    _weight_names,
)
from warp_nn.runtime.formats.safetensors import SafeTensorArchive, SafeTensorNamespace

from warp_nn.runtime.operators import SparseExpertPlan


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
        elif (
            name.endswith("norm.weight")
            or name.endswith("A_log")
            or name.endswith("dt_bias")
            or name.endswith(".D")
        ):
            tensors[name] = ("F32", shape, values.tobytes())
        else:
            tensors[name] = ("BF16", shape, _bfloat16_bytes(values))
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_safetensors(path / "model.safetensors", tensors)


def _write_tiny_omni(path):
    config = {
        "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
        "llm_config": {
            "model_type": "nemotron_h",
            "hidden_size": 4,
            "intermediate_size": 6,
            "vocab_size": 8,
            "num_hidden_layers": 1,
            "hybrid_override_pattern": "E",
            "mamba_num_heads": 1,
            "mamba_head_dim": 4,
            "n_groups": 1,
            "ssm_state_size": 2,
            "conv_kernel": 3,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 8,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 3,
            "moe_shared_expert_intermediate_size": 5,
            "routed_scaling_factor": 2.5,
            "n_group": 1,
            "topk_group": 1,
        },
    }
    language = config["llm_config"]
    shapes = {
        "backbone.embeddings.weight": (8, 4),
        "backbone.norm_f.weight": (4,),
        "lm_head.weight": (8, 4),
        "backbone.layers.0.norm.weight": (4,),
        "backbone.layers.0.mixer.gate.weight": (4, 4),
        "backbone.layers.0.mixer.gate.e_score_correction_bias": (4,),
        "backbone.layers.0.mixer.shared_experts.up_proj.weight": (5, 4),
        "backbone.layers.0.mixer.shared_experts.down_proj.weight": (4, 5),
    }
    for expert in range(4):
        shapes[f"backbone.layers.0.mixer.experts.{expert}.up_proj.weight"] = (3, 4)
        shapes[f"backbone.layers.0.mixer.experts.{expert}.down_proj.weight"] = (4, 3)
    rng = np.random.default_rng(101)
    tensors = {}
    for name in _weight_names(language):
        shape = shapes[name]
        values = (
            np.ones(shape, dtype=np.float32)
            if name.endswith("norm.weight")
            else rng.normal(0.0, 0.15, shape).astype(np.float32)
        )
        full_name = "language_model." + name
        if name.endswith("e_score_correction_bias"):
            tensors[full_name] = ("F32", shape, values.tobytes())
        else:
            tensors[full_name] = ("BF16", shape, _bfloat16_bytes(values))
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_safetensors(path / "model.safetensors", tensors)


def test_nested_omni_config_dispatches_to_nemotron_runner(tmp_path, monkeypatch):
    import warp_nn.runtime as runtime

    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "NemotronH_Omni", "llm_config": {"model_type": "nemotron_h"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runtime, "NemotronHRunner", lambda path, **options: (path, options)
    )
    assert runtime.create_text_runner(tmp_path, cache_capacity=8) == (
        tmp_path,
        {"cache_capacity": 8},
    )


def test_nested_bf16_moe_prefill_and_embedding_override(tmp_path):
    model_path = tmp_path / "tiny-omni"
    _write_tiny_omni(model_path)
    runner = NemotronHRunner(
        model_path,
        device="cpu",
        cache_capacity=4,
        prefill_chunk_size=2,
        use_cublas=False,
    )
    expert_up = runner.weights["backbone.layers.0.mixer.experts.up_proj.weight"]
    expert_down = runner.weights["backbone.layers.0.mixer.experts.down_proj.weight"]
    assert expert_up.shape == (4, 3, 4)
    assert expert_down.shape == (4, 4, 3)

    ordinary = runner.prefill([1, 2]).numpy().copy()
    embeddings = wp.array(
        np.full((1, 4), 0.5, dtype=np.float32),
        dtype=runner.dtype,
        device=runner.device,
    )
    overridden = runner.prefill_with_embeddings([1, 2], embeddings, [1]).numpy()
    assert ordinary.shape == overridden.shape == (1, 2, 8)
    assert np.isfinite(overridden).all()
    assert not np.allclose(ordinary, overridden)


def test_nemotron_omni_bf16_language_manifest():
    path = local_model_root() / "nvidia" / "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    if not (path / "model.safetensors.index.json").is_file():
        pytest.skip("local Nemotron Omni BF16 checkpoint is unavailable")
    document = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config, prefix = _language_config(document)
    _validate_config(config)
    assert prefix == "language_model."
    assert config["attention_head_dim"] == 128
    assert config["hybrid_override_pattern"].count("E") == 23
    archive = SafeTensorNamespace(SafeTensorArchive(path), prefix)
    assert set(_weight_names(config)) <= set(archive.names)


def test_sparse_expert_plan_matches_reference():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(2, 4)).astype(np.float32)
    gate = rng.normal(size=(4, 4)).astype(np.float32)
    correction = np.array([0.02, -0.03, 0.01, 0.04], dtype=np.float32)
    expert_up = rng.normal(size=(4, 3, 4)).astype(np.float32)
    expert_down = rng.normal(size=(4, 4, 3)).astype(np.float32)
    shared_up = rng.normal(size=(5, 4)).astype(np.float32)
    shared_down = rng.normal(size=(4, 5)).astype(np.float32)
    arrays = [
        wp.array(value, device="cpu")
        for value in (
            x,
            gate,
            correction,
            expert_up,
            expert_down,
            shared_up,
            shared_down,
        )
    ]
    plan = SparseExpertPlan(*arrays, top_k=2, scale=2.5)
    actual = plan.execute().numpy()

    logits = x @ gate.T
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    ranking = probabilities + correction
    indices = np.argsort(-ranking, axis=1, kind="stable")[:, :2]
    weights = np.take_along_axis(probabilities, indices, axis=1)
    weights = weights / weights.sum(axis=1, keepdims=True) * 2.5
    expected = np.maximum(x @ shared_up.T, 0.0) ** 2 @ shared_down.T
    for row in range(x.shape[0]):
        for slot, expert in enumerate(indices[row]):
            hidden = np.maximum(x[row] @ expert_up[expert].T, 0.0) ** 2
            expected[row] += weights[row, slot] * (hidden @ expert_down[expert].T)
    np.testing.assert_array_equal(plan.routing_indices.numpy(), indices)
    np.testing.assert_allclose(
        plan.routing_weights.numpy(), weights, rtol=1.0e-5, atol=1.0e-6
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)


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
        model_path,
        device="cuda:0",
        cache_capacity=8,
        prefill_chunk_size=4,
        use_cublas=use_cublas,
    )

    assert (
        runner.weights["backbone.layers.1.mixer.up_proj.weight"].dtype.__name__
        == "bfloat16"
    )
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
