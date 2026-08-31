# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np

from tests.utilities import write_safetensors
from warp_nn.runtime.analysis import analyze_model, write_model_graph
from warp_nn.runtime.analysis.report import render_report


def _write_model(path):
    path.mkdir()
    config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 8,
        "intermediate_size": 12,
        "vocab_size": 16,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 4096,
    }
    (path / "config.json").write_text(
        json.dumps({"text_config": config}), encoding="utf-8"
    )
    shapes = {
        "model.language_model.embed_tokens.weight": (16, 8),
        "model.language_model.layers.0.input_layernorm.weight": (8,),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": (16, 8),
        "model.language_model.layers.0.mlp.up_proj.weight": (12, 8),
        "model.language_model.layers.1.input_layernorm.weight": (8,),
        "model.language_model.layers.1.self_attn.q_proj.weight": (8, 8),
        "model.language_model.layers.1.mlp.up_proj.weight": (12, 8),
        "model.language_model.norm.weight": (8,),
        "lm_head.weight": (16, 8),
    }
    tensors = {
        name: ("F16", shape, np.zeros(shape, dtype=np.float16).tobytes())
        for name, shape in shapes.items()
    }
    write_safetensors(path / "model.safetensors", tensors)


def test_analyze_model_builds_architecture_and_tensor_levels(tmp_path):
    model = tmp_path / "model"
    _write_model(model)

    graph = analyze_model(model)
    summary = graph["summary"]
    assert summary == {
        "name": "model",
        "path": str(model.resolve()),
        "format": "safetensors",
        "architecture": "qwen3_5_text",
        "parameters": 664,
        "bytes": 1328,
        "tensorCount": 9,
        "layers": 2,
        "hiddenSize": 8,
        "attentionHeads": 2,
        "kvHeads": 1,
        "contextLength": 4096,
        "formats": {"F16": 9},
        "headerOnly": True,
    }
    components = [node for node in graph["nodes"] if node["type"] == "component"]
    tensors = [node for node in graph["nodes"] if node["type"] == "tensor"]
    assert len(tensors) == 9
    assert {node.get("layerType") for node in components} >= {
        "linear_attention",
        "full_attention",
    }
    attention = next(
        node
        for node in components
        if node.get("layer") == 0 and node["kind"] == "attention"
    )
    assert attention["label"] == "Layer 0 · Linear Attention"
    assert attention["parameters"] == 128
    assert any(
        edge["source"] == attention["id"] and edge["kind"] == "contains"
        for edge in graph["edges"]
    )


def test_write_model_graph_is_standalone_and_escapes_script_data(tmp_path):
    model = tmp_path / "model"
    _write_model(model)
    output = write_model_graph(model, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")
    graph = analyze_model(model)
    graph["summary"]["name"] = "</script>"
    escaped = render_report(graph)

    assert html.startswith("<!doctype html>")
    assert 'type="application/json"' in html
    assert "<\\/script>" in escaped
    assert "https://" not in html
    assert "http://" not in html
    assert "<script src=" not in html
    assert "Model atlas" in html
