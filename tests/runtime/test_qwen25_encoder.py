# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import warp as wp

from tests.runtime.test_qwen3_encoder import _tiny_checkpoint, _write_safetensors
from warp_nn.runtime.qwen.encoder import QwenEncoder, load_qwen_encoder_config


def _tiny_qwen25(path):
    config, weights = _tiny_checkpoint(path)
    config["model_type"] = "qwen2_5_vl"
    config["attention_bias"] = True
    config["rope_scaling"] = {"type": "mrope", "mrope_section": [1]}
    config.pop("head_dim")
    prefix = "model.layers.0.self_attn."
    weights.pop(prefix + "q_norm.weight")
    weights.pop(prefix + "k_norm.weight")
    rng = np.random.default_rng(29)
    weights[prefix + "q_proj.bias"] = rng.normal(0, 0.1, 4).astype(np.float32)
    weights[prefix + "k_proj.bias"] = rng.normal(0, 0.1, 2).astype(np.float32)
    weights[prefix + "v_proj.bias"] = rng.normal(0, 0.1, 2).astype(np.float32)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _write_safetensors(path / "model.safetensors", weights)
    return config, weights


def _rms(x, scale, epsilon):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * scale


def _reference(token_ids, config, weights):
    epsilon = config["rms_norm_eps"]
    hidden = weights["model.embed_tokens.weight"][token_ids]
    root = "model.layers.0."
    attention = root + "self_attn."
    x = _rms(hidden, weights[root + "input_layernorm.weight"], epsilon)
    q = x @ weights[attention + "q_proj.weight"].T + weights[attention + "q_proj.bias"]
    k = x @ weights[attention + "k_proj.weight"].T + weights[attention + "k_proj.bias"]
    v = x @ weights[attention + "v_proj.weight"].T + weights[attention + "v_proj.bias"]
    q = q.reshape(-1, 2, 2)
    k = k.reshape(-1, 1, 2)
    v = v.reshape(-1, 1, 2)
    positions = np.arange(len(token_ids), dtype=np.float32)
    inverse = config["rope_theta"] ** (-np.arange(0, 2, 2, dtype=np.float32) / 2)
    cosine = np.cos(positions[:, None] * inverse)
    sine = np.sin(positions[:, None] * inverse)

    def rotate(values):
        first, second = values[..., :1], values[..., 1:]
        return np.concatenate(
            (
                first * cosine[:, None] - second * sine[:, None],
                second * cosine[:, None] + first * sine[:, None],
            ),
            axis=-1,
        )

    q, k = rotate(q), rotate(k)
    attended = np.empty_like(q)
    for token in range(len(token_ids)):
        for head in range(2):
            scores = q[token, head] @ k[: token + 1, 0].T / np.sqrt(2.0)
            probability = np.exp(scores - scores.max())
            probability /= probability.sum()
            attended[token, head] = probability @ v[: token + 1, 0]
    projected = attended.reshape(-1, 4) @ weights[attention + "o_proj.weight"].T
    residual = hidden + projected
    x = _rms(residual, weights[root + "post_attention_layernorm.weight"], epsilon)
    gate = x @ weights[root + "mlp.gate_proj.weight"].T
    up = x @ weights[root + "mlp.up_proj.weight"].T
    hidden = (
        residual
        + (gate / (1 + np.exp(-gate)) * up) @ weights[root + "mlp.down_proj.weight"].T
    )
    return _rms(hidden, weights["model.norm.weight"], epsilon)


def test_qwen25_language_encoder_matches_numpy_and_derives_head_dim(tmp_path):
    config, weights = _tiny_qwen25(tmp_path / "qwen")
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tmp_path / "qwen" / "tokenizer.json").replace(tokenizer_path / "tokenizer.json")
    loaded = load_qwen_encoder_config(tmp_path / "qwen")
    assert loaded["head_dim"] == 2
    runner = QwenEncoder(
        tmp_path / "qwen",
        dtype=wp.float16,
        device="cpu",
        use_cublas=False,
        tokenizer_path=tokenizer_path,
    )
    token_ids = [3, 9, 4]
    np.testing.assert_allclose(
        runner.encode_ids(token_ids).numpy()[0],
        _reference(token_ids, config, weights),
        rtol=3.0e-3,
        atol=3.0e-3,
    )
    assert not runner.qk_norm
    assert runner.attention_bias


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is not available")
def test_qwen25_language_encoder_cuda_graph_replay(tmp_path):
    config, weights = _tiny_qwen25(tmp_path / "qwen")
    runner = QwenEncoder(
        tmp_path / "qwen", dtype=wp.float16, device="cuda:0", use_cublas=False
    )
    runner.encode_ids([3, 9, 4]).numpy()
    second = runner.encode_ids([4, 9, 3]).numpy()[0].copy()
    first = runner.encode_ids([3, 9, 4]).numpy()[0].copy()
    np.testing.assert_allclose(
        second, _reference([4, 9, 3], config, weights), rtol=3.0e-3, atol=3.0e-3
    )
    np.testing.assert_allclose(
        first, _reference([3, 9, 4], config, weights), rtol=3.0e-3, atol=3.0e-3
    )
    assert not np.allclose(first, second)
