# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.formats.safetensors import SafeTensorArchive
from warp_nn.runtime.qwen.encoder import (
    Qwen3Encoder,
    load_qwen3_encoder_config,
    qwen3_encoder_weight_names,
)
from warp_nn.runtime.qwen.causal import Qwen3CausalLM, qwen3_causal_weight_names
from warp_nn.runtime.tokenizers import _BYTE_ENCODER


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    offset = 0
    header = {}
    chunks = []
    for name, value in tensors.items():
        value = np.ascontiguousarray(value, dtype=np.float32)
        data = value.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def _tiny_checkpoint(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    rng = np.random.default_rng(17)
    vocabulary = {
        character: index for index, character in enumerate(_BYTE_ENCODER.values())
    }
    vocabulary_size = len(vocabulary) + 2
    config = {
        "model_type": "qwen3",
        "attention_bias": False,
        "hidden_act": "silu",
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": vocabulary_size,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1.0e-6,
        "rope_theta": 1000.0,
        "layer_types": ["full_attention"],
    }
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "normalizer": {"type": "NFC"},
                "model": {"type": "BPE", "vocab": vocabulary, "merges": []},
                "added_tokens": [
                    {
                        "id": len(vocabulary),
                        "content": "<|endoftext|>",
                        "special": True,
                    },
                    {
                        "id": len(vocabulary) + 1,
                        "content": "<|im_end|>",
                        "special": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def matrix(rows, columns, scale=0.18):
        return rng.normal(0.0, scale, (rows, columns)).astype(np.float32)

    prefix = "model.layers.0."
    weights = {
        "model.embed_tokens.weight": matrix(vocabulary_size, 4),
        "model.norm.weight": rng.uniform(0.8, 1.2, 4).astype(np.float32),
        prefix + "input_layernorm.weight": rng.uniform(0.8, 1.2, 4).astype(np.float32),
        prefix + "post_attention_layernorm.weight": rng.uniform(0.8, 1.2, 4).astype(
            np.float32
        ),
        prefix + "self_attn.q_proj.weight": matrix(4, 4),
        prefix + "self_attn.k_proj.weight": matrix(2, 4),
        prefix + "self_attn.v_proj.weight": matrix(2, 4),
        prefix + "self_attn.q_norm.weight": rng.uniform(0.8, 1.2, 2).astype(np.float32),
        prefix + "self_attn.k_norm.weight": rng.uniform(0.8, 1.2, 2).astype(np.float32),
        prefix + "self_attn.o_proj.weight": matrix(4, 4),
        prefix + "mlp.gate_proj.weight": matrix(6, 4),
        prefix + "mlp.up_proj.weight": matrix(6, 4),
        prefix + "mlp.down_proj.weight": matrix(4, 6),
    }
    _write_safetensors(path / "model.safetensors", weights)
    return config, weights


def _rms(x, scale, epsilon):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * scale


def _reference(token_ids, config, weights):
    epsilon = config["rms_norm_eps"]
    hidden = weights["model.embed_tokens.weight"][token_ids]
    prefix = "model.layers.0."
    x = _rms(hidden, weights[prefix + "input_layernorm.weight"], epsilon)
    q = x @ weights[prefix + "self_attn.q_proj.weight"].T
    k = x @ weights[prefix + "self_attn.k_proj.weight"].T
    v = (x @ weights[prefix + "self_attn.v_proj.weight"].T).reshape(-1, 1, 2)
    q = _rms(q.reshape(-1, 2, 2), weights[prefix + "self_attn.q_norm.weight"], epsilon)
    k = _rms(k.reshape(-1, 1, 2), weights[prefix + "self_attn.k_norm.weight"], epsilon)
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

    q = rotate(q)
    k = rotate(k)
    attention = np.empty_like(q)
    for token in range(len(token_ids)):
        for head in range(2):
            scores = q[token, head] @ k[: token + 1, 0].T / np.sqrt(2.0)
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            attention[token, head] = probabilities @ v[: token + 1, 0]
    projected = attention.reshape(-1, 4) @ weights[prefix + "self_attn.o_proj.weight"].T
    residual = hidden + projected
    x = _rms(residual, weights[prefix + "post_attention_layernorm.weight"], epsilon)
    gate = x @ weights[prefix + "mlp.gate_proj.weight"].T
    up = x @ weights[prefix + "mlp.up_proj.weight"].T
    swiglu = gate / (1.0 + np.exp(-gate)) * up
    hidden = residual + swiglu @ weights[prefix + "mlp.down_proj.weight"].T
    return _rms(hidden, weights["model.norm.weight"], epsilon)


def test_qwen3_encoder_matches_numpy_and_embedding_gather(tmp_path):
    config, weights = _tiny_checkpoint(tmp_path / "qwen")
    runner = Qwen3Encoder(
        tmp_path / "qwen",
        dtype=wp.float16,
        device="cpu",
        use_cublas=False,
    )
    token_ids = [3, 9, 4]
    actual = runner.encode_ids(token_ids).numpy()[0]
    expected = _reference(token_ids, config, weights)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-3, atol=3.0e-3)
    embedded = runner.embed_ids([[3, 9], [4, 7]]).numpy()
    np.testing.assert_allclose(
        embedded,
        weights["model.embed_tokens.weight"][[[3, 9], [4, 7]]],
        rtol=5.0e-4,
        atol=1.0e-4,
    )


def test_qwen3_encoder_rejects_invalid_ids(tmp_path):
    config, _ = _tiny_checkpoint(tmp_path / "qwen")
    runner = Qwen3Encoder(
        tmp_path / "qwen",
        dtype=wp.float16,
        device="cpu",
        use_cublas=False,
    )
    with pytest.raises(ValueError, match="must not be empty"):
        runner.encode_ids([])
    with pytest.raises(ValueError, match="outside"):
        runner.encode_ids([config["vocab_size"]])
    with pytest.raises(ValueError, match="one-dimensional"):
        runner.encode_ids([[1, 2]])


def test_qwen3_causal_prefill_decode_matches_full_reference(tmp_path):
    config, weights = _tiny_checkpoint(tmp_path / "qwen")
    config["tie_word_embeddings"] = True
    (tmp_path / "qwen" / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    runner = Qwen3CausalLM(
        tmp_path / "qwen",
        dtype=wp.float16,
        device="cpu",
        cache_capacity=16,
        prefill_chunk_size=2,
        use_cublas=False,
    )
    logits = runner.prefill([3, 9, 4]).numpy()[0, 0]
    expected = _reference([3, 9, 4], config, weights)[-1]
    np.testing.assert_allclose(
        logits,
        expected @ weights["model.embed_tokens.weight"].T,
        rtol=4.0e-3,
        atol=4.0e-3,
    )
    decoded = runner.decode(7).numpy()[0, 0]
    expected = _reference([3, 9, 4, 7], config, weights)[-1]
    np.testing.assert_allclose(
        decoded,
        expected @ weights["model.embed_tokens.weight"].T,
        rtol=4.0e-3,
        atol=4.0e-3,
    )
    assert "lm_head.weight" not in qwen3_causal_weight_names(config)


def test_official_qwen3_embedding_checkpoint_contract():
    path = Path(
        "/home/twidmer/.cache/warp-nn/models/ACE-Step/Ace-Step1.5/Qwen3-Embedding-0.6B"
    )
    model = path / "model.safetensors"
    if not model.is_file():
        pytest.skip("official ACE-Step Qwen3 embedding tensors are still downloading")
    config = load_qwen3_encoder_config(path)
    archive = SafeTensorArchive(path)
    names = qwen3_encoder_weight_names(config)
    assert not ({name.removeprefix("model.") for name in names} - set(archive.names))
    assert archive.metadata("embed_tokens.weight").shape == (151669, 1024)
    assert archive.metadata("layers.0.self_attn.q_proj.weight").shape == (
        2048,
        1024,
    )


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is not available")
def test_qwen3_encoder_cuda_graph_replay(tmp_path):
    config, weights = _tiny_checkpoint(tmp_path / "qwen")
    runner = Qwen3Encoder(
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


def test_official_qwen3_embedding_finite_bf16_cuda():
    path = Path(
        "/home/twidmer/.cache/warp-nn/models/ACE-Step/Ace-Step1.5/Qwen3-Embedding-0.6B"
    )
    if not (path / "model.safetensors").is_file():
        pytest.skip("official ACE-Step Qwen3 embedding tensors are still downloading")
    if not wp.is_cuda_available():
        pytest.skip("CUDA is not available")
    runner = Qwen3Encoder(path, dtype=wp.bfloat16, device="cuda:0", use_cublas=False)
    output = runner.encode_ids([10, 20, 30]).numpy()
    assert output.shape == (1, 3, 1024)
    assert np.isfinite(output).all()
