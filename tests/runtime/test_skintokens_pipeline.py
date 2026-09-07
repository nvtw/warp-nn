# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest

from warp_nn.runtime.skintokens.pipeline import (
    SkinTokensCheckpoint,
    SkinTokensPipeline,
    sample_token,
)


def test_checkpoint_loader_validates_artifacts(tmp_path):
    config = {
        "schema": "qtmesh-skintokens-onnx-v1",
        "num_points": 8192,
        "tokens_per_skin": 4,
        "tokens_skin_cond": 384,
        "vae_latent_channels": 512,
        "llm": {
            "hidden_size": 896,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "full_vocab_size": 33036,
        },
    }
    (tmp_path / "skintokens.json").write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError, match="graph files"):
        SkinTokensCheckpoint.load(tmp_path)
    for filename in (
        "embed.onnx",
        "mesh_cond.onnx",
        "vae_cond.onnx",
        "decoder.onnx",
        "skin_decode.onnx",
    ):
        (tmp_path / filename).touch()
    checkpoint = SkinTokensCheckpoint.load(tmp_path)
    assert checkpoint.num_points == 8192
    assert checkpoint.num_vertex_samples == 0
    assert checkpoint.vocabulary == 33036
    assert checkpoint.graph_paths["decoder"] == tmp_path / "decoder.onnx"

    config.update(
        {
            "num_vertex_samples": 1024,
            "dtype": "bf16",
            "batched_skin_decode": True,
            "external_fps_indices": True,
            "fps_candidates": {"seed": 7, "mesh": 2048, "vae": 1536},
        }
    )
    (tmp_path / "skintokens.json").write_text(json.dumps(config))
    checkpoint = SkinTokensCheckpoint.load(tmp_path)
    assert checkpoint.num_vertex_samples == 1024
    assert checkpoint.activation_dtype.__name__ == "bfloat16"
    assert checkpoint.external_fps_indices
    assert checkpoint.fps_seed == 7
    assert checkpoint.mesh_fps_candidates == 2048
    assert checkpoint.vae_fps_candidates == 1536
    assert checkpoint.batched_skin_decode


def test_constrained_sampling_is_seeded_and_never_selects_forbidden_tokens():
    logits = np.asarray((100.0, 4.0, 3.0, 2.0, 1.0), dtype=np.float32)
    allowed = np.asarray((1, 2, 3), dtype=np.int64)
    first = sample_token(
        logits,
        allowed,
        (1,),
        rng=np.random.default_rng(17),
        top_k=3,
        top_p=1.0,
        temperature=0.7,
        repetition_penalty=2.0,
    )
    second = sample_token(
        logits,
        allowed,
        (1,),
        rng=np.random.default_rng(17),
        top_k=3,
        top_p=1.0,
        temperature=0.7,
        repetition_penalty=2.0,
    )
    assert first == second
    assert first in allowed
    assert (
        sample_token(
            logits,
            allowed,
            (),
            rng=np.random.default_rng(0),
            temperature=0.0,
        )
        == 1
    )


def test_dynamic_skin_decode_chunks_and_pads_without_a_second_batch():
    pipeline = object.__new__(SkinTokensPipeline)
    pipeline.checkpoint = SimpleNamespace(
        batched_skin_decode=True, num_points=3, tokens_per_skin=4
    )
    pipeline.skin_decode_batch_size = 2
    pipeline.tokenizer = SimpleNamespace(switch=258, skeleton_vocab=267)
    calls = []

    def decode(ids, _cond, _latents):
        calls.append(ids.copy())
        return np.tile(ids[:, 0], (3, 1))

    pipeline._decode_weights_batched = decode
    local = np.arange(12, dtype=np.int64).reshape(3, 4)
    tokens = np.concatenate(((258,), local.reshape(-1) + 267, (33035,)))
    skeleton = SimpleNamespace(joints=np.zeros((3, 3), dtype=np.float32))
    weights = pipeline._decode_weights(tokens, None, None, skeleton)
    assert [call.shape for call in calls] == [(2, 4), (2, 4)]
    np.testing.assert_array_equal(calls[1][0], calls[1][1])
    np.testing.assert_array_equal(weights[0], (0, 4, 8))
