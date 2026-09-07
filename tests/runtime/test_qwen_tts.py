# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from warp_nn.runtime.qwen.tts import (
    SUPPORTED_TTS_LANGUAGES,
    _backbone_mapping,
    _sample_tts_logits,
    load_qwen3_tts_config,
)


def _decoder(hidden, layers, heads, kv_heads, vocabulary):
    return {
        "hidden_size": hidden,
        "intermediate_size": hidden * 3,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": hidden // heads,
        "vocab_size": vocabulary,
        "max_position_embeddings": 4096,
        "hidden_act": "silu",
    }


def _config():
    predictor = _decoder(1024, 5, 16, 8, 2048)
    predictor["num_code_groups"] = 16
    talker = _decoder(2048, 28, 16, 8, 3072)
    talker.update(
        {
            "num_code_groups": 16,
            "text_hidden_size": 2048,
            "code_predictor_config": predictor,
        }
    )
    return {
        "model_type": "qwen3_tts",
        "tts_model_type": "base",
        "talker_config": talker,
    }


def test_qwen3_tts_config_contract(tmp_path):
    document = _config()
    (tmp_path / "config.json").write_text(json.dumps(document), encoding="utf-8")
    assert load_qwen3_tts_config(tmp_path) == document
    assert "english" in SUPPORTED_TTS_LANGUAGES
    assert "german" in SUPPORTED_TTS_LANGUAGES

    document["talker_config"]["num_code_groups"] = 8
    (tmp_path / "config.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="code-group"):
        load_qwen3_tts_config(tmp_path)


def test_qwen3_tts_backbone_mapping_keeps_nested_checkpoint_names():
    mapping = _backbone_mapping(
        _config()["talker_config"],
        "talker.model",
        "talker.codec_head.weight",
    )
    assert mapping["model.layers.0.self_attn.q_proj.weight"] == (
        "talker.model.layers.0.self_attn.q_proj.weight"
    )
    assert mapping["lm_head.weight"] == "talker.codec_head.weight"
    assert "model.embed_tokens.weight" not in mapping


def test_qwen3_tts_sampling_suppresses_non_acoustic_tokens_but_preserves_eos():
    logits = np.zeros((1, 1, 3072), dtype=np.float32)
    logits[..., 2500] = 100.0
    logits[..., 2150] = 10.0
    token = _sample_tts_logits(
        logits,
        rng=np.random.default_rng(0),
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        token_stop=2048,
        eos_token=2150,
    )
    assert token == 2150

    token = _sample_tts_logits(
        logits,
        rng=np.random.default_rng(0),
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        token_stop=2048,
        eos_token=2150,
        allow_eos=False,
    )
    assert token < 2048


def test_qwen3_tts_repetition_penalty_matches_hugging_face_policy():
    logits = np.zeros(4, dtype=np.float32)
    logits[1] = 2.0
    logits[2] = 1.5
    assert (
        _sample_tts_logits(
            logits,
            rng=np.random.default_rng(0),
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=2.0,
            previous_tokens=(1,),
        )
        == 2
    )
