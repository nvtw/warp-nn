# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from tests.utilities import local_model_root, write_safetensors
from warp_nn.runtime.ace_step.planner import (
    AUDIO_CODE_TOKEN_BASE,
    AUDIO_CODE_TOKEN_STOP,
    AceAudioCodeDecoder,
    AceStepPlanner,
    audio_code_decoder_weight_names,
    audio_code_from_token_id,
    audio_code_token_id,
    format_planner_prompt,
    format_planner_unconditional_prompt,
    fsq_indices_to_codes,
    parse_planner_metadata,
)
from warp_nn.runtime.formats.safetensors import SafeTensorArchive
from warp_nn.runtime.qwen.causal import qwen3_causal_weight_names
from warp_nn.runtime.qwen.encoder import load_qwen3_encoder_config


def test_audio_token_grammar_and_fsq_inverse():
    assert audio_code_token_id(0) == 151669
    assert audio_code_token_id(63999) == 215668
    assert audio_code_from_token_id(151670) == 1
    with pytest.raises(ValueError, match="63999"):
        audio_code_token_id(64000)
    with pytest.raises(ValueError, match="not an ACE"):
        audio_code_from_token_id(151668)
    actual = fsq_indices_to_codes([0, 1, 7, 8, 63999])
    expected = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-0.75, -1, -1, -1, -1, -1],
            [0.75, -1, -1, -1, -1, -1],
            [-1, -0.75, -1, -1, -1, -1],
            [0.75, 0.75, 0.75, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_planner_prompt_and_metadata_contract():
    phase_one = format_planner_prompt("warm piano", "[Instrumental]")
    assert phase_one.startswith("<|im_start|>system\n# Instruction\n")
    assert "# Caption\nwarm piano\n\n# Lyric\n[Instrumental]" in phase_one
    assert phase_one.endswith("<|im_start|>assistant\n")
    phase_two = format_planner_prompt(
        "warm piano", "[Instrumental]", cot="bpm: 84\nkeyscale: C major"
    )
    assert phase_two.endswith("</think>\n\n")
    assert format_planner_unconditional_prompt().endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert "\nNO USER INPUT<|im_end|>\n" in format_planner_unconditional_prompt()
    assert parse_planner_metadata(
        "bpm: 84\ncaption: A warm piano theme that develops slowly.\n"
        "  A quiet countermelody enters later.\nduration: 12\n"
        "key: C major\ntime_signature: 4"
    ) == {
        "bpm": "84",
        "caption": (
            "A warm piano theme that develops slowly. "
            "A quiet countermelody enters later."
        ),
        "duration": "12",
        "keyscale": "C major",
        "timesignature": "4",
    }


class _FakeTokenizer:
    def encode(self, text):
        return [99] if text == "</think>" else [1, 2]

    def decode(self, token_ids):
        assert token_ids == [10]
        return (
            "bpm: 120\n"
            "caption: A developing piano miniature.\n"
            "duration: 99\n"
            "keyscale: C major\n"
            "language: en\n"
            "timesignature: 4/4"
        )


class _FakeRunner:
    config = {"vocab_size": 217204}
    tokenizer = _FakeTokenizer()

    def __init__(self):
        self.greedy = iter((10, 99))
        self.code_tokens = iter((AUDIO_CODE_TOKEN_BASE + 7, AUDIO_CODE_TOKEN_BASE + 9))
        self.ranges = []

    def prefill(self, token_ids):
        return object()

    def decode(self, token_id):
        return object()

    def sample_greedy(self, logits):
        return next(self.greedy)

    def sample_greedy_range(self, logits, start, stop):
        self.ranges.append((start, stop))
        return next(self.code_tokens)


def test_two_phase_planner_enforces_audio_vocabulary_and_rate():
    runner = _FakeRunner()
    planner = AceStepPlanner(runner)
    result = planner.generate(
        "piano",
        "[Instrumental]",
        duration_seconds=0.4,
        temperature=0.0,
        cfg_scale=1.0,
    )
    assert result.cot == (
        "bpm: 120\n"
        "caption: A developing piano miniature.\n"
        "duration: 0.4\n"
        "keyscale: C major\n"
        "language: en\n"
        "timesignature: 4"
    )
    assert result.metadata == {
        name: value
        for name, value in (line.split(": ", 1) for line in result.cot.splitlines())
    }
    assert result.audio_codes == (7, 9)
    assert runner.ranges == [
        (AUDIO_CODE_TOKEN_BASE, AUDIO_CODE_TOKEN_STOP),
        (AUDIO_CODE_TOKEN_BASE, AUDIO_CODE_TOKEN_STOP),
    ]


def _tiny_decoder_checkpoint(path: Path):
    config = {
        "fsq_input_levels": [8, 8, 8, 5, 5, 5],
        "fsq_input_num_quantizers": 1,
        "num_attention_pooler_hidden_layers": 2,
        "encoder_hidden_size": 4,
        "encoder_intermediate_size": 6,
        "encoder_num_attention_heads": 2,
        "encoder_num_key_value_heads": 1,
        "head_dim": 2,
        "rms_norm_eps": 1.0e-6,
        "rope_theta": 1000.0,
        "layer_types": ["full_attention", "full_attention"],
        "sliding_window": 8,
        "pool_window_size": 2,
        "audio_acoustic_hidden_dim": 3,
    }
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rng = np.random.default_rng(71)

    def matrix(rows, columns, scale=0.12):
        return rng.normal(0, scale, (rows, columns)).astype(np.float32)

    weights = {
        "tokenizer.quantizer.project_out.weight": matrix(4, 6),
        "tokenizer.quantizer.project_out.bias": matrix(1, 4)[0],
        "detokenizer.embed_tokens.weight": matrix(4, 4),
        "detokenizer.embed_tokens.bias": matrix(1, 4)[0],
        "detokenizer.special_tokens": matrix(2, 4)[None],
        "detokenizer.norm.weight": rng.uniform(0.8, 1.2, 4).astype(np.float32),
        "detokenizer.proj_out.weight": matrix(3, 4),
        "detokenizer.proj_out.bias": matrix(1, 3)[0],
    }
    for index in range(2):
        prefix = f"detokenizer.layers.{index}."
        weights.update(
            {
                prefix + "input_layernorm.weight": rng.uniform(0.8, 1.2, 4).astype(
                    np.float32
                ),
                prefix + "post_attention_layernorm.weight": rng.uniform(
                    0.8, 1.2, 4
                ).astype(np.float32),
                prefix + "self_attn.q_proj.weight": matrix(4, 4),
                prefix + "self_attn.k_proj.weight": matrix(2, 4),
                prefix + "self_attn.v_proj.weight": matrix(2, 4),
                prefix + "self_attn.q_norm.weight": rng.uniform(0.8, 1.2, 2).astype(
                    np.float32
                ),
                prefix + "self_attn.k_norm.weight": rng.uniform(0.8, 1.2, 2).astype(
                    np.float32
                ),
                prefix + "self_attn.o_proj.weight": matrix(4, 4),
                prefix + "mlp.gate_proj.weight": matrix(6, 4),
                prefix + "mlp.up_proj.weight": matrix(6, 4),
                prefix + "mlp.down_proj.weight": matrix(4, 6),
            }
        )
    write_safetensors(
        path / "model.safetensors",
        {
            name: ("F32", value.shape, np.ascontiguousarray(value).tobytes())
            for name, value in weights.items()
        },
    )
    return config, weights


def _rms(values, weight, epsilon):
    return (
        values
        / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + epsilon)
        * weight
    )


def _decoder_reference(indices, config, weights):
    scalar = fsq_indices_to_codes(indices)
    hidden = scalar @ weights["tokenizer.quantizer.project_out.weight"].T
    hidden += weights["tokenizer.quantizer.project_out.bias"]
    hidden = hidden @ weights["detokenizer.embed_tokens.weight"].T
    hidden += weights["detokenizer.embed_tokens.bias"]
    hidden = hidden[:, None, :] + weights["detokenizer.special_tokens"]
    positions = np.arange(config["pool_window_size"], dtype=np.float32)
    cosine = np.cos(positions[:, None])
    sine = np.sin(positions[:, None])

    def rotate(values):
        first, second = values[..., :1], values[..., 1:]
        return np.concatenate(
            (
                first * cosine[None] - second * sine[None],
                second * cosine[None] + first * sine[None],
            ),
            axis=-1,
        )

    for index in range(2):
        prefix = f"detokenizer.layers.{index}."
        x = _rms(hidden, weights[prefix + "input_layernorm.weight"], 1.0e-6)
        q = x @ weights[prefix + "self_attn.q_proj.weight"].T
        k = x @ weights[prefix + "self_attn.k_proj.weight"].T
        v = (x @ weights[prefix + "self_attn.v_proj.weight"].T).reshape(-1, 2, 1, 2)
        q = _rms(
            q.reshape(-1, 2, 2, 2), weights[prefix + "self_attn.q_norm.weight"], 1.0e-6
        )
        k = _rms(
            k.reshape(-1, 2, 1, 2), weights[prefix + "self_attn.k_norm.weight"], 1.0e-6
        )
        q, k = rotate(q), rotate(k)
        attention = np.empty_like(q)
        for batch in range(len(indices)):
            for token in range(2):
                for head in range(2):
                    scores = q[batch, token, head] @ k[batch, :, 0].T / np.sqrt(2)
                    probabilities = np.exp(scores - scores.max())
                    probabilities /= probabilities.sum()
                    attention[batch, token, head] = probabilities @ v[batch, :, 0]
        projected = (
            attention.reshape(-1, 2, 4) @ weights[prefix + "self_attn.o_proj.weight"].T
        )
        residual = hidden + projected
        x = _rms(residual, weights[prefix + "post_attention_layernorm.weight"], 1.0e-6)
        gate = x @ weights[prefix + "mlp.gate_proj.weight"].T
        up = x @ weights[prefix + "mlp.up_proj.weight"].T
        hidden = (
            residual
            + (gate / (1 + np.exp(-gate)) * up)
            @ weights[prefix + "mlp.down_proj.weight"].T
        )
    hidden = _rms(hidden, weights["detokenizer.norm.weight"], 1.0e-6)
    output = hidden @ weights["detokenizer.proj_out.weight"].T
    return output + weights["detokenizer.proj_out.bias"]


def test_audio_code_decoder_matches_numpy_oracle(tmp_path):
    config, weights = _tiny_decoder_checkpoint(tmp_path / "decoder")
    decoder = AceAudioCodeDecoder(
        tmp_path / "decoder", dtype=wp.float16, device="cpu", use_cublas=False
    )
    actual = decoder.decode([0, 63999]).numpy()[0]
    expected = _decoder_reference([0, 63999], config, weights).reshape(-1, 3)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-2, atol=8.0e-3)


def test_local_ace_planner_and_decoder_manifests():
    root = local_model_root() / "ACE-Step"
    planner = root / "acestep-5Hz-lm-4B"
    decoder = root / "acestep-v15-xl-sft"
    if not planner.is_dir() or not decoder.is_dir():
        pytest.skip("local ACE-Step planner/XL checkpoints are unavailable")
    planner_config = load_qwen3_encoder_config(planner)
    planner_archive = SafeTensorArchive(planner)
    assert set(qwen3_causal_weight_names(planner_config, planner_archive.names)) <= set(
        planner_archive.names
    )
    assert planner_archive.metadata("model.embed_tokens.weight").shape == (217204, 2560)
    decoder_archive = SafeTensorArchive(decoder)
    assert set(audio_code_decoder_weight_names()) <= set(decoder_archive.names)
    assert decoder_archive.metadata("tokenizer.quantizer.project_out.weight").shape == (
        2048,
        6,
    )
    assert decoder_archive.metadata("detokenizer.proj_out.weight").shape == (64, 2048)
