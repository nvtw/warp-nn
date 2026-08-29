# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from warp_nn.runtime.chat import generate_tokens, split_reasoning, split_tool_prefix


class _Runner:
    cache_capacity = 8

    def __init__(self):
        self.tokens = iter((2, 3))
        self.decoded = []

    def _logits(self):
        token = next(self.tokens)
        values = np.zeros((1, 1, 4), dtype=np.float16)
        values[0, 0, token] = 1.0
        return values

    def prefill(self, token_ids):
        assert token_ids == [1]
        return self._logits()

    def decode(self, token_id):
        assert token_id == 2
        self.decoded.append(token_id)
        return self._logits()

    def sample_greedy(self, logits):
        return int(np.argmax(logits))


class _Tokenizer:
    eos_token_id = 3


class _TopKRunner(_Runner):
    def __init__(self):
        super().__init__()
        self.top_k_reads = 0

    def read_top_k(self, logits, top_k):
        self.top_k_reads += 1
        values = np.asarray(logits).reshape(-1, logits.shape[-1])[-1]
        tokens = np.argsort(values)[::-1][:top_k]
        return values[tokens], tokens


class _PenaltyRunner:
    cache_capacity = 8

    def prefill(self, token_ids):
        del token_ids
        return np.array([[[0.0, 0.0, 2.0, 1.0]]], dtype=np.float32)

    def decode(self, token_id):
        assert token_id == 2
        return np.array([[[0.0, 0.0, 2.0, 1.5]]], dtype=np.float32)

    def sample_greedy(self, logits):
        return int(np.argmax(logits))


def test_generate_tokens_uses_common_runner_interface():
    assert list(generate_tokens(_Runner(), _Tokenizer(), [1], max_new_tokens=4)) == [
        2,
        3,
    ]


def test_generate_tokens_uses_bounded_top_k_reader():
    runner = _TopKRunner()
    assert list(
        generate_tokens(
            runner,
            _Tokenizer(),
            [1],
            max_new_tokens=4,
            temperature=0.01,
            top_k=2,
            top_p=0.95,
            seed=7,
        )
    ) == [2, 3]
    assert runner.top_k_reads == 2


def test_generate_tokens_validates_sampling_before_fast_top_k():
    with pytest.raises(ValueError, match="invalid top_k, top_p, or presence_penalty"):
        list(
            generate_tokens(
                _TopKRunner(),
                _Tokenizer(),
                [1],
                max_new_tokens=1,
                temperature=1.0,
                top_k=2,
                top_p=1.1,
            )
        )


def test_generate_tokens_top_k_one_preserves_presence_penalty():
    assert list(
        generate_tokens(
            _PenaltyRunner(),
            _Tokenizer(),
            [1],
            max_new_tokens=3,
            temperature=1.0,
            top_k=1,
            presence_penalty=1.0,
            seed=4,
        )
    ) == [2, 3]


def test_generate_tokens_enqueues_decode_before_yielding():
    runner = _Runner()
    tokens = generate_tokens(runner, _Tokenizer(), [1], max_new_tokens=4)
    assert next(tokens) == 2
    assert runner.decoded == [2]


def test_generate_tokens_accepts_multiple_end_tokens():
    tokenizer = _Tokenizer()
    tokenizer.eos_token_ids = (2, 3)
    assert list(generate_tokens(_Runner(), tokenizer, [1], max_new_tokens=4)) == [2]


def test_split_tool_prefix_preserves_partial_marker():
    assert split_tool_prefix("answer<tool_", "<tool_call>") == (
        "answer",
        "<tool_",
        False,
    )
    assert split_tool_prefix("<tool_call>body", "<tool_call>") == (
        "",
        "<tool_call>body",
        True,
    )


def test_split_reasoning():
    assert split_reasoning("Reason\n</think>\n\nAnswer", True) == ("Answer", "Reason")
    assert split_reasoning("Answer", False) == ("Answer", None)
