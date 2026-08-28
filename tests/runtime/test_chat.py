# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from warp_nn.runtime.chat import generate_tokens


class _Runner:
    cache_capacity = 8

    def __init__(self):
        self.tokens = iter((2, 3))

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
        return self._logits()

    def sample_greedy(self, logits):
        return int(np.argmax(logits))


class _Tokenizer:
    eos_token_id = 3


def test_generate_tokens_uses_common_runner_interface():
    assert list(generate_tokens(_Runner(), _Tokenizer(), [1], max_new_tokens=4)) == [2, 3]
