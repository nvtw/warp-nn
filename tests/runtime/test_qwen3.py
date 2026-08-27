# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np

from warp_nn.runtime import Qwen3Tokenizer, sample_token
from warp_nn.runtime.qwen3 import _BYTE_ENCODER


def _write_tokenizer(path: Path):
    vocabulary = {character: index for index, character in enumerate(_BYTE_ENCODER.values())}
    merges = [["h", "e"], ["he", "l"], ["he" + "l", "l"], ["hell", "o"]]
    for left, right in merges:
        token = left + right
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    added_tokens = [
        {
            "id": len(vocabulary),
            "content": "<|endoftext|>",
            "special": True,
        },
        {
            "id": len(vocabulary) + 1,
            "content": "<|im_start|>",
            "special": True,
        },
        {
            "id": len(vocabulary) + 2,
            "content": "<|im_end|>",
            "special": True,
        },
    ]
    path.write_text(
        json.dumps(
            {
                "normalizer": {"type": "NFC"},
                "added_tokens": added_tokens,
                "model": {"type": "BPE", "vocab": vocabulary, "merges": merges},
            }
        ),
        encoding="utf-8",
    )


def test_qwen_tokenizer_and_chat_template(tmp_path):
    path = tmp_path / "tokenizer.json"
    _write_tokenizer(path)
    tokenizer = Qwen3Tokenizer(path)

    text = "hello  Café\n你好 👋"
    token_ids = tokenizer.encode(text)
    assert tokenizer.decode(token_ids) == text
    assert tokenizer._vocabulary["hello"] in token_ids

    messages = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hello"}]
    formatted = tokenizer.format_chat(messages, enable_thinking=False)
    assert formatted == (
        "<|im_start|>system\nBe concise.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert tokenizer.decode(tokenizer.encode_chat(messages, enable_thinking=False)) == formatted


def test_sample_token():
    logits = np.array([[[0.0, 1.0, 4.0, 3.0]]], dtype=np.float16)
    assert sample_token(logits, temperature=0.0) == 2
    rng = np.random.default_rng(73)
    samples = {sample_token(logits, top_k=2, rng=rng) for _ in range(20)}
    assert samples <= {2, 3}
    assert samples
