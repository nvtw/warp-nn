# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import codecs
import json
from pathlib import Path

import numpy as np

from examples.qwen3_onnx_chat import _generate
from warp_nn.runtime import Qwen3Tokenizer, parse_qwen_tool_calls, sample_token
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
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    streamed = "".join(decoder.decode(tokenizer.token_bytes(token_id)) for token_id in token_ids)
    streamed += decoder.decode(b"", final=True)
    assert streamed == text

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


def test_qwen_tool_template_and_parser(tmp_path):
    path = tmp_path / "tokenizer.json"
    _write_tokenizer(path)
    tokenizer = Qwen3Tokenizer(path)
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Read it."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": "read", "arguments": '{"path":"README.md"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Warp NN"},
    ]
    formatted = tokenizer.format_chat(messages, tools=tools, enable_thinking=False)
    assert '"name":"read"' in formatted
    assert "<function=read>\n<parameter=path>\nREADME.md\n</parameter>" in formatted
    assert "<tool_response>\nWarp NN\n</tool_response>" in formatted
    assert formatted.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    text, calls = parse_qwen_tool_calls(
        'I will inspect it.\n<tool_call>\n<function=read>\n<parameter=path>\n"README.md"\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    assert text == "I will inspect it."
    assert calls == [{"name": "read", "arguments": {"path": "README.md"}}]


def test_console_generation_streams_text_and_hides_tool_markup(capsys):
    class Runner:
        def sample_greedy(self, logits):
            return logits

        def decode(self, token_id):
            return {1: 2, 2: 0}[token_id]

    class Tokenizer:
        eos_token_id = 0

        pieces = {
            1: b"Checking. <tool_",
            2: b"call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>",
        }

        def token_bytes(self, token_id, skip_special_tokens=False):
            return self.pieces.get(token_id, b"")

        def decode(self, token_ids, skip_special_tokens=False):
            return b"".join(self.pieces.get(token_id, b"") for token_id in token_ids).decode()

        def parse_tool_calls(self, text):
            return parse_qwen_tool_calls(text)

    cached_ids = []
    generated, text, calls = _generate(Runner(), Tokenizer(), 1, 4, 0.0, cached_ids, "<tool_call>")
    assert capsys.readouterr().out == "Checking. "
    assert generated == [1, 2, 0]
    assert cached_ids == [1, 2]
    assert text == "Checking."
    assert calls == [{"name": "read_file", "arguments": {"path": "README.md"}}]
