# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import codecs
import json
from pathlib import Path

import numpy as np

from examples.qwen_chat import _generate
from warp_nn.runtime import Qwen3Tokenizer, parse_qwen_tool_calls, sample_token
from warp_nn.runtime.chat import ChatEncodingCache
from warp_nn.runtime.qwen3 import _BYTE_ENCODER


def _write_tokenizer(path: Path):
    vocabulary = {
        character: index for index, character in enumerate(_BYTE_ENCODER.values())
    }
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
    streamed = "".join(
        decoder.decode(tokenizer.token_bytes(token_id)) for token_id in token_ids
    )
    streamed += decoder.decode(b"", final=True)
    assert streamed == text

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]
    formatted = tokenizer.format_chat(messages, enable_thinking=False)
    assert formatted == (
        "<|im_start|>system\nBe concise.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert (
        tokenizer.decode(tokenizer.encode_chat(messages, enable_thinking=False))
        == formatted
    )

    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [tokenizer.eos_token_id, tokenizer.pad_token_id]}),
        encoding="utf-8",
    )
    tokenizer = Qwen3Tokenizer(path)
    assert tokenizer.eos_token_ids == (tokenizer.eos_token_id, tokenizer.pad_token_id)

    cached = tokenizer.encode_chat(messages, enable_thinking=False) + tokenizer.encode(
        "Hello"
    )
    messages.extend(
        [
            {
                "role": "assistant",
                "content": tokenizer.generation_prefix(False) + "Hello",
            },
            {"role": "user", "content": "Again"},
        ]
    )
    assert (
        tokenizer.encode_chat(messages, enable_thinking=False)[: len(cached)] == cached
    )


def test_incremental_chat_encoding_matches_full_history(tmp_path):
    path = tmp_path / "tokenizer.json"
    _write_tokenizer(path)
    tokenizer = Qwen3Tokenizer(path)
    cache = ChatEncodingCache(tokenizer)
    messages = [{"role": "user", "content": "Hello"}]

    initial = cache.encode_chat(messages, enable_thinking=False)
    assert initial == tokenizer.encode_chat(messages, enable_thinking=False)
    generated = tokenizer.encode("Hello") + [tokenizer.eos_token_id]
    cache.extend_raw(generated)
    messages.extend(
        [
            {
                "role": "assistant",
                "content": tokenizer.generation_prefix(False) + "Hello",
            },
            {"role": "user", "content": "Again"},
        ]
    )
    assert cache.encode_chat(messages, enable_thinking=False) == tokenizer.encode_chat(
        messages, enable_thinking=False
    )

    edited = [{"role": "user", "content": "Changed"}]
    assert cache.encode_chat(edited, enable_thinking=False) == tokenizer.encode_chat(
        edited, enable_thinking=False
    )


def test_nemotron_tokenizer_metadata(tmp_path):
    path = tmp_path / "tokenizer.json"
    _write_tokenizer(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["normalizer"] = None
    data["added_tokens"] = [
        item for item in data["added_tokens"] if item["content"] != "<|endoftext|>"
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 2, "pad_token_id": 0}), encoding="utf-8"
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": 'set enable_thinking set system_message = "" <think></think>'
            }
        ),
        encoding="utf-8",
    )

    tokenizer = Qwen3Tokenizer(path)
    text = "Cafe\u0301"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.pad_token_id == 0
    assert tokenizer.eos_token_ids == (2,)
    assert tokenizer.default_enable_thinking
    assert tokenizer.generation_prefix(False) == "<think></think>"
    assert tokenizer.format_chat(
        [{"role": "user", "content": "Hello"}], enable_thinking=False
    ) == (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>"
    )


def test_sample_token():
    logits = np.array([[[0.0, 1.0, 4.0, 3.0]]], dtype=np.float16)
    assert sample_token(logits, temperature=0.0) == 2
    rng = np.random.default_rng(73)
    samples = {sample_token(logits, top_k=2, rng=rng) for _ in range(20)}
    assert samples <= {2, 3}
    assert samples
    assert (
        sample_token(
            logits,
            temperature=0.01,
            presence_penalty=2.0,
            previous_tokens=[2],
            rng=np.random.default_rng(1),
        )
        == 3
    )


def test_qwen_tool_template_and_parser(tmp_path):
    path = tmp_path / "tokenizer.json"
    _write_tokenizer(path)
    tokenizer = Qwen3Tokenizer(path)
    tools = [
        {
            "type": "function",
            "function": {"name": "read", "parameters": {"type": "object"}},
        }
    ]
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Read it."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                }
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
            return b"".join(
                self.pieces.get(token_id, b"") for token_id in token_ids
            ).decode()

        def parse_tool_calls(self, text):
            return parse_qwen_tool_calls(text)

    cached_ids = []
    generated, text, calls = _generate(
        Runner(), Tokenizer(), 1, 4, 0.0, cached_ids, "<tool_call>"
    )
    assert capsys.readouterr().out == "Checking. "
    assert generated == [1, 2, 0]
    assert cached_ids == [1, 2]
    assert text == "Checking."
    assert calls == [{"name": "read_file", "arguments": {"path": "README.md"}}]

    cached_ids = []
    generated, text, calls = _generate(
        Runner(),
        Tokenizer(),
        1,
        4,
        0.0,
        cached_ids,
        "<tool_call>",
        cancelled=lambda: True,
    )
    assert (generated, text, calls, cached_ids) == ([], "", [], [])


def test_qwen3_json_tool_dialect(tmp_path):
    path = tmp_path / "tokenizer.json"
    _write_tokenizer(path)
    (tmp_path / "chat_template.jinja").write_text(
        '"arguments": <args-json-object>', encoding="utf-8"
    )
    tokenizer = Qwen3Tokenizer(path)
    formatted = tokenizer.format_chat(
        [
            {"role": "user", "content": "Read it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
        ],
        tools=[
            {"type": "function", "function": {"name": "read_file", "parameters": {}}}
        ],
    )
    assert (
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "README.md"}}\n</tool_call>'
        in formatted
    )
    text, calls = parse_qwen_tool_calls(
        'Checking.\n<tool_call>\n{"name":"read_file","arguments":{"path":"README.md"}}\n</tool_call>'
    )
    assert text == "Checking."
    assert calls == [{"name": "read_file", "arguments": {"path": "README.md"}}]


def test_qwen38_reasoning_template_controls(tmp_path):
    path = tmp_path / "tokenizer.json"
    _write_tokenizer(path)
    (tmp_path / "chat_template.jinja").write_text(
        "reasoning_effort <think>\\n", encoding="utf-8"
    )
    tokenizer = Qwen3Tokenizer(path)

    assert tokenizer.default_enable_thinking
    formatted = tokenizer.format_chat(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
        reasoning_effort="low",
    )
    assert "Reasoning effort is set to low." in formatted
    assert "Be concise." in formatted
    assert formatted.endswith("<|im_start|>assistant\n<think>\n")

    history = tokenizer.format_chat(
        [{"role": "assistant", "content": "Answer", "reasoning_content": "Reason"}],
        add_generation_prompt=False,
        reasoning_effort="medium",
    )
    assert "<think>\nReason\n</think>\n\nAnswer" in history
    assert "<think>" not in tokenizer.format_chat(
        [{"role": "assistant", "content": "<think>\nReason\n</think>\n\nAnswer"}],
        add_generation_prompt=False,
        reasoning_effort="medium",
        preserve_thinking=False,
    )
