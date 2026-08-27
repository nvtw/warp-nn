# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal stateful runner for Qwen-style decoder ONNX graphs."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import warp as wp

from warp_nn.runtime.onnx_runtime import OnnxRuntime


def _byte_alphabet() -> tuple[dict[int, str], dict[str, int]]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = list(values)
    missing = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + missing)
            missing += 1
    encoder = dict(zip(values, map(chr, characters)))
    return encoder, {character: value for value, character in encoder.items()}


_BYTE_ENCODER, _BYTE_DECODER = _byte_alphabet()


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _pretokenize(text: str):
    """Implement the Unicode split expression stored in Qwen's tokenizer."""
    index = 0
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    while index < len(text):
        contraction = next((item for item in contractions if text[index : index + len(item)].lower() == item), None)
        if contraction is not None:
            yield text[index : index + len(contraction)]
            index += len(contraction)
            continue

        letter_start = index
        character = text[index]
        if character not in "\r\n" and not _is_letter(character) and not _is_number(character):
            letter_start += 1
        if letter_start < len(text) and _is_letter(text[letter_start]):
            end = letter_start + 1
            while end < len(text) and _is_letter(text[end]):
                end += 1
            yield text[index:end]
            index = end
            continue
        if _is_number(character):
            yield character
            index += 1
            continue

        symbol_start = index + 1 if character == " " else index
        if symbol_start < len(text):
            symbol = text[symbol_start]
            if not symbol.isspace() and not _is_letter(symbol) and not _is_number(symbol):
                end = symbol_start + 1
                while end < len(text):
                    symbol = text[end]
                    if symbol.isspace() or _is_letter(symbol) or _is_number(symbol):
                        break
                    end += 1
                while end < len(text) and text[end] in "\r\n":
                    end += 1
                yield text[index:end]
                index = end
                continue

        if character.isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            last_newline = max(text.rfind("\r", index, end), text.rfind("\n", index, end))
            if last_newline >= index:
                end = last_newline + 1
            elif end < len(text) and end - index > 1:
                end -= 1
            yield text[index:end]
            index = end
            continue

        yield character
        index += 1


class Qwen3Tokenizer:
    """Dependency-free ByteLevel-BPE tokenizer for Qwen3 tokenizer JSON files."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if path.is_dir():
            path /= "tokenizer.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data["model"]
        if model["type"] != "BPE" or data["normalizer"] != {"type": "NFC"}:
            raise ValueError("Qwen3Tokenizer: expected Qwen's NFC ByteLevel-BPE tokenizer")
        self._vocabulary = model["vocab"]
        self._tokens = {token_id: token for token, token_id in self._vocabulary.items()}
        self._merge_ranks = {tuple(pair): rank for rank, pair in enumerate(model["merges"])}
        self._added_tokens = {item["content"]: item["id"] for item in data["added_tokens"]}
        self._added_by_id = {item["id"]: item for item in data["added_tokens"]}
        alternatives = "|".join(re.escape(token) for token in sorted(self._added_tokens, key=len, reverse=True))
        self._added_pattern = re.compile(f"({alternatives})")
        self.eos_token_id = self._added_tokens["<|im_end|>"]
        self.pad_token_id = self._added_tokens["<|endoftext|>"]

    @lru_cache(maxsize=8192)
    def _bpe(self, piece: str) -> tuple[str, ...]:
        symbols = list(piece)
        while len(symbols) > 1:
            candidates = {(symbols[index], symbols[index + 1]) for index in range(len(symbols) - 1)}
            pair = min(candidates, key=lambda item: self._merge_ranks.get(item, len(self._merge_ranks)))
            if pair not in self._merge_ranks:
                break
            merged = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        return tuple(symbols)

    def encode(self, text: str) -> list[int]:
        """Encode text, recognizing Qwen's added control tokens."""
        token_ids = []
        for chunk in self._added_pattern.split(text):
            if not chunk:
                continue
            if chunk in self._added_tokens:
                token_ids.append(self._added_tokens[chunk])
                continue
            chunk = unicodedata.normalize("NFC", chunk)
            for piece in _pretokenize(chunk):
                byte_piece = "".join(_BYTE_ENCODER[value] for value in piece.encode("utf-8"))
                token_ids.extend(self._vocabulary[token] for token in self._bpe(byte_piece))
        return token_ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to UTF-8 text."""
        output = []
        byte_characters = []

        def flush_bytes():
            if byte_characters:
                output.append(
                    bytes(_BYTE_DECODER[character] for character in byte_characters).decode("utf-8", "replace")
                )
                byte_characters.clear()

        for token_id in token_ids:
            added = self._added_by_id.get(int(token_id))
            if added is not None:
                flush_bytes()
                if not skip_special_tokens or not added["special"]:
                    output.append(added["content"])
            else:
                byte_characters.extend(self._tokens[int(token_id)])
        flush_bytes()
        return "".join(output)

    def format_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        """Format ordinary system/user/assistant messages using Qwen3's template."""
        formatted = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                raise ValueError("Qwen3Tokenizer.format_chat supports text system/user/assistant messages")
            formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            formatted.append("<|im_start|>assistant\n")
            if not enable_thinking:
                formatted.append("<think>\n\n</think>\n\n")
        return "".join(formatted)

    def encode_chat(self, messages: Sequence[Mapping[str, str]], **kwargs) -> list[int]:
        """Format and encode a chat prompt."""
        return self.encode(self.format_chat(messages, **kwargs))


def sample_token(
    logits: wp.array | np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one token from the last logits row on the host."""
    values = logits.numpy() if hasattr(logits, "numpy") else np.asarray(logits)
    values = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])[-1]
    if temperature <= 0.0:
        return int(np.argmax(values))
    if top_k < 0 or not 0.0 < top_p <= 1.0:
        raise ValueError("sample_token requires top_k >= 0 and 0 < top_p <= 1")
    candidates = np.arange(values.size)
    if 0 < top_k < values.size:
        candidates = np.argpartition(values, -top_k)[-top_k:]
        values = values[candidates]
    values = values / temperature
    probabilities = np.exp(values - np.max(values))
    probabilities /= probabilities.sum()
    if top_p < 1.0:
        order = np.argsort(probabilities)[::-1]
        keep = np.cumsum(probabilities[order]) - probabilities[order] < top_p
        candidates = candidates[order[keep]]
        probabilities = probabilities[order[keep]]
        probabilities /= probabilities.sum()
    return int((rng or np.random.default_rng()).choice(candidates, p=probabilities))


class Qwen3OnnxRunner:
    """Run prefill and token-by-token decode while keeping weights on device."""

    def __init__(self, path: str, device: str | wp.Device | None = None):
        self.runtime = OnnxRuntime(path, device=device)
        self._past_names = [name for name in self.runtime.input_names if name.startswith("past_key_values.")]
        self._present_for_past = {
            name: f"present.{name.split('.')[1]}.{name.split('.')[2]}" for name in self._past_names
        }
        if not self._past_names or any(
            name not in self.runtime.output_names for name in self._present_for_past.values()
        ):
            raise ValueError("Qwen3OnnxRunner: model does not expose compatible past/present KV-cache tensors")
        self._cache_shapes = {name: self.runtime._shapes[name] for name in self._past_names}
        rotary_lengths = [
            self.runtime._shapes[name][0] for name in ("cos_cache", "sin_cache") if name in self.runtime._shapes
        ]
        if not rotary_lengths:
            raise ValueError("Qwen3OnnxRunner: model does not contain rotary embedding caches")
        self.max_sequence_length = min(rotary_lengths)
        self._past: dict[str, wp.array] = {}
        self.sequence_length = 0

    def reset(self) -> None:
        """Discard the current conversation's KV cache."""
        self._past.clear()
        self.sequence_length = 0

    def prefill(self, token_ids: Sequence[int]) -> wp.array:
        """Reset state, process a prompt, and return its logits."""
        self.reset()
        return self._forward(token_ids)

    def decode(self, token_id: int) -> wp.array:
        """Append one token and return its logits."""
        if self.sequence_length == 0:
            raise RuntimeError("Qwen3OnnxRunner.decode requires a preceding prefill call")
        return self._forward([token_id])

    def _forward(self, token_ids: Sequence[int]) -> wp.array:
        current_length = len(token_ids)
        if current_length == 0:
            raise ValueError("Qwen3OnnxRunner: token_ids must not be empty")
        total_length = self.sequence_length + current_length
        if total_length > self.max_sequence_length:
            raise ValueError(
                f"Qwen3OnnxRunner: sequence length {total_length} exceeds model limit {self.max_sequence_length}"
            )

        shapes = {
            "input_ids": (1, current_length),
            "attention_mask": (1, total_length),
        }
        for name, base_shape in self._cache_shapes.items():
            shapes[name] = (1, base_shape[1], self.sequence_length, base_shape[3])
        self.runtime.resize_inputs(shapes)

        device = self.runtime._device
        inputs = {
            "input_ids": wp.array(np.asarray(token_ids, dtype=np.int64)[None, :], dtype=wp.int64, device=device),
            "attention_mask": wp.ones((1, total_length), dtype=wp.int64, device=device),
        }
        for name, shape in shapes.items():
            if name in ("input_ids", "attention_mask"):
                continue
            if name in self._past:
                inputs[name] = self._past[name]
            else:
                inputs[name] = wp.zeros(shape, dtype=self.runtime._input_dtypes[name], device=device)

        outputs = self.runtime(inputs)
        self._past = {name: outputs[present] for name, present in self._present_for_past.items()}
        self.sequence_length = total_length
        return outputs["logits"]
