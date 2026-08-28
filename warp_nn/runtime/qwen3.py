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

from warp_nn.runtime.kernels import (
    _get_greedy_argmax_kernels,
    _initialize_attention_mask,
    _initialize_generation_state,
    _set_decode_token,
)
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

_TOOL_PROMPT = """# Tools

You have access to the following functions:

<tools>{tools}
</tools>

If you choose to call a function, reply with no suffix in this format:

<tool_call>
<function=function_name>
<parameter=parameter_name>
value
</parameter>
</function>
</tool_call>

Required parameters must be specified. You may reason before the function call, but not after it. If no function
applies, answer normally."""


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

    def token_bytes(self, token_id: int, skip_special_tokens: bool = False) -> bytes:
        """Return one token as bytes for use with an incremental UTF-8 decoder."""
        added = self._added_by_id.get(int(token_id))
        if added is not None:
            if skip_special_tokens and added["special"]:
                return b""
            return added["content"].encode("utf-8")
        return bytes(_BYTE_DECODER[character] for character in self._tokens[int(token_id)])

    def format_chat(
        self,
        messages: Sequence[Mapping[str, object]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        tools: Sequence[Mapping[str, object]] | None = None,
    ) -> str:
        """Format text messages and OpenAI function tools using Qwen's template."""
        formatted = []
        first = 0
        if tools:
            definitions = "".join(f"\n{json.dumps(tool, ensure_ascii=False, separators=(',', ':'))}" for tool in tools)
            system = _TOOL_PROMPT.format(tools=definitions)
            if messages and messages[0].get("role") in ("system", "developer"):
                content = messages[0].get("content")
                if not isinstance(content, str):
                    raise ValueError("Qwen3Tokenizer.format_chat requires text message content")
                if content.strip():
                    system += "\n\n" + content.strip()
                first = 1
            formatted.append(f"<|im_start|>system\n{system}<|im_end|>\n")

        index = first
        while index < len(messages):
            message = messages[index]
            role = message.get("role")
            content = message.get("content")
            if role == "developer":
                role = "system"
            if role == "tool":
                responses = []
                while index < len(messages) and messages[index].get("role") == "tool":
                    tool_content = messages[index].get("content")
                    if not isinstance(tool_content, str):
                        raise ValueError("Qwen3Tokenizer.format_chat requires text tool results")
                    responses.append(f"\n<tool_response>\n{tool_content.strip()}\n</tool_response>")
                    index += 1
                formatted.append(f"<|im_start|>user{''.join(responses)}<|im_end|>\n")
                continue
            if role not in ("system", "user", "assistant") or content is not None and not isinstance(content, str):
                raise ValueError("Qwen3Tokenizer.format_chat supports text OpenAI chat messages")
            body = "" if content is None else content.strip()
            if role == "assistant":
                for tool_call in message.get("tool_calls") or ():
                    function = tool_call.get("function", tool_call)
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, Mapping):
                        raise ValueError("Qwen3Tokenizer.format_chat requires object-valued tool arguments")
                    parameters = "".join(
                        f"<parameter={name}>\n{_format_tool_value(value)}\n</parameter>\n"
                        for name, value in arguments.items()
                    )
                    separator = "\n\n" if body else ""
                    body += (
                        f"{separator}<tool_call>\n<function={function['name']}>\n"
                        f"{parameters}</function>\n</tool_call>"
                    )
            formatted.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
            index += 1
        if add_generation_prompt:
            formatted.append("<|im_start|>assistant\n")
            if not enable_thinking:
                formatted.append("<think>\n\n</think>\n\n")
        return "".join(formatted)

    def encode_chat(self, messages: Sequence[Mapping[str, str]], **kwargs) -> list[int]:
        """Format and encode a chat prompt."""
        return self.encode(self.format_chat(messages, **kwargs))


def _format_tool_value(value: object) -> str:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def parse_qwen_tool_calls(text: str) -> tuple[str, list[dict[str, object]]]:
    """Extract Qwen XML function calls and return remaining assistant text."""
    calls = []
    spans = []
    pattern = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>", re.DOTALL)
    parameter_pattern = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
    for match in pattern.finditer(text):
        arguments = {}
        for parameter in parameter_pattern.finditer(match.group(2)):
            value = parameter.group(2).strip()
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
            arguments[parameter.group(1).strip()] = value
        calls.append({"name": match.group(1).strip(), "arguments": arguments})
        spans.append(match.span())
    for start, end in reversed(spans):
        text = text[:start] + text[end:]
    return text.strip(), calls


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

    def __init__(
        self,
        path: str,
        device: str | wp.Device | None = None,
        cache_capacity: int | None = None,
        prefill_chunk_size: int | None = None,
        use_cublas: bool = True,
    ):
        self.runtime = OnnxRuntime(path, device=device, use_cublas=use_cublas, _defer_preallocation=True)
        self._past_names = [name for name in self.runtime.input_names if name.startswith("past_key_values.")]
        self._present_for_past = {
            name: f"present.{name.split('.')[1]}.{name.split('.')[2]}" for name in self._past_names
        }
        if not self._past_names or any(
            name not in self.runtime.output_names for name in self._present_for_past.values()
        ):
            raise ValueError("Qwen3OnnxRunner: model does not expose compatible past/present KV-cache tensors")
        self._cache_shapes = {name: self.runtime._shapes[name] for name in self._past_names}
        self._variable_cache_names = {
            name for name in self._past_names if name.endswith(".key") or name.endswith(".value")
        }
        rotary_lengths = [
            self.runtime._shapes[name][0] for name in ("cos_cache", "sin_cache") if name in self.runtime._shapes
        ]
        config_path = Path(path).with_name("config.json")
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        text_config = config.get("text_config", config)
        configured_length = int(text_config.get("max_position_embeddings", 0))
        if not rotary_lengths and not configured_length:
            raise ValueError("Qwen3OnnxRunner: model does not declare a maximum sequence length")
        self.max_sequence_length = min(rotary_lengths) if rotary_lengths else configured_length
        self.cache_capacity = cache_capacity or self.max_sequence_length
        if not 0 < self.cache_capacity <= self.max_sequence_length:
            raise ValueError("Qwen3OnnxRunner: cache_capacity must be within the model's rotary cache")
        if prefill_chunk_size is not None and not 1 < prefill_chunk_size <= self.cache_capacity:
            raise ValueError("Qwen3OnnxRunner: prefill_chunk_size must be between 2 and cache_capacity")
        self.prefill_chunk_size = prefill_chunk_size
        self._cache = {
            name: wp.zeros(
                (1, shape[1], self.cache_capacity, shape[3]) if name in self._variable_cache_names else shape,
                dtype=self.runtime._input_dtypes[name],
                device=self.runtime._device,
            )
            for name, shape in self._cache_shapes.items()
        }
        self._decode_input_ids = wp.zeros((1, 1), dtype=wp.int64, device=self.runtime._device)
        self._decode_position_ids = wp.zeros((1, 1), dtype=wp.int64, device=self.runtime._device)
        self._decode_attention_mask = wp.zeros((1, self.cache_capacity), dtype=wp.int64, device=self.runtime._device)
        decode_shapes = {"input_ids": (1, 1), "attention_mask": (1, self.cache_capacity)}
        if "position_ids" in self.runtime.input_names:
            decode_shapes["position_ids"] = (1, 1)
        decode_shapes.update({name: tuple(cache.shape) for name, cache in self._cache.items()})
        self._decode_runtime = self.runtime._fork(decode_shapes, share_kv_cache=True)
        self._decode_inputs = {
            "input_ids": self._decode_input_ids,
            "attention_mask": self._decode_attention_mask,
            **self._cache,
        }
        if "position_ids" in self.runtime.input_names:
            self._decode_inputs["position_ids"] = self._decode_position_ids
        self._chunk_runtime = None
        if prefill_chunk_size is not None:
            self._chunk_input_ids = wp.zeros((1, prefill_chunk_size), dtype=wp.int64, device=self.runtime._device)
            self._chunk_position_ids = wp.zeros((1, prefill_chunk_size), dtype=wp.int64, device=self.runtime._device)
            chunk_shapes = {
                "input_ids": (1, prefill_chunk_size),
                "attention_mask": (1, self.cache_capacity),
                **{name: tuple(cache.shape) for name, cache in self._cache.items()},
            }
            if "position_ids" in self.runtime.input_names:
                chunk_shapes["position_ids"] = (1, prefill_chunk_size)
            self._chunk_runtime = self.runtime._fork(chunk_shapes, share_kv_cache=True)
            self._chunk_inputs = {
                "input_ids": self._chunk_input_ids,
                "attention_mask": self._decode_attention_mask,
                **self._cache,
            }
            if "position_ids" in self.runtime.input_names:
                self._chunk_inputs["position_ids"] = self._chunk_position_ids
        self._decode_position = wp.zeros(1, dtype=wp.int32, device=self.runtime._device)
        self._generated_count = wp.zeros(1, dtype=wp.int32, device=self.runtime._device)
        self._generated_ids = wp.zeros(self.cache_capacity, dtype=wp.int64, device=self.runtime._device)
        self._generation_finished = wp.zeros(1, dtype=wp.int32, device=self.runtime._device)
        self._sample_partial_values = wp.empty(128, dtype=wp.float32, device=self.runtime._device)
        self._sample_partial_tokens = wp.empty(128, dtype=wp.int32, device=self.runtime._device)
        self._sampled_token = wp.empty(1, dtype=wp.int32, device=self.runtime._device)
        self._sampled_token_host = wp.empty(1, dtype=wp.int32, device="cpu", pinned=self.runtime._device.is_cuda)
        self._sampled_token_host_view = self._sampled_token_host.numpy()
        self._greedy_argmax_kernels = _get_greedy_argmax_kernels(1024, 128)
        self._decode_graph = None
        self._decode_graph_outputs = None
        self._generation_graph = None
        self._generation_graph_eos = None
        self._past: dict[str, wp.array] = {}
        self.sequence_length = 0

    def reset(self) -> None:
        """Discard the current conversation's KV cache."""
        self._past.clear()
        self.sequence_length = 0

    def prefill(self, token_ids: Sequence[int]) -> wp.array:
        """Reset state, process a prompt, and return its logits."""
        self.reset()
        current_length = len(token_ids)
        if current_length == 0:
            raise ValueError("Qwen3OnnxRunner: token_ids must not be empty")
        if current_length >= self.cache_capacity:
            raise ValueError("Qwen3OnnxRunner: prompt must leave room for at least one decoded token")
        if self._chunk_runtime is not None and current_length >= self.prefill_chunk_size:
            return self._prefill_chunked(token_ids)
        shapes = {"input_ids": (1, current_length), "attention_mask": (1, current_length)}
        if "position_ids" in self.runtime.input_names:
            shapes["position_ids"] = (1, current_length)
        for name, base_shape in self._cache_shapes.items():
            shapes[name] = (1, base_shape[1], 0, base_shape[3]) if name in self._variable_cache_names else base_shape
        self.runtime.resize_inputs(shapes)
        inputs = {
            "input_ids": wp.array(
                np.asarray(token_ids, dtype=np.int64)[None, :], dtype=wp.int64, device=self.runtime._device
            ),
            "attention_mask": wp.ones((1, current_length), dtype=wp.int64, device=self.runtime._device),
        }
        if "position_ids" in self.runtime.input_names:
            inputs["position_ids"] = wp.array(
                np.arange(current_length, dtype=np.int64)[None, :], dtype=wp.int64, device=self.runtime._device
            )
        for name, shape in shapes.items():
            if name not in inputs:
                inputs[name] = wp.zeros(shape, dtype=self.runtime._input_dtypes[name], device=self.runtime._device)
        outputs = self.runtime(inputs)
        for name, destination in self._cache.items():
            source = outputs[self._present_for_past[name]]
            wp.copy(destination.flatten(), source.flatten(), count=source.size)
        self.sequence_length = current_length
        self._prepare_decode()
        return outputs["logits"]

    def _prefill_chunked(self, token_ids: Sequence[int]) -> wp.array:
        """Prefill through bounded fixed-size chunks and return the final logits."""
        for cache in self._cache.values():
            cache.zero_()
        wp.launch(
            _initialize_attention_mask,
            dim=self.cache_capacity,
            inputs=[self._decode_attention_mask, 0],
            device=self.runtime._device,
        )
        self.sequence_length = 0
        return self._append(token_ids)

    def append(self, token_ids: Sequence[int]) -> wp.array:
        """Process new prompt tokens while retaining the existing KV cache."""
        if self.sequence_length == 0:
            raise RuntimeError("Qwen3OnnxRunner.append requires a preceding prefill call")
        return self._append(token_ids)

    def _append(self, token_ids: Sequence[int]) -> wp.array:
        if not token_ids:
            raise ValueError("Qwen3OnnxRunner.append requires at least one token")
        if self.sequence_length + len(token_ids) > self.cache_capacity:
            raise ValueError("Qwen3OnnxRunner: appended tokens exceed the KV-cache capacity")

        chunk_size = self.prefill_chunk_size or len(token_ids) + 1
        full_length = len(token_ids) // chunk_size * chunk_size
        outputs = None
        consumed = 0
        for start in range(0, full_length, chunk_size):
            position = self.sequence_length
            end = position + chunk_size
            self._chunk_input_ids.assign(np.asarray(token_ids[start : start + chunk_size], dtype=np.int64)[None, :])
            if "position_ids" in self.runtime.input_names:
                self._chunk_position_ids.assign(np.arange(position, end, dtype=np.int64)[None, :])
            wp.launch(
                _initialize_attention_mask,
                dim=self.cache_capacity,
                inputs=[self._decode_attention_mask, end],
                device=self.runtime._device,
            )
            outputs = self._chunk_runtime(self._chunk_inputs)
            self.sequence_length = end
            consumed += chunk_size

        logits = outputs["logits"] if outputs is not None else None
        for token_id in token_ids[consumed:]:
            logits = self.decode(token_id)
        self._past = dict(self._cache)
        return logits

    def decode(self, token_id: int) -> wp.array:
        """Append one token and return its logits."""
        if self.sequence_length == 0:
            raise RuntimeError("Qwen3OnnxRunner.decode requires a preceding prefill call")
        if self.sequence_length >= self.cache_capacity:
            raise ValueError("Qwen3OnnxRunner: KV cache is full")
        self._stage_decode_token(token_id)
        if self.runtime._device.is_cuda:
            if self._decode_graph is None:
                wp.capture_begin(device=self.runtime._device)
                try:
                    self._decode_graph_outputs = self._decode_runtime(self._decode_inputs)
                    self._decode_graph = wp.capture_end(device=self.runtime._device)
                except Exception:
                    wp.capture_end(device=self.runtime._device)
                    raise
            wp.capture_launch(self._decode_graph)
            outputs = self._decode_graph_outputs
        else:
            outputs = self._decode_runtime(self._decode_inputs)
        self.sequence_length += 1
        return outputs["logits"]

    def _launch_greedy_partials(self, logits: wp.array) -> None:
        wp.launch_tiled(
            self._greedy_argmax_kernels[0],
            dim=128,
            inputs=[logits, self._sample_partial_values, self._sample_partial_tokens],
            block_dim=256,
            device=self.runtime._device,
        )

    def sample_greedy(self, logits: wp.array) -> int:
        """Select the largest logit while transferring only its token ID to the host."""
        if not self.runtime._device.is_cuda:
            return sample_token(logits, temperature=0.0)
        if logits.device != self.runtime._device or logits.dtype != wp.float16 or logits.ndim != 3:
            raise TypeError("Qwen3OnnxRunner.sample_greedy expects a 3-D FP16 array on the runner device")
        self._launch_greedy_partials(logits)
        wp.launch_tiled(
            self._greedy_argmax_kernels[1],
            dim=1,
            inputs=[self._sample_partial_values, self._sample_partial_tokens, self._sampled_token, logits.shape[2]],
            block_dim=128,
            device=self.runtime._device,
        )
        wp.copy(self._sampled_token_host, self._sampled_token, count=1)
        wp.synchronize_stream(self.runtime._device)
        return int(self._sampled_token_host_view[0])

    def generate_greedy(self, token_ids: Sequence[int], max_new_tokens: int, eos_token_id: int) -> list[int]:
        """Generate with an allocation-free captured CUDA graph and device-side argmax."""
        if max_new_tokens <= 0:
            return []
        if len(token_ids) + max_new_tokens > self.cache_capacity:
            raise ValueError("Qwen3OnnxRunner: requested generation exceeds KV-cache capacity")
        prompt_logits = self.prefill(token_ids)
        wp.launch(
            _initialize_generation_state,
            dim=1,
            inputs=[
                self._decode_position,
                self._generated_count,
                self._generation_finished,
                self.sequence_length,
            ],
            device=self.runtime._device,
        )
        sample_inputs = [
            self._sample_partial_values,
            self._sample_partial_tokens,
            prompt_logits.shape[2],
            self._decode_input_ids,
            self._decode_attention_mask,
            self._decode_position_ids,
            self._decode_position,
            self._generated_count,
            self._generated_ids,
            self._generation_finished,
            eos_token_id,
        ]
        self._launch_greedy_partials(prompt_logits)
        wp.launch_tiled(
            self._greedy_argmax_kernels[2], dim=1, inputs=sample_inputs, block_dim=128, device=self.runtime._device
        )
        if max_new_tokens > 1 and (self._generation_graph is None or self._generation_graph_eos != eos_token_id):
            wp.capture_begin(device=self.runtime._device)
            try:
                outputs = self._decode_runtime(self._decode_inputs)
                self._launch_greedy_partials(outputs["logits"])
                wp.launch_tiled(
                    self._greedy_argmax_kernels[2],
                    dim=1,
                    inputs=sample_inputs,
                    block_dim=128,
                    device=self.runtime._device,
                )
                self._generation_graph = wp.capture_end(device=self.runtime._device)
                self._generation_graph_eos = eos_token_id
            except Exception:
                wp.capture_end(device=self.runtime._device)
                raise
        for _ in range(max_new_tokens - 1):
            wp.capture_launch(self._generation_graph)
        generated = self._generated_ids.numpy()[:max_new_tokens].tolist()
        if eos_token_id in generated:
            generated = generated[: generated.index(eos_token_id) + 1]
        self.sequence_length += len(generated)
        return generated

    def _stage_decode_token(self, token_id: int) -> None:
        wp.launch(
            _set_decode_token,
            dim=1,
            inputs=[
                self._decode_input_ids,
                self._decode_attention_mask,
                self._decode_position_ids,
                token_id,
                self.sequence_length,
            ],
            device=self.runtime._device,
        )

    def _prepare_decode(self) -> None:
        wp.launch(
            _initialize_attention_mask,
            dim=self.cache_capacity,
            inputs=[self._decode_attention_mask, self.sequence_length],
            device=self.runtime._device,
        )
        self._past = dict(self._cache)
