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

_PARAMETER_TOOL_PROMPT = """# Tools

You have access to the following functions:

<tools>{tools}
</tools>

Use a function only when the request requires it. Answer directly for conversation, translation, or creative writing.

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

_JSON_TOOL_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

Use a function only when the request requires it. Answer directly for conversation, translation, or creative writing.

You are provided with function signatures within <tools></tools> XML tags:
<tools>{tools}
</tools>

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

_REASONING_INSTRUCTIONS = {
    "xhigh": (
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
        "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
    ),
    "medium": "",
    "low": (
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
        "without unnecessary elaboration."
    ),
}


def _is_letter(character: str) -> bool:
    return unicodedata.category(character)[0] in "LM"


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _pretokenize_gpt(text: str, number_width: int):
    """Implement the shared GPT/Qwen/Llama Unicode split expression."""
    index = 0
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    while index < len(text):
        contraction = next(
            (
                item
                for item in contractions
                if text[index : index + len(item)].lower() == item
            ),
            None,
        )
        if contraction is not None:
            yield text[index : index + len(contraction)]
            index += len(contraction)
            continue

        letter_start = index
        character = text[index]
        if (
            character not in "\r\n"
            and not _is_letter(character)
            and not _is_number(character)
        ):
            letter_start += 1
        if letter_start < len(text) and _is_letter(text[letter_start]):
            end = letter_start + 1
            while end < len(text) and _is_letter(text[end]):
                end += 1
            yield text[index:end]
            index = end
            continue
        if _is_number(character):
            end = index + 1
            while (
                end < len(text) and end - index < number_width and _is_number(text[end])
            ):
                end += 1
            yield text[index:end]
            index = end
            continue

        symbol_start = index + 1 if character == " " else index
        if symbol_start < len(text):
            symbol = text[symbol_start]
            if (
                not symbol.isspace()
                and not _is_letter(symbol)
                and not _is_number(symbol)
            ):
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
            last_newline = max(
                text.rfind("\r", index, end), text.rfind("\n", index, end)
            )
            if last_newline >= index:
                end = last_newline + 1
            elif end < len(text) and end - index > 1:
                end -= 1
            yield text[index:end]
            index = end
            continue

        yield character
        index += 1


def _pretokenize(text: str):
    """Implement Qwen's one-digit Unicode split expression."""
    yield from _pretokenize_gpt(text, 1)


def _pretokenize_llama3(text: str):
    """Implement Llama 3's otherwise-shared split with 1-3 digit groups."""
    yield from _pretokenize_gpt(text, 3)


def _pretokenize_o200k(text: str):
    """Implement the dependency-free o200k Unicode split used by Muse."""
    index = 0
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    while index < len(text):
        start = index
        if (
            text[index] not in "\r\n"
            and not _is_letter(text[index])
            and not _is_number(text[index])
        ):
            index += 1
        if index < len(text) and _is_letter(text[index]):
            index += 1
            while index < len(text) and _is_letter(text[index]):
                if unicodedata.category(
                    text[index - 1]
                ) == "Ll" and unicodedata.category(text[index]) in ("Lu", "Lt"):
                    break
                index += 1
            contraction = next(
                (
                    item
                    for item in contractions
                    if text[index : index + len(item)].lower() == item
                ),
                None,
            )
            if contraction:
                index += len(contraction)
            yield text[start:index]
            continue
        index = start
        if _is_number(text[index]):
            end = index + 1
            while end < len(text) and end - index < 3 and _is_number(text[end]):
                end += 1
            yield text[index:end]
            index = end
            continue

        symbol_start = index + 1 if text[index] == " " else index
        if symbol_start < len(text):
            symbol = text[symbol_start]
            if (
                not symbol.isspace()
                and not _is_letter(symbol)
                and not _is_number(symbol)
            ):
                end = symbol_start + 1
                while end < len(text):
                    symbol = text[end]
                    if symbol.isspace() or _is_letter(symbol) or _is_number(symbol):
                        break
                    end += 1
                while end < len(text) and text[end] in "\r\n/":
                    end += 1
                yield text[index:end]
                index = end
                continue

        if text[index].isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            last_newline = max(
                text.rfind("\r", index, end), text.rfind("\n", index, end)
            )
            if last_newline >= index:
                end = last_newline + 1
            elif end < len(text) and end - index > 1:
                end -= 1
            yield text[index:end]
            index = end
            continue

        yield text[index]
        index += 1


class Qwen3Tokenizer:
    """Dependency-free ByteLevel-BPE tokenizer for Qwen and Nemotron chat models."""

    tool_call_start = "<tool_call>"

    def __init__(self, path: str | Path | Mapping, *, pretokenizer=None):
        if isinstance(path, Mapping):
            data = path
            directory = None
        else:
            path = Path(path)
            if path.is_dir():
                directory = path
                path /= "tokenizer.json"
            else:
                directory = path.parent
            data = json.loads(path.read_text(encoding="utf-8"))
        model = data["model"]
        normalizer = data.get("normalizer")
        if model["type"] != "BPE" or normalizer not in (None, {"type": "NFC"}):
            raise ValueError(
                "Qwen3Tokenizer: expected an NFC or unnormalized ByteLevel-BPE tokenizer"
            )
        self._normalize_nfc = normalizer is not None
        self._vocabulary = model["vocab"]
        self._tokens = {token_id: token for token, token_id in self._vocabulary.items()}
        self._merge_ranks = {
            tuple(pair): rank for rank, pair in enumerate(model["merges"])
        }
        self._ignore_merges = bool(model.get("ignore_merges", False))
        pretokenizer = pretokenizer or data.get("_pretokenizer")
        self._pretokenize = {
            None: _pretokenize,
            "qwen": _pretokenize,
            "llama3": _pretokenize_llama3,
            "o200k": _pretokenize_o200k,
        }.get(pretokenizer)
        if self._pretokenize is None:
            raise ValueError(f"unknown ByteLevel pretokenizer '{pretokenizer}'")
        self._added_tokens = {
            item["content"]: item["id"] for item in data["added_tokens"]
        }
        self._added_by_id = {item["id"]: item for item in data["added_tokens"]}
        alternatives = "|".join(
            re.escape(token)
            for token in sorted(self._added_tokens, key=len, reverse=True)
        )
        self._added_pattern = re.compile(f"({alternatives})")
        tokenizer_config_path = (
            directory / "tokenizer_config.json" if directory is not None else None
        )
        tokenizer_config = data.get("tokenizer_config", {})
        if tokenizer_config_path is not None and tokenizer_config_path.is_file():
            tokenizer_config = json.loads(
                tokenizer_config_path.read_text(encoding="utf-8")
            )
        generation_path = (
            directory / "generation_config.json" if directory is not None else None
        )
        self.generation_config = data.get("generation_config", {})
        if generation_path is not None and generation_path.is_file():
            self.generation_config = json.loads(
                generation_path.read_text(encoding="utf-8")
            )
        eos_token = tokenizer_config.get("eos_token")
        if isinstance(eos_token, dict):
            eos_token = eos_token.get("content")
        eos_token_id = self.generation_config.get(
            "eos_token_id",
            self._added_tokens.get(eos_token, self._added_tokens.get("<|im_end|>", 0)),
        )
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        self.eos_token_id = int(eos_token_id)
        pad_token = tokenizer_config.get("pad_token")
        self.pad_token_id = int(
            self.generation_config.get(
                "pad_token_id",
                self._added_tokens.get(
                    pad_token, self._added_tokens.get("<|endoftext|>", 0)
                ),
            )
        )
        end_tokens = self.generation_config.get("eos_token_id", self.eos_token_id)
        end_tokens = end_tokens if isinstance(end_tokens, list) else [end_tokens]
        self.eos_token_ids = tuple(
            dict.fromkeys(int(token_id) for token_id in end_tokens)
        )
        if not self.eos_token_ids:
            raise ValueError("Qwen3Tokenizer: generation_config has no EOS token")
        template_path = (
            directory / "chat_template.jinja" if directory is not None else None
        )
        template = data.get("chat_template", tokenizer_config.get("chat_template", ""))
        if template_path is not None and template_path.is_file():
            template = template_path.read_text(encoding="utf-8")
        self._tool_dialect = "json" if "args-json-object" in template else "parameters"
        self.supports_reasoning_effort = "reasoning_effort" in template
        self.default_enable_thinking = (
            self.supports_reasoning_effort or "set enable_thinking" in template
        )
        self._compact_empty_thinking = "<think></think>" in template
        self._always_system_prompt = 'set system_message = ""' in template

    @lru_cache(maxsize=8192)
    def _bpe(self, piece: str) -> tuple[str, ...]:
        if self._ignore_merges and piece in self._vocabulary:
            return (piece,)
        symbols = list(piece)
        while len(symbols) > 1:
            candidates = {
                (symbols[index], symbols[index + 1])
                for index in range(len(symbols) - 1)
            }
            pair = min(
                candidates,
                key=lambda item: self._merge_ranks.get(item, len(self._merge_ranks)),
            )
            if pair not in self._merge_ranks:
                break
            merged = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and (symbols[index], symbols[index + 1]) == pair
                ):
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
            if self._normalize_nfc:
                chunk = unicodedata.normalize("NFC", chunk)
            for piece in self._pretokenize(chunk):
                byte_piece = "".join(
                    _BYTE_ENCODER[value] for value in piece.encode("utf-8")
                )
                token_ids.extend(
                    self._vocabulary[token] for token in self._bpe(byte_piece)
                )
        return token_ids

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = False
    ) -> str:
        """Decode token IDs to UTF-8 text."""
        output = []
        byte_characters = []

        def flush_bytes():
            if byte_characters:
                output.append(
                    bytes(
                        _BYTE_DECODER[character] for character in byte_characters
                    ).decode("utf-8", "replace")
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
        return bytes(
            _BYTE_DECODER[character] for character in self._tokens[int(token_id)]
        )

    def format_chat(
        self,
        messages: Sequence[Mapping[str, object]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        tools: Sequence[Mapping[str, object]] | None = None,
        reasoning_effort: str | None = None,
        preserve_thinking: bool = True,
    ) -> str:
        """Format text messages and OpenAI function tools using Qwen's template."""
        formatted = []
        first = 0
        reasoning_instruction = ""
        if reasoning_effort is not None and not self.supports_reasoning_effort:
            raise ValueError(
                "This Qwen chat template does not support reasoning_effort"
            )
        if self.supports_reasoning_effort and enable_thinking:
            reasoning_effort = reasoning_effort or "xhigh"
            try:
                reasoning_instruction = _REASONING_INSTRUCTIONS[reasoning_effort]
            except KeyError as error:
                raise ValueError(
                    "reasoning_effort must be 'low', 'medium', or 'xhigh'"
                ) from error
        elif reasoning_effort is not None:
            raise ValueError("reasoning_effort requires thinking mode")
        if tools:
            definitions = "".join(
                f"\n{json.dumps(tool, ensure_ascii=False, separators=(',', ':'))}"
                for tool in tools
            )
            prompt = (
                _JSON_TOOL_PROMPT
                if self._tool_dialect == "json"
                else _PARAMETER_TOOL_PROMPT
            ).format(tools=definitions)
            system = prompt
            if reasoning_instruction:
                system = reasoning_instruction + "\n\n" + system
            if messages and messages[0].get("role") in ("system", "developer"):
                content = messages[0].get("content")
                if not isinstance(content, str):
                    raise ValueError(
                        "Qwen3Tokenizer.format_chat requires text message content"
                    )
                if content.strip():
                    system = (
                        content.strip() + "\n\n" + prompt
                        if self._tool_dialect == "json"
                        else prompt + "\n\n" + content.strip()
                    )
                first = 1
            formatted.append(f"<|im_start|>system\n{system}<|im_end|>\n")
        elif reasoning_instruction:
            system = reasoning_instruction
            if messages and messages[0].get("role") in ("system", "developer"):
                content = messages[0].get("content")
                if not isinstance(content, str):
                    raise ValueError(
                        "Qwen3Tokenizer.format_chat requires text message content"
                    )
                if content.strip():
                    system += "\n\n" + content.strip()
                first = 1
            formatted.append(f"<|im_start|>system\n{system}<|im_end|>\n")
        elif self._always_system_prompt and (
            not messages or messages[0].get("role") not in ("system", "developer")
        ):
            formatted.append("<|im_start|>system\n<|im_end|>\n")

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
                        raise ValueError(
                            "Qwen3Tokenizer.format_chat requires text tool results"
                        )
                    responses.append(
                        f"\n<tool_response>\n{tool_content.strip()}\n</tool_response>"
                    )
                    index += 1
                formatted.append(f"<|im_start|>user{''.join(responses)}<|im_end|>\n")
                continue
            if (
                role not in ("system", "user", "assistant")
                or content is not None
                and not isinstance(content, str)
            ):
                raise ValueError(
                    "Qwen3Tokenizer.format_chat supports text OpenAI chat messages"
                )
            body = "" if content is None else content.strip()
            if role == "assistant":
                reasoning = message.get("reasoning_content", message.get("reasoning"))
                if (
                    preserve_thinking
                    and isinstance(reasoning, str)
                    and reasoning.strip()
                    and not body.startswith("<think>")
                ):
                    body = f"<think>\n{reasoning.strip()}\n</think>\n\n{body}"
                elif not preserve_thinking and body.startswith("<think>"):
                    body = re.sub(
                        r"^<think>.*?</think>\s*", "", body, count=1, flags=re.DOTALL
                    )
                for tool_call in message.get("tool_calls") or ():
                    function = tool_call.get("function", tool_call)
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, Mapping):
                        raise ValueError(
                            "Qwen3Tokenizer.format_chat requires object-valued tool arguments"
                        )
                    separator = "" if not body or body.endswith("\n") else "\n\n"
                    if self._tool_dialect == "json":
                        call = {"name": function["name"], "arguments": arguments}
                        body += f"{separator}<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>"
                    else:
                        parameters = "".join(
                            f"<parameter={name}>\n{_format_tool_value(value)}\n</parameter>\n"
                            for name, value in arguments.items()
                        )
                        body += (
                            f"{separator}<tool_call>\n<function={function['name']}>\n"
                            f"{parameters}</function>\n</tool_call>"
                        )
            formatted.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
            index += 1
        if add_generation_prompt:
            formatted.append("<|im_start|>assistant\n")
            formatted.append(self.generation_prefix(enable_thinking))
        return "".join(formatted)

    def generation_prefix(self, enable_thinking: bool) -> str:
        """Return tokens inserted before generated assistant content."""
        if not enable_thinking:
            return (
                "<think></think>"
                if self._compact_empty_thinking
                else "<think>\n\n</think>\n\n"
            )
        return "<think>\n" if self.default_enable_thinking else ""

    def encode_chat(
        self, messages: Sequence[Mapping[str, object]], **kwargs
    ) -> list[int]:
        """Format and encode a chat prompt."""
        return self.encode(self.format_chat(messages, **kwargs))

    def parse_tool_calls(self, text: str) -> tuple[str, list[dict[str, object]]]:
        """Extract structured function calls from Qwen assistant text."""
        return parse_qwen_tool_calls(text)


def _format_tool_value(value: object) -> str:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def parse_qwen_tool_calls(text: str) -> tuple[str, list[dict[str, object]]]:
    """Extract Qwen XML function calls and return remaining assistant text."""
    calls = []
    spans = []
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    function_pattern = re.compile(r"<function=([^>\n]+)>\s*(.*?)</function>", re.DOTALL)
    parameter_pattern = re.compile(
        r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL
    )
    for match in pattern.finditer(text):
        body = match.group(1).strip()
        if body.startswith("{"):
            try:
                call = json.loads(body)
            except json.JSONDecodeError:
                continue
            if not isinstance(call, Mapping) or not isinstance(call.get("name"), str):
                continue
            arguments = call.get("arguments", {})
            if not isinstance(arguments, Mapping):
                continue
            calls.append({"name": call["name"], "arguments": dict(arguments)})
            spans.append(match.span())
            continue
        function = function_pattern.fullmatch(body)
        if function is None:
            continue
        arguments = {}
        for parameter in parameter_pattern.finditer(function.group(2)):
            value = parameter.group(2).strip()
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
            arguments[parameter.group(1).strip()] = value
        calls.append({"name": function.group(1).strip(), "arguments": arguments})
        spans.append(match.span())
    for start, end in reversed(spans):
        text = text[:start] + text[end:]
    return text.strip(), calls
