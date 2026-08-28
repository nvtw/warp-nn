# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared text-generation helpers for stateful language-model runners."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol

import numpy as np


class Tokenizer(Protocol):
    eos_token_id: int
    eos_token_ids: tuple[int, ...]
    tool_call_start: str | None

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] | None = None,
        enable_thinking: bool = False,
        reasoning_effort: str | None = None,
        preserve_thinking: bool = True,
    ) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str: ...

    def token_bytes(self, token_id: int, skip_special_tokens: bool = False) -> bytes: ...

    def parse_tool_calls(self, text: str) -> tuple[str, list[dict[str, object]]]: ...


class Runner(Protocol):
    cache_capacity: int

    def prefill(self, token_ids: Sequence[int]) -> Any: ...

    def decode(self, token_id: int) -> Any: ...

    def sample_greedy(self, logits: Any) -> int: ...


def is_eos_token(tokenizer: Tokenizer, token_id: int) -> bool:
    """Return whether ``token_id`` is any model-declared end token."""
    return token_id in getattr(tokenizer, "eos_token_ids", (tokenizer.eos_token_id,))


def split_tool_prefix(text: str, marker: str) -> tuple[str, str, bool]:
    """Split streamable text from a possible structured tool-call prefix."""
    start = text.find(marker)
    if start >= 0:
        return text[:start], text[start:], True
    keep = min(len(text), len(marker) - 1)
    while keep and not marker.startswith(text[-keep:]):
        keep -= 1
    return (text[:-keep], text[-keep:], False) if keep else (text, "", False)


def split_reasoning(text: str, enable_thinking: bool) -> tuple[str, str | None]:
    """Separate a Qwen thinking response into answer and reasoning text."""
    if not enable_thinking:
        return text, None
    reasoning, marker, answer = text.partition("</think>")
    return (answer.lstrip(), reasoning.strip()) if marker else ("", reasoning.strip())


def sample_token(
    logits: Any,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    previous_tokens: Sequence[int] = (),
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one token from the last logits row on the host."""
    values = logits.numpy() if hasattr(logits, "numpy") else np.asarray(logits)
    values = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])[-1].copy()
    if temperature <= 0.0:
        return int(np.argmax(values))
    if top_k < 0 or not 0.0 < top_p <= 1.0 or not -2.0 <= presence_penalty <= 2.0:
        raise ValueError("invalid top_k, top_p, or presence_penalty")
    if presence_penalty and previous_tokens:
        seen = np.asarray(tuple(previous_tokens), dtype=np.int64)
        seen = seen[(seen >= 0) & (seen < values.size)]
        values[np.unique(seen)] -= presence_penalty
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


def generate_tokens(
    runner: Runner,
    tokenizer: Tokenizer,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    seed: int | None = None,
) -> Iterator[int]:
    """Generate tokens incrementally through the common stateful runner API."""
    logits = runner.prefill(prompt_ids)
    generated = []
    rng = np.random.default_rng(seed)
    for _ in range(max_new_tokens):
        token_id = (
            runner.sample_greedy(logits)
            if temperature <= 0.0
            else sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                presence_penalty=presence_penalty,
                previous_tokens=generated,
                rng=rng,
            )
        )
        generated.append(token_id)
        eos = is_eos_token(tokenizer, token_id)
        next_logits = None if eos else runner.decode(token_id)
        yield token_id
        if eos:
            break
        logits = next_logits
