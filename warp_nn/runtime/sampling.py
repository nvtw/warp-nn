# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared sampling policy; runners provide logits and optional bounded top-k reads."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def validate_sampling(temperature, top_k, top_p, presence_penalty):
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if (
        not isinstance(top_k, (int, np.integer))
        or top_k < 0
        or not 0.0 < top_p <= 1.0
        or not -2.0 <= presence_penalty <= 2.0
    ):
        raise ValueError("invalid top_k, top_p, or presence_penalty")


def sample_candidates(
    values: np.ndarray,
    candidates: np.ndarray,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    """Apply host probability policy to an already selected candidate set."""
    values = np.asarray(values, dtype=np.float64) / temperature
    candidates = np.asarray(candidates, dtype=np.int64)
    probabilities = np.exp(values - np.max(values))
    probabilities /= probabilities.sum()
    if top_p < 1.0:
        order = np.argsort(probabilities)[::-1]
        keep = np.cumsum(probabilities[order]) - probabilities[order] < top_p
        candidates = candidates[order[keep]]
        probabilities = probabilities[order[keep]]
        probabilities /= probabilities.sum()
    return int(rng.choice(candidates, p=probabilities))


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
    values = (
        np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])[-1].copy()
    )
    validate_sampling(temperature, top_k, top_p, presence_penalty)
    if presence_penalty and len(previous_tokens):
        seen = np.asarray(tuple(previous_tokens), dtype=np.int64)
        seen = seen[(seen >= 0) & (seen < values.size)]
        values[np.unique(seen)] -= presence_penalty
    if temperature == 0.0:
        return int(np.argmax(values))
    candidates = np.arange(values.size)
    if 0 < top_k < values.size:
        candidates = np.argpartition(values, -top_k)[-top_k:]
        values = values[candidates]
    return sample_candidates(
        values, candidates, temperature, top_p, rng or np.random.default_rng()
    )


def sample_runner_token(
    runner: Any,
    logits: Any,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    previous_tokens: Sequence[int] = (),
    rng: np.random.Generator | None = None,
) -> int:
    """Sample through a runner's bounded device path when one is available."""
    validate_sampling(temperature, top_k, top_p, presence_penalty)
    if (temperature == 0.0 or top_k == 1) and presence_penalty == 0.0:
        return runner.sample_greedy(logits)
    read_top_k = getattr(runner, "read_top_k", None)
    if callable(read_top_k) and presence_penalty == 0.0 and 1 < top_k <= 64:
        values, candidates = read_top_k(logits, top_k)
        return sample_candidates(
            values,
            candidates,
            temperature,
            top_p,
            rng or np.random.default_rng(),
        )
    return sample_token(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        presence_penalty=presence_penalty,
        previous_tokens=previous_tokens,
        rng=rng,
    )
