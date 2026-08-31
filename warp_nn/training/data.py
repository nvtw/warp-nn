# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free causal SFT batch preparation."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import warp as wp


@dataclass(frozen=True)
class SFTExample:
    """Already-tokenized prompt and desired response."""

    prompt: Sequence[int]
    response: Sequence[int]


@dataclass(frozen=True)
class SFTBatch:
    """Fixed rectangular causal batch ready for a training plan."""

    input_ids: np.ndarray
    targets: np.ndarray
    lengths: np.ndarray
    positions: np.ndarray

    @property
    def batch(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def sequence(self) -> int:
        return int(self.positions.shape[1])

    def upload(self, device=None) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        """Upload flattened IDs/targets and batched metadata to one device."""
        return (
            wp.array(self.input_ids.reshape(-1), dtype=wp.int32, device=device),
            wp.array(self.targets.reshape(-1), dtype=wp.int32, device=device),
            wp.array(self.lengths, dtype=wp.int32, device=device),
            wp.array(self.positions, dtype=wp.int64, device=device),
        )


def prepare_sft_batch(
    examples: Sequence[SFTExample],
    sequence: int,
    *,
    pad_token_id: int,
    eos_token_id: int | None = None,
    ignore_index: int = -100,
    train_on_prompt: bool = False,
    truncation: str = "error",
) -> SFTBatch:
    """Create shifted causal inputs and labels without cross-example attention.

    Each example occupies one batch row. Targets predicting prompt tokens and
    all padding are ignored unless ``train_on_prompt`` is enabled. Overlength
    data raises by default; ``truncation="right"`` is an explicit quality
    tradeoff. Multi-example sequence packing is intentionally a separate API
    because it needs attention/recurrent-state isolation.
    """
    examples = tuple(examples)
    if not examples:
        raise ValueError("an SFT batch requires at least one example")
    if sequence <= 0:
        raise ValueError("sequence must be positive")
    if truncation not in ("error", "right"):
        raise ValueError("truncation must be 'error' or 'right'")
    batch = len(examples)
    inputs = np.full((batch, sequence), pad_token_id, dtype=np.int32)
    targets = np.full((batch, sequence), ignore_index, dtype=np.int32)
    lengths = np.zeros(batch, dtype=np.int32)
    positions = np.broadcast_to(
        np.arange(sequence, dtype=np.int64), (batch, sequence)
    ).copy()
    maximum_tokens = sequence + 1
    for row, example in enumerate(examples):
        prompt = [int(token) for token in example.prompt]
        response = [int(token) for token in example.response]
        if not response:
            raise ValueError(f"SFT example {row} has an empty response")
        tokens = prompt + response
        if eos_token_id is not None and tokens[-1] != eos_token_id:
            tokens.append(int(eos_token_id))
        if len(tokens) > maximum_tokens:
            if truncation == "error":
                raise ValueError(
                    f"SFT example {row} needs {len(tokens) - 1} positions, "
                    f"but sequence is {sequence}"
                )
            tokens = tokens[:maximum_tokens]
        length = len(tokens) - 1
        if length <= 0 or len(prompt) >= len(tokens):
            raise ValueError(
                f"SFT example {row} has no response target after truncation"
            )
        inputs[row, :length] = tokens[:-1]
        targets[row, :length] = tokens[1:]
        if not train_on_prompt:
            prompt_targets = min(max(len(prompt) - 1, 0), length)
            targets[row, :prompt_targets] = ignore_index
        lengths[row] = length
    return SFTBatch(inputs, targets, lengths, positions)
