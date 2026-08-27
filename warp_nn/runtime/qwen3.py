# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal stateful runner for Qwen-style decoder ONNX graphs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from warp_nn.runtime.onnx_runtime import OnnxRuntime


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
