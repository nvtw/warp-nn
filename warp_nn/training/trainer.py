# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small fixed-buffer SFT trainer with optional whole-step CUDA capture."""

from pathlib import Path
from typing import Mapping

import warp as wp

from .checkpoint import restore_lora_collection, save_lora_collection
from .data import SFTBatch
from .model import CausalLMTrainingPlan


class LoRATrainer:
    """Own stable batch buffers and launch one reusable training step graph."""

    def __init__(
        self,
        model: CausalLMTrainingPlan,
        cosine: wp.array,
        sine: wp.array,
    ):
        first_attention = model.stack.blocks[0].attention
        batch = int(first_attention.batch)
        sequence = int(first_attention.sequence)
        if model.rows != batch * sequence:
            raise ValueError("model rows do not match attention batch geometry")
        if (
            not isinstance(cosine, wp.array)
            or not isinstance(sine, wp.array)
            or cosine.shape != sine.shape
            or cosine.ndim != 2
            or cosine.dtype != model.dtype
            or sine.dtype != model.dtype
            or cosine.device != model.device
            or sine.device != model.device
        ):
            raise ValueError("rotary caches must be matching model-dtype matrices")
        self.model = model
        self.device = model.device
        self.batch = batch
        self.sequence = sequence
        self.cosine = cosine
        self.sine = sine
        self.input_ids = wp.empty(model.rows, dtype=wp.int32, device=self.device)
        self.targets = wp.empty(model.rows, dtype=wp.int32, device=self.device)
        self.lengths = wp.empty(batch, dtype=wp.int32, device=self.device)
        self.positions = wp.empty((batch, sequence), dtype=wp.int64, device=self.device)
        self.graph = None
        self._loaded = False

    @property
    def _inputs(self):
        return (
            self.input_ids,
            self.targets,
            self.lengths,
            self.positions,
            self.cosine,
            self.sine,
        )

    def load_batch(self, batch: SFTBatch) -> None:
        """Copy a host batch into stable graph-bound buffers."""
        if batch.batch != self.batch or batch.sequence != self.sequence:
            raise ValueError(f"batch must have geometry {(self.batch, self.sequence)}")
        self.input_ids.assign(batch.input_ids.reshape(-1))
        self.targets.assign(batch.targets.reshape(-1))
        self.lengths.assign(batch.lengths)
        self.positions.assign(batch.positions)
        self._loaded = True

    def evaluate_loss(self) -> float:
        """Run forward only and return the synchronized mean loss."""
        if not self._loaded:
            raise RuntimeError("load a batch before evaluation")
        return float(self.model.forward(*self._inputs).numpy()[0])

    def capture(self) -> None:
        """Compile and capture the complete zero-grad through AdamW step."""
        if not self.device.is_cuda:
            raise RuntimeError("CUDA graph capture requires a CUDA model")
        if not self._loaded:
            raise RuntimeError("load a batch before capture")
        self.model.adapters.zero_grad()
        self.model.forward(*self._inputs)
        self.model.backward(*self._inputs)
        self.model.adapters.zero_grad()
        wp.synchronize_device(self.device)
        wp.capture_begin(device=self.device)
        try:
            self.model.train_step(*self._inputs)
            self.graph = wp.capture_end(device=self.device)
        except Exception:
            wp.capture_end(device=self.device)
            raise

    def step(self) -> wp.array:
        """Launch one direct or captured update and return the fixed loss scalar."""
        if not self._loaded:
            raise RuntimeError("load a batch before training")
        if self.graph is None:
            return self.model.train_step(*self._inputs)
        wp.capture_launch(self.graph)
        return self.model.output.loss

    def save_adapters(
        self,
        path: str | Path,
        *,
        base_identifier: str | None = None,
        metadata: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> None:
        """Atomically save current FP32 masters in standard safetensors."""
        save_lora_collection(
            path,
            self.model.adapters,
            base_identifier=base_identifier,
            caller_metadata=metadata,
        )

    def load_adapters(self, path: str | Path):
        """Restore adapters in place; AdamW moments and step restart from zero."""
        return restore_lora_collection(path, self.model.adapters)
