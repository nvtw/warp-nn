# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small fixed-buffer SFT trainer with optional whole-step CUDA capture."""

from pathlib import Path
from typing import Mapping

import warp as wp

from .checkpoint import (
    restore_lora_collection,
    restore_lora_training_state,
    save_lora_collection,
    save_lora_training_state,
)
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
        self.segment_bounds = wp.empty(
            (batch, sequence, 2), dtype=wp.int32, device=self.device
        )
        self._segment_bounds_input = None
        self.graph = None
        self.accumulation_graph = None
        self.update_graph = None
        self._loaded = False
        self._accumulating = False

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
        segment_bounds = batch.segment_bounds
        was_segmented = self._segment_bounds_input is not None
        if segment_bounds is not None and segment_bounds.shape != (
            self.batch,
            self.sequence,
            2,
        ):
            raise ValueError(
                f"batch segment bounds must have shape {(self.batch, self.sequence, 2)}"
            )
        if segment_bounds is not None:
            self.segment_bounds.assign(segment_bounds)
            self._segment_bounds_input = self.segment_bounds
        else:
            self._segment_bounds_input = None
        if was_segmented != (self._segment_bounds_input is not None):
            self.graph = None
            self.accumulation_graph = None
        self._loaded = True

    def evaluate_loss(self) -> float:
        """Run forward only and return the synchronized mean loss."""
        if not self._loaded:
            raise RuntimeError("load a batch before evaluation")
        return float(
            self.model.forward(
                *self._inputs, segment_bounds=self._segment_bounds_input
            ).numpy()[0]
        )

    def _capture_graph(self, launch):
        wp.capture_begin(device=self.device)
        try:
            launch()
            return wp.capture_end(device=self.device)
        except Exception:
            wp.capture_end(device=self.device)
            raise

    def capture(self) -> None:
        """Compile and capture the complete zero-grad through AdamW step."""
        if self._accumulating:
            raise RuntimeError("finish gradient accumulation before capture")
        if not self.device.is_cuda:
            raise RuntimeError("CUDA graph capture requires a CUDA model")
        if not self._loaded:
            raise RuntimeError("load a batch before capture")
        self.model.adapters.zero_grad()
        self.model.forward(*self._inputs, segment_bounds=self._segment_bounds_input)
        self.model.backward(*self._inputs, segment_bounds=self._segment_bounds_input)
        self.model.adapters.zero_grad()
        wp.synchronize_device(self.device)
        self.graph = self._capture_graph(
            lambda: self.model.train_step(
                *self._inputs, segment_bounds=self._segment_bounds_input
            )
        )

    def _accumulate(self) -> wp.array:
        loss = self.model.forward(
            *self._inputs,
            segment_bounds=self._segment_bounds_input,
            reduction="sum",
        )
        self.model.backward(
            *self._inputs,
            segment_bounds=self._segment_bounds_input,
            reduction="sum",
            accumulate=True,
        )
        self.model.adapters.optimizer.accumulate_valid_tokens(
            self.model.output.valid_count
        )
        return loss

    def capture_accumulation(self) -> None:
        """Capture one summed-gradient microbatch and its separate AdamW update."""
        if self._accumulating:
            raise RuntimeError("finish gradient accumulation before capture")
        if not self.device.is_cuda:
            raise RuntimeError("CUDA graph capture requires a CUDA model")
        if not self._loaded:
            raise RuntimeError("load a batch before capture")
        optimizer = self.model.adapters.optimizer
        if not optimizer.normalize_by_valid_tokens:
            raise RuntimeError(
                "gradient accumulation requires normalize_by_valid_tokens=True"
            )
        self.model.adapters.zero_grad()
        self._accumulate()
        self.model.adapters.zero_grad()
        wp.synchronize_device(self.device)
        self.accumulation_graph = self._capture_graph(self._accumulate)
        self.update_graph = self._capture_graph(self.model.adapters.step)
        self.model.adapters.zero_grad()

    def begin_accumulation(self) -> None:
        """Begin an exact valid-token-normalized microbatch accumulation window."""
        if self._accumulating:
            raise RuntimeError("gradient accumulation is already active")
        if not self.model.adapters.optimizer.normalize_by_valid_tokens:
            raise RuntimeError(
                "gradient accumulation requires normalize_by_valid_tokens=True"
            )
        self.model.adapters.zero_grad()
        self._accumulating = True

    def accumulate(self) -> wp.array:
        """Add the loaded microbatch's summed gradients to the active window."""
        if not self._loaded:
            raise RuntimeError("load a batch before training")
        if not self._accumulating:
            raise RuntimeError("begin gradient accumulation first")
        if self.accumulation_graph is None:
            return self._accumulate()
        wp.capture_launch(self.accumulation_graph)
        return self.model.output.loss

    def finish_accumulation(self) -> wp.array:
        """Apply one AdamW update normalized by all accumulated valid tokens."""
        if not self._accumulating:
            raise RuntimeError("begin gradient accumulation first")
        if self.update_graph is None:
            self.model.adapters.step()
        else:
            wp.capture_launch(self.update_graph)
        self._accumulating = False
        return self.model.output.loss

    def step(self) -> wp.array:
        """Launch one direct or captured update and return the fixed loss scalar."""
        if self._accumulating:
            raise RuntimeError("use accumulate while gradient accumulation is active")
        if not self._loaded:
            raise RuntimeError("load a batch before training")
        if self.graph is None:
            return self.model.train_step(
                *self._inputs, segment_bounds=self._segment_bounds_input
            )
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

    def save_training_state(
        self, path: str | Path, *, base_identifier: str | None = None
    ) -> None:
        """Save exact-resume LoRA and AdamW state separately from adapter export."""
        if self._accumulating:
            raise RuntimeError("finish gradient accumulation before checkpointing")
        save_lora_training_state(
            path, self.model.adapters, base_identifier=base_identifier
        )

    def load_training_state(
        self, path: str | Path, *, base_identifier: str | None = None
    ):
        """Restore an exact trajectory in place without invalidating CUDA graphs."""
        if self._accumulating:
            raise RuntimeError("finish gradient accumulation before checkpointing")
        return restore_lora_training_state(
            path, self.model.adapters, base_identifier=base_identifier
        )
