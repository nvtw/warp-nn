# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-buffer composition for one trainable LoRA Linear operation."""

from __future__ import annotations

import warp as wp

from .linear import (
    _lora_split_count,
    lora_backward,
    lora_forward,
)


class LoRALinearTrainingPlan:
    """Own reusable activation scratch and FP32 gradients for one LoRA Linear.

    The output is a Warp-Tape boundary primal. The explicit Linear kernels are
    outside the tape; callers may consume the output in a differentiable
    primitive island, then pass output.grad to backward. Callers must call
    zero_grad (or otherwise clear output.grad) and reset or replace the tape
    before each microbatch.
    """

    def __init__(
        self,
        rows: int,
        in_features: int,
        out_features: int,
        rank: int,
        dtype: type,
        *,
        train_base: bool = False,
        device: str | wp.context.Device | None = None,
        cublas=None,
    ):
        if min(rows, in_features, out_features, rank) <= 0:
            raise ValueError("LoRA Linear dimensions must be positive")
        if dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("LoRA Linear training supports FP16 or BF16 storage")
        self.device = wp.get_device(device)
        self.cublas = cublas
        self.rows = rows
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.dtype = dtype
        self.output = wp.empty(
            (rows, out_features), dtype=dtype, device=self.device, requires_grad=True
        )
        self.hidden = wp.empty((rows, rank), dtype=wp.float32, device=self.device)
        self.grad_input = wp.empty((rows, in_features), dtype=dtype, device=self.device)
        self.grad_hidden = wp.empty((rows, rank), dtype=wp.float32, device=self.device)
        self.grad_a = wp.empty(
            (rank, in_features), dtype=wp.float32, device=self.device
        )
        self.grad_b = wp.empty(
            (out_features, rank), dtype=wp.float32, device=self.device
        )
        self.grad_weight = (
            wp.empty((out_features, in_features), dtype=wp.float32, device=self.device)
            if train_base
            else None
        )
        self.forward_matmul_splits = _lora_split_count(
            rows, rank, in_features, dtype, self.device
        )
        self.backward_matmul_splits = _lora_split_count(
            rows, rank, out_features, dtype, self.device
        )
        workspace_splits = max(self.forward_matmul_splits, self.backward_matmul_splits)
        self.matmul_workspace = (
            wp.empty(
                (workspace_splits * rows, rank),
                dtype=wp.float32,
                device=self.device,
            )
            if workspace_splits > 1
            else None
        )

    def forward(self, x, weight, lora_a, lora_b, *, scale: float) -> wp.array:
        lora_forward(
            x,
            weight,
            lora_a,
            lora_b,
            self.hidden,
            self.output,
            scale,
            cublas=self.cublas,
            matmul_workspace=self.matmul_workspace,
            matmul_splits=self.forward_matmul_splits,
        )
        return self.output

    def backward(
        self,
        x,
        weight,
        lora_a,
        lora_b,
        grad_output,
        *,
        scale: float,
        accumulate: bool = False,
    ) -> wp.array:
        lora_backward(
            x,
            weight,
            lora_a,
            lora_b,
            self.hidden,
            grad_output,
            self.grad_hidden,
            self.grad_input,
            self.grad_a,
            self.grad_b,
            scale,
            self.grad_weight,
            accumulate=accumulate,
            matmul_workspace=self.matmul_workspace,
            matmul_splits=self.backward_matmul_splits,
            cublas=self.cublas,
        )
        return self.grad_input

    def zero_grad(self) -> None:
        """Clear parameter and Tape-boundary gradients for a fresh microbatch."""
        self.output.grad.zero_()
        self.grad_a.zero_()
        self.grad_b.zero_()
        if self.grad_weight is not None:
            self.grad_weight.zero_()
