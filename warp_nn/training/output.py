# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-safe causal language-model output composition."""

import math

import warp as wp

from .bridges import cast_to_float32
from .linear import linear_backward, linear_forward
from .loss import LowPrecisionCrossEntropyPlan
from .qk import QKTransformPlan


class CausalLMOutputPlan:
    """Frozen RMSNorm and vocabulary head with memory-bounded cross-entropy.

    This composes existing reusable kernels rather than carrying a second head
    implementation. The vocabulary gradient overwrites the logits buffer. LM
    head and normalization weights are frozen, which is the standard LoRA SFT
    path; backward returns an FP32 gradient for the transformer stack.
    """

    def __init__(
        self,
        rows: int,
        norm_weight: wp.array,
        lm_head: wp.array,
        *,
        epsilon: float = 1.0e-6,
        ignore_index: int = -100,
        cublas=None,
    ):
        if rows <= 0:
            raise ValueError("rows must be positive")
        if not isinstance(norm_weight, wp.array) or norm_weight.ndim != 1:
            raise TypeError("norm_weight must be a 1-D Warp array")
        if not isinstance(lm_head, wp.array) or lm_head.ndim != 2:
            raise TypeError("lm_head must be a 2-D Warp array")
        if norm_weight.dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("output weights must use FP16 or BF16")
        if lm_head.dtype != norm_weight.dtype:
            raise TypeError("norm_weight and lm_head dtypes must match")
        if lm_head.shape[1] != norm_weight.shape[0]:
            raise ValueError("LM head input width must match normalization width")
        if lm_head.device != norm_weight.device:
            raise ValueError("output weights must be on the same device")
        if not norm_weight.is_contiguous or not lm_head.is_contiguous:
            raise ValueError("output weights must be contiguous")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        self.device = norm_weight.device
        self.rows = rows
        self.hidden = norm_weight.shape[0]
        self.classes = lm_head.shape[0]
        self.dtype = norm_weight.dtype
        self.norm_weight = norm_weight
        self.lm_head = lm_head
        self.cublas = cublas
        self.norm = QKTransformPlan(
            1,
            1,
            rows,
            self.hidden,
            self.dtype,
            rotary_dim=0,
            epsilon=epsilon,
            device=self.device,
        )
        self.normalized = self.norm.output.reshape((rows, self.hidden))
        self.logits = wp.empty(
            (rows, self.classes), dtype=self.dtype, device=self.device
        )
        self.loss_plan = LowPrecisionCrossEntropyPlan(
            rows,
            self.classes,
            dtype=self.dtype,
            ignore_index=ignore_index,
            in_place=True,
            device=self.device,
        )
        self.normalized_grad = wp.empty(
            (rows, self.hidden), dtype=self.dtype, device=self.device
        )
        self.normalized_grad_fp32 = wp.empty(
            (rows, self.hidden), dtype=wp.float32, device=self.device
        )
        self.input_grad = self.norm.input_grad.reshape((rows, self.hidden))

    def _inputs(self, x: wp.array, targets: wp.array) -> None:
        if not isinstance(x, wp.array) or x.shape != (self.rows, self.hidden):
            raise ValueError(f"x must have shape {(self.rows, self.hidden)}")
        if x.dtype != self.dtype or x.device != self.device or not x.is_contiguous:
            raise ValueError("x must match output-plan dtype and device")
        if not isinstance(targets, wp.array) or targets.shape != (self.rows,):
            raise ValueError(f"targets must have shape {(self.rows,)}")
        if targets.dtype != wp.int32 or targets.device != self.device:
            raise ValueError("targets must be int32 on the output-plan device")

    def forward(
        self, x: wp.array, targets: wp.array, *, reduction: str = "mean"
    ) -> wp.array:
        """Compute and return the scalar loss in fixed storage."""
        self._inputs(x, targets)
        self.norm.forward(x.reshape(self.norm.shape), self.norm_weight)
        linear_forward(
            self.normalized, self.lm_head, self.logits, cublas=self.cublas
        )
        return self.loss_plan.forward(self.logits, targets, reduction=reduction)

    def backward(
        self,
        x: wp.array,
        targets: wp.array,
        *,
        loss_scale: float = 1.0,
        reduction: str = "mean",
    ) -> wp.array:
        """Overwrite logits with dlogits and return the FP32 stack gradient."""
        self._inputs(x, targets)
        logits_grad = self.loss_plan.backward(
            self.logits,
            targets,
            loss_scale=loss_scale,
            reduction=reduction,
        )
        linear_backward(
            self.normalized,
            self.lm_head,
            logits_grad,
            self.normalized_grad,
            cublas=self.cublas,
        )
        cast_to_float32(self.normalized_grad, self.normalized_grad_fp32)
        self.norm.backward(
            x.reshape(self.norm.shape),
            self.norm_weight,
            self.normalized_grad_fp32.reshape(self.norm.shape),
        )
        return self.input_grad

    @property
    def loss(self) -> wp.array:
        return self.loss_plan.loss

    @property
    def valid_count(self) -> wp.array:
        return self.loss_plan.valid_count
