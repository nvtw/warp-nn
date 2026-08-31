# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-buffer composition shared by Qwen and Muse transformer stacks."""

from collections.abc import Sequence

import warp as wp

from .bridges import cast_from_float32


class LoRATransformerStackPlan:
    """Chain compatible LoRA transformer blocks through one reusable bridge.

    Each block retains its own forward state. Backward walks those blocks in
    reverse and casts the FP32 activation gradient into one shared low-precision
    buffer before invoking the preceding block. Adapter gradients remain FP32
    and are owned by the one adapter collection shared by every block.
    """

    def __init__(self, blocks: Sequence[object]):
        blocks = tuple(blocks)
        if not blocks:
            raise ValueError("a transformer stack requires at least one block")
        first = blocks[0]
        required = ("adapters", "device", "dtype", "rows", "hidden", "output")
        if len({id(block) for block in blocks}) != len(blocks):
            raise ValueError(
                "a transformer plan instance cannot be reused as two layers"
            )
        if any(not hasattr(first, name) for name in required):
            raise TypeError(
                "transformer blocks do not expose the training-plan interface"
            )
        if first.dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("transformer stack storage must use FP16 or BF16")
        for index, block in enumerate(blocks):
            if any(not hasattr(block, name) for name in required):
                raise TypeError(
                    f"transformer block {index} has an incomplete interface"
                )
            if block.adapters is not first.adapters:
                raise ValueError(
                    "all transformer blocks must share one adapter collection"
                )
            if (
                block.device != first.device
                or block.dtype != first.dtype
                or block.rows != first.rows
                or block.hidden != first.hidden
                or block.output.shape != (first.rows, first.hidden)
            ):
                raise ValueError("all transformer blocks must share one fixed geometry")
        self.blocks = blocks
        self.adapters = first.adapters
        self.device = first.device
        self.dtype = first.dtype
        self.rows = first.rows
        self.hidden = first.hidden
        self.shape = (self.rows, self.hidden)
        self.output = blocks[-1].output
        self.boundary_grad = wp.empty(self.shape, dtype=self.dtype, device=self.device)

    def _input(self, value: wp.array, name: str) -> None:
        if not isinstance(value, wp.array) or value.shape != self.shape:
            raise ValueError(f"{name} must have shape {self.shape}")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError(f"{name} must match stack dtype and device")
        if not value.is_contiguous:
            raise ValueError(f"{name} must be contiguous")

    def forward(
        self, x: wp.array, lengths: wp.array, positions=None, cosine=None, sine=None
    ) -> wp.array:
        """Execute all layers in model order into their fixed output buffers."""
        self._input(x, "x")
        value = x
        for block in self.blocks:
            value = block.forward(value, lengths, positions, cosine, sine)
        return value

    def backward(
        self,
        x: wp.array,
        lengths: wp.array,
        grad_output: wp.array,
        positions=None,
        cosine=None,
        sine=None,
        *,
        accumulate: bool = False,
    ) -> wp.array:
        """Reverse every layer and return the first layer's fixed FP32 gradient."""
        self._input(x, "x")
        self._input(grad_output, "grad_output")
        gradient = grad_output
        for index in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[index]
            block_input = x if index == 0 else self.blocks[index - 1].output
            input_gradient = block.backward(
                block_input,
                lengths,
                gradient,
                positions,
                cosine,
                sine,
                accumulate=accumulate,
            )
            if index:
                cast_from_float32(input_gradient, self.boundary_grad)
                gradient = self.boundary_grad
        return input_gradient

    def zero_grad(self) -> None:
        """Clear all adapter gradients for a fresh accumulation window."""
        self.adapters.zero_grad()

    def step(self) -> None:
        """Update the shared adapter collection through its FP32 masters."""
        self.adapters.step()
