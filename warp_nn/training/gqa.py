# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Fixed-buffer composition of LoRA projections and grouped-query attention."""

from functools import lru_cache

import warp as wp

from .adapters import LoRAAdapterCollection
from .attention import gqa_attention_backward, gqa_attention_forward
from .bridges import cast_from_float32, merge_heads, split_heads


@lru_cache(maxsize=None)
def _sum_input_gradients_kernel(dtype: type):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        query: wp.array1d(dtype=DTYPE),
        key: wp.array1d(dtype=DTYPE),
        value: wp.array1d(dtype=DTYPE),
        output: wp.array1d(dtype=wp.float32),
    ):
        index = wp.tid()
        output[index] = (
            wp.float32(query[index]) + wp.float32(key[index]) + wp.float32(value[index])
        )

    kernel.module.options["enable_backward"] = False
    return kernel


class GQALoRAAttentionPlan:
    """Compose four named LoRA projections with exact causal/sliding GQA.

    The plan owns every intermediate and saved backward buffer. It intentionally
    covers the reusable attention core only: model-specific Q/K normalization and
    RoPE belong in surrounding plans. One instance supports one outstanding
    forward and is safe to capture because execution performs no allocation.
    """

    def __init__(
        self,
        adapters: LoRAAdapterCollection,
        *,
        query: str,
        key: str,
        value: str,
        output: str,
        batch: int,
        sequence: int,
        query_heads: int,
        kv_heads: int,
        head_size: int,
        window: int = 0,
    ):
        if min(batch, sequence, query_heads, kv_heads, head_size) <= 0:
            raise ValueError("GQA LoRA dimensions must be positive")
        if query_heads % kv_heads:
            raise ValueError("query heads must be divisible by key/value heads")
        if window < 0:
            raise ValueError("GQA window must be non-negative")
        names = (query, key, value, output)
        if len(set(names)) != 4 or any(name not in adapters.targets for name in names):
            raise ValueError(
                "GQA LoRA projection names must be distinct adapter targets"
            )

        rows = batch * sequence
        targets = tuple(adapters.targets[name] for name in names)
        dtype = targets[0].weight.dtype
        device = targets[0].weight.device
        if any(
            target.weight.dtype != dtype or target.weight.device != device
            for target in targets
        ):
            raise ValueError("GQA LoRA targets must share one dtype and device")
        if any(target.plan.rows != rows for target in targets):
            raise ValueError("GQA LoRA target row counts must equal batch * sequence")

        hidden = targets[0].weight.shape[1]
        query_width = query_heads * head_size
        kv_width = kv_heads * head_size
        expected = (
            (query_width, hidden),
            (kv_width, hidden),
            (kv_width, hidden),
            (hidden, query_width),
        )
        actual = tuple(target.weight.shape for target in targets)
        if actual != expected:
            raise ValueError(
                f"GQA LoRA projection shapes must be {expected}, got {actual}"
            )

        self.adapters = adapters
        self.names = names
        self.device = device
        self.dtype = dtype
        self.batch = batch
        self.sequence = sequence
        self.rows = rows
        self.hidden = hidden
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_size = head_size
        self.window = window
        self.scale = head_size**-0.5

        query_shape = (batch, query_heads, sequence, head_size)
        kv_shape = (batch, kv_heads, sequence, head_size)
        self.query = wp.empty(query_shape, dtype=dtype, device=device)
        self.key = wp.empty(kv_shape, dtype=dtype, device=device)
        self.value = wp.empty(kv_shape, dtype=dtype, device=device)
        self.core = wp.empty(query_shape, dtype=dtype, device=device)
        self.merged = wp.empty((rows, query_width), dtype=dtype, device=device)
        self.lse = wp.empty(query_shape[:3], dtype=wp.float32, device=device)
        self.workspace = wp.empty(query_shape, dtype=wp.float32, device=device)
        self.core_grad = wp.empty(query_shape, dtype=dtype, device=device)
        self.query_grad = wp.empty(query_shape, dtype=wp.float32, device=device)
        self.key_grad = wp.empty(kv_shape, dtype=wp.float32, device=device)
        self.value_grad = wp.empty(kv_shape, dtype=wp.float32, device=device)
        self.delta = wp.empty(query_shape[:3], dtype=wp.float32, device=device)
        self.query_grad_storage = wp.empty(query_shape, dtype=dtype, device=device)
        self.key_grad_storage = wp.empty(kv_shape, dtype=dtype, device=device)
        self.value_grad_storage = wp.empty(kv_shape, dtype=dtype, device=device)
        self.input_grad = wp.empty((rows, hidden), dtype=wp.float32, device=device)

    @property
    def output(self) -> wp.array:
        """Return the fixed output buffer of the O projection."""
        return self.adapters.targets[self.names[3]].plan.output

    def forward(self, x: wp.array, lengths: wp.array) -> wp.array:
        """Execute Q/K/V projections, exact GQA, and the O projection."""
        query_name, key_name, value_name, output_name = self.names
        split_heads(self.adapters.forward(query_name, x), self.query)
        split_heads(self.adapters.forward(key_name, x), self.key)
        split_heads(self.adapters.forward(value_name, x), self.value)
        gqa_attention_forward(
            self.query,
            self.key,
            self.value,
            lengths,
            self.core,
            self.lse,
            self.workspace,
            scale=self.scale,
            window=self.window,
        )
        merge_heads(self.core, self.merged)
        return self.adapters.forward(output_name, self.merged)

    def backward(
        self,
        x: wp.array,
        lengths: wp.array,
        grad_output: wp.array,
        *,
        accumulate: bool = False,
    ) -> wp.array:
        """Run O, Flash-GQA, and Q/K/V backward into fixed FP32 input gradient."""
        query_name, key_name, value_name, output_name = self.names
        merged_grad = self.adapters.backward(
            output_name, self.merged, grad_output, accumulate=accumulate
        )
        split_heads(merged_grad, self.core_grad)
        gqa_attention_backward(
            self.query,
            self.key,
            self.value,
            self.core_grad,
            lengths,
            self.lse,
            self.query_grad,
            self.key_grad,
            self.value_grad,
            self.delta,
            scale=self.scale,
            window=self.window,
        )
        for gradient, storage, name in (
            (self.query_grad, self.query_grad_storage, query_name),
            (self.key_grad, self.key_grad_storage, key_name),
            (self.value_grad, self.value_grad_storage, value_name),
        ):
            cast_from_float32(gradient, storage)
            merge_heads(storage, self.adapters.targets[name].plan.output.grad)

        query_input = self.adapters.backward(
            query_name,
            x,
            self.adapters.targets[query_name].plan.output.grad,
            accumulate=accumulate,
        )
        key_input = self.adapters.backward(
            key_name,
            x,
            self.adapters.targets[key_name].plan.output.grad,
            accumulate=accumulate,
        )
        value_input = self.adapters.backward(
            value_name,
            x,
            self.adapters.targets[value_name].plan.output.grad,
            accumulate=accumulate,
        )
        wp.launch(
            _sum_input_gradients_kernel(self.dtype),
            dim=self.input_grad.size,
            inputs=[query_input.flatten(), key_input.flatten(), value_input.flatten()],
            outputs=[self.input_grad.flatten()],
            device=self.device,
        )
        return self.input_grad
