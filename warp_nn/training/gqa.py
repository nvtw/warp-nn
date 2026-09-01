# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Fixed-buffer composition of LoRA projections and grouped-query attention."""

from functools import lru_cache

import warp as wp

from .adapters import LoRAAdapterCollection
from .attention import gqa_attention_backward, gqa_attention_forward
from .bridges import cast_from_float32, merge_heads, split_heads
from .qk import QKTransformPlan


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


@lru_cache(maxsize=None)
def _gate_kernels(dtype: type):
    DTYPE = dtype

    @wp.func
    def sigmoid(value: wp.float32):
        if value >= wp.float32(0.0):
            return wp.float32(1.0) / (wp.float32(1.0) + wp.exp(-value))
        exponential = wp.exp(value)
        return exponential / (wp.float32(1.0) + exponential)

    @wp.kernel(enable_backward=False, module="unique")
    def forward(
        core: wp.array1d(dtype=DTYPE),
        gate: wp.array1d(dtype=DTYPE),
        output: wp.array1d(dtype=DTYPE),
    ):
        index = wp.tid()
        output[index] = DTYPE(
            wp.float32(core[index]) * sigmoid(wp.float32(gate[index]))
        )

    @wp.kernel(enable_backward=False, module="unique")
    def backward(
        core: wp.array1d(dtype=DTYPE),
        gate: wp.array1d(dtype=DTYPE),
        output_grad: wp.array1d(dtype=DTYPE),
        core_grad: wp.array1d(dtype=DTYPE),
        gate_grad: wp.array1d(dtype=DTYPE),
    ):
        index = wp.tid()
        gate_value = sigmoid(wp.float32(gate[index]))
        gradient = wp.float32(output_grad[index])
        core_grad[index] = DTYPE(gradient * gate_value)
        gate_grad[index] = DTYPE(
            gradient
            * wp.float32(core[index])
            * gate_value
            * (wp.float32(1.0) - gate_value)
        )

    @wp.kernel(enable_backward=False, module="unique")
    def add_input(gate: wp.array1d(dtype=DTYPE), output: wp.array1d(dtype=wp.float32)):
        index = wp.tid()
        output[index] += wp.float32(gate[index])

    for kernel in (forward, backward, add_input):
        kernel.module.options["enable_backward"] = False
    return forward, backward, add_input


@lru_cache(maxsize=None)
def _packed_query_gate_kernels(dtype: type, interleaved: bool = False):
    """Split and reassemble per-head ``[Q, gate]`` projection storage."""
    DTYPE = dtype
    INTERLEAVED = interleaved

    @wp.kernel(enable_backward=False, module="unique")
    def unpack(
        packed: wp.array2d(dtype=DTYPE),
        query: wp.array4d(dtype=DTYPE),
        gate: wp.array2d(dtype=DTYPE),
    ):
        row, head, column = wp.tid()
        sequence = query.shape[2]
        head_size = query.shape[3]
        batch = row // sequence
        token = row % sequence
        source = head * head_size * 2
        source_column = column
        if INTERLEAVED:
            half = head_size // 2
            source_column = (column % half) * 2 + column // half
        query[batch, head, token, column] = packed[row, source + source_column]
        gate[row, head * head_size + column] = packed[row, source + head_size + column]

    @wp.kernel(enable_backward=False, module="unique")
    def pack_gradient(
        query: wp.array4d(dtype=DTYPE),
        gate: wp.array2d(dtype=DTYPE),
        packed: wp.array2d(dtype=DTYPE),
    ):
        row, head, column = wp.tid()
        sequence = query.shape[2]
        head_size = query.shape[3]
        batch = row // sequence
        token = row % sequence
        destination = head * head_size * 2
        destination_column = column
        if INTERLEAVED:
            half = head_size // 2
            destination_column = (column % half) * 2 + column // half
        packed[row, destination + destination_column] = query[
            batch, head, token, column
        ]
        packed[row, destination + head_size + column] = gate[
            row, head * head_size + column
        ]

    for kernel in (unpack, pack_gradient):
        kernel.module.options["enable_backward"] = False
    return unpack, pack_gradient


class GQALoRAAttentionPlan:
    """Compose named LoRA projections with exact causal/sliding GQA.

    The plan owns every intermediate and saved backward buffer. It supports optional
    frozen Q/K normalization and RoPE transforms plus a separate or packed sigmoid gate. One
    instance supports one outstanding
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
        gate: str | None = None,
        packed_query_gate: bool = False,
        batch: int,
        sequence: int,
        query_heads: int,
        kv_heads: int,
        head_size: int,
        window: int = 0,
        query_transform: QKTransformPlan | None = None,
        key_transform: QKTransformPlan | None = None,
        query_norm_weight: wp.array | None = None,
        key_norm_weight: wp.array | None = None,
        interleaved_rope_weights: bool = False,
    ):
        if min(batch, sequence, query_heads, kv_heads, head_size) <= 0:
            raise ValueError("GQA LoRA dimensions must be positive")
        if query_heads % kv_heads:
            raise ValueError("query heads must be divisible by key/value heads")
        if window < 0:
            raise ValueError("GQA window must be non-negative")
        if interleaved_rope_weights and head_size % 2:
            raise ValueError("interleaved RoPE weights require an even head size")
        if gate is not None and packed_query_gate:
            raise ValueError("separate and packed GQA gates are mutually exclusive")
        names = (query, key, value, output)
        projection_names = names + ((gate,) if gate is not None else ())
        if len(set(projection_names)) != len(projection_names) or any(
            name not in adapters.targets for name in projection_names
        ):
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
            (query_width * (2 if packed_query_gate else 1), hidden),
            (kv_width, hidden),
            (kv_width, hidden),
            (hidden, query_width),
        )
        actual = tuple(target.weight.shape for target in targets)
        if actual != expected:
            raise ValueError(
                f"GQA LoRA projection shapes must be {expected}, got {actual}"
            )
        if gate is not None:
            gate_target = adapters.targets[gate]
            if (
                gate_target.weight.shape != (query_width, hidden)
                or gate_target.weight.dtype != dtype
                or gate_target.weight.device != device
                or gate_target.plan.rows != rows
            ):
                raise ValueError(
                    "GQA gate projection shape, dtype, device, and rows must match"
                )

        transform_items = (
            (
                query_transform,
                query_norm_weight,
                (batch, query_heads, sequence, head_size),
            ),
            (key_transform, key_norm_weight, (batch, kv_heads, sequence, head_size)),
        )
        enabled = tuple(transform is not None for transform, _, _ in transform_items)
        if enabled not in ((False, False), (True, True)):
            raise ValueError("query and key transforms must be enabled together")
        for transform, weight, shape in transform_items:
            if transform is None:
                if weight is not None:
                    raise ValueError("normalization weights require Q/K transforms")
                continue
            if (
                transform.shape != shape
                or transform.dtype != dtype
                or transform.device != device
            ):
                raise ValueError("Q/K transform shape, dtype, and device must match")
            if (
                not isinstance(weight, wp.array)
                or weight.shape != (head_size,)
                or weight.dtype != dtype
                or weight.device != device
            ):
                raise ValueError("Q/K normalization weights must match head storage")

        self.adapters = adapters
        self.names = names
        self.gate_name = gate
        self.packed_query_gate = packed_query_gate
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
        self.interleaved_rope_weights = bool(interleaved_rope_weights)
        self.scale = head_size**-0.5
        self.query_transform = query_transform
        self.key_transform = key_transform
        self.query_norm_weight = query_norm_weight
        self.key_norm_weight = key_norm_weight

        query_shape = (batch, query_heads, sequence, head_size)
        kv_shape = (batch, kv_heads, sequence, head_size)
        self.query = wp.empty(query_shape, dtype=dtype, device=device)
        self.key = wp.empty(kv_shape, dtype=dtype, device=device)
        self.value = wp.empty(kv_shape, dtype=dtype, device=device)
        self.core = wp.empty(query_shape, dtype=dtype, device=device)
        self.merged = wp.empty((rows, query_width), dtype=dtype, device=device)
        self.packed_gate = (
            wp.empty((rows, query_width), dtype=dtype, device=device)
            if packed_query_gate
            else None
        )
        self.packed_gate_grad = (
            wp.empty((rows, query_width), dtype=dtype, device=device)
            if packed_query_gate
            else None
        )
        self.gated = (
            wp.empty((rows, query_width), dtype=dtype, device=device)
            if gate is not None or packed_query_gate
            else None
        )
        self.merged_core_grad = (
            wp.empty((rows, query_width), dtype=dtype, device=device)
            if gate is not None or packed_query_gate
            else None
        )
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

    def forward(
        self,
        x: wp.array,
        lengths: wp.array,
        positions=None,
        cosine=None,
        sine=None,
        *,
        segment_bounds=None,
    ) -> wp.array:
        """Execute Q/K/V projections, exact GQA, and the O projection."""
        query_name, key_name, value_name, output_name = self.names
        gate_values = (
            self.adapters.forward(self.gate_name, x)
            if self.gate_name is not None
            else None
        )
        query_projection = self.adapters.forward(query_name, x)
        if self.packed_query_gate:
            wp.launch(
                _packed_query_gate_kernels(self.dtype, self.interleaved_rope_weights)[
                    0
                ],
                dim=(self.rows, self.query_heads, self.head_size),
                inputs=[query_projection],
                outputs=[self.query, self.packed_gate],
                device=self.device,
            )
            gate_values = self.packed_gate
        else:
            split_heads(
                query_projection,
                self.query,
                interleaved=self.interleaved_rope_weights,
            )
        split_heads(
            self.adapters.forward(key_name, x),
            self.key,
            interleaved=self.interleaved_rope_weights,
        )
        split_heads(self.adapters.forward(value_name, x), self.value)
        query_ready, key_ready = self.query, self.key
        if self.query_transform is not None:
            query_ready = self.query_transform.forward(
                self.query, self.query_norm_weight, positions, cosine, sine
            )
            key_ready = self.key_transform.forward(
                self.key, self.key_norm_weight, positions, cosine, sine
            )
        gqa_attention_forward(
            query_ready,
            key_ready,
            self.value,
            lengths,
            self.core,
            self.lse,
            self.workspace,
            segment_bounds=segment_bounds,
            scale=self.scale,
            window=self.window,
        )
        merge_heads(self.core, self.merged)
        projection_input = self.merged
        if gate_values is not None:
            wp.launch(
                _gate_kernels(self.dtype)[0],
                dim=self.merged.size,
                inputs=[self.merged.flatten(), gate_values.flatten()],
                outputs=[self.gated.flatten()],
                device=self.device,
            )
            projection_input = self.gated
        return self.adapters.forward(output_name, projection_input)

    def backward(
        self,
        x: wp.array,
        lengths: wp.array,
        grad_output: wp.array,
        positions=None,
        cosine=None,
        sine=None,
        *,
        segment_bounds=None,
        accumulate: bool = False,
    ) -> wp.array:
        """Run O, Flash-GQA, and Q/K/V backward into fixed FP32 input gradient."""
        query_name, key_name, value_name, output_name = self.names
        has_gate = self.gate_name is not None or self.packed_query_gate
        projection_input = self.gated if has_gate else self.merged
        merged_grad = self.adapters.backward(
            output_name, projection_input, grad_output, accumulate=accumulate
        )
        core_gradient = merged_grad
        if has_gate:
            gate_values = (
                self.adapters.targets[self.gate_name].plan.output
                if self.gate_name is not None
                else self.packed_gate
            )
            gate_grad = (
                gate_values.grad
                if self.gate_name is not None
                else self.packed_gate_grad
            )
            wp.launch(
                _gate_kernels(self.dtype)[1],
                dim=self.merged.size,
                inputs=[
                    self.merged.flatten(),
                    gate_values.flatten(),
                    merged_grad.flatten(),
                ],
                outputs=[
                    self.merged_core_grad.flatten(),
                    gate_grad.flatten(),
                ],
                device=self.device,
            )
            core_gradient = self.merged_core_grad
        split_heads(core_gradient, self.core_grad)
        query_ready = (
            self.query if self.query_transform is None else self.query_transform.output
        )
        key_ready = (
            self.key if self.key_transform is None else self.key_transform.output
        )
        gqa_attention_backward(
            query_ready,
            key_ready,
            self.value,
            self.core_grad,
            lengths,
            self.lse,
            self.query_grad,
            self.key_grad,
            self.value_grad,
            self.delta,
            segment_bounds=segment_bounds,
            scale=self.scale,
            window=self.window,
        )
        query_projection_grad, key_projection_grad = self.query_grad, self.key_grad
        if self.query_transform is not None:
            query_projection_grad = self.query_transform.backward(
                self.query,
                self.query_norm_weight,
                self.query_grad,
                positions,
                cosine,
                sine,
            )
            key_projection_grad = self.key_transform.backward(
                self.key,
                self.key_norm_weight,
                self.key_grad,
                positions,
                cosine,
                sine,
            )
        for gradient, storage in (
            (query_projection_grad, self.query_grad_storage),
            (key_projection_grad, self.key_grad_storage),
            (self.value_grad, self.value_grad_storage),
        ):
            cast_from_float32(gradient, storage)
        if self.packed_query_gate:
            wp.launch(
                _packed_query_gate_kernels(self.dtype, self.interleaved_rope_weights)[
                    1
                ],
                dim=(self.rows, self.query_heads, self.head_size),
                inputs=[self.query_grad_storage, self.packed_gate_grad],
                outputs=[self.adapters.targets[query_name].plan.output.grad],
                device=self.device,
            )
        else:
            merge_heads(
                self.query_grad_storage,
                self.adapters.targets[query_name].plan.output.grad,
                interleaved=self.interleaved_rope_weights,
            )
        merge_heads(
            self.key_grad_storage,
            self.adapters.targets[key_name].plan.output.grad,
            interleaved=self.interleaved_rope_weights,
        )
        merge_heads(
            self.value_grad_storage,
            self.adapters.targets[value_name].plan.output.grad,
        )

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
        if self.gate_name is not None:
            gate_input = self.adapters.backward(
                self.gate_name,
                x,
                self.adapters.targets[self.gate_name].plan.output.grad,
                accumulate=accumulate,
            )
            wp.launch(
                _gate_kernels(self.dtype)[2],
                dim=self.input_grad.size,
                inputs=[gate_input.flatten(), self.input_grad.flatten()],
                device=self.device,
            )
        return self.input_grad
