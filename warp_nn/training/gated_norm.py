# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Fixed-buffer RMSNorm-times-SiLU gate with explicit training gradients."""

from dataclasses import dataclass
from functools import lru_cache
import math

import warp as wp


_DTYPES = (wp.float16, wp.bfloat16, wp.float32)


@dataclass(frozen=True)
class _Kernels:
    forward: object
    backward: object


@lru_cache(maxsize=None)
def _kernels(dtype: type, width: int):
    DTYPE = dtype
    WIDTH = width

    @wp.func
    def square(value: DTYPE):
        value_fp32 = wp.float32(value)
        return value_fp32 * value_fp32

    @wp.func
    def to_fp32(value: DTYPE):
        return wp.float32(value)

    @wp.func
    def silu(value: DTYPE):
        value_fp32 = wp.float32(value)
        return value_fp32 / (wp.float32(1.0) + wp.exp(-value_fp32))

    @wp.func
    def apply(
        value: DTYPE,
        gate: DTYPE,
        scale: DTYPE,
        inverse: wp.float32,
    ):
        return DTYPE(wp.float32(value) * inverse * wp.float32(scale) * silu(gate))

    @wp.func
    def weighted_gradient(
        gradient: DTYPE,
        gate: DTYPE,
        scale: DTYPE,
    ):
        return wp.float32(gradient) * wp.float32(scale) * silu(gate)

    @wp.func
    def input_gradient(
        value: DTYPE,
        weighted: wp.float32,
        inverse: wp.float32,
        dot: wp.float32,
    ):
        return inverse * weighted - (
            wp.float32(value) * inverse * inverse * inverse * dot / wp.float32(WIDTH)
        )

    @wp.func
    def gate_gradient(
        value: DTYPE,
        gate: DTYPE,
        scale: DTYPE,
        gradient: DTYPE,
        inverse: wp.float32,
    ):
        gate_fp32 = wp.float32(gate)
        probability = wp.float32(1.0) / (wp.float32(1.0) + wp.exp(-gate_fp32))
        derivative = probability * (
            wp.float32(1.0) + gate_fp32 * (wp.float32(1.0) - probability)
        )
        return DTYPE(
            wp.float32(gradient)
            * wp.float32(value)
            * inverse
            * wp.float32(scale)
            * derivative
        )

    @wp.kernel(enable_backward=False, module="unique")
    def forward(
        x: wp.array2d(dtype=DTYPE),
        gate: wp.array2d(dtype=DTYPE),
        scale: wp.array1d(dtype=DTYPE),
        inverse: wp.array1d(dtype=wp.float32),
        output: wp.array2d(dtype=DTYPE),
        epsilon: wp.float32,
    ):
        row = wp.tid()
        values = wp.tile_load(x[row], shape=(WIDTH,))
        gates = wp.tile_load(gate[row], shape=(WIDTH,))
        scales = wp.tile_load(scale, shape=(WIDTH,))
        mean_square = wp.tile_extract(
            wp.tile_sum(wp.tile_map(square, values)), 0
        ) / wp.float32(WIDTH)
        inverse_value = wp.float32(1.0) / wp.sqrt(mean_square + epsilon)
        inverse[row] = inverse_value
        wp.tile_store(
            output[row],
            wp.tile_map(apply, values, gates, scales, inverse_value),
        )

    @wp.kernel(enable_backward=False, module="unique")
    def backward(
        x: wp.array2d(dtype=DTYPE),
        gate: wp.array2d(dtype=DTYPE),
        scale: wp.array1d(dtype=DTYPE),
        inverse: wp.array1d(dtype=wp.float32),
        output_grad: wp.array2d(dtype=DTYPE),
        input_grad: wp.array2d(dtype=wp.float32),
        gate_grad: wp.array2d(dtype=DTYPE),
    ):
        row = wp.tid()
        values = wp.tile_load(x[row], shape=(WIDTH,))
        gates = wp.tile_load(gate[row], shape=(WIDTH,))
        scales = wp.tile_load(scale, shape=(WIDTH,))
        gradients = wp.tile_load(output_grad[row], shape=(WIDTH,))
        inverse_value = inverse[row]
        weighted = wp.tile_map(weighted_gradient, gradients, gates, scales)
        dot = wp.tile_extract(wp.tile_sum(weighted * wp.tile_map(to_fp32, values)), 0)
        wp.tile_store(
            input_grad[row],
            wp.tile_map(input_gradient, values, weighted, inverse_value, dot),
        )
        wp.tile_store(
            gate_grad[row],
            wp.tile_map(
                gate_gradient,
                values,
                gates,
                scales,
                gradients,
                inverse_value,
            ),
        )

    result = _Kernels(forward, backward)
    for kernel in result.__dict__.values():
        kernel.module.options["enable_backward"] = False
    return result


class GatedRMSNormPlan:
    """Apply per-head RMSNorm and a SiLU gate with frozen scales."""

    def __init__(
        self, rows: int, width: int, dtype: type, *, epsilon: float, device=None
    ):
        if min(rows, width) <= 0:
            raise ValueError("gated RMSNorm dimensions must be positive")
        if dtype not in _DTYPES:
            raise TypeError("gated RMSNorm must use FP32, FP16, or BF16")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("gated RMSNorm epsilon must be finite and positive")
        self.rows, self.width, self.dtype = rows, width, dtype
        self.device, self.epsilon = wp.get_device(device), float(epsilon)
        self._kernels = _kernels(dtype, width)
        self.inverse = wp.empty(rows, dtype=wp.float32, device=self.device)
        self.output = wp.empty((rows, width), dtype=dtype, device=self.device)
        self.input_grad = wp.empty((rows, width), dtype=wp.float32, device=self.device)
        self.gate_grad = wp.empty((rows, width), dtype=dtype, device=self.device)
        self.block_dim = min(512, max(32, 1 << (width - 1).bit_length()))

    def forward(self, x, gate, scale):
        self._validate(x, gate, scale)
        wp.launch_tiled(
            self._kernels.forward,
            dim=self.rows,
            inputs=[x, gate, scale, self.inverse, self.output, self.epsilon],
            block_dim=self.block_dim,
            device=self.device,
        )
        return self.output

    def backward(self, x, gate, scale, output_grad):
        self._validate(x, gate, scale)
        self._validate_array(output_grad, (self.rows, self.width), self.dtype)
        wp.launch_tiled(
            self._kernels.backward,
            dim=self.rows,
            inputs=[
                x,
                gate,
                scale,
                self.inverse,
                output_grad,
                self.input_grad,
                self.gate_grad,
            ],
            block_dim=self.block_dim,
            device=self.device,
        )
        return self.input_grad, self.gate_grad

    def _validate(self, x, gate, scale):
        self._validate_array(x, (self.rows, self.width), self.dtype)
        self._validate_array(gate, (self.rows, self.width), self.dtype)
        self._validate_array(scale, (self.width,), self.dtype)

    def _validate_array(self, array, shape, dtype):
        if (
            not isinstance(array, wp.array)
            or array.shape != shape
            or array.dtype != dtype
            or array.device != self.device
            or not array.is_contiguous
        ):
            raise ValueError(
                f"gated RMSNorm input must be contiguous {shape} {dtype} on {self.device}"
            )
