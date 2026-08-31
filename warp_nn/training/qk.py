# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Graph-safe per-head RMS normalization and split-half rotary transforms."""

from functools import lru_cache
import math

import warp as wp


@wp.kernel(enable_backward=False)
def _inverse_rms(
    mean_square: wp.array1d(dtype=wp.float32),
    epsilon: wp.float32,
    inverse: wp.array1d(dtype=wp.float32),
):
    row = wp.tid()
    inverse[row] = wp.float32(1.0) / wp.sqrt(mean_square[row] + epsilon)


@lru_cache(maxsize=None)
def _qk_kernels(dtype: type, rotary_dim: int, head_size: int):
    DTYPE = dtype
    ROTARY_DIM = rotary_dim
    HEAD_SIZE = head_size

    @wp.func
    def square(value: DTYPE):
        value_fp32 = wp.float32(value)
        return value_fp32 * value_fp32

    @wp.func
    def weighted_dot(
        value: DTYPE,
        weight: DTYPE,
        gradient: wp.float32,
        weight_offset: wp.float32,
    ):
        return gradient * (wp.float32(weight) + weight_offset) * wp.float32(value)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def mean_square(x: wp.array2d(dtype=DTYPE), output: wp.array1d(dtype=wp.float32)):
        row = wp.tid()
        values = wp.tile_load(x[row], shape=(HEAD_SIZE,))
        output[row] = wp.tile_extract(
            wp.tile_sum(wp.tile_map(square, values)), 0
        ) / wp.float32(HEAD_SIZE)

    @wp.kernel(enable_backward=False, module="unique")
    def forward(
        x: wp.array4d(dtype=DTYPE),
        weight: wp.array1d(dtype=DTYPE),
        positions: wp.array2d(dtype=wp.int64),
        cosine: wp.array2d(dtype=DTYPE),
        sine: wp.array2d(dtype=DTYPE),
        inverse: wp.array1d(dtype=wp.float32),
        scale: wp.float32,
        weight_offset: wp.float32,
        output: wp.array4d(dtype=DTYPE),
    ):
        batch, head, token, column = wp.tid()
        row = (batch * x.shape[1] + head) * x.shape[2] + token
        normalized = (
            wp.float32(x[batch, head, token, column])
            * inverse[row]
            * (wp.float32(weight[column]) + weight_offset)
            * scale
        )
        if column < ROTARY_DIM:
            half = ROTARY_DIM // 2
            cache_column = column % half
            partner = column + half if column < half else column - half
            sign = wp.float32(-1.0) if column < half else wp.float32(1.0)
            partner_value = (
                wp.float32(x[batch, head, token, partner])
                * inverse[row]
                * (wp.float32(weight[partner]) + weight_offset)
                * scale
            )
            position = positions[batch, token]
            normalized = normalized * wp.float32(
                cosine[position, cache_column]
            ) + sign * partner_value * wp.float32(sine[position, cache_column])
        output[batch, head, token, column] = DTYPE(normalized)

    @wp.kernel(enable_backward=False, module="unique")
    def reverse_rotation(
        x: wp.array4d(dtype=DTYPE),
        positions: wp.array2d(dtype=wp.int64),
        cosine: wp.array2d(dtype=DTYPE),
        sine: wp.array2d(dtype=DTYPE),
        output_grad: wp.array4d(dtype=wp.float32),
        scale: wp.float32,
        normalized_grad: wp.array4d(dtype=wp.float32),
    ):
        batch, head, token, column = wp.tid()
        gradient = output_grad[batch, head, token, column]
        if column < ROTARY_DIM:
            half = ROTARY_DIM // 2
            cache_column = column % half
            partner = column + half if column < half else column - half
            sign = wp.float32(-1.0) if column < half else wp.float32(1.0)
            position = positions[batch, token]
            gradient = gradient * wp.float32(
                cosine[position, cache_column]
            ) - sign * output_grad[batch, head, token, partner] * wp.float32(
                sine[position, cache_column]
            )
        gradient *= scale
        normalized_grad[batch, head, token, column] = gradient

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def reduce_dot(
        x: wp.array2d(dtype=DTYPE),
        weight: wp.array1d(dtype=DTYPE),
        normalized_grad: wp.array2d(dtype=wp.float32),
        weight_offset: wp.float32,
        dot: wp.array1d(dtype=wp.float32),
    ):
        row = wp.tid()
        values = wp.tile_load(x[row], shape=(HEAD_SIZE,))
        weights = wp.tile_load(weight, shape=(HEAD_SIZE,))
        gradients = wp.tile_load(normalized_grad[row], shape=(HEAD_SIZE,))
        dot[row] = wp.tile_extract(
            wp.tile_sum(
                wp.tile_map(weighted_dot, values, weights, gradients, weight_offset)
            ),
            0,
        )

    @wp.kernel(enable_backward=False, module="unique")
    def input_gradient(
        x: wp.array4d(dtype=DTYPE),
        weight: wp.array1d(dtype=DTYPE),
        inverse: wp.array1d(dtype=wp.float32),
        normalized_grad: wp.array4d(dtype=wp.float32),
        dot: wp.array1d(dtype=wp.float32),
        weight_offset: wp.float32,
        output: wp.array4d(dtype=wp.float32),
    ):
        batch, head, token, column = wp.tid()
        row = (batch * x.shape[1] + head) * x.shape[2] + token
        inverse_value = inverse[row]
        output[batch, head, token, column] = normalized_grad[
            batch, head, token, column
        ] * (wp.float32(weight[column]) + weight_offset) * inverse_value - wp.float32(
            x[batch, head, token, column]
        ) * inverse_value * inverse_value * inverse_value * dot[row] / wp.float32(
            x.shape[3]
        )

    for kernel in (mean_square, forward, reverse_rotation, reduce_dot, input_gradient):
        kernel.module.options["enable_backward"] = False
    return mean_square, forward, reverse_rotation, reduce_dot, input_gradient


class QKTransformPlan:
    """Apply frozen per-head RMSNorm, optional Q scaling, and optional RoPE.

    The analytical backward returns FP32 input gradients and intentionally does
    not train the normalization weight. ``rotary_dim=0`` selects normalization
    only, which covers Muse's NoPE full-attention layers.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        sequence: int,
        head_size: int,
        dtype: type,
        *,
        rotary_dim: int | None = None,
        epsilon: float = 1.0e-6,
        scale: float = 1.0,
        weight_offset: float = 0.0,
        device=None,
    ):
        if dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Q/K transforms support FP16 or BF16 storage")
        if min(batch, heads, sequence, head_size) <= 0:
            raise ValueError("Q/K transform dimensions must be positive")
        if rotary_dim is None:
            rotary_dim = head_size
        if rotary_dim < 0 or rotary_dim > head_size or rotary_dim % 2:
            raise ValueError("rotary_dim must be even and between zero and head_size")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("Q/K epsilon must be finite and positive")
        if not math.isfinite(scale) or not math.isfinite(weight_offset):
            raise ValueError("Q/K scale and weight offset must be finite")
        self.device = wp.get_device(device)
        self.shape = (batch, heads, sequence, head_size)
        self.row_shape = (batch * heads * sequence, head_size)
        self.dtype = dtype
        self.rotary_dim = rotary_dim
        self.epsilon = float(epsilon)
        self.scale = float(scale)
        self.weight_offset = float(weight_offset)
        self._kernels = _qk_kernels(dtype, rotary_dim, head_size)
        self.block_dim = min(512, max(32, 1 << (head_size - 1).bit_length()))
        rows = self.row_shape[0]
        self.mean_square = wp.empty(rows, dtype=wp.float32, device=self.device)
        self.inverse_rms = wp.empty(rows, dtype=wp.float32, device=self.device)
        self.output = wp.empty(self.shape, dtype=dtype, device=self.device)
        self.normalized_grad = wp.empty(
            self.shape, dtype=wp.float32, device=self.device
        )
        self.dot = wp.empty(rows, dtype=wp.float32, device=self.device)
        self.input_grad = wp.empty(self.shape, dtype=wp.float32, device=self.device)
        self._dummy_positions = wp.zeros(
            (batch, sequence), dtype=wp.int64, device=self.device
        )
        self._dummy_cache = wp.ones((1, 1), dtype=dtype, device=self.device)

    def _check(self, array, shape, dtype, name):
        if (
            not isinstance(array, wp.array)
            or array.shape != shape
            or array.dtype != dtype
        ):
            raise TypeError(f"{name} must be a {dtype} Warp array with shape {shape}")
        if array.device != self.device or not array.is_contiguous:
            raise ValueError(f"{name} must be contiguous on {self.device}")

    def _rope_inputs(self, positions, cosine, sine):
        if self.rotary_dim == 0:
            return self._dummy_positions, self._dummy_cache, self._dummy_cache
        batch, _, sequence, _ = self.shape
        self._check(positions, (batch, sequence), wp.int64, "positions")
        frequency = self.rotary_dim // 2
        for array, name in ((cosine, "cosine"), (sine, "sine")):
            if (
                not isinstance(array, wp.array)
                or array.ndim != 2
                or array.shape[1] != frequency
                or array.dtype != self.dtype
            ):
                raise TypeError(
                    f"{name} must be a {self.dtype} Warp matrix with width {frequency}"
                )
            if array.device != self.device or not array.is_contiguous:
                raise ValueError(f"{name} must be contiguous on {self.device}")
        return positions, cosine, sine

    def forward(self, x, weight, positions=None, cosine=None, sine=None):
        """Write and return the fixed transformed head tensor."""
        self._check(x, self.shape, self.dtype, "x")
        self._check(weight, (self.shape[3],), self.dtype, "weight")
        positions, cosine, sine = self._rope_inputs(positions, cosine, sine)
        wp.launch_tiled(
            self._kernels[0],
            dim=self.row_shape[0],
            inputs=[x.reshape(self.row_shape), self.mean_square],
            block_dim=self.block_dim,
            device=self.device,
        )
        wp.launch(
            _inverse_rms,
            dim=self.row_shape[0],
            inputs=[self.mean_square, self.epsilon],
            outputs=[self.inverse_rms],
            device=self.device,
        )
        wp.launch(
            self._kernels[1],
            dim=self.shape,
            inputs=[
                x,
                weight,
                positions,
                cosine,
                sine,
                self.inverse_rms,
                self.scale,
                self.weight_offset,
            ],
            outputs=[self.output],
            device=self.device,
        )
        return self.output

    def backward(self, x, weight, output_grad, positions=None, cosine=None, sine=None):
        """Reverse the latest transform and return the fixed FP32 input gradient."""
        self._check(x, self.shape, self.dtype, "x")
        self._check(weight, (self.shape[3],), self.dtype, "weight")
        self._check(output_grad, self.shape, wp.float32, "output gradient")
        positions, cosine, sine = self._rope_inputs(positions, cosine, sine)
        wp.launch(
            self._kernels[2],
            dim=self.shape,
            inputs=[
                x,
                positions,
                cosine,
                sine,
                output_grad,
                self.scale,
            ],
            outputs=[self.normalized_grad],
            device=self.device,
        )
        wp.launch_tiled(
            self._kernels[3],
            dim=self.row_shape[0],
            inputs=[
                x.reshape(self.row_shape),
                weight,
                self.normalized_grad.reshape(self.row_shape),
                self.weight_offset,
                self.dot,
            ],
            block_dim=self.block_dim,
            device=self.device,
        )
        wp.launch(
            self._kernels[4],
            dim=self.shape,
            inputs=[
                x,
                weight,
                self.inverse_rms,
                self.normalized_grad,
                self.dot,
                self.weight_offset,
            ],
            outputs=[self.input_grad],
            device=self.device,
        )
        return self.input_grad
