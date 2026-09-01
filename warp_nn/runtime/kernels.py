# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable Warp inference kernels and kernel factories."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import warp as wp

from warp_nn.modules.layers._common import tile_transposed_gemm_2d
from warp_nn.runtime._cuda import (
    decode_ue4m3,
    dp4a,
    encode_ue4m3,
    expand_int4x4_high,
    expand_int4x4_low,
    get_grouped_decode_projection,
    get_nvfp4_mma_projection,
    get_small_batch_grouped_projection,
    get_prefill_mma_projection,
    get_q8_grouped_decode_projection,
    get_q8_prefill_mma_projection,
    subgroup_sum,
    subgroup_max_broadcast,
    quantize_e2m1_pair,
    warp_max_broadcast,
)
from warp_nn.utils.config import get_kernel_config

# ---------------------------------------------------------------------------
# Inference kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _gemm_transb_kernel(
    A: wp.array2d[Any],  # (M, K)
    B: wp.array2d[Any],  # (N, K) — stored transposed
    bias: wp.array1d[Any],  # (N,)
    C: wp.array2d[Any],  # (M, N)
    K: int,
    alpha: float,
    beta: float,
):
    """``C = alpha * A @ B.T + beta * bias`` with ``transB=1``."""
    i, j = wp.tid()

    s = A.dtype(0.0)
    for k in range(K):
        s += A[i, k] * B[j, k]

    C[i, j] = A.dtype(alpha) * s + A.dtype(beta) * bias[j]


def _create_gemm_transb_tiled_kernel(config):
    """Build tiled ``A @ B.T`` using ``config`` for tile and block sizes."""

    @wp.kernel
    def kernel(
        A: wp.array2d[float],
        B: wp.array2d[float],
        bias: wp.array2d[float],
        alpha: float,
        beta: float,
        C: wp.array2d[float],
    ):
        """Compute tiled ``C = alpha * A @ B.T + beta * bias``."""
        i, j = wp.tid()
        offset = (i * wp.static(config.tile_2d[0]), j * wp.static(config.tile_2d[1]))
        out = wp.static(tile_transposed_gemm_2d(config.tile_2d))(B, A, index=(i, j))
        shape_t = (wp.static(config.tile_2d[1]), wp.static(config.tile_2d[0]))
        shape_b = (wp.static(config.tile_2d[1]), 1)
        offset_b = (j * wp.static(config.tile_2d[1]), 0)
        tiled_bias = wp.tile_broadcast(
            wp.tile_load(bias, shape=shape_b, offset=offset_b), shape=shape_t
        )
        wp.tile_store(
            C, wp.tile_transpose(alpha * out + beta * tiled_bias), offset=offset
        )

    return kernel


_GEMM_CONFIG = get_kernel_config()
_GEMM_TRANSB_TILED_KERNEL = _create_gemm_transb_tiled_kernel(_GEMM_CONFIG)


@wp.kernel
def _linear_kernel(
    x: wp.array2d[Any], weight: wp.array2d[Any], output: wp.array2d[Any]
):
    """Compute the fallback dense projection ``output = x @ weight.T``."""
    row, column = wp.tid()
    total = wp.float32(0.0)
    for inner in range(x.shape[1]):
        total += wp.float32(x[row, inner]) * wp.float32(weight[column, inner])
    output[row, column] = x.dtype(total)


def _create_linear_vector_kernel(dtype: type):
    """Build a row-by-weight dot-product kernel for small activation batches."""
    DTYPE = dtype
    TILE_WIDTH = 256

    @wp.func
    def multiply(left: DTYPE, right: DTYPE):
        return wp.float32(DTYPE(left)) * wp.float32(right)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        x: wp.array2d(dtype=DTYPE),
        weight: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        """Project small row batches with one reduction tile per output."""
        item = wp.tid()
        row = item / weight.shape[0]
        column = item % weight.shape[0]
        partials = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for inner_tile in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = inner_tile * TILE_WIDTH
            activations = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            weights = wp.tile_load(
                weight[column], shape=(TILE_WIDTH,), offset=(offset,)
            )
            partials += wp.tile_map(multiply, activations, weights)
        output[row, column] = DTYPE(wp.tile_extract(wp.tile_sum(partials), 0))

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_linear_vector_kernel(dtype: type):
    """Return a cached small-batch dense projection kernel."""
    return _create_linear_vector_kernel(dtype)


def _create_grouped_decode_linear_kernel(dtype: type):
    """Build a single-token projection sharing activations across eight outputs."""
    DTYPE = dtype
    project = get_grouped_decode_projection(dtype)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        x: wp.array2d(dtype=DTYPE),
        weight: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        inner: int,
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        wp.static(project)(x, weight, output, wp.tid(), inner)

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_grouped_decode_linear_kernel(dtype: type):
    """Return the cached eight-output single-token projection kernel."""
    return _create_grouped_decode_linear_kernel(dtype)


def _create_small_batch_grouped_linear_kernel(
    dtype: type, rows: int, outputs_per_group: int
):
    """Build a small-batch projection that reuses weights across rows."""
    DTYPE = dtype
    project = get_small_batch_grouped_projection(dtype, rows, outputs_per_group)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        x: wp.array2d(dtype=DTYPE),
        weight: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        inner: int,
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        wp.static(project)(x, weight, output, wp.tid(), inner)

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_small_batch_grouped_linear_kernel(
    dtype: type, rows: int, outputs_per_group: int = 8
):
    return _create_small_batch_grouped_linear_kernel(dtype, rows, outputs_per_group)


def _create_prefill_mma_linear_kernel(dtype: type, tile_m: int, tile_n: int):
    """Build an SM80+ dense projection wrapper for one tile geometry."""
    DTYPE = dtype
    project = get_prefill_mma_projection(dtype, tile_m, tile_n)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        x: wp.array2d(dtype=DTYPE),
        weight: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        columns: int,
        inner: int,
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        wp.static(project)(x, weight, output, wp.tid(), columns, inner)

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_prefill_mma_linear_kernel(dtype: type, tile_m: int, tile_n: int):
    """Return a cached SM80+ dense projection kernel."""
    return _create_prefill_mma_linear_kernel(dtype, tile_m, tile_n)


def _create_q8_grouped_decode_linear_kernel(dtype: type, outputs_per_group: int):
    """Build a signed-Q8 grouped DP4A decode wrapper."""
    DTYPE = dtype
    project = get_q8_grouped_decode_projection(dtype, outputs_per_group)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        activations: wp.array3d[wp.uint32],
        activation_scales: wp.array2d[wp.float32],
        weights: wp.array3d[wp.uint32],
        weight_scales: wp.array2d[wp.float16],
        output: wp.array2d(dtype=DTYPE),
        blocks: int,
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        wp.static(project)(
            activations,
            activation_scales,
            weights,
            weight_scales,
            output,
            wp.tid(),
            blocks,
        )

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_q8_grouped_decode_linear_kernel(dtype: type, outputs_per_group: int):
    """Return the cached signed-Q8 grouped decode kernel."""
    return _create_q8_grouped_decode_linear_kernel(dtype, outputs_per_group)


def _create_q8_prefill_mma_linear_kernel(dtype: type, tile_m: int):
    """Build a shared signed-INT8 tensor-core projection wrapper."""
    DTYPE = dtype
    project = get_q8_prefill_mma_projection(dtype, tile_m)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        activations: wp.array2d[wp.int8],
        activation_scales: wp.array2d[wp.float32],
        weights: wp.array3d[wp.int8],
        weight_scales: wp.array2d[wp.float16],
        output: wp.array2d(dtype=DTYPE),
        columns: int,
        blocks: int,
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        wp.static(project)(
            activations,
            activation_scales,
            weights,
            weight_scales,
            output,
            wp.tid(),
            columns,
            blocks,
        )

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_q8_prefill_mma_linear_kernel(dtype: type, tile_m: int):
    """Return a cached signed-INT8 tensor-core projection kernel."""
    return _create_q8_prefill_mma_linear_kernel(dtype, tile_m)


def _create_nvfp4_mma_linear_kernel(dtype: type, reuse_weights: bool, split_k: int):
    """Build the exact-SM120 native NVFP4 projection wrapper."""
    DTYPE = dtype
    project = get_nvfp4_mma_projection(dtype, reuse_weights, split_k)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        activations: wp.array2d[wp.uint8],
        activation_scales: wp.array2d[wp.uint8],
        activation_global_scales: wp.array1d[wp.float32],
        weights: wp.array2d[wp.uint8],
        weight_scales: wp.array2d[wp.uint8],
        output: wp.array2d(dtype=DTYPE),
        columns: int,
        blocks64: int,
        global_scale: float,
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - bind dtype in the Warp closure
        wp.static(project)(
            activations,
            activation_scales,
            activation_global_scales,
            weights,
            weight_scales,
            output,
            wp.tid(),
            columns,
            blocks64,
            global_scale,
        )

    return kernel


@lru_cache(maxsize=None)
def _get_nvfp4_mma_linear_kernel(
    dtype: type, reuse_weights: bool = False, split_k: int = 0
):
    return _create_nvfp4_mma_linear_kernel(dtype, reuse_weights, split_k)


def _create_linear_tiled_kernel(dtype: type, tile_m: int, tile_k: int):
    """Build a typed tensor-core-friendly dense projection kernel."""
    DTYPE = dtype
    TILE_M = tile_m
    TILE_N = 32
    TILE_K = tile_k

    @wp.func
    def cast_output(value: wp.float32):
        return DTYPE(value)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        x: wp.array2d(dtype=DTYPE),
        weight: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        """Compute a tiled ``output = x @ weight.T`` projection."""
        tile_row, tile_column = wp.tid()
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        accumulator = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
        for inner_tile in range((x.shape[1] + TILE_K - 1) / TILE_K):
            inner_offset = inner_tile * TILE_K
            activations = wp.tile_load(
                x, shape=(TILE_M, TILE_K), offset=(tile_row * TILE_M, inner_offset)
            )
            weights = wp.tile_load(
                weight,
                shape=(TILE_N, TILE_K),
                offset=(tile_column * TILE_N, inner_offset),
            )
            wp.tile_matmul(activations, wp.tile_transpose(weights), accumulator)
        wp.tile_store(
            output,
            wp.tile_map(cast_output, accumulator),
            offset=(tile_row * TILE_M, tile_column * TILE_N),
        )

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_linear_tiled_kernel(dtype: type, tile_m: int, tile_k: int):
    """Return a dense projection kernel for one reusable tile geometry."""
    return _create_linear_tiled_kernel(dtype, tile_m, tile_k), (tile_m, 32)


@wp.kernel
def _elu_kernel(
    x: wp.array2d[Any],
    y: wp.array2d[Any],
    alpha: float,
):
    """Apply ELU elementwise; ``alpha`` controls the negative branch."""
    i, j = wp.tid()
    v = x[i, j]
    y[i, j] = wp.where(
        v >= x.dtype(0.0), v, x.dtype(alpha) * (wp.exp(v) - x.dtype(1.0))
    )


@wp.kernel
def _unary_kernel(x: wp.array2d[Any], operation: int, y: wp.array2d[Any]):
    """Apply the unary operation selected by ``operation`` elementwise."""
    i, j = wp.tid()
    value = x[i, j]
    if operation == 0:
        y[i, j] = wp.max(value, x.dtype(0.0))
    elif operation == 1:
        y[i, j] = wp.tanh(value)
    elif operation == 2:
        y[i, j] = wp.sqrt(value)
    elif operation == 3:
        value_fp32 = wp.float32(value)
        y[i, j] = x.dtype(wp.float32(1.0) / (wp.float32(1.0) + wp.exp(-value_fp32)))
    else:
        value_fp32 = wp.float32(value)
        y[i, j] = x.dtype(
            wp.max(value_fp32, wp.float32(0.0))
            + wp.log(wp.float32(1.0) + wp.exp(-wp.abs(value_fp32)))
        )


@wp.kernel
def _binary_broadcast_kernel(
    lhs: wp.array2d[Any],
    rhs: wp.array2d[Any],
    operation: int,
    out: wp.array2d[Any],
):
    """Apply ``operation`` with modulo broadcasting over two 2-D inputs."""
    i, j = wp.tid()
    left = lhs[i % lhs.shape[0], j % lhs.shape[1]]
    right = rhs[i % rhs.shape[0], j % rhs.shape[1]]
    if operation == 0:
        out[i, j] = left + right
    elif operation == 1:
        out[i, j] = left - right
    elif operation == 2:
        out[i, j] = left * right
    else:
        out[i, j] = left / right


@wp.kernel
def _reduce_mean_rows_kernel(x: wp.array2d[Any], out: wp.array2d[Any]):
    """Reduce each matrix row to its mean."""
    row = wp.tid()
    total = x.dtype(0.0)
    for column in range(x.shape[1]):
        total += x[row, column]
    out[row, 0] = total / x.dtype(x.shape[1])


@lru_cache(maxsize=None)
def _get_masked_mean_pool_kernel(dtype):
    """Create a fixed-dtype masked token mean used by encoder models."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def masked_mean_pool(
        hidden: wp.array2d(dtype=DTYPE),
        mask: wp.array1d[wp.bool],
        output: wp.array2d(dtype=DTYPE),
    ):
        column = wp.tid()
        total = wp.float32(0.0)
        count = wp.float32(0.0)
        for token in range(hidden.shape[0]):
            if mask[token]:
                total += wp.float32(hidden[token, column])
                count += wp.float32(1.0)
        output[0, column] = DTYPE(total / wp.max(count, wp.float32(1.0)))

    return masked_mean_pool


@wp.kernel(enable_backward=False)
def _reduce_max_1d_kernel(x: wp.array1d[Any], out: wp.array1d[Any]):
    """Reduce a 1-D array to its maximum."""
    value = x[0]
    for index in range(1, x.shape[0]):
        value = wp.max(value, x[index])
    out[0] = value


@wp.kernel(enable_backward=False)
def _cast_kernel(x: wp.array1d[Any], out: wp.array1d[Any]):
    """Cast each element to the output array dtype."""
    index = wp.tid()
    out[index] = out.dtype(x[index])


@wp.kernel(enable_backward=False)
def _clamp_kernel(
    x: wp.array1d[Any],
    output: wp.array1d[Any],
    minimum: wp.float32,
    maximum: wp.float32,
):
    """Clamp a contiguous floating array with FP32 bounds."""
    index = wp.tid()
    output[index] = output.dtype(wp.clamp(wp.float32(x[index]), minimum, maximum))


@wp.kernel(enable_backward=False, module="unique")
def _overlap_tile_blend_kernel(
    tile: wp.array4d[Any],
    canvas: wp.array4d[Any],
    origin_y: int,
    origin_x: int,
    overlap_y: int,
    overlap_x: int,
    target_height: int,
    target_width: int,
):
    """Blend one NCHW tile into an in-place canvas with cropped bounds."""
    batch, channel, row, column = wp.tid()
    target_y = origin_y + row
    target_x = origin_x + column
    if target_y < target_height and target_x < target_width:
        value = wp.float32(tile[batch, channel, row, column])
        previous = wp.float32(canvas[batch, channel, target_y, target_x])
        if origin_y > 0 and row < overlap_y:
            alpha_y = wp.float32(row) / wp.float32(overlap_y)
            value = previous * (wp.float32(1.0) - alpha_y) + value * alpha_y
        if origin_x > 0 and column < overlap_x:
            alpha_x = wp.float32(column) / wp.float32(overlap_x)
            value = previous * (wp.float32(1.0) - alpha_x) + value * alpha_x
        canvas[batch, channel, target_y, target_x] = canvas.dtype(value)


@wp.kernel(enable_backward=False)
def _transpose_021_kernel(x: wp.array3d[Any], output: wp.array3d[Any]):
    """Transpose a rank-3 tensor from axes 0-1-2 to 0-2-1."""
    i, j, k = wp.tid()
    output[i, j, k] = x[i, k, j]


@wp.kernel(enable_backward=False)
def _transpose_0213_kernel(x: wp.array4d[Any], output: wp.array4d[Any]):
    """Transpose a rank-4 tensor from axes 0-1-2-3 to 0-2-1-3."""
    i, j, k, column = wp.tid()
    output[i, j, k, column] = x[i, k, j, column]


@wp.kernel(enable_backward=False, module="unique")
def _split_last_axis_kernel(
    x: wp.array2d[Any],
    output: wp.array2d[Any],
    input_offset: int,
):
    """Copy a last-axis slice starting at ``input_offset``."""
    row, column = wp.tid()
    output[row, column] = x[row, input_offset + column]


@wp.kernel(enable_backward=False)
def _tile_3d_kernel(x: wp.array3d[Any], output: wp.array3d[Any]):
    """Repeat a rank-3 tensor to fill ``output``."""
    i, j, k = wp.tid()
    output[i, j, k] = x[i % x.shape[0], j % x.shape[1], k % x.shape[2]]


@wp.kernel(enable_backward=False)
def _where_broadcast_kernel(
    condition: wp.array2d[wp.bool],
    x: wp.array2d[Any],
    y: wp.array2d[Any],
    output: wp.array2d[Any],
):
    """Select elementwise with modulo-broadcast conditions."""
    row, column = wp.tid()
    output[row, column] = wp.where(
        condition[row % condition.shape[0], column % condition.shape[1]],
        x[row, column],
        y[row, column],
    )


@wp.kernel(enable_backward=False, module="unique")
def _rotary_embedding_kernel(
    x: wp.array4d[Any],
    position_ids: wp.array2d[wp.int64],
    cos_cache: wp.array2d[Any],
    sin_cache: wp.array2d[Any],
    output: wp.array4d[Any],
    rotary_dim: int,
    interleaved: bool,
    position_offset: bool,
):
    """Apply rotary embedding over ``rotary_dim`` channels.
    ``interleaved`` selects adjacent rather than split-half pairs."""
    batch, head, sequence, column = wp.tid()
    if column >= rotary_dim:
        output[batch, head, sequence, column] = x[batch, head, sequence, column]
        return

    half = rotary_dim / 2
    if interleaved:
        cache_column = column / 2
        partner = column + 1 if column % 2 == 0 else column - 1
        sign = wp.float32(-1.0) if column % 2 == 0 else wp.float32(1.0)
    else:
        cache_column = column % half
        partner = column + half if column < half else column - half
        sign = wp.float32(-1.0) if column < half else wp.float32(1.0)
    position = (
        position_ids[0, 0] + wp.int64(sequence)
        if position_offset
        else position_ids[batch, sequence]
    )
    value = wp.float32(x[batch, head, sequence, column])
    rotated = sign * wp.float32(x[batch, head, sequence, partner])
    output[batch, head, sequence, column] = x.dtype(
        value * wp.float32(cos_cache[position, cache_column])
        + rotated * wp.float32(sin_cache[position, cache_column])
    )


@wp.kernel
def _batch_normalization_kernel(
    x: wp.array2d[Any],
    scale: wp.array1d[Any],
    bias: wp.array1d[Any],
    mean: wp.array1d[Any],
    variance: wp.array1d[Any],
    epsilon: float,
    relu: bool,
    y: wp.array2d[Any],
):
    """Apply inference batch normalization with optional ReLU fusion."""
    row, column = wp.tid()
    unit = (x[row, column] - mean[column]) / wp.sqrt(
        variance[column] + x.dtype(epsilon)
    )
    value = unit * scale[column] + bias[column]
    y[row, column] = wp.where(relu, wp.max(value, x.dtype(0.0)), value)


@wp.func
def _inverse_sqrt(value: Any):
    return value.dtype(1.0) / wp.sqrt(value)


def _create_rms_normalization_kernel(width: int):
    """Create a deterministic one-block-per-row RMS normalization kernel."""

    @wp.kernel
    def kernel(
        x: wp.array2d[Any],
        epsilon: wp.array1d[Any],
        scale: wp.array1d[Any],
        output: wp.array2d[Any],
    ):
        """Normalize each row using a fixed compile-time ``width``."""
        row = wp.tid()
        values = wp.tile_load(x, shape=(1, wp.static(width)), offset=(row, 0))
        sum_squares = wp.tile_sum(values * values)
        epsilon_tile = wp.tile_load(epsilon, shape=(1,), offset=(0,))
        inverse_rms = wp.tile_map(
            _inverse_sqrt, sum_squares / x.dtype(wp.static(width)) + epsilon_tile
        )
        inverse_rms = wp.tile_broadcast(inverse_rms, shape=(1, wp.static(width)))
        scales = wp.tile_broadcast(
            wp.tile_load(scale, shape=(wp.static(width),), offset=(0,)),
            shape=(1, wp.static(width)),
        )
        wp.tile_store(output, values * inverse_rms * scales, offset=(row, 0))

    return kernel


@wp.kernel
def _lstm_gates_kernel(
    x: wp.array2d[Any],  # (batch, input_size)
    h_prev: wp.array2d[Any],  # (batch, hidden_size)
    W: wp.array2d[Any],  # (4*hidden_size, input_size)
    R: wp.array2d[Any],  # (4*hidden_size, hidden_size)
    gates: wp.array2d[Any],  # (batch, 4*hidden_size) output
    input_size: int,
    hidden_size: int,
):
    """``gates = x @ W.T + h_prev @ R.T`` (one thread per (batch, gate))."""
    b, j = wp.tid()

    s = x.dtype(0.0)
    for k in range(input_size):
        s += x[b, k] * W[j, k]
    for k in range(hidden_size):
        s += h_prev[b, k] * R[j, k]

    gates[b, j] = s


@wp.kernel
def _lstm_cell_update_kernel(
    gates: wp.array2d[Any],  # (batch, 4*hidden_size); already x@W.T + h_prev@R.T
    c_prev: wp.array2d[Any],  # (batch, hidden_size)
    Bx: wp.array1d[Any],  # (4*hidden_size,)
    Bh: wp.array1d[Any],  # (4*hidden_size,)
    h_out: wp.array2d[Any],  # (batch, hidden_size)
    c_out: wp.array2d[Any],  # (batch, hidden_size)
    hidden_size: int,
):
    """Apply LSTM gates and update hidden and cell state."""
    b, h = wp.tid()

    s_i = (
        gates[b, 0 * hidden_size + h]
        + Bx[0 * hidden_size + h]
        + Bh[0 * hidden_size + h]
    )
    s_o = (
        gates[b, 1 * hidden_size + h]
        + Bx[1 * hidden_size + h]
        + Bh[1 * hidden_size + h]
    )
    s_f = (
        gates[b, 2 * hidden_size + h]
        + Bx[2 * hidden_size + h]
        + Bh[2 * hidden_size + h]
    )
    s_c = (
        gates[b, 3 * hidden_size + h]
        + Bx[3 * hidden_size + h]
        + Bh[3 * hidden_size + h]
    )

    one = gates.dtype(1.0)
    g_i = one / (one + wp.exp(-s_i))
    g_o = one / (one + wp.exp(-s_o))
    g_f = one / (one + wp.exp(-s_f))
    g_c = wp.tanh(s_c)

    c_new = g_f * c_prev[b, h] + g_i * g_c
    c_out[b, h] = c_new
    h_out[b, h] = g_o * wp.tanh(c_new)


@wp.kernel
def _gather_block_quantized_int8_kernel(
    data: wp.array2d[wp.uint8],
    indices: wp.array2d[wp.int64],
    scales: wp.array2d[wp.float16],
    zero_points: wp.array2d[wp.uint8],
    output: wp.array3d[wp.float16],
    block_size: int,
):
    """Gather and dequantize INT8 embedding rows by ``block_size``."""
    batch, sequence, column = wp.tid()
    row = indices[batch, sequence]
    block = column / block_size
    output[batch, sequence, column] = wp.float16(
        (wp.float32(data[row, column]) - wp.float32(zero_points[row, block]))
        * wp.float32(scales[row, block])
    )


@wp.kernel(enable_backward=False)
def _dequantize_e4m3_kernel(
    packed: wp.array1d[wp.uint8], scale: wp.array1d[Any], output: wp.array1d[Any]
):
    """Convert finite E4M3 values to ``output.dtype`` and apply one scale."""
    index = wp.tid()
    bits = wp.int32(packed[index])
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    magnitude = wp.float32(mantissa) * wp.float32(0.001953125)
    if exponent != 0:
        magnitude = (
            wp.float32(1.0) + wp.float32(mantissa) * wp.float32(0.125)
        ) * wp.pow(wp.float32(2.0), wp.float32(exponent - 7))
    if bits & 128:
        magnitude = -magnitude
    output[index] = output.dtype(magnitude * wp.float32(scale[0]))


def _create_gather_q8_0_rows_kernel(dtype: type):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        values: wp.array3d[wp.int8],
        indices: wp.array2d[wp.int64],
        scales: wp.array2d[wp.float16],
        output: wp.array3d(dtype=DTYPE),
    ):
        batch, sequence, column = wp.tid()
        row = indices[batch, sequence]
        block = column / 32
        output[batch, sequence, column] = DTYPE(
            wp.float32(values[row, block, column % 32]) * wp.float32(scales[row, block])
        )

    return kernel


@lru_cache(maxsize=None)
def _get_gather_q8_0_rows_kernel(dtype: type):
    return _create_gather_q8_0_rows_kernel(dtype)


@wp.kernel(module="unique")
def _gather_rows_kernel(
    data: wp.array2d[Any], indices: wp.array2d[wp.int64], output: wp.array3d[Any]
):
    """Gather matrix rows for batched token indices."""
    batch, sequence, column = wp.tid()
    output[batch, sequence, column] = data[indices[batch, sequence], column]


@wp.kernel(enable_backward=False, module="unique")
def _reorder_heads_kernel(x: wp.array2d[Any], output: wp.array2d[Any], head_size: int):
    """Reorder row-major packed heads into head-major rows."""
    row, head, column = wp.tid()
    output[head * x.shape[0] + row, column] = x[row, head * head_size + column]


@wp.kernel(enable_backward=False, module="unique")
def _split_attention_heads_kernel(x: wp.array3d[Any], output: wp.array4d[Any]):
    """Convert packed [batch, sequence, heads * width] to explicit heads."""
    batch, head, sequence, column = wp.tid()
    output[batch, head, sequence, column] = x[
        batch, sequence, head * output.shape[3] + column
    ]


@wp.kernel(enable_backward=False, module="unique")
def _merge_attention_heads_kernel(x: wp.array4d[Any], output: wp.array3d[Any]):
    """Convert explicit heads to packed [batch, sequence, heads * width]."""
    batch, head, sequence, column = wp.tid()
    output[batch, sequence, head * x.shape[3] + column] = x[
        batch, head, sequence, column
    ]


@wp.kernel(enable_backward=False, module="unique")
def _adaptive_rms_modulation_kernel(
    normalized: wp.array3d[Any],
    scale_shift_table: wp.array3d[Any],
    timestep_modulation: wp.array3d[Any],
    output: wp.array3d[Any],
    shift_index: int,
    scale_index: int,
):
    """Apply broadcast AdaLN shift/scale after a precomputed RMSNorm."""
    batch, sequence, column = wp.tid()
    shift = wp.float32(scale_shift_table[0, shift_index, column]) + wp.float32(
        timestep_modulation[batch, shift_index, column]
    )
    scale = wp.float32(scale_shift_table[0, scale_index, column]) + wp.float32(
        timestep_modulation[batch, scale_index, column]
    )
    output[batch, sequence, column] = normalized.dtype(
        wp.float32(normalized[batch, sequence, column]) * (wp.float32(1.0) + scale)
        + shift
    )


@wp.kernel(enable_backward=False, module="unique")
def _modulated_residual_kernel(
    residual: wp.array3d[Any],
    branch: wp.array3d[Any],
    scale_shift_table: wp.array3d[Any],
    timestep_modulation: wp.array3d[Any],
    output: wp.array3d[Any],
    gate_index: int,
    use_gate: bool,
):
    """Add a branch with an optional broadcast timestep/table gate."""
    batch, sequence, column = wp.tid()
    gate = wp.float32(1.0)
    if use_gate:
        gate = wp.float32(scale_shift_table[0, gate_index, column]) + wp.float32(
            timestep_modulation[batch, gate_index, column]
        )
    output[batch, sequence, column] = residual.dtype(
        wp.float32(residual[batch, sequence, column])
        + wp.float32(branch[batch, sequence, column]) * gate
    )


@wp.kernel(enable_backward=False, module="unique")
def _bias_activation_kernel(
    x: wp.array2d[Any], bias: wp.array1d[Any], output: wp.array2d[Any], activation: int
):
    """Add a vector bias and optionally apply SiLU or tanh-approximate GELU."""
    row, column = wp.tid()
    value = wp.float32(x[row, column]) + wp.float32(bias[column])
    if activation == 1:
        value = value / (wp.float32(1.0) + wp.exp(-value))
    elif activation == 2:
        cubic = value * value * value
        value = (
            wp.float32(0.5)
            * value
            * (
                wp.float32(1.0)
                + wp.tanh(
                    wp.float32(0.7978845608028654)
                    * (value + wp.float32(0.044715) * cubic)
                )
            )
        )
    output[row, column] = x.dtype(value)


@wp.kernel(enable_backward=False, module="unique")
def _adaptive_layer_norm_kernel(
    normalized: wp.array3d[Any],
    modulation: wp.array3d[Any],
    output: wp.array3d[Any],
    shift_index: int,
    scale_index: int,
):
    """Apply broadcast adaptive shift/scale after affine-free LayerNorm."""
    batch, sequence, column = wp.tid()
    shift = wp.float32(modulation[batch, shift_index, column])
    scale = wp.float32(modulation[batch, scale_index, column])
    output[batch, sequence, column] = normalized.dtype(
        wp.float32(normalized[batch, sequence, column]) * (wp.float32(1.0) + scale)
        + shift
    )


@wp.kernel(enable_backward=False, module="unique")
def _broadcast_gated_residual_kernel(
    residual: wp.array3d[Any],
    branch: wp.array3d[Any],
    modulation: wp.array3d[Any],
    output: wp.array3d[Any],
    gate_index: int,
):
    """Add a branch multiplied by one batch-broadcast modulation vector."""
    batch, sequence, column = wp.tid()
    output[batch, sequence, column] = residual.dtype(
        wp.float32(residual[batch, sequence, column])
        + wp.float32(branch[batch, sequence, column])
        * wp.float32(modulation[batch, gate_index, column])
    )


@wp.kernel(enable_backward=False, module="unique")
def _sinusoidal_embedding_kernel(
    values: wp.array1d[wp.float32],
    output: wp.array2d[Any],
    maximum_period: wp.float32,
    scale: wp.float32,
    frequency_shift: wp.float32,
    flip_sin_cos: bool,
):
    """Create a graph-safe diffusion-style sinusoidal embedding."""
    batch, column = wp.tid()
    half = output.shape[1] / 2
    if column >= half * 2:
        output[batch, column] = output.dtype(0.0)
    else:
        frequency_column = column % half
        frequency = wp.exp(
            -wp.log(maximum_period)
            * wp.float32(frequency_column)
            / (wp.float32(half) - frequency_shift)
        )
        angle = values[batch] * scale * frequency
        use_cos = (column >= half) != flip_sin_cos
        output[batch, column] = output.dtype(
            wp.cos(angle) if use_cos else wp.sin(angle)
        )


@wp.kernel(enable_backward=False, module="unique")
def _rotary_cache_kernel(
    x: wp.array4d[Any],
    cosine: wp.array2d[Any],
    sine: wp.array2d[Any],
    output: wp.array4d[Any],
):
    """Apply adjacent-pair RoPE from one precomputed cache row per token."""
    batch, head, sequence, column = wp.tid()
    pair = column / 2
    partner = column + 1 if column % 2 == 0 else column - 1
    sign = wp.float32(-1.0) if column % 2 == 0 else wp.float32(1.0)
    output[batch, head, sequence, column] = x.dtype(
        wp.float32(x[batch, head, sequence, column])
        * wp.float32(cosine[sequence, pair])
        + sign
        * wp.float32(x[batch, head, sequence, partner])
        * wp.float32(sine[sequence, pair])
    )


@wp.kernel(enable_backward=False, module="unique")
def _concatenate_attention_streams_kernel(
    first: wp.array4d[Any], second: wp.array4d[Any], output: wp.array4d[Any]
):
    """Concatenate two [B,H,S,D] streams along their sequence axis."""
    batch, head, sequence, column = wp.tid()
    if sequence < first.shape[2]:
        output[batch, head, sequence, column] = first[batch, head, sequence, column]
    else:
        output[batch, head, sequence, column] = second[
            batch, head, sequence - first.shape[2], column
        ]


@wp.kernel(enable_backward=False, module="unique")
def _concatenate_validity_kernel(
    first: wp.array2d[wp.bool], second: wp.array2d[wp.bool], output: wp.array2d[wp.bool]
):
    """Concatenate two batch/sequence validity masks."""
    batch, sequence = wp.tid()
    if sequence < first.shape[1]:
        output[batch, sequence] = first[batch, sequence]
    else:
        output[batch, sequence] = second[batch, sequence - first.shape[1]]


@wp.kernel(enable_backward=False, module="unique")
def _split_attention_streams_kernel(
    joint: wp.array4d[Any], first: wp.array4d[Any], second: wp.array4d[Any]
):
    """Split a joint [B,H,S,D] result into two fixed sequence streams."""
    batch, head, sequence, column = wp.tid()
    if sequence < first.shape[2]:
        first[batch, head, sequence, column] = joint[batch, head, sequence, column]
    else:
        second[batch, head, sequence - first.shape[2], column] = joint[
            batch, head, sequence, column
        ]


@wp.kernel(enable_backward=False, module="unique")
def _sequence_slice_kernel(
    x: wp.array3d[Any], output: wp.array3d[Any], source_offset: int
):
    """Copy one contiguous fixed-length slice along a rank-three sequence axis."""
    batch, sequence, column = wp.tid()
    output[batch, sequence, column] = x[batch, sequence + source_offset, column]


@wp.kernel(enable_backward=False, module="unique")
def _reorder_interleaved_heads_kernel(
    x: wp.array2d[Any], output: wp.array2d[Any], head_size: int
):
    """Reorder GGUF interleaved-RoPE heads into split-half head-major rows."""
    row, head, column = wp.tid()
    half = head_size / 2
    source_column = (column % half) * 2 + column / half
    output[head * x.shape[0] + row, column] = x[row, head * head_size + source_column]


@wp.kernel(enable_backward=False, module="unique")
def _unpack_gated_heads_kernel(
    x: wp.array2d[Any],
    values: wp.array2d[Any],
    gate: wp.array2d[Any],
    head_size: int,
    interleaved: bool,
):
    """Split per-head value/gate pairs and reorder values head-major."""
    row, head, column = wp.tid()
    offset = head * head_size * 2
    half = head_size / 2
    source_column = (column % half) * 2 + column / half if interleaved else column
    values[head * x.shape[0] + row, column] = x[row, offset + source_column]
    gate[row, head * head_size + column] = x[row, offset + head_size + column]


@wp.kernel(enable_backward=False, module="unique")
def _unpack_gated_heads_decode_batch_kernel(
    x: wp.array2d[Any],
    values: wp.array4d[Any],
    gate: wp.array2d[Any],
    head_size: int,
    interleaved: bool,
):
    """Split gated heads into batch-major one-token storage."""
    batch, head, column = wp.tid()
    offset = head * head_size * 2
    half = head_size / 2
    source = column
    if interleaved:
        source = (column % half) * 2 + column / half
    values[batch, head, 0, column] = x[batch, offset + source]
    gate[batch, head * head_size + column] = x[batch, offset + head_size + column]


@wp.kernel(enable_backward=False, module="unique")
def _reorder_heads_decode_batch_kernel(
    x: wp.array2d[Any], output: wp.array4d[Any], head_size: int, interleaved: bool
):
    """Convert packed rows to batch-major one-token heads."""
    batch, head, column = wp.tid()
    half = head_size / 2
    source = column
    if interleaved:
        source = (column % half) * 2 + column / half
    output[batch, head, 0, column] = x[batch, head * head_size + source]


@lru_cache(maxsize=None)
def _get_append_head_cache_decode_batch_kernel(
    mapped: bool = False, circular: bool = False
):
    """Build a cache append with static physical-slot and ring-cache policy."""
    MAPPED = mapped
    CIRCULAR = circular

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        x: wp.array4d[Any],
        positions: wp.array1d[wp.int32],
        active: wp.array1d[wp.bool],
        slot_indices: wp.array1d[wp.int32],
        cache: wp.array2d[Any],
        capacity: int,
    ):
        batch, head, column = wp.tid()
        if active[batch]:
            slot = slot_indices[batch] if wp.static(MAPPED) else batch
            position = (
                positions[batch] % capacity if wp.static(CIRCULAR) else positions[batch]
            )
            row = (slot * x.shape[1] + head) * capacity + position
            cache[row, column] = x[batch, head, 0, column]

    return kernel


@wp.func
def _float_from_fp16_bytes(low: wp.uint8, high: wp.uint8) -> wp.float32:
    """Decode one little-endian IEEE FP16 value without an aligned typed view."""
    bits = wp.int32(low) | (wp.int32(high) << 8)
    sign = wp.float32(1.0)
    if (bits & 0x8000) != 0:
        sign = wp.float32(-1.0)
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    if exponent == 0:
        return sign * wp.float32(fraction) * wp.float32(5.960464477539063e-8)
    return (
        sign
        * (wp.float32(1.0) + wp.float32(fraction) * wp.float32(0.0009765625))
        * wp.pow(wp.float32(2.0), wp.float32(exponent - 15))
    )


@lru_cache(maxsize=None)
def _get_gather_q2_k_rows_kernel(dtype: type):
    """Gather GGML Q2_K rows while dequantizing only requested elements."""
    if dtype not in (wp.float16, wp.bfloat16, wp.float32):
        raise TypeError("Q2_K gather output must be FP16, BF16, or FP32")

    @wp.kernel(enable_backward=False)
    def kernel(
        blocks: wp.array3d(dtype=wp.uint8),
        indices: wp.array2d(dtype=wp.int64),
        output: wp.array3d(dtype=dtype),
    ):
        batch, row, column = wp.tid()
        source_row = indices[batch, row]
        block = column >> 8
        local = column & 255
        scale_index = local >> 4
        quant_group = (local & 127) >> 4
        quant_index = 16 + ((local >> 7) << 5) + ((quant_group & 1) << 4) + (local & 15)
        shift = (quant_group >> 1) << 1
        scale = wp.int32(blocks[source_row, block, scale_index])
        quant = (wp.int32(blocks[source_row, block, quant_index]) >> shift) & 3
        d = _float_from_fp16_bytes(
            blocks[source_row, block, 80], blocks[source_row, block, 81]
        )
        minimum = _float_from_fp16_bytes(
            blocks[source_row, block, 82], blocks[source_row, block, 83]
        )
        output[batch, row, column] = dtype(
            d * wp.float32(scale & 15) * wp.float32(quant)
            - minimum * wp.float32(scale >> 4)
        )

    return kernel


@wp.func
def _q3_k_scale(
    blocks: wp.array3d(dtype=wp.uint8), row: int, block: int, index: int
) -> int:
    """Unpack one signed six-bit Q3_K group scale."""
    group = index >> 2
    lane = index & 3
    lower = wp.int32(0)
    if group == 0:
        lower = wp.int32(blocks[row, block, 96 + lane]) & 15
    elif group == 1:
        lower = wp.int32(blocks[row, block, 100 + lane]) & 15
    elif group == 2:
        lower = (wp.int32(blocks[row, block, 96 + lane]) >> 4) & 15
    else:
        lower = (wp.int32(blocks[row, block, 100 + lane]) >> 4) & 15
    upper = (wp.int32(blocks[row, block, 104 + lane]) >> (group << 1)) & 3
    return (lower | (upper << 4)) - 32


@lru_cache(maxsize=None)
def _get_q3_k_linear_kernel(dtype: type):
    """Return a fused GGML Q3_K matrix-vector projection."""
    if dtype not in (wp.float16, wp.bfloat16, wp.float32):
        raise TypeError("Q3_K Linear output must be FP16, BF16, or FP32")

    @wp.kernel(enable_backward=False)
    def kernel(
        x: wp.array2d(dtype=dtype),
        blocks: wp.array3d(dtype=wp.uint8),
        output: wp.array2d(dtype=dtype),
        inner_blocks: int,
    ):
        thread = wp.tid()
        lane = thread & 31
        item = thread >> 5
        column = item % blocks.shape[0]
        row = item // blocks.shape[0]
        total = wp.float32(0.0)
        for block in range(inner_blocks):
            d = _float_from_fp16_bytes(
                blocks[column, block, 108], blocks[column, block, 109]
            )
            for part in range(8):
                local = lane + part * 32
                group = local >> 4
                component = local & 15
                group_scale = d * wp.float32(_q3_k_scale(blocks, column, block, group))
                half = group >> 3
                local_group = group & 7
                shift = (local_group >> 1) << 1
                mask = 1 << (half * 4 + (local_group >> 1))
                quant_index = 32 + half * 32 + (local_group & 1) * 16 + component
                quant = (wp.int32(blocks[column, block, quant_index]) >> shift) & 3
                high = (
                    wp.int32(
                        blocks[
                            column,
                            block,
                            component + (local_group & 1) * 16,
                        ]
                    )
                    & mask
                )
                value = quant
                if high == 0:
                    value -= 4
                total += (
                    wp.float32(x[row, block * 256 + local])
                    * group_scale
                    * wp.float32(value)
                )
        total = subgroup_sum(total, 32)
        if lane == 0:
            output[row, column] = dtype(total)

    return kernel


@wp.kernel(enable_backward=False, module="unique")
def _append_head_cache_kernel(
    x: wp.array2d[Any],
    positions: wp.array2d[wp.int64],
    cache: wp.array2d[Any],
    heads: int,
    head_size: int,
):
    """Append head-major token rows at their device-side positions."""
    head, row, column = wp.tid()
    capacity = cache.shape[0] / heads
    cache[head * capacity + wp.int32(positions[0, row]), column] = x[
        head * positions.shape[1] + row, column
    ]


@wp.kernel(enable_backward=False, module="unique")
def _append_circular_head_cache_kernel(
    x: wp.array2d[Any],
    positions: wp.array2d[wp.int64],
    cache: wp.array2d[Any],
    heads: int,
    head_size: int,
):
    """Append head-major rows to a circular device-side cache."""
    head, row, column = wp.tid()
    capacity = cache.shape[0] / heads
    position = wp.int32(positions[0, row] % wp.int64(capacity))
    cache[head * capacity + position, column] = x[
        head * positions.shape[1] + row, column
    ]


@wp.kernel(enable_backward=False, module="unique")
def _sigmoid_gate_kernel(
    x: wp.array2d[Any], gate: wp.array2d[Any], output: wp.array2d[Any]
):
    """Multiply activations by a sigmoid gate."""
    row, column = wp.tid()
    gate_value = wp.float32(gate[row, column])
    output[row, column] = x.dtype(
        wp.float32(x[row, column]) / (wp.float32(1.0) + wp.exp(-gate_value))
    )


@wp.kernel(enable_backward=False, module="unique")
def _scale_kernel(x: wp.array2d[Any], output: wp.array2d[Any], scale: float):
    """Multiply an array by one scalar, allowing in-place output."""
    row, column = wp.tid()
    output[row, column] = x.dtype(wp.float32(x[row, column]) * wp.float32(scale))


@wp.kernel(enable_backward=False, module="unique")
def _logit_softcap_kernel(
    x: wp.array3d[Any], output: wp.array3d[Any], multiplier: float, cap: float
):
    """Apply ``cap * tanh(x * multiplier / cap)`` to logits."""
    batch, row, column = wp.tid()
    value = wp.float32(x[batch, row, column]) * wp.float32(multiplier) / wp.float32(cap)
    output[batch, row, column] = x.dtype(wp.float32(cap) * wp.tanh(value))


@wp.kernel(enable_backward=False)
def _gather_single_index_kernel(
    data: wp.array1d[Any],
    output: wp.array1d[Any],
    index: int,
    axis_size: int,
    stride: int,
):
    """Gather one index along a flattened axis with the given ``stride``."""
    output_index = wp.tid()
    prefix = output_index / stride
    suffix = output_index % stride
    output[output_index] = data[(prefix * axis_size + index) * stride + suffix]


def _create_quantize_int8_kernel(dtype: type, scale_dtype: type, zero_scale_one: bool):
    """Build symmetric block-32 INT8 quantization for input and scale dtypes."""
    DTYPE = dtype
    SCALE_DTYPE = scale_dtype
    ZERO_SCALE_ONE = zero_scale_one

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        activations: wp.array2d(dtype=DTYPE),
        quantized: wp.array2d[wp.int8],
        scales: wp.array2d(dtype=SCALE_DTYPE),
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - binds the factory dtype for Warp
        thread = wp.tid()
        lane = thread % 32
        block = (thread / 32) % scales.shape[1]
        row = (thread / 32) / scales.shape[1]
        column = block * 32 + lane
        value = wp.float32(activations[row, column])
        maximum = warp_max_broadcast(wp.abs(value))
        scale = maximum / 127.0
        quantization_scale = scale if maximum > 0.0 else 1.0
        quantized[row, column] = wp.int8(
            wp.clamp(wp.round(value / quantization_scale), -127.0, 127.0)
        )
        if lane == 0:
            scales[row, block] = SCALE_DTYPE(
                quantization_scale if ZERO_SCALE_ONE else scale
            )

    return kernel


@lru_cache(maxsize=None)
def _get_quantize_int8_kernel(dtype: type, scale_dtype: type, zero_scale_one: bool):
    return _create_quantize_int8_kernel(dtype, scale_dtype, zero_scale_one)


def _get_quantize_activation_int8_kernel(dtype: type):
    return _get_quantize_int8_kernel(dtype, wp.float32, True)


_quantize_activation_int8_kernel = _get_quantize_activation_int8_kernel(wp.float16)


def _create_quantize_nvfp4_kernel(dtype: type):
    """Build dynamic block-16 E2M1 quantization with E4M3 scales."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        values: wp.array2d(dtype=DTYPE),
        packed: wp.array2d[wp.uint8],
        scales: wp.array2d[wp.uint8],
        global_scales: wp.array1d[wp.float32],
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - bind dtype in the Warp closure
        thread = wp.tid()
        lane = thread % 16
        block = (thread / 16) % scales.shape[1]
        row = (thread / 16) / scales.shape[1]
        global_scale = global_scales[row]
        value = wp.float32(values[row, block * 16 + lane])
        value = value / global_scale if global_scale > 0.0 else 0.0
        maximum = subgroup_max_broadcast(wp.abs(value), 16)
        scale_code = encode_ue4m3(maximum / 6.0)
        scale = decode_ue4m3(scale_code)
        inverse_scale = 1.0 / scale if scale > 0.0 else 0.0
        packed_code = quantize_e2m1_pair(value, inverse_scale)
        if lane % 2 == 0:
            packed[row, block * 8 + lane / 2] = wp.uint8(packed_code)
        if lane == 0:
            scales[row, block] = wp.uint8(scale_code)

    return kernel


@lru_cache(maxsize=None)
def _get_quantize_nvfp4_kernel(dtype: type):
    """Return block-16 dynamic NVFP4 quantization for a floating input."""
    if dtype not in (wp.float16, wp.bfloat16, wp.float32):
        raise TypeError("NVFP4 quantization requires FP16, BF16, or FP32 input")
    return _create_quantize_nvfp4_kernel(dtype)


def _create_nvfp4_row_scale_kernel(dtype: type):
    """Build a deterministic tiled row-maximum reduction for NVFP4."""
    DTYPE = dtype
    TILE_WIDTH = 256

    @wp.func
    def absolute(value: DTYPE):
        return wp.abs(wp.float32(DTYPE(value)))

    @wp.func
    def maximum(left: wp.float32, right: wp.float32):
        return wp.max(left, right)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        values: wp.array2d(dtype=DTYPE),
        global_scales: wp.array1d[wp.float32],
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - bind dtype in the Warp closure
        row = wp.tid()
        maxima = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for tile in range((values.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            loaded = wp.tile_load(
                values[row], shape=(TILE_WIDTH,), offset=(tile * TILE_WIDTH,)
            )
            maxima = wp.tile_map(maximum, maxima, wp.tile_map(absolute, loaded))
        row_maximum = wp.tile_extract(wp.tile_reduce(maximum, maxima), 0)
        global_scales[row] = row_maximum / wp.float32(2688.0)

    return kernel


@lru_cache(maxsize=None)
def _get_nvfp4_row_scale_kernel(dtype: type):
    if dtype not in (wp.float16, wp.bfloat16, wp.float32):
        raise TypeError("NVFP4 row scaling requires FP16, BF16, or FP32 input")
    return _create_nvfp4_row_scale_kernel(dtype)


@wp.kernel(enable_backward=False)
def _repack_gguf_nvfp4_kernel(
    source: wp.array3d[wp.uint8], output: wp.array3d[wp.uint8]
):
    """Convert GGUF's split-half nibbles to adjacent E2M1 pairs."""
    row, block, output_byte = wp.tid()
    first = output_byte * 2
    subblock = first / 16
    local = first % 16
    shift = 0 if local < 8 else 4
    source_byte = subblock * 8 + local % 8
    low = (wp.int32(source[row, block, source_byte]) >> shift) & 15
    high = (wp.int32(source[row, block, source_byte + 1]) >> shift) & 15
    output[row, block, output_byte] = wp.uint8(low | (high << 4))


@wp.kernel(enable_backward=False, module="unique", grid_stride=False)
def _matmul_int4_q8_kernel(
    activations: wp.array3d[wp.uint32],
    activation_scales: wp.array2d[wp.float32],
    weights: wp.array3d[wp.uint32],
    weight_scales: wp.array2d[wp.float16],
    output: wp.array2d[wp.float16],
):
    """Multiply Q8 activations by packed, block-scaled INT4 weights."""
    thread = wp.tid()
    lane = thread % 4
    item = thread / 4
    row = item / weights.shape[0]
    column = item % weights.shape[0]
    total = wp.float32(0.0)
    for block in range(weights.shape[1]):
        packed_weights = weights[column, block, lane]
        packed_activation_0 = wp.int32(activations[row, block, lane * 2])
        packed_activation_1 = wp.int32(activations[row, block, lane * 2 + 1])
        block_total = dp4a(
            expand_int4x4_low(wp.int32(packed_weights)), packed_activation_0, 0
        )
        block_total = dp4a(
            expand_int4x4_high(wp.int32(packed_weights)),
            packed_activation_1,
            block_total,
        )
        activation_sum = dp4a(0x01010101, packed_activation_0, 0)
        activation_sum = dp4a(0x01010101, packed_activation_1, activation_sum)
        block_total -= 8 * activation_sum
        total += (
            wp.float32(block_total)
            * activation_scales[row, block]
            * wp.float32(weight_scales[column, block])
        )
    total = subgroup_sum(total, 4)
    if lane == 0:
        output[row, column] = wp.float16(total)


@lru_cache(maxsize=None)
def _get_matmul_int8_q8_kernel(
    reduction_width: int,
    dtype: type = wp.float16,
    signed_weights: bool = False,
    outputs_per_group: int = 1,
):
    """Build a block-scaled INT8 matrix-vector kernel."""
    if outputs_per_group not in (1, 2):
        raise ValueError("Q8 output grouping must be 1 or 2")
    REDUCTION_WIDTH = reduction_width
    WORDS_PER_LANE = 8 // reduction_width
    DTYPE = dtype
    SIGNED_WEIGHTS = signed_weights
    OUTPUTS_PER_GROUP = outputs_per_group

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        activations: wp.array3d[wp.uint32],
        activation_scales: wp.array2d[wp.float32],
        weights: wp.array3d[wp.uint32],
        weight_scales: wp.array2d[wp.float16],
        output: wp.array2d(dtype=DTYPE),
    ):
        output_zero = DTYPE(0.0)  # noqa: F841 - binds the factory dtype for Warp
        thread = wp.tid()
        lane = thread % REDUCTION_WIDTH
        item = thread / REDUCTION_WIDTH
        groups = (weights.shape[0] + OUTPUTS_PER_GROUP - 1) / OUTPUTS_PER_GROUP
        row = item / groups
        column_0 = (item % groups) * OUTPUTS_PER_GROUP
        totals = wp.vec2f(0.0, 0.0)
        for block in range(weights.shape[1]):
            activation_scale = activation_scales[row, block]
            for group in range(WORDS_PER_LANE):
                word = lane + group * REDUCTION_WIDTH
                packed_activation = wp.int32(activations[row, block, word])
                for output_item in range(OUTPUTS_PER_GROUP):
                    column = column_0 + output_item
                    if column < weights.shape[0]:
                        if wp.static(SIGNED_WEIGHTS):
                            packed_weights = wp.int32(weights[column, block, word])
                        else:
                            packed_weights = wp.int32(
                                weights[column, block, word] ^ wp.uint32(0x80808080)
                            )
                        block_total = dp4a(packed_weights, packed_activation, 0)
                        totals[output_item] += (
                            wp.float32(block_total)
                            * activation_scale
                            * wp.float32(weight_scales[column, block])
                        )
        for output_item in range(OUTPUTS_PER_GROUP):
            column = column_0 + output_item
            total = subgroup_sum(totals[output_item], REDUCTION_WIDTH)
            if lane == 0 and column < weights.shape[0]:
                output[row, column] = DTYPE(total)

    return kernel


@lru_cache(maxsize=None)
def _get_matmul_int4_tile_gemm_kernel(tile_m: int, tile_n: int, blocks_per_tile: int):
    """Build tiled INT4 GEMM for ``tile_m`` by ``tile_n`` output tiles.
    ``blocks_per_tile`` controls the K span loaded per iteration."""
    packed_width = 16 * blocks_per_tile
    activation_width = 32 * blocks_per_tile

    @wp.func
    def dequantize_low(value: wp.uint8, scale: wp.float16):
        return wp.float16((wp.float32(wp.int32(value) & 15) - 8.0) * wp.float32(scale))

    @wp.func
    def dequantize_high(value: wp.uint8, scale: wp.float16):
        return wp.float16((wp.float32(wp.int32(value) >> 4) - 8.0) * wp.float32(scale))

    @wp.func
    def to_float16(value: wp.float32):
        return wp.float16(value)

    @wp.func
    def add_one(value: wp.int32):
        return value + 1

    @wp.func
    def scale_index(value: wp.int32):
        return value / 16

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        activations: wp.array2d[wp.float16],
        weights: wp.array2d[wp.uint8],
        scales: wp.array2d[wp.float16],
        output: wp.array2d[wp.float16],
    ):
        """Dequantize packed INT4 tiles and multiply them with FP16 activations."""
        row_tile, column_tile = wp.tid()
        row = row_tile * tile_m
        column = column_tile * tile_n
        total = wp.tile_zeros(shape=(tile_m, tile_n), dtype=wp.float32)
        even_columns = wp.tile_arange(0, packed_width, dtype=wp.int32) * 2
        odd_columns = wp.tile_map(add_one, even_columns)
        scale_indices = wp.tile_map(
            scale_index, wp.tile_arange(0, packed_width, dtype=wp.int32)
        )
        for block_tile in range(
            (scales.shape[1] + blocks_per_tile - 1) / blocks_per_tile
        ):
            activation = wp.tile_load(
                activations,
                shape=(tile_m, activation_width),
                offset=(row, block_tile * activation_width),
            )
            packed = wp.tile_load(
                weights,
                shape=(tile_n, packed_width),
                offset=(column, block_tile * packed_width),
            )
            block_scales = wp.tile_load(
                scales,
                shape=(tile_n, blocks_per_tile),
                offset=(column, block_tile * blocks_per_tile),
            )
            scale = block_scales[:, scale_indices]
            low = wp.tile_transpose(wp.tile_map(dequantize_low, packed, scale))
            high = wp.tile_transpose(wp.tile_map(dequantize_high, packed, scale))
            wp.tile_matmul(activation[:, even_columns], low, total)
            wp.tile_matmul(activation[:, odd_columns], high, total)
        wp.tile_store(output, wp.tile_map(to_float16, total), offset=(row, column))

    return kernel


def _nbits_reduction_width(
    bits: int, packed_block_size: int, warp_reduction: bool
) -> int:
    if not warp_reduction:
        return 1
    width = 1 << (packed_block_size - 1).bit_length()
    return min(32, width // 2 if bits == 8 and width > 1 else width)


def _create_matmul_nbits_kernel(
    bits: int, block_size: int, dtype: type, warp_reduction: bool
):
    """Build generic packed N-bit matmul for the given quantization block.
    ``warp_reduction`` enables a subgroup reduction on CUDA."""
    values_per_byte = 8 // bits
    packed_block_size = block_size // values_per_byte
    reduction_width = _nbits_reduction_width(bits, packed_block_size, warp_reduction)
    load_stride = reduction_width
    loads_per_lane = (packed_block_size + load_stride - 1) // load_stride

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        activations: wp.array2d(dtype=dtype),
        weights: wp.array3d[wp.uint8],
        scales: wp.array2d(dtype=dtype),
        zero_points: wp.array2d[wp.uint8],
        output: wp.array2d(dtype=dtype),
        has_zero_points: bool,
    ):
        """Multiply activations by packed N-bit weights with optional zero points."""
        thread = wp.tid()
        lane = thread % reduction_width if warp_reduction else 0
        item = thread / reduction_width if warp_reduction else thread
        row = item / weights.shape[0]
        column = item % weights.shape[0]
        total = wp.float32(0.0)

        for block in range(weights.shape[1]):
            zero = 1 << (bits - 1)
            if has_zero_points:
                packed_zero = wp.int32(zero_points[column, block / values_per_byte])
                zero = (packed_zero >> ((block % values_per_byte) * bits)) & (
                    (1 << bits) - 1
                )
            block_total = wp.float32(0.0)
            for group in range(loads_per_lane):
                packed_offset = lane + group * load_stride
                if packed_offset < packed_block_size:
                    packed = wp.int32(weights[column, block, packed_offset])
                    activation_offset = (
                        block * block_size + packed_offset * values_per_byte
                    )
                    for value_index in range(values_per_byte):
                        quantized = (packed >> (value_index * bits)) & ((1 << bits) - 1)
                        block_total += wp.float32(
                            activations[row, activation_offset + value_index]
                        ) * wp.float32(quantized - zero)
            total += block_total * wp.float32(scales[column, block])

        if warp_reduction:
            total = subgroup_sum(total, reduction_width)
        if lane == 0:
            output[row, column] = dtype(total)

    return kernel


@lru_cache(maxsize=None)
def _get_matmul_nbits_kernel(
    bits: int, block_size: int, dtype: type, warp_reduction: bool
):
    """Return the reduction width and cached generic N-bit matmul kernel."""
    packed_block_size = block_size * bits // 8
    reduction_width = _nbits_reduction_width(bits, packed_block_size, warp_reduction)
    return reduction_width, _create_matmul_nbits_kernel(
        bits, block_size, dtype, warp_reduction
    )


def _create_dequantize_nbits_kernel(bits: int, block_size: int, dtype: type):
    """Build a kernel that expands packed N-bit blocks to ``dtype``."""
    values_per_byte = 8 // bits
    packed_block_size = block_size // values_per_byte

    @wp.kernel(enable_backward=False)
    def kernel(
        weights: wp.array3d[wp.uint8],
        scales: wp.array2d(dtype=dtype),
        zero_points: wp.array2d[wp.uint8],
        output: wp.array2d(dtype=dtype),
        has_zero_points: bool,
    ):
        """Dequantize packed weights with optional packed zero points."""
        column, packed_index = wp.tid()
        block = packed_index / packed_block_size
        packed_offset = packed_index - block * packed_block_size
        packed = wp.int32(weights[column, block, packed_offset])
        zero = 1 << (bits - 1)
        if has_zero_points:
            packed_zero = wp.int32(zero_points[column, block / values_per_byte])
            zero = (packed_zero >> ((block % values_per_byte) * bits)) & (
                (1 << bits) - 1
            )
        scale = wp.float32(scales[column, block])
        output_offset = block * block_size + packed_offset * values_per_byte
        for value_index in range(values_per_byte):
            quantized = (packed >> (value_index * bits)) & ((1 << bits) - 1)
            output[column, output_offset + value_index] = dtype(
                wp.float32(quantized - zero) * scale
            )

    return kernel


@wp.kernel(enable_backward=False)
def _causal_conv_1d_kernel(
    x: wp.array3d[Any],
    weight: wp.array3d[Any],
    bias: wp.array1d[Any],
    past: wp.array3d[Any],
    output: wp.array3d[Any],
    kernel_size: int,
    has_bias: bool,
    silu: bool,
):
    """Apply depthwise causal 1-D convolution using ``past`` for left context.
    ``silu`` optionally fuses the activation."""
    batch, channel, position = wp.tid()
    value = wp.float32(0.0)
    if has_bias:
        value = wp.float32(bias[channel])
    for kernel_index in range(kernel_size):
        input_index = position + kernel_index - (kernel_size - 1)
        input_value = wp.float32(0.0)
        if input_index < 0:
            input_value = wp.float32(
                past[batch, channel, input_index + kernel_size - 1]
            )
        else:
            input_value = wp.float32(x[batch, channel, input_index])
        value += input_value * wp.float32(weight[channel, 0, kernel_index])
    if silu:
        value = value / (wp.float32(1.0) + wp.exp(-value))
    output[batch, channel, position] = x.dtype(value)


@wp.kernel(enable_backward=False)
def _causal_conv_state_kernel(
    x: wp.array3d[Any],
    past: wp.array3d[Any],
    present: wp.array3d[Any],
):
    """Build the next causal-convolution state from input and prior state."""
    batch, channel, state_index = wp.tid()
    sequence_length = x.shape[2]
    source_index = sequence_length + state_index
    if source_index < past.shape[2]:
        present[batch, channel, state_index] = past[batch, channel, source_index]
    else:
        present[batch, channel, state_index] = x[
            batch, channel, source_index - past.shape[2]
        ]


@wp.kernel(enable_backward=False)
def _causal_conv_state_inplace_kernel(x: wp.array3d[Any], state: wp.array3d[Any]):
    """Advance causal-convolution state in place."""
    batch, channel = wp.tid()
    sequence_length = x.shape[2]
    for state_index in range(state.shape[2]):
        source_index = sequence_length + state_index
        if source_index < state.shape[2]:
            state[batch, channel, state_index] = state[batch, channel, source_index]
        else:
            state[batch, channel, state_index] = x[
                batch, channel, source_index - state.shape[2]
            ]


@wp.kernel(enable_backward=False, module="unique")
def _causal_conv_rows_kernel(
    x: wp.array2d[Any],
    weight: wp.array3d[Any],
    bias: wp.array1d[Any],
    state: wp.array2d[Any],
    output: wp.array2d[Any],
    has_bias: bool,
):
    """Apply optionally biased SiLU causal convolution to row-major tokens."""
    token, channel = wp.tid()
    total = wp.float32(bias[channel]) if has_bias else wp.float32(0.0)
    for kernel_index in range(weight.shape[2]):
        source_token = token + kernel_index - state.shape[1]
        value = (
            wp.float32(state[channel, source_token + state.shape[1]])
            if source_token < 0
            else wp.float32(x[source_token, channel])
        )
        total += value * wp.float32(weight[channel, 0, kernel_index])
    output[token, channel] = x.dtype(total / (wp.float32(1.0) + wp.exp(-total)))


@wp.kernel(enable_backward=False, module="unique")
def _update_conv_rows_state_kernel(x: wp.array2d[Any], state: wp.array2d[Any]):
    """Advance a row-major causal-convolution state in place."""
    channel = wp.tid()
    for state_index in range(state.shape[1]):
        source = x.shape[0] + state_index
        if source < state.shape[1]:
            state[channel, state_index] = state[channel, source]
        else:
            state[channel, state_index] = x[source - state.shape[1], channel]


@lru_cache(maxsize=None)
def _get_causal_conv_decode_batch_kernels(mapped: bool = False):
    """Build convolution kernels with statically selected slot indexing."""
    MAPPED = mapped

    @wp.kernel(enable_backward=False, module="unique")
    def causal(
        x: wp.array2d[Any],
        weight: wp.array3d[Any],
        state: wp.array3d[Any],
        active: wp.array1d[wp.bool],
        slot_indices: wp.array1d[wp.int32],
        output: wp.array2d[Any],
    ):
        batch, channel = wp.tid()
        if not active[batch]:
            output[batch, channel] = x.dtype(0.0)
            return
        slot = slot_indices[batch] if wp.static(MAPPED) else batch
        total = wp.float32(0.0)
        width = state.shape[2]
        for kernel_index in range(weight.shape[2]):
            value = (
                wp.float32(state[slot, channel, kernel_index])
                if kernel_index < width
                else wp.float32(x[batch, channel])
            )
            total += value * wp.float32(weight[channel, 0, kernel_index])
        output[batch, channel] = x.dtype(total / (wp.float32(1.0) + wp.exp(-total)))

    @wp.kernel(enable_backward=False, module="unique")
    def update(
        x: wp.array2d[Any],
        state: wp.array3d[Any],
        active: wp.array1d[wp.bool],
        slot_indices: wp.array1d[wp.int32],
    ):
        batch, channel = wp.tid()
        if active[batch]:
            slot = slot_indices[batch] if wp.static(MAPPED) else batch
            for index in range(state.shape[2] - 1):
                state[slot, channel, index] = state[slot, channel, index + 1]
            state[slot, channel, state.shape[2] - 1] = x[batch, channel]

    return causal, update


@wp.kernel(enable_backward=False, module="unique")
def _prepare_gated_delta_kernel(
    a: wp.array2d[Any],
    b: wp.array2d[Any],
    a_log: wp.array1d[Any],
    dt_bias: wp.array1d[Any],
    a_is_decay: bool,
    decay: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
):
    """Compute multiplicative FP32 decay and beta for gated-delta attention."""
    row, head = wp.tid()
    b_value = wp.float32(b[row, head])
    beta[row, head] = wp.float32(1.0) / (wp.float32(1.0) + wp.exp(-b_value))
    dt = wp.float32(a[row, head]) + wp.float32(dt_bias[head])
    softplus = wp.max(dt, wp.float32(0.0)) + wp.log(
        wp.float32(1.0) + wp.exp(-wp.abs(dt))
    )
    a_value = wp.float32(a_log[head])
    log_decay = (a_value if a_is_decay else -wp.exp(a_value)) * softplus
    decay[row, head] = wp.exp(log_decay)


_dequantize_nbits_kernel_cache = {}


def _get_dequantize_nbits_kernel(bits: int, block_size: int, dtype: type):
    """Return a cached packed N-bit dequantization kernel."""
    key = (bits, block_size, dtype)
    if key not in _dequantize_nbits_kernel_cache:
        _dequantize_nbits_kernel_cache[key] = _create_dequantize_nbits_kernel(*key)
    return _dequantize_nbits_kernel_cache[key]


def _create_rms_norm_kernels(tile_width: int, dtype: type, scale_dtype: type):
    """Build RMSNorm and residual-RMSNorm kernels for ``tile_width``.
    Tiles zero-pad widths that are not exact multiples."""
    TILE_WIDTH = tile_width
    DTYPE = dtype
    SCALE_DTYPE = scale_dtype

    @wp.func
    def square(value: dtype):
        value_fp32 = wp.float32(dtype(value))
        return value_fp32 * value_fp32

    @wp.func
    def skip_square(value: dtype, skip: dtype):
        value_fp32 = wp.float32(dtype(value)) + wp.float32(skip)
        return value_fp32 * value_fp32

    @wp.func
    def add(value: dtype, skip: dtype):
        return dtype(wp.float32(value) + wp.float32(skip))

    @wp.func
    def normalize(
        value: dtype, scale: SCALE_DTYPE, inverse_rms: float, scale_offset: float
    ):
        return dtype(
            wp.float32(value)
            * (wp.float32(SCALE_DTYPE(scale)) + scale_offset)
            * inverse_rms
        )

    @wp.func
    def skip_normalize(
        value: dtype,
        skip: dtype,
        scale: SCALE_DTYPE,
        inverse_rms: float,
        scale_offset: float,
    ):
        return dtype(
            (wp.float32(value) + wp.float32(skip))
            * (wp.float32(SCALE_DTYPE(scale)) + scale_offset)
            * inverse_rms
        )

    @wp.kernel(enable_backward=False, module="unique")
    def rms_norm(
        x: wp.array2d(dtype=DTYPE),
        scale: wp.array1d(dtype=SCALE_DTYPE),
        output: wp.array2d(dtype=DTYPE),
        epsilon: float,
        scale_offset: float,
    ):
        """Apply row-wise RMS normalization."""
        row = wp.tid()
        typed_zero = DTYPE(0.0)
        scale_zero = SCALE_DTYPE(0.0)
        partials = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            values = wp.tile_load(
                x[row], shape=(TILE_WIDTH,), offset=(tile_index * TILE_WIDTH,)
            )
            partials += wp.tile_map(square, values)
        inverse_rms = wp.float32(1.0) / wp.sqrt(
            wp.tile_extract(wp.tile_sum(partials), 0) / wp.float32(x.shape[1])
            + wp.float32(epsilon)
            + wp.float32(typed_zero)
            + wp.float32(scale_zero)
        )
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            scales = wp.tile_load(scale, shape=(TILE_WIDTH,), offset=(offset,))
            wp.tile_store(
                output[row],
                wp.tile_map(normalize, values, scales, inverse_rms, scale_offset),
                offset=(offset,),
            )

    @wp.kernel(enable_backward=False, module="unique")
    def skip_rms_norm(
        x: wp.array2d(dtype=DTYPE),
        skip: wp.array2d(dtype=DTYPE),
        scale: wp.array1d(dtype=SCALE_DTYPE),
        output: wp.array2d(dtype=DTYPE),
        residual: wp.array2d(dtype=DTYPE),
        epsilon: float,
        scale_offset: float,
    ):
        """Add a residual, store it, and apply row-wise RMS normalization."""
        row = wp.tid()
        typed_zero = DTYPE(0.0)
        scale_zero = SCALE_DTYPE(0.0)
        partials = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            skips = wp.tile_load(skip[row], shape=(TILE_WIDTH,), offset=(offset,))
            partials += wp.tile_map(skip_square, values, skips)
            wp.tile_store(
                residual[row], wp.tile_map(add, values, skips), offset=(offset,)
            )
        inverse_rms = wp.float32(1.0) / wp.sqrt(
            wp.tile_extract(wp.tile_sum(partials), 0) / wp.float32(x.shape[1])
            + wp.float32(epsilon)
            + wp.float32(typed_zero)
            + wp.float32(scale_zero)
        )
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            skips = wp.tile_load(skip[row], shape=(TILE_WIDTH,), offset=(offset,))
            scales = wp.tile_load(scale, shape=(TILE_WIDTH,), offset=(offset,))
            normalized = wp.tile_map(
                skip_normalize, values, skips, scales, inverse_rms, scale_offset
            )
            wp.tile_store(output[row], normalized, offset=(offset,))

    return rms_norm, skip_rms_norm


_rms_norm_kernel_cache = {}


def _get_rms_norm_kernels(width: int, dtype: type, scale_dtype: type | None = None):
    """Return cached RMSNorm kernels and the padded tile width."""
    tile_width = min(512, max(32, 1 << (width - 1).bit_length()))
    key = (tile_width, dtype, scale_dtype or dtype)
    if key not in _rms_norm_kernel_cache:
        _rms_norm_kernel_cache[key] = _create_rms_norm_kernels(*key)
    return tile_width, _rms_norm_kernel_cache[key]


def _create_layer_norm_kernel(tile_width: int, dtype: type):
    """Build an affine-free row-wise LayerNorm kernel."""
    TILE_WIDTH = tile_width
    DTYPE = dtype

    @wp.func
    def to_float(value: dtype):
        return wp.float32(dtype(value))

    @wp.func
    def square(value: dtype):
        value_fp32 = wp.float32(dtype(value))
        return value_fp32 * value_fp32

    @wp.func
    def normalize(value: dtype, mean: float, inverse_std: float):
        return dtype((wp.float32(dtype(value)) - mean) * inverse_std)

    @wp.kernel(enable_backward=False, module="unique")
    def layer_norm(
        x: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        epsilon: wp.float32,
    ):
        row = wp.tid()
        typed_zero = DTYPE(0.0)
        sums = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        squares = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            values = wp.tile_load(
                x[row], shape=(TILE_WIDTH,), offset=(tile_index * TILE_WIDTH,)
            )
            sums += wp.tile_map(to_float, values)
            squares += wp.tile_map(square, values)
        mean = wp.tile_extract(wp.tile_sum(sums), 0) / wp.float32(x.shape[1])
        variance = (
            wp.tile_extract(wp.tile_sum(squares), 0) / wp.float32(x.shape[1])
            - mean * mean
        )
        inverse_std = wp.float32(1.0) / wp.sqrt(
            wp.max(variance, wp.float32(0.0)) + epsilon + wp.float32(typed_zero)
        )
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            wp.tile_store(
                output[row],
                wp.tile_map(normalize, values, mean, inverse_std),
                offset=(offset,),
            )

    return layer_norm


_layer_norm_kernel_cache = {}


def _get_layer_norm_kernel(width: int, dtype: type):
    """Return a cached affine-free LayerNorm kernel and padded tile width."""
    tile_width = min(512, max(32, 1 << (width - 1).bit_length()))
    key = (tile_width, dtype)
    if key not in _layer_norm_kernel_cache:
        _layer_norm_kernel_cache[key] = _create_layer_norm_kernel(*key)
    return tile_width, _layer_norm_kernel_cache[key]


def _create_gated_rms_norm_kernel(
    tile_width: int, dtype: type, scale_dtype: type, norm_before_gate: bool
):
    """Build fused RMSNorm-times-SiLU gating for recurrent attention."""
    TILE_WIDTH = tile_width
    DTYPE = dtype
    SCALE_DTYPE = scale_dtype
    NORM_BEFORE_GATE = norm_before_gate

    @wp.func
    def square(value: dtype):
        value_fp32 = wp.float32(dtype(value))
        return value_fp32 * value_fp32

    @wp.func
    def gated_square(value: dtype, gate: dtype):
        gate_fp32 = wp.float32(gate)
        gated = (
            wp.float32(dtype(value))
            * gate_fp32
            / (wp.float32(1.0) + wp.exp(-gate_fp32))
        )
        return gated * gated

    @wp.func
    def normalize_gate(
        value: dtype, gate: dtype, scale: scale_dtype, inverse_rms: float
    ):
        gate_fp32 = wp.float32(gate)
        silu = gate_fp32 / (wp.float32(1.0) + wp.exp(-gate_fp32))
        return dtype(
            wp.float32(value) * wp.float32(scale_dtype(scale)) * inverse_rms * silu
        )

    @wp.func
    def gate_normalize(
        value: dtype, gate: dtype, scale: scale_dtype, inverse_rms: float
    ):
        gate_fp32 = wp.float32(gate)
        silu = gate_fp32 / (wp.float32(1.0) + wp.exp(-gate_fp32))
        return dtype(
            wp.float32(value) * silu * wp.float32(scale_dtype(scale)) * inverse_rms
        )

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        x: wp.array2d(dtype=DTYPE),
        gate: wp.array2d(dtype=DTYPE),
        scale: wp.array2d(dtype=SCALE_DTYPE),
        output: wp.array2d(dtype=DTYPE),
        epsilon: float,
    ):
        """Normalize each row, then multiply by its SiLU gate."""
        row = wp.tid()
        scale_row = row % scale.shape[0]
        typed_zero = DTYPE(0.0)
        scale_zero = SCALE_DTYPE(0.0)
        partials = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            if NORM_BEFORE_GATE:
                partials += wp.tile_map(square, values)
            else:
                gates = wp.tile_load(gate[row], shape=(TILE_WIDTH,), offset=(offset,))
                partials += wp.tile_map(gated_square, values, gates)
        inverse_rms = wp.float32(1.0) / wp.sqrt(
            wp.tile_extract(wp.tile_sum(partials), 0) / wp.float32(x.shape[1])
            + wp.float32(epsilon)
            + wp.float32(typed_zero)
            + wp.float32(scale_zero)
        )
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            gates = wp.tile_load(gate[row], shape=(TILE_WIDTH,), offset=(offset,))
            scales = wp.tile_load(
                scale[scale_row], shape=(TILE_WIDTH,), offset=(offset,)
            )
            if NORM_BEFORE_GATE:
                normalized = wp.tile_map(
                    normalize_gate, values, gates, scales, inverse_rms
                )
            else:
                normalized = wp.tile_map(
                    gate_normalize, values, gates, scales, inverse_rms
                )
            wp.tile_store(output[row], normalized, offset=(offset,))

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_gated_rms_norm_kernel(
    width: int,
    dtype: type,
    norm_before_gate: bool = True,
    scale_dtype: type | None = None,
):
    """Return a cached recurrent gated-RMSNorm kernel and tile width."""
    tile_width = min(512, max(32, 1 << (width - 1).bit_length()))
    return tile_width, _create_gated_rms_norm_kernel(
        tile_width, dtype, scale_dtype or dtype, norm_before_gate
    )


@wp.kernel(enable_backward=False, module="unique")
def _relu2_kernel(x: wp.array2d[Any], output: wp.array2d[Any]):
    """Apply squared ReLU elementwise."""
    row, column = wp.tid()
    value = wp.max(wp.float32(x[row, column]), wp.float32(0.0))
    output[row, column] = x.dtype(value * value)


def _create_lp_normalization_kernel(tile_width: int, dtype: type):
    """Build row-wise L2 normalization using ``tile_width`` lanes."""
    TILE_WIDTH = tile_width
    DTYPE = dtype

    @wp.func
    def square(value: dtype):
        value_fp32 = wp.float32(dtype(value))
        return value_fp32 * value_fp32

    @wp.func
    def normalize(value: dtype, inverse_norm: float):
        return dtype(wp.float32(value) * inverse_norm)

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        x: wp.array2d(dtype=DTYPE), output: wp.array2d(dtype=DTYPE), epsilon: float
    ):
        """Normalize each row to unit L2 norm."""
        row = wp.tid()
        typed_zero = DTYPE(0.0)
        partials = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            values = wp.tile_load(
                x[row], shape=(TILE_WIDTH,), offset=(tile_index * TILE_WIDTH,)
            )
            partials += wp.tile_map(square, values)
        norm = wp.sqrt(
            wp.tile_extract(wp.tile_sum(partials), 0)
            + wp.float32(epsilon)
            + wp.float32(typed_zero)
        )
        inverse_norm = wp.float32(1.0) / wp.max(norm, wp.float32(1.0e-12))
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            wp.tile_store(
                output[row],
                wp.tile_map(normalize, values, inverse_norm),
                offset=(offset,),
            )

    return kernel


_lp_normalization_kernel_cache = {}


def _get_lp_normalization_kernel(width: int, dtype: type):
    """Return a cached L2-normalization kernel and padded tile width."""
    tile_width = min(512, max(32, 1 << (width - 1).bit_length()))
    key = (tile_width, dtype)
    if key not in _lp_normalization_kernel_cache:
        _lp_normalization_kernel_cache[key] = _create_lp_normalization_kernel(*key)
    return tile_width, _lp_normalization_kernel_cache[key]


@lru_cache(maxsize=None)
def _get_reduce_sum_rows_kernel(width: int, dtype: type):
    """Build a row-sum kernel and choose its padded tile width."""
    tile_width = min(512, max(32, 1 << (width - 1).bit_length()))
    TILE_WIDTH = tile_width
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(x: wp.array2d(dtype=DTYPE), out: wp.array1d(dtype=DTYPE)):
        """Reduce each matrix row with ``tile_sum``."""
        row = wp.tid()
        values = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=DTYPE)
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values += wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
        wp.tile_store(out, wp.tile_sum(values), offset=row)

    return tile_width, kernel


def _linear_attention_value_blocks(value_size: int) -> int:
    """Return the number of independently scheduled value-column tiles."""
    return value_size // min(32, value_size & -value_size)


def _create_linear_attention_kernel(
    key_size: int,
    value_size: int,
    dtype: type,
    state_dtype: type,
    scalar_gated_delta: bool,
):
    """Build recurrent linear attention for fixed key and value widths.
    Value channels are processed in tiles of at most 32."""
    KEY_SIZE = key_size
    VALUE_SIZE = value_size
    VALUE_TILE = min(32, value_size & -value_size)
    VALUE_BLOCKS = _linear_attention_value_blocks(value_size)
    DTYPE = dtype
    STATE_DTYPE = state_dtype
    SCALAR_GATED_DELTA = scalar_gated_delta
    if SCALAR_GATED_DELTA and STATE_DTYPE != wp.float32:
        raise ValueError("the scalar gated-delta path requires float32 state")

    @wp.func
    def exp_value(value: state_dtype):
        return state_dtype(wp.exp(wp.float32(value)))

    @wp.func
    def to_state(value: dtype):
        return state_dtype(wp.float32(dtype(value)))

    @wp.func
    def to_output(value: state_dtype):
        return dtype(wp.float32(state_dtype(value)))

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        query: wp.array2d(dtype=DTYPE),
        key: wp.array2d(dtype=DTYPE),
        value: wp.array2d(dtype=DTYPE),
        past: wp.array2d(dtype=STATE_DTYPE),
        decay: wp.array2d(dtype=STATE_DTYPE),
        beta: wp.array2d(dtype=STATE_DTYPE),
        output: wp.array2d(dtype=DTYPE),
        present: wp.array2d(dtype=STATE_DTYPE),
        sequence_length: int,
        query_heads: int,
        key_heads: int,
        value_heads: int,
        tiled_value_heads: bool,
        needs_decay: bool,
        decay_per_key: bool,
        needs_beta: bool,
        beta_per_head: bool,
        scale: float,
    ):
        """Update recurrent attention state and emit the current sequence."""
        item = wp.tid()
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        value_block = item % VALUE_BLOCKS
        state_item = item / VALUE_BLOCKS
        batch = state_item / value_heads
        value_head = state_item % value_heads
        # HF groups value heads below each key head. Some checkpoint formats
        # transpose those two head axes for contiguous tiled execution.
        key_head = (
            value_head % key_heads
            if tiled_value_heads
            else value_head * key_heads / value_heads
        )
        state_offset = state_item * KEY_SIZE
        value_offset = value_block * VALUE_TILE
        state = wp.tile_load(
            past, shape=(KEY_SIZE, VALUE_TILE), offset=(state_offset, value_offset)
        )

        for token in range(sequence_length):
            token_row = batch * sequence_length + token
            if SCALAR_GATED_DELTA:
                key_row = wp.tile_map(
                    to_state,
                    wp.tile_load(
                        key,
                        shape=(1, KEY_SIZE),
                        offset=(token_row, key_head * KEY_SIZE),
                    ),
                )
                value_row = wp.tile_map(
                    to_state,
                    wp.tile_load(
                        value,
                        shape=(1, VALUE_TILE),
                        offset=(token_row, value_head * VALUE_SIZE + value_offset),
                    ),
                )
                query_head = (
                    key_head * query_heads / key_heads
                    if tiled_value_heads
                    else value_head * query_heads / value_heads
                )
                query_row = wp.tile_map(
                    to_state,
                    wp.tile_load(
                        query,
                        shape=(1, KEY_SIZE),
                        offset=(token_row, query_head * KEY_SIZE),
                    ),
                )
                decay_value = STATE_DTYPE(decay[token_row, value_head])
                beta_value = STATE_DTYPE(beta[token_row, value_head])
                probes = wp.tile_zeros(shape=(2, KEY_SIZE), dtype=STATE_DTYPE)
                wp.tile_assign(probes, key_row, offset=(0, 0))
                wp.tile_assign(probes, query_row, offset=(1, 0))
                projections = wp.tile_zeros(shape=(2, VALUE_TILE), dtype=STATE_DTYPE)
                wp.tile_matmul(probes, state, projections)
                retrieved_unscaled = wp.tile_view(
                    projections, offset=(0, 0), shape=(1, VALUE_TILE)
                )
                query_unscaled = wp.tile_view(
                    projections, offset=(1, 0), shape=(1, VALUE_TILE)
                )
                delta = beta_value * (value_row - decay_value * retrieved_unscaled)
                query_key = wp.tile_extract(wp.tile_sum(query_row * key_row), 0)
                wp.tile_matmul(
                    wp.tile_transpose(key_row),
                    delta,
                    state,
                    alpha=STATE_DTYPE(1.0),
                    beta=decay_value,
                )
                result = decay_value * query_unscaled + query_key * delta
                wp.tile_store(
                    output,
                    wp.tile_map(to_output, STATE_DTYPE(scale) * result),
                    offset=(token_row, value_head * VALUE_SIZE + value_offset),
                )
                continue
            if needs_decay:
                if decay_per_key:
                    decay_row = wp.tile_load(
                        decay,
                        shape=(1, KEY_SIZE),
                        offset=(token_row, value_head * KEY_SIZE),
                    )
                    decay_column = wp.tile_transpose(wp.tile_map(exp_value, decay_row))
                    state *= wp.tile_broadcast(
                        decay_column, shape=(KEY_SIZE, VALUE_TILE)
                    )
                else:
                    state *= STATE_DTYPE(
                        wp.exp(wp.float32(decay[token_row, value_head]))
                    )

            key_row = wp.tile_map(
                to_state,
                wp.tile_load(
                    key, shape=(1, KEY_SIZE), offset=(token_row, key_head * KEY_SIZE)
                ),
            )
            value_row = wp.tile_map(
                to_state,
                wp.tile_load(
                    value,
                    shape=(1, VALUE_TILE),
                    offset=(token_row, value_head * VALUE_SIZE + value_offset),
                ),
            )
            if needs_beta:
                retrieved = wp.tile_zeros(shape=(1, VALUE_TILE), dtype=STATE_DTYPE)
                wp.tile_matmul(key_row, state, retrieved)
                beta_value = (
                    beta[token_row, value_head] if beta_per_head else beta[token_row, 0]
                )
                delta = STATE_DTYPE(beta_value) * (value_row - retrieved)
            else:
                delta = value_row
            wp.tile_matmul(wp.tile_transpose(key_row), delta, state)

            if query_heads >= value_heads:
                heads_per_group = query_heads / value_heads
                for group in range(heads_per_group):
                    query_head = value_head * heads_per_group + group
                    query_row = wp.tile_map(
                        to_state,
                        wp.tile_load(
                            query,
                            shape=(1, KEY_SIZE),
                            offset=(token_row, query_head * KEY_SIZE),
                        ),
                    )
                    result = wp.tile_zeros(shape=(1, VALUE_TILE), dtype=STATE_DTYPE)
                    wp.tile_matmul(query_row, state, result)
                    wp.tile_store(
                        output,
                        wp.tile_map(to_output, STATE_DTYPE(scale) * result),
                        offset=(token_row, query_head * VALUE_SIZE + value_offset),
                    )
            else:
                query_head = (
                    key_head * query_heads / key_heads
                    if tiled_value_heads
                    else value_head * query_heads / value_heads
                )
                query_row = wp.tile_map(
                    to_state,
                    wp.tile_load(
                        query,
                        shape=(1, KEY_SIZE),
                        offset=(token_row, query_head * KEY_SIZE),
                    ),
                )
                result = wp.tile_zeros(shape=(1, VALUE_TILE), dtype=STATE_DTYPE)
                wp.tile_matmul(query_row, state, result)
                wp.tile_store(
                    output,
                    wp.tile_map(to_output, STATE_DTYPE(scale) * result),
                    offset=(token_row, value_head * VALUE_SIZE + value_offset),
                )

        wp.tile_store(present, state, offset=(state_offset, value_offset))

    kernel.module.options["enable_backward"] = False
    return kernel


_linear_attention_kernel_cache = {}


def _get_linear_attention_kernel(
    key_size: int,
    value_size: int,
    dtype: type,
    state_dtype: type | None = None,
    scalar_gated_delta: bool = False,
):
    """Return a cached recurrent linear-attention kernel."""
    key = (key_size, value_size, dtype, state_dtype or dtype, scalar_gated_delta)
    if key not in _linear_attention_kernel_cache:
        _linear_attention_kernel_cache[key] = _create_linear_attention_kernel(*key)
    return _linear_attention_kernel_cache[key]


@lru_cache(maxsize=None)
def _get_gated_delta_decode_batch_kernel(
    key_size: int, value_size: int, dtype: type, mapped: bool = False
):
    """Build masked one-token scalar gated-delta attention for a batch."""
    KEY_SIZE = key_size
    VALUE_SIZE = value_size
    VALUE_TILE = min(32, value_size & -value_size)
    VALUE_BLOCKS = _linear_attention_value_blocks(value_size)
    DTYPE = dtype
    MAPPED = mapped

    @wp.func
    def to_state(value: dtype):
        return wp.float32(dtype(value))

    @wp.func
    def to_output(value: wp.float32):
        return dtype(value)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        query: wp.array2d(dtype=DTYPE),
        key: wp.array2d(dtype=DTYPE),
        value: wp.array2d(dtype=DTYPE),
        state: wp.array2d[wp.float32],
        decay: wp.array2d[wp.float32],
        beta: wp.array2d[wp.float32],
        active: wp.array1d[wp.bool],
        slot_indices: wp.array1d[wp.int32],
        output: wp.array2d(dtype=DTYPE),
        query_heads: int,
        key_heads: int,
        value_heads: int,
        scale: float,
    ):
        item = wp.tid()
        value_block = item % VALUE_BLOCKS
        state_item = item / VALUE_BLOCKS
        batch = state_item / value_heads
        value_head = state_item % value_heads
        value_offset = value_block * VALUE_TILE
        if not active[batch]:
            for column in range(VALUE_TILE):
                output[batch, value_head * VALUE_SIZE + value_offset + column] = DTYPE(
                    0.0
                )
            return
        key_head = value_head % key_heads
        slot = slot_indices[batch] if wp.static(MAPPED) else batch
        state_offset = (slot * value_heads + value_head) * KEY_SIZE
        state_tile = wp.tile_load(
            state, shape=(KEY_SIZE, VALUE_TILE), offset=(state_offset, value_offset)
        )
        key_row = wp.tile_map(
            to_state,
            wp.tile_load(key, shape=(1, KEY_SIZE), offset=(batch, key_head * KEY_SIZE)),
        )
        value_row = wp.tile_map(
            to_state,
            wp.tile_load(
                value,
                shape=(1, VALUE_TILE),
                offset=(batch, value_head * VALUE_SIZE + value_offset),
            ),
        )
        query_head = key_head * query_heads / key_heads
        query_row = wp.tile_map(
            to_state,
            wp.tile_load(
                query, shape=(1, KEY_SIZE), offset=(batch, query_head * KEY_SIZE)
            ),
        )
        decay_value = decay[batch, value_head]
        beta_value = beta[batch, value_head]
        probes = wp.tile_zeros(shape=(2, KEY_SIZE), dtype=wp.float32)
        wp.tile_assign(probes, key_row, offset=(0, 0))
        wp.tile_assign(probes, query_row, offset=(1, 0))
        projections = wp.tile_zeros(shape=(2, VALUE_TILE), dtype=wp.float32)
        wp.tile_matmul(probes, state_tile, projections)
        retrieved = wp.tile_view(projections, offset=(0, 0), shape=(1, VALUE_TILE))
        query_result = wp.tile_view(projections, offset=(1, 0), shape=(1, VALUE_TILE))
        delta = beta_value * (value_row - decay_value * retrieved)
        query_key = wp.tile_extract(wp.tile_sum(query_row * key_row), 0)
        wp.tile_matmul(
            wp.tile_transpose(key_row),
            delta,
            state_tile,
            alpha=wp.float32(1.0),
            beta=decay_value,
        )
        result = decay_value * query_result + query_key * delta
        wp.tile_store(
            output,
            wp.tile_map(to_output, wp.float32(scale) * result),
            offset=(batch, value_head * VALUE_SIZE + value_offset),
        )
        wp.tile_store(state, state_tile, offset=(state_offset, value_offset))

    return kernel


def _create_mamba2_decode_kernel(
    head_dim: int, state_size: int, heads_per_group: int, dtype: type
):
    """Build one-token Mamba-2 selective-state update and projection."""
    HEAD_DIM = head_dim
    STATE_TILE = max(32, 1 << (state_size - 1).bit_length())
    HEADS_PER_GROUP = heads_per_group
    DTYPE = dtype

    @wp.func
    def to_float(value: dtype):
        return wp.float32(dtype(value))

    @wp.func
    def update_state(
        value: wp.float32, b: wp.float32, decay: wp.float32, source: wp.float32
    ):
        return value * decay + b * source

    @wp.func
    def multiply(value: wp.float32, scale: wp.float32):
        return value * scale

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        x: wp.array2d(dtype=DTYPE),
        b: wp.array2d(dtype=DTYPE),
        c: wp.array2d(dtype=DTYPE),
        dt: wp.array1d(dtype=DTYPE),
        a_log: wp.array1d[wp.float32],
        dt_bias: wp.array1d[wp.float32],
        d: wp.array1d[wp.float32],
        state: wp.array2d[wp.float32],
        output: wp.array2d(dtype=DTYPE),
        time_step_min: float,
        time_step_max: float,
    ):
        """Update FP32 state and emit one Mamba-2 token for each head."""
        item = wp.tid()
        head = item / HEAD_DIM
        channel = item % HEAD_DIM
        group = head / HEADS_PER_GROUP
        step_input = wp.float32(dt[head]) + dt_bias[head]
        step = wp.max(step_input, 0.0) + wp.log(1.0 + wp.exp(-wp.abs(step_input)))
        step = wp.clamp(step, wp.float32(time_step_min), wp.float32(time_step_max))
        decay = wp.exp(-wp.exp(a_log[head]) * step)
        source = step * wp.float32(x[head, channel])

        state_row = head * HEAD_DIM + channel
        values = wp.tile_load(state[state_row], shape=(STATE_TILE,))
        b_values = wp.tile_map(to_float, wp.tile_load(b[group], shape=(STATE_TILE,)))
        c_values = wp.tile_map(to_float, wp.tile_load(c[group], shape=(STATE_TILE,)))
        values = wp.tile_map(update_state, values, b_values, decay, source)
        wp.tile_store(state[state_row], values)
        projected = wp.tile_extract(
            wp.tile_sum(wp.tile_map(multiply, values, c_values)), 0
        )
        output[head, channel] = DTYPE(
            projected + d[head] * wp.float32(x[head, channel])
        )

    kernel.module.options["enable_backward"] = False
    return STATE_TILE, kernel


@lru_cache(maxsize=None)
def _get_mamba2_decode_kernel(
    head_dim: int, state_size: int, heads_per_group: int, dtype: type
):
    """Return the cached one-token Mamba-2 kernel and reduction width."""
    if min(head_dim, state_size, heads_per_group) <= 0:
        raise ValueError("Mamba-2 dimensions must be positive")
    return _create_mamba2_decode_kernel(head_dim, state_size, heads_per_group, dtype)


def _create_mamba2_prefill_kernel(
    head_dim: int, state_size: int, heads_per_group: int, dtype: type
):
    """Build chunked Mamba-2 prefill with state kept on-chip."""
    HEAD_DIM = head_dim
    STATE_SIZE = state_size
    CHANNEL_TILE = min(32, HEAD_DIM & -HEAD_DIM)
    CHANNEL_BLOCKS = HEAD_DIM // CHANNEL_TILE
    HEADS_PER_GROUP = heads_per_group
    DTYPE = dtype

    @wp.func
    def to_float(value: dtype):
        return wp.float32(dtype(value))

    @wp.func
    def to_output(value: wp.float32):
        return dtype(value)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        x: wp.array2d(dtype=DTYPE),
        b: wp.array2d(dtype=DTYPE),
        c: wp.array2d(dtype=DTYPE),
        dt: wp.array2d(dtype=DTYPE),
        a_log: wp.array1d[wp.float32],
        dt_bias: wp.array1d[wp.float32],
        d: wp.array1d[wp.float32],
        state: wp.array2d[wp.float32],
        output: wp.array2d(dtype=DTYPE),
        sequence_length: int,
        time_step_min: float,
        time_step_max: float,
    ):
        """Process one sequence chunk and update its persistent FP32 state."""
        item = wp.tid()
        typed_zero = DTYPE(0.0)
        head = item / CHANNEL_BLOCKS
        channel_offset = (item % CHANNEL_BLOCKS) * CHANNEL_TILE
        group = head / HEADS_PER_GROUP
        state_row = head * HEAD_DIM + channel_offset
        values = wp.tile_transpose(
            wp.tile_load(
                state,
                shape=(CHANNEL_TILE, STATE_SIZE),
                offset=(state_row, 0),
                storage="register",
            )
        )

        for token in range(sequence_length):
            step_input = wp.float32(dt[token, head] + typed_zero) + dt_bias[head]
            step = wp.max(step_input, 0.0) + wp.log(1.0 + wp.exp(-wp.abs(step_input)))
            step = wp.clamp(step, wp.float32(time_step_min), wp.float32(time_step_max))
            wp.tile_assign(values, values * wp.exp(-wp.exp(a_log[head]) * step))

            x_row = wp.tile_map(
                to_float,
                wp.tile_load(
                    x,
                    shape=(1, CHANNEL_TILE),
                    offset=(token, head * HEAD_DIM + channel_offset),
                ),
            )
            b_column = wp.tile_transpose(
                wp.tile_map(
                    to_float,
                    wp.tile_load(
                        b, shape=(1, STATE_SIZE), offset=(token, group * STATE_SIZE)
                    ),
                )
            )
            wp.tile_matmul(b_column, step * x_row, values)

            c_row = wp.tile_map(
                to_float,
                wp.tile_load(
                    c, shape=(1, STATE_SIZE), offset=(token, group * STATE_SIZE)
                ),
            )
            projected = wp.tile_zeros(shape=(1, CHANNEL_TILE), dtype=wp.float32)
            wp.tile_matmul(c_row, values, projected)
            wp.tile_store(
                output,
                wp.tile_map(to_output, projected + d[head] * x_row),
                offset=(token, head * HEAD_DIM + channel_offset),
            )

        wp.tile_store(state, wp.tile_transpose(values), offset=(state_row, 0))

    kernel.module.options["enable_backward"] = False
    return CHANNEL_BLOCKS, 128, kernel


@lru_cache(maxsize=None)
def _get_mamba2_prefill_kernel(
    head_dim: int, state_size: int, heads_per_group: int, dtype: type
):
    """Return cached prefill, channel blocks per head, and block width."""
    if min(head_dim, state_size, heads_per_group) <= 0:
        raise ValueError("Mamba-2 dimensions must be positive")
    return _create_mamba2_prefill_kernel(head_dim, state_size, heads_per_group, dtype)


def _create_swiglu_kernel(dtype: type):
    """Build a fused SiLU-gate-times-up-projection kernel."""

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        gate: wp.array2d(dtype=dtype),
        up: wp.array2d(dtype=dtype),
        output: wp.array2d(dtype=dtype),
    ):
        """Compute ``silu(gate) * up`` elementwise."""
        row, column = wp.tid()
        value = wp.float32(gate[row, column])
        silu = value / (wp.float32(1.0) + wp.exp(-value))
        output[row, column] = dtype(silu * wp.float32(up[row, column]))

    return kernel


_swiglu_kernel_cache = {}


def _get_swiglu_kernel(dtype: type):
    """Return a cached fused SwiGLU kernel for ``dtype``."""
    if dtype not in _swiglu_kernel_cache:
        _swiglu_kernel_cache[dtype] = _create_swiglu_kernel(dtype)
    return _swiglu_kernel_cache[dtype]


@wp.kernel
def _gqa_copy_past_fp16_kernel(
    past_key: wp.array4d[wp.float16],
    past_value: wp.array4d[wp.float16],
    present_key: wp.array4d[wp.float16],
    present_value: wp.array4d[wp.float16],
):
    """Copy FP16 key/value cache tensors to their present buffers."""
    batch, head, token, column = wp.tid()
    present_key[batch, head, token, column] = past_key[batch, head, token, column]
    present_value[batch, head, token, column] = past_value[batch, head, token, column]


@wp.kernel
def _gqa_prepare_fp16_kernel(
    query: wp.array3d[wp.float16],
    key: wp.array3d[wp.float16],
    value: wp.array3d[wp.float16],
    sequence_lengths_minus_one: wp.array1d[wp.int32],
    cos_cache: wp.array2d[wp.float16],
    sin_cache: wp.array2d[wp.float16],
    rotated_query: wp.array4d[wp.float16],
    present_key: wp.array4d[wp.float16],
    present_value: wp.array4d[wp.float16],
    query_heads: int,
    kv_heads: int,
    sequence_length: int,
    past_length: int,
    head_size: int,
    share_cache: bool,
    do_rotary: bool,
):
    """Rotate queries and append keys and values to the FP16 cache.
    ``share_cache`` selects absolute rather than appended cache positions."""
    batch, head, token, column = wp.tid()
    position = wp.int32(sequence_lengths_minus_one[batch]) - sequence_length + 1 + token
    cache_token = position if share_cache else past_length + token

    if head < query_heads:
        offset = head * head_size
        current = wp.float32(query[batch, token, offset + column])
        if do_rotary:
            cache_column = column % (head_size // 2)
            paired_column = (
                column + head_size // 2
                if column < head_size // 2
                else column - head_size // 2
            )
            sign = wp.float32(-1.0) if column < head_size // 2 else wp.float32(1.0)
            paired = wp.float32(query[batch, token, offset + paired_column])
            current = current * wp.float32(
                cos_cache[position, cache_column]
            ) + sign * paired * wp.float32(sin_cache[position, cache_column])
        rotated_query[batch, head, token, column] = wp.float16(current)

    if head < kv_heads:
        offset = head * head_size
        current = wp.float32(key[batch, token, offset + column])
        if do_rotary:
            cache_column = column % (head_size // 2)
            paired_column = (
                column + head_size // 2
                if column < head_size // 2
                else column - head_size // 2
            )
            sign = wp.float32(-1.0) if column < head_size // 2 else wp.float32(1.0)
            paired = wp.float32(key[batch, token, offset + paired_column])
            current = current * wp.float32(
                cos_cache[position, cache_column]
            ) + sign * paired * wp.float32(sin_cache[position, cache_column])
        present_key[batch, head, cache_token, column] = wp.float16(current)
        present_value[batch, head, cache_token, column] = value[
            batch, token, offset + column
        ]


def _create_gqa_attention_kernel(head_size: int, dtype: type):
    """Build numerically stable grouped-query attention for one dtype."""
    DTYPE = dtype

    @wp.func
    def dot(left: DTYPE, right: DTYPE):
        return wp.float32(DTYPE(left)) * wp.float32(right)

    @wp.func
    def accumulate(
        total: wp.float32, value: DTYPE, old_scale: wp.float32, weight: wp.float32
    ):
        return total * old_scale + wp.float32(DTYPE(value)) * weight

    @wp.func
    def normalize(total: wp.float32, denominator: wp.float32):
        return DTYPE(total / denominator)

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        query: wp.array2d(dtype=DTYPE),
        key: wp.array2d(dtype=DTYPE),
        value: wp.array2d(dtype=DTYPE),
        sequence_lengths_minus_one: wp.array1d[wp.int32],
        output: wp.array2d(dtype=DTYPE),
        query_heads: int,
        kv_heads: int,
        sequence_length: int,
        total_length: int,
        scale: float,
        window: int,
    ):
        """Apply causal GQA with an optional circular sliding window."""
        index = wp.tid()
        query_token = index % sequence_length
        head = (index // sequence_length) % query_heads
        batch = index // (sequence_length * query_heads)
        kv_head = head // (query_heads // kv_heads)
        valid_keys = (
            wp.int32(sequence_lengths_minus_one[batch])
            - sequence_length
            + query_token
            + 2
        )
        first_key = wp.max(0, valid_keys - window) if window > 0 else 0
        query_row = (batch * query_heads + head) * sequence_length + query_token
        query_values = wp.tile_load(query[query_row], shape=(head_size,))
        accumulator = wp.tile_zeros(shape=(head_size,), dtype=wp.float32)
        maximum = wp.float32(-3.402823466e38) + wp.float32(DTYPE(0.0))
        denominator = wp.float32(0.0)
        for key_token in range(first_key, valid_keys):
            cache_token = key_token % total_length if window > 0 else key_token
            cache_row = (batch * kv_heads + kv_head) * total_length + cache_token
            key_values = wp.tile_load(key[cache_row], shape=(head_size,))
            score = wp.tile_extract(
                wp.tile_sum(wp.tile_map(dot, query_values, key_values)), 0
            )
            score *= wp.float32(scale)
            new_maximum = wp.max(maximum, score)
            old_scale = wp.exp(maximum - new_maximum)
            weight = wp.exp(score - new_maximum)
            denominator = denominator * old_scale + weight
            value_values = wp.tile_load(value[cache_row], shape=(head_size,))
            accumulator = wp.tile_map(
                accumulate, accumulator, value_values, old_scale, weight
            )
            maximum = new_maximum
        normalized = wp.tile_map(normalize, accumulator, denominator)
        wp.tile_store(
            output[batch * sequence_length + query_token],
            normalized,
            offset=(head * head_size,),
        )

    kernel.module.options["enable_backward"] = False
    return kernel


def _create_bidirectional_gqa_attention_kernel(head_size: int, dtype: type):
    """Build stable full/sliding GQA for fixed query and key sequences."""
    DTYPE = dtype

    @wp.func
    def dot(left: DTYPE, right: DTYPE):
        return wp.float32(DTYPE(left)) * wp.float32(DTYPE(right))

    @wp.func
    def accumulate(
        total: wp.float32, value: DTYPE, old_scale: wp.float32, weight: wp.float32
    ):
        return total * old_scale + wp.float32(DTYPE(value)) * weight

    @wp.func
    def normalize(total: wp.float32, denominator: wp.float32):
        return DTYPE(total / denominator)

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
        query_valid: wp.array2d[wp.bool],
        key_valid: wp.array2d[wp.bool],
        output: wp.array4d(dtype=DTYPE),
        scale: wp.float32,
        window: int,
    ):
        batch, head, query_token = wp.tid()
        kv_head = head / (query.shape[1] / key.shape[1])
        accumulator = wp.tile_zeros(shape=(head_size,), dtype=wp.float32)
        maximum = wp.float32(-3.402823466e38) + wp.float32(DTYPE(0.0))
        denominator = wp.float32(0.0)
        if query_valid[batch, query_token]:
            query_values = wp.tile_load(
                query[batch, head, query_token], shape=(head_size,)
            )
            for key_token in range(key.shape[2]):
                in_window = window <= 0 or wp.abs(query_token - key_token) <= window
                if key_valid[batch, key_token] and in_window:
                    key_values = wp.tile_load(
                        key[batch, kv_head, key_token], shape=(head_size,)
                    )
                    score = wp.tile_extract(
                        wp.tile_sum(wp.tile_map(dot, query_values, key_values)), 0
                    )
                    score *= scale
                    new_maximum = wp.max(maximum, score)
                    old_scale = wp.exp(maximum - new_maximum)
                    weight = wp.exp(score - new_maximum)
                    denominator = denominator * old_scale + weight
                    value_values = wp.tile_load(
                        value[batch, kv_head, key_token], shape=(head_size,)
                    )
                    accumulator = wp.tile_map(
                        accumulate, accumulator, value_values, old_scale, weight
                    )
                    maximum = new_maximum
        safe_denominator = wp.max(denominator, wp.float32(1.0e-20))
        normalized = wp.tile_map(normalize, accumulator, safe_denominator)
        wp.tile_store(output[batch, head, query_token], normalized)

    kernel.module.options["enable_backward"] = False
    return kernel


def _create_partitioned_gqa_attention_kernels(
    head_size: int,
    dtype: type,
    partitions: int,
    rows_per_group: int,
    heads_per_group: int,
    mapped: bool = False,
):
    """Build blockwise parallel decode attention and its softmax reduction."""
    DTYPE = dtype
    PARTITIONS = partitions
    GROUP = heads_per_group
    ROWS_PER_GROUP = rows_per_group
    QUERY_GROUP = GROUP * ROWS_PER_GROUP
    KEY_TILE = 32
    MAPPED = mapped

    @wp.func
    def maximum_value(left: wp.float32, right: wp.float32):
        return wp.max(left, right)

    @wp.func
    def scale_and_mask_score(
        score: wp.float32,
        key_position: wp.int32,
        first_key: wp.int32,
        end_key: wp.int32,
        tile_end: wp.int32,
        scale: wp.float32,
    ):
        return (
            score * scale
            if key_position >= first_key
            and key_position < end_key
            and key_position < tile_end
            else wp.float32(-3.402823466e38)
        )

    @wp.func
    def mask_cache_value(value: DTYPE, key_offset: wp.int32, valid_keys: wp.int32):
        return value if key_offset < valid_keys else DTYPE(0.0)

    @wp.func
    def exp_difference(value: wp.float32, maximum: wp.float32):
        return wp.exp(value - maximum)

    @wp.func
    def masked_exp_difference(
        value: wp.float32,
        maximum: wp.float32,
        key_position: wp.int32,
        first_key: wp.int32,
        end_key: wp.int32,
        tile_end: wp.int32,
    ):
        return (
            wp.exp(value - maximum)
            if key_position >= first_key
            and key_position < end_key
            and key_position < tile_end
            else wp.float32(0.0)
        )

    @wp.func
    def circular_cache_index(
        offset: wp.int32, key_start: wp.int32, length: wp.int32, base: wp.int32
    ):
        return (offset + key_start) % length + base

    @wp.func
    def grouped_query_index(
        offset: wp.int32,
        base: wp.int32,
        stride: wp.int32,
        valid_heads: wp.int32,
        valid_rows: wp.int32,
    ):
        row = wp.min(offset / GROUP, valid_rows - 1)
        head = wp.min(offset % GROUP, valid_heads - 1)
        return base + head * stride + row

    @wp.func
    def row_end_for_member(offset: wp.int32, first_end: wp.int32, valid_rows: wp.int32):
        return first_end + wp.min(offset / GROUP, valid_rows - 1)

    @wp.func
    def row_first_from_end(end: wp.int32, window: wp.int32):
        return wp.max(0, end - window) if window > 0 else 0

    @wp.func
    def add_offset(offset: wp.int32, base: wp.int32):
        return base + offset

    @wp.func
    def scale_value(value: wp.float32, scale: wp.float32):
        return value * scale

    @wp.func
    def normalize(value: wp.float32, denominator: wp.float32):
        return DTYPE(value / denominator)

    @wp.kernel(enable_backward=False, module="unique")
    def partial(
        query: wp.array2d(dtype=DTYPE),
        key: wp.array2d(dtype=DTYPE),
        value: wp.array2d(dtype=DTYPE),
        sequence_lengths_minus_one: wp.array1d[wp.int32],
        slot_indices: wp.array1d[wp.int32],
        partial_maximum: wp.array1d[wp.float32],
        partial_denominator: wp.array1d[wp.float32],
        partial_output: wp.array2d[wp.float32],
        query_heads: int,
        kv_heads: int,
        sequence_length: int,
        total_length: int,
        scale: float,
        window: int,
    ):
        item = wp.tid()
        partition = item % PARTITIONS
        group_item = item / PARTITIONS
        queries_per_kv = query_heads / kv_heads
        groups_per_kv = (queries_per_kv + GROUP - 1) / GROUP
        groups_per_batch = kv_heads * groups_per_kv
        row_groups = (sequence_length + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP
        sequence_item = group_item / groups_per_batch
        row_group = sequence_item % row_groups
        query_token_0 = row_group * ROWS_PER_GROUP
        valid_rows = wp.min(ROWS_PER_GROUP, sequence_length - query_token_0)
        batch = sequence_item / row_groups
        cache_batch = slot_indices[batch] if wp.static(MAPPED) else batch
        group = group_item % groups_per_batch
        kv_head = group / groups_per_kv
        subgroup = group % groups_per_kv
        head_0 = kv_head * queries_per_kv + subgroup * GROUP
        valid_heads = wp.min(GROUP, (kv_head + 1) * queries_per_kv - head_0)
        query_item_0 = (batch * query_heads + head_0) * sequence_length + query_token_0

        first_row_end = (
            wp.int32(sequence_lengths_minus_one[batch])
            - sequence_length
            + query_token_0
            + 2
        )
        common_end = first_row_end + valid_rows - 1
        common_first = wp.max(0, first_row_end - window) if window > 0 else 0
        key_count = common_end - common_first
        keys_per_partition = (key_count + PARTITIONS - 1) / PARTITIONS
        partition_start = common_first + partition * keys_per_partition
        partition_end = wp.min(common_end, partition_start + keys_per_partition)

        query_members = wp.tile_arange(QUERY_GROUP, dtype=wp.int32)
        query_indices = wp.tile_map(
            grouped_query_index,
            query_members,
            query_item_0,
            sequence_length,
            valid_heads,
            valid_rows,
        )
        queries = wp.tile_load_indexed(
            query,
            indices=query_indices,
            shape=(QUERY_GROUP, head_size),
            offset=(0, 0),
            axis=0,
        )
        accumulator = wp.tile_zeros(shape=(QUERY_GROUP, head_size), dtype=wp.float32)
        maximum = wp.tile_full(
            shape=(QUERY_GROUP,),
            value=wp.float32(-3.402823466e38) + wp.float32(DTYPE(0.0)),
            dtype=wp.float32,
        )
        denominator = wp.tile_zeros(shape=(QUERY_GROUP,), dtype=wp.float32)
        row_ends = wp.tile_map(
            row_end_for_member, query_members, first_row_end, valid_rows
        )
        row_firsts = wp.tile_map(row_first_from_end, row_ends, window)
        row_end_group = wp.tile_broadcast(
            wp.tile_reshape(row_ends, shape=(QUERY_GROUP, 1)),
            shape=(QUERY_GROUP, KEY_TILE),
        )
        row_first_group = wp.tile_broadcast(
            wp.tile_reshape(row_firsts, shape=(QUERY_GROUP, 1)),
            shape=(QUERY_GROUP, KEY_TILE),
        )
        key_offsets = wp.tile_arange(KEY_TILE, dtype=wp.int32)
        cache_offset_group = wp.tile_broadcast(
            wp.tile_reshape(key_offsets, shape=(KEY_TILE, 1)),
            shape=(KEY_TILE, head_size),
        )

        for key_start in range(partition_start, partition_end, KEY_TILE):
            valid_tile_keys = wp.min(KEY_TILE, partition_end - key_start)
            cache_base = (cache_batch * kv_heads + kv_head) * total_length
            cache_row = cache_base + key_start
            cache_indices = wp.tile_map(
                circular_cache_index, key_offsets, key_start, total_length, cache_base
            )
            if window > 0:
                key_values = wp.tile_load_indexed(
                    key,
                    indices=cache_indices,
                    shape=(KEY_TILE, head_size),
                    offset=(0, 0),
                    axis=0,
                )
            else:
                key_values = wp.tile_load(
                    key,
                    shape=(KEY_TILE, head_size),
                    offset=(cache_row, 0),
                )
            if valid_tile_keys < KEY_TILE:
                key_values = wp.tile_map(
                    mask_cache_value,
                    key_values,
                    cache_offset_group,
                    valid_tile_keys,
                )
            scores = wp.tile_zeros(shape=(QUERY_GROUP, KEY_TILE), dtype=wp.float32)
            wp.tile_matmul(queries, wp.tile_transpose(key_values), scores)
            key_positions = wp.tile_map(add_offset, key_offsets, key_start)
            key_position_group = wp.tile_broadcast(
                wp.tile_reshape(key_positions, shape=(1, KEY_TILE)),
                shape=(QUERY_GROUP, KEY_TILE),
            )
            scores = wp.tile_map(
                scale_and_mask_score,
                scores,
                key_position_group,
                row_first_group,
                row_end_group,
                partition_end,
                wp.float32(scale),
            )
            block_maximum = wp.tile_reduce(maximum_value, scores, axis=1)
            new_maximum = wp.tile_map(maximum_value, maximum, block_maximum)
            old_scale = wp.tile_map(exp_difference, maximum, new_maximum)
            maximum_group = wp.tile_broadcast(
                wp.tile_reshape(new_maximum, shape=(QUERY_GROUP, 1)),
                shape=(QUERY_GROUP, KEY_TILE),
            )
            probabilities = wp.tile_map(
                masked_exp_difference,
                scores,
                maximum_group,
                key_position_group,
                row_first_group,
                row_end_group,
                partition_end,
            )
            denominator = denominator * old_scale + wp.tile_sum(probabilities, axis=1)
            old_scale_group = wp.tile_broadcast(
                wp.tile_reshape(old_scale, shape=(QUERY_GROUP, 1)),
                shape=(QUERY_GROUP, head_size),
            )
            typed_probabilities = wp.tile_astype(probabilities, dtype=DTYPE)
            if window > 0:
                value_values = wp.tile_load_indexed(
                    value,
                    indices=cache_indices,
                    shape=(KEY_TILE, head_size),
                    offset=(0, 0),
                    axis=0,
                )
            else:
                value_values = wp.tile_load(
                    value,
                    shape=(KEY_TILE, head_size),
                    offset=(cache_row, 0),
                )
            if valid_tile_keys < KEY_TILE:
                value_values = wp.tile_map(
                    mask_cache_value,
                    value_values,
                    cache_offset_group,
                    valid_tile_keys,
                )
            contribution = wp.tile_zeros(
                shape=(QUERY_GROUP, head_size), dtype=wp.float32
            )
            wp.tile_matmul(typed_probabilities, value_values, contribution)
            accumulator = accumulator * old_scale_group + contribution
            maximum = new_maximum

        for row_member in range(ROWS_PER_GROUP):
            if row_member < valid_rows:
                for head_member in range(GROUP):
                    if head_member < valid_heads:
                        member = row_member * GROUP + head_member
                        output_head_item = (
                            (batch * sequence_length + query_token_0 + row_member)
                            * query_heads
                            + head_0
                            + head_member
                        )
                        partial_item = output_head_item * PARTITIONS + partition
                        partial_maximum[partial_item] = wp.tile_extract(maximum, member)
                        partial_denominator[partial_item] = wp.tile_extract(
                            denominator, member
                        )
                        accumulator_row = wp.tile_view(
                            accumulator,
                            offset=(member, 0),
                            shape=(1, head_size),
                        )
                        wp.tile_store(
                            partial_output, accumulator_row, offset=(partial_item, 0)
                        )

    @wp.kernel(enable_backward=False, module="unique")
    def reduce(
        partial_maximum: wp.array1d[wp.float32],
        partial_denominator: wp.array1d[wp.float32],
        partial_output: wp.array2d[wp.float32],
        output: wp.array2d(dtype=DTYPE),
        query_heads: int,
    ):
        head_item = wp.tid()
        typed_zero = DTYPE(0.0)
        head = head_item % query_heads
        output_row = head_item / query_heads
        offset = head_item * PARTITIONS
        maximum = wp.float32(-3.402823466e38) + wp.float32(typed_zero)
        for partition in range(PARTITIONS):
            maximum = wp.max(maximum, partial_maximum[offset + partition])

        denominator = wp.float32(0.0)
        accumulator = wp.tile_zeros(shape=(head_size,), dtype=wp.float32)
        for partition in range(PARTITIONS):
            item = offset + partition
            partial_scale = wp.exp(partial_maximum[item] - maximum)
            denominator += partial_denominator[item] * partial_scale
            values = wp.tile_load(partial_output[item], shape=(head_size,))
            accumulator += wp.tile_map(scale_value, values, partial_scale)
        wp.tile_store(
            output[output_row],
            wp.tile_map(normalize, accumulator, denominator),
            offset=(head * head_size,),
        )

    partial.module.options["enable_backward"] = False
    partial.module.mark_modified()
    reduce.module.options["enable_backward"] = False
    reduce.module.mark_modified()
    return partial, reduce


_gqa_attention_kernel_cache = {}
_bidirectional_gqa_attention_kernel_cache = {}
_partitioned_gqa_attention_kernel_cache = {}


def _get_gqa_attention_kernel(head_size: int, dtype: type = wp.float16):
    """Return cached GQA kernel and a head-sized CUDA block dimension."""
    key = (head_size, dtype)
    if key not in _gqa_attention_kernel_cache:
        _gqa_attention_kernel_cache[key] = _create_gqa_attention_kernel(*key)
    block_dim = min(1024, max(32, 1 << (head_size - 1).bit_length()))
    return block_dim, _gqa_attention_kernel_cache[key]


def _get_bidirectional_gqa_attention_kernel(head_size: int, dtype: type = wp.float16):
    """Return fixed-sequence bidirectional GQA and its tile block dimension."""
    key = (head_size, dtype)
    if key not in _bidirectional_gqa_attention_kernel_cache:
        _bidirectional_gqa_attention_kernel_cache[key] = (
            _create_bidirectional_gqa_attention_kernel(*key)
        )
    block_dim = min(1024, max(32, 1 << (head_size - 1).bit_length()))
    return block_dim, _bidirectional_gqa_attention_kernel_cache[key]


def _get_partitioned_gqa_attention_kernels(
    head_size: int,
    dtype: type = wp.float16,
    partitions: int = 256,
    rows_per_group: int = 1,
    heads_per_group: int = 4,
    mapped: bool = False,
):
    """Return cached partitioned decode attention kernels and their launch dimensions."""
    key = (head_size, dtype, partitions, rows_per_group, heads_per_group, mapped)
    if key not in _partitioned_gqa_attention_kernel_cache:
        _partitioned_gqa_attention_kernel_cache[key] = (
            _create_partitioned_gqa_attention_kernels(*key)
        )
    block_dim = min(1024, max(32, 1 << (head_size - 1).bit_length()))
    return block_dim, partitions, _partitioned_gqa_attention_kernel_cache[key]


@wp.kernel
def _initialize_attention_mask(mask: wp.array2d[wp.int64], length: int):
    """Set the first ``length`` mask entries and clear the remainder."""
    index = wp.tid()
    mask[0, index] = wp.int64(1) if index < length else wp.int64(0)


@wp.kernel
def _set_decode_token(
    input_ids: wp.array2d[wp.int64],
    attention_mask: wp.array2d[wp.int64],
    position_ids: wp.array2d[wp.int64],
    token_id: int,
    position: int,
):
    """Stage one decode token, mask entry, and position entirely on device."""
    input_ids[0, 0] = wp.int64(token_id)
    attention_mask[0, position] = wp.int64(1)
    position_ids[0, 0] = wp.int64(position)


@wp.kernel(enable_backward=False, module="unique")
def _stage_token_position(
    input_ids: wp.array2d[wp.int64],
    position_ids: wp.array2d[wp.int64],
    sequence_end: wp.array1d[wp.int32],
    token_id: int,
    position: int,
):
    """Stage one token, its position, and the inclusive sequence end."""
    input_ids[0, 0] = wp.int64(token_id)
    position_ids[0, 0] = wp.int64(position)
    sequence_end[0] = wp.int32(position)


@lru_cache(maxsize=None)
def _get_stage_decode_batch_kernel(mrope: bool = False):
    """Stage batch tokens with a static standard-RoPE or MRoPE layout."""
    MROPE = mrope

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        input_ids: wp.array2d[wp.int64],
        cache_positions: wp.array1d[wp.int32],
        rope_positions: wp.array2d[wp.int64],
        sequence_end: wp.array1d[wp.int32],
        active: wp.array1d[wp.bool],
        token_ids: wp.array1d[wp.int64],
        positions: wp.array1d[wp.int32],
        rope_deltas: wp.array1d[wp.int32],
    ):
        batch = wp.tid()
        input_ids[0, batch] = token_ids[batch]
        cache_positions[batch] = positions[batch]
        sequence_end[batch] = wp.where(active[batch], positions[batch], wp.int32(-1))
        rope_position = wp.int64(positions[batch] + rope_deltas[batch])
        if wp.static(MROPE):
            rope_positions[0, batch] = rope_position
            rope_positions[1, batch] = rope_position
            rope_positions[2, batch] = rope_position
        else:
            rope_positions[batch, 0] = rope_position

    return kernel


@wp.kernel(enable_backward=False, module="unique")
def _set_sequence_end(sequence_end: wp.array1d[wp.int32], position: int):
    """Update the inclusive device-side sequence end."""
    sequence_end[0] = wp.int32(position)


@wp.kernel
def _initialize_generation_state(
    position: wp.array1d[wp.int32],
    generated_count: wp.array1d[wp.int32],
    finished: wp.array1d[wp.int32],
    prompt_length: int,
):
    """Reset device-side generation counters at ``prompt_length``."""
    position[0] = wp.int32(prompt_length)
    generated_count[0] = wp.int32(0)
    finished[0] = wp.int32(0)


@lru_cache(maxsize=None)
def _get_greedy_argmax_kernels(
    tile_width: int, partial_count: int, dtype: type = wp.float16
):
    """Build hierarchical greedy argmax kernels.
    ``tile_width`` scans logits; ``partial_count`` sizes the final reduction."""
    TILE_WIDTH = tile_width
    PARTIAL_COUNT = partial_count
    DTYPE = dtype

    @wp.func
    def add_offset(index: wp.int32, offset: wp.int32):
        return index + offset

    @wp.func
    def mask_logit(value: DTYPE, index: wp.int32, vocabulary: wp.int32):
        return (
            wp.float32(DTYPE(value))
            if index < vocabulary
            else wp.float32(-3.402823466e38)
        )

    @wp.func
    def matching_token(
        value: wp.float32, token: wp.int32, maximum: wp.float32, vocabulary: wp.int32
    ):
        return token if value == maximum else vocabulary

    @wp.kernel(enable_backward=False, module="unique")
    def partial_argmax(
        logits: wp.array3d(dtype=DTYPE),
        values: wp.array1d[wp.float32],
        tokens: wp.array1d[wp.int32],
    ):
        """Find one deterministic argmax candidate per vocabulary partition."""
        partial = wp.tid()
        vocabulary = logits.shape[2]
        tile_count = (vocabulary + TILE_WIDTH - 1) / TILE_WIDTH
        best_value = wp.float32(-3.402823466e38) + wp.float32(DTYPE(0.0))
        best_token = vocabulary
        local_indices = wp.tile_arange(0, TILE_WIDTH, dtype=wp.int32)
        for tile_id in range(partial, tile_count, PARTIAL_COUNT):
            offset = tile_id * TILE_WIDTH
            indices = wp.tile_map(add_offset, local_indices, offset)
            tile = wp.tile_map(
                mask_logit,
                wp.tile_load(
                    logits[0, logits.shape[1] - 1], shape=TILE_WIDTH, offset=offset
                ),
                indices,
                vocabulary,
            )
            local_token = wp.tile_extract(wp.tile_argmax(tile), 0)
            value = wp.tile_extract(tile, local_token)
            token = offset + local_token
            if value > best_value or (value == best_value and token < best_token):
                best_value = value
                best_token = token
        wp.tile_store(
            values,
            wp.tile_full(shape=1, value=best_value, dtype=wp.float32),
            offset=partial,
        )
        wp.tile_store(
            tokens,
            wp.tile_full(shape=1, value=best_token, dtype=wp.int32),
            offset=partial,
        )

    @wp.func
    def select_token(
        values: wp.array1d[wp.float32], tokens: wp.array1d[wp.int32], vocabulary: int
    ):
        value_tile = wp.tile_load(values, shape=PARTIAL_COUNT)
        token_tile = wp.tile_load(tokens, shape=PARTIAL_COUNT)
        maximum_index = wp.tile_extract(wp.tile_argmax(value_tile), 0)
        maximum = wp.tile_extract(value_tile, maximum_index)
        candidates = wp.tile_map(
            matching_token, value_tile, token_tile, maximum, vocabulary
        )
        winner = wp.tile_extract(wp.tile_argmin(candidates), 0)
        return wp.tile_extract(token_tile, winner)

    @wp.kernel(enable_backward=False, module="unique")
    def store_token(
        values: wp.array1d[wp.float32],
        tokens: wp.array1d[wp.int32],
        output: wp.array1d[wp.int32],
        vocabulary: int,
    ):
        """Reduce partial candidates and store the winning token."""
        wp.tile_store(
            output,
            wp.tile_full(
                shape=1, value=select_token(values, tokens, vocabulary), dtype=wp.int32
            ),
        )

    @wp.kernel(enable_backward=False, module="unique")
    def advance_generation(
        values: wp.array1d[wp.float32],
        tokens: wp.array1d[wp.int32],
        vocabulary: int,
        input_ids: wp.array2d[wp.int64],
        attention_mask: wp.array2d[wp.int64],
        position_ids: wp.array2d[wp.int64],
        position: wp.array1d[wp.int32],
        generated_count: wp.array1d[wp.int32],
        generated_ids: wp.array1d[wp.int64],
        finished: wp.array1d[wp.int32],
        eos_token_id: int,
    ):
        """Select, store, and stage the next token unless generation finished."""
        if finished[0] != 0:
            return
        best_token = select_token(values, tokens, vocabulary)
        count = generated_count[0]
        generated_ids[count] = wp.int64(best_token)
        generated_count[0] = count + 1
        input_ids[0, 0] = wp.int64(best_token)
        token_position = position[0]
        attention_mask[0, token_position] = wp.int64(1)
        position_ids[0, 0] = wp.int64(token_position)
        position[0] = token_position + 1
        if best_token == eos_token_id:
            finished[0] = wp.int32(1)

    return partial_argmax, store_token, advance_generation


@lru_cache(maxsize=None)
def _get_top_k_kernels(tile_width: int, top_k: int, dtype: type):
    """Build an exact hierarchical top-k over the final vocabulary row."""
    TILE_WIDTH = tile_width
    TOP_K = top_k
    MERGE_GROUPS = 16
    MERGE_CANDIDATES = MERGE_GROUPS * top_k
    DTYPE = dtype

    @wp.func
    def add_offset(index: wp.int32, offset: wp.int32):
        return index + offset

    @wp.func
    def mask_logit(value: DTYPE, index: wp.int32, vocabulary: wp.int32):
        return wp.float32(DTYPE(value)) if index < vocabulary else wp.float32(-wp.inf)

    @wp.func
    def mask_token(index: wp.int32, vocabulary: wp.int32):
        return index if index < vocabulary else wp.int32(2147483647)

    @wp.func
    def remove_selected(value: wp.float32, index: wp.int32, selected: wp.int32):
        return wp.float32(-wp.inf) if index == selected else value

    @wp.func
    def remove_selected_token(token: wp.int32, index: wp.int32, selected: wp.int32):
        return wp.int32(2147483647) if index == selected else token

    @wp.func
    def matching_token(value: wp.float32, token: wp.int32, maximum: wp.float32):
        return token if value == maximum else wp.int32(2147483647)

    @wp.func
    def mask_merge_value(value: wp.float32, index: wp.int32, count: wp.int32):
        return value if index < count else wp.float32(-wp.inf)

    @wp.func
    def mask_merge_token(token: wp.int32, index: wp.int32, count: wp.int32):
        return token if index < count else wp.int32(2147483647)

    @wp.kernel(enable_backward=False, module="unique")
    def partial_top_k(
        logits: wp.array3d(dtype=DTYPE),
        values: wp.array1d[wp.float32],
        tokens: wp.array1d[wp.int32],
    ):
        """Extract top-k candidates from one contiguous vocabulary tile."""
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        partial = wp.tid()
        offset = partial * TILE_WIDTH
        vocabulary = logits.shape[2]
        local_indices = wp.tile_arange(0, TILE_WIDTH, dtype=wp.int32)
        indices = wp.tile_map(add_offset, local_indices, offset)
        active_tokens = wp.tile_map(mask_token, indices, vocabulary)
        candidates = wp.tile_map(
            mask_logit,
            wp.tile_load(
                logits[0, logits.shape[1] - 1], shape=TILE_WIDTH, offset=offset
            ),
            indices,
            vocabulary,
        )
        for rank in range(TOP_K):
            maximum_index = wp.tile_extract(wp.tile_argmax(candidates), 0)
            maximum = wp.tile_extract(candidates, maximum_index)
            tied_tokens = wp.tile_map(
                matching_token, candidates, active_tokens, maximum
            )
            local_token = wp.tile_extract(wp.tile_argmin(tied_tokens), 0)
            wp.tile_store(
                values,
                wp.tile_full(
                    shape=1,
                    value=wp.tile_extract(candidates, local_token),
                    dtype=wp.float32,
                ),
                offset=partial * TOP_K + rank,
            )
            wp.tile_store(
                tokens,
                wp.tile_full(
                    shape=1,
                    value=wp.tile_extract(active_tokens, local_token),
                    dtype=wp.int32,
                ),
                offset=partial * TOP_K + rank,
            )
            candidates = wp.tile_map(
                remove_selected, candidates, local_indices, local_token
            )
            active_tokens = wp.tile_map(
                remove_selected_token, active_tokens, local_indices, local_token
            )

    @wp.kernel(enable_backward=False, module="unique")
    def merge_top_k(
        values: wp.array1d[wp.float32],
        tokens: wp.array1d[wp.int32],
        output_values: wp.array1d[wp.float32],
        output_tokens: wp.array1d[wp.int32],
        input_groups: int,
    ):
        """Merge up to 16 adjacent top-k groups into one sorted group."""
        group = wp.tid()
        offset = group * MERGE_CANDIDATES
        local_indices = wp.tile_arange(0, MERGE_CANDIDATES, dtype=wp.int32)
        indices = wp.tile_map(add_offset, local_indices, offset)
        count = input_groups * TOP_K
        candidates = wp.tile_map(
            mask_merge_value,
            wp.tile_load(values, shape=MERGE_CANDIDATES, offset=offset),
            indices,
            count,
        )
        active_tokens = wp.tile_map(
            mask_merge_token,
            wp.tile_load(tokens, shape=MERGE_CANDIDATES, offset=offset),
            indices,
            count,
        )
        for rank in range(TOP_K):
            maximum_index = wp.tile_extract(wp.tile_argmax(candidates), 0)
            maximum = wp.tile_extract(candidates, maximum_index)
            tied_tokens = wp.tile_map(
                matching_token, candidates, active_tokens, maximum
            )
            selected = wp.tile_extract(wp.tile_argmin(tied_tokens), 0)
            output_values[group * TOP_K + rank] = wp.tile_extract(candidates, selected)
            output_tokens[group * TOP_K + rank] = wp.tile_extract(
                active_tokens, selected
            )
            candidates = wp.tile_map(
                remove_selected, candidates, local_indices, selected
            )
            active_tokens = wp.tile_map(
                remove_selected_token, active_tokens, local_indices, selected
            )

    return partial_top_k, merge_top_k


def _array_type(dtype: type, ndim: int):
    return wp.array(dtype=dtype, ndim=ndim)


_KERNEL_OVERLOADS: dict[tuple[Any, ...], Any] = {}


def _kernel_for_dtype(kernel, dtype: type, *parameter_types: type | tuple[int]):
    """Return one cached specialization of a generic same-dtype kernel."""
    key = (kernel, dtype, parameter_types)
    if key not in _KERNEL_OVERLOADS:
        signature = [
            _array_type(dtype, item[0]) if isinstance(item, tuple) else item
            for item in parameter_types
        ]
        _KERNEL_OVERLOADS[key] = wp.overload(kernel, signature)
    return _KERNEL_OVERLOADS[key]


def _clamp_kernel_for_dtype(dtype: type):
    return _kernel_for_dtype(_clamp_kernel, dtype, (1,), (1,), wp.float32, wp.float32)


def _cast_kernel_for_dtypes(source_dtype: type, target_dtype: type):
    key = (_cast_kernel, source_dtype, target_dtype)
    if key not in _KERNEL_OVERLOADS:
        _KERNEL_OVERLOADS[key] = wp.overload(
            _cast_kernel,
            [wp.array1d(dtype=source_dtype), wp.array1d(dtype=target_dtype)],
        )
    return _KERNEL_OVERLOADS[key]


@lru_cache(maxsize=None)
def _seeded_normal_kernel(dtype: type):
    """Return deterministic independent standard-normal filling for one dtype."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def fill(output: wp.array1d(dtype=DTYPE), seed: int):
        index = wp.tid()
        state = wp.rand_init(seed, index)
        output[index] = DTYPE(wp.randn(state))

    return fill


@lru_cache(maxsize=None)
def _temporal_conv2d_slice_kernel(dtype: type):
    """Extract one OITHW temporal plane into a contiguous OIHW weight."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def extract(
        source: wp.array1d(dtype=DTYPE),
        output: wp.array4d(dtype=DTYPE),
        temporal_size: int,
        temporal_index: int,
    ):
        out_channel, in_channel, row, column = wp.tid()
        source_index = (
            (
                (
                    (out_channel * output.shape[1] + in_channel) * temporal_size
                    + temporal_index
                )
                * output.shape[2]
                + row
            )
            * output.shape[3]
        ) + column
        output[out_channel, in_channel, row, column] = DTYPE(source[source_index])

    return extract


@lru_cache(maxsize=None)
def _true_cfg_kernel(dtype: type, width: int):
    """Fuse true-CFG combination and positive-prediction norm rescaling."""
    DTYPE = dtype
    WIDTH = int(width)

    @wp.func
    def square(value: dtype):
        value_fp32 = wp.float32(dtype(value))
        return value_fp32 * value_fp32

    @wp.func
    def combine(positive: dtype, negative: dtype, scale: wp.float32):
        return wp.float32(dtype(negative)) + scale * (
            wp.float32(dtype(positive)) - wp.float32(dtype(negative))
        )

    @wp.func
    def rescale(value: wp.float32, ratio: wp.float32):
        return dtype(value * ratio)

    @wp.kernel(enable_backward=False, module="unique")
    def true_cfg(
        positive: wp.array3d(dtype=DTYPE),
        negative: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        scale: wp.float32,
    ):
        row = wp.tid()
        batch = row / positive.shape[1]
        token = row % positive.shape[1]
        positive_values = wp.tile_load(positive[batch, token], shape=(WIDTH,))
        negative_values = wp.tile_load(negative[batch, token], shape=(WIDTH,))
        combined = wp.tile_map(combine, positive_values, negative_values, scale)
        positive_norm = wp.sqrt(
            wp.tile_extract(wp.tile_sum(wp.tile_map(square, positive_values)), 0)
        )
        combined_norm = wp.sqrt(wp.tile_extract(wp.tile_sum(combined * combined), 0))
        ratio = positive_norm / wp.max(
            combined_norm, wp.float32(1.0e-20) + wp.float32(DTYPE(0.0))
        )
        wp.tile_store(output[batch, token], wp.tile_map(rescale, combined, ratio))

    return true_cfg


@lru_cache(maxsize=None)
def _spatial_diffusion_kernels(dtype: type):
    """Reusable spatial patch, channel-affine, and Euler update kernels."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def pack_patches(
        x: wp.array4d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        patch_size: int,
    ):
        batch, token, packed_channel = wp.tid()
        patch_area = patch_size * patch_size
        channel = packed_channel / patch_area
        patch_index = packed_channel % patch_area
        patch_y = patch_index / patch_size
        patch_x = patch_index % patch_size
        grid_width = x.shape[3] / patch_size
        grid_y = token / grid_width
        grid_x = token % grid_width
        output[batch, token, packed_channel] = DTYPE(
            wp.float32(
                x[
                    batch,
                    channel,
                    grid_y * patch_size + patch_y,
                    grid_x * patch_size + patch_x,
                ]
            )
        )

    @wp.kernel(enable_backward=False, module="unique")
    def unpack_patches(
        x: wp.array3d(dtype=DTYPE),
        output: wp.array4d(dtype=DTYPE),
        patch_size: int,
    ):
        batch, channel, row, column = wp.tid()
        grid_width = output.shape[3] / patch_size
        grid_y = row / patch_size
        grid_x = column / patch_size
        patch_y = row % patch_size
        patch_x = column % patch_size
        packed_channel = (
            channel * patch_size * patch_size + patch_y * patch_size + patch_x
        )
        output[batch, channel, row, column] = DTYPE(
            wp.float32(x[batch, grid_y * grid_width + grid_x, packed_channel])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def channel_affine(
        x: wp.array4d(dtype=DTYPE),
        scale: wp.array1d(dtype=wp.float32),
        bias: wp.array1d(dtype=wp.float32),
        output: wp.array4d(dtype=DTYPE),
    ):
        batch, channel, row, column = wp.tid()
        output[batch, channel, row, column] = DTYPE(
            wp.float32(x[batch, channel, row, column]) * wp.float32(scale[channel])
            + wp.float32(bias[channel])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def flow_euler_step(
        sample: wp.array3d(dtype=DTYPE),
        velocity: wp.array3d(dtype=DTYPE),
        sigma: wp.array1d(dtype=wp.float32),
        next_sigma: wp.array1d(dtype=wp.float32),
    ):
        batch, token, channel = wp.tid()
        dt = wp.float32(next_sigma[batch]) - wp.float32(sigma[batch])
        sample[batch, token, channel] = DTYPE(
            wp.float32(sample[batch, token, channel])
            + dt * wp.float32(velocity[batch, token, channel])
        )

    return pack_patches, unpack_patches, channel_affine, flow_euler_step


@lru_cache(maxsize=None)
def _merge_lora_kernel(dtype: type):
    """Return an in-place LoRA merge kernel specialized for one weight dtype."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def merge_lora(
        weight: wp.array2d(dtype=DTYPE),
        a: wp.array2d(dtype=DTYPE),
        b: wp.array2d(dtype=DTYPE),
        scale: wp.float32,
    ):
        row, column = wp.tid()
        delta = wp.float32(0.0)
        for rank in range(a.shape[0]):
            delta += wp.float32(b[row, rank]) * wp.float32(a[rank, column])
        weight[row, column] = DTYPE(wp.float32(weight[row, column]) + scale * delta)

    return merge_lora


def _where_kernel_for_dtype(dtype: type):
    key = (_where_broadcast_kernel, dtype)
    if key not in _KERNEL_OVERLOADS:
        _KERNEL_OVERLOADS[key] = wp.overload(
            _where_broadcast_kernel,
            [
                wp.array2d(dtype=wp.bool),
                wp.array2d(dtype=dtype),
                wp.array2d(dtype=dtype),
                wp.array2d(dtype=dtype),
            ],
        )
    return _KERNEL_OVERLOADS[key]


def _rotary_embedding_kernel_for_dtype(dtype: type):
    key = (_rotary_embedding_kernel, dtype)
    if key not in _KERNEL_OVERLOADS:
        _KERNEL_OVERLOADS[key] = wp.overload(
            _rotary_embedding_kernel,
            [
                wp.array4d(dtype=dtype),
                wp.array2d(dtype=wp.int64),
                wp.array2d(dtype=dtype),
                wp.array2d(dtype=dtype),
                wp.array4d(dtype=dtype),
                int,
                bool,
                bool,
            ],
        )
    return _KERNEL_OVERLOADS[key]


@lru_cache(maxsize=None)
def _get_mrope_decode_batch_kernel(dtype: type):
    """Apply Qwen MRoPE to batch-major one-token heads."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        x: wp.array4d(dtype=DTYPE),
        position_ids: wp.array2d[wp.int64],
        cos_cache: wp.array2d(dtype=DTYPE),
        sin_cache: wp.array2d(dtype=DTYPE),
        output: wp.array4d(dtype=DTYPE),
        rotary_dim: int,
    ):
        batch, head, column = wp.tid()
        if column >= rotary_dim:
            output[batch, head, 0, column] = x[batch, head, 0, column]
            return
        half = rotary_dim / 2
        cache_column = column % half
        partner = column + half if column < half else column - half
        sign = wp.float32(-1.0) if column < half else wp.float32(1.0)
        position = position_ids[cache_column % 3, batch]
        value = wp.float32(x[batch, head, 0, column])
        rotated = sign * wp.float32(x[batch, head, 0, partner])
        output[batch, head, 0, column] = DTYPE(
            value * wp.float32(cos_cache[position, cache_column])
            + rotated * wp.float32(sin_cache[position, cache_column])
        )

    return kernel


@lru_cache(maxsize=None)
def _get_mrope_embedding_kernel(dtype: type):
    """Apply split-half Qwen MRoPE using explicit temporal/height/width positions."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        x: wp.array4d(dtype=DTYPE),
        position_ids: wp.array2d[wp.int64],
        cos_cache: wp.array2d(dtype=DTYPE),
        sin_cache: wp.array2d(dtype=DTYPE),
        output: wp.array4d(dtype=DTYPE),
        rotary_dim: int,
    ):
        batch, head, sequence, column = wp.tid()
        if column >= rotary_dim:
            output[batch, head, sequence, column] = x[batch, head, sequence, column]
            return
        half = rotary_dim / 2
        cache_column = column % half
        partner = column + half if column < half else column - half
        sign = wp.float32(-1.0) if column < half else wp.float32(1.0)
        # Qwen3.8 sections [11,11,10] are interleaved T,H,W over 32 frequencies.
        residue = cache_column % 3
        axis = residue
        position = position_ids[axis, sequence]
        value = wp.float32(x[batch, head, sequence, column])
        rotated = sign * wp.float32(x[batch, head, sequence, partner])
        output[batch, head, sequence, column] = DTYPE(
            value * wp.float32(cos_cache[position, cache_column])
            + rotated * wp.float32(sin_cache[position, cache_column])
        )

    return kernel


@wp.kernel(enable_backward=False, module="unique")
def _overlay_embedding_rows_kernel(
    embedding: wp.array2d[Any],
    visual: wp.array2d[Any],
    source_indices: wp.array1d[wp.int32],
):
    row, column = wp.tid()
    source = source_indices[row]
    if source >= 0:
        embedding[row, column] = visual[source, column]


@wp.kernel(enable_backward=False, module="unique")
def _stage_mrope_token_position(
    input_ids: wp.array2d[wp.int64],
    cache_positions: wp.array2d[wp.int64],
    rope_positions: wp.array2d[wp.int64],
    sequence_end: wp.array1d[wp.int32],
    token_id: int,
    cache_position: int,
    rope_position: int,
):
    input_ids[0, 0] = wp.int64(token_id)
    cache_positions[0, 0] = wp.int64(cache_position)
    rope_positions[0, 0] = wp.int64(rope_position)
    rope_positions[1, 0] = wp.int64(rope_position)
    rope_positions[2, 0] = wp.int64(rope_position)
    sequence_end[0] = cache_position


@lru_cache(maxsize=None)
def _encoder_kernels(dtype: type, head_size: int):
    DTYPE = dtype
    HEAD_SIZE = head_size

    @wp.kernel(enable_backward=False, module="unique")
    def add_bias(x: wp.array2d(dtype=DTYPE), bias: wp.array1d(dtype=DTYPE)):
        row, column = wp.tid()
        x[row, column] = DTYPE(wp.float32(x[row, column]) + wp.float32(bias[column]))

    @wp.kernel(enable_backward=False, module="unique")
    def bias_gelu(x: wp.array2d(dtype=DTYPE), bias: wp.array1d(dtype=DTYPE)):
        row, column = wp.tid()
        value = wp.float32(x[row, column]) + wp.float32(bias[column])
        # Exact PyTorch GELU default (erf), rather than the tanh approximation.
        value *= wp.float32(0.5) * (
            wp.float32(1.0) + wp.erf(value * wp.float32(0.7071067811865476))
        )
        x[row, column] = DTYPE(value)

    @wp.kernel(enable_backward=False, module="unique")
    def residual_layer_norm(
        branch: wp.array2d(dtype=DTYPE),
        residual: wp.array2d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        scale: wp.array1d(dtype=DTYPE),
        shift: wp.array1d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        epsilon: wp.float32,
    ):
        row = wp.tid()
        width = branch.shape[1]
        mean = wp.float32(0.0)
        for column in range(width):
            mean += (
                wp.float32(branch[row, column])
                + wp.float32(bias[column])
                + wp.float32(residual[row, column])
            )
        mean /= wp.float32(width)
        variance = wp.float32(0.0)
        for column in range(width):
            value = (
                wp.float32(branch[row, column])
                + wp.float32(bias[column])
                + wp.float32(residual[row, column])
                - mean
            )
            variance += value * value
        inverse = wp.float32(1.0) / wp.sqrt(variance / wp.float32(width) + epsilon)
        for column in range(width):
            value = (
                wp.float32(branch[row, column])
                + wp.float32(bias[column])
                + wp.float32(residual[row, column])
            )
            output[row, column] = DTYPE(
                (value - mean) * inverse * wp.float32(scale[column])
                + wp.float32(shift[column])
            )

    @wp.kernel(enable_backward=False, module="unique")
    def split_qkv(
        packed: wp.array2d(dtype=DTYPE),
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
    ):
        batch, head, token, column = wp.tid()
        hidden = query.shape[1] * query.shape[3]
        row = batch * query.shape[2] + token
        offset = head * query.shape[3] + column
        query[batch, head, token, column] = DTYPE(packed[row, offset])
        key[batch, head, token, column] = DTYPE(packed[row, hidden + offset])
        value[batch, head, token, column] = DTYPE(packed[row, hidden * 2 + offset])

    @wp.kernel(enable_backward=False, module="unique")
    def merge_heads(x: wp.array4d(dtype=DTYPE), output: wp.array2d(dtype=DTYPE)):
        batch, head, token, column = wp.tid()
        output[batch * x.shape[2] + token, head * x.shape[3] + column] = DTYPE(
            x[batch, head, token, column]
        )

    @wp.func
    def dot(left: DTYPE, right: DTYPE):
        return wp.float32(DTYPE(left)) * wp.float32(DTYPE(right))

    @wp.func
    def update(
        total: wp.float32,
        current: DTYPE,
        old_scale: wp.float32,
        probability: wp.float32,
    ):
        return total * old_scale + wp.float32(DTYPE(current)) * probability

    @wp.func
    def normalize(total: wp.float32, denominator: wp.float32):
        return total / denominator

    @wp.kernel(enable_backward=False, module="unique")
    def full_attention(
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
        valid: wp.array2d(dtype=wp.bool),
        output: wp.array4d(dtype=DTYPE),
        scale: wp.float32,
    ):
        item = wp.tid()
        sequence = query.shape[2]
        token = item % sequence
        head = (item / sequence) % query.shape[1]
        batch = item / (query.shape[1] * sequence)
        kv_head = head / (query.shape[1] / key.shape[1])
        accumulator = wp.tile_zeros(shape=(HEAD_SIZE,), dtype=wp.float32)
        q = wp.tile_load(query[batch, head, token], shape=(HEAD_SIZE,))
        maximum = wp.float32(-3.402823466e38)
        denominator = wp.float32(0.0)
        for source in range(sequence):
            if valid[batch, source]:
                k = wp.tile_load(key[batch, kv_head, source], shape=(HEAD_SIZE,))
                score = wp.tile_extract(wp.tile_sum(wp.tile_map(dot, q, k)), 0) * scale
                new_maximum = wp.max(maximum, score)
                old_scale = wp.exp(maximum - new_maximum)
                probability = wp.exp(score - new_maximum)
                denominator = denominator * old_scale + probability
                v = wp.tile_load(value[batch, kv_head, source], shape=(HEAD_SIZE,))
                accumulator = wp.tile_map(
                    update, accumulator, v, old_scale, probability
                )
                maximum = new_maximum
        accumulator = wp.tile_map(normalize, accumulator, denominator)
        wp.tile_store(
            output[batch, head, token], wp.tile_astype(accumulator, dtype=DTYPE)
        )

    return (
        add_bias,
        bias_gelu,
        residual_layer_norm,
        split_qkv,
        merge_heads,
        full_attention,
    )


@lru_cache(maxsize=None)
def _channels_last_1d_kernels(dtype: type):
    """Return reusable general channels-last 1D kernels for one compute dtype."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def conv1d_nlc(
        x: wp.array3d(dtype=DTYPE),
        weight: wp.array3d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        stride: int,
        padding: int,
        dilation: int,
        use_bias: bool,
    ):
        batch, position, out_channel = wp.tid()
        total = wp.float32(0.0)
        if use_bias:
            total = wp.float32(bias[out_channel])
        for kernel_index in range(weight.shape[2]):
            source = position * stride - padding + kernel_index * dilation
            if source >= 0 and source < x.shape[1]:
                for in_channel in range(x.shape[2]):
                    total += wp.float32(x[batch, source, in_channel]) * wp.float32(
                        weight[out_channel, in_channel, kernel_index]
                    )
        output[batch, position, out_channel] = DTYPE(total)

    @wp.kernel(enable_backward=False, module="unique")
    def conv_transpose1d_nlc(
        x: wp.array3d(dtype=DTYPE),
        weight: wp.array3d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        stride: int,
        padding: int,
        dilation: int,
        use_bias: bool,
    ):
        batch, position, out_channel = wp.tid()
        total = wp.float32(0.0)
        if use_bias:
            total = wp.float32(bias[out_channel])
        for kernel_index in range(weight.shape[2]):
            numerator = position + padding - kernel_index * dilation
            if numerator >= 0 and numerator % stride == 0:
                source = numerator / stride
                if source < x.shape[1]:
                    for in_channel in range(x.shape[2]):
                        total += wp.float32(x[batch, source, in_channel]) * wp.float32(
                            weight[in_channel, out_channel, kernel_index]
                        )
        output[batch, position, out_channel] = DTYPE(total)

    @wp.kernel(enable_backward=False, module="unique")
    def snake1d(
        x: wp.array3d(dtype=DTYPE),
        alpha: wp.array1d(dtype=DTYPE),
        beta: wp.array1d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        logscale: bool,
    ):
        batch, position, channel = wp.tid()
        value = wp.float32(x[batch, position, channel])
        alpha_value = wp.float32(alpha[channel])
        beta_value = wp.float32(beta[channel])
        if logscale:
            alpha_value = wp.exp(alpha_value)
            beta_value = wp.exp(beta_value)
        periodic = wp.sin(alpha_value * value)
        output[batch, position, channel] = DTYPE(
            value + periodic * periodic / (beta_value + wp.float32(1.0e-9))
        )

    return conv1d_nlc, conv_transpose1d_nlc, snake1d


@lru_cache(maxsize=None)
def _spatial_vae_kernels(dtype: type, channels: int):
    """Return reusable channels-last VAE normalization and layout kernels."""
    DTYPE = dtype
    CHANNELS = int(channels)

    @wp.func
    def square(value: dtype):
        value_fp32 = wp.float32(dtype(value))
        return value_fp32 * value_fp32

    @wp.func
    def normalize(value: dtype, gamma: dtype, inverse_rms: wp.float32, activate: int):
        result = wp.float32(value) * wp.float32(gamma) * inverse_rms
        if activate != 0:
            result = result / (wp.float32(1.0) + wp.exp(-result))
        return dtype(result)

    @wp.kernel(enable_backward=False, module="unique")
    def rms_norm(
        x: wp.array4d(dtype=DTYPE),
        gamma: wp.array1d(dtype=DTYPE),
        output: wp.array4d(dtype=DTYPE),
        epsilon: wp.float32,
        activate: int,
    ):
        pixel = wp.tid()
        image_area = x.shape[1] * x.shape[2]
        batch = pixel / image_area
        position = pixel % image_area
        row = position / x.shape[2]
        column = position % x.shape[2]
        values = wp.tile_load(x[batch, row, column], shape=(CHANNELS,))
        scales = wp.tile_load(gamma, shape=(CHANNELS,))
        inverse_rms = wp.float32(1.0) / wp.sqrt(
            wp.tile_extract(wp.tile_sum(wp.tile_map(square, values)), 0)
            / wp.float32(CHANNELS)
            + epsilon
            + wp.float32(DTYPE(0.0))
        )
        wp.tile_store(
            output[batch, row, column],
            wp.tile_map(normalize, values, scales, inverse_rms, activate),
        )

    @wp.kernel(enable_backward=False, module="unique")
    def nearest_upsample(
        x: wp.array4d(dtype=DTYPE), output: wp.array4d(dtype=DTYPE), scale: int
    ):
        batch, row, column, channel = wp.tid()
        output[batch, row, column, channel] = DTYPE(
            x[batch, row / scale, column / scale, channel]
        )

    @wp.kernel(enable_backward=False, module="unique")
    def residual_add(
        left: wp.array4d(dtype=DTYPE),
        right: wp.array4d(dtype=DTYPE),
        output: wp.array4d(dtype=DTYPE),
    ):
        batch, row, column, channel = wp.tid()
        output[batch, row, column, channel] = DTYPE(
            wp.float32(left[batch, row, column, channel])
            + wp.float32(right[batch, row, column, channel])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def split_qkv(
        packed: wp.array4d(dtype=DTYPE),
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
    ):
        batch, token, channel = wp.tid()
        row = token / packed.shape[2]
        column = token % packed.shape[2]
        query[batch, 0, token, channel] = DTYPE(packed[batch, row, column, channel])
        key[batch, 0, token, channel] = DTYPE(
            packed[batch, row, column, channel + CHANNELS]
        )
        value[batch, 0, token, channel] = DTYPE(
            packed[batch, row, column, channel + CHANNELS * 2]
        )

    @wp.kernel(enable_backward=False, module="unique")
    def merge_attention(x: wp.array4d(dtype=DTYPE), output: wp.array4d(dtype=DTYPE)):
        batch, token, channel = wp.tid()
        row = token / output.shape[2]
        column = token % output.shape[2]
        output[batch, row, column, channel] = DTYPE(x[batch, 0, token, channel])

    return rms_norm, nearest_upsample, residual_add, split_qkv, merge_attention


@lru_cache(maxsize=None)
def _channels_last_2d_kernels(dtype: type):
    """Return a reusable general channels-last Conv2D fallback."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def conv2d_nhwc(
        x: wp.array4d(dtype=DTYPE),
        weight: wp.array4d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        output: wp.array4d(dtype=DTYPE),
        stride_y: int,
        stride_x: int,
        padding_top: int,
        padding_left: int,
        dilation_y: int,
        dilation_x: int,
        use_bias: bool,
    ):
        batch, row, column, out_channel = wp.tid()
        total = wp.float32(0.0)
        if use_bias:
            total = wp.float32(bias[out_channel])
        for kernel_y in range(weight.shape[2]):
            source_y = row * stride_y - padding_top + kernel_y * dilation_y
            if source_y >= 0 and source_y < x.shape[1]:
                for kernel_x in range(weight.shape[3]):
                    source_x = column * stride_x - padding_left + kernel_x * dilation_x
                    if source_x >= 0 and source_x < x.shape[2]:
                        for in_channel in range(x.shape[3]):
                            total += wp.float32(
                                x[batch, source_y, source_x, in_channel]
                            ) * wp.float32(
                                weight[out_channel, in_channel, kernel_y, kernel_x]
                            )
        output[batch, row, column, out_channel] = DTYPE(total)

    return (conv2d_nhwc,)


@lru_cache(maxsize=None)
def _conv2d_mma_kernels(
    dtype: type,
    kernel_height: int,
    kernel_width: int,
    tile_m: int = 16,
    tile_n: int = 32,
):
    """Return packed-weight tensor-core NHWC Conv2D kernels and edge fallback."""
    DTYPE = dtype
    KERNEL_HEIGHT = kernel_height
    KERNEL_WIDTH = kernel_width
    TILE_M = tile_m
    TILE_N = tile_n
    TILE_K = 16

    @wp.kernel(enable_backward=False, module="unique")
    def pack_weight(
        weight: wp.array4d(dtype=DTYPE),
        packed: wp.array4d(dtype=DTYPE),
    ):
        out_channel, in_channel, kernel_y, kernel_x = wp.tid()
        packed[kernel_y, kernel_x, out_channel, in_channel] = DTYPE(
            weight[out_channel, in_channel, kernel_y, kernel_x]
        )

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def interior(
        x: wp.array4d(dtype=DTYPE),
        packed_weight: wp.array4d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        output: wp.array4d(dtype=DTYPE),
        first_x_tile: int,
        first_y: int,
        padding_top: int,
        padding_left: int,
        use_bias: bool,
    ):
        x_tile, y_offset, batch_channel_tile = wp.tid()
        out_channel_tiles = output.shape[3] / TILE_N
        out_channel_tile = batch_channel_tile % out_channel_tiles
        batch = batch_channel_tile / out_channel_tiles
        output_x = (x_tile + first_x_tile) * TILE_M
        output_y = y_offset + first_y
        accumulator = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
        for kernel_y in range(KERNEL_HEIGHT):
            source_y = output_y - padding_top + kernel_y
            for kernel_x in range(KERNEL_WIDTH):
                source_x = output_x - padding_left + kernel_x
                for inner_tile in range((x.shape[3] + TILE_K - 1) / TILE_K):
                    inner = inner_tile * TILE_K
                    activation = wp.tile_load(
                        x[batch, source_y],
                        shape=(TILE_M, TILE_K),
                        offset=(source_x, inner),
                    )
                    weights = wp.tile_load(
                        packed_weight[kernel_y, kernel_x],
                        shape=(TILE_N, TILE_K),
                        offset=(out_channel_tile * TILE_N, inner),
                    )
                    wp.tile_matmul(activation, wp.tile_transpose(weights), accumulator)
        if use_bias:
            tiled_bias = wp.tile_load(
                bias, shape=(TILE_N,), offset=(out_channel_tile * TILE_N,)
            )
            accumulator += wp.tile_astype(
                wp.tile_broadcast(tiled_bias, shape=(TILE_M, TILE_N)),
                dtype=wp.float32,
            )
        wp.tile_store(
            output[batch, output_y],
            wp.tile_astype(accumulator, dtype=DTYPE),
            offset=(output_x, out_channel_tile * TILE_N),
        )

    @wp.kernel(enable_backward=False, module="unique")
    def boundary(
        x: wp.array4d(dtype=DTYPE),
        packed_weight: wp.array4d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        output: wp.array4d(dtype=DTYPE),
        interior_x_begin: int,
        interior_x_end: int,
        interior_y_begin: int,
        interior_y_end: int,
        padding_top: int,
        padding_left: int,
        use_bias: bool,
    ):
        batch, boundary_position, out_channel = wp.tid()
        width = output.shape[2]
        top_count = interior_y_begin * width
        bottom_count = (output.shape[1] - interior_y_end) * width
        if boundary_position < top_count:
            row = boundary_position / width
            column = boundary_position % width
        elif boundary_position < top_count + bottom_count:
            offset = boundary_position - top_count
            row = interior_y_end + offset / width
            column = offset % width
        else:
            offset = boundary_position - top_count - bottom_count
            edge_width = interior_x_begin + output.shape[2] - interior_x_end
            row = interior_y_begin + offset / edge_width
            edge_column = offset % edge_width
            column = edge_column
            if edge_column >= interior_x_begin:
                column = interior_x_end + edge_column - interior_x_begin
        total = wp.float32(0.0)
        if use_bias:
            total = wp.float32(bias[out_channel])
        for kernel_y in range(KERNEL_HEIGHT):
            source_y = row - padding_top + kernel_y
            if source_y >= 0 and source_y < x.shape[1]:
                for kernel_x in range(KERNEL_WIDTH):
                    source_x = column - padding_left + kernel_x
                    if source_x >= 0 and source_x < x.shape[2]:
                        for in_channel in range(x.shape[3]):
                            total += wp.float32(
                                x[batch, source_y, source_x, in_channel]
                            ) * wp.float32(
                                packed_weight[
                                    kernel_y, kernel_x, out_channel, in_channel
                                ]
                            )
        output[batch, row, column, out_channel] = DTYPE(total)

    for kernel in (pack_weight, interior, boundary):
        kernel.module.options["enable_backward"] = False
    return pack_weight, interior, boundary


@lru_cache(maxsize=None)
def _conv1d_mma_kernels(
    dtype: type, kernel_size: int, tile_m: int = 16, tile_n: int = 32
):
    """Return packed-weight tensor-core Conv1D kernels and edge fallback."""
    DTYPE = dtype
    KERNEL_SIZE = kernel_size
    TILE_M = tile_m
    TILE_N = tile_n
    TILE_K = 16

    @wp.kernel(enable_backward=False, module="unique")
    def pack_weight(weight: wp.array3d(dtype=DTYPE), packed: wp.array3d(dtype=DTYPE)):
        out_channel, in_channel, tap = wp.tid()
        packed[tap, out_channel, in_channel] = DTYPE(
            weight[out_channel, in_channel, tap]
        )

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def interior(
        x: wp.array3d(dtype=DTYPE),
        packed_weight: wp.array3d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        first_tile: int,
        padding: int,
        dilation: int,
        use_bias: bool,
    ):
        tile_row, tile_column, batch = wp.tid()
        output_row = (tile_row + first_tile) * TILE_M
        accumulator = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
        for tap in range(KERNEL_SIZE):
            source_row = output_row - padding + tap * dilation
            for inner_tile in range((x.shape[2] + TILE_K - 1) / TILE_K):
                inner = inner_tile * TILE_K
                activation = wp.tile_load(
                    x[batch], shape=(TILE_M, TILE_K), offset=(source_row, inner)
                )
                weights = wp.tile_load(
                    packed_weight[tap],
                    shape=(TILE_N, TILE_K),
                    offset=(tile_column * TILE_N, inner),
                )
                wp.tile_matmul(activation, wp.tile_transpose(weights), accumulator)
        if use_bias:
            tiled_bias = wp.tile_load(
                bias, shape=(TILE_N,), offset=(tile_column * TILE_N,)
            )
            accumulator += wp.tile_astype(
                wp.tile_broadcast(tiled_bias, shape=(TILE_M, TILE_N)),
                dtype=wp.float32,
            )
        wp.tile_store(
            output[batch],
            wp.tile_astype(accumulator, dtype=DTYPE),
            offset=(output_row, tile_column * TILE_N),
        )

    @wp.kernel(enable_backward=False, module="unique")
    def boundary(
        x: wp.array3d(dtype=DTYPE),
        packed_weight: wp.array3d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        interior_begin: int,
        interior_end: int,
        padding: int,
        dilation: int,
        use_bias: bool,
    ):
        batch, boundary_position, out_channel = wp.tid()
        position = boundary_position
        if boundary_position >= interior_begin:
            position = interior_end + boundary_position - interior_begin
        total = wp.float32(0.0)
        if use_bias:
            total = wp.float32(bias[out_channel])
        for tap in range(KERNEL_SIZE):
            source = position - padding + tap * dilation
            if source >= 0 and source < x.shape[1]:
                for in_channel in range(x.shape[2]):
                    total += wp.float32(x[batch, source, in_channel]) * wp.float32(
                        packed_weight[tap, out_channel, in_channel]
                    )
        output[batch, position, out_channel] = DTYPE(total)

    for kernel in (pack_weight, interior, boundary):
        kernel.module.options["enable_backward"] = False
    return pack_weight, interior, boundary


@lru_cache(maxsize=None)
def _conv_transpose1d_mma_kernels(
    dtype: type, kernel_size: int, tile_m: int = 16, tile_n: int = 32
):
    """Return tensor-core ConvTranspose1D kernels using residue-class GEMMs."""
    DTYPE = dtype
    KERNEL_SIZE = kernel_size
    TILE_M = tile_m
    TILE_N = tile_n
    TILE_K = 16

    @wp.kernel(enable_backward=False, module="unique")
    def pack_weight(weight: wp.array3d(dtype=DTYPE), packed: wp.array3d(dtype=DTYPE)):
        in_channel, out_channel, tap = wp.tid()
        packed[tap, out_channel, in_channel] = DTYPE(
            weight[in_channel, out_channel, tap]
        )

    @wp.kernel(enable_backward=False, module="unique")
    def pack_input(x: wp.array3d(dtype=DTYPE), padded: wp.array3d(dtype=DTYPE)):
        batch, position, channel = wp.tid()
        padded[batch, position + 1, channel] = DTYPE(x[batch, position, channel])

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def project(
        padded_input: wp.array3d(dtype=DTYPE),
        packed_weight: wp.array3d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        scratch: wp.array4d(dtype=DTYPE),
        stride: int,
        padding: int,
        use_bias: bool,
    ):
        tile_row, tile_column, batch_residue = wp.tid()
        residue = batch_residue % stride
        batch = batch_residue / stride
        quotient = tile_row * TILE_M
        accumulator = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
        tap = (residue + padding) % stride
        while tap < KERNEL_SIZE:
            source_row = quotient + (residue + padding - tap) / stride + 1
            for inner_tile in range((padded_input.shape[2] + TILE_K - 1) / TILE_K):
                inner = inner_tile * TILE_K
                activation = wp.tile_load(
                    padded_input[batch],
                    shape=(TILE_M, TILE_K),
                    offset=(source_row, inner),
                )
                weights = wp.tile_load(
                    packed_weight[tap],
                    shape=(TILE_N, TILE_K),
                    offset=(tile_column * TILE_N, inner),
                )
                wp.tile_matmul(activation, wp.tile_transpose(weights), accumulator)
            tap += stride
        if use_bias:
            tiled_bias = wp.tile_load(
                bias, shape=(TILE_N,), offset=(tile_column * TILE_N,)
            )
            accumulator += wp.tile_astype(
                wp.tile_broadcast(tiled_bias, shape=(TILE_M, TILE_N)),
                dtype=wp.float32,
            )
        wp.tile_store(
            scratch[batch, residue],
            wp.tile_astype(accumulator, dtype=DTYPE),
            offset=(quotient, tile_column * TILE_N),
        )

    @wp.kernel(enable_backward=False, module="unique")
    def unpack(scratch: wp.array4d(dtype=DTYPE), output: wp.array3d(dtype=DTYPE)):
        batch, position, channel = wp.tid()
        stride = scratch.shape[1]
        output[batch, position, channel] = DTYPE(
            scratch[batch, position % stride, position / stride, channel]
        )

    for kernel in (pack_weight, pack_input, project, unpack):
        kernel.module.options["enable_backward"] = False
    return pack_weight, pack_input, project, unpack
