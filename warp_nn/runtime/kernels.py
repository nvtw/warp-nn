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

from typing import Any

from functools import lru_cache

import warp as wp

from warp_nn.modules.layers._common import tile_transposed_gemm_2d
from warp_nn.runtime._cuda import dp4a, expand_int4x4_high, expand_int4x4_low, subgroup_sum, warp_max_broadcast
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
        tiled_bias = wp.tile_broadcast(wp.tile_load(bias, shape=shape_b, offset=offset_b), shape=shape_t)
        wp.tile_store(C, wp.tile_transpose(alpha * out + beta * tiled_bias), offset=offset)

    return kernel


_GEMM_CONFIG = get_kernel_config()
_GEMM_TRANSB_TILED_KERNEL = _create_gemm_transb_tiled_kernel(_GEMM_CONFIG)


@wp.kernel
def _linear_kernel(x: wp.array2d[Any], weight: wp.array2d[Any], output: wp.array2d[Any]):
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
    def kernel(x: wp.array2d(dtype=DTYPE), weight: wp.array2d(dtype=DTYPE), output: wp.array2d(dtype=DTYPE)):
        """Project small row batches with one reduction tile per output."""
        item = wp.tid()
        row = item / weight.shape[0]
        column = item % weight.shape[0]
        partials = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for inner_tile in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = inner_tile * TILE_WIDTH
            activations = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            weights = wp.tile_load(weight[column], shape=(TILE_WIDTH,), offset=(offset,))
            partials += wp.tile_map(multiply, activations, weights)
        output[row, column] = DTYPE(wp.tile_extract(wp.tile_sum(partials), 0))

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_linear_vector_kernel(dtype: type):
    """Return a cached small-batch dense projection kernel."""
    return _create_linear_vector_kernel(dtype)


def _create_linear_tiled_kernel(dtype: type, tile_m: int):
    """Build a typed tensor-core-friendly dense projection kernel."""
    DTYPE = dtype
    TILE_M = tile_m
    TILE_N = 32
    TILE_K = 32

    @wp.func
    def cast_output(value: wp.float32):
        return DTYPE(value)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(x: wp.array2d(dtype=DTYPE), weight: wp.array2d(dtype=DTYPE), output: wp.array2d(dtype=DTYPE)):
        """Compute a tiled ``output = x @ weight.T`` projection."""
        tile_row, tile_column = wp.tid()
        typed_zero = DTYPE(0.0)
        accumulator = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
        for inner_tile in range((x.shape[1] + TILE_K - 1) / TILE_K):
            inner_offset = inner_tile * TILE_K
            activations = wp.tile_load(x, shape=(TILE_M, TILE_K), offset=(tile_row * TILE_M, inner_offset))
            weights = wp.tile_load(weight, shape=(TILE_N, TILE_K), offset=(tile_column * TILE_N, inner_offset))
            wp.tile_matmul(activations, wp.tile_transpose(weights), accumulator)
        wp.tile_store(
            output,
            wp.tile_map(cast_output, accumulator),
            offset=(tile_row * TILE_M, tile_column * TILE_N),
        )

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_linear_tiled_kernel(dtype: type, rows: int):
    """Return a dense projection kernel and its tile shape for ``rows``."""
    tile_m = 8 if rows < 16 else 32
    return _create_linear_tiled_kernel(dtype, tile_m), (tile_m, 32)


@wp.kernel
def _elu_kernel(
    x: wp.array2d[Any],
    y: wp.array2d[Any],
    alpha: float,
):
    """Apply ELU elementwise; ``alpha`` controls the negative branch."""
    i, j = wp.tid()
    v = x[i, j]
    y[i, j] = wp.where(v >= x.dtype(0.0), v, x.dtype(alpha) * (wp.exp(v) - x.dtype(1.0)))


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
        y[i, j] = x.dtype(wp.max(value_fp32, wp.float32(0.0)) + wp.log(wp.float32(1.0) + wp.exp(-wp.abs(value_fp32))))


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
def _transpose_021_kernel(x: wp.array3d[Any], output: wp.array3d[Any]):
    """Transpose a rank-3 tensor from axes 0-1-2 to 0-2-1."""
    i, j, k = wp.tid()
    output[i, j, k] = x[i, k, j]


@wp.kernel(enable_backward=False)
def _transpose_0213_kernel(x: wp.array4d[Any], output: wp.array4d[Any]):
    """Transpose a rank-4 tensor from axes 0-1-2-3 to 0-2-1-3."""
    i, j, k, l = wp.tid()
    output[i, j, k, l] = x[i, k, j, l]


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
    position = position_ids[0, 0] + wp.int64(sequence) if position_offset else position_ids[batch, sequence]
    value = wp.float32(x[batch, head, sequence, column])
    rotated = sign * wp.float32(x[batch, head, sequence, partner])
    output[batch, head, sequence, column] = x.dtype(
        value * wp.float32(cos_cache[position, cache_column]) + rotated * wp.float32(sin_cache[position, cache_column])
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
    unit = (x[row, column] - mean[column]) / wp.sqrt(variance[column] + x.dtype(epsilon))
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
        inverse_rms = wp.tile_map(_inverse_sqrt, sum_squares / x.dtype(wp.static(width)) + epsilon_tile)
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

    s_i = gates[b, 0 * hidden_size + h] + Bx[0 * hidden_size + h] + Bh[0 * hidden_size + h]
    s_o = gates[b, 1 * hidden_size + h] + Bx[1 * hidden_size + h] + Bh[1 * hidden_size + h]
    s_f = gates[b, 2 * hidden_size + h] + Bx[2 * hidden_size + h] + Bh[2 * hidden_size + h]
    s_c = gates[b, 3 * hidden_size + h] + Bx[3 * hidden_size + h] + Bh[3 * hidden_size + h]

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
        (wp.float32(data[row, column]) - wp.float32(zero_points[row, block])) * wp.float32(scales[row, block])
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
        magnitude = (wp.float32(1.0) + wp.float32(mantissa) * wp.float32(0.125)) * wp.pow(
            wp.float32(2.0), wp.float32(exponent - 7)
        )
    if bits & 128:
        magnitude = -magnitude
    output[index] = output.dtype(magnitude * wp.float32(scale[0]))


@wp.kernel(module="unique")
def _gather_rows_kernel(data: wp.array2d[Any], indices: wp.array2d[wp.int64], output: wp.array3d[Any]):
    """Gather matrix rows for batched token indices."""
    batch, sequence, column = wp.tid()
    output[batch, sequence, column] = data[indices[batch, sequence], column]


@wp.kernel(enable_backward=False, module="unique")
def _reorder_heads_kernel(x: wp.array2d[Any], output: wp.array2d[Any], head_size: int):
    """Reorder row-major packed heads into head-major rows."""
    row, head, column = wp.tid()
    output[head * x.shape[0] + row, column] = x[row, head * head_size + column]


@wp.kernel(enable_backward=False, module="unique")
def _unpack_gated_heads_kernel(x: wp.array2d[Any], values: wp.array2d[Any], gate: wp.array2d[Any], head_size: int):
    """Split per-head value/gate pairs and reorder values head-major."""
    row, head, column = wp.tid()
    offset = head * head_size * 2
    values[head * x.shape[0] + row, column] = x[row, offset + column]
    gate[row, head * head_size + column] = x[row, offset + head_size + column]


@wp.kernel(enable_backward=False, module="unique")
def _append_head_cache_kernel(
    x: wp.array2d[Any], positions: wp.array2d[wp.int64], cache: wp.array2d[Any], heads: int, head_size: int
):
    """Append head-major token rows at their device-side positions."""
    head, row, column = wp.tid()
    capacity = cache.shape[0] / heads
    cache[head * capacity + wp.int32(positions[0, row]), column] = x[head * positions.shape[1] + row, column]


@wp.kernel(enable_backward=False, module="unique")
def _sigmoid_gate_kernel(x: wp.array2d[Any], gate: wp.array2d[Any], output: wp.array2d[Any]):
    """Multiply activations by a sigmoid gate."""
    row, column = wp.tid()
    gate_value = wp.float32(gate[row, column])
    output[row, column] = x.dtype(wp.float32(x[row, column]) / (wp.float32(1.0) + wp.exp(-gate_value)))


@wp.kernel(enable_backward=False)
def _gather_single_index_kernel(
    data: wp.array1d[Any], output: wp.array1d[Any], index: int, axis_size: int, stride: int
):
    """Gather one index along a flattened axis with the given ``stride``."""
    output_index = wp.tid()
    prefix = output_index / stride
    suffix = output_index % stride
    output[output_index] = data[(prefix * axis_size + index) * stride + suffix]


@wp.kernel(enable_backward=False, module="unique", grid_stride=False)
def _quantize_activation_int8_kernel(
    activations: wp.array2d[wp.float16],
    quantized: wp.array2d[wp.int8],
    scales: wp.array2d[wp.float32],
):
    """Quantize FP16 rows to symmetric INT8 in 32-value blocks."""
    thread = wp.tid()
    lane = thread % 32
    block = (thread / 32) % scales.shape[1]
    row = (thread / 32) / scales.shape[1]
    column = block * 32 + lane
    value = wp.float32(activations[row, column])
    maximum = warp_max_broadcast(wp.abs(value))
    scale = maximum / 127.0 if maximum > 0.0 else 1.0
    quantized[row, column] = wp.int8(wp.clamp(wp.round(value / scale), -127.0, 127.0))
    if lane == 0:
        scales[row, block] = scale


@wp.func
def _expand_int4x4(value: wp.uint32) -> wp.int32:
    return wp.int32(
        (value & wp.uint32(0x000F))
        | ((value & wp.uint32(0x00F0)) << wp.uint32(4))
        | ((value & wp.uint32(0x0F00)) << wp.uint32(8))
        | ((value & wp.uint32(0xF000)) << wp.uint32(12))
    )


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
        block_total = dp4a(expand_int4x4_low(wp.int32(packed_weights)), packed_activation_0, 0)
        block_total = dp4a(expand_int4x4_high(wp.int32(packed_weights)), packed_activation_1, block_total)
        activation_sum = dp4a(0x01010101, packed_activation_0, 0)
        activation_sum = dp4a(0x01010101, packed_activation_1, activation_sum)
        block_total -= 8 * activation_sum
        total += wp.float32(block_total) * activation_scales[row, block] * wp.float32(weight_scales[column, block])
    total = subgroup_sum(total, 4)
    if lane == 0:
        output[row, column] = wp.float16(total)


@lru_cache(maxsize=None)
def _get_matmul_int8_q8_kernel(reduction_width: int):
    """Build an INT8 matrix-vector kernel using ``reduction_width`` lanes."""
    REDUCTION_WIDTH = reduction_width
    WORDS_PER_LANE = 8 // reduction_width

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        activations: wp.array3d[wp.uint32],
        activation_scales: wp.array2d[wp.float32],
        weights: wp.array3d[wp.uint32],
        weight_scales: wp.array2d[wp.float16],
        output: wp.array2d[wp.float16],
    ):
        """Multiply Q8 activations by block-scaled INT8 weights."""
        thread = wp.tid()
        lane = thread % REDUCTION_WIDTH
        item = thread / REDUCTION_WIDTH
        row = item / weights.shape[0]
        column = item % weights.shape[0]
        total = wp.float32(0.0)
        for block in range(weights.shape[1]):
            block_total = 0
            for group in range(WORDS_PER_LANE):
                word = lane + group * REDUCTION_WIDTH
                packed_activation = wp.int32(activations[row, block, word])
                signed_weights = wp.int32(weights[column, block, word] ^ wp.uint32(0x80808080))
                block_total = dp4a(signed_weights, packed_activation, block_total)
            total += wp.float32(block_total) * activation_scales[row, block] * wp.float32(weight_scales[column, block])
        total = subgroup_sum(total, REDUCTION_WIDTH)
        if lane == 0:
            output[row, column] = wp.float16(total)

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
        scale_indices = wp.tile_map(scale_index, wp.tile_arange(0, packed_width, dtype=wp.int32))
        for block_tile in range((scales.shape[1] + blocks_per_tile - 1) / blocks_per_tile):
            activation = wp.tile_load(
                activations, shape=(tile_m, activation_width), offset=(row, block_tile * activation_width)
            )
            packed = wp.tile_load(weights, shape=(tile_n, packed_width), offset=(column, block_tile * packed_width))
            block_scales = wp.tile_load(
                scales, shape=(tile_n, blocks_per_tile), offset=(column, block_tile * blocks_per_tile)
            )
            scale = block_scales[:, scale_indices]
            low = wp.tile_transpose(wp.tile_map(dequantize_low, packed, scale))
            high = wp.tile_transpose(wp.tile_map(dequantize_high, packed, scale))
            wp.tile_matmul(activation[:, even_columns], low, total)
            wp.tile_matmul(activation[:, odd_columns], high, total)
        wp.tile_store(output, wp.tile_map(to_float16, total), offset=(row, column))

    return kernel


def _nbits_reduction_width(bits: int, packed_block_size: int, warp_reduction: bool) -> int:
    if not warp_reduction:
        return 1
    width = 1 << (packed_block_size - 1).bit_length()
    return min(32, width // 2 if bits == 8 and width > 1 else width)


def _create_matmul_nbits_kernel(bits: int, block_size: int, dtype: type, warp_reduction: bool):
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
                zero = (packed_zero >> ((block % values_per_byte) * bits)) & ((1 << bits) - 1)
            block_total = wp.float32(0.0)
            for group in range(loads_per_lane):
                packed_offset = lane + group * load_stride
                if packed_offset < packed_block_size:
                    packed = wp.int32(weights[column, block, packed_offset])
                    activation_offset = block * block_size + packed_offset * values_per_byte
                    for value_index in range(values_per_byte):
                        quantized = (packed >> (value_index * bits)) & ((1 << bits) - 1)
                        block_total += wp.float32(activations[row, activation_offset + value_index]) * wp.float32(
                            quantized - zero
                        )
            total += block_total * wp.float32(scales[column, block])

        if warp_reduction:
            total = subgroup_sum(total, reduction_width)
        if lane == 0:
            output[row, column] = dtype(total)

    return kernel


@lru_cache(maxsize=None)
def _get_matmul_nbits_kernel(bits: int, block_size: int, dtype: type, warp_reduction: bool):
    """Return the reduction width and cached generic N-bit matmul kernel."""
    packed_block_size = block_size * bits // 8
    reduction_width = _nbits_reduction_width(bits, packed_block_size, warp_reduction)
    return reduction_width, _create_matmul_nbits_kernel(bits, block_size, dtype, warp_reduction)


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
            zero = (packed_zero >> ((block % values_per_byte) * bits)) & ((1 << bits) - 1)
        scale = wp.float32(scales[column, block])
        output_offset = block * block_size + packed_offset * values_per_byte
        for value_index in range(values_per_byte):
            quantized = (packed >> (value_index * bits)) & ((1 << bits) - 1)
            output[column, output_offset + value_index] = dtype(wp.float32(quantized - zero) * scale)

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
            input_value = wp.float32(past[batch, channel, input_index + kernel_size - 1])
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
        present[batch, channel, state_index] = x[batch, channel, source_index - past.shape[2]]


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
            state[batch, channel, state_index] = x[batch, channel, source_index - state.shape[2]]


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


@wp.kernel(enable_backward=False, module="unique")
def _prepare_gated_delta_kernel(
    a: wp.array2d[Any],
    b: wp.array2d[Any],
    a_log: wp.array1d[Any],
    dt_bias: wp.array1d[Any],
    decay: wp.array2d[wp.float32],
    beta: wp.array2d[wp.float32],
):
    """Compute FP32 decay and beta controls for gated-delta attention."""
    row, head = wp.tid()
    b_value = wp.float32(b[row, head])
    beta[row, head] = wp.float32(1.0) / (wp.float32(1.0) + wp.exp(-b_value))
    dt = wp.float32(a[row, head]) + wp.float32(dt_bias[head])
    softplus = wp.max(dt, wp.float32(0.0)) + wp.log(wp.float32(1.0) + wp.exp(-wp.abs(dt)))
    decay[row, head] = -wp.exp(wp.float32(a_log[head])) * softplus


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
    def normalize(value: dtype, scale: SCALE_DTYPE, inverse_rms: float, scale_offset: float):
        return dtype(wp.float32(value) * (wp.float32(SCALE_DTYPE(scale)) + scale_offset) * inverse_rms)

    @wp.func
    def skip_normalize(value: dtype, skip: dtype, scale: SCALE_DTYPE, inverse_rms: float, scale_offset: float):
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
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(tile_index * TILE_WIDTH,))
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
                output[row], wp.tile_map(normalize, values, scales, inverse_rms, scale_offset), offset=(offset,)
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
            wp.tile_store(residual[row], wp.tile_map(add, values, skips), offset=(offset,))
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
            normalized = wp.tile_map(skip_normalize, values, skips, scales, inverse_rms, scale_offset)
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


def _create_gated_rms_norm_kernel(tile_width: int, dtype: type):
    """Build fused RMSNorm-times-SiLU gating for recurrent attention."""
    TILE_WIDTH = tile_width
    DTYPE = dtype

    @wp.func
    def square(value: dtype):
        value_fp32 = wp.float32(dtype(value))
        return value_fp32 * value_fp32

    @wp.func
    def normalize_gate(value: dtype, gate: dtype, scale: dtype, inverse_rms: float):
        gate_fp32 = wp.float32(gate)
        silu = gate_fp32 / (wp.float32(1.0) + wp.exp(-gate_fp32))
        return dtype(wp.float32(value) * wp.float32(scale) * inverse_rms * silu)

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        x: wp.array2d(dtype=DTYPE),
        gate: wp.array2d(dtype=DTYPE),
        scale: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        epsilon: float,
    ):
        """Normalize each row, then multiply by its SiLU gate."""
        row = wp.tid()
        scale_row = row % scale.shape[0]
        typed_zero = DTYPE(0.0)
        partials = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            partials += wp.tile_map(square, values)
        inverse_rms = wp.float32(1.0) / wp.sqrt(
            wp.tile_extract(wp.tile_sum(partials), 0) / wp.float32(x.shape[1])
            + wp.float32(epsilon)
            + wp.float32(typed_zero)
        )
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            gates = wp.tile_load(gate[row], shape=(TILE_WIDTH,), offset=(offset,))
            scales = wp.tile_load(scale[scale_row], shape=(TILE_WIDTH,), offset=(offset,))
            wp.tile_store(
                output[row], wp.tile_map(normalize_gate, values, gates, scales, inverse_rms), offset=(offset,)
            )

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_gated_rms_norm_kernel(width: int, dtype: type):
    """Return a cached recurrent gated-RMSNorm kernel and tile width."""
    tile_width = min(512, max(32, 1 << (width - 1).bit_length()))
    return tile_width, _create_gated_rms_norm_kernel(tile_width, dtype)


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
    def kernel(x: wp.array2d(dtype=DTYPE), output: wp.array2d(dtype=DTYPE), epsilon: float):
        """Normalize each row to unit L2 norm."""
        row = wp.tid()
        typed_zero = DTYPE(0.0)
        partials = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(tile_index * TILE_WIDTH,))
            partials += wp.tile_map(square, values)
        norm = wp.sqrt(wp.tile_extract(wp.tile_sum(partials), 0) + wp.float32(epsilon) + wp.float32(typed_zero))
        inverse_norm = wp.float32(1.0) / wp.max(norm, wp.float32(1.0e-12))
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            wp.tile_store(output[row], wp.tile_map(normalize, values, inverse_norm), offset=(offset,))

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


def _create_linear_attention_kernel(key_size: int, value_size: int, dtype: type, state_dtype: type):
    """Build recurrent linear attention for fixed key and value widths.
    Value channels are processed in tiles of at most 64."""
    KEY_SIZE = key_size
    VALUE_SIZE = value_size
    VALUE_TILE = min(64, value_size & -value_size)
    VALUE_BLOCKS = VALUE_SIZE // VALUE_TILE
    DTYPE = dtype
    STATE_DTYPE = state_dtype

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
        needs_decay: bool,
        decay_per_key: bool,
        needs_beta: bool,
        beta_per_head: bool,
        scale: float,
    ):
        """Update recurrent attention state and emit the current sequence."""
        item = wp.tid()
        typed_zero = DTYPE(0.0)
        value_block = item % VALUE_BLOCKS
        state_item = item / VALUE_BLOCKS
        batch = state_item / value_heads
        value_head = state_item % value_heads
        key_head = value_head * key_heads / value_heads
        state_offset = state_item * KEY_SIZE
        value_offset = value_block * VALUE_TILE
        state = wp.tile_load(past, shape=(KEY_SIZE, VALUE_TILE), offset=(state_offset, value_offset))

        for token in range(sequence_length):
            token_row = batch * sequence_length + token
            if needs_decay:
                if decay_per_key:
                    decay_row = wp.tile_load(decay, shape=(1, KEY_SIZE), offset=(token_row, value_head * KEY_SIZE))
                    decay_column = wp.tile_transpose(wp.tile_map(exp_value, decay_row))
                    state *= wp.tile_broadcast(decay_column, shape=(KEY_SIZE, VALUE_TILE))
                else:
                    state *= STATE_DTYPE(wp.exp(wp.float32(decay[token_row, value_head])))

            key_row = wp.tile_map(
                to_state, wp.tile_load(key, shape=(1, KEY_SIZE), offset=(token_row, key_head * KEY_SIZE))
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
                beta_value = beta[token_row, value_head] if beta_per_head else beta[token_row, 0]
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
                        wp.tile_load(query, shape=(1, KEY_SIZE), offset=(token_row, query_head * KEY_SIZE)),
                    )
                    result = wp.tile_zeros(shape=(1, VALUE_TILE), dtype=STATE_DTYPE)
                    wp.tile_matmul(query_row, state, result)
                    wp.tile_store(
                        output,
                        wp.tile_map(to_output, STATE_DTYPE(scale) * result),
                        offset=(token_row, query_head * VALUE_SIZE + value_offset),
                    )
            else:
                query_head = value_head * query_heads / value_heads
                query_row = wp.tile_map(
                    to_state,
                    wp.tile_load(query, shape=(1, KEY_SIZE), offset=(token_row, query_head * KEY_SIZE)),
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


def _get_linear_attention_kernel(key_size: int, value_size: int, dtype: type, state_dtype: type | None = None):
    """Return a cached recurrent linear-attention kernel."""
    key = (key_size, value_size, dtype, state_dtype or dtype)
    if key not in _linear_attention_kernel_cache:
        _linear_attention_kernel_cache[key] = _create_linear_attention_kernel(*key)
    return _linear_attention_kernel_cache[key]


def _create_mamba2_decode_kernel(head_dim: int, state_size: int, heads_per_group: int, dtype: type):
    """Build one-token Mamba-2 selective-state update and projection."""
    HEAD_DIM = head_dim
    STATE_TILE = max(32, 1 << (state_size - 1).bit_length())
    HEADS_PER_GROUP = heads_per_group
    DTYPE = dtype

    @wp.func
    def to_float(value: dtype):
        return wp.float32(dtype(value))

    @wp.func
    def update_state(value: wp.float32, b: wp.float32, decay: wp.float32, source: wp.float32):
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
        projected = wp.tile_extract(wp.tile_sum(wp.tile_map(multiply, values, c_values)), 0)
        output[head, channel] = DTYPE(projected + d[head] * wp.float32(x[head, channel]))

    kernel.module.options["enable_backward"] = False
    return STATE_TILE, kernel


@lru_cache(maxsize=None)
def _get_mamba2_decode_kernel(head_dim: int, state_size: int, heads_per_group: int, dtype: type):
    """Return the cached one-token Mamba-2 kernel and reduction width."""
    if min(head_dim, state_size, heads_per_group) <= 0:
        raise ValueError("Mamba-2 dimensions must be positive")
    return _create_mamba2_decode_kernel(head_dim, state_size, heads_per_group, dtype)


def _create_mamba2_prefill_kernel(head_dim: int, state_size: int, heads_per_group: int, dtype: type):
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
                wp.tile_load(x, shape=(1, CHANNEL_TILE), offset=(token, head * HEAD_DIM + channel_offset)),
            )
            b_column = wp.tile_transpose(
                wp.tile_map(
                    to_float,
                    wp.tile_load(b, shape=(1, STATE_SIZE), offset=(token, group * STATE_SIZE)),
                )
            )
            wp.tile_matmul(b_column, step * x_row, values)

            c_row = wp.tile_map(
                to_float,
                wp.tile_load(c, shape=(1, STATE_SIZE), offset=(token, group * STATE_SIZE)),
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
def _get_mamba2_prefill_kernel(head_dim: int, state_size: int, heads_per_group: int, dtype: type):
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
            paired_column = column + head_size // 2 if column < head_size // 2 else column - head_size // 2
            sign = wp.float32(-1.0) if column < head_size // 2 else wp.float32(1.0)
            paired = wp.float32(query[batch, token, offset + paired_column])
            current = current * wp.float32(cos_cache[position, cache_column]) + sign * paired * wp.float32(
                sin_cache[position, cache_column]
            )
        rotated_query[batch, head, token, column] = wp.float16(current)

    if head < kv_heads:
        offset = head * head_size
        current = wp.float32(key[batch, token, offset + column])
        if do_rotary:
            cache_column = column % (head_size // 2)
            paired_column = column + head_size // 2 if column < head_size // 2 else column - head_size // 2
            sign = wp.float32(-1.0) if column < head_size // 2 else wp.float32(1.0)
            paired = wp.float32(key[batch, token, offset + paired_column])
            current = current * wp.float32(cos_cache[position, cache_column]) + sign * paired * wp.float32(
                sin_cache[position, cache_column]
            )
        present_key[batch, head, cache_token, column] = wp.float16(current)
        present_value[batch, head, cache_token, column] = value[batch, token, offset + column]


def _create_gqa_attention_kernel(head_size: int, dtype: type):
    """Build numerically stable grouped-query attention for one dtype."""
    DTYPE = dtype

    @wp.func
    def dot(left: DTYPE, right: DTYPE):
        return wp.float32(DTYPE(left)) * wp.float32(right)

    @wp.func
    def accumulate(total: wp.float32, value: DTYPE, old_scale: wp.float32, weight: wp.float32):
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
    ):
        """Apply causal grouped-query attention over the populated cache."""
        index = wp.tid()
        query_token = index % sequence_length
        head = (index // sequence_length) % query_heads
        batch = index // (sequence_length * query_heads)
        kv_head = head // (query_heads // kv_heads)
        valid_keys = wp.int32(sequence_lengths_minus_one[batch]) - sequence_length + query_token + 2
        query_row = (batch * query_heads + head) * sequence_length + query_token
        query_values = wp.tile_load(query[query_row], shape=(head_size,))
        accumulator = wp.tile_zeros(shape=(head_size,), dtype=wp.float32)
        maximum = wp.float32(-3.402823466e38) + wp.float32(DTYPE(0.0))
        denominator = wp.float32(0.0)
        for key_token in range(valid_keys):
            cache_row = (batch * kv_heads + kv_head) * total_length + key_token
            key_values = wp.tile_load(key[cache_row], shape=(head_size,))
            score = wp.tile_extract(wp.tile_sum(wp.tile_map(dot, query_values, key_values)), 0)
            score *= wp.float32(scale)
            new_maximum = wp.max(maximum, score)
            old_scale = wp.exp(maximum - new_maximum)
            weight = wp.exp(score - new_maximum)
            denominator = denominator * old_scale + weight
            value_values = wp.tile_load(value[cache_row], shape=(head_size,))
            accumulator = wp.tile_map(accumulate, accumulator, value_values, old_scale, weight)
            maximum = new_maximum
        normalized = wp.tile_map(normalize, accumulator, denominator)
        wp.tile_store(output[batch * sequence_length + query_token], normalized, offset=(head * head_size,))

    kernel.module.options["enable_backward"] = False
    return kernel


_gqa_attention_kernel_cache = {}


def _get_gqa_attention_kernel(head_size: int, dtype: type = wp.float16):
    """Return cached GQA kernel and a head-sized CUDA block dimension."""
    key = (head_size, dtype)
    if key not in _gqa_attention_kernel_cache:
        _gqa_attention_kernel_cache[key] = _create_gqa_attention_kernel(*key)
    block_dim = min(1024, max(32, 1 << (head_size - 1).bit_length()))
    return block_dim, _gqa_attention_kernel_cache[key]


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
def _get_greedy_argmax_kernels(tile_width: int, partial_count: int, dtype: type = wp.float16):
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
        return wp.float32(DTYPE(value)) if index < vocabulary else wp.float32(-3.402823466e38)

    @wp.func
    def matching_token(value: wp.float32, token: wp.int32, maximum: wp.float32, vocabulary: wp.int32):
        return token if value == maximum else vocabulary

    @wp.kernel(enable_backward=False, module="unique")
    def partial_argmax(logits: wp.array3d(dtype=DTYPE), values: wp.array1d[wp.float32], tokens: wp.array1d[wp.int32]):
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
                wp.tile_load(logits[0, logits.shape[1] - 1], shape=TILE_WIDTH, offset=offset),
                indices,
                vocabulary,
            )
            local_token = wp.tile_extract(wp.tile_argmax(tile), 0)
            value = wp.tile_extract(tile, local_token)
            token = offset + local_token
            if value > best_value or (value == best_value and token < best_token):
                best_value = value
                best_token = token
        wp.tile_store(values, wp.tile_full(shape=1, value=best_value, dtype=wp.float32), offset=partial)
        wp.tile_store(tokens, wp.tile_full(shape=1, value=best_token, dtype=wp.int32), offset=partial)

    @wp.func
    def select_token(values: wp.array1d[wp.float32], tokens: wp.array1d[wp.int32], vocabulary: int):
        value_tile = wp.tile_load(values, shape=PARTIAL_COUNT)
        token_tile = wp.tile_load(tokens, shape=PARTIAL_COUNT)
        maximum_index = wp.tile_extract(wp.tile_argmax(value_tile), 0)
        maximum = wp.tile_extract(value_tile, maximum_index)
        candidates = wp.tile_map(matching_token, value_tile, token_tile, maximum, vocabulary)
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
        wp.tile_store(output, wp.tile_full(shape=1, value=select_token(values, tokens, vocabulary), dtype=wp.int32))

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


def _array_type(dtype: type, ndim: int):
    return wp.array(dtype=dtype, ndim=ndim)


_KERNEL_OVERLOADS: dict[tuple[Any, ...], Any] = {}


def _kernel_for_dtype(kernel, dtype: type, *parameter_types: type | tuple[int]):
    """Return one cached specialization of a generic same-dtype kernel."""
    key = (kernel, dtype, parameter_types)
    if key not in _KERNEL_OVERLOADS:
        signature = [_array_type(dtype, item[0]) if isinstance(item, tuple) else item for item in parameter_types]
        _KERNEL_OVERLOADS[key] = wp.overload(kernel, signature)
    return _KERNEL_OVERLOADS[key]


def _cast_kernel_for_dtypes(source_dtype: type, target_dtype: type):
    key = (_cast_kernel, source_dtype, target_dtype)
    if key not in _KERNEL_OVERLOADS:
        _KERNEL_OVERLOADS[key] = wp.overload(
            _cast_kernel,
            [wp.array1d(dtype=source_dtype), wp.array1d(dtype=target_dtype)],
        )
    return _KERNEL_OVERLOADS[key]


def _where_kernel_for_dtype(dtype: type):
    key = (_where_broadcast_kernel, dtype)
    if key not in _KERNEL_OVERLOADS:
        _KERNEL_OVERLOADS[key] = wp.overload(
            _where_broadcast_kernel,
            [wp.array2d(dtype=wp.bool), wp.array2d(dtype=dtype), wp.array2d(dtype=dtype), wp.array2d(dtype=dtype)],
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
