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

"""Graph-capturable ONNX inference runtime for Warp-NN policy networks.

Only the ``onnx`` package (pure protobuf parser) is required -- no
``onnxruntime`` or ``torch``.  Weights are loaded once onto the target
Warp device; inference executes a pre-built list of lightweight op
descriptors that dispatch to dedicated inference kernels without host
round-trips or device allocation.

Dense layers reuse Warp-NN's tiled matrix multiplication for efficient
single-policy and batched-policy execution. Elementwise and normalization
operators use deterministic one-writer kernels. All runtime-owned buffers
are allocated during construction, so execution is CUDA-graph capturable
after warmup.

Supported ONNX operators (all graph-capturable after one warmup call):

* **Add**, **Sub**, **Mul**, **Div** -- 2-D tensors with optional 1-D broadcasting
* **BatchNormalization** -- 2-D inference mode
* **Gemm** -- ``C = alpha * A @ B.T + beta * bias`` with ``transB=1``
* **Elu**, **Relu**, **Sqrt**, **Tanh** -- elementwise activation/math
* **ReduceMean** -- 2-D row reduction with ``keepdims=1``
* **Constant** -- tensor-valued constants resolved during construction
* **Reshape** -- view-only reshape with a construction-time shape tensor
* **GatherBlockQuantized** -- Qwen-style INT8 block-quantized embedding lookup
* **MatMulNBits** -- Qwen-style INT4/INT8 block-quantized matrix multiplication
* **CausalConvWithState** -- stateful causal depthwise 1-D convolution
* **LinearAttention** -- fused recurrent linear/delta attention with GQA
* **SimplifiedLayerNormalization**, **SkipSimplifiedLayerNormalization** --
  Qwen FP16 last-axis RMS normalization, with optional residual addition
* **GroupQueryAttention** -- causal FP16 grouped-query attention with
  non-interleaved rotary embeddings and an external FP16 KV cache
* **RotaryEmbedding** -- BNSH rotary position embedding with split-half or
  interleaved pairs
* **Cast**, **Gather**, **ReduceSum**, **Shape** -- the integer metadata subset
  used by transformer attention-mask subgraphs
* **Squeeze** -- alias passthrough (the output array shares memory with the
  input). Only used to drop unit dims, no copy is performed.
* **LSTM** -- forward, single-direction, single-layer, ``seq_length=1``. The
  full step (gate GEMM + cell update) executes in two on-device kernels.

Floating-point operators preserve matching FP16, BF16, FP32, or FP64 model
dtypes. Non-floating initializers such as shape and axis tensors retain their
declared dtype as well.

Example::

    from warp_nn.runtime import OnnxRuntime

    rt = OnnxRuntime("policy.onnx", device="cuda:0")
    out = rt({"observation": wp.array2d(obs, dtype=wp.float32, device="cuda:0")})
    actions = out["action"]
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

from warp_nn.modules.layers._common import tile_transposed_gemm_2d
from warp_nn.runtime._cublas import try_create_cublas
from warp_nn.utils.config import get_kernel_config
from warp_nn.utils.device import parse_device
from warp_nn.utils.ops import resolve_dim


def _require_onnx():
    """Lazy import of the ``onnx`` package with a friendly error message."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - exercised only on missing dep
        raise ImportError(
            "OnnxRuntime requires the optional `onnx` package. "
            "Install it with `pip install onnx>=1.16.0` or `pip install warp-nn[onnx]`."
        ) from exc
    return onnx, numpy_helper


_FLOAT_DTYPES = (wp.float16, wp.bfloat16, wp.float32, wp.float64)


def _warp_dtype_from_onnx(onnx, elem_type: int):
    """Map an ONNX tensor element type to a Warp scalar type."""
    mapping = {
        onnx.TensorProto.FLOAT16: wp.float16,
        onnx.TensorProto.BFLOAT16: wp.bfloat16,
        onnx.TensorProto.FLOAT: wp.float32,
        onnx.TensorProto.DOUBLE: wp.float64,
        onnx.TensorProto.INT8: wp.int8,
        onnx.TensorProto.INT16: wp.int16,
        onnx.TensorProto.INT32: wp.int32,
        onnx.TensorProto.INT64: wp.int64,
        onnx.TensorProto.UINT8: wp.uint8,
        onnx.TensorProto.UINT16: wp.uint16,
        onnx.TensorProto.UINT32: wp.uint32,
        onnx.TensorProto.UINT64: wp.uint64,
        onnx.TensorProto.BOOL: wp.bool,
    }
    try:
        return mapping[elem_type]
    except KeyError as exc:
        type_name = onnx.TensorProto.DataType.Name(elem_type)
        raise NotImplementedError(f"OnnxRuntime: unsupported tensor dtype '{type_name}'") from exc


def _require_matching_float_dtypes(op, dtypes: dict[str, type], names: list[str]) -> type:
    """Validate homogeneous floating-point inputs and return their dtype."""
    dtype = dtypes[names[0]]
    if dtype not in _FLOAT_DTYPES:
        raise NotImplementedError(f"OnnxRuntime {op.op_type}: dtype '{dtype.__name__}' is not supported")
    mismatched = {name: dtypes[name].__name__ for name in names if dtypes[name] != dtype}
    if mismatched:
        actual = {name: dtypes[name].__name__ for name in names}
        raise ValueError(f"OnnxRuntime {op.op_type}: input dtypes must match, got {actual}")
    return dtype


# ---------------------------------------------------------------------------
# Inference kernels
# ---------------------------------------------------------------------------
#
# Simple per-output-element kernels: one thread writes one cell.  Policies
# seen in practice are tiny (batch=1, hidden<=128), so the tiled variants
# used by the training modules are unnecessary here.


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
    @wp.kernel
    def kernel(
        A: wp.array2d[float],
        B: wp.array2d[float],
        bias: wp.array2d[float],
        alpha: float,
        beta: float,
        C: wp.array2d[float],
    ):
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
def _elu_kernel(
    x: wp.array2d[Any],
    y: wp.array2d[Any],
    alpha: float,
):
    i, j = wp.tid()
    v = x[i, j]
    y[i, j] = wp.where(v >= x.dtype(0.0), v, x.dtype(alpha) * (wp.exp(v) - x.dtype(1.0)))


@wp.kernel
def _unary_kernel(x: wp.array2d[Any], operation: int, y: wp.array2d[Any]):
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
    row = wp.tid()
    total = x.dtype(0.0)
    for column in range(x.shape[1]):
        total += x[row, column]
    out[row, 0] = total / x.dtype(x.shape[1])


@wp.kernel(enable_backward=False)
def _reduce_sum_rows_kernel(x: wp.array2d[Any], out: wp.array1d[Any]):
    row = wp.tid()
    total = x.dtype(0)
    for column in range(x.shape[1]):
        total += x[row, column]
    out[row] = total


@wp.kernel(enable_backward=False)
def _reduce_max_1d_kernel(x: wp.array1d[Any], out: wp.array1d[Any]):
    value = x[0]
    for index in range(1, x.shape[0]):
        value = wp.max(value, x[index])
    out[0] = value


@wp.kernel(enable_backward=False)
def _cast_kernel(x: wp.array1d[Any], out: wp.array1d[Any]):
    index = wp.tid()
    out[index] = out.dtype(x[index])


@wp.kernel(enable_backward=False)
def _transpose_021_kernel(x: wp.array3d[Any], output: wp.array3d[Any]):
    i, j, k = wp.tid()
    output[i, j, k] = x[i, k, j]


@wp.kernel(enable_backward=False)
def _transpose_0213_kernel(x: wp.array4d[Any], output: wp.array4d[Any]):
    i, j, k, l = wp.tid()
    output[i, j, k, l] = x[i, k, j, l]


@wp.kernel(enable_backward=False)
def _split_last_axis_kernel(
    x: wp.array2d[Any],
    output: wp.array2d[Any],
    input_offset: int,
):
    row, column = wp.tid()
    output[row, column] = x[row, input_offset + column]


@wp.kernel(enable_backward=False)
def _tile_3d_kernel(x: wp.array3d[Any], output: wp.array3d[Any]):
    i, j, k = wp.tid()
    output[i, j, k] = x[i % x.shape[0], j % x.shape[1], k % x.shape[2]]


@wp.kernel(enable_backward=False)
def _where_broadcast_kernel(
    condition: wp.array2d[wp.bool],
    x: wp.array2d[Any],
    y: wp.array2d[Any],
    output: wp.array2d[Any],
):
    row, column = wp.tid()
    output[row, column] = wp.where(
        condition[row % condition.shape[0], column % condition.shape[1]],
        x[row, column],
        y[row, column],
    )


@wp.kernel(enable_backward=False)
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
    batch, sequence, column = wp.tid()
    row = indices[batch, sequence]
    block = column / block_size
    output[batch, sequence, column] = wp.float16(
        (wp.float32(data[row, column]) - wp.float32(zero_points[row, block])) * wp.float32(scales[row, block])
    )


@wp.kernel
def _gather_rows_kernel(data: wp.array2d[Any], indices: wp.array2d[wp.int64], output: wp.array3d[Any]):
    batch, sequence, column = wp.tid()
    output[batch, sequence, column] = data[indices[batch, sequence], column]


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
#endif
    return value;
    """
)
def _warp_sum(value: float) -> float: ...


def _create_matmul_nbits_kernel(bits: int, block_size: int, dtype: type, warp_reduction: bool):
    values_per_byte = 8 // bits
    packed_block_size = block_size // values_per_byte
    load_stride = 32 if warp_reduction else 1
    loads_per_lane = (packed_block_size + load_stride - 1) // load_stride

    @wp.kernel(enable_backward=False)
    def kernel(
        activations: wp.array2d(dtype=dtype),
        weights: wp.array3d[wp.uint8],
        scales: wp.array2d(dtype=dtype),
        zero_points: wp.array2d[wp.uint8],
        output: wp.array2d(dtype=dtype),
        has_zero_points: bool,
    ):
        thread = wp.tid()
        lane = thread & 31 if warp_reduction else 0
        item = thread / 32 if warp_reduction else thread
        row = item / weights.shape[0]
        column = item % weights.shape[0]
        total = wp.float32(0.0)

        for block in range(weights.shape[1]):
            zero = 1 << (bits - 1)
            if has_zero_points:
                packed_zero = wp.int32(zero_points[column, block / values_per_byte])
                zero = (packed_zero >> ((block % values_per_byte) * bits)) & ((1 << bits) - 1)
            scale = wp.float32(scales[column, block])
            for group in range(loads_per_lane):
                packed_offset = lane + group * load_stride
                if packed_offset < packed_block_size:
                    packed = wp.int32(weights[column, block, packed_offset])
                    activation_offset = block * block_size + packed_offset * values_per_byte
                    for value_index in range(values_per_byte):
                        quantized = (packed >> (value_index * bits)) & ((1 << bits) - 1)
                        total += (
                            wp.float32(activations[row, activation_offset + value_index])
                            * wp.float32(quantized - zero)
                            * scale
                        )

        if warp_reduction:
            total = _warp_sum(total)
        if lane == 0:
            output[row, column] = dtype(total)

    return kernel


_matmul_nbits_kernel_cache = {}


def _get_matmul_nbits_kernel(bits: int, block_size: int, dtype: type, warp_reduction: bool):
    key = (bits, block_size, dtype, warp_reduction)
    if key not in _matmul_nbits_kernel_cache:
        _matmul_nbits_kernel_cache[key] = _create_matmul_nbits_kernel(*key)
    return _matmul_nbits_kernel_cache[key]


def _create_dequantize_nbits_kernel(bits: int, block_size: int, dtype: type):
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
    batch, channel, state_index = wp.tid()
    sequence_length = x.shape[2]
    source_index = sequence_length + state_index
    if source_index < past.shape[2]:
        present[batch, channel, state_index] = past[batch, channel, source_index]
    else:
        present[batch, channel, state_index] = x[batch, channel, source_index - past.shape[2]]


_dequantize_nbits_kernel_cache = {}


def _get_dequantize_nbits_kernel(bits: int, block_size: int, dtype: type):
    key = (bits, block_size, dtype)
    if key not in _dequantize_nbits_kernel_cache:
        _dequantize_nbits_kernel_cache[key] = _create_dequantize_nbits_kernel(*key)
    return _dequantize_nbits_kernel_cache[key]


def _create_rms_norm_kernels(tile_width: int, dtype: type):
    TILE_WIDTH = tile_width
    DTYPE = dtype

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
    def normalize(value: dtype, scale: dtype, inverse_rms: float):
        return dtype(wp.float32(value) * wp.float32(scale) * inverse_rms)

    @wp.func
    def skip_normalize(value: dtype, skip: dtype, scale: dtype, inverse_rms: float):
        return dtype((wp.float32(value) + wp.float32(skip)) * wp.float32(scale) * inverse_rms)

    @wp.kernel(enable_backward=False)
    def rms_norm(
        x: wp.array2d(dtype=DTYPE),
        scale: wp.array1d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        epsilon: float,
    ):
        row = wp.tid()
        typed_zero = DTYPE(0.0)
        partials = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(tile_index * TILE_WIDTH,))
            partials += wp.tile_map(square, values)
        inverse_rms = wp.float32(1.0) / wp.sqrt(
            wp.tile_extract(wp.tile_sum(partials), 0) / wp.float32(x.shape[1])
            + wp.float32(epsilon)
            + wp.float32(typed_zero)
        )
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            scales = wp.tile_load(scale, shape=(TILE_WIDTH,), offset=(offset,))
            wp.tile_store(output[row], wp.tile_map(normalize, values, scales, inverse_rms), offset=(offset,))

    @wp.kernel(enable_backward=False)
    def skip_rms_norm(
        x: wp.array2d(dtype=DTYPE),
        skip: wp.array2d(dtype=DTYPE),
        scale: wp.array1d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        residual: wp.array2d(dtype=DTYPE),
        epsilon: float,
    ):
        row = wp.tid()
        typed_zero = DTYPE(0.0)
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
        )
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            skips = wp.tile_load(skip[row], shape=(TILE_WIDTH,), offset=(offset,))
            scales = wp.tile_load(scale, shape=(TILE_WIDTH,), offset=(offset,))
            normalized = wp.tile_map(skip_normalize, values, skips, scales, inverse_rms)
            wp.tile_store(output[row], normalized, offset=(offset,))

    return rms_norm, skip_rms_norm


_rms_norm_kernel_cache = {}


def _get_rms_norm_kernels(width: int, dtype: type):
    tile_width = min(512, max(32, 1 << (width - 1).bit_length()))
    key = (tile_width, dtype)
    if key not in _rms_norm_kernel_cache:
        _rms_norm_kernel_cache[key] = _create_rms_norm_kernels(*key)
    return tile_width, _rms_norm_kernel_cache[key]


def _create_lp_normalization_kernel(tile_width: int, dtype: type):
    TILE_WIDTH = tile_width
    DTYPE = dtype

    @wp.func
    def square(value: dtype):
        value_fp32 = wp.float32(dtype(value))
        return value_fp32 * value_fp32

    @wp.func
    def normalize(value: dtype, inverse_norm: float):
        return dtype(wp.float32(value) * inverse_norm)

    @wp.kernel(enable_backward=False)
    def kernel(x: wp.array2d(dtype=DTYPE), output: wp.array2d(dtype=DTYPE)):
        row = wp.tid()
        typed_zero = DTYPE(0.0)
        partials = wp.tile_zeros(shape=(TILE_WIDTH,), dtype=wp.float32)
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(tile_index * TILE_WIDTH,))
            partials += wp.tile_map(square, values)
        norm = wp.sqrt(wp.tile_extract(wp.tile_sum(partials), 0) + wp.float32(typed_zero))
        inverse_norm = wp.float32(1.0) / wp.max(norm, wp.float32(1.0e-12))
        for tile_index in range((x.shape[1] + TILE_WIDTH - 1) / TILE_WIDTH):
            offset = tile_index * TILE_WIDTH
            values = wp.tile_load(x[row], shape=(TILE_WIDTH,), offset=(offset,))
            wp.tile_store(output[row], wp.tile_map(normalize, values, inverse_norm), offset=(offset,))

    return kernel


_lp_normalization_kernel_cache = {}


def _get_lp_normalization_kernel(width: int, dtype: type):
    tile_width = min(512, max(32, 1 << (width - 1).bit_length()))
    key = (tile_width, dtype)
    if key not in _lp_normalization_kernel_cache:
        _lp_normalization_kernel_cache[key] = _create_lp_normalization_kernel(*key)
    return tile_width, _lp_normalization_kernel_cache[key]


def _create_linear_attention_kernel(key_size: int, value_size: int, dtype: type):
    KEY_SIZE = key_size
    VALUE_SIZE = value_size
    VALUE_TILE = min(64, value_size & -value_size)
    VALUE_BLOCKS = VALUE_SIZE // VALUE_TILE
    DTYPE = dtype

    @wp.func
    def exp_value(value: dtype):
        return dtype(wp.exp(wp.float32(value)))

    @wp.kernel(enable_backward=False)
    def kernel(
        query: wp.array2d(dtype=DTYPE),
        key: wp.array2d(dtype=DTYPE),
        value: wp.array2d(dtype=DTYPE),
        past: wp.array2d(dtype=DTYPE),
        decay: wp.array2d(dtype=DTYPE),
        beta: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        present: wp.array2d(dtype=DTYPE),
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
        item = wp.tid()
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
                    state *= DTYPE(wp.exp(wp.float32(decay[token_row, value_head])))

            key_row = wp.tile_load(key, shape=(1, KEY_SIZE), offset=(token_row, key_head * KEY_SIZE))
            value_row = wp.tile_load(
                value,
                shape=(1, VALUE_TILE),
                offset=(token_row, value_head * VALUE_SIZE + value_offset),
            )
            if needs_beta:
                retrieved = wp.tile_zeros(shape=(1, VALUE_TILE), dtype=DTYPE)
                wp.tile_matmul(key_row, state, retrieved)
                beta_value = beta[token_row, value_head] if beta_per_head else beta[token_row, 0]
                delta = DTYPE(beta_value) * (value_row - retrieved)
            else:
                delta = value_row
            wp.tile_matmul(wp.tile_transpose(key_row), delta, state)

            if query_heads >= value_heads:
                heads_per_group = query_heads / value_heads
                for group in range(heads_per_group):
                    query_head = value_head * heads_per_group + group
                    query_row = wp.tile_load(query, shape=(1, KEY_SIZE), offset=(token_row, query_head * KEY_SIZE))
                    result = wp.tile_zeros(shape=(1, VALUE_TILE), dtype=DTYPE)
                    wp.tile_matmul(query_row, state, result)
                    wp.tile_store(
                        output,
                        DTYPE(scale) * result,
                        offset=(token_row, query_head * VALUE_SIZE + value_offset),
                    )
            else:
                query_head = value_head * query_heads / value_heads
                query_row = wp.tile_load(query, shape=(1, KEY_SIZE), offset=(token_row, query_head * KEY_SIZE))
                result = wp.tile_zeros(shape=(1, VALUE_TILE), dtype=DTYPE)
                wp.tile_matmul(query_row, state, result)
                wp.tile_store(
                    output,
                    DTYPE(scale) * result,
                    offset=(token_row, value_head * VALUE_SIZE + value_offset),
                )

        wp.tile_store(present, state, offset=(state_offset, value_offset))

    return kernel


_linear_attention_kernel_cache = {}


def _get_linear_attention_kernel(key_size: int, value_size: int, dtype: type):
    key = (key_size, value_size, dtype)
    if key not in _linear_attention_kernel_cache:
        _linear_attention_kernel_cache[key] = _create_linear_attention_kernel(*key)
    return _linear_attention_kernel_cache[key]


def _create_swiglu_kernel(dtype: type):
    @wp.kernel(enable_backward=False)
    def kernel(
        gate: wp.array2d(dtype=dtype),
        up: wp.array2d(dtype=dtype),
        output: wp.array2d(dtype=dtype),
    ):
        row, column = wp.tid()
        value = wp.float32(gate[row, column])
        silu = value / (wp.float32(1.0) + wp.exp(-value))
        output[row, column] = dtype(silu * wp.float32(up[row, column]))

    return kernel


_swiglu_kernel_cache = {}


@wp.kernel
def _gqa_copy_past_fp16_kernel(
    past_key: wp.array4d[wp.float16],
    past_value: wp.array4d[wp.float16],
    present_key: wp.array4d[wp.float16],
    present_value: wp.array4d[wp.float16],
):
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


@wp.func
def _gqa_dot_fp16(left: wp.float16, right: wp.float16):
    return wp.float32(left) * wp.float32(right)


@wp.func
def _gqa_accumulate_fp16(total: wp.float32, value: wp.float16, old_scale: wp.float32, weight: wp.float32):
    return total * old_scale + wp.float32(value) * weight


@wp.func
def _gqa_normalize_fp16(total: wp.float32, denominator: wp.float32):
    return wp.float16(total / denominator)


def _create_gqa_attention_kernel(head_size: int):
    @wp.kernel(enable_backward=False)
    def kernel(
        query: wp.array2d[wp.float16],
        key: wp.array2d[wp.float16],
        value: wp.array2d[wp.float16],
        sequence_lengths_minus_one: wp.array1d[wp.int32],
        output: wp.array2d[wp.float16],
        query_heads: int,
        kv_heads: int,
        sequence_length: int,
        total_length: int,
        scale: float,
    ):
        index = wp.tid()
        query_token = index % sequence_length
        head = (index // sequence_length) % query_heads
        batch = index // (sequence_length * query_heads)
        kv_head = head // (query_heads // kv_heads)
        valid_keys = wp.int32(sequence_lengths_minus_one[batch]) - sequence_length + query_token + 2
        query_row = (batch * query_heads + head) * sequence_length + query_token
        query_values = wp.tile_load(query[query_row], shape=(head_size,))
        accumulator = wp.tile_zeros(shape=(head_size,), dtype=wp.float32)
        maximum = wp.float32(-3.402823466e38)
        denominator = wp.float32(0.0)
        for key_token in range(valid_keys):
            cache_row = (batch * kv_heads + kv_head) * total_length + key_token
            key_values = wp.tile_load(key[cache_row], shape=(head_size,))
            score = wp.tile_extract(wp.tile_sum(wp.tile_map(_gqa_dot_fp16, query_values, key_values)), 0)
            score *= wp.float32(scale)
            new_maximum = wp.max(maximum, score)
            old_scale = wp.exp(maximum - new_maximum)
            weight = wp.exp(score - new_maximum)
            denominator = denominator * old_scale + weight
            value_values = wp.tile_load(value[cache_row], shape=(head_size,))
            accumulator = wp.tile_map(_gqa_accumulate_fp16, accumulator, value_values, old_scale, weight)
            maximum = new_maximum
        normalized = wp.tile_map(_gqa_normalize_fp16, accumulator, denominator)
        wp.tile_store(output[batch * sequence_length + query_token], normalized, offset=(head * head_size,))

    return kernel


_gqa_attention_kernel_cache = {}


def _get_gqa_attention_kernel(head_size: int):
    if head_size not in _gqa_attention_kernel_cache:
        _gqa_attention_kernel_cache[head_size] = _create_gqa_attention_kernel(head_size)
    block_dim = min(1024, max(32, 1 << (head_size - 1).bit_length()))
    return block_dim, _gqa_attention_kernel_cache[head_size]


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_ATTR_DECODERS = {
    1: lambda a: a.f,  # FLOAT
    2: lambda a: a.i,  # INT
    3: lambda a: a.s.decode("utf-8") if isinstance(a.s, (bytes, bytearray)) else a.s,  # STRING
    4: lambda a: a.t,  # TENSOR
    7: lambda a: list(a.ints),  # INTS
}


@dataclass
class _Op:
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    attr_names: set[str] = field(default_factory=set)


def _decode_attrs(node) -> tuple[dict[str, Any], set[str]]:
    out: dict[str, Any] = {}
    all_names: set[str] = set()
    for attr in node.attribute:
        all_names.add(attr.name)
        decoder = _ATTR_DECODERS.get(attr.type)
        if decoder is not None:
            out[attr.name] = decoder(attr)
    return out, all_names


def _fuse_inference_ops(ops: list[_Op], graph_outputs: set[str], initializer_names: set[str]) -> list[_Op]:
    """Fuse common inference chains without changing the ONNX artifact."""
    consumers: dict[str, int] = {}
    for op in ops:
        for name in op.inputs:
            consumers[name] = consumers.get(name, 0) + 1
    fused: list[_Op] = []
    index = 0
    while index < len(ops):
        if index + 2 < len(ops):
            sigmoid, silu_mul, output_mul = ops[index : index + 3]
            gate = sigmoid.inputs[0] if len(sigmoid.inputs) == 1 else ""
            sigmoid_out = sigmoid.outputs[0]
            silu_out = silu_mul.outputs[0]
            silu_inputs = set(silu_mul.inputs)
            matches = (
                sigmoid.op_type == "Sigmoid"
                and silu_mul.op_type == "Mul"
                and silu_inputs == {gate, sigmoid_out}
                and output_mul.op_type == "Mul"
                and silu_out in output_mul.inputs
                and consumers.get(sigmoid_out) == 1
                and consumers.get(silu_out) == 1
                and sigmoid_out not in graph_outputs
                and silu_out not in graph_outputs
            )
            if matches:
                up = output_mul.inputs[1] if output_mul.inputs[0] == silu_out else output_mul.inputs[0]
                fused.append(_Op(op_type="_SwiGLU", inputs=[gate, up], outputs=list(output_mul.outputs)))
                index += 3
                continue
        if index + 1 < len(ops):
            norm, relu = ops[index : index + 2]
            norm_out = norm.outputs[0]
            matches = (
                norm.op_type == "BatchNormalization"
                and len(norm.inputs) == 5
                and len(norm.outputs) == 1
                and int(norm.attrs.get("training_mode", 0)) == 0
                and relu.op_type == "Relu"
                and relu.inputs[0] == norm_out
                and consumers.get(norm_out) == 1
                and norm_out not in graph_outputs
            )
            if matches:
                fused.append(
                    _Op(
                        op_type="_BatchNormalizationRelu",
                        inputs=list(norm.inputs),
                        outputs=list(relu.outputs),
                        attrs={"epsilon": norm.attrs.get("epsilon", 1.0e-5)},
                    )
                )
                index += 2
                continue
        if index + 4 < len(ops):
            square, reduce, add, sqrt, divide = ops[index : index + 5]
            x_name = square.inputs[0] if len(square.inputs) == 2 and square.inputs[0] == square.inputs[1] else ""
            square_out = square.outputs[0]
            reduce_out = reduce.outputs[0]
            add_out = add.outputs[0]
            sqrt_out = sqrt.outputs[0]
            epsilon_name = ""
            if len(add.inputs) == 2:
                if add.inputs[0] == reduce_out:
                    epsilon_name = add.inputs[1]
                elif add.inputs[1] == reduce_out:
                    epsilon_name = add.inputs[0]
            matches = (
                square.op_type == "Mul"
                and bool(x_name)
                and reduce.op_type == "ReduceMean"
                and reduce.inputs[0] == square_out
                and tuple(int(axis) for axis in reduce.attrs.get("axes", [])) in ((1,), (-1,))
                and int(reduce.attrs.get("keepdims", 1)) == 1
                and add.op_type == "Add"
                and bool(epsilon_name)
                and sqrt.op_type == "Sqrt"
                and sqrt.inputs[0] == add_out
                and divide.op_type == "Div"
                and divide.inputs == [x_name, sqrt_out]
                and all(consumers.get(name) == 1 for name in (square_out, reduce_out, add_out, sqrt_out))
                and all(name not in graph_outputs for name in (square_out, reduce_out, add_out, sqrt_out))
            )
            if matches:
                output_name = divide.outputs[0]
                scale_name = ""
                consumed = 5
                if index + 5 < len(ops):
                    scale_op = ops[index + 5]
                    if (
                        scale_op.op_type == "Mul"
                        and len(scale_op.inputs) == 2
                        and output_name in scale_op.inputs
                        and consumers.get(output_name) == 1
                        and output_name not in graph_outputs
                    ):
                        candidate = scale_op.inputs[1] if scale_op.inputs[0] == output_name else scale_op.inputs[0]
                        if candidate in initializer_names:
                            scale_name = candidate
                            output_name = scale_op.outputs[0]
                            consumed = 6
                fused.append(
                    _Op(
                        op_type="_RmsNormalization",
                        inputs=[x_name, epsilon_name, scale_name],
                        outputs=[output_name],
                    )
                )
                index += consumed
                continue
        fused.append(ops[index])
        index += 1
    return fused


def _np_to_warp(arr_np: np.ndarray, device: wp.context.Device, requires_grad: bool = False) -> wp.array:
    arr_np = np.ascontiguousarray(arr_np)
    dtype = wp.dtype_from_numpy(arr_np.dtype)
    return wp.array(
        arr_np,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad and dtype in _FLOAT_DTYPES,
    )


class OnnxRuntime:
    """Lightweight ONNX inference engine for graph-capturable MLP policies.

    Args:
        path: Path to an ``.onnx`` file.
        device: Warp device string (e.g. ``"cuda:0"``).  ``None`` uses the
            current default device.
        batch_size: Fixed batch dimension used to pre-allocate intermediate
            buffers.  Defaults to ``1``.
        input_shapes: Optional exact shapes for graph inputs with symbolic
            dimensions, such as a transformer's sequence and KV-cache lengths.
        input_batch_axes: Optional batch-axis override for graph inputs.  If
            an integer is provided, it is applied to every graph input; if a
            dictionary is provided, it maps graph input names to their batch
            axis.  The selected axes are replaced with ``batch_size`` even
            when the ONNX model exported them as fixed dimensions.
        requires_grad: Whether runtime-owned tensors, including initializers
            and intermediate buffers, should allocate gradient storage.  Keep
            this disabled for inference/replay and enable it when computing
            gradients through ONNX runtime outputs.
        use_cublas: Use an available system cuBLAS library for multi-row
            quantized matrix multiplication.  Falls back to Warp when cuBLAS
            is unavailable.
    """

    def __init__(
        self,
        path: str,
        device: str | wp.Device | None = None,
        batch_size: int = 1,
        input_batch_axes: int | dict[str, int] | None = None,
        input_shapes: dict[str, tuple[int, ...]] | None = None,
        requires_grad: bool = False,
        use_cublas: bool = True,
    ):
        self._device = parse_device(device)
        self._requires_grad = requires_grad

        onnx, numpy_helper = _require_onnx()
        model_path = Path(path)
        model = onnx.load(model_path, load_external_data=False)
        try:
            onnx.checker.check_model(model_path)
        except onnx.checker.ValidationError as exc:
            # Qwen's optimized graph places this ORT operator in the default
            # domain, where the ONNX checker cannot resolve its schema.
            if "No Op registered for SimplifiedLayerNormalization" not in str(exc):
                raise ValueError(f"OnnxRuntime: invalid ONNX model: {exc}") from exc
        graph = model.graph

        self._tensors: dict[str, wp.array] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._dtypes: dict[str, type] = {}

        for init in graph.initializer:
            external = onnx.external_data_helper.uses_external_data(init)
            try:
                arr_np = numpy_helper.to_array(init, base_dir=str(model_path.parent))
                tensor = _np_to_warp(arr_np, self._device, requires_grad=self._requires_grad)
            finally:
                if external:
                    init.ClearField("raw_data")
            self._tensors[init.name] = tensor
            self._shapes[init.name] = tuple(tensor.shape)
            self._dtypes[init.name] = tensor.dtype

        initializer_names = {init.name for init in graph.initializer}
        self._initializer_names = initializer_names
        self.input_names: list[str] = [inp.name for inp in graph.input if inp.name not in initializer_names]
        self.output_names: list[str] = [out.name for out in graph.output]
        self._input_dims = {
            inp.name: tuple(
                dim.dim_value if dim.HasField("dim_value") and dim.dim_value > 0 else None
                for dim in inp.type.tensor_type.shape.dim
            )
            for inp in graph.input
            if inp.name not in initializer_names
        }

        if isinstance(input_batch_axes, dict):
            unknown_inputs = set(input_batch_axes) - set(self.input_names)
            if unknown_inputs:
                raise KeyError(
                    f"OnnxRuntime: input_batch_axes references unknown graph inputs {sorted(unknown_inputs)}"
                )
        if input_shapes is not None:
            unknown_inputs = set(input_shapes) - set(self.input_names)
            if unknown_inputs:
                raise KeyError(f"OnnxRuntime: input_shapes references unknown graph inputs {sorted(unknown_inputs)}")

        for inp in graph.input:
            if inp.name in initializer_names:
                continue
            dims = list(inp.type.tensor_type.shape.dim)
            explicit_shape = input_shapes.get(inp.name) if input_shapes is not None else None
            if explicit_shape is not None:
                if len(explicit_shape) != len(dims):
                    raise ValueError(
                        f"OnnxRuntime: input '{inp.name}' shape {explicit_shape} does not match rank {len(dims)}"
                    )
                for axis, (declared, actual) in enumerate(zip(dims, explicit_shape)):
                    if actual < 0 or (declared.HasField("dim_value") and declared.dim_value != actual):
                        raise ValueError(
                            f"OnnxRuntime: input '{inp.name}' shape {explicit_shape} conflicts with dimension {axis}"
                        )
            batch_axis = None
            if input_batch_axes is not None:
                if isinstance(input_batch_axes, dict):
                    batch_axis = input_batch_axes.get(inp.name)
                else:
                    batch_axis = input_batch_axes
                if batch_axis is not None:
                    if batch_axis < 0:
                        batch_axis += len(dims)
                    if batch_axis < 0 or batch_axis >= len(dims):
                        raise ValueError(
                            f"OnnxRuntime: input '{inp.name}' batch axis {batch_axis} is out of range "
                            f"for rank-{len(dims)} input"
                        )
            shape = []
            for axis, d in enumerate(dims):
                if explicit_shape is not None:
                    shape.append(explicit_shape[axis])
                elif axis == batch_axis:
                    shape.append(batch_size)
                elif d.HasField("dim_value") and d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    shape.append(batch_size)
            self._shapes[inp.name] = tuple(shape)
            self._dtypes[inp.name] = _warp_dtype_from_onnx(onnx, inp.type.tensor_type.elem_type)
        self._input_dtypes = {name: self._dtypes[name] for name in self.input_names}

        self._ops: list[_Op] = []
        for node in graph.node:
            decoded, all_names = _decode_attrs(node)
            if node.op_type == "Constant" and "value" in decoded:
                decoded["_value"] = _np_to_warp(
                    numpy_helper.to_array(decoded["value"]),
                    self._device,
                    requires_grad=self._requires_grad,
                )
            self._ops.append(
                _Op(
                    op_type=node.op_type,
                    inputs=list(node.input),
                    outputs=list(node.output),
                    attrs=decoded,
                    attr_names=all_names,
                )
            )
        if not self._requires_grad:
            self._ops = _fuse_inference_ops(self._ops, set(self.output_names), initializer_names)

        self._cublas = try_create_cublas() if use_cublas and self._device.is_cuda else None
        self._matmul_scratch = {}
        self._preallocate_buffers()

    def resize_inputs(self, input_shapes: dict[str, tuple[int, ...]], share_kv_cache: bool = False) -> None:
        """Rebuild shape-dependent buffers while retaining loaded initializers."""
        if set(input_shapes) != set(self.input_names):
            missing = sorted(set(self.input_names) - set(input_shapes))
            extra = sorted(set(input_shapes) - set(self.input_names))
            raise KeyError(f"OnnxRuntime: resize_inputs requires every graph input; missing={missing}, extra={extra}")
        for name, shape in input_shapes.items():
            declared = self._input_dims[name]
            if len(shape) != len(declared) or any(
                actual < 0 or (fixed is not None and actual != fixed) for actual, fixed in zip(shape, declared)
            ):
                raise ValueError(f"OnnxRuntime: input '{name}' shape {shape} conflicts with declared shape {declared}")

        self._tensors = {name: self._tensors[name] for name in self._initializer_names}
        self._shapes = {name: tuple(tensor.shape) for name, tensor in self._tensors.items()}
        self._dtypes = {name: tensor.dtype for name, tensor in self._tensors.items()}
        for name, shape in input_shapes.items():
            self._shapes[name] = tuple(shape)
            self._dtypes[name] = self._input_dtypes[name]
        for op in self._ops:
            op.attrs = {name: value for name, value in op.attrs.items() if not name.startswith("_") or name == "_value"}
            if op.op_type == "GroupQueryAttention":
                op.attrs["_share_cache"] = share_kv_cache
        self._preallocate_buffers()

    def _fork(self, input_shapes: dict[str, tuple[int, ...]], share_kv_cache: bool = False) -> OnnxRuntime:
        """Create another execution plan sharing this runtime's initializers."""
        runtime = object.__new__(OnnxRuntime)
        runtime._device = self._device
        runtime._requires_grad = self._requires_grad
        runtime._initializer_names = self._initializer_names
        runtime.input_names = list(self.input_names)
        runtime.output_names = list(self.output_names)
        runtime._input_dims = dict(self._input_dims)
        runtime._input_dtypes = dict(self._input_dtypes)
        runtime._cublas = self._cublas
        runtime._matmul_scratch = {}
        runtime._tensors = {name: self._tensors[name] for name in self._initializer_names}
        runtime._shapes = {name: tuple(tensor.shape) for name, tensor in runtime._tensors.items()}
        runtime._dtypes = {name: tensor.dtype for name, tensor in runtime._tensors.items()}
        runtime._ops = [
            _Op(
                op_type=op.op_type,
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={name: value for name, value in op.attrs.items() if not name.startswith("_") or name == "_value"},
                attr_names=set(op.attr_names),
            )
            for op in self._ops
        ]
        runtime.resize_inputs(input_shapes, share_kv_cache=share_kv_cache)
        return runtime

    def _preallocate_buffers(self) -> None:
        for op in self._ops:
            handler = _SHAPE_DISPATCH.get(op.op_type)
            if handler is None:
                supported = sorted(name for name in _OP_DISPATCH if not name.startswith("_"))
                raise NotImplementedError(f"OnnxRuntime: unsupported op '{op.op_type}'.  Supported ops: {supported}")
            handler(op, self._shapes, self._dtypes, self._tensors, self._device, self._requires_grad)

        matmuls = [op for op in self._ops if op.op_type == "MatMulNBits" and op.attrs["_rows"] > 1]
        if self._cublas is not None:
            for dtype in (wp.float16, wp.bfloat16):
                typed_matmuls = [op for op in matmuls if op.attrs["_dtype"] == dtype]
                if not typed_matmuls:
                    continue
                scratch_elements = max(int(op.attrs["K"]) * int(op.attrs["N"]) for op in typed_matmuls)
                scratch = self._matmul_scratch.get(dtype)
                if scratch is None or scratch.size < scratch_elements:
                    scratch = wp.empty(scratch_elements, dtype=dtype, device=self._device)
                    self._matmul_scratch[dtype] = scratch
                for op in typed_matmuls:
                    K = int(op.attrs["K"])
                    N = int(op.attrs["N"])
                    op.attrs["_cublas"] = self._cublas
                    op.attrs["_dequantized_weights"] = scratch[: K * N].reshape((N, K))

    def __call__(self, inputs: dict[str, wp.array]) -> dict[str, wp.array]:
        """Run forward inference.

        Args:
            inputs: Mapping of ONNX input names to Warp arrays already on
                the correct device.  2-D ``wp.array2d`` is the typical case.

        Returns:
            Mapping of ONNX output names to Warp result arrays.
        """
        tensors = self._tensors

        declared_inputs = set(self.input_names)
        for name in inputs:
            if name not in declared_inputs:
                raise KeyError(f"OnnxRuntime: unknown input '{name}'")

        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"OnnxRuntime: missing input '{name}'")
            arr = inputs[name]
            expected_shape = self._shapes[name]
            if tuple(arr.shape) != expected_shape:
                raise ValueError(f"OnnxRuntime: input '{name}' has shape {tuple(arr.shape)}, expected {expected_shape}")
            expected_dtype = self._dtypes[name]
            if arr.dtype != expected_dtype:
                raise TypeError(
                    f"OnnxRuntime: input '{name}' has dtype '{arr.dtype.__name__}', "
                    f"expected '{expected_dtype.__name__}'"
                )
            tensors[name] = arr

        for op in self._ops:
            dispatch = _OP_DISPATCH.get(op.op_type)
            if dispatch is None:
                raise NotImplementedError(f"OnnxRuntime: unsupported op '{op.op_type}'")
            dispatch(op, tensors, self._shapes, self._device)

        return {name: tensors[name] for name in self.output_names}


def _shape_gemm(op, shapes, dtypes, tensors, device, requires_grad=False):
    A_shape = shapes[op.inputs[0]]
    B_shape = shapes[op.inputs[1]]
    transA = int(op.attrs.get("transA", 0))
    transB = int(op.attrs.get("transB", 0))
    if transA:
        raise NotImplementedError("OnnxRuntime Gemm: transA=1 is not graph-capturable in this runtime")
    if transB != 1:
        raise NotImplementedError("OnnxRuntime Gemm: only transB=1 policy weights are supported")
    if len(op.inputs) < 3 or not op.inputs[2]:
        raise NotImplementedError("OnnxRuntime Gemm: bias input is required for graph-capturable policy execution")
    if len(A_shape) != 2 or len(B_shape) != 2:
        raise NotImplementedError("OnnxRuntime Gemm: only 2-D tensors are supported")
    M = A_shape[0]
    N = B_shape[0]
    K = A_shape[1]
    if B_shape[1] != K:
        raise ValueError(f"OnnxRuntime Gemm: incompatible shapes {A_shape} and {B_shape}")
    bias_shape = shapes[op.inputs[2]]
    if bias_shape != (N,):
        raise ValueError(f"OnnxRuntime Gemm: bias '{op.inputs[2]}' has shape {bias_shape}, expected {(N,)}")
    dtype = _require_matching_float_dtypes(op, dtypes, op.inputs[:3])
    out_shape = (M, N)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(out_shape, dtype=dtype, device=device, requires_grad=requires_grad)
    shapes[out_name] = out_shape
    dtypes[out_name] = dtype
    op.attrs["_bias_2d"] = tensors[op.inputs[2]].reshape((N, 1))
    op.attrs["_requires_grad"] = requires_grad


def _shape_elementwise_unary(op, shapes, dtypes, tensors, device, requires_grad=False):
    in_shape = shapes[op.inputs[0]]
    dtype = _require_matching_float_dtypes(op, dtypes, [op.inputs[0]])
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(in_shape, dtype=dtype, device=device, requires_grad=requires_grad)
    shapes[out_name] = in_shape
    dtypes[out_name] = dtype
    width = in_shape[-1] if in_shape else 1
    op.attrs["_shape_2d"] = (int(np.prod(in_shape)) // width, width)


def _shape_elementwise_binary(op, shapes, dtypes, tensors, device, requires_grad=False):
    lhs_shape = shapes[op.inputs[0]]
    rhs_shape = shapes[op.inputs[1]]
    rank = max(len(lhs_shape), len(rhs_shape))
    lhs_aligned = (1,) * (rank - len(lhs_shape)) + lhs_shape
    rhs_aligned = (1,) * (rank - len(rhs_shape)) + rhs_shape
    out_shape = []
    for lhs_size, rhs_size in zip(lhs_aligned, rhs_aligned):
        if lhs_size != rhs_size and lhs_size != 1 and rhs_size != 1:
            raise ValueError(f"OnnxRuntime {op.op_type}: shapes {lhs_shape} and {rhs_shape} do not broadcast")
        out_shape.append(max(lhs_size, rhs_size))
    out_shape = tuple(out_shape)
    width = out_shape[-1] if out_shape else 1
    out_rows = int(np.prod(out_shape)) // width

    def as_2d(aligned_shape):
        prefix = aligned_shape[:-1]
        if prefix == out_shape[:-1]:
            rows = out_rows
        elif all(size == 1 for size in prefix):
            rows = 1
        else:
            raise NotImplementedError(
                f"OnnxRuntime {op.op_type}: broadcast pattern {lhs_shape} with {rhs_shape} is not supported"
            )
        return (rows, aligned_shape[-1] if aligned_shape else 1)

    lhs_2d = as_2d(lhs_aligned)
    rhs_2d = as_2d(rhs_aligned)
    out_2d = (out_rows, width)
    if requires_grad and (lhs_aligned != out_shape or rhs_aligned != out_shape):
        raise NotImplementedError(f"OnnxRuntime {op.op_type}: broadcast gradients are not supported deterministically")
    dtype = dtypes[op.inputs[0]]
    if dtypes[op.inputs[1]] != dtype or (dtype not in _FLOAT_DTYPES and dtype not in (wp.int32, wp.int64)):
        raise TypeError(f"OnnxRuntime {op.op_type}: input dtypes must match")
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(out_shape, dtype=dtype, device=device, requires_grad=requires_grad)
    shapes[out_name] = out_shape
    dtypes[out_name] = dtype
    op.attrs["_out_shape_2d"] = out_2d
    op.attrs["_lhs_shape_2d"] = lhs_2d
    op.attrs["_rhs_shape_2d"] = rhs_2d


def _shape_reduce_mean(op, shapes, dtypes, tensors, device, requires_grad=False):
    in_shape = shapes[op.inputs[0]]
    axes = tuple(int(axis) for axis in op.attrs.get("axes", []))
    keepdims = int(op.attrs.get("keepdims", 1))
    if len(in_shape) != 2 or axes not in ((1,), (-1,)) or keepdims != 1:
        raise NotImplementedError("OnnxRuntime ReduceMean: only 2-D row reductions with keepdims=1 are supported")
    dtype = _require_matching_float_dtypes(op, dtypes, [op.inputs[0]])
    out_shape = (in_shape[0], 1)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(out_shape, dtype=dtype, device=device, requires_grad=requires_grad)
    shapes[out_name] = out_shape
    dtypes[out_name] = dtype


def _shape_lp_normalization(op, shapes, dtypes, tensors, device, requires_grad=False):
    in_shape = shapes[op.inputs[0]]
    axis = int(op.attrs.get("axis", -1))
    if axis < 0:
        axis += len(in_shape)
    if axis != len(in_shape) - 1 or int(op.attrs.get("p", 2)) != 2:
        raise NotImplementedError("OnnxRuntime LpNormalization: only last-axis L2 normalization is supported")
    dtype = _require_matching_float_dtypes(op, dtypes, [op.inputs[0]])
    width = in_shape[-1]
    rows = int(np.prod(in_shape[:-1]))
    tensors[op.outputs[0]] = wp.zeros(in_shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = in_shape
    dtypes[op.outputs[0]] = dtype
    op.attrs["_rows"] = rows
    op.attrs["_width"] = width
    op.attrs["_tile_width"], op.attrs["_kernel"] = _get_lp_normalization_kernel(width, dtype)


def _shape_reduce_sum(op, shapes, dtypes, tensors, device, requires_grad=False):
    axes = tuple(int(value) for value in tensors[op.inputs[1]].numpy().reshape(-1))
    in_shape = shapes[op.inputs[0]]
    if len(in_shape) != 2 or axes != (1,) or int(op.attrs.get("keepdims", 1)) != 0:
        raise NotImplementedError("OnnxRuntime ReduceSum: only Qwen's 2-D integer row reduction is supported")
    dtype = dtypes[op.inputs[0]]
    if dtype not in (wp.int32, wp.int64):
        raise TypeError("OnnxRuntime ReduceSum: expected an INT32 or INT64 input")
    out_shape = (in_shape[0],)
    tensors[op.outputs[0]] = wp.zeros(out_shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = out_shape
    dtypes[op.outputs[0]] = dtype


def _shape_reduce_max(op, shapes, dtypes, tensors, device, requires_grad=False):
    in_shape = shapes[op.inputs[0]]
    if len(in_shape) != 1 or int(op.attrs.get("keepdims", 1)) != 0:
        raise NotImplementedError("OnnxRuntime ReduceMax: only full 1-D reductions are supported")
    dtype = dtypes[op.inputs[0]]
    tensors[op.outputs[0]] = wp.zeros(1, dtype=dtype, device=device)
    shapes[op.outputs[0]] = ()
    dtypes[op.outputs[0]] = dtype
    op.attrs["_kernel"] = _kernel_for_dtype(_reduce_max_1d_kernel, dtype, (1,), (1,))


def _shape_shape(op, shapes, dtypes, tensors, device, requires_grad=False):
    if op.attr_names:
        raise NotImplementedError("OnnxRuntime Shape: start/end attributes are not supported")
    value = np.asarray(shapes[op.inputs[0]], dtype=np.int64)
    tensors[op.outputs[0]] = _np_to_warp(value, device)
    shapes[op.outputs[0]] = value.shape
    dtypes[op.outputs[0]] = wp.int64


def _shape_gather(op, shapes, dtypes, tensors, device, requires_grad=False):
    if int(op.attrs.get("axis", 0)) != 0:
        raise NotImplementedError("OnnxRuntime Gather: only constant axis-0 gathers are supported")
    if (
        op.inputs[0] in tensors
        and len(shapes[op.inputs[0]]) == 2
        and len(shapes[op.inputs[1]]) == 2
        and dtypes[op.inputs[0]] in _FLOAT_DTYPES
        and dtypes[op.inputs[1]] == wp.int64
    ):
        out_shape = (*shapes[op.inputs[1]], shapes[op.inputs[0]][1])
        tensors[op.outputs[0]] = wp.zeros(out_shape, dtype=dtypes[op.inputs[0]], device=device)
        shapes[op.outputs[0]] = out_shape
        dtypes[op.outputs[0]] = dtypes[op.inputs[0]]
        op.attrs["_dynamic"] = True
        return
    if op.inputs[0] not in tensors or op.inputs[1] not in tensors:
        raise NotImplementedError("OnnxRuntime Gather: unsupported dynamic gather")
    value = np.take(tensors[op.inputs[0]].numpy(), tensors[op.inputs[1]].numpy().astype(np.int64), axis=0)
    tensors[op.outputs[0]] = _np_to_warp(value, device)
    shapes[op.outputs[0]] = tuple(tensors[op.outputs[0]].shape)
    dtypes[op.outputs[0]] = tensors[op.outputs[0]].dtype


def _shape_cast(op, shapes, dtypes, tensors, device, requires_grad=False):
    target_dtype = {
        1: wp.float32,
        6: wp.int32,
        10: wp.float16,
        16: wp.bfloat16,
    }.get(int(op.attrs.get("to", 0)))
    if target_dtype is None:
        raise NotImplementedError("OnnxRuntime Cast: unsupported target dtype")
    shape = shapes[op.inputs[0]]
    tensors[op.outputs[0]] = wp.zeros(shape, dtype=target_dtype, device=device)
    shapes[op.outputs[0]] = shape
    dtypes[op.outputs[0]] = target_dtype
    op.attrs["_kernel"] = _cast_kernel_for_dtypes(dtypes[op.inputs[0]], target_dtype)


def _shape_batch_normalization(op, shapes, dtypes, tensors, device, requires_grad=False):
    if requires_grad:
        raise NotImplementedError("OnnxRuntime BatchNormalization: deterministic gradients are not supported")
    if len(op.inputs) != 5:
        raise NotImplementedError("OnnxRuntime BatchNormalization: training inputs are not supported")
    in_shape = shapes[op.inputs[0]]
    if len(in_shape) != 2:
        raise NotImplementedError("OnnxRuntime BatchNormalization: only 2-D tensors are supported")
    width = in_shape[1]
    for name in op.inputs[1:]:
        if shapes[name] != (width,):
            raise ValueError(
                f"OnnxRuntime BatchNormalization: parameter '{name}' has shape {shapes[name]}, expected {(width,)}"
            )
    if int(op.attrs.get("training_mode", 0)) != 0:
        raise NotImplementedError("OnnxRuntime BatchNormalization: training mode is not supported")
    dtype = _require_matching_float_dtypes(op, dtypes, op.inputs)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(in_shape, dtype=dtype, device=device, requires_grad=requires_grad)
    shapes[out_name] = in_shape
    dtypes[out_name] = dtype


def _shape_rms_normalization(op, shapes, dtypes, tensors, device, requires_grad=False):
    if requires_grad:
        raise RuntimeError("internal inference fusion cannot require gradients")
    shape = shapes[op.inputs[0]]
    if len(shape) != 2:
        raise ValueError(f"OnnxRuntime fused RMS normalization requires a 2-D input, got {shape}")
    if shapes[op.inputs[1]] != (1,):
        raise ValueError("OnnxRuntime fused RMS normalization epsilon must have shape (1,)")
    width = shape[1]
    dtype_names = [name for name in op.inputs if name]
    dtype = _require_matching_float_dtypes(op, dtypes, dtype_names)
    if op.inputs[2]:
        if shapes[op.inputs[2]] != (width,):
            raise ValueError("OnnxRuntime fused RMS normalization scale has invalid shape")
        op.attrs["_scale"] = tensors[op.inputs[2]]
    else:
        op.attrs["_scale"] = wp.ones(width, dtype=dtype, device=device)
    kernel = _create_rms_normalization_kernel(width)
    op.attrs["_kernel"] = _kernel_for_dtype(kernel, dtype, (2,), (1,), (1,), (2,))
    tensors[op.outputs[0]] = wp.zeros(shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = shape
    dtypes[op.outputs[0]] = dtype


def _shape_constant(op, shapes, dtypes, tensors, device, requires_grad=False):
    if set(op.attr_names) != {"value"}:
        raise NotImplementedError("OnnxRuntime Constant: only the tensor-valued 'value' attribute is supported")
    value = op.attrs["_value"]
    tensors[op.outputs[0]] = value
    shapes[op.outputs[0]] = tuple(value.shape)
    dtypes[op.outputs[0]] = value.dtype


def _shape_range(op, shapes, dtypes, tensors, device, requires_grad=False):
    if len(op.inputs) != 3 or any(name not in tensors for name in op.inputs):
        raise NotImplementedError("OnnxRuntime Range: inputs must be construction-time constants")
    dtype = dtypes[op.inputs[0]]
    if any(dtypes[name] != dtype for name in op.inputs):
        raise TypeError("OnnxRuntime Range: input dtypes must match")
    values = [tensor.numpy().reshape(-1)[0] for tensor in (tensors[name] for name in op.inputs)]
    result = np.arange(*values, dtype=wp.dtype_to_numpy(dtype))
    tensors[op.outputs[0]] = _np_to_warp(result, device)
    shapes[op.outputs[0]] = result.shape
    dtypes[op.outputs[0]] = dtype


def _shape_slice(op, shapes, dtypes, tensors, device, requires_grad=False):
    if any(name not in tensors for name in op.inputs):
        raise NotImplementedError("OnnxRuntime Slice: inputs must be construction-time constants")
    data = tensors[op.inputs[0]].numpy()
    starts = tensors[op.inputs[1]].numpy().reshape(-1)
    ends = tensors[op.inputs[2]].numpy().reshape(-1)
    axes = tensors[op.inputs[3]].numpy().reshape(-1) if len(op.inputs) > 3 and op.inputs[3] else np.arange(len(starts))
    steps = (
        tensors[op.inputs[4]].numpy().reshape(-1)
        if len(op.inputs) > 4 and op.inputs[4]
        else np.ones(len(starts), dtype=np.int64)
    )
    slices = [slice(None)] * data.ndim
    for start, end, axis, step in zip(starts, ends, axes, steps):
        slices[int(axis)] = slice(int(start), int(end), int(step))
    result = np.ascontiguousarray(data[tuple(slices)])
    tensors[op.outputs[0]] = _np_to_warp(result, device)
    shapes[op.outputs[0]] = result.shape
    dtypes[op.outputs[0]] = dtypes[op.inputs[0]]


def _shape_where(op, shapes, dtypes, tensors, device, requires_grad=False):
    condition_shape, x_shape, y_shape = (shapes[name] for name in op.inputs)
    if x_shape != y_shape or len(x_shape) < 1:
        raise NotImplementedError("OnnxRuntime Where: data inputs must have the same non-scalar shape")
    if condition_shape not in (x_shape, (x_shape[-1],)):
        raise NotImplementedError("OnnxRuntime Where: condition must match the data or its last axis")
    if dtypes[op.inputs[0]] != wp.bool or dtypes[op.inputs[1]] != dtypes[op.inputs[2]]:
        raise TypeError("OnnxRuntime Where: expected a boolean condition and matching data dtypes")
    dtype = dtypes[op.inputs[1]]
    rows = int(np.prod(x_shape[:-1]))
    width = x_shape[-1]
    tensors[op.outputs[0]] = wp.zeros(x_shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = x_shape
    dtypes[op.outputs[0]] = dtype
    op.attrs["_shape_2d"] = (rows, width)
    op.attrs["_condition_shape_2d"] = (rows, width) if condition_shape == x_shape else (1, width)
    op.attrs["_kernel"] = _where_kernel_for_dtype(dtype)


def _shape_rotary_embedding(op, shapes, dtypes, tensors, device, requires_grad=False):
    if requires_grad:
        raise NotImplementedError("OnnxRuntime RotaryEmbedding: gradients are not supported")
    shape = shapes[op.inputs[0]]
    if len(shape) != 4:
        raise NotImplementedError("OnnxRuntime RotaryEmbedding: expected BNSH input layout")
    batch, heads, sequence, head_size = shape
    num_heads = int(op.attrs.get("num_heads", 0))
    rotary_dim = int(op.attrs.get("rotary_embedding_dim", head_size))
    if num_heads not in (0, heads) or rotary_dim <= 0 or rotary_dim > head_size or rotary_dim % 2:
        raise ValueError("OnnxRuntime RotaryEmbedding: invalid head count or rotary dimension")
    position_shape = shapes[op.inputs[1]]
    if position_shape == (1,):
        position_shape_2d = (1, 1)
        position_offset = True
    elif position_shape == (batch, sequence):
        position_shape_2d = position_shape
        position_offset = False
    else:
        raise ValueError("OnnxRuntime RotaryEmbedding: position IDs must be a scalar offset or [batch, sequence]")
    cache_shape = shapes[op.inputs[2]]
    if cache_shape != shapes[op.inputs[3]] or len(cache_shape) != 2 or cache_shape[1] != rotary_dim // 2:
        raise ValueError("OnnxRuntime RotaryEmbedding: cosine and sine cache shapes do not match")
    if dtypes[op.inputs[1]] != wp.int64:
        raise TypeError("OnnxRuntime RotaryEmbedding: position IDs must be INT64")
    dtype = _require_matching_float_dtypes(op, dtypes, [op.inputs[0], op.inputs[2], op.inputs[3]])
    tensors[op.outputs[0]] = wp.zeros(shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = shape
    dtypes[op.outputs[0]] = dtype
    op.attrs["_position_shape_2d"] = position_shape_2d
    op.attrs["_position_offset"] = position_offset
    op.attrs["_kernel"] = _rotary_embedding_kernel_for_dtype(dtype)


def _shape_reshape(op, shapes, dtypes, tensors, device, requires_grad=False):
    if len(op.inputs) != 2 or op.inputs[1] not in tensors:
        raise NotImplementedError("OnnxRuntime Reshape: the shape input must be constant")

    in_shape = shapes[op.inputs[0]]
    requested = [int(value) for value in tensors[op.inputs[1]].numpy().reshape(-1)]
    allowzero = int(op.attrs.get("allowzero", 0))
    out_shape = []
    inferred_axis = None
    for axis, dimension in enumerate(requested):
        if dimension == 0 and not allowzero:
            if axis >= len(in_shape):
                raise ValueError("OnnxRuntime Reshape: a copied dimension is outside the input rank")
            dimension = in_shape[axis]
        elif dimension == -1:
            if inferred_axis is not None:
                raise ValueError("OnnxRuntime Reshape: at most one dimension may be inferred")
            inferred_axis = axis
            dimension = 1
        elif dimension < 0:
            raise ValueError(f"OnnxRuntime Reshape: invalid dimension {dimension}")
        out_shape.append(dimension)

    input_size = int(np.prod(in_shape))
    known_size = int(np.prod(out_shape))
    if inferred_axis is not None:
        if 0 in out_shape:
            raise ValueError("OnnxRuntime Reshape: zero dimensions cannot be combined with an inferred dimension")
        if known_size == 0 or input_size % known_size:
            raise ValueError("OnnxRuntime Reshape: input size is not divisible by the requested shape")
        out_shape[inferred_axis] = input_size // known_size
    elif known_size != input_size:
        raise ValueError("OnnxRuntime Reshape: input and output shapes must have the same number of elements")

    op.attrs["_out_shape"] = tuple(out_shape)
    shapes[op.outputs[0]] = tuple(out_shape)
    dtypes[op.outputs[0]] = dtypes[op.inputs[0]]


def _shape_gather_block_quantized(op, shapes, dtypes, tensors, device, requires_grad=False):
    if requires_grad:
        raise NotImplementedError("OnnxRuntime GatherBlockQuantized: gradients are not supported")
    if int(op.attrs.get("bits", 4)) != 8 or int(op.attrs.get("block_size", 128)) != 128:
        raise NotImplementedError("OnnxRuntime GatherBlockQuantized: only 8-bit blocks of 128 are supported")
    if len(op.inputs) != 4:
        raise NotImplementedError("OnnxRuntime GatherBlockQuantized: zero points are required")

    data_shape = shapes[op.inputs[0]]
    indices_shape = shapes[op.inputs[1]]
    scales_shape = shapes[op.inputs[2]]
    zero_points_shape = shapes[op.inputs[3]]
    if len(data_shape) != 2 or len(indices_shape) != 2:
        raise NotImplementedError("OnnxRuntime GatherBlockQuantized: only 2-D data and indices are supported")
    expected_quant_shape = (data_shape[0], (data_shape[1] + 127) // 128)
    if scales_shape != expected_quant_shape or zero_points_shape != expected_quant_shape:
        raise ValueError(
            "OnnxRuntime GatherBlockQuantized: scales and zero points must have shape " f"{expected_quant_shape}"
        )
    expected_dtypes = (wp.uint8, wp.int64, wp.float16, wp.uint8)
    actual_dtypes = tuple(dtypes[name] for name in op.inputs)
    if actual_dtypes != expected_dtypes:
        raise TypeError(
            "OnnxRuntime GatherBlockQuantized: expected uint8, int64, float16, uint8 inputs, "
            f"got {tuple(dtype.__name__ for dtype in actual_dtypes)}"
        )

    out_shape = (*indices_shape, data_shape[1])
    tensors[op.outputs[0]] = wp.zeros(out_shape, dtype=wp.float16, device=device)
    shapes[op.outputs[0]] = out_shape
    dtypes[op.outputs[0]] = wp.float16


def _shape_matmul_nbits(op, shapes, dtypes, tensors, device, requires_grad=False):
    if requires_grad:
        raise NotImplementedError("OnnxRuntime MatMulNBits: gradients are not supported")
    bits = int(op.attrs.get("bits", 4))
    block_size = int(op.attrs.get("block_size", 128))
    if bits not in (4, 8) or block_size < 16 or block_size & (block_size - 1):
        raise NotImplementedError(
            "OnnxRuntime MatMulNBits: only 4/8-bit power-of-two blocks of at least 16 are supported"
        )
    if int(op.attrs.get("accuracy_level", 0)) not in (0, 4) or len(op.inputs) not in (3, 4):
        raise NotImplementedError("OnnxRuntime MatMulNBits: unsupported accuracy level or input count")

    activation_shape = shapes[op.inputs[0]]
    if len(activation_shape) not in (2, 3):
        raise NotImplementedError("OnnxRuntime MatMulNBits: only 2-D and 3-D activations are supported")
    K = int(op.attrs.get("K", activation_shape[-1]))
    N = int(op.attrs.get("N", shapes[op.inputs[1]][0]))
    if activation_shape[-1] != K:
        raise ValueError(f"OnnxRuntime MatMulNBits: activation width is {activation_shape[-1]}, expected {K}")

    if K % block_size:
        raise NotImplementedError("OnnxRuntime MatMulNBits: partial quantization blocks are not supported")
    blocks = K // block_size
    packed_block = block_size * bits // 8
    expected_weight_shape = (N, blocks, packed_block)
    expected_scale_shape = (N, blocks)
    expected_zero_shape = (N, (blocks * bits + 7) // 8)
    if shapes[op.inputs[1]] != expected_weight_shape:
        raise ValueError(f"OnnxRuntime MatMulNBits: weights must have shape {expected_weight_shape}")
    has_zero_points = len(op.inputs) == 4 and bool(op.inputs[3])
    if shapes[op.inputs[2]] != expected_scale_shape or (
        has_zero_points and shapes[op.inputs[3]] != expected_zero_shape
    ):
        raise ValueError(
            f"OnnxRuntime MatMulNBits: scales and zero points must have shapes "
            f"{expected_scale_shape} and {expected_zero_shape}"
        )
    dtype = dtypes[op.inputs[0]]
    actual_dtypes = tuple(dtypes[name] for name in op.inputs if name)
    expected_dtypes = (dtype, wp.uint8, dtype, wp.uint8) if has_zero_points else (dtype, wp.uint8, dtype)
    if dtype not in (wp.float16, wp.bfloat16) or actual_dtypes != expected_dtypes:
        raise TypeError(
            "OnnxRuntime MatMulNBits: expected matching FP16/BF16 activations and scales with uint8 weights, "
            f"got {tuple(dtype.__name__ for dtype in actual_dtypes)}"
        )

    rows = int(np.prod(activation_shape[:-1]))
    out_shape = (*activation_shape[:-1], N)
    tensors[op.outputs[0]] = wp.zeros(out_shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = out_shape
    dtypes[op.outputs[0]] = dtype
    zero_name = "__onnx_runtime_default_nbits_zero"
    if not has_zero_points and zero_name not in tensors:
        tensors[zero_name] = wp.zeros((1, 1), dtype=wp.uint8, device=device)
    op.attrs["_rows"] = rows
    op.attrs["K"] = K
    op.attrs["N"] = N
    op.attrs["_block_size"] = block_size
    op.attrs["_dtype"] = dtype
    op.attrs["_has_zero_points"] = has_zero_points
    op.attrs["_zero_points"] = tensors[op.inputs[3]] if has_zero_points else tensors[zero_name]
    op.attrs["_output_2d"] = tensors[op.outputs[0]].reshape((rows, N))


def _shape_causal_conv_with_state(op, shapes, dtypes, tensors, device, requires_grad=False):
    if requires_grad:
        raise NotImplementedError("OnnxRuntime CausalConvWithState: gradients are not supported")
    if int(op.attrs.get("ndim", 1)) != 1:
        raise NotImplementedError("OnnxRuntime CausalConvWithState: only 1-D convolution is supported")
    activation = op.attrs.get("activation", "none").lower()
    if activation not in ("none", "silu", "swish"):
        raise NotImplementedError(f"OnnxRuntime CausalConvWithState: unsupported activation '{activation}'")

    x_shape = shapes[op.inputs[0]]
    weight_shape = shapes[op.inputs[1]]
    if len(x_shape) != 3 or len(weight_shape) != 3 or weight_shape[:2] != (x_shape[1], 1):
        raise ValueError(
            "OnnxRuntime CausalConvWithState: expected input [batch, channels, sequence] and "
            "weights [channels, 1, kernel]"
        )
    batch, channels, sequence_length = x_shape
    kernel_size = weight_shape[2]
    if sequence_length < 1 or kernel_size < 1:
        raise ValueError("OnnxRuntime CausalConvWithState: sequence and kernel sizes must be positive")

    has_bias = len(op.inputs) > 2 and bool(op.inputs[2])
    has_past = len(op.inputs) > 3 and bool(op.inputs[3])
    dtype_names = [op.inputs[0], op.inputs[1]]
    if has_bias:
        if shapes[op.inputs[2]] != (channels,):
            raise ValueError(f"OnnxRuntime CausalConvWithState: bias must have shape {(channels,)}")
        dtype_names.append(op.inputs[2])
    state_shape = (batch, channels, kernel_size - 1)
    if has_past:
        if shapes[op.inputs[3]] != state_shape:
            raise ValueError(f"OnnxRuntime CausalConvWithState: past state must have shape {state_shape}")
        dtype_names.append(op.inputs[3])
    dtype = _require_matching_float_dtypes(op, dtypes, dtype_names)

    bias = tensors[op.inputs[2]] if has_bias else wp.zeros(1, dtype=dtype, device=device)
    past = wp.zeros(state_shape, dtype=dtype, device=device) if not has_past else None
    tensors[op.outputs[0]] = wp.zeros(x_shape, dtype=dtype, device=device)
    tensors[op.outputs[1]] = wp.zeros(state_shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = x_shape
    shapes[op.outputs[1]] = state_shape
    dtypes[op.outputs[0]] = dtype
    dtypes[op.outputs[1]] = dtype
    op.attrs["_kernel_size"] = kernel_size
    op.attrs["_bias"] = bias
    op.attrs["_past"] = past
    op.attrs["_has_past"] = has_past
    op.attrs["_has_bias"] = has_bias
    op.attrs["_silu"] = activation != "none"
    op.attrs["_kernel"] = _kernel_for_dtype(
        _causal_conv_1d_kernel, dtype, (3,), (3,), (1,), (3,), (3,), int, bool, bool
    )
    op.attrs["_state_kernel"] = _kernel_for_dtype(_causal_conv_state_kernel, dtype, (3,), (3,), (3,))


def _shape_linear_attention(op, shapes, dtypes, tensors, device, requires_grad=False):
    if requires_grad:
        raise NotImplementedError("OnnxRuntime LinearAttention: gradients are not supported")
    if len(op.inputs) < 3 or not op.inputs[1] or not op.inputs[2]:
        raise NotImplementedError("OnnxRuntime LinearAttention: separate key and value inputs are required")
    query_shape, key_shape, value_shape = (shapes[name] for name in op.inputs[:3])
    if (
        any(len(shape) != 3 for shape in (query_shape, key_shape, value_shape))
        or key_shape[:2] != query_shape[:2]
        or value_shape[:2] != query_shape[:2]
    ):
        raise ValueError("OnnxRuntime LinearAttention: query, key, and value must be matching rank-3 tensors")

    batch, sequence_length, query_hidden = query_shape
    query_heads = int(op.attrs.get("q_num_heads", 0))
    value_heads = int(op.attrs.get("kv_num_heads", 0))
    if query_heads < 1 or value_heads < 1 or query_hidden % query_heads:
        raise ValueError("OnnxRuntime LinearAttention: invalid q_num_heads or kv_num_heads")
    key_size = query_hidden // query_heads
    if key_shape[2] % key_size or value_shape[2] % value_heads:
        raise ValueError("OnnxRuntime LinearAttention: key or value hidden size is not divisible by its head size")
    key_heads = key_shape[2] // key_size
    value_size = value_shape[2] // value_heads
    if value_heads % key_heads or max(query_heads, value_heads) % min(query_heads, value_heads):
        raise ValueError("OnnxRuntime LinearAttention: incompatible query, key, and value head counts")

    update_rule = op.attrs.get("update_rule", "gated_delta").lower()
    if update_rule not in ("linear", "gated", "delta", "gated_delta"):
        raise NotImplementedError(f"OnnxRuntime LinearAttention: unsupported update rule '{update_rule}'")
    needs_decay = update_rule in ("gated", "gated_delta")
    needs_beta = update_rule in ("delta", "gated_delta")
    has_past = len(op.inputs) > 3 and bool(op.inputs[3])
    has_decay = len(op.inputs) > 4 and bool(op.inputs[4])
    has_beta = len(op.inputs) > 5 and bool(op.inputs[5])
    if needs_decay != has_decay or needs_beta != has_beta:
        raise ValueError(f"OnnxRuntime LinearAttention: update rule '{update_rule}' has inconsistent gate inputs")

    state_shape = (batch, value_heads, key_size, value_size)
    dtype_names = list(op.inputs[:3])
    if has_past:
        if shapes[op.inputs[3]] != state_shape:
            raise ValueError(f"OnnxRuntime LinearAttention: past state must have shape {state_shape}")
        dtype_names.append(op.inputs[3])
    decay_per_key = False
    if has_decay:
        decay_shape = shapes[op.inputs[4]]
        if decay_shape == (batch, sequence_length, value_heads * key_size):
            decay_per_key = True
        elif decay_shape != (batch, sequence_length, value_heads):
            raise ValueError("OnnxRuntime LinearAttention: invalid decay shape")
        dtype_names.append(op.inputs[4])
    beta_per_head = False
    if has_beta:
        beta_shape = shapes[op.inputs[5]]
        if beta_shape == (batch, sequence_length, value_heads):
            beta_per_head = True
        elif beta_shape != (batch, sequence_length, 1):
            raise ValueError("OnnxRuntime LinearAttention: invalid beta shape")
        dtype_names.append(op.inputs[5])
    dtype = _require_matching_float_dtypes(op, dtypes, dtype_names)
    if dtype not in (wp.float16, wp.bfloat16, wp.float32):
        raise NotImplementedError("OnnxRuntime LinearAttention: only FP16, BF16, and FP32 are supported")

    output_heads = max(query_heads, value_heads)
    output_shape = (batch, sequence_length, output_heads * value_size)
    tensors[op.outputs[0]] = wp.zeros(output_shape, dtype=dtype, device=device)
    tensors[op.outputs[1]] = wp.zeros(state_shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = output_shape
    shapes[op.outputs[1]] = state_shape
    dtypes[op.outputs[0]] = dtype
    dtypes[op.outputs[1]] = dtype

    op.attrs["_kernel"] = _get_linear_attention_kernel(key_size, value_size, dtype)
    op.attrs["_block_dim"] = 256
    op.attrs["_batch"] = batch
    op.attrs["_sequence_length"] = sequence_length
    op.attrs["_query_heads"] = query_heads
    op.attrs["_key_heads"] = key_heads
    op.attrs["_value_heads"] = value_heads
    op.attrs["_key_size"] = key_size
    op.attrs["_value_size"] = value_size
    op.attrs["_value_blocks"] = value_size // min(64, value_size & -value_size)
    op.attrs["_has_past"] = has_past
    op.attrs["_past"] = None if has_past else wp.zeros(state_shape, dtype=dtype, device=device)
    op.attrs["_decay"] = None if has_decay else wp.zeros((1, 1), dtype=dtype, device=device)
    op.attrs["_beta"] = None if has_beta else wp.zeros((1, 1), dtype=dtype, device=device)
    op.attrs["_needs_decay"] = needs_decay
    op.attrs["_decay_per_key"] = decay_per_key
    op.attrs["_needs_beta"] = needs_beta
    op.attrs["_beta_per_head"] = beta_per_head
    configured_scale = float(op.attrs.get("scale", 0.0))
    op.attrs["_scale"] = configured_scale if configured_scale != 0.0 else key_size**-0.5


def _shape_simplified_layer_normalization(op, shapes, dtypes, tensors, device, requires_grad=False):
    if requires_grad:
        raise NotImplementedError("OnnxRuntime SimplifiedLayerNormalization: gradients are not supported")
    shape = shapes[op.inputs[0]]
    if not shape or int(op.attrs.get("axis", -1)) != -1:
        raise NotImplementedError("OnnxRuntime SimplifiedLayerNormalization: only the last axis is supported")
    width = shape[-1]
    dtype = _require_matching_float_dtypes(op, dtypes, op.inputs)
    if dtype not in (wp.float16, wp.bfloat16) or shapes[op.inputs[1]] != (width,):
        raise ValueError("OnnxRuntime SimplifiedLayerNormalization: expected matching FP16/BF16 input and scale")
    rows = int(np.prod(shape[:-1]))
    tensors[op.outputs[0]] = wp.zeros(shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = shape
    dtypes[op.outputs[0]] = dtype
    op.attrs["_rows"] = rows
    op.attrs["_width"] = width
    op.attrs["_output_2d"] = tensors[op.outputs[0]].reshape((rows, width))
    op.attrs["_tile_width"], op.attrs["_rms_norm_kernels"] = _get_rms_norm_kernels(width, dtype)


def _shape_skip_simplified_layer_normalization(op, shapes, dtypes, tensors, device, requires_grad=False):
    if requires_grad:
        raise NotImplementedError("OnnxRuntime SkipSimplifiedLayerNormalization: gradients are not supported")
    shape = shapes[op.inputs[0]]
    if not shape or shapes[op.inputs[1]] != shape:
        raise NotImplementedError(
            "OnnxRuntime SkipSimplifiedLayerNormalization: matching non-scalar inputs are required"
        )
    width = shape[-1]
    dtype = _require_matching_float_dtypes(op, dtypes, op.inputs)
    if dtype not in (wp.float16, wp.bfloat16) or shapes[op.inputs[2]] != (width,):
        raise ValueError("OnnxRuntime SkipSimplifiedLayerNormalization: expected FP16/BF16 inputs and scale")
    if any(output for output in op.outputs[1:3]):
        raise NotImplementedError("OnnxRuntime SkipSimplifiedLayerNormalization: statistics outputs are not supported")

    rows = int(np.prod(shape[:-1]))
    tensors[op.outputs[0]] = wp.zeros(shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = shape
    dtypes[op.outputs[0]] = dtype
    if len(op.outputs) > 3 and op.outputs[3]:
        residual = wp.zeros(shape, dtype=dtype, device=device)
        tensors[op.outputs[3]] = residual
        shapes[op.outputs[3]] = shape
        dtypes[op.outputs[3]] = dtype
    else:
        residual = wp.zeros(shape, dtype=dtype, device=device)
    op.attrs["_rows"] = rows
    op.attrs["_width"] = width
    op.attrs["_output_2d"] = tensors[op.outputs[0]].reshape((rows, width))
    op.attrs["_residual_2d"] = residual.reshape((rows, width))
    op.attrs["_tile_width"], op.attrs["_rms_norm_kernels"] = _get_rms_norm_kernels(width, dtype)


def _shape_swiglu(op, shapes, dtypes, tensors, device, requires_grad=False):
    shape = shapes[op.inputs[0]]
    if not shape or shapes[op.inputs[1]] != shape:
        raise NotImplementedError("OnnxRuntime fused SwiGLU requires matching non-scalar inputs")
    dtype = _require_matching_float_dtypes(op, dtypes, op.inputs)
    if dtype not in (wp.float16, wp.bfloat16):
        raise TypeError("OnnxRuntime fused SwiGLU requires FP16/BF16 inputs")
    tensors[op.outputs[0]] = wp.zeros(shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = shape
    dtypes[op.outputs[0]] = dtype
    op.attrs["_shape_2d"] = (int(np.prod(shape[:-1])), shape[-1])
    op.attrs["_output_2d"] = tensors[op.outputs[0]].reshape(op.attrs["_shape_2d"])
    if dtype not in _swiglu_kernel_cache:
        _swiglu_kernel_cache[dtype] = _create_swiglu_kernel(dtype)
    op.attrs["_kernel"] = _swiglu_kernel_cache[dtype]


def _shape_group_query_attention(op, shapes, dtypes, tensors, device, requires_grad=False):
    if requires_grad:
        raise NotImplementedError("OnnxRuntime GroupQueryAttention: gradients are not supported")
    query_heads = int(op.attrs.get("num_heads", 0))
    kv_heads = int(op.attrs.get("kv_num_heads", 0))
    query_shape, key_shape, value_shape = (shapes[name] for name in op.inputs[:3])
    if len(query_shape) != 3 or len(key_shape) != 3 or value_shape != key_shape or query_shape[:2] != key_shape[:2]:
        raise ValueError("OnnxRuntime GroupQueryAttention: expected matching 3-D Q/K/V inputs")
    batch, sequence_length, hidden_size = query_shape
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads != 0 or hidden_size % query_heads != 0:
        raise ValueError("OnnxRuntime GroupQueryAttention: invalid query/KV head counts")
    head_size = hidden_size // query_heads
    if key_shape[2] != kv_heads * head_size or head_size % 2 != 0:
        raise ValueError("OnnxRuntime GroupQueryAttention: Q/K/V head dimensions do not match")
    past_shape = (batch, kv_heads, shapes[op.inputs[3]][2], head_size)
    if shapes[op.inputs[3]] != past_shape or shapes[op.inputs[4]] != past_shape:
        raise ValueError("OnnxRuntime GroupQueryAttention: invalid past KV-cache shape")
    if shapes[op.inputs[5]] != (batch,) or dtypes[op.inputs[5]] != wp.int32:
        raise ValueError("OnnxRuntime GroupQueryAttention: sequence lengths must be an INT32 batch vector")
    if shapes[op.inputs[6]] not in ((), (1,)) or dtypes[op.inputs[6]] != wp.int32:
        raise ValueError("OnnxRuntime GroupQueryAttention: total sequence length must be an INT32 scalar")
    do_rotary = bool(op.attrs.get("do_rotary", 0))
    if any(dtypes[name] != wp.float16 for name in op.inputs[:5]):
        raise TypeError("OnnxRuntime GroupQueryAttention: Q/K/V and caches must be FP16")
    if do_rotary:
        if (
            len(op.inputs) < 9
            or not op.inputs[7]
            or not op.inputs[8]
            or shapes[op.inputs[7]] != shapes[op.inputs[8]]
            or len(shapes[op.inputs[7]]) != 2
            or shapes[op.inputs[7]][1] != head_size // 2
        ):
            raise ValueError("OnnxRuntime GroupQueryAttention: invalid rotary cache shape")
        if dtypes[op.inputs[7]] != wp.float16 or dtypes[op.inputs[8]] != wp.float16:
            raise TypeError("OnnxRuntime GroupQueryAttention: rotary tables must be FP16")
        cos_cache, sin_cache = tensors[op.inputs[7]], tensors[op.inputs[8]]
    else:
        dummy_name = "__onnx_runtime_gqa_dummy_fp16"
        if dummy_name not in tensors:
            tensors[dummy_name] = wp.zeros((1, 1), dtype=wp.float16, device=device)
        cos_cache = sin_cache = tensors[dummy_name]
    if (
        len(op.inputs) < 7
        or any(op.inputs[9:])
        or int(op.attrs.get("rotary_interleaved", 0)) != 0
        or float(op.attrs.get("softcap", 0.0)) != 0.0
    ):
        raise NotImplementedError("OnnxRuntime GroupQueryAttention: unsupported optional inputs or attributes")

    past_length = past_shape[2]
    share_cache = bool(op.attrs.get("_share_cache", False))
    total_length = past_length if share_cache else past_length + sequence_length
    output_shape = query_shape
    present_shape = (batch, kv_heads, total_length, head_size)
    tensors[op.outputs[0]] = wp.zeros(output_shape, dtype=wp.float16, device=device)
    if not share_cache:
        tensors[op.outputs[1]] = wp.zeros(present_shape, dtype=wp.float16, device=device)
        tensors[op.outputs[2]] = wp.zeros(present_shape, dtype=wp.float16, device=device)
    shapes[op.outputs[0]], shapes[op.outputs[1]], shapes[op.outputs[2]] = output_shape, present_shape, present_shape
    dtypes[op.outputs[0]] = dtypes[op.outputs[1]] = dtypes[op.outputs[2]] = wp.float16
    rotated_shape = (batch, query_heads, sequence_length, head_size)
    rotated_name = f"__onnx_runtime_gqa_query_{rotated_shape}"
    if rotated_name not in tensors:
        tensors[rotated_name] = wp.zeros(rotated_shape, dtype=wp.float16, device=device)
    attention_block_dim, attention_kernel = _get_gqa_attention_kernel(head_size)
    op.attrs.update(
        {
            "_batch": batch,
            "_sequence_length": sequence_length,
            "_past_length": past_length,
            "_total_length": total_length,
            "_head_size": head_size,
            "_cos_cache": cos_cache,
            "_sin_cache": sin_cache,
            "_do_rotary": do_rotary,
            "_rotated_query": tensors[rotated_name],
            "_attention_block_dim": attention_block_dim,
            "_attention_kernel": attention_kernel,
        }
    )


def _shape_squeeze(op, shapes, dtypes, tensors, device, requires_grad=False):
    in_shape = shapes[op.inputs[0]]
    axes = None
    if len(op.inputs) > 1 and op.inputs[1] in tensors:
        axes_tensor = tensors[op.inputs[1]]
        if hasattr(axes_tensor, "numpy"):
            axes = [int(v) for v in axes_tensor.numpy().tolist()]
    if axes is None:
        out_shape = tuple(d for d in in_shape if d != 1)
    else:
        rank = len(in_shape)
        axes_norm = {a if a >= 0 else a + rank for a in axes}
        out_shape = tuple(d for i, d in enumerate(in_shape) if i not in axes_norm)
    if axes is not None and any(in_shape[axis] != 1 for axis in axes_norm):
        raise ValueError("OnnxRuntime Squeeze: selected axes must have size one")
    shapes[op.outputs[0]] = out_shape
    dtypes[op.outputs[0]] = dtypes[op.inputs[0]]
    op.attrs["_out_shape"] = out_shape


def _shape_unsqueeze(op, shapes, dtypes, tensors, device, requires_grad=False):
    if len(op.inputs) != 2 or op.inputs[1] not in tensors:
        raise NotImplementedError("OnnxRuntime Unsqueeze: the axes input must be constant")
    in_shape = shapes[op.inputs[0]]
    axes = [int(value) for value in tensors[op.inputs[1]].numpy().reshape(-1)]
    out_rank = len(in_shape) + len(axes)
    axes_norm = {axis if axis >= 0 else axis + out_rank for axis in axes}
    if len(axes_norm) != len(axes) or any(axis < 0 or axis >= out_rank for axis in axes_norm):
        raise ValueError("OnnxRuntime Unsqueeze: invalid or duplicate axes")
    source = iter(in_shape)
    out_shape = tuple(1 if axis in axes_norm else next(source) for axis in range(out_rank))
    shapes[op.outputs[0]] = out_shape
    dtypes[op.outputs[0]] = dtypes[op.inputs[0]]
    op.attrs["_out_shape"] = out_shape


def _shape_transpose(op, shapes, dtypes, tensors, device, requires_grad=False):
    in_shape = shapes[op.inputs[0]]
    perm = tuple(int(axis) for axis in op.attrs.get("perm", reversed(range(len(in_shape)))))
    if sorted(perm) != list(range(len(in_shape))):
        raise ValueError("OnnxRuntime Transpose: invalid permutation")
    if perm == (0, 2, 1):
        kernel = _transpose_021_kernel
    elif perm == (0, 2, 1, 3):
        kernel = _transpose_0213_kernel
    else:
        raise NotImplementedError(f"OnnxRuntime Transpose: permutation {perm} is not supported")
    out_shape = tuple(in_shape[axis] for axis in perm)
    dtype = dtypes[op.inputs[0]]
    tensors[op.outputs[0]] = wp.zeros(out_shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = out_shape
    dtypes[op.outputs[0]] = dtype
    op.attrs["_kernel"] = _kernel_for_dtype(kernel, dtype, (len(in_shape),), (len(in_shape),))


def _shape_split(op, shapes, dtypes, tensors, device, requires_grad=False):
    in_shape = shapes[op.inputs[0]]
    axis = int(op.attrs.get("axis", 0))
    if axis < 0:
        axis += len(in_shape)
    if axis != len(in_shape) - 1:
        raise NotImplementedError("OnnxRuntime Split: only the last axis is supported")
    if len(op.inputs) > 1 and op.inputs[1]:
        if op.inputs[1] not in tensors:
            raise NotImplementedError("OnnxRuntime Split: split sizes must be constant")
        split_sizes = [int(value) for value in tensors[op.inputs[1]].numpy().reshape(-1)]
    elif "split" in op.attrs:
        split_sizes = [int(value) for value in op.attrs["split"]]
    else:
        if in_shape[-1] % len(op.outputs):
            raise ValueError("OnnxRuntime Split: axis size is not evenly divisible")
        split_sizes = [in_shape[-1] // len(op.outputs)] * len(op.outputs)
    if len(split_sizes) != len(op.outputs) or sum(split_sizes) != in_shape[-1]:
        raise ValueError("OnnxRuntime Split: invalid split sizes")

    dtype = dtypes[op.inputs[0]]
    rows = int(np.prod(in_shape[:-1]))
    for name, width in zip(op.outputs, split_sizes):
        out_shape = (*in_shape[:-1], width)
        tensors[name] = wp.zeros(out_shape, dtype=dtype, device=device)
        shapes[name] = out_shape
        dtypes[name] = dtype
    op.attrs["_rows"] = rows
    op.attrs["_split_sizes"] = split_sizes
    op.attrs["_kernel"] = _kernel_for_dtype(_split_last_axis_kernel, dtype, (2,), (2,), int)


def _shape_tile(op, shapes, dtypes, tensors, device, requires_grad=False):
    if len(op.inputs) != 2 or op.inputs[1] not in tensors or len(shapes[op.inputs[0]]) != 3:
        raise NotImplementedError("OnnxRuntime Tile: only rank-3 tensors with constant repeats are supported")
    repeats = tuple(int(value) for value in tensors[op.inputs[1]].numpy().reshape(-1))
    in_shape = shapes[op.inputs[0]]
    if len(repeats) != 3 or any(repeat < 0 for repeat in repeats):
        raise ValueError("OnnxRuntime Tile: invalid repeats")
    out_shape = tuple(size * repeat for size, repeat in zip(in_shape, repeats))
    dtype = dtypes[op.inputs[0]]
    tensors[op.outputs[0]] = wp.zeros(out_shape, dtype=dtype, device=device)
    shapes[op.outputs[0]] = out_shape
    dtypes[op.outputs[0]] = dtype
    op.attrs["_kernel"] = _kernel_for_dtype(_tile_3d_kernel, dtype, (3,), (3,))


def _shape_lstm(op, shapes, dtypes, tensors, device, requires_grad=False):
    for unsupported in ("activations", "activation_alpha", "activation_beta"):
        if unsupported in op.attr_names:
            raise NotImplementedError(
                f"OnnxRuntime LSTM: attribute '{unsupported}' is not supported "
                f"(only default sigmoid/tanh/tanh activations)"
            )
    if op.attrs.get("clip", 0.0):
        raise NotImplementedError(
            f"OnnxRuntime LSTM: non-default 'clip' attribute is not supported (got {op.attrs['clip']})"
        )
    if op.attrs.get("input_forget", 0):
        raise NotImplementedError(
            f"OnnxRuntime LSTM: non-default 'input_forget' attribute is not supported (got {op.attrs['input_forget']})"
        )

    if len(op.inputs) > 4 and op.inputs[4]:
        raise NotImplementedError("OnnxRuntime LSTM: 'sequence_lens' input is not supported")
    if len(op.inputs) > 7 and op.inputs[7]:
        raise NotImplementedError("OnnxRuntime LSTM: peephole input 'P' is not supported")

    direction = op.attrs.get("direction", "forward")
    if direction not in ("forward", b"forward"):
        raise NotImplementedError("OnnxRuntime LSTM: only forward direction is supported")

    layout = int(op.attrs.get("layout", 0))
    if layout != 0:
        raise NotImplementedError("OnnxRuntime LSTM: layout must be 0 (layout=1 not supported)")

    X_shape = shapes[op.inputs[0]]
    if len(X_shape) != 3:
        raise NotImplementedError("OnnxRuntime LSTM: input X must be 3-D")
    if layout == 0:
        seq_len, batch, input_size = X_shape
    else:
        batch, seq_len, input_size = X_shape
    if seq_len != 1:
        raise NotImplementedError("OnnxRuntime LSTM: only seq_length=1 is supported (single-step inference)")

    W_shape = shapes[op.inputs[1]]
    if len(W_shape) != 3 or W_shape[0] != 1:
        raise NotImplementedError("OnnxRuntime LSTM: only num_directions=1 is supported")
    hidden_size = int(op.attrs.get("hidden_size", W_shape[1] // 4))

    if W_shape != (1, 4 * hidden_size, input_size):
        raise ValueError(f"OnnxRuntime LSTM: W has shape {W_shape}, expected {(1, 4 * hidden_size, input_size)}")

    R_shape = shapes[op.inputs[2]]
    if R_shape != (1, 4 * hidden_size, hidden_size):
        raise ValueError(f"OnnxRuntime LSTM: R has shape {R_shape}, expected {(1, 4 * hidden_size, hidden_size)}")

    dtype_inputs = [name for name in (op.inputs[0], op.inputs[1], op.inputs[2]) if name]
    for index in (3, 5, 6):
        if len(op.inputs) > index and op.inputs[index]:
            dtype_inputs.append(op.inputs[index])
    dtype = _require_matching_float_dtypes(op, dtypes, dtype_inputs)

    W_full = tensors[op.inputs[1]]
    R_full = tensors[op.inputs[2]]
    cache: dict[str, wp.array] = {}
    cache["W"] = W_full.reshape((4 * hidden_size, input_size))
    cache["R"] = R_full.reshape((4 * hidden_size, hidden_size))

    if len(op.inputs) > 3 and op.inputs[3] and op.inputs[3] in tensors:
        B_full = tensors[op.inputs[3]]
        B_shape_in = shapes[op.inputs[3]]
        if B_shape_in != (1, 8 * hidden_size):
            raise ValueError(f"OnnxRuntime LSTM: B has shape {B_shape_in}, expected {(1, 8 * hidden_size)}")
        B_2d = B_full.reshape((8 * hidden_size,))
        cache["Bx"] = B_2d[: 4 * hidden_size]
        cache["Bh"] = B_2d[4 * hidden_size :]
    else:
        cache["Bx"] = wp.zeros(
            4 * hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        cache["Bh"] = wp.zeros(
            4 * hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    cache["gates"] = wp.zeros(
        (batch, 4 * hidden_size),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    cache["input_size"] = input_size
    cache["hidden_size"] = hidden_size
    cache["batch"] = batch
    cache["layout"] = layout
    cache["dtype"] = dtype
    op.attrs["_cache"] = cache

    h_buf = wp.zeros(
        (batch, hidden_size),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    c_buf = wp.zeros(
        (batch, hidden_size),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    cache["h_out"] = h_buf
    cache["c_out"] = c_buf

    if layout == 0:
        Y_shape = (1, 1, batch, hidden_size)
    else:
        Y_shape = (batch, 1, 1, hidden_size)
    Yh_shape = (1, batch, hidden_size)

    if len(op.outputs) > 0 and op.outputs[0]:
        tensors[op.outputs[0]] = h_buf.reshape(Y_shape)
        shapes[op.outputs[0]] = Y_shape
        dtypes[op.outputs[0]] = dtype
    if len(op.outputs) > 1 and op.outputs[1]:
        tensors[op.outputs[1]] = h_buf.reshape(Yh_shape)
        shapes[op.outputs[1]] = Yh_shape
        dtypes[op.outputs[1]] = dtype
    if len(op.outputs) > 2 and op.outputs[2]:
        tensors[op.outputs[2]] = c_buf.reshape(Yh_shape)
        shapes[op.outputs[2]] = Yh_shape
        dtypes[op.outputs[2]] = dtype


def _exec_gemm(op, tensors, shapes, device):
    A = tensors[op.inputs[0]]
    B = tensors[op.inputs[1]]
    bias = tensors[op.inputs[2]]
    out = tensors[op.outputs[0]]
    alpha = float(op.attrs.get("alpha", 1.0))
    beta = float(op.attrs.get("beta", 1.0))
    M = shapes[op.inputs[0]][0]
    N, K = shapes[op.inputs[1]]

    if op.attrs["_requires_grad"] or A.dtype != wp.float32:
        wp.launch(
            _kernel_for_dtype(_gemm_transb_kernel, A.dtype, (2,), (2,), (1,), (2,), int, float, float),
            dim=(M, N),
            inputs=[A, B, bias, out, K, alpha, beta],
            device=device,
        )
    else:
        wp.launch_tiled(
            _GEMM_TRANSB_TILED_KERNEL,
            dim=resolve_dim(config=_GEMM_CONFIG, shape=(M, N), tiled=True),
            inputs=[A, B, op.attrs["_bias_2d"], alpha, beta],
            outputs=[out],
            device=device,
            block_dim=_GEMM_CONFIG.block_dim,
        )


def _exec_elu(op, tensors, shapes, device):
    x = tensors[op.inputs[0]]
    alpha = float(op.attrs.get("alpha", 1.0))
    out = tensors[op.outputs[0]]
    shape = op.attrs["_shape_2d"]
    kernel = _kernel_for_dtype(_elu_kernel, x.dtype, (2,), (2,), float)
    wp.launch(kernel, dim=shape, inputs=[x.reshape(shape), out.reshape(shape), alpha], device=device)


def _exec_unary(op, tensors, shapes, device):
    operation = {"Relu": 0, "Tanh": 1, "Sqrt": 2, "Sigmoid": 3, "Softplus": 4}[op.op_type]
    shape_2d = op.attrs["_shape_2d"]
    wp.launch(
        _kernel_for_dtype(_unary_kernel, tensors[op.inputs[0]].dtype, (2,), int, (2,)),
        dim=shape_2d,
        inputs=[tensors[op.inputs[0]].reshape(shape_2d), operation],
        outputs=[tensors[op.outputs[0]].reshape(shape_2d)],
        device=device,
    )


def _exec_binary(op, tensors, shapes, device):
    lhs = tensors[op.inputs[0]].reshape(op.attrs["_lhs_shape_2d"])
    rhs = tensors[op.inputs[1]].reshape(op.attrs["_rhs_shape_2d"])
    operation = {"Add": 0, "Sub": 1, "Mul": 2, "Div": 3}[op.op_type]
    wp.launch(
        _kernel_for_dtype(_binary_broadcast_kernel, lhs.dtype, (2,), (2,), int, (2,)),
        dim=op.attrs["_out_shape_2d"],
        inputs=[lhs, rhs, operation],
        outputs=[tensors[op.outputs[0]].reshape(op.attrs["_out_shape_2d"])],
        device=device,
    )


def _exec_reduce_mean(op, tensors, shapes, device):
    wp.launch(
        _kernel_for_dtype(_reduce_mean_rows_kernel, tensors[op.inputs[0]].dtype, (2,), (2,)),
        dim=shapes[op.inputs[0]][0],
        inputs=[tensors[op.inputs[0]]],
        outputs=[tensors[op.outputs[0]]],
        device=device,
    )


def _exec_reduce_sum(op, tensors, shapes, device):
    wp.launch(
        _kernel_for_dtype(_reduce_sum_rows_kernel, tensors[op.inputs[0]].dtype, (2,), (1,)),
        dim=shapes[op.inputs[0]][0],
        inputs=[tensors[op.inputs[0]], tensors[op.outputs[0]]],
        device=device,
    )


def _exec_static(op, tensors, shapes, device):
    pass


def _exec_gather(op, tensors, shapes, device):
    if not op.attrs.get("_dynamic"):
        return
    data = tensors[op.inputs[0]]
    wp.launch(
        _kernel_for_dtype(
            _gather_rows_kernel,
            data.dtype,
            (2,),
            _array_type(wp.int64, 2),
            (3,),
        ),
        dim=shapes[op.outputs[0]],
        inputs=[data, tensors[op.inputs[1]], tensors[op.outputs[0]]],
        device=device,
    )


def _exec_cast(op, tensors, shapes, device):
    size = int(np.prod(shapes[op.inputs[0]]))
    wp.launch(
        op.attrs["_kernel"],
        dim=size,
        inputs=[tensors[op.inputs[0]].reshape((size,)), tensors[op.outputs[0]].reshape((size,))],
        device=device,
    )


def _exec_lp_normalization(op, tensors, shapes, device):
    rows = op.attrs["_rows"]
    width = op.attrs["_width"]
    wp.launch_tiled(
        op.attrs["_kernel"],
        dim=rows,
        inputs=[
            tensors[op.inputs[0]].reshape((rows, width)),
            tensors[op.outputs[0]].reshape((rows, width)),
        ],
        block_dim=op.attrs["_tile_width"],
        device=device,
    )


def _exec_reduce_max(op, tensors, shapes, device):
    wp.launch(
        op.attrs["_kernel"],
        dim=1,
        inputs=[tensors[op.inputs[0]], tensors[op.outputs[0]]],
        device=device,
    )


def _exec_batch_normalization(op, tensors, shapes, device):
    wp.launch(
        _kernel_for_dtype(
            _batch_normalization_kernel,
            tensors[op.inputs[0]].dtype,
            (2,),
            (1,),
            (1,),
            (1,),
            (1,),
            float,
            bool,
            (2,),
        ),
        dim=shapes[op.inputs[0]],
        inputs=[
            tensors[op.inputs[0]],
            tensors[op.inputs[1]],
            tensors[op.inputs[2]],
            tensors[op.inputs[3]],
            tensors[op.inputs[4]],
            float(op.attrs.get("epsilon", 1.0e-5)),
            op.op_type == "_BatchNormalizationRelu",
        ],
        outputs=[tensors[op.outputs[0]]],
        device=device,
    )


def _exec_rms_normalization(op, tensors, shapes, device):
    wp.launch_tiled(
        op.attrs["_kernel"],
        dim=shapes[op.inputs[0]][0],
        inputs=[tensors[op.inputs[0]], tensors[op.inputs[1]], op.attrs["_scale"]],
        outputs=[tensors[op.outputs[0]]],
        device=device,
        block_dim=_GEMM_CONFIG.block_dim,
    )


def _exec_constant(op, tensors, shapes, device):
    pass


def _exec_reshape(op, tensors, shapes, device):
    tensors[op.outputs[0]] = tensors[op.inputs[0]].reshape(op.attrs["_out_shape"])


def _exec_transpose(op, tensors, shapes, device):
    wp.launch(
        op.attrs["_kernel"],
        dim=shapes[op.outputs[0]],
        inputs=[tensors[op.inputs[0]], tensors[op.outputs[0]]],
        device=device,
    )


def _exec_split(op, tensors, shapes, device):
    rows = op.attrs["_rows"]
    source = tensors[op.inputs[0]].reshape((rows, shapes[op.inputs[0]][-1]))
    offset = 0
    for name, width in zip(op.outputs, op.attrs["_split_sizes"]):
        wp.launch(
            op.attrs["_kernel"],
            dim=(rows, width),
            inputs=[source, tensors[name].reshape((rows, width)), offset],
            device=device,
        )
        offset += width


def _exec_tile(op, tensors, shapes, device):
    wp.launch(
        op.attrs["_kernel"],
        dim=shapes[op.outputs[0]],
        inputs=[tensors[op.inputs[0]], tensors[op.outputs[0]]],
        device=device,
    )


def _exec_where(op, tensors, shapes, device):
    shape_2d = op.attrs["_shape_2d"]
    wp.launch(
        op.attrs["_kernel"],
        dim=shape_2d,
        inputs=[
            tensors[op.inputs[0]].reshape(op.attrs["_condition_shape_2d"]),
            tensors[op.inputs[1]].reshape(shape_2d),
            tensors[op.inputs[2]].reshape(shape_2d),
            tensors[op.outputs[0]].reshape(shape_2d),
        ],
        device=device,
    )


def _exec_rotary_embedding(op, tensors, shapes, device):
    wp.launch(
        op.attrs["_kernel"],
        dim=shapes[op.inputs[0]],
        inputs=[
            tensors[op.inputs[0]],
            tensors[op.inputs[1]].reshape(op.attrs["_position_shape_2d"]),
            tensors[op.inputs[2]],
            tensors[op.inputs[3]],
            tensors[op.outputs[0]],
            int(op.attrs.get("rotary_embedding_dim", shapes[op.inputs[0]][-1])),
            bool(op.attrs.get("interleaved", 0)),
            op.attrs["_position_offset"],
        ],
        device=device,
    )


def _exec_gather_block_quantized(op, tensors, shapes, device):
    wp.launch(
        _gather_block_quantized_int8_kernel,
        dim=shapes[op.outputs[0]],
        inputs=[
            tensors[op.inputs[0]],
            tensors[op.inputs[1]],
            tensors[op.inputs[2]],
            tensors[op.inputs[3]],
            tensors[op.outputs[0]],
            128,
        ],
        device=device,
    )


def _exec_matmul_nbits(op, tensors, shapes, device):
    K = int(op.attrs["K"])
    N = int(op.attrs["N"])
    bits = int(op.attrs["bits"])
    block_size = op.attrs["_block_size"]
    dtype = op.attrs["_dtype"]
    zero_points = op.attrs["_zero_points"]
    has_zero_points = op.attrs["_has_zero_points"]
    if "_cublas" in op.attrs:
        weights = tensors[op.inputs[1]]
        dequantized = op.attrs["_dequantized_weights"]
        wp.launch(
            _get_dequantize_nbits_kernel(bits, block_size, dtype),
            dim=(N, weights.shape[1] * weights.shape[2]),
            inputs=[weights, tensors[op.inputs[2]], zero_points, dequantized, has_zero_points],
            device=device,
        )
        op.attrs["_cublas"].gemm(
            tensors[op.inputs[0]].ptr,
            dequantized.ptr,
            op.attrs["_output_2d"].ptr,
            op.attrs["_rows"],
            N,
            K,
            wp.get_stream(device).cuda_stream,
            2 if dtype == wp.float16 else 14,
        )
        return
    if device.is_cuda:
        wp.launch(
            _get_matmul_nbits_kernel(bits, block_size, dtype, True),
            dim=op.attrs["_rows"] * N * 32,
            inputs=[
                tensors[op.inputs[0]].reshape((op.attrs["_rows"], K)),
                tensors[op.inputs[1]],
                tensors[op.inputs[2]],
                zero_points,
                op.attrs["_output_2d"],
                has_zero_points,
            ],
            block_dim=128,
            device=device,
        )
        return
    wp.launch(
        _get_matmul_nbits_kernel(bits, block_size, dtype, False),
        dim=(op.attrs["_rows"], N),
        inputs=[
            tensors[op.inputs[0]].reshape((op.attrs["_rows"], K)),
            tensors[op.inputs[1]],
            tensors[op.inputs[2]],
            zero_points,
            op.attrs["_output_2d"],
            has_zero_points,
        ],
        device=device,
    )


def _exec_causal_conv_with_state(op, tensors, shapes, device):
    x = tensors[op.inputs[0]]
    past = tensors[op.inputs[3]] if op.attrs["_has_past"] else op.attrs["_past"]
    wp.launch(
        op.attrs["_kernel"],
        dim=x.shape,
        inputs=[
            x,
            tensors[op.inputs[1]],
            op.attrs["_bias"],
            past,
            tensors[op.outputs[0]],
            op.attrs["_kernel_size"],
            op.attrs["_has_bias"],
            op.attrs["_silu"],
        ],
        device=device,
    )
    if op.attrs["_kernel_size"] > 1:
        wp.launch(
            op.attrs["_state_kernel"],
            dim=tensors[op.outputs[1]].shape,
            inputs=[x, past, tensors[op.outputs[1]]],
            device=device,
        )


def _exec_linear_attention(op, tensors, shapes, device):
    batch = op.attrs["_batch"]
    sequence_length = op.attrs["_sequence_length"]
    query_heads = op.attrs["_query_heads"]
    key_heads = op.attrs["_key_heads"]
    value_heads = op.attrs["_value_heads"]
    key_size = op.attrs["_key_size"]
    value_size = op.attrs["_value_size"]
    past = tensors[op.inputs[3]] if op.attrs["_has_past"] else op.attrs["_past"]
    decay = tensors[op.inputs[4]] if op.attrs["_needs_decay"] else op.attrs["_decay"]
    beta = tensors[op.inputs[5]] if op.attrs["_needs_beta"] else op.attrs["_beta"]
    wp.launch_tiled(
        op.attrs["_kernel"],
        dim=batch * value_heads * op.attrs["_value_blocks"],
        inputs=[
            tensors[op.inputs[0]].reshape((batch * sequence_length, query_heads * key_size)),
            tensors[op.inputs[1]].reshape((batch * sequence_length, key_heads * key_size)),
            tensors[op.inputs[2]].reshape((batch * sequence_length, value_heads * value_size)),
            past.reshape((batch * value_heads * key_size, value_size)),
            decay.reshape((int(np.prod(decay.shape[:-1])), decay.shape[-1])),
            beta.reshape((int(np.prod(beta.shape[:-1])), beta.shape[-1])),
            tensors[op.outputs[0]].reshape((batch * sequence_length, max(query_heads, value_heads) * value_size)),
            tensors[op.outputs[1]].reshape((batch * value_heads * key_size, value_size)),
            sequence_length,
            query_heads,
            key_heads,
            value_heads,
            op.attrs["_needs_decay"],
            op.attrs["_decay_per_key"],
            op.attrs["_needs_beta"],
            op.attrs["_beta_per_head"],
            op.attrs["_scale"],
        ],
        block_dim=op.attrs["_block_dim"],
        device=device,
    )


def _exec_simplified_layer_normalization(op, tensors, shapes, device):
    wp.launch_tiled(
        op.attrs["_rms_norm_kernels"][0],
        dim=op.attrs["_rows"],
        inputs=[
            tensors[op.inputs[0]].reshape((op.attrs["_rows"], op.attrs["_width"])),
            tensors[op.inputs[1]],
            op.attrs["_output_2d"],
            float(op.attrs.get("epsilon", 1.0e-5)),
        ],
        block_dim=op.attrs["_tile_width"],
        device=device,
    )


def _exec_skip_simplified_layer_normalization(op, tensors, shapes, device):
    shape_2d = (op.attrs["_rows"], op.attrs["_width"])
    wp.launch_tiled(
        op.attrs["_rms_norm_kernels"][1],
        dim=op.attrs["_rows"],
        inputs=[
            tensors[op.inputs[0]].reshape(shape_2d),
            tensors[op.inputs[1]].reshape(shape_2d),
            tensors[op.inputs[2]],
            op.attrs["_output_2d"],
            op.attrs["_residual_2d"],
            float(op.attrs.get("epsilon", 1.0e-5)),
        ],
        block_dim=op.attrs["_tile_width"],
        device=device,
    )


def _exec_swiglu(op, tensors, shapes, device):
    wp.launch(
        op.attrs["_kernel"],
        dim=op.attrs["_shape_2d"],
        inputs=[
            tensors[op.inputs[0]].reshape(op.attrs["_shape_2d"]),
            tensors[op.inputs[1]].reshape(op.attrs["_shape_2d"]),
            op.attrs["_output_2d"],
        ],
        device=device,
    )


def _exec_group_query_attention(op, tensors, shapes, device):
    batch = op.attrs["_batch"]
    sequence_length = op.attrs["_sequence_length"]
    past_length = op.attrs["_past_length"]
    total_length = op.attrs["_total_length"]
    head_size = op.attrs["_head_size"]
    query_heads = int(op.attrs["num_heads"])
    kv_heads = int(op.attrs["kv_num_heads"])
    share_cache = bool(op.attrs.get("_share_cache", False))
    if share_cache:
        present_key = tensors[op.inputs[3]]
        present_value = tensors[op.inputs[4]]
        tensors[op.outputs[1]] = present_key
        tensors[op.outputs[2]] = present_value
    else:
        present_key = tensors[op.outputs[1]]
        present_value = tensors[op.outputs[2]]
    if past_length and not share_cache:
        wp.launch(
            _gqa_copy_past_fp16_kernel,
            dim=(batch, kv_heads, past_length, head_size),
            inputs=[tensors[op.inputs[3]], tensors[op.inputs[4]], present_key, present_value],
            device=device,
        )
    wp.launch(
        _gqa_prepare_fp16_kernel,
        dim=(batch, query_heads, sequence_length, head_size),
        inputs=[
            tensors[op.inputs[0]],
            tensors[op.inputs[1]],
            tensors[op.inputs[2]],
            tensors[op.inputs[5]],
            op.attrs["_cos_cache"],
            op.attrs["_sin_cache"],
            op.attrs["_rotated_query"],
            present_key,
            present_value,
            query_heads,
            kv_heads,
            sequence_length,
            past_length,
            head_size,
            share_cache,
            op.attrs["_do_rotary"],
        ],
        device=device,
    )
    wp.launch_tiled(
        op.attrs["_attention_kernel"],
        dim=batch * query_heads * sequence_length,
        inputs=[
            op.attrs["_rotated_query"].reshape((batch * query_heads * sequence_length, head_size)),
            present_key.reshape((batch * kv_heads * total_length, head_size)),
            present_value.reshape((batch * kv_heads * total_length, head_size)),
            tensors[op.inputs[5]],
            tensors[op.outputs[0]].reshape((batch * sequence_length, query_heads * head_size)),
            query_heads,
            kv_heads,
            sequence_length,
            total_length,
            float(op.attrs.get("scale", head_size**-0.5)),
        ],
        block_dim=op.attrs["_attention_block_dim"],
        device=device,
    )


def _exec_squeeze(op, tensors, shapes, device):
    src = tensors[op.inputs[0]]
    out_shape = op.attrs["_out_shape"]
    tensors[op.outputs[0]] = src.reshape(out_shape)
    shapes[op.outputs[0]] = out_shape


def _exec_lstm(op, tensors, shapes, device):
    cache = op.attrs["_cache"]
    input_size: int = cache["input_size"]
    hidden_size: int = cache["hidden_size"]
    batch: int = cache["batch"]
    layout: int = cache["layout"]

    X = tensors[op.inputs[0]]
    if layout == 0:
        x_t = X.reshape((batch, input_size))
    else:
        x_t = X.reshape((batch, input_size))

    if len(op.inputs) > 5 and op.inputs[5] and op.inputs[5] in tensors:
        h_prev = tensors[op.inputs[5]].reshape((batch, hidden_size))
    else:
        if "h_prev_zero" not in cache:
            cache["h_prev_zero"] = wp.zeros((batch, hidden_size), dtype=cache["dtype"], device=device)
        h_prev = cache["h_prev_zero"]
    if len(op.inputs) > 6 and op.inputs[6] and op.inputs[6] in tensors:
        c_prev = tensors[op.inputs[6]].reshape((batch, hidden_size))
    else:
        if "c_prev_zero" not in cache:
            cache["c_prev_zero"] = wp.zeros((batch, hidden_size), dtype=cache["dtype"], device=device)
        c_prev = cache["c_prev_zero"]

    gates = cache["gates"]
    h_out = cache["h_out"]
    c_out = cache["c_out"]

    wp.launch(
        _kernel_for_dtype(_lstm_gates_kernel, cache["dtype"], (2,), (2,), (2,), (2,), (2,), int, int),
        dim=(batch, 4 * hidden_size),
        inputs=[x_t, h_prev, cache["W"], cache["R"], gates, input_size, hidden_size],
        device=device,
    )
    wp.launch(
        _kernel_for_dtype(_lstm_cell_update_kernel, cache["dtype"], (2,), (2,), (1,), (1,), (2,), (2,), int),
        dim=(batch, hidden_size),
        inputs=[gates, c_prev, cache["Bx"], cache["Bh"], h_out, c_out, hidden_size],
        device=device,
    )


_OP_DISPATCH: dict[str, Any] = {
    "_BatchNormalizationRelu": _exec_batch_normalization,
    "_RmsNormalization": _exec_rms_normalization,
    "_SwiGLU": _exec_swiglu,
    "Add": _exec_binary,
    "BatchNormalization": _exec_batch_normalization,
    "Cast": _exec_cast,
    "CausalConvWithState": _exec_causal_conv_with_state,
    "Constant": _exec_constant,
    "Div": _exec_binary,
    "Elu": _exec_elu,
    "Gemm": _exec_gemm,
    "Gather": _exec_gather,
    "GatherBlockQuantized": _exec_gather_block_quantized,
    "GroupQueryAttention": _exec_group_query_attention,
    "LSTM": _exec_lstm,
    "LinearAttention": _exec_linear_attention,
    "LpNormalization": _exec_lp_normalization,
    "MatMulNBits": _exec_matmul_nbits,
    "Mul": _exec_binary,
    "ReduceMean": _exec_reduce_mean,
    "ReduceMax": _exec_reduce_max,
    "ReduceSum": _exec_reduce_sum,
    "Range": _exec_static,
    "Relu": _exec_unary,
    "Reshape": _exec_reshape,
    "RotaryEmbedding": _exec_rotary_embedding,
    "Shape": _exec_static,
    "Sigmoid": _exec_unary,
    "Sqrt": _exec_unary,
    "Softplus": _exec_unary,
    "SimplifiedLayerNormalization": _exec_simplified_layer_normalization,
    "Squeeze": _exec_squeeze,
    "Sub": _exec_binary,
    "SkipSimplifiedLayerNormalization": _exec_skip_simplified_layer_normalization,
    "Slice": _exec_static,
    "Split": _exec_split,
    "Tanh": _exec_unary,
    "Tile": _exec_tile,
    "Transpose": _exec_transpose,
    "Unsqueeze": _exec_squeeze,
    "Where": _exec_where,
}

_SHAPE_DISPATCH: dict[str, Any] = {
    "_BatchNormalizationRelu": _shape_batch_normalization,
    "_RmsNormalization": _shape_rms_normalization,
    "_SwiGLU": _shape_swiglu,
    "Add": _shape_elementwise_binary,
    "BatchNormalization": _shape_batch_normalization,
    "Cast": _shape_cast,
    "CausalConvWithState": _shape_causal_conv_with_state,
    "Constant": _shape_constant,
    "Div": _shape_elementwise_binary,
    "Elu": _shape_elementwise_unary,
    "Gemm": _shape_gemm,
    "Gather": _shape_gather,
    "GatherBlockQuantized": _shape_gather_block_quantized,
    "GroupQueryAttention": _shape_group_query_attention,
    "LSTM": _shape_lstm,
    "LinearAttention": _shape_linear_attention,
    "LpNormalization": _shape_lp_normalization,
    "MatMulNBits": _shape_matmul_nbits,
    "Mul": _shape_elementwise_binary,
    "ReduceMean": _shape_reduce_mean,
    "ReduceMax": _shape_reduce_max,
    "ReduceSum": _shape_reduce_sum,
    "Range": _shape_range,
    "Relu": _shape_elementwise_unary,
    "Reshape": _shape_reshape,
    "RotaryEmbedding": _shape_rotary_embedding,
    "Shape": _shape_shape,
    "Sigmoid": _shape_elementwise_unary,
    "Sqrt": _shape_elementwise_unary,
    "Softplus": _shape_elementwise_unary,
    "SimplifiedLayerNormalization": _shape_simplified_layer_normalization,
    "Squeeze": _shape_squeeze,
    "Sub": _shape_elementwise_binary,
    "SkipSimplifiedLayerNormalization": _shape_skip_simplified_layer_normalization,
    "Slice": _shape_slice,
    "Split": _shape_split,
    "Tanh": _shape_elementwise_unary,
    "Tile": _shape_tile,
    "Transpose": _shape_transpose,
    "Unsqueeze": _shape_unsqueeze,
    "Where": _shape_where,
}
