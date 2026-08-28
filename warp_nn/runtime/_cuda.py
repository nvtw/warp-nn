# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small CUDA intrinsic wrappers used by Warp runtime kernels."""

from functools import lru_cache

import warp as wp


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    for (int offset = width / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(__activemask(), value, offset, width);
#endif
    return value;
    """
)
def subgroup_sum(value: float, width: int) -> float: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    for (int offset = 16; offset > 0; offset >>= 1)
        value = max(value, __shfl_down_sync(__activemask(), value, offset));
    value = __shfl_sync(__activemask(), value, 0);
#endif
    return value;
    """
)
def warp_max_broadcast(value: float) -> float: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __dp4a(a, b, total);
#else
    for (int shift = 0; shift < 32; shift += 8)
        total += int8_t(a >> shift) * int8_t(b >> shift);
    return total;
#endif
    """
)
def dp4a(a: int, b: int, total: int) -> int: ...


@wp.func_native(
    """
    unsigned packed = (unsigned)value;
#if defined(__CUDA_ARCH__)
    unsigned duplicated = __byte_perm(packed, 0, 0x1100);
#else
    unsigned first = packed & 0xff;
    unsigned second = (packed >> 8) & 0xff;
    unsigned duplicated = first | (first << 8) | (second << 16) | (second << 24);
#endif
    return (int)((duplicated & 0x000f000f) | ((duplicated & 0xf000f000) >> 4));
    """
)
def expand_int4x4_low(value: int) -> int: ...


@wp.func_native(
    """
    unsigned packed = (unsigned)value;
#if defined(__CUDA_ARCH__)
    unsigned duplicated = __byte_perm(packed, 0, 0x3322);
#else
    unsigned first = (packed >> 16) & 0xff;
    unsigned second = packed >> 24;
    unsigned duplicated = first | (first << 8) | (second << 16) | (second << 24);
#endif
    return (int)((duplicated & 0x000f000f) | ((duplicated & 0xf000f000) >> 4));
    """
)
def expand_int4x4_high(value: int) -> int: ...


_GROUPED_DECODE_PROJECTION = r"""
#if defined(__CUDA_ARCH__)
    const int lane = tid & 31;
    constexpr int outputs = 8;
    const int column = (tid >> 5) * outputs;
    const NATIVE_TYPE* activations = x.data;
    const NATIVE_TYPE* weights = weight.data;
    float totals[outputs] = {};

    for (int k = lane * 8; k < inner; k += 256) {
        const uint4 activation =
            *reinterpret_cast<const uint4*>(activations + k);
        uint4 packed_weights[outputs];
        #pragma unroll
        for (int output_index = 0; output_index < outputs; ++output_index) {
            const uint4* address = reinterpret_cast<const uint4*>(
                weights + (column + output_index) * inner + k);
            #if NATIVE_BF16
            packed_weights[output_index] = __ldcs(address);
            #else
            packed_weights[output_index] = *address;
            #endif
        }
        #if NATIVE_BF16
        const unsigned* activation_words =
            reinterpret_cast<const unsigned*>(&activation);
        #pragma unroll
        for (int word = 0; word < 4; ++word) {
            float value = __uint_as_float(activation_words[word] << 16);
            #pragma unroll
            for (int output_index = 0; output_index < outputs; ++output_index) {
                const unsigned packed =
                    reinterpret_cast<const unsigned*>(&packed_weights[output_index])[word];
                totals[output_index] = fmaf(
                    value, __uint_as_float(packed << 16), totals[output_index]);
            }
            value = __uint_as_float(activation_words[word] & 0xffff0000u);
            #pragma unroll
            for (int output_index = 0; output_index < outputs; ++output_index) {
                const unsigned packed =
                    reinterpret_cast<const unsigned*>(&packed_weights[output_index])[word];
                totals[output_index] = fmaf(
                    value,
                    __uint_as_float(packed & 0xffff0000u),
                    totals[output_index]);
            }
        }
        #else
        const NATIVE_TYPE* activation_values =
            reinterpret_cast<const NATIVE_TYPE*>(&activation);
        #pragma unroll
        for (int component = 0; component < 8; ++component) {
            const float value = float(activation_values[component]);
            #pragma unroll
            for (int output_index = 0; output_index < outputs; ++output_index) {
                const NATIVE_TYPE* values = reinterpret_cast<const NATIVE_TYPE*>(
                    &packed_weights[output_index]);
                totals[output_index] = fmaf(
                    value, float(values[component]), totals[output_index]);
            }
        }
        #endif
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int output_index = 0; output_index < outputs; ++output_index)
            totals[output_index] += __shfl_down_sync(
                0xffffffffu, totals[output_index], offset);
    }
    if (lane == 0) {
        uint4 packed;
        NATIVE_TYPE* values = reinterpret_cast<NATIVE_TYPE*>(&packed);
        #pragma unroll
        for (int output_index = 0; output_index < outputs; ++output_index)
            values[output_index] = NATIVE_TYPE(totals[output_index]);
        *reinterpret_cast<uint4*>(output.data + column) = packed;
    }
#endif
"""


@lru_cache(maxsize=None)
def get_grouped_decode_projection(dtype: type):
    """Return a native eight-output decode projection for FP16/BF16 storage."""
    if dtype == wp.float16:
        native_type, native_bf16 = "wp::float16", "0"
    elif dtype == wp.bfloat16:
        native_type, native_bf16 = "wp::bfloat16", "1"
    else:
        raise TypeError("Grouped decode projection requires FP16 or BF16")
    snippet = _GROUPED_DECODE_PROJECTION.replace("NATIVE_TYPE", native_type).replace(
        "NATIVE_BF16", native_bf16
    )

    @wp.func_native(snippet)
    def project(
        x: wp.array2d[dtype],
        weight: wp.array2d[dtype],
        output: wp.array2d[dtype],
        tid: int,
        inner: int,
    ): ...

    return project
