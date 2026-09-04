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
    for (int offset = width / 2; offset > 0; offset >>= 1)
        value = max(value, __shfl_down_sync(__activemask(), value, offset, width));
    value = __shfl_sync(__activemask(), value, 0, width);
#endif
    return value;
    """
)
def subgroup_max_broadcast(value: float, width: int) -> float: ...


_BIDIRECTIONAL_ATTENTION_D128 = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Eight warps own 16 query rows each while sharing one 16-row K/V tile.
    // Scores are never materialized globally: online softmax and FP32 output
    // accumulators preserve exact attention with 46.5 KiB of static shared memory.
    constexpr int QUERY_ROWS = 128;
    constexpr int WARP_ROWS = 16;
    constexpr int HEAD_SIZE = 128;
    constexpr int KEY_ROWS = 16;
    constexpr int Q_LD = 136;
    constexpr int KV_LD = 136;
    constexpr int P_LD = 16;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int lane_group = lane & 3;
    const int row = lane >> 2;
    const int query_tiles = (query_length + QUERY_ROWS - 1) / QUERY_ROWS;
    const int tile = blockIdx.x;
    const int tile_index = tile % query_tiles;
    const int head = (tile / query_tiles) % query_heads;
    const int batch = tile / (query_tiles * query_heads);
    const int kv_head = head / (query_heads / kv_heads);
    const int query_start = tile_index * QUERY_ROWS;
    const int warp_query_start = query_start + warp * WARP_ROWS;
    const int query_base = (batch * query_heads + head) * query_length;
    const int key_base = (batch * kv_heads + kv_head) * key_length;

    __shared__ __align__(16) unsigned short q_shared[8 * WARP_ROWS * Q_LD];
    __shared__ __align__(16) unsigned short k_shared[KEY_ROWS * KV_LD];
    __shared__ __align__(16) unsigned short v_shared[KEY_ROWS * KV_LD];
    __shared__ __align__(16) unsigned short p_shared[8 * WARP_ROWS * P_LD];

    const NATIVE_TYPE* query_values = query.data;
    const NATIVE_TYPE* key_values = key.data;
    const NATIVE_TYPE* value_values = value.data;
    NATIVE_TYPE* output_values = output.data;

    for (int copy = threadIdx.x; copy < 8 * WARP_ROWS * (HEAD_SIZE / 8); copy += 256) {
        const int query_row = copy / (HEAD_SIZE / 8);
        const int segment = copy % (HEAD_SIZE / 8);
        unsigned short* destination = q_shared + query_row * Q_LD + segment * 8;
        const int source_row = query_start + query_row;
        if (source_row < query_length) {
            const NATIVE_TYPE* source = query_values + (query_base + source_row) * HEAD_SIZE + segment * 8;
            *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(source);
        } else {
            *reinterpret_cast<uint4*>(destination) = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    float maximum_0 = -3.402823466e38f;
    float maximum_1 = -3.402823466e38f;
    float denominator_0 = 0.0f;
    float denominator_1 = 0.0f;
    float accumulators[8][8] = {};
    const int query_0 = warp_query_start + row;
    const int query_1 = query_0 + 8;
    const bool query_valid_0 = query_0 < query_length && query_valid.data[batch * query_length + query_0];
    const bool query_valid_1 = query_1 < query_length && query_valid.data[batch * query_length + query_1];
    unsigned short* warp_q = q_shared + warp * WARP_ROWS * Q_LD;
    unsigned short* warp_p = p_shared + warp * WARP_ROWS * P_LD;

    for (int key_start = 0; key_start < key_length; key_start += KEY_ROWS) {
        for (int copy = threadIdx.x; copy < KEY_ROWS * (HEAD_SIZE / 8); copy += 256) {
            const int key_row = copy / (HEAD_SIZE / 8);
            const int segment = copy % (HEAD_SIZE / 8);
            const int source_row = key_start + key_row;
            unsigned short* key_destination = k_shared + key_row * KV_LD + segment * 8;
            unsigned short* value_destination = v_shared + key_row * KV_LD + segment * 8;
            if (source_row < key_length) {
                const NATIVE_TYPE* key_source = key_values + (key_base + source_row) * HEAD_SIZE + segment * 8;
                const NATIVE_TYPE* value_source = value_values + (key_base + source_row) * HEAD_SIZE + segment * 8;
                *reinterpret_cast<uint4*>(key_destination) = *reinterpret_cast<const uint4*>(key_source);
                *reinterpret_cast<uint4*>(value_destination) = *reinterpret_cast<const uint4*>(value_source);
            } else {
                *reinterpret_cast<uint4*>(key_destination) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(value_destination) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        float scores[8] = {};
        const int quadrant = lane >> 3;
        const int local_row = lane & 7;
        #pragma unroll
        for (int part = 0; part < HEAD_SIZE / 16; ++part) {
            unsigned a0, a1, a2, a3, b0, b1, b2, b3;
            const unsigned pa = static_cast<unsigned>(__cvta_generic_to_shared(
                warp_q + (local_row + ((quadrant & 1) * 8)) * Q_LD
                + part * 16 + ((quadrant >> 1) * 8)));
            const unsigned pb = static_cast<unsigned>(__cvta_generic_to_shared(
                k_shared + (local_row + ((quadrant >> 1) * 8)) * KV_LD
                + part * 16 + ((quadrant & 1) * 8)));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(pa) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(pb) : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.PTX_TYPE.PTX_TYPE.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(scores[0]), "+f"(scores[1]), "+f"(scores[2]), "+f"(scores[3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.PTX_TYPE.PTX_TYPE.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(scores[4]), "+f"(scores[5]), "+f"(scores[6]), "+f"(scores[7])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));
        }

        const int key_column_0 = key_start + lane_group * 2;
        const int key_column_1 = key_column_0 + 1;
        const int key_column_8 = key_column_0 + 8;
        const int key_column_9 = key_column_0 + 9;
        const bool key_valid_0 = key_column_0 < key_length && key_valid.data[batch * key_length + key_column_0];
        const bool key_valid_1 = key_column_1 < key_length && key_valid.data[batch * key_length + key_column_1];
        const bool key_valid_8 = key_column_8 < key_length && key_valid.data[batch * key_length + key_column_8];
        const bool key_valid_9 = key_column_9 < key_length && key_valid.data[batch * key_length + key_column_9];
        #define VALID_PAIR(Q, K, QVALID, KVALID) ((QVALID) && (KVALID) && (window <= 0 || abs((Q) - (K)) <= window))
        const bool valid_00 = VALID_PAIR(query_0, key_column_0, query_valid_0, key_valid_0);
        const bool valid_01 = VALID_PAIR(query_0, key_column_1, query_valid_0, key_valid_1);
        const bool valid_08 = VALID_PAIR(query_0, key_column_8, query_valid_0, key_valid_8);
        const bool valid_09 = VALID_PAIR(query_0, key_column_9, query_valid_0, key_valid_9);
        const bool valid_10 = VALID_PAIR(query_1, key_column_0, query_valid_1, key_valid_0);
        const bool valid_11 = VALID_PAIR(query_1, key_column_1, query_valid_1, key_valid_1);
        const bool valid_18 = VALID_PAIR(query_1, key_column_8, query_valid_1, key_valid_8);
        const bool valid_19 = VALID_PAIR(query_1, key_column_9, query_valid_1, key_valid_9);
        #undef VALID_PAIR
        scores[0] = valid_00 ? scores[0] * scale : -3.402823466e38f;
        scores[1] = valid_01 ? scores[1] * scale : -3.402823466e38f;
        scores[2] = valid_10 ? scores[2] * scale : -3.402823466e38f;
        scores[3] = valid_11 ? scores[3] * scale : -3.402823466e38f;
        scores[4] = valid_08 ? scores[4] * scale : -3.402823466e38f;
        scores[5] = valid_09 ? scores[5] * scale : -3.402823466e38f;
        scores[6] = valid_18 ? scores[6] * scale : -3.402823466e38f;
        scores[7] = valid_19 ? scores[7] * scale : -3.402823466e38f;
        float block_max_0 = max(max(scores[0], scores[1]), max(scores[4], scores[5]));
        float block_max_1 = max(max(scores[2], scores[3]), max(scores[6], scores[7]));
        block_max_0 = max(block_max_0, __shfl_xor_sync(0xffffffffu, block_max_0, 2, 4));
        block_max_0 = max(block_max_0, __shfl_xor_sync(0xffffffffu, block_max_0, 1, 4));
        block_max_1 = max(block_max_1, __shfl_xor_sync(0xffffffffu, block_max_1, 2, 4));
        block_max_1 = max(block_max_1, __shfl_xor_sync(0xffffffffu, block_max_1, 1, 4));
        const float new_maximum_0 = max(maximum_0, block_max_0);
        const float new_maximum_1 = max(maximum_1, block_max_1);
        const float old_scale_0 = expf(maximum_0 - new_maximum_0);
        const float old_scale_1 = expf(maximum_1 - new_maximum_1);
        float probabilities[8];
        probabilities[0] = valid_00 ? expf(scores[0] - new_maximum_0) : 0.0f;
        probabilities[1] = valid_01 ? expf(scores[1] - new_maximum_0) : 0.0f;
        probabilities[2] = valid_10 ? expf(scores[2] - new_maximum_1) : 0.0f;
        probabilities[3] = valid_11 ? expf(scores[3] - new_maximum_1) : 0.0f;
        probabilities[4] = valid_08 ? expf(scores[4] - new_maximum_0) : 0.0f;
        probabilities[5] = valid_09 ? expf(scores[5] - new_maximum_0) : 0.0f;
        probabilities[6] = valid_18 ? expf(scores[6] - new_maximum_1) : 0.0f;
        probabilities[7] = valid_19 ? expf(scores[7] - new_maximum_1) : 0.0f;
        float probability_sum_0 = probabilities[0] + probabilities[1] + probabilities[4] + probabilities[5];
        float probability_sum_1 = probabilities[2] + probabilities[3] + probabilities[6] + probabilities[7];
        probability_sum_0 += __shfl_xor_sync(0xffffffffu, probability_sum_0, 2, 4);
        probability_sum_0 += __shfl_xor_sync(0xffffffffu, probability_sum_0, 1, 4);
        probability_sum_1 += __shfl_xor_sync(0xffffffffu, probability_sum_1, 2, 4);
        probability_sum_1 += __shfl_xor_sync(0xffffffffu, probability_sum_1, 1, 4);
        denominator_0 = denominator_0 * old_scale_0 + probability_sum_0;
        denominator_1 = denominator_1 * old_scale_1 + probability_sum_1;
        maximum_0 = new_maximum_0;
        maximum_1 = new_maximum_1;
        #pragma unroll
        for (int output_part = 0; output_part < 8; ++output_part) {
            accumulators[output_part][0] *= old_scale_0;
            accumulators[output_part][1] *= old_scale_0;
            accumulators[output_part][2] *= old_scale_1;
            accumulators[output_part][3] *= old_scale_1;
            accumulators[output_part][4] *= old_scale_0;
            accumulators[output_part][5] *= old_scale_0;
            accumulators[output_part][6] *= old_scale_1;
            accumulators[output_part][7] *= old_scale_1;
        }

        NATIVE_TYPE* probability_values = reinterpret_cast<NATIVE_TYPE*>(warp_p);
        const int probability_column = lane_group * 2;
        probability_values[row * P_LD + probability_column] = NATIVE_TYPE(probabilities[0]);
        probability_values[row * P_LD + probability_column + 1] = NATIVE_TYPE(probabilities[1]);
        probability_values[(row + 8) * P_LD + probability_column] = NATIVE_TYPE(probabilities[2]);
        probability_values[(row + 8) * P_LD + probability_column + 1] = NATIVE_TYPE(probabilities[3]);
        probability_values[row * P_LD + probability_column + 8] = NATIVE_TYPE(probabilities[4]);
        probability_values[row * P_LD + probability_column + 9] = NATIVE_TYPE(probabilities[5]);
        probability_values[(row + 8) * P_LD + probability_column + 8] = NATIVE_TYPE(probabilities[6]);
        probability_values[(row + 8) * P_LD + probability_column + 9] = NATIVE_TYPE(probabilities[7]);
        __syncwarp();

        unsigned probability_a0, probability_a1, probability_a2, probability_a3;
        const unsigned probability_address = static_cast<unsigned>(__cvta_generic_to_shared(
            warp_p + (local_row + ((quadrant & 1) * 8)) * P_LD + ((quadrant >> 1) * 8)));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(probability_a0), "=r"(probability_a1), "=r"(probability_a2), "=r"(probability_a3)
            : "r"(probability_address) : "memory");
        #pragma unroll
        for (int output_part = 0; output_part < 8; ++output_part) {
            unsigned b0, b1, b2, b3;
            const unsigned value_address = static_cast<unsigned>(__cvta_generic_to_shared(
                v_shared + (lane & 15) * KV_LD + output_part * 16 + (lane >> 4) * 8));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(value_address) : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.PTX_TYPE.PTX_TYPE.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(accumulators[output_part][0]), "+f"(accumulators[output_part][1]), "+f"(accumulators[output_part][2]), "+f"(accumulators[output_part][3])
                : "r"(probability_a0), "r"(probability_a1), "r"(probability_a2), "r"(probability_a3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.PTX_TYPE.PTX_TYPE.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(accumulators[output_part][4]), "+f"(accumulators[output_part][5]), "+f"(accumulators[output_part][6]), "+f"(accumulators[output_part][7])
                : "r"(probability_a0), "r"(probability_a1), "r"(probability_a2), "r"(probability_a3), "r"(b2), "r"(b3));
        }
        __syncthreads();
    }

    const float inverse_0 = denominator_0 > 0.0f ? 1.0f / denominator_0 : 0.0f;
    const float inverse_1 = denominator_1 > 0.0f ? 1.0f / denominator_1 : 0.0f;
    const int output_column = lane_group * 2;
    #pragma unroll
    for (int output_part = 0; output_part < 8; ++output_part) {
        const int column = output_part * 16 + output_column;
        if (query_0 < query_length) {
            output_values[(query_base + query_0) * HEAD_SIZE + column] = NATIVE_TYPE(accumulators[output_part][0] * inverse_0);
            output_values[(query_base + query_0) * HEAD_SIZE + column + 1] = NATIVE_TYPE(accumulators[output_part][1] * inverse_0);
            output_values[(query_base + query_0) * HEAD_SIZE + column + 8] = NATIVE_TYPE(accumulators[output_part][4] * inverse_0);
            output_values[(query_base + query_0) * HEAD_SIZE + column + 9] = NATIVE_TYPE(accumulators[output_part][5] * inverse_0);
        }
        if (query_1 < query_length) {
            output_values[(query_base + query_1) * HEAD_SIZE + column] = NATIVE_TYPE(accumulators[output_part][2] * inverse_1);
            output_values[(query_base + query_1) * HEAD_SIZE + column + 1] = NATIVE_TYPE(accumulators[output_part][3] * inverse_1);
            output_values[(query_base + query_1) * HEAD_SIZE + column + 8] = NATIVE_TYPE(accumulators[output_part][6] * inverse_1);
            output_values[(query_base + query_1) * HEAD_SIZE + column + 9] = NATIVE_TYPE(accumulators[output_part][7] * inverse_1);
        }
    }
#endif
"""


@lru_cache(maxsize=None)
def get_bidirectional_attention_d128(dtype: type):
    """Return dependency-free SM80+ exact attention for 128-wide heads."""
    if dtype == wp.float16:
        native_type, ptx_type = "wp::float16", "f16"
    elif dtype == wp.bfloat16:
        native_type, ptx_type = "wp::bfloat16", "bf16"
    else:
        raise TypeError("Native attention requires FP16 or BF16")
    snippet = _BIDIRECTIONAL_ATTENTION_D128.replace("NATIVE_TYPE", native_type).replace(
        "PTX_TYPE", ptx_type
    )

    @wp.func_native(snippet)
    def attention(
        query: wp.array2d[dtype],
        key: wp.array2d[dtype],
        value: wp.array2d[dtype],
        query_valid: wp.array2d[wp.bool],
        key_valid: wp.array2d[wp.bool],
        output: wp.array2d[dtype],
        query_heads: int,
        kv_heads: int,
        query_length: int,
        key_length: int,
        scale: float,
        window: int,
    ): ...

    return attention


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    unsigned short storage;
    asm volatile(
        "{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}"
        : "=h"(storage) : "f"(value), "f"(0.0f));
    return static_cast<int>(storage & 0xffu);
#else
    return 0;
#endif
    """
)
def encode_ue4m3(value: float) -> int: ...


@wp.func_native(
    """
    const unsigned bits = unsigned(encoded) & 0xffu;
    const unsigned exponent = (bits >> 3) & 15u;
    const float mantissa = static_cast<float>(bits & 7u);
    if (exponent == 0u)
        return ldexpf(mantissa, -9);
    return ldexpf(1.0f + mantissa * 0.125f, static_cast<int>(exponent) - 7);
    """
)
def decode_ue4m3(encoded: int) -> float: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    const float high = __shfl_down_sync(__activemask(), value, 1, 16);
    unsigned short storage;
    asm volatile(
        "{.reg .b8 fp4; cvt.rn.satfinite.e2m1x2.f32 fp4, %2, %1; "
        "mov.b16 %0, {fp4, 0};}"
        : "=h"(storage) : "f"(value * inverse_scale),
                            "f"(high * inverse_scale));
    return static_cast<int>(storage & 0xffu);
#else
    return 0;
#endif
    """
)
def quantize_e2m1_pair(value: float, inverse_scale: float) -> int: ...


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


_SMALL_BATCH_GROUPED_PROJECTION = r"""
#if defined(__CUDA_ARCH__)
    const int lane = tid & 31;
    constexpr int rows = BATCH_ROWS;
    constexpr int outputs = GROUP_OUTPUTS;
    const int column = (tid >> 5) * outputs;
    const NATIVE_TYPE* activations = x.data;
    const NATIVE_TYPE* weights = weight.data;
    float totals[rows][outputs] = {};

    for (int k = lane * 8; k < inner; k += 256) {
        uint4 packed_activations[rows];
        #pragma unroll
        for (int row = 0; row < rows; ++row)
            packed_activations[row] = *reinterpret_cast<const uint4*>(
                activations + row * inner + k);
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
        #pragma unroll
        for (int row = 0; row < rows; ++row) {
            const unsigned* activation_words =
                reinterpret_cast<const unsigned*>(&packed_activations[row]);
            #pragma unroll
            for (int word = 0; word < 4; ++word) {
                float low = __uint_as_float(activation_words[word] << 16);
                float high = __uint_as_float(activation_words[word] & 0xffff0000u);
                #pragma unroll
                for (int output_index = 0; output_index < outputs; ++output_index) {
                    const unsigned packed = reinterpret_cast<const unsigned*>(
                        &packed_weights[output_index])[word];
                    totals[row][output_index] = fmaf(
                        low, __uint_as_float(packed << 16), totals[row][output_index]);
                    totals[row][output_index] = fmaf(
                        high, __uint_as_float(packed & 0xffff0000u),
                        totals[row][output_index]);
                }
            }
        }
        #else
        #pragma unroll
        for (int row = 0; row < rows; ++row) {
            const NATIVE_TYPE* activation_values =
                reinterpret_cast<const NATIVE_TYPE*>(&packed_activations[row]);
            #pragma unroll
            for (int component = 0; component < 8; ++component) {
                const float value = float(activation_values[component]);
                #pragma unroll
                for (int output_index = 0; output_index < outputs; ++output_index) {
                    const NATIVE_TYPE* values = reinterpret_cast<const NATIVE_TYPE*>(
                        &packed_weights[output_index]);
                    totals[row][output_index] = fmaf(
                        value, float(values[component]), totals[row][output_index]);
                }
            }
        }
        #endif
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int row = 0; row < rows; ++row) {
            #pragma unroll
            for (int output_index = 0; output_index < outputs; ++output_index)
                totals[row][output_index] += __shfl_down_sync(
                    0xffffffffu, totals[row][output_index], offset);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int row = 0; row < rows; ++row) {
            NATIVE_TYPE* destination = output.data + row * output.shape[1] + column;
            #pragma unroll
            for (int output_index = 0; output_index < outputs; ++output_index)
                destination[output_index] = NATIVE_TYPE(totals[row][output_index]);
        }
    }
#endif
"""


@lru_cache(maxsize=None)
def get_small_batch_grouped_projection(
    dtype: type, batch_rows: int, group_outputs: int
):
    """Return a projection that shares each weight load across small batches."""
    if dtype == wp.float16:
        native_type, native_bf16 = "wp::float16", "0"
    elif dtype == wp.bfloat16:
        native_type, native_bf16 = "wp::bfloat16", "1"
    else:
        raise TypeError("Small-batch grouped projection requires FP16 or BF16")
    if batch_rows not in (2, 4, 8) or group_outputs not in (4, 8):
        raise ValueError(
            "Small-batch grouped projection requires 2/4/8 rows and 4/8 outputs"
        )
    snippet = (
        _SMALL_BATCH_GROUPED_PROJECTION.replace("NATIVE_TYPE", native_type)
        .replace("NATIVE_BF16", native_bf16)
        .replace("BATCH_ROWS", str(batch_rows))
        .replace("GROUP_OUTPUTS", str(group_outputs))
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


_PREFILL_MMA_PROJECTION = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    BLOCK_SETUP
    const int column_tiles = columns / TILE_N;
    const int row_base = (block / column_tiles) * TILE_M;
    const int column = (block % column_tiles) * TILE_N;
    constexpr int WARP_COLUMNS = TILE_N / 16;
    const int warp_row = warp / WARP_COLUMNS;
    const int warp_column = warp % WARP_COLUMNS;
    constexpr int LD = STAGE_K + 8;
    constexpr int B_LD = B_STRIDE;
    constexpr int A_SIZE = TILE_M * LD;
    constexpr int B_SIZE = B_ROWS * B_LD;
    constexpr int STAGE_SIZE = A_SIZE + B_SIZE;
    __shared__ __align__(16) unsigned short smem[2 * STAGE_SIZE];
    const NATIVE_TYPE* xp = x.data;
    const NATIVE_TYPE* weightp = weight.data;
    OUTPUT_POINTER
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;

    #pragma unroll
    constexpr int A_SEGMENTS = STAGE_K / 8;
    for (int copy = threadIdx.x; copy < TILE_M * A_SEGMENTS; copy += BLOCK_DIM) {
        const int row = copy / A_SEGMENTS;
        const int segment = copy % A_SEGMENTS;
        unsigned short* dst = smem + row * LD + segment * 8;
        const NATIVE_TYPE* src = xp + (row_base + row) * inner + k_begin + segment * 8;
        const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    }
    B_INITIAL_LOAD
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    for (int k = k_begin, stage = 0; k < k_end; k += STAGE_K, stage ^= 1) {
        if (k + STAGE_K < k_end) {
            unsigned short* next = smem + (stage ^ 1) * STAGE_SIZE;
            #pragma unroll
            for (int copy = threadIdx.x; copy < TILE_M * A_SEGMENTS; copy += BLOCK_DIM) {
                const int row = copy / A_SEGMENTS;
                const int segment = copy % A_SEGMENTS;
                unsigned short* dst = next + row * LD + segment * 8;
                const NATIVE_TYPE* src = xp + (row_base + row) * inner + k + STAGE_K + segment * 8;
                const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
            }
            B_NEXT_LOAD
            asm volatile("cp.async.commit_group;");
        }

        unsigned short* current = smem + stage * STAGE_SIZE;
        unsigned short* sa = current + warp_row * 16 * LD;
        unsigned short* sb = current + A_SIZE;
        const int quadrant = lane >> 3;
        const int local_row = lane & 7;
        #pragma unroll
        for (int part = 0; part < STAGE_K / 16; ++part) {
            unsigned a0, a1, a2, a3, b0, b1, b2, b3;
            const unsigned pa = static_cast<unsigned>(__cvta_generic_to_shared(sa + (local_row + ((quadrant & 1) * 8)) * LD + part * 16 + ((quadrant >> 1) * 8)));
            B_FRAGMENT_ADDRESS
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(pa) : "memory");
            B_FRAGMENT_LOAD
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.PTX_TYPE.PTX_TYPE.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.PTX_TYPE.PTX_TYPE.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));
        }
        if (k + STAGE_K < k_end) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
    }

    const int row = warp_row * 16 + (lane >> 2);
    const int col = column + warp_column * 16 + (lane & 3) * 2;
    OUTPUT_EPILOGUE
#endif
"""


_PREFILL_MMA_LOW_PRECISION_EPILOGUE = r"""
    op[row * columns + col] = OUTPUT_TYPE(c0);
    op[row * columns + col + 1] = OUTPUT_TYPE(c1);
    op[(row + 8) * columns + col] = OUTPUT_TYPE(c2);
    op[(row + 8) * columns + col + 1] = OUTPUT_TYPE(c3);
    op[row * columns + col + 8] = OUTPUT_TYPE(c4);
    op[row * columns + col + 9] = OUTPUT_TYPE(c5);
    op[(row + 8) * columns + col + 8] = OUTPUT_TYPE(c6);
    op[(row + 8) * columns + col + 9] = OUTPUT_TYPE(c7);
"""


_PREFILL_MMA_FP32_EPILOGUE = r"""
    *reinterpret_cast<float2*>(op + row * columns + col) = make_float2(c0, c1);
    *reinterpret_cast<float2*>(op + (row + 8) * columns + col) = make_float2(c2, c3);
    *reinterpret_cast<float2*>(op + row * columns + col + 8) = make_float2(c4, c5);
    *reinterpret_cast<float2*>(op + (row + 8) * columns + col + 8) = make_float2(c6, c7);
"""


_PREFILL_MMA_TRANSPOSED_RIGHT = {
    "B_STRIDE": "STAGE_K + 8",
    "B_ROWS": "TILE_N",
    "B_INITIAL_LOAD": r"""
    #pragma unroll
    constexpr int B_SEGMENTS = STAGE_K / 8;
    for (int copy = threadIdx.x; copy < TILE_N * B_SEGMENTS; copy += BLOCK_DIM) {
        const int row = copy / B_SEGMENTS;
        const int segment = copy % B_SEGMENTS;
        unsigned short* dst = smem + A_SIZE + row * B_LD + segment * 8;
        const NATIVE_TYPE* src = weightp + (column + row) * inner + k_begin + segment * 8;
        const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    }
""",
    "B_NEXT_LOAD": r"""
            #pragma unroll
            for (int copy = threadIdx.x; copy < TILE_N * B_SEGMENTS; copy += BLOCK_DIM) {
                const int row = copy / B_SEGMENTS;
                const int segment = copy % B_SEGMENTS;
                unsigned short* dst = next + A_SIZE + row * B_LD + segment * 8;
                const NATIVE_TYPE* src = weightp + (column + row) * inner + k + STAGE_K + segment * 8;
                const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
            }
""",
    "B_FRAGMENT_ADDRESS": r"""
            const unsigned pb = static_cast<unsigned>(__cvta_generic_to_shared(
                sb + (warp_column * 16 + local_row + ((quadrant >> 1) * 8)) * B_LD
                + part * 16 + ((quadrant & 1) * 8)));
""",
    "B_FRAGMENT_LOAD": r"""
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(pb) : "memory");
""",
}


_PREFILL_MMA_REGULAR_RIGHT = {
    "B_STRIDE": "TILE_N + 8",
    "B_ROWS": "STAGE_K",
    "B_INITIAL_LOAD": r"""
    #pragma unroll
    for (int copy = threadIdx.x; copy < TILE_N * (STAGE_K / 8); copy += BLOCK_DIM) {
        const int row = copy / (TILE_N / 8);
        const int segment = copy % (TILE_N / 8);
        unsigned short* dst = smem + A_SIZE + row * B_LD + segment * 8;
        const NATIVE_TYPE* src = weightp + (k_begin + row) * columns + column + segment * 8;
        const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    }
""",
    "B_NEXT_LOAD": r"""
            #pragma unroll
            for (int copy = threadIdx.x; copy < TILE_N * (STAGE_K / 8); copy += BLOCK_DIM) {
                const int row = copy / (TILE_N / 8);
                const int segment = copy % (TILE_N / 8);
                unsigned short* dst = next + A_SIZE + row * B_LD + segment * 8;
                const NATIVE_TYPE* src = weightp + (k + STAGE_K + row) * columns + column + segment * 8;
                const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
            }
""",
    "B_FRAGMENT_ADDRESS": r"""
            const unsigned pb = static_cast<unsigned>(__cvta_generic_to_shared(
                sb + (part * 16 + (lane & 15)) * B_LD
                + warp_column * 16 + (lane >> 4) * 8));
""",
    "B_FRAGMENT_LOAD": r"""
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(pb) : "memory");
""",
}


def _prefill_mma_projection_snippet(
    dtype: type,
    tile_m: int,
    tile_n: int,
    *,
    transposed_right: bool,
    block_setup: str,
    output_pointer: str,
    output_type: str,
    stage_k: int,
) -> str:
    """Specialize the one shared MMA pipeline for direct or split-K output."""
    if (tile_m, tile_n) not in (
        (16, 64),
        (64, 64),
        (64, 32),
        (64, 16),
        (128, 32),
    ):
        raise ValueError("Unsupported prefill MMA tile geometry")
    block_dim = tile_m * tile_n // 8
    if stage_k not in (32, 64):
        raise ValueError("Prefill MMA stage K must be 32 or 64")
    if dtype == wp.float16:
        native_type, ptx_type = "wp::float16", "f16"
    elif dtype == wp.bfloat16:
        native_type, ptx_type = "wp::bfloat16", "bf16"
    else:
        raise TypeError("Prefill MMA projection requires FP16 or BF16")
    layout = (
        _PREFILL_MMA_TRANSPOSED_RIGHT
        if transposed_right
        else _PREFILL_MMA_REGULAR_RIGHT
    )
    snippet = _PREFILL_MMA_PROJECTION
    for marker, replacement in layout.items():
        snippet = snippet.replace(marker, replacement)
    epilogue = (
        _PREFILL_MMA_FP32_EPILOGUE
        if output_type == "static_cast<float>"
        else _PREFILL_MMA_LOW_PRECISION_EPILOGUE
    )
    return (
        snippet.replace("BLOCK_SETUP", block_setup)
        .replace("OUTPUT_EPILOGUE", epilogue)
        .replace("OUTPUT_POINTER", output_pointer)
        .replace("OUTPUT_TYPE", output_type)
        .replace("NATIVE_TYPE", native_type)
        .replace("PTX_TYPE", ptx_type)
        .replace("TILE_M", str(tile_m))
        .replace("TILE_N", str(tile_n))
        .replace("BLOCK_DIM", str(block_dim))
        .replace("STAGE_K", str(stage_k))
    )


@lru_cache(maxsize=None)
def get_prefill_mma_projection(
    dtype: type,
    tile_m: int,
    tile_n: int,
    *,
    transposed_right: bool = True,
    stage_k: int = 32,
):
    """Return an SM80+ GEMM primitive for either right-operand storage layout."""
    snippet = _prefill_mma_projection_snippet(
        dtype,
        tile_m,
        tile_n,
        transposed_right=transposed_right,
        block_setup=(
            "const int block = tid / BLOCK_DIM; const int k_begin = 0; "
            "const int k_end = inner;"
        ),
        output_pointer="NATIVE_TYPE* op = output.data + row_base * columns;",
        output_type="NATIVE_TYPE",
        stage_k=stage_k,
    )

    @wp.func_native(snippet)
    def project(
        x: wp.array2d[dtype],
        weight: wp.array2d[dtype],
        output: wp.array2d[dtype],
        tid: int,
        columns: int,
        inner: int,
    ): ...

    return project


@lru_cache(maxsize=None)
def get_prefill_mma_split_k_projection(
    dtype: type,
    tile_m: int,
    tile_n: int,
    *,
    transposed_right: bool = True,
):
    """Return a 64-stage SM80+ GEMM primitive with deterministic FP32 K-splits."""
    snippet = _prefill_mma_projection_snippet(
        dtype,
        tile_m,
        tile_n,
        transposed_right=transposed_right,
        block_setup=(
            "const int launch_block = tid / BLOCK_DIM; "
            "const int split = launch_block % splits; "
            "const int block = launch_block / splits; "
            "const int split_inner = inner / splits; "
            "const int k_begin = split * split_inner; "
            "const int k_end = k_begin + split_inner;"
        ),
        output_pointer=(
            "float* op = output.data + (split * rows + row_base) * columns;"
        ),
        output_type="static_cast<float>",
        stage_k=64,
    )

    @wp.func_native(snippet)
    def project(
        x: wp.array2d[dtype],
        weight: wp.array2d[dtype],
        output: wp.array2d[wp.float32],
        tid: int,
        rows: int,
        columns: int,
        inner: int,
        splits: int,
    ): ...

    return project


_Q8_GROUPED_DECODE_PROJECTION = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    const int lane = tid & 7;
    const int column_0 = (tid >> 3) * OUTPUTS_PER_GROUP;
    float total_0 = 0.0f;
    #if OUTPUTS_PER_GROUP == 2
    const int column_1 = column_0 + 1;
    float total_1 = 0.0f;
    #endif
    for (int block = 0; block < blocks; ++block) {
        const int activation = static_cast<int>(activations.data[block * 8 + lane]);
        const int weight_0 = static_cast<int>(
            weights.data[(column_0 * blocks + block) * 8 + lane]);
        #if OUTPUTS_PER_GROUP == 2
        const int weight_1 = static_cast<int>(
            weights.data[(column_1 * blocks + block) * 8 + lane]);
        #endif
        int dot_0;
        #if OUTPUTS_PER_GROUP == 2
        int dot_1;
        #endif
        const int zero = 0;
        asm("dp4a.s32.s32 %0, %1, %2, %3;"
            : "=r"(dot_0) : "r"(weight_0), "r"(activation), "r"(zero));
        #if OUTPUTS_PER_GROUP == 2
        asm("dp4a.s32.s32 %0, %1, %2, %3;"
            : "=r"(dot_1) : "r"(weight_1), "r"(activation), "r"(zero));
        #endif
        float activation_scale =
            lane == 0 ? activation_scales.data[block] : 0.0f;
        float weight_scale_0 = lane == 0
            ? static_cast<float>(weight_scales.data[column_0 * blocks + block])
            : 0.0f;
        #if OUTPUTS_PER_GROUP == 2
        float weight_scale_1 = lane == 0
            ? static_cast<float>(weight_scales.data[column_1 * blocks + block])
            : 0.0f;
        #endif
        activation_scale = __shfl_sync(0xffffffffu, activation_scale, 0, 8);
        weight_scale_0 = __shfl_sync(0xffffffffu, weight_scale_0, 0, 8);
        #if OUTPUTS_PER_GROUP == 2
        weight_scale_1 = __shfl_sync(0xffffffffu, weight_scale_1, 0, 8);
        #endif
        total_0 += static_cast<float>(dot_0) * activation_scale * weight_scale_0;
        #if OUTPUTS_PER_GROUP == 2
        total_1 += static_cast<float>(dot_1) * activation_scale * weight_scale_1;
        #endif
    }
    for (int offset = 4; offset > 0; offset >>= 1) {
        total_0 += __shfl_down_sync(0xffffffffu, total_0, offset, 8);
        #if OUTPUTS_PER_GROUP == 2
        total_1 += __shfl_down_sync(0xffffffffu, total_1, offset, 8);
        #endif
    }
    if (lane == 0) {
        output.data[column_0] = NATIVE_TYPE(total_0);
        #if OUTPUTS_PER_GROUP == 2
        output.data[column_1] = NATIVE_TYPE(total_1);
        #endif
    }
#endif
"""


@lru_cache(maxsize=None)
def get_q8_grouped_decode_projection(dtype: type, outputs_per_group: int):
    """Return a signed-Q8 grouped DP4A decode projection."""
    if outputs_per_group not in (1, 2):
        raise ValueError("Q8 native decode grouping must be 1 or 2")
    if dtype == wp.float16:
        native_type = "wp::float16"
    elif dtype == wp.bfloat16:
        native_type = "wp::bfloat16"
    else:
        raise TypeError("Q8 grouped decode requires FP16 or BF16 output")
    snippet = _Q8_GROUPED_DECODE_PROJECTION.replace("NATIVE_TYPE", native_type).replace(
        "OUTPUTS_PER_GROUP", str(outputs_per_group)
    )

    @wp.func_native(snippet)
    def project(
        activations: wp.array3d[wp.uint32],
        activation_scales: wp.array2d[wp.float32],
        weights: wp.array3d[wp.uint32],
        weight_scales: wp.array2d[wp.float16],
        output: wp.array2d[dtype],
        tid: int,
        blocks: int,
    ): ...

    return project


_Q8_PREFILL_MMA_16X32_PROJECTION = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int block = tid >> 7;
    const int row_tiles = output.shape.dims[0] >> 4;
    const int row_base = (block % row_tiles) << 4;
    const int column_base = ((block / row_tiles) << 5) + (warp << 3);
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int row_0 = row_base + group;
    const int row_1 = row_0 + 8;
    const int column_0 = column_base + (thread_in_group << 1);
    const int column_1 = column_0 + 1;
    float total_0 = 0.0f, total_1 = 0.0f;
    float total_2 = 0.0f, total_3 = 0.0f;

    for (int k_block = 0; k_block < blocks; ++k_block) {
        const int k = k_block << 5;
        const int fragment = thread_in_group << 2;
        const signed char* a_row_0 = activations.data + row_0 * (blocks << 5) + k;
        const signed char* a_row_1 = activations.data + row_1 * (blocks << 5) + k;
        const unsigned a0 = *reinterpret_cast<const unsigned*>(a_row_0 + fragment);
        const unsigned a1 = *reinterpret_cast<const unsigned*>(a_row_1 + fragment);
        const unsigned a2 = *reinterpret_cast<const unsigned*>(a_row_0 + fragment + 16);
        const unsigned a3 = *reinterpret_cast<const unsigned*>(a_row_1 + fragment + 16);
        const int weight_column = column_base + group;
        const signed char* b_column = weights.data + weight_column * (blocks << 5) + k;
        const unsigned b0 = *reinterpret_cast<const unsigned*>(b_column + fragment);
        const unsigned b1 = *reinterpret_cast<const unsigned*>(b_column + fragment + 16);
        int d0, d1, d2, d3;
        const int zero = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "r"(zero), "r"(zero), "r"(zero), "r"(zero));
        const float activation_scale_0 = activation_scales.data[row_0 * blocks + k_block];
        const float activation_scale_1 = activation_scales.data[row_1 * blocks + k_block];
        const float weight_scale_0 = static_cast<float>(weight_scales.data[column_0 * blocks + k_block]);
        const float weight_scale_1 = static_cast<float>(weight_scales.data[column_1 * blocks + k_block]);
        total_0 += static_cast<float>(d0) * activation_scale_0 * weight_scale_0;
        total_1 += static_cast<float>(d1) * activation_scale_0 * weight_scale_1;
        total_2 += static_cast<float>(d2) * activation_scale_1 * weight_scale_0;
        total_3 += static_cast<float>(d3) * activation_scale_1 * weight_scale_1;
    }
    output.data[row_0 * columns + column_0] = NATIVE_TYPE(total_0);
    output.data[row_0 * columns + column_1] = NATIVE_TYPE(total_1);
    output.data[row_1 * columns + column_0] = NATIVE_TYPE(total_2);
    output.data[row_1 * columns + column_1] = NATIVE_TYPE(total_3);
#endif
"""


_Q8_PREFILL_MMA_64X32_PROJECTION = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int block = tid >> 9;
    const int row_tiles = output.shape.dims[0] >> 6;
    const int row_base = (block % row_tiles) << 6;
    const int column_tile = (block / row_tiles) << 5;
    const int warp_row = warp >> 2;
    const int warp_column = warp & 3;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int local_row_0 = (warp_row << 4) + group;
    const int local_row_1 = local_row_0 + 8;
    const int local_column_0 = (warp_column << 3) + (thread_in_group << 1);
    const int local_column_1 = local_column_0 + 1;
    constexpr int VALUE_STAGE = 64 * 32 + 32 * 32;
    float total_0 = 0.0f, total_1 = 0.0f;
    float total_2 = 0.0f, total_3 = 0.0f;
    __shared__ __align__(16) signed char values[2 * VALUE_STAGE];
    __shared__ float activation_scale_tile[2 * 64];
    __shared__ wp::float16 weight_scale_tile[2 * 32];

    const int copy = threadIdx.x;
    if (copy < 128) {
        const int row = copy >> 1;
        const int segment = copy & 1;
        signed char* dst = values + row * 32 + segment * 16;
        const signed char* src = activations.data +
            (row_base + row) * (blocks << 5) + segment * 16;
        const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    } else if (copy < 192) {
        const int weight_copy = copy - 128;
        const int row = weight_copy >> 1;
        const int segment = weight_copy & 1;
        signed char* dst = values + 64 * 32 + row * 32 + segment * 16;
        const signed char* src = weights.data +
            (column_tile + row) * (blocks << 5) + segment * 16;
        const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    }
    if (copy < 64)
        activation_scale_tile[copy] = activation_scales.data[(row_base + copy) * blocks];
    if (copy >= 64 && copy < 96)
        weight_scale_tile[copy - 64] = weight_scales.data[(column_tile + copy - 64) * blocks];
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    for (int k_block = 0, stage = 0; k_block < blocks; ++k_block, stage ^= 1) {
        if (k_block + 1 < blocks) {
            signed char* next = values + (stage ^ 1) * VALUE_STAGE;
            if (copy < 128) {
                const int row = copy >> 1;
                const int segment = copy & 1;
                signed char* dst = next + row * 32 + segment * 16;
                const signed char* src = activations.data +
                    (row_base + row) * (blocks << 5) + ((k_block + 1) << 5) + segment * 16;
                const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
            } else if (copy < 192) {
                const int weight_copy = copy - 128;
                const int row = weight_copy >> 1;
                const int segment = weight_copy & 1;
                signed char* dst = next + 64 * 32 + row * 32 + segment * 16;
                const signed char* src = weights.data +
                    (column_tile + row) * (blocks << 5) + ((k_block + 1) << 5) + segment * 16;
                const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
            }
            if (copy < 64)
                activation_scale_tile[(stage ^ 1) * 64 + copy] =
                    activation_scales.data[(row_base + copy) * blocks + k_block + 1];
            if (copy >= 64 && copy < 96)
                weight_scale_tile[(stage ^ 1) * 32 + copy - 64] =
                    weight_scales.data[(column_tile + copy - 64) * blocks + k_block + 1];
            asm volatile("cp.async.commit_group;");
        }

        signed char* current = values + stage * VALUE_STAGE;
        const int fragment = thread_in_group << 2;
        const signed char* a_row_0 = current + local_row_0 * 32;
        const signed char* a_row_1 = current + local_row_1 * 32;
        const unsigned a0 = *reinterpret_cast<const unsigned*>(a_row_0 + fragment);
        const unsigned a1 = *reinterpret_cast<const unsigned*>(a_row_1 + fragment);
        const unsigned a2 = *reinterpret_cast<const unsigned*>(a_row_0 + fragment + 16);
        const unsigned a3 = *reinterpret_cast<const unsigned*>(a_row_1 + fragment + 16);
        const signed char* b_row = current + 64 * 32 + ((warp_column << 3) + group) * 32;
        const unsigned b0 = *reinterpret_cast<const unsigned*>(b_row + fragment);
        const unsigned b1 = *reinterpret_cast<const unsigned*>(b_row + fragment + 16);
        int d0, d1, d2, d3;
        const int zero = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "r"(zero), "r"(zero), "r"(zero), "r"(zero));
        const float activation_scale_0 = activation_scale_tile[stage * 64 + local_row_0];
        const float activation_scale_1 = activation_scale_tile[stage * 64 + local_row_1];
        const float weight_scale_0 = static_cast<float>(
            weight_scale_tile[stage * 32 + local_column_0]);
        const float weight_scale_1 = static_cast<float>(
            weight_scale_tile[stage * 32 + local_column_1]);
        total_0 += static_cast<float>(d0) * activation_scale_0 * weight_scale_0;
        total_1 += static_cast<float>(d1) * activation_scale_0 * weight_scale_1;
        total_2 += static_cast<float>(d2) * activation_scale_1 * weight_scale_0;
        total_3 += static_cast<float>(d3) * activation_scale_1 * weight_scale_1;
        if (k_block + 1 < blocks) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
    }
    const int row_0 = row_base + local_row_0;
    const int row_1 = row_base + local_row_1;
    const int column_0 = column_tile + local_column_0;
    const int column_1 = column_0 + 1;
    output.data[row_0 * columns + column_0] = NATIVE_TYPE(total_0);
    output.data[row_0 * columns + column_1] = NATIVE_TYPE(total_1);
    output.data[row_1 * columns + column_0] = NATIVE_TYPE(total_2);
    output.data[row_1 * columns + column_1] = NATIVE_TYPE(total_3);
#endif
"""


@lru_cache(maxsize=None)
def get_q8_prefill_mma_projection(dtype: type, tile_m: int):
    """Return an SM80+ block-Q8 projection using signed INT8 tensor cores."""
    if dtype == wp.float16:
        native_type = "wp::float16"
    elif dtype == wp.bfloat16:
        native_type = "wp::bfloat16"
    else:
        raise TypeError("Q8 prefill MMA projection requires FP16 or BF16 output")
    snippets = {
        16: _Q8_PREFILL_MMA_16X32_PROJECTION,
        64: _Q8_PREFILL_MMA_64X32_PROJECTION,
    }
    try:
        snippet = snippets[tile_m].replace("NATIVE_TYPE", native_type)
    except KeyError as exc:
        raise ValueError(f"Unsupported Q8 prefill tile height {tile_m}") from exc

    @wp.func_native(snippet)
    def project(
        activations: wp.array2d[wp.int8],
        activation_scales: wp.array2d[wp.float32],
        weights: wp.array3d[wp.int8],
        weight_scales: wp.array2d[wp.float16],
        output: wp.array2d[dtype],
        tid: int,
        columns: int,
        blocks: int,
    ): ...

    return project


_NVFP4_MMA_PROJECTION = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int column_tiles = columns >> 3;
#if REUSE_WEIGHTS
    __shared__ unsigned shared_weight_words[256];
    __shared__ unsigned shared_scale_words[32];
    const int tile = blockIdx.x;
    const int row_base = ((tile / column_tiles) << 6) + ((threadIdx.x >> 5) << 4);
    const int column_base = (tile - (tile / column_tiles) * column_tiles) << 3;
#elif SPLIT_K
    __shared__ float split_partials[1024];
    const int local_warp = threadIdx.x >> 5;
    const int tiles_per_block = WARPS_PER_BLOCK / SPLIT_K;
    const int tile = blockIdx.x * tiles_per_block + local_warp / SPLIT_K;
    const int row_tile = tile / column_tiles;
    const int row_base = row_tile << 4;
    const int column_base = (tile - row_tile * column_tiles) << 3;
#else
    const int warp = tid >> 5;
    const int row_tile = warp / column_tiles;
    const int row_base = row_tile << 4;
    const int column_base = (warp - row_tile * column_tiles) << 3;
#endif
    float total_0 = 0.0f, total_1 = 0.0f, total_2 = 0.0f, total_3 = 0.0f;

#if SPLIT_K
    for (int block = local_warp % SPLIT_K; block < blocks64; block += SPLIT_K) {
#else
    for (int block = 0; block < blocks64; ++block) {
#endif
        const int packed_base = block << 5;
        const int scale_base = block << 2;
        const int fragment = thread_in_group << 2;
        const unsigned char* a_row_0 = activations.data +
            (row_base + group) * activations.shape[1] + packed_base;
        const unsigned char* a_row_1 = activations.data +
            (row_base + group + 8) * activations.shape[1] + packed_base;
        const unsigned a0 = *reinterpret_cast<const unsigned*>(a_row_0 + fragment);
        const unsigned a1 = *reinterpret_cast<const unsigned*>(a_row_1 + fragment);
        const unsigned a2 = *reinterpret_cast<const unsigned*>(a_row_0 + fragment + 16);
        const unsigned a3 = *reinterpret_cast<const unsigned*>(a_row_1 + fragment + 16);
#if REUSE_WEIGHTS
        const int staged = block & 3;
        if (staged == 0) {
            for (int load_group = 0; load_group < 2; ++load_group) {
                const int index = threadIdx.x + load_group * 128;
                const int staged_block = index >> 6;
                const int offset = index & 63;
                const int source_block = block + staged_block;
                unsigned value = 0;
                if (source_block < blocks64) {
                    const int shared_row = offset >> 3;
                    const int shared_word = offset & 7;
                    const unsigned char* source = weights.data +
                        (column_base + shared_row) * weights.shape[1] +
                        (source_block << 5);
                    value = *reinterpret_cast<const unsigned*>(
                        source + (shared_word << 2));
                }
                shared_weight_words[index] = value;
            }
            if (threadIdx.x < 32) {
                const int staged_block = threadIdx.x >> 3;
                const int shared_row = threadIdx.x & 7;
                const int source_block = block + staged_block;
                unsigned value = 0;
                if (source_block < blocks64) {
                    const unsigned char* source = weight_scales.data +
                        (column_base + shared_row) * weight_scales.shape[1] +
                        (source_block << 2);
                    value = *reinterpret_cast<const unsigned*>(source);
                }
                shared_scale_words[threadIdx.x] = value;
            }
            __syncthreads();
        }
        const unsigned char* b_row = reinterpret_cast<const unsigned char*>(
            shared_weight_words + staged * 64 + (group << 3));
#else
        const unsigned char* b_row = weights.data +
            (column_base + group) * weights.shape[1] + packed_base;
#endif
        const unsigned b0 = *reinterpret_cast<const unsigned*>(b_row + fragment);
        const unsigned b1 = *reinterpret_cast<const unsigned*>(b_row + fragment + 16);

        // CUTLASS SFALayout: thread strides <8,0,1>, value stride 16.
        const int a_scale_row = row_base + group + ((lane & 1) << 3);
        const unsigned char* as = activation_scales.data +
            a_scale_row * activation_scales.shape[1] + scale_base;
        // CUTLASS SFBLayout: thread strides <0,1>, value stride 8.
#if REUSE_WEIGHTS
        const unsigned char* bs = reinterpret_cast<const unsigned char*>(
            shared_scale_words + staged * 8 + group);
#else
        const unsigned char* bs = weight_scales.data +
            (column_base + group) * weight_scales.shape[1] + scale_base;
#endif
        const unsigned sfa = *reinterpret_cast<const unsigned*>(as);
        const unsigned sfb = *reinterpret_cast<const unsigned*>(bs);
        const unsigned short selector = 0;

        float d0, d1, d2, d3;
        const float zero = 0.0f;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
            "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
            "{%14}, {%15,%16}, {%17}, {%18,%19};"
            : "=&f"(d0), "=&f"(d1), "=&f"(d2), "=&f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "f"(zero), "f"(zero), "f"(zero), "f"(zero),
              "r"(sfa), "h"(selector), "h"(selector), "r"(sfb),
              "h"(selector), "h"(selector));
        total_0 += d0; total_1 += d1; total_2 += d2; total_3 += d3;
#if REUSE_WEIGHTS
        if (staged == 3 || block + 1 == blocks64) {
            __syncthreads();
        }
#endif
    }

#if SPLIT_K
    const int partial_index = (threadIdx.x >> 5) * 128 + lane * 4;
    split_partials[partial_index] = total_0;
    split_partials[partial_index + 1] = total_1;
    split_partials[partial_index + 2] = total_2;
    split_partials[partial_index + 3] = total_3;
    __syncthreads();
    if (local_warp % SPLIT_K == 0) {
        for (int split = 1; split < SPLIT_K; ++split) {
            const int offset = partial_index + split * 128;
            total_0 += split_partials[offset];
            total_1 += split_partials[offset + 1];
            total_2 += split_partials[offset + 2];
            total_3 += split_partials[offset + 3];
        }
    }
#endif

    const int row_0 = row_base + group;
    const int row_1 = row_0 + 8;
    const int column_0 = column_base + (thread_in_group << 1);
    const float output_scale_0 = activation_global_scales.data[row_0] * weight_global_scale;
    const float output_scale_1 = activation_global_scales.data[row_1] * weight_global_scale;
#if SPLIT_K
    if (local_warp % SPLIT_K == 0) {
#endif
        output.data[row_0 * columns + column_0] = NATIVE_TYPE(total_0 * output_scale_0);
        output.data[row_0 * columns + column_0 + 1] = NATIVE_TYPE(total_1 * output_scale_0);
        output.data[row_1 * columns + column_0] = NATIVE_TYPE(total_2 * output_scale_1);
        output.data[row_1 * columns + column_0 + 1] = NATIVE_TYPE(total_3 * output_scale_1);
#if SPLIT_K
    }
#endif
#endif
"""


@lru_cache(maxsize=None)
def get_nvfp4_mma_projection(
    dtype: type, reuse_weights: bool = False, split_k: int = 0
):
    """Return the SM120a m16n8k64 block-scaled NVFP4 projection."""
    if dtype == wp.float16:
        native_type = "wp::float16"
    elif dtype == wp.bfloat16:
        native_type = "wp::bfloat16"
    else:
        raise TypeError("NVFP4 MMA output requires FP16 or BF16")

    source = _NVFP4_MMA_PROJECTION.replace("NATIVE_TYPE", native_type).replace(
        "REUSE_WEIGHTS", "1" if reuse_weights else "0"
    )
    source = source.replace("SPLIT_K", str(split_k))
    source = source.replace("WARPS_PER_BLOCK", str(max(4, split_k)))

    @wp.func_native(source)
    def project(
        activations: wp.array2d[wp.uint8],
        activation_scales: wp.array2d[wp.uint8],
        activation_global_scales: wp.array1d[wp.float32],
        weights: wp.array2d[wp.uint8],
        weight_scales: wp.array2d[wp.uint8],
        output: wp.array2d[dtype],
        tid: int,
        columns: int,
        blocks64: int,
        weight_global_scale: float,
    ): ...

    return project
