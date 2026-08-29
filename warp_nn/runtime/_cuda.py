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


_PREFILL_MMA_PROJECTION = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int block = tid >> 7;
    const int column_tiles = columns / 64;
    const int row_base = (block / column_tiles) * 16;
    const int column = (block % column_tiles) * 64;
    constexpr int LD = 40;
    constexpr int A_SIZE = 16 * LD;
    constexpr int B_SIZE = 64 * LD;
    constexpr int STAGE_SIZE = A_SIZE + B_SIZE;
    __shared__ __align__(16) unsigned short smem[2 * STAGE_SIZE];
    const NATIVE_TYPE* xp = x.data;
    const NATIVE_TYPE* weightp = weight.data;
    NATIVE_TYPE* op = output.data + row_base * columns;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;

    if (threadIdx.x < 64) {
        const int row = threadIdx.x >> 2;
        const int segment = threadIdx.x & 3;
        unsigned short* dst = smem + row * LD + segment * 8;
        const NATIVE_TYPE* src = xp + (row_base + row) * inner + segment * 8;
        const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    }
    #pragma unroll
    for (int copy = 0; copy < 2; ++copy) {
        const int local = threadIdx.x + copy * 128;
        const int row = local >> 2;
        const int segment = local & 3;
        unsigned short* dst = smem + A_SIZE + row * LD + segment * 8;
        const NATIVE_TYPE* src = weightp + (column + row) * inner + segment * 8;
        const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    for (int k = 0, stage = 0; k < inner; k += 32, stage ^= 1) {
        if (k + 32 < inner) {
            unsigned short* next = smem + (stage ^ 1) * STAGE_SIZE;
            if (threadIdx.x < 64) {
                const int row = threadIdx.x >> 2;
                const int segment = threadIdx.x & 3;
                unsigned short* dst = next + row * LD + segment * 8;
                const NATIVE_TYPE* src = xp + (row_base + row) * inner + k + 32 + segment * 8;
                const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
            }
            #pragma unroll
            for (int copy = 0; copy < 2; ++copy) {
                const int local = threadIdx.x + copy * 128;
                const int row = local >> 2;
                const int segment = local & 3;
                unsigned short* dst = next + A_SIZE + row * LD + segment * 8;
                const NATIVE_TYPE* src = weightp + (column + row) * inner + k + 32 + segment * 8;
                const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
            }
            asm volatile("cp.async.commit_group;");
        }

        unsigned short* current = smem + stage * STAGE_SIZE;
        unsigned short* sa = current;
        unsigned short* sb = current + A_SIZE + warp * 16 * LD;
        const int quadrant = lane >> 3;
        const int local_row = lane & 7;
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            unsigned a0, a1, a2, a3, b0, b1, b2, b3;
            const unsigned pa = static_cast<unsigned>(__cvta_generic_to_shared(sa + (local_row + ((quadrant & 1) * 8)) * LD + part * 16 + ((quadrant >> 1) * 8)));
            const unsigned pb = static_cast<unsigned>(__cvta_generic_to_shared(sb + (local_row + ((quadrant >> 1) * 8)) * LD + part * 16 + ((quadrant & 1) * 8)));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(pa) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(pb) : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.PTX_TYPE.PTX_TYPE.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.PTX_TYPE.PTX_TYPE.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));
        }
        if (k + 32 < inner) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
    }

    const int row = lane >> 2;
    const int col = column + warp * 16 + (lane & 3) * 2;
    op[row * columns + col] = NATIVE_TYPE(c0);
    op[row * columns + col + 1] = NATIVE_TYPE(c1);
    op[(row + 8) * columns + col] = NATIVE_TYPE(c2);
    op[(row + 8) * columns + col + 1] = NATIVE_TYPE(c3);
    op[row * columns + col + 8] = NATIVE_TYPE(c4);
    op[row * columns + col + 9] = NATIVE_TYPE(c5);
    op[(row + 8) * columns + col + 8] = NATIVE_TYPE(c6);
    op[(row + 8) * columns + col + 9] = NATIVE_TYPE(c7);
#endif
"""


@lru_cache(maxsize=None)
def get_prefill_mma_projection(dtype: type):
    """Return an SM80+ M16xN64 projection primitive for FP16/BF16 storage."""
    if dtype == wp.float16:
        native_type, ptx_type = "wp::float16", "f16"
    elif dtype == wp.bfloat16:
        native_type, ptx_type = "wp::bfloat16", "bf16"
    else:
        raise TypeError("Prefill MMA projection requires FP16 or BF16")
    snippet = _PREFILL_MMA_PROJECTION.replace("NATIVE_TYPE", native_type).replace(
        "PTX_TYPE", ptx_type
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
