# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimized FP16/BF16 linear kernels for decode and 16-row prefill."""

from functools import lru_cache

import warp as wp


_DECODE_LINEAR = r"""
#if defined(__CUDA_ARCH__)
    const int lane = tid & 31;
    constexpr int OUTPUTS = 8;
    const int column = (tid >> 5) * OUTPUTS;
    const NATIVE_TYPE* xp = x.data;
    const NATIVE_TYPE* wp = weight.data;
    float totals[OUTPUTS] = {};

    for (int k = lane * 8; k < inner; k += 256) {
        uint4 av = *reinterpret_cast<const uint4*>(xp + k);
        uint4 weights[OUTPUTS];
        #pragma unroll
        for (int output = 0; output < OUTPUTS; ++output) {
            #if NATIVE_BF16
            weights[output] = __ldcs(reinterpret_cast<const uint4*>(wp + (column + output) * inner + k));
            #else
            weights[output] = *reinterpret_cast<const uint4*>(wp + (column + output) * inner + k);
            #endif
        }
        #if NATIVE_BF16
        const unsigned* a = reinterpret_cast<const unsigned*>(&av);
        #pragma unroll
        for (int word = 0; word < 4; ++word) {
            float value = __uint_as_float(a[word] << 16);
            #pragma unroll
            for (int output = 0; output < OUTPUTS; ++output) {
                unsigned packed = reinterpret_cast<const unsigned*>(&weights[output])[word];
                totals[output] = fmaf(value, __uint_as_float(packed << 16), totals[output]);
            }
            value = __uint_as_float(a[word] & 0xffff0000u);
            #pragma unroll
            for (int output = 0; output < OUTPUTS; ++output) {
                unsigned packed = reinterpret_cast<const unsigned*>(&weights[output])[word];
                totals[output] = fmaf(value, __uint_as_float(packed & 0xffff0000u), totals[output]);
            }
        }
        #else
        const NATIVE_TYPE* a = reinterpret_cast<const NATIVE_TYPE*>(&av);
        #pragma unroll
        for (int component = 0; component < 8; ++component) {
            float value = float(a[component]);
            #pragma unroll
            for (int output = 0; output < OUTPUTS; ++output) {
                const NATIVE_TYPE* values = reinterpret_cast<const NATIVE_TYPE*>(&weights[output]);
                totals[output] = fmaf(value, float(values[component]), totals[output]);
            }
        }
        #endif
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        #pragma unroll
        for (int output = 0; output < OUTPUTS; ++output)
            totals[output] += __shfl_down_sync(0xffffffffu, totals[output], offset);
    }
    if (lane == 0) {
        uint4 packed;
        NATIVE_TYPE* values = reinterpret_cast<NATIVE_TYPE*>(&packed);
        #pragma unroll
        for (int index = 0; index < OUTPUTS; ++index)
            values[index] = NATIVE_TYPE(totals[index]);
        *reinterpret_cast<uint4*>(output.data + column) = packed;
    }
#endif
"""


_MMA_16X64 = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int block = tid >> 7;
    const int split = block % splits;
    const int column = (block / splits) * 64;
    constexpr int LD = 40;
    constexpr int A_SIZE = 16 * LD;
    constexpr int B_SIZE = 64 * LD;
    constexpr int STAGE_SIZE = A_SIZE + B_SIZE;
    __shared__ __align__(16) unsigned short smem[2 * STAGE_SIZE];
    const NATIVE_TYPE* xp = x.data;
    const NATIVE_TYPE* weightp = weight.data;
    float* op = output.data;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;

    const int split_inner = inner / splits;
    const int k_begin = split * split_inner;
    const int k_end = k_begin + split_inner;
    if (threadIdx.x < 64) {
        int local = threadIdx.x;
        int row = local >> 2;
        int segment = local & 3;
        unsigned short* dst = smem + row * LD + segment * 8;
        const NATIVE_TYPE* src = xp + row * inner + k_begin + segment * 8;
        unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    }
    #pragma unroll
    for (int copy = 0; copy < 2; ++copy) {
        int local = threadIdx.x + copy * 128;
        int row = local >> 2;
        int segment = local & 3;
        unsigned short* dst = smem + A_SIZE + row * LD + segment * 8;
        const NATIVE_TYPE* src = weightp + (column + row) * inner + k_begin + segment * 8;
        unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    for (int k = k_begin, stage = 0; k < k_end; k += 32, stage ^= 1) {
        if (k + 32 < k_end) {
            unsigned short* next = smem + (stage ^ 1) * STAGE_SIZE;
            if (threadIdx.x < 64) {
                int local = threadIdx.x;
                int row = local >> 2;
                int segment = local & 3;
                unsigned short* dst = next + row * LD + segment * 8;
                const NATIVE_TYPE* src = xp + row * inner + k + 32 + segment * 8;
                unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
            }
            #pragma unroll
            for (int copy = 0; copy < 2; ++copy) {
                int local = threadIdx.x + copy * 128;
                int row = local >> 2;
                int segment = local & 3;
                unsigned short* dst = next + A_SIZE + row * LD + segment * 8;
                const NATIVE_TYPE* src = weightp + (column + row) * inner + k + 32 + segment * 8;
                unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
            }
            asm volatile("cp.async.commit_group;");
        }

        unsigned short* current = smem + stage * STAGE_SIZE;
        unsigned short* sa = current;
        unsigned short* sb = current + A_SIZE + warp * 16 * LD;
        const int q = lane >> 3;
        const int r = lane & 7;
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            unsigned a0, a1, a2, a3, b0, b1, b2, b3;
            unsigned pa = static_cast<unsigned>(__cvta_generic_to_shared(
                sa + (r + ((q & 1) * 8)) * LD + part * 16 + ((q >> 1) * 8)));
            unsigned pb = static_cast<unsigned>(__cvta_generic_to_shared(
                sb + (r + ((q >> 1) * 8)) * LD + part * 16 + ((q & 1) * 8)));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(pa) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(pb) : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.PTX_TYPE.PTX_TYPE.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.PTX_TYPE.PTX_TYPE.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));
        }
        if (k + 32 < k_end) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
    }

    const int row = lane >> 2;
    const int col = column + warp * 16 + (lane & 3) * 2;
    op += split * 16 * columns;
    op[row * columns + col] = c0;
    op[row * columns + col + 1] = c1;
    op[(row + 8) * columns + col] = c2;
    op[(row + 8) * columns + col + 1] = c3;
    op[row * columns + col + 8] = c4;
    op[row * columns + col + 9] = c5;
    op[(row + 8) * columns + col + 8] = c6;
    op[(row + 8) * columns + col + 9] = c7;
#endif
"""


@lru_cache(maxsize=None)
def _get_mma_16x64(dtype):
    """Return the SM80 16x64 MMA primitive for FP16 or BF16 storage."""
    if dtype == wp.float16:
        native_type, ptx_type = "wp::float16", "f16"
    elif dtype == wp.bfloat16:
        native_type, ptx_type = "wp::bfloat16", "bf16"
    else:
        raise TypeError("MMA requires FP16 or BF16")
    snippet = _MMA_16X64.replace("NATIVE_TYPE", native_type).replace("PTX_TYPE", ptx_type)

    @wp.func_native(snippet)
    def mma(
        x: wp.array[dtype],
        weight: wp.array[dtype],
        output: wp.array[wp.float32],
        tid: int,
        columns: int,
        inner: int,
        splits: int,
    ): ...

    return mma


def _create_decode_linear_kernel(dtype: type):
    """Build the vec8 single-token projection with eight outputs per warp."""
    DTYPE = dtype
    if dtype == wp.float16:
        native_type, native_bf16 = "wp::float16", "0"
    elif dtype == wp.bfloat16:
        native_type, native_bf16 = "wp::bfloat16", "1"
    else:
        raise TypeError("Decode projection requires FP16 or BF16")
    snippet = _DECODE_LINEAR.replace("NATIVE_TYPE", native_type).replace("NATIVE_BF16", native_bf16)

    @wp.func_native(snippet)
    def project(
        x: wp.array[DTYPE],
        weight: wp.array[DTYPE],
        output: wp.array[DTYPE],
        tid: int,
        columns: int,
        inner: int,
    ): ...

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        x: wp.array(dtype=DTYPE),
        weight: wp.array(dtype=DTYPE),
        output: wp.array(dtype=DTYPE),
        columns: int,
        inner: int,
    ):
        typed_zero = DTYPE(0.0)
        wp.static(project)(x, weight, output, wp.tid(), columns, inner)

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def get_decode_linear_kernel(dtype: type):
    """Return the cached single-token projection kernel."""
    return _create_decode_linear_kernel(dtype)


def _create_prefill_linear_kernels(dtype: type):
    """Build the adaptive split-K 16-row projection and epilogue."""
    DTYPE = dtype
    mma = _get_mma_16x64(dtype)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def project(
        x: wp.array(dtype=DTYPE),
        weight: wp.array(dtype=DTYPE),
        output: wp.array(dtype=wp.float32),
        columns: int,
        inner: int,
        splits: int,
    ):
        typed_zero = DTYPE(0.0)
        wp.static(mma)(x, weight, output, wp.tid(), columns, inner, splits)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def combine(partials: wp.array(dtype=wp.float32), output: wp.array(dtype=DTYPE), output_size: int, splits: int):
        index = wp.tid()
        value = wp.float32(0.0)
        for split in range(splits):
            value += partials[split * output_size + index]
        output[index] = DTYPE(value)

    project.module.options["enable_backward"] = False
    combine.module.options["enable_backward"] = False
    return project, combine


@lru_cache(maxsize=None)
def get_prefill_linear_kernels(dtype: type):
    """Return the cached 16-row projection and private epilogue kernels."""
    return _create_prefill_linear_kernels(dtype)
