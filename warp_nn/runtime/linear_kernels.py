# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimized FP16/BF16 linear kernels for decode and 16-row prefill."""

from functools import lru_cache

import warp as wp

from warp_nn.runtime._cuda import subgroup_sum


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
    for (int item = threadIdx.x; item < 320; item += 128) {
        bool is_a = item < 64;
        int local = is_a ? item : item - 64;
        int row = local >> 2;
        int segment = local & 3;
        unsigned short* dst = smem + (is_a ? row * LD : A_SIZE + row * LD) + segment * 8;
        const NATIVE_TYPE* src = is_a ? xp + row * inner + k_begin + segment * 8
                                      : weightp + (column + row) * inner + k_begin + segment * 8;
        unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        if (is_a)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
        else
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    for (int k = k_begin, stage = 0; k < k_end; k += 32, stage ^= 1) {
        if (k + 32 < k_end) {
            unsigned short* next = smem + (stage ^ 1) * STAGE_SIZE;
            for (int item = threadIdx.x; item < 320; item += 128) {
                bool is_a = item < 64;
                int local = is_a ? item : item - 64;
                int row = local >> 2;
                int segment = local & 3;
                unsigned short* dst = next + (is_a ? row * LD : A_SIZE + row * LD) + segment * 8;
                const NATIVE_TYPE* src = is_a ? xp + row * inner + k + 32 + segment * 8
                                              : weightp + (column + row) * inner + k + 32 + segment * 8;
                unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                if (is_a)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
                else
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
    """Build the vec4 single-token projection with four outputs per warp."""
    DTYPE = dtype
    VTYPE = wp.vec4h if dtype == wp.float16 else wp.types.vector(4, wp.bfloat16)
    SUBGROUP = 32

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        x: wp.array2d(dtype=VTYPE),
        weight: wp.array2d(dtype=VTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        thread = wp.tid()
        lane = thread % SUBGROUP
        column = (thread / SUBGROUP) * 4
        total0 = wp.float32(0.0)
        total1 = wp.float32(0.0)
        total2 = wp.float32(0.0)
        total3 = wp.float32(0.0)
        for inner in range(lane, x.shape[1], SUBGROUP):
            activation = VTYPE(x[0, inner])
            weights0 = VTYPE(weight[column, inner])
            weights1 = VTYPE(weight[column + 1, inner])
            weights2 = VTYPE(weight[column + 2, inner])
            weights3 = VTYPE(weight[column + 3, inner])
            for component in range(4):
                value = wp.float32(activation[component])
                total0 += value * wp.float32(weights0[component])
                total1 += value * wp.float32(weights1[component])
                total2 += value * wp.float32(weights2[component])
                total3 += value * wp.float32(weights3[component])
        total0 = subgroup_sum(total0, SUBGROUP)
        total1 = subgroup_sum(total1, SUBGROUP)
        total2 = subgroup_sum(total2, SUBGROUP)
        total3 = subgroup_sum(total3, SUBGROUP)
        if lane == 0:
            output[0, column] = DTYPE(total0)
            output[0, column + 1] = DTYPE(total1)
            output[0, column + 2] = DTYPE(total2)
            output[0, column + 3] = DTYPE(total3)

    kernel.module.options["enable_backward"] = False
    return kernel, VTYPE


@lru_cache(maxsize=None)
def get_decode_linear_kernel(dtype: type):
    """Return the cached single-token projection kernel and packed dtype."""
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
