# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small native CUDA matrix primitives used by Warp kernels."""

from functools import lru_cache

import warp as wp


_MMA_16X16 = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int column = (tid >> 5) * 16;
    constexpr int LD = 40;
    constexpr int TILE = 16 * LD;
    __shared__ __align__(16) unsigned short smem[8 * 2 * TILE];
    unsigned short* sa = smem + warp * (2 * TILE);
    unsigned short* sb = sa + TILE;
    const NATIVE_TYPE* xp = x.data;
    const NATIVE_TYPE* weightp = weight.data;
    NATIVE_TYPE* op = output.data;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;

    const int load_row = lane >> 1;
    const int load_col = (lane & 1) * 8;
    for (int k = 0; k < inner; k += 32) {
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            *reinterpret_cast<uint4*>(sa + load_row * LD + part * 16 + load_col) =
                *reinterpret_cast<const uint4*>(xp + load_row * inner + k + part * 16 + load_col);
            *reinterpret_cast<uint4*>(sb + load_row * LD + part * 16 + load_col) =
                *reinterpret_cast<const uint4*>(weightp + (column + load_row) * inner + k + part * 16 + load_col);
        }
        __syncwarp();

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
    }

    const int row = lane >> 2;
    const int col = (lane & 3) * 2;
    op[row * columns + column + col] = NATIVE_TYPE(c0);
    op[row * columns + column + col + 1] = NATIVE_TYPE(c1);
    op[(row + 8) * columns + column + col] = NATIVE_TYPE(c2);
    op[(row + 8) * columns + column + col + 1] = NATIVE_TYPE(c3);
    op[row * columns + column + col + 8] = NATIVE_TYPE(c4);
    op[row * columns + column + col + 9] = NATIVE_TYPE(c5);
    op[(row + 8) * columns + column + col + 8] = NATIVE_TYPE(c6);
    op[(row + 8) * columns + column + col + 9] = NATIVE_TYPE(c7);
#endif
"""


@lru_cache(maxsize=None)
def get_mma_16x16(dtype):
    """Return the SM80 16x16 MMA primitive for FP16 or BF16 storage."""
    if dtype == wp.float16:
        native_type, ptx_type = "wp::float16", "f16"
    elif dtype == wp.bfloat16:
        native_type, ptx_type = "wp::bfloat16", "bf16"
    else:
        raise TypeError("MMA requires FP16 or BF16")
    snippet = _MMA_16X16.replace("NATIVE_TYPE", native_type).replace("PTX_TYPE", ptx_type)

    @wp.func_native(snippet)
    def mma(
        x: wp.array[dtype],
        weight: wp.array[dtype],
        output: wp.array[dtype],
        tid: int,
        columns: int,
        inner: int,
    ): ...

    return mma
