# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small CUDA intrinsic wrappers used by Warp runtime kernels."""

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
