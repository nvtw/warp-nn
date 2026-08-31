# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-free dtype, layout, and gradient bridges for training plans."""

from dataclasses import dataclass
from functools import lru_cache

import warp as wp


__all__ = [
    "accumulate_fp32_gradient",
    "add_fp32_gradients",
    "cast_from_float32",
    "cast_to_float32",
    "merge_heads",
    "split_heads",
]

_STORAGE_DTYPES = (wp.float16, wp.bfloat16)


@dataclass(frozen=True)
class _BridgeKernels:
    to_float32: object
    from_float32: object
    split_heads: object
    merge_heads: object


@lru_cache(maxsize=None)
def _get_bridge_kernels(dtype: type, interleaved: bool = False) -> _BridgeKernels:
    if dtype not in _STORAGE_DTYPES:
        raise TypeError("training bridges support FP16 and BF16 storage")
    DTYPE = dtype
    INTERLEAVED = interleaved

    @wp.kernel(enable_backward=False, module="unique")
    def to_float32(
        source: wp.array1d(dtype=DTYPE), output: wp.array1d(dtype=wp.float32)
    ):
        index = wp.tid()
        output[index] = wp.float32(source[index])

    @wp.kernel(enable_backward=False, module="unique")
    def from_float32(
        source: wp.array1d(dtype=wp.float32), output: wp.array1d(dtype=DTYPE)
    ):
        index = wp.tid()
        output[index] = DTYPE(source[index])

    @wp.kernel(enable_backward=False, module="unique")
    def split_heads(packed: wp.array2d(dtype=DTYPE), heads: wp.array4d(dtype=DTYPE)):
        batch, head, token, column = wp.tid()
        packed_row = batch * heads.shape[2] + token
        source_column = column
        if INTERLEAVED:
            half = heads.shape[3] // 2
            source_column = (column % half) * 2 + column // half
        packed_column = head * heads.shape[3] + source_column
        heads[batch, head, token, column] = packed[packed_row, packed_column]

    @wp.kernel(enable_backward=False, module="unique")
    def merge_heads(heads: wp.array4d(dtype=DTYPE), packed: wp.array2d(dtype=DTYPE)):
        packed_row, packed_column = wp.tid()
        token = packed_row % heads.shape[2]
        batch = packed_row // heads.shape[2]
        column = packed_column % heads.shape[3]
        head = packed_column // heads.shape[3]
        source_column = column
        if INTERLEAVED:
            half = heads.shape[3] // 2
            source_column = column // 2 + (column % 2) * half
        packed[packed_row, packed_column] = heads[batch, head, token, source_column]

    return _BridgeKernels(to_float32, from_float32, split_heads, merge_heads)


@wp.kernel(enable_backward=False)
def _add_fp32_gradients(
    left: wp.array1d(dtype=wp.float32),
    right: wp.array1d(dtype=wp.float32),
    output: wp.array1d(dtype=wp.float32),
):
    index = wp.tid()
    output[index] = left[index] + right[index]


@wp.kernel(enable_backward=False)
def _accumulate_fp32_gradient(
    source: wp.array1d(dtype=wp.float32),
    destination: wp.array1d(dtype=wp.float32),
):
    index = wp.tid()
    destination[index] += source[index]


def _check_array(name: str, value: wp.array, *, ndim: int | None = None) -> None:
    if not isinstance(value, wp.array):
        raise TypeError(f"{name} must be a Warp array")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {value.ndim}-D")
    if not value.is_contiguous:
        raise ValueError(f"{name} must be contiguous")


def _check_pair(source: wp.array, output: wp.array) -> None:
    if source.shape != output.shape:
        raise ValueError(
            f"source and output shapes must match, got {source.shape} and {output.shape}"
        )
    if source.device != output.device:
        raise ValueError("source and output must be on the same device")


def cast_to_float32(source: wp.array, output: wp.array) -> None:
    """Cast contiguous FP16/BF16 ``source`` into caller-owned FP32 ``output``."""
    _check_array("source", source)
    _check_array("output", output)
    if source.dtype not in _STORAGE_DTYPES or output.dtype != wp.float32:
        raise TypeError("cast_to_float32 requires FP16/BF16 source and FP32 output")
    _check_pair(source, output)
    wp.launch(
        _get_bridge_kernels(source.dtype).to_float32,
        dim=source.size,
        inputs=[source.flatten()],
        outputs=[output.flatten()],
        device=source.device,
    )


def cast_from_float32(source: wp.array, output: wp.array) -> None:
    """Cast contiguous FP32 ``source`` into caller-owned FP16/BF16 ``output``."""
    _check_array("source", source)
    _check_array("output", output)
    if source.dtype != wp.float32 or output.dtype not in _STORAGE_DTYPES:
        raise TypeError("cast_from_float32 requires FP32 source and FP16/BF16 output")
    _check_pair(source, output)
    wp.launch(
        _get_bridge_kernels(output.dtype).from_float32,
        dim=source.size,
        inputs=[source.flatten()],
        outputs=[output.flatten()],
        device=source.device,
    )


def split_heads(
    packed: wp.array, heads: wp.array, *, interleaved: bool = False
) -> None:
    """Permute packed ``[B*S, H*D]`` storage into GQA ``[B, H, S, D]``."""
    _check_array("packed", packed, ndim=2)
    _check_array("heads", heads, ndim=4)
    if packed.dtype not in _STORAGE_DTYPES or heads.dtype != packed.dtype:
        raise TypeError("packed and heads must share an FP16 or BF16 dtype")
    if packed.device != heads.device:
        raise ValueError("packed and heads must be on the same device")
    batch, head_count, sequence, head_size = heads.shape
    expected = (batch * sequence, head_count * head_size)
    if packed.shape != expected:
        raise ValueError(f"packed must have shape {expected}, got {packed.shape}")
    if interleaved and head_size % 2:
        raise ValueError("interleaved head storage requires an even head size")
    wp.launch(
        _get_bridge_kernels(packed.dtype, bool(interleaved)).split_heads,
        dim=heads.shape,
        inputs=[packed],
        outputs=[heads],
        device=packed.device,
    )


def merge_heads(
    heads: wp.array, packed: wp.array, *, interleaved: bool = False
) -> None:
    """Inverse-permute GQA ``[B, H, S, D]`` into packed ``[B*S, H*D]``."""
    _check_array("heads", heads, ndim=4)
    _check_array("packed", packed, ndim=2)
    if heads.dtype not in _STORAGE_DTYPES or packed.dtype != heads.dtype:
        raise TypeError("heads and packed must share an FP16 or BF16 dtype")
    if heads.device != packed.device:
        raise ValueError("heads and packed must be on the same device")
    batch, head_count, sequence, head_size = heads.shape
    expected = (batch * sequence, head_count * head_size)
    if packed.shape != expected:
        raise ValueError(f"packed must have shape {expected}, got {packed.shape}")
    if interleaved and head_size % 2:
        raise ValueError("interleaved head storage requires an even head size")
    wp.launch(
        _get_bridge_kernels(heads.dtype, bool(interleaved)).merge_heads,
        dim=packed.shape,
        inputs=[heads],
        outputs=[packed],
        device=heads.device,
    )


def add_fp32_gradients(left: wp.array, right: wp.array, output: wp.array) -> None:
    """Write the elementwise sum of two contiguous FP32 gradients."""
    for name, value in (("left", left), ("right", right), ("output", output)):
        _check_array(name, value)
        if value.dtype != wp.float32:
            raise TypeError(f"{name} must use FP32 storage")
    if left.shape != right.shape or left.shape != output.shape:
        raise ValueError("left, right, and output shapes must match")
    if left.device != right.device or left.device != output.device:
        raise ValueError("left, right, and output must be on the same device")
    wp.launch(
        _add_fp32_gradients,
        dim=left.size,
        inputs=[left.flatten(), right.flatten()],
        outputs=[output.flatten()],
        device=left.device,
    )


def accumulate_fp32_gradient(source: wp.array, destination: wp.array) -> None:
    """Add one contiguous FP32 gradient into a fixed destination in place."""
    _check_array("source", source)
    _check_array("destination", destination)
    if source.dtype != wp.float32 or destination.dtype != wp.float32:
        raise TypeError("source and destination must use FP32 storage")
    _check_pair(source, destination)
    wp.launch(
        _accumulate_fp32_gradient,
        dim=source.size,
        inputs=[source.flatten(), destination.flatten()],
        device=source.device,
    )
