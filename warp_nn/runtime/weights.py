# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Small format-neutral helpers for streaming inference-weight conversion."""

from functools import lru_cache
import math

import warp as wp


@lru_cache(maxsize=None)
def _cast_kernel(source_dtype, target_dtype):
    SOURCE = source_dtype
    TARGET = target_dtype

    @wp.kernel(enable_backward=False, module="unique")
    def cast(source: wp.array(dtype=SOURCE), output: wp.array(dtype=TARGET)):
        index = wp.tid()
        output[index] = TARGET(source[index])

    return cast


@lru_cache(maxsize=None)
def _merge_lora_kernel(dtype):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def merge_lora(
        weight: wp.array2d(dtype=DTYPE),
        a: wp.array2d(dtype=DTYPE),
        b: wp.array2d(dtype=DTYPE),
        scale: wp.float32,
    ):
        row, column = wp.tid()
        delta = wp.float32(0.0)
        for rank in range(a.shape[0]):
            delta += wp.float32(b[row, rank]) * wp.float32(a[rank, column])
        weight[row, column] = DTYPE(wp.float32(weight[row, column]) + scale * delta)

    return merge_lora


def merge_lora_weight(weight, a, b, scale):
    """Merge one low-rank update into a caller-owned inference weight."""
    if (
        weight.ndim != 2
        or a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != weight.shape[1]
        or b.shape != (weight.shape[0], a.shape[0])
    ):
        raise ValueError("LoRA A and B shapes do not match the base weight")
    if a.dtype != weight.dtype or b.dtype != weight.dtype:
        raise TypeError("LoRA and base weight dtypes must match")
    if a.device != weight.device or b.device != weight.device:
        raise ValueError("LoRA and base weight devices must match")
    if not math.isfinite(scale):
        raise ValueError("LoRA scale must be finite")
    wp.launch(
        _merge_lora_kernel(weight.dtype),
        dim=weight.shape,
        inputs=[weight, a, b, wp.float32(scale)],
        device=weight.device,
    )


def load_cast_weights(archive, names, device, dtype=None):
    """Load selected tensors one at a time and optionally cast floating weights.

    The sequential lifecycle bounds peak memory to final weights plus one source
    tensor.  It works with every archive exposing the shared ``metadata/load``
    interface and never creates a CPU numeric conversion copy.
    """
    device = wp.get_device(device)
    names = tuple(dict.fromkeys(names))
    if dtype is None:
        return archive.load(device, names)
    if dtype not in (wp.float16, wp.bfloat16, wp.float32):
        raise TypeError("weight conversion target must be FP16, BF16, or FP32")
    output = {}
    for name in sorted(
        names, key=lambda item: archive.metadata(item).nbytes, reverse=True
    ):
        source = archive.load(device, [name])[name]
        if source.dtype == dtype:
            output[name] = source
            continue
        if source.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError(f"cannot cast non-floating weight '{name}'")
        converted = wp.empty(source.shape, dtype=dtype, device=device)
        wp.launch(
            _cast_kernel(source.dtype, dtype),
            dim=source.size,
            inputs=[source.flatten(), converted.flatten()],
            device=device,
        )
        if device.is_cuda:
            wp.synchronize_stream(wp.get_stream(device))
        output[name] = converted
    return output
