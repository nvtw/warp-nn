# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Format-neutral, opt-in block quantization for native text runners."""

from __future__ import annotations

import math
from collections.abc import Iterable

import warp as wp

from warp_nn.runtime.gguf import BlockQuantizedTensor
from warp_nn.runtime.kernels import _get_quantize_int8_kernel

_PROJECTION_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.gate_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.out_proj.weight",
)


def normalize_weight_quantization(value: str | None) -> str | None:
    """Validate an optional native-runner weight quantization mode."""
    if value is None:
        return None
    normalized = value.lower()
    if normalized != "q8_0":
        raise ValueError("weight_quantization must be None or 'q8_0'")
    return normalized


def is_q8_linear_weight(name: str, metadata) -> bool:
    """Select ordinary two-dimensional projection weights, not model state."""
    shape = tuple(metadata.shape)
    return (
        metadata.format in ("F16", "BF16")
        and len(shape) == 2
        and shape[-1] % 32 == 0
        and name.endswith(_PROJECTION_SUFFIXES)
    )


def q8_storage_nbytes(shape: tuple[int, ...]) -> int:
    """Return contiguous Q8 values plus FP16 block-scale storage."""
    elements = math.prod(shape)
    if not shape or shape[-1] % 32:
        raise ValueError("Q8_0 requires an inner width divisible by 32")
    return elements + elements // 32 * 2


def quantize_q8_0_weight(weight: wp.array) -> BlockQuantizedTensor:
    """Quantize one contiguous FP16/BF16 matrix into shared Q8 runtime storage."""
    if not weight.device.is_cuda:
        raise TypeError("Q8_0 weight quantization requires CUDA")
    if weight.dtype not in (wp.float16, wp.bfloat16) or weight.ndim != 2:
        raise TypeError("Q8_0 weight quantization requires an FP16/BF16 matrix")
    rows, inner = weight.shape
    if inner % 32:
        raise ValueError("Q8_0 weight inner width must be divisible by 32")
    blocks = inner // 32
    values_2d = wp.empty((rows, inner), dtype=wp.int8, device=weight.device)
    scales = wp.empty((rows, blocks), dtype=wp.float16, device=weight.device)
    wp.launch(
        _get_quantize_int8_kernel(weight.dtype, wp.float16, False),
        dim=rows * inner,
        inputs=[weight, values_2d, scales],
        block_dim=128,
        device=weight.device,
    )
    values = values_2d.reshape((rows, blocks, 32))
    words = wp.array(
        ptr=values.ptr,
        dtype=wp.uint32,
        shape=(rows, blocks, 8),
        capacity=values.capacity,
        device=weight.device,
        copy=False,
    )
    return BlockQuantizedTensor(values, words, scales, tuple(weight.shape), "Q8_0")


def estimate_loaded_weight_bytes(archive, names: Iterable[str], mode: str | None):
    """Return final bytes and largest transient source for a load policy."""
    mode = normalize_weight_quantization(mode)
    final = 0
    largest_source = 0
    for name in names:
        metadata = archive.metadata(name)
        if mode == "q8_0" and is_q8_linear_weight(name, metadata):
            final += q8_storage_nbytes(tuple(metadata.shape))
            largest_source = max(largest_source, metadata.nbytes)
        else:
            final += metadata.nbytes
    return final, largest_source


def load_native_weights(archive, device, names: Iterable[str], mode: str | None):
    """Load native weights, quantizing selected matrices one at a time."""
    mode = normalize_weight_quantization(mode)
    names = tuple(names)
    if mode is None:
        return archive.load(device, names)
    device = wp.get_device(device)
    if not device.is_cuda:
        raise TypeError("Q8_0 weight quantization requires CUDA")
    selected = sorted(
        (name for name in names if is_q8_linear_weight(name, archive.metadata(name))),
        key=lambda name: archive.metadata(name).nbytes,
        reverse=True,
    )
    selected_set = set(selected)
    output = archive.load(device, [name for name in names if name not in selected_set])
    for name in selected:
        loaded = archive.load(device, [name])
        source = loaded.pop(name)
        output[name] = quantize_q8_0_weight(source)
        wp.synchronize_stream(device)
        del source, loaded
    return output
