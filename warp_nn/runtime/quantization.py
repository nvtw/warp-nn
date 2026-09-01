# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Format-neutral, opt-in block quantization for native text runners."""

from __future__ import annotations

import math
from collections.abc import Iterable

import warp as wp

from warp_nn.runtime.formats.gguf import BlockQuantizedTensor
from warp_nn.runtime.kernels import (
    _get_nvfp4_mma_linear_kernel,
    _get_nvfp4_row_scale_kernel,
    _get_quantize_int8_kernel,
    _get_quantize_nvfp4_kernel,
    _repack_gguf_nvfp4_kernel,
)

NVFP4_BLOCK_SIZE = 16
NVFP4_MMA_K = 64
NVFP4_MMA_M = 16
NVFP4_MMA_N = 8

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


def enable_nvfp4_native(device=None):
    """Enable the exact SM120a target for an opt-in NVFP4 model."""
    device = wp.get_device(device)
    if not device.is_cuda or device.arch != 120:
        name = getattr(device, "name", str(device))
        raise RuntimeError(
            f"native NVFP4 inference requires an SM120 GPU; selected device is {name}"
        )
    if wp.config.cuda_arch_suffix not in ("a", "f"):
        wp.config.cuda_arch_suffix = "a"
    return device


def launch_quantize_nvfp4(
    values,
    packed,
    scales,
    global_scales,
    *,
    compute_global_scale: bool = True,
    stream=None,
) -> None:
    """Quantize a dense matrix to adjacent E2M1 pairs and block-16 E4M3 scales."""
    if values.shape[1] % NVFP4_BLOCK_SIZE:
        raise ValueError("NVFP4 inner dimension must be divisible by 16")
    expected_columns = values.shape[1] // 2
    expected_scale_columns = values.shape[1] // NVFP4_BLOCK_SIZE
    padded_rows = packed.shape[0]
    if (
        padded_rows < values.shape[0]
        or packed.shape[1] != expected_columns
        or scales.shape != (padded_rows, expected_scale_columns)
        or global_scales.shape != (padded_rows,)
    ):
        raise ValueError(
            "packed/scales must share a row count at least as large as input and "
            "have K/2 and K/16 columns"
        )
    if compute_global_scale:
        wp.launch_tiled(
            _get_nvfp4_row_scale_kernel(values.dtype),
            dim=values.shape[0],
            inputs=[values, global_scales],
            block_dim=128,
            stream=stream,
        )
    wp.launch(
        _get_quantize_nvfp4_kernel(values.dtype),
        dim=values.size,
        inputs=[values, packed, scales, global_scales],
        block_dim=256,
        stream=stream,
    )


def repack_gguf_nvfp4_weight(weight: BlockQuantizedTensor) -> BlockQuantizedTensor:
    """Repack one GGUF NVFP4 weight once into tensor-core fragment order."""
    if weight.format != "NVFP4":
        raise TypeError("expected an NVFP4 block-quantized tensor")
    if weight.values.ndim != 3 or weight.values.shape[2] != 32:
        raise ValueError("NVFP4 GGUF values must have shape [rows, blocks, 32]")
    output = wp.empty_like(weight.values)
    wp.launch(
        _repack_gguf_nvfp4_kernel,
        dim=weight.values.shape,
        inputs=[weight.values, output],
        device=weight.values.device,
    )
    words = wp.array(
        ptr=output.ptr,
        dtype=wp.uint32,
        shape=(output.shape[0], output.shape[1], 8),
        capacity=output.capacity,
        device=output.device,
        copy=False,
    )
    return BlockQuantizedTensor(output, words, weight.scales, weight.shape, "NVFP4_MMA")


def launch_nvfp4_linear(
    activations,
    activation_scales,
    activation_global_scales,
    weights,
    weight_scales,
    output,
    *,
    global_scale: float = 1.0,
    reuse_weights: bool = False,
    split_k: int = 1,
    stream=None,
) -> None:
    """Launch native NVFP4 output = activations @ weights.T.

    Scales use natural [row, K / 16] order; the MMA intrinsic assembles scale
    registers without a persistent 512-byte cuBLASLt swizzle. Operator planning
    pads short decode batches to M=16 so they reuse this same native kernel.
    """
    enable_nvfp4_native(output.device)
    if split_k not in (1, 2, 4, 8):
        raise ValueError("NVFP4 split_k must be 1, 2, 4, or 8")
    if reuse_weights and split_k != 1:
        raise ValueError("NVFP4 weight reuse and split-K are mutually exclusive")
    rows, columns = output.shape
    if reuse_weights and rows % 64:
        raise ValueError("NVFP4 weight reuse requires M divisible by 64")
    inner = activations.shape[1] * 2
    if rows % NVFP4_MMA_M or columns % NVFP4_MMA_N or inner % NVFP4_MMA_K:
        raise ValueError("native NVFP4 MMA requires M%16 == N%8 == K%64 == 0")
    if split_k > 1:
        output_tiles = (rows // NVFP4_MMA_M) * (columns // NVFP4_MMA_N)
        tiles_per_block = 4 // split_k if split_k <= 4 else 1
        if output_tiles % tiles_per_block:
            raise ValueError("NVFP4 split-K output tiles do not fill a thread block")
    if weights.shape != (columns, inner // 2):
        raise ValueError("NVFP4 weight shape does not match output and inner size")
    expected_activation_scales = (rows, inner // NVFP4_BLOCK_SIZE)
    expected_weight_scales = (columns, inner // NVFP4_BLOCK_SIZE)
    if (
        activation_scales.shape != expected_activation_scales
        or activation_global_scales.shape != (rows,)
        or weight_scales.shape != expected_weight_scales
    ):
        raise ValueError("NVFP4 scale shape does not match its packed matrix")
    wp.launch(
        _get_nvfp4_mma_linear_kernel(
            output.dtype,
            reuse_weights=reuse_weights,
            split_k=split_k if split_k > 1 else 0,
        ),
        dim=((rows // NVFP4_MMA_M) * (columns // NVFP4_MMA_N) * 32 * split_k),
        inputs=[
            activations,
            activation_scales,
            activation_global_scales,
            weights,
            weight_scales,
            output,
            columns,
            inner // NVFP4_MMA_K,
            global_scale,
        ],
        block_dim=max(128, split_k * 32),
        stream=stream,
    )


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
            if metadata.format == "NVFP4":
                largest_source = max(largest_source, metadata.nbytes)
    return final, largest_source


def load_native_weights(archive, device, names: Iterable[str], mode: str | None):
    """Load weights, preparing selected block formats one matrix at a time."""
    mode = normalize_weight_quantization(mode)
    names = tuple(names)
    device = wp.get_device(device)
    nvfp4 = {name for name in names if archive.metadata(name).format == "NVFP4"}
    if nvfp4:
        enable_nvfp4_native(device)
    if mode == "q8_0" and not device.is_cuda:
        raise TypeError("Q8_0 weight quantization requires CUDA")
    q8 = {
        name
        for name in names
        if mode == "q8_0" and is_q8_linear_weight(name, archive.metadata(name))
    }
    selected_set = nvfp4 | q8
    selected = sorted(
        selected_set,
        key=lambda name: archive.metadata(name).nbytes,
        reverse=True,
    )
    output = archive.load(device, [name for name in names if name not in selected_set])
    for name in selected:
        loaded = archive.load(device, [name])
        source = loaded.pop(name)
        output[name] = (
            repack_gguf_nvfp4_weight(source)
            if name in nvfp4
            else quantize_q8_0_weight(source)
        )
        wp.synchronize_stream(device)
        del source, loaded
    return output
