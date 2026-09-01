# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Format-independent operator execution for preplanned inference graphs."""

from __future__ import annotations

from typing import Any, Iterable
from collections.abc import Mapping

from dataclasses import dataclass, field
import math

import numpy as np
import warp as wp

from warp_nn.runtime.formats.gguf import BlockQuantizedTensor, PackedQuantizedTensor

from warp_nn.runtime.kernels import (
    _adaptive_layer_norm_kernel,
    _bias_activation_kernel,
    _broadcast_gated_residual_kernel,
    _concatenate_attention_streams_kernel,
    _concatenate_validity_kernel,
    _get_layer_norm_kernel,
    _merge_attention_heads_kernel,
    _rotary_cache_kernel,
    _sequence_slice_kernel,
    _seeded_normal_kernel,
    _sinusoidal_embedding_kernel,
    _split_attention_heads_kernel,
    _split_attention_streams_kernel,
    _GEMM_CONFIG,
    _GEMM_TRANSB_TILED_KERNEL,
    _gather_block_quantized_int8_kernel,
    _get_grouped_decode_linear_kernel,
    _get_small_batch_grouped_linear_kernel,
    _get_bidirectional_gqa_attention_kernel,
    _get_linear_tiled_kernel,
    _get_prefill_mma_linear_kernel,
    _get_partitioned_gqa_attention_kernels,
    _get_linear_vector_kernel,
    _get_nvfp4_mma_linear_kernel,
    _get_nvfp4_row_scale_kernel,
    _get_q8_grouped_decode_linear_kernel,
    _get_q8_prefill_mma_linear_kernel,
    _get_q3_k_linear_kernel,
    _get_matmul_int8_q8_kernel,
    _get_quantize_activation_int8_kernel,
    _get_quantize_nvfp4_kernel,
    _get_rms_norm_kernels,
    _get_swiglu_kernel,
    _gqa_copy_past_fp16_kernel,
    _gqa_prepare_fp16_kernel,
    _linear_kernel,
    _channels_last_1d_kernels,
    _clamp_kernel_for_dtype,
    _overlap_tile_blend_kernel,
    _channels_last_2d_kernels,
    _conv1d_mma_kernels,
    _conv2d_mma_kernels,
    _conv_transpose1d_mma_kernels,
    _adaptive_rms_modulation_kernel,
    _encoder_kernels,
    _modulated_residual_kernel,
    _quantize_activation_int8_kernel,
    _spatial_diffusion_kernels,
    _true_cfg_kernel,
    _spatial_vae_kernels,
)
from warp_nn.runtime.quantization import enable_nvfp4_native, repack_gguf_nvfp4_weight
from warp_nn.utils.ops import resolve_dim


def _decode_attention_partitions(head_size: int) -> int:
    """Choose bounded decode parallelism from attention-head geometry."""
    return max(64, min(256, int(head_size)))


def _decode_attention_head_group(
    query_heads: int, kv_heads: int, head_size: int
) -> int:
    """Choose bounded K/V reuse from the grouped-query ratio."""
    if kv_heads <= 0 or query_heads < kv_heads or query_heads % kv_heads:
        raise ValueError("query_heads must be a positive multiple of kv_heads")
    if head_size > 128:
        return 4
    queries_per_kv = query_heads // kv_heads
    return max(4, min(16, 1 << (queries_per_kv.bit_length() - 1)))


def _attention_group_geometry(
    query_heads: int, kv_heads: int | None, head_size: int, rows: int
) -> tuple[int, int]:
    """Choose a bounded query tile without introducing partial head groups."""
    heads_per_group = 4
    if kv_heads and (rows == 1 or rows >= 16):
        candidate = _decode_attention_head_group(query_heads, kv_heads, head_size)
        queries_per_kv = query_heads // kv_heads
        if rows == 1 or queries_per_kv % candidate == 0:
            heads_per_group = candidate
    rows_per_group = (
        1 if rows < 16 else max(1, min(4, 2048 // head_size // heads_per_group))
    )
    return rows_per_group, heads_per_group


def _allocate_partitioned_gqa(
    heads: int,
    head_size: int,
    dtype: type,
    device,
    partitions: int = 256,
    rows: int = 1,
    rows_per_group: int | None = None,
    heads_per_group: int | None = None,
    kv_heads: int | None = None,
    mapped: bool = False,
):
    """Allocate one reusable workspace for partitioned decode attention."""
    default_rows, default_heads = _attention_group_geometry(
        heads, kv_heads, head_size, rows
    )
    if rows_per_group is None:
        rows_per_group = default_rows
    if heads_per_group is None:
        heads_per_group = default_heads
    block_dim, partitions, kernels = _get_partitioned_gqa_attention_kernels(
        head_size, dtype, partitions, rows_per_group, heads_per_group, mapped
    )
    items = rows * heads * partitions
    return (
        block_dim,
        partitions,
        rows_per_group,
        heads_per_group,
        kernels,
        wp.empty(items, dtype=wp.float32, device=device),
        wp.empty(items, dtype=wp.float32, device=device),
        wp.empty((items, head_size), dtype=wp.float32, device=device),
    )


def _launch_partitioned_gqa(
    workspace,
    query,
    key,
    value,
    sequence_end,
    output,
    query_heads: int,
    kv_heads: int,
    total_length: int,
    scale: float,
    window: int,
    device,
    sequence_length: int | None = None,
    slot_indices=None,
):
    """Launch partitioned decode attention using a reusable workspace."""
    (
        block_dim,
        partitions,
        rows_per_group,
        heads_per_group,
        kernels,
        partial_maximum,
        partial_denominator,
        partial_output,
    ) = workspace
    queries_per_kv = query_heads // kv_heads
    groups_per_batch = kv_heads * (
        (queries_per_kv + heads_per_group - 1) // heads_per_group
    )
    sequence_length = output.shape[0] if sequence_length is None else sequence_length
    row_groups = (sequence_length + rows_per_group - 1) // rows_per_group
    batches = query.shape[0] // (query_heads * sequence_length)
    wp.launch_tiled(
        kernels[0],
        dim=batches * row_groups * groups_per_batch * partitions,
        inputs=[
            query,
            key,
            value,
            sequence_end,
            sequence_end if slot_indices is None else slot_indices,
            partial_maximum,
            partial_denominator,
            partial_output,
            query_heads,
            kv_heads,
            sequence_length,
            total_length,
            scale,
            window,
        ],
        block_dim=block_dim,
        device=device,
    )
    wp.launch_tiled(
        kernels[1],
        dim=batches * sequence_length * query_heads,
        inputs=[
            partial_maximum,
            partial_denominator,
            partial_output,
            output,
            query_heads,
        ],
        block_dim=block_dim,
        device=device,
    )


@dataclass
class Operation:
    """A preplanned operation whose private attributes hold launch state."""

    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


def execute_operations(
    operations: Iterable[Operation],
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device,
) -> None:
    """Launch a preplanned operation sequence on the current Warp stream."""
    for operation in operations:
        try:
            dispatch = _OP_DISPATCH[operation.op_type]
        except KeyError as exc:
            raise NotImplementedError(
                f"Unsupported operation '{operation.op_type}'"
            ) from exc
        dispatch(operation, tensors, shapes, device)


def _exec_linear(op, tensors, shapes, device):
    x = tensors[op.inputs[0]].reshape((op.attrs["_rows"], op.attrs["_inner"]))
    weight = tensors[op.inputs[1]]
    output = tensors[op.outputs[0]].reshape((op.attrs["_rows"], op.attrs["_columns"]))
    if "_q3_k_kernel" in op.attrs:
        wp.launch(
            op.attrs["_q3_k_kernel"],
            dim=op.attrs["_rows"] * op.attrs["_columns"] * 32,
            inputs=[x, weight.blocks, output, op.attrs["_inner"] // 256],
            block_dim=128,
            device=device,
        )
        return
    if "_nvfp4_activations" in op.attrs:
        quantize_kernel = op.attrs.get("_nvfp4_quantize_kernel")
        if quantize_kernel is not None:
            wp.launch_tiled(
                op.attrs["_nvfp4_row_scale_kernel"],
                dim=op.attrs["_rows"],
                inputs=[x, op.attrs["_nvfp4_global_scales"]],
                block_dim=128,
                device=device,
            )
            wp.launch(
                quantize_kernel,
                dim=op.attrs["_rows"] * op.attrs["_inner"],
                inputs=[
                    x,
                    op.attrs["_nvfp4_activations"],
                    op.attrs["_nvfp4_scales"],
                    op.attrs["_nvfp4_global_scales"],
                ],
                block_dim=256,
                device=device,
            )
        wp.launch(
            op.attrs["_nvfp4_mma_kernel"],
            dim=(
                op.attrs["_nvfp4_padded_rows"]
                // 16
                * (op.attrs["_columns"] // 8)
                * 32
                * op.attrs.get("_nvfp4_grid_multiplier", 1)
            ),
            inputs=[
                op.attrs["_nvfp4_activations"],
                op.attrs["_nvfp4_scales"],
                op.attrs["_nvfp4_global_scales"],
                weight.values.reshape((op.attrs["_columns"], op.attrs["_inner"] // 2)),
                weight.scales.reshape((op.attrs["_columns"], op.attrs["_inner"] // 16)),
                op.attrs["_nvfp4_output"],
                op.attrs["_columns"],
                op.attrs["_inner"] // 64,
                float(op.attrs.get("_output_scale", 1.0)),
            ],
            block_dim=op.attrs.get("_nvfp4_block_dim", 128),
            device=device,
        )
        return
    if "_q8_activations" in op.attrs:
        quantize_kernel = op.attrs.get("_q8_quantize_kernel")
        if quantize_kernel is not None:
            wp.launch(
                quantize_kernel,
                dim=op.attrs["_rows"] * op.attrs["_inner"],
                inputs=[x, op.attrs["_q8_activations"], op.attrs["_q8_scales"]],
                block_dim=32,
                device=device,
            )
        q8_decode = op.attrs.get("_q8_grouped_decode_kernel")
        if q8_decode is not None:
            wp.launch(
                q8_decode,
                dim=(op.attrs["_columns"] // op.attrs["_q8_decode_outputs_per_group"])
                * 8,
                inputs=[
                    op.attrs["_q8_activation_words"],
                    op.attrs["_q8_scales"],
                    weight.words,
                    weight.scales,
                    output,
                    op.attrs["_inner"] // 32,
                ],
                block_dim=128,
                device=device,
            )
            return
        q8_mma = op.attrs.get("_q8_mma_kernel")
        if q8_mma is not None:
            wp.launch(
                q8_mma,
                dim=(
                    op.attrs["_rows"]
                    // op.attrs["_q8_mma_tile_m"]
                    * (op.attrs["_columns"] // 32)
                    * op.attrs["_q8_mma_tile_m"]
                    * 8
                ),
                inputs=[
                    op.attrs["_q8_activations"],
                    op.attrs["_q8_scales"],
                    weight.values,
                    weight.scales,
                    output,
                    op.attrs["_columns"],
                    op.attrs["_inner"] // 32,
                ],
                block_dim=op.attrs["_q8_mma_tile_m"] * 8,
                device=device,
            )
            return
        wp.launch(
            op.attrs["_q8_kernel"],
            dim=(
                op.attrs["_rows"]
                * (op.attrs["_columns"] + op.attrs["_q8_outputs_per_group"] - 1)
                // op.attrs["_q8_outputs_per_group"]
                * 8
            ),
            inputs=[
                op.attrs["_q8_activation_words"],
                op.attrs["_q8_scales"],
                weight.words,
                weight.scales,
                output,
            ],
            block_dim=128,
            device=device,
        )
        return
    cublas = op.attrs.get("_cublas")
    if cublas is not None:
        cublas.gemm(
            x.ptr,
            weight.ptr,
            output.ptr,
            op.attrs["_rows"],
            op.attrs["_columns"],
            op.attrs["_inner"],
            wp.get_stream(device).cuda_stream,
            2 if x.dtype == wp.float16 else 14,
        )
    elif device.is_cuda:
        if op.attrs.get("_prefill_mma_kernel"):
            tile_m, tile_n = op.attrs["_mma_tile_shape"]
            wp.launch(
                op.attrs["_kernel"],
                dim=(op.attrs["_rows"] // tile_m)
                * (op.attrs["_columns"] // tile_n)
                * op.attrs["_mma_block_dim"],
                inputs=[x, weight, output, op.attrs["_columns"], op.attrs["_inner"]],
                block_dim=op.attrs["_mma_block_dim"],
                device=device,
            )
        elif op.attrs.get("_grouped_decode_kernel"):
            wp.launch(
                op.attrs["_kernel"],
                dim=(weight.shape[0] // 8) * 32,
                inputs=[x, weight, output, op.attrs["_inner"]],
                block_dim=128,
                device=device,
            )
        elif op.attrs.get("_small_batch_grouped_kernel"):
            outputs_per_group = op.attrs["_small_batch_outputs_per_group"]
            wp.launch(
                op.attrs["_kernel"],
                dim=(weight.shape[0] // outputs_per_group) * 32,
                inputs=[x, weight, output, op.attrs["_inner"]],
                block_dim=128,
                device=device,
            )
        elif op.attrs.get("_vector_kernel"):
            wp.launch_tiled(
                op.attrs["_kernel"],
                dim=x.shape[0] * weight.shape[0],
                inputs=[x, weight, output],
                block_dim=128,
                device=device,
            )
        else:
            tile_m, tile_n = op.attrs["_tile_shape"]
            wp.launch_tiled(
                op.attrs["_kernel"],
                dim=(
                    (x.shape[0] + tile_m - 1) // tile_m,
                    (weight.shape[0] + tile_n - 1) // tile_n,
                ),
                inputs=[x, weight, output],
                block_dim=128,
                device=device,
            )
    else:
        wp.launch(
            _linear_kernel,
            dim=output.shape,
            inputs=[x, weight, output],
            device=device,
        )


def plan_linear(
    op: Operation,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device,
    cublas=None,
    quantized_activation_cache=None,
):
    """Allocate and specialize a dense projection operation."""
    rows, inner = shapes[op.inputs[0]]
    columns, weight_inner = shapes[op.inputs[1]]
    if weight_inner != inner:
        raise ValueError(
            f"Linear has incompatible shapes {(rows, inner)} and {(columns, weight_inner)}"
        )
    activation = tensors[op.inputs[0]]
    dtype = activation.dtype
    weight = tensors[op.inputs[1]]
    if isinstance(weight, PackedQuantizedTensor):
        if weight.format != "Q3_K":
            raise TypeError(f"Linear does not support packed {weight.format} weights")
        if not device.is_cuda or dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Q3_K Linear requires CUDA FP16/BF16 activations")
        if inner % 256:
            raise ValueError("Q3_K Linear requires an inner width divisible by 256")
        output = wp.empty((rows, columns), dtype=dtype, device=device)
        tensors[op.outputs[0]] = output
        shapes[op.outputs[0]] = output.shape
        op.attrs.update(
            {
                "_rows": rows,
                "_columns": columns,
                "_inner": inner,
                "_q3_k_kernel": _get_q3_k_linear_kernel(dtype),
            }
        )
        return
    if isinstance(weight, BlockQuantizedTensor):
        if weight.format in ("NVFP4", "NVFP4_MMA"):
            enable_nvfp4_native(device)
            if dtype not in (wp.float16, wp.bfloat16):
                raise TypeError("NVFP4 Linear requires FP16/BF16 activations")
            if inner % 64 or columns % 8:
                raise ValueError("NVFP4 Linear requires K%64 == N%8 == 0")
            if weight.format == "NVFP4":
                weight = repack_gguf_nvfp4_weight(weight)
                tensors[op.inputs[1]] = weight
            padded_rows = (rows + 15) // 16 * 16
            padded_output = wp.empty((padded_rows, columns), dtype=dtype, device=device)
            output = wp.array(
                ptr=padded_output.ptr,
                dtype=dtype,
                shape=(rows, columns),
                capacity=padded_output.capacity,
                device=device,
                copy=False,
            )
            tensors[op.outputs[0]] = output
            shapes[op.outputs[0]] = output.shape
            cache_key = (weight.format, op.inputs[0], padded_rows, inner, dtype)
            cached_activation = (
                quantized_activation_cache.get(cache_key)
                if quantized_activation_cache is not None
                else None
            )
            if cached_activation is None:
                quantized = wp.zeros(
                    (padded_rows, inner // 2), dtype=wp.uint8, device=device
                )
                activation_scales = wp.zeros(
                    (padded_rows, inner // 16), dtype=wp.uint8, device=device
                )
                activation_global_scales = wp.zeros(
                    padded_rows, dtype=wp.float32, device=device
                )
                cached_activation = (
                    quantized,
                    activation_scales,
                    activation_global_scales,
                )
                if quantized_activation_cache is not None:
                    quantized_activation_cache[cache_key] = cached_activation
                op.attrs["_nvfp4_quantize_kernel"] = _get_quantize_nvfp4_kernel(dtype)
                op.attrs["_nvfp4_row_scale_kernel"] = _get_nvfp4_row_scale_kernel(dtype)
            quantized, activation_scales, activation_global_scales = cached_activation
            # Prefill reuses each K256 weight tile across four M16 warps. Decode's
            # long-K down projection instead needs more warps to saturate SM120.
            reuse_weights = padded_rows >= 64 and padded_rows % 64 == 0
            split_k = (
                8 if padded_rows == 16 and columns >= 1024 and inner >= 8192 else 0
            )
            op.attrs.update(
                {
                    "_rows": rows,
                    "_columns": columns,
                    "_inner": inner,
                    "_nvfp4_padded_rows": padded_rows,
                    "_nvfp4_activations": quantized,
                    "_nvfp4_scales": activation_scales,
                    "_nvfp4_global_scales": activation_global_scales,
                    "_nvfp4_output": padded_output,
                    "_nvfp4_mma_kernel": _get_nvfp4_mma_linear_kernel(
                        dtype, reuse_weights, split_k
                    ),
                    "_nvfp4_grid_multiplier": split_k or 1,
                    "_nvfp4_block_dim": max(128, split_k * 32),
                }
            )
            return
        if weight.format != "Q8_0":
            raise TypeError(f"Linear does not support block {weight.format} weights")
        if not device.is_cuda or dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Q8_0 Linear requires CUDA FP16/BF16 activations")
        if inner % 32:
            raise ValueError("Q8_0 Linear requires an inner width divisible by 32")
        output = wp.empty((rows, columns), dtype=dtype, device=device)
        tensors[op.outputs[0]] = output
        shapes[op.outputs[0]] = output.shape
        op.attrs.update({"_rows": rows, "_columns": columns, "_inner": inner})
        blocks = inner // 32
        cache_key = (weight.format, op.inputs[0], rows, inner, dtype)
        cached_activation = (
            quantized_activation_cache.get(cache_key)
            if quantized_activation_cache is not None
            else None
        )
        if cached_activation is None:
            quantized = wp.empty((rows, inner), dtype=wp.int8, device=device)
            activation_words = wp.array(
                ptr=quantized.ptr,
                capacity=quantized.capacity,
                dtype=wp.uint32,
                shape=(rows, blocks, 8),
                device=device,
            )
            activation_scales = wp.empty(
                (rows, blocks), dtype=wp.float32, device=device
            )
            if quantized_activation_cache is not None:
                quantized_activation_cache[cache_key] = (
                    quantized,
                    activation_words,
                    activation_scales,
                )
            op.attrs["_q8_quantize_kernel"] = _get_quantize_activation_int8_kernel(
                dtype
            )
        else:
            quantized, activation_words, activation_scales = cached_activation
        op.attrs["_q8_activations"] = quantized
        op.attrs["_q8_activation_words"] = activation_words
        op.attrs["_q8_scales"] = activation_scales
        if (
            device.arch >= 80
            and rows % 16 == 0
            and columns % 32 == 0
            and quantized.is_contiguous
            and weight.values.is_contiguous
            and weight.scales.is_contiguous
            and output.is_contiguous
            and quantized.ptr % 4 == 0
            and weight.values.ptr % 4 == 0
        ):
            tile_m = (
                64
                if rows % 64 == 0
                and quantized.ptr % 16 == 0
                and weight.values.ptr % 16 == 0
                else 16
            )
            op.attrs["_q8_mma_tile_m"] = tile_m
            op.attrs["_q8_mma_kernel"] = _get_q8_prefill_mma_linear_kernel(
                dtype, tile_m
            )
        # Share activation loads only when halving the grid retains two CTAs per SM.
        grouped_blocks = (rows * ((columns + 1) // 2) * 8 + 127) // 128
        outputs_per_group = 2 if grouped_blocks >= 2 * device.sm_count else 1
        op.attrs["_q8_outputs_per_group"] = outputs_per_group
        if rows == 1 and columns % outputs_per_group == 0 and device.arch >= 61:
            op.attrs["_q8_decode_outputs_per_group"] = outputs_per_group
            op.attrs["_q8_grouped_decode_kernel"] = (
                _get_q8_grouped_decode_linear_kernel(dtype, outputs_per_group)
            )
        op.attrs["_q8_kernel"] = _get_matmul_int8_q8_kernel(
            8, dtype, True, outputs_per_group
        )
        return
    if weight.dtype != dtype or dtype not in (
        wp.float16,
        wp.bfloat16,
        wp.float32,
    ):
        raise TypeError("Linear requires matching FP16, BF16, or FP32 inputs")
    output = wp.empty((rows, columns), dtype=dtype, device=device)
    tensors[op.outputs[0]] = output
    shapes[op.outputs[0]] = output.shape
    op.attrs.update({"_rows": rows, "_columns": columns, "_inner": inner})
    grouped_decode = (
        device.is_cuda
        and rows == 1
        and dtype in (wp.float16, wp.bfloat16)
        and columns % 8 == 0
        and inner % 8 == 0
        and (columns + 31) // 32 >= device.sm_count
        and activation.is_contiguous
        and weight.is_contiguous
        and output.is_contiguous
        and activation.ptr % 16 == 0
        and weight.ptr % 16 == 0
        and output.ptr % 16 == 0
    )
    small_batch_grouped = (
        bool(op.attrs.get("_small_batch_decode"))
        and device.is_cuda
        and rows in (2, 4, 8)
        and dtype in (wp.float16, wp.bfloat16)
        and columns % 8 == 0
        and inner % 8 == 0
        and (columns + 31) // 32 >= device.sm_count
        and activation.is_contiguous
        and weight.is_contiguous
        and output.is_contiguous
        and activation.ptr % 16 == 0
        and weight.ptr % 16 == 0
        and output.ptr % 16 == 0
    )
    if grouped_decode:
        op.attrs["_kernel"] = _get_grouped_decode_linear_kernel(dtype)
        op.attrs["_grouped_decode_kernel"] = True
    elif small_batch_grouped:
        outputs_per_group = int(op.attrs.get("_small_batch_outputs_per_group", 8))
        op.attrs["_kernel"] = _get_small_batch_grouped_linear_kernel(
            dtype, rows, outputs_per_group
        )
        op.attrs["_small_batch_grouped_kernel"] = True
        op.attrs["_small_batch_outputs_per_group"] = outputs_per_group
    elif cublas is not None and device.is_cuda and dtype in (wp.float16, wp.bfloat16):
        op.attrs["_cublas"] = cublas
    elif device.is_cuda:
        mma_geometry = None
        mma_common = (
            device.arch >= 80
            and dtype in (wp.float16, wp.bfloat16)
            and inner % 32 == 0
            and activation.is_contiguous
            and weight.is_contiguous
            and output.is_contiguous
            and activation.ptr % 16 == 0
            and weight.ptr % 16 == 0
            and output.ptr % 16 == 0
        )
        contraction_tile_m = 64 if rows % 64 == 0 else 16
        contraction_blocks = rows // contraction_tile_m * (columns // 64)
        if (
            mma_common
            and rows >= 64
            and rows % 64 == 0
            and columns % 64 == 0
            and columns <= inner
            and contraction_blocks >= device.sm_count
        ):
            mma_geometry = (64, 64, 512)
        elif (
            mma_common
            and rows >= 16
            and rows % 16 == 0
            and columns % 64 == 0
            and columns <= inner
            and rows // 16 * (columns // 64) >= (device.sm_count + 1) // 2
        ):
            mma_geometry = (16, 64, 128)
        expansion_tile_m = 128 if rows % 128 == 0 else 64
        expansion_blocks = rows // expansion_tile_m * (columns // 32)
        if (
            mma_common
            and rows >= 64
            and rows % 64 == 0
            and columns % 32 == 0
            and columns > inner
            and expansion_blocks >= 2 * device.sm_count
        ):
            mma_geometry = (expansion_tile_m, 32, expansion_tile_m * 4)
        if mma_geometry is not None:
            tile_m, tile_n, block_dim = mma_geometry
            op.attrs["_kernel"] = _get_prefill_mma_linear_kernel(dtype, tile_m, tile_n)
            op.attrs["_prefill_mma_kernel"] = True
            op.attrs["_mma_tile_shape"] = (tile_m, tile_n)
            op.attrs["_mma_block_dim"] = block_dim
        elif rows < 8:
            op.attrs["_kernel"] = _get_linear_vector_kernel(dtype)
            op.attrs["_vector_kernel"] = True
        else:
            tile_m = 8 if rows < 16 else 16 if rows < 32 else 32
            candidate_blocks = (rows + 63) // 64 * ((columns + 31) // 32)
            if rows >= 64 and candidate_blocks >= 2 * device.sm_count:
                tile_m = 64
            tile_k = 128 if tile_m == 16 else 32
            op.attrs["_kernel"], op.attrs["_tile_shape"] = _get_linear_tiled_kernel(
                dtype, tile_m, tile_k
            )


def _plan_rms_norm_buffers(op, x_name, scale_name, tensors, shapes, device, dtype):
    shape = shapes[x_name]
    rows, width = int(np.prod(shape[:-1])), shape[-1]
    dtype = dtype or tensors[x_name].dtype
    scale_dtype = tensors[scale_name].dtype
    if (
        dtype not in (wp.float16, wp.bfloat16)
        or scale_dtype
        not in (
            wp.float16,
            wp.bfloat16,
            wp.float32,
        )
        or shapes[scale_name] != (width,)
    ):
        raise ValueError("RMSNorm requires a matching width-sized scale")
    output = wp.empty(shape, dtype=dtype, device=device)
    tensors[op.outputs[0]] = output
    shapes[op.outputs[0]] = shape
    op.attrs.update(
        {"_rows": rows, "_width": width, "_output_2d": output.reshape((rows, width))}
    )
    op.attrs["_tile_width"], op.attrs["_rms_norm_kernels"] = _get_rms_norm_kernels(
        width, dtype, scale_dtype
    )


def plan_rms_norm(
    op: Operation,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device,
    dtype=None,
):
    """Allocate and specialize last-axis RMS normalization."""
    _plan_rms_norm_buffers(
        op, op.inputs[0], op.inputs[1], tensors, shapes, device, dtype
    )


def plan_residual_rms_norm(
    op: Operation,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device,
    dtype=None,
):
    """Allocate and specialize fused residual addition and RMSNorm."""
    shape = shapes[op.inputs[0]]
    if shapes[op.inputs[1]] != shape or (
        op.inputs[0] in tensors
        and op.inputs[1] in tensors
        and tensors[op.inputs[1]].dtype != tensors[op.inputs[0]].dtype
    ):
        raise ValueError("ResidualRMSNorm requires matching activation shapes")
    _plan_rms_norm_buffers(
        op, op.inputs[0], op.inputs[2], tensors, shapes, device, dtype
    )
    residual = wp.empty(
        shape, dtype=dtype or tensors[op.inputs[0]].dtype, device=device
    )
    if len(op.outputs) > 3 and op.outputs[3]:
        tensors[op.outputs[3]] = residual
        shapes[op.outputs[3]] = shape
    op.attrs["_residual_2d"] = residual.reshape((op.attrs["_rows"], op.attrs["_width"]))


def plan_swiglu(
    op: Operation,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device,
    dtype=None,
):
    """Allocate and specialize fused SiLU-gate multiplication."""
    shape = shapes[op.inputs[0]]
    if shapes[op.inputs[1]] != shape:
        raise ValueError("SwiGLU requires matching activation shapes")
    dtype = dtype or tensors[op.inputs[0]].dtype
    if (
        op.inputs[1] in tensors and tensors[op.inputs[1]].dtype != dtype
    ) or dtype not in (wp.float16, wp.bfloat16):
        raise TypeError("SwiGLU requires matching FP16 or BF16 inputs")
    output = wp.empty(shape, dtype=dtype, device=device)
    tensors[op.outputs[0]] = output
    shapes[op.outputs[0]] = shape
    op.attrs["_shape_2d"] = (int(np.prod(shape[:-1])), shape[-1])
    op.attrs["_output_2d"] = output.reshape(op.attrs["_shape_2d"])
    op.attrs["_kernel"] = _get_swiglu_kernel(dtype)


def _exec_gemm(op, tensors, shapes, device):
    A = tensors[op.inputs[0]]
    B = tensors[op.inputs[1]]
    bias = tensors[op.inputs[2]]
    out = tensors[op.outputs[0]]
    alpha = float(op.attrs.get("alpha", 1.0))
    beta = float(op.attrs.get("beta", 1.0))
    M = shapes[op.inputs[0]][0]
    N, K = shapes[op.inputs[1]]

    if op.attrs["_requires_grad"] or A.dtype != wp.float32:
        wp.launch(
            op.attrs["_kernel"],
            dim=(M, N),
            inputs=[A, B, bias, out, K, alpha, beta],
            device=device,
        )
    else:
        wp.launch_tiled(
            _GEMM_TRANSB_TILED_KERNEL,
            dim=resolve_dim(config=_GEMM_CONFIG, shape=(M, N), tiled=True),
            inputs=[A, B, op.attrs["_bias_2d"], alpha, beta],
            outputs=[out],
            device=device,
            block_dim=_GEMM_CONFIG.block_dim,
        )


def _exec_elu(op, tensors, shapes, device):
    x = tensors[op.inputs[0]]
    alpha = float(op.attrs.get("alpha", 1.0))
    out = tensors[op.outputs[0]]
    shape = op.attrs["_shape_2d"]
    wp.launch(
        op.attrs["_kernel"],
        dim=shape,
        inputs=[x.reshape(shape), out.reshape(shape), alpha],
        device=device,
    )


def _exec_unary(op, tensors, shapes, device):
    operation = {"Relu": 0, "Tanh": 1, "Sqrt": 2, "Sigmoid": 3, "Softplus": 4}[
        op.op_type
    ]
    shape_2d = op.attrs["_shape_2d"]
    wp.launch(
        op.attrs["_kernel"],
        dim=shape_2d,
        inputs=[tensors[op.inputs[0]].reshape(shape_2d), operation],
        outputs=[tensors[op.outputs[0]].reshape(shape_2d)],
        device=device,
    )


def _exec_binary(op, tensors, shapes, device):
    if op.attrs.get("_static_output"):
        return
    lhs = tensors[op.inputs[0]].reshape(op.attrs["_lhs_shape_2d"])
    rhs = tensors[op.inputs[1]].reshape(op.attrs["_rhs_shape_2d"])
    operation = {"Add": 0, "Sub": 1, "Mul": 2, "Div": 3}[op.op_type]
    wp.launch(
        op.attrs["_kernel"],
        dim=op.attrs["_out_shape_2d"],
        inputs=[lhs, rhs, operation],
        outputs=[tensors[op.outputs[0]].reshape(op.attrs["_out_shape_2d"])],
        device=device,
    )


def _exec_reduce_mean(op, tensors, shapes, device):
    wp.launch(
        op.attrs["_kernel"],
        dim=shapes[op.inputs[0]][0],
        inputs=[tensors[op.inputs[0]]],
        outputs=[tensors[op.outputs[0]]],
        device=device,
    )


def _exec_reduce_sum(op, tensors, shapes, device):
    wp.launch_tiled(
        op.attrs["_kernel"],
        dim=shapes[op.inputs[0]][0],
        inputs=[tensors[op.inputs[0]], tensors[op.outputs[0]]],
        block_dim=op.attrs["_tile_width"],
        device=device,
    )


def _exec_static(op, tensors, shapes, device):
    pass


def _exec_gather(op, tensors, shapes, device):
    if op.attrs.get("_static_output"):
        return
    if "_single_index" in op.attrs:
        data = tensors[op.inputs[0]]
        output = tensors[op.outputs[0]]
        wp.launch(
            op.attrs["_kernel"],
            dim=output.size,
            inputs=[
                data.flatten(),
                output.flatten(),
                op.attrs["_single_index"],
                op.attrs["_axis_size"],
                op.attrs["_stride"],
            ],
            device=device,
        )
        return
    if not op.attrs.get("_dynamic"):
        return
    data = tensors[op.inputs[0]]
    wp.launch(
        op.attrs["_kernel"],
        dim=shapes[op.outputs[0]],
        inputs=[data, tensors[op.inputs[1]], tensors[op.outputs[0]]],
        device=device,
    )


def _exec_cast(op, tensors, shapes, device):
    if op.attrs.get("_static_output"):
        return
    size = int(np.prod(shapes[op.inputs[0]]))
    wp.launch(
        op.attrs["_kernel"],
        dim=size,
        inputs=[
            tensors[op.inputs[0]].reshape((size,)),
            tensors[op.outputs[0]].reshape((size,)),
        ],
        device=device,
    )


def _exec_lp_normalization(op, tensors, shapes, device):
    rows = op.attrs["_rows"]
    width = op.attrs["_width"]
    wp.launch_tiled(
        op.attrs["_kernel"],
        dim=rows,
        inputs=[
            tensors[op.inputs[0]].reshape((rows, width)),
            tensors[op.outputs[0]].reshape((rows, width)),
            float(op.attrs.get("_epsilon", 0.0)),
        ],
        block_dim=op.attrs["_tile_width"],
        device=device,
    )


def _exec_reduce_max(op, tensors, shapes, device):
    wp.launch(
        op.attrs["_kernel"],
        dim=1,
        inputs=[tensors[op.inputs[0]], tensors[op.outputs[0]]],
        device=device,
    )


def _exec_batch_normalization(op, tensors, shapes, device):
    wp.launch(
        op.attrs["_kernel"],
        dim=shapes[op.inputs[0]],
        inputs=[
            tensors[op.inputs[0]],
            tensors[op.inputs[1]],
            tensors[op.inputs[2]],
            tensors[op.inputs[3]],
            tensors[op.inputs[4]],
            float(op.attrs.get("epsilon", 1.0e-5)),
            op.op_type == "_BatchNormalizationRelu",
        ],
        outputs=[tensors[op.outputs[0]]],
        device=device,
    )


def _exec_rms_normalization(op, tensors, shapes, device):
    wp.launch_tiled(
        op.attrs["_kernel"],
        dim=shapes[op.inputs[0]][0],
        inputs=[tensors[op.inputs[0]], tensors[op.inputs[1]], op.attrs["_scale"]],
        outputs=[tensors[op.outputs[0]]],
        device=device,
        block_dim=_GEMM_CONFIG.block_dim,
    )


def _exec_constant(op, tensors, shapes, device):
    pass


def _exec_reshape(op, tensors, shapes, device):
    tensors[op.outputs[0]] = tensors[op.inputs[0]].reshape(op.attrs["_out_shape"])


def _exec_transpose(op, tensors, shapes, device):
    if op.attrs["_view_only"]:
        return
    wp.launch(
        op.attrs["_kernel"],
        dim=shapes[op.outputs[0]],
        inputs=[tensors[op.inputs[0]], tensors[op.outputs[0]]],
        device=device,
    )


def _exec_split(op, tensors, shapes, device):
    if op.attrs["_view_only"]:
        return
    rows = op.attrs["_rows"]
    source = tensors[op.inputs[0]].reshape((rows, shapes[op.inputs[0]][-1]))
    offset = 0
    for name, width in zip(op.outputs, op.attrs["_split_sizes"]):
        wp.launch(
            op.attrs["_kernel"],
            dim=(rows, width),
            inputs=[source, tensors[name].reshape((rows, width)), offset],
            device=device,
        )
        offset += width


def _exec_tile(op, tensors, shapes, device):
    wp.launch(
        op.attrs["_kernel"],
        dim=shapes[op.outputs[0]],
        inputs=[tensors[op.inputs[0]], tensors[op.outputs[0]]],
        device=device,
    )


def _exec_where(op, tensors, shapes, device):
    shape_2d = op.attrs["_shape_2d"]
    wp.launch(
        op.attrs["_kernel"],
        dim=shape_2d,
        inputs=[
            tensors[op.inputs[0]].reshape(op.attrs["_condition_shape_2d"]),
            tensors[op.inputs[1]].reshape(shape_2d),
            tensors[op.inputs[2]].reshape(shape_2d),
            tensors[op.outputs[0]].reshape(shape_2d),
        ],
        device=device,
    )


def _exec_rotary_embedding(op, tensors, shapes, device):
    wp.launch(
        op.attrs["_kernel"],
        dim=shapes[op.inputs[0]],
        inputs=[
            tensors[op.inputs[0]],
            tensors[op.inputs[1]].reshape(op.attrs["_position_shape_2d"]),
            tensors[op.inputs[2]],
            tensors[op.inputs[3]],
            tensors[op.outputs[0]],
            int(op.attrs.get("rotary_embedding_dim", shapes[op.inputs[0]][-1])),
            bool(op.attrs.get("interleaved", 0)),
            op.attrs["_position_offset"],
        ],
        device=device,
    )


def _exec_gather_block_quantized(op, tensors, shapes, device):
    wp.launch(
        _gather_block_quantized_int8_kernel,
        dim=shapes[op.outputs[0]],
        inputs=[
            tensors[op.inputs[0]],
            tensors[op.inputs[1]],
            tensors[op.inputs[2]],
            tensors[op.inputs[3]],
            tensors[op.outputs[0]],
            128,
        ],
        device=device,
    )


def _exec_matmul_nbits(op, tensors, shapes, device):
    K = int(op.attrs["K"])
    N = int(op.attrs["N"])
    bits = int(op.attrs["bits"])
    block_size = op.attrs["_block_size"]
    dtype = op.attrs["_dtype"]
    zero_points = op.attrs["_zero_points"]
    has_zero_points = op.attrs["_has_zero_points"]
    if "_q8_activations" in op.attrs:
        activations = tensors[op.inputs[0]].reshape((op.attrs["_rows"], K))
        wp.launch(
            _quantize_activation_int8_kernel,
            dim=op.attrs["_rows"] * K,
            inputs=[activations, op.attrs["_q8_activations"], op.attrs["_q8_scales"]],
            block_dim=128,
            device=device,
        )
        wp.launch(
            op.attrs["_q8_kernel"],
            dim=op.attrs["_rows"] * N * op.attrs["_q8_width"],
            inputs=[
                op.attrs["_q8_activation_words"],
                op.attrs["_q8_scales"],
                op.attrs["_q8_weight_words"],
                tensors[op.inputs[2]],
                op.attrs["_output_2d"],
            ],
            block_dim=128,
            device=device,
        )
        return
    if "_cublas" in op.attrs:
        weights = tensors[op.inputs[1]]
        dequantized = op.attrs["_dequantized_weights"]
        wp.launch(
            op.attrs["_dequantize_kernel"],
            dim=(N, weights.shape[1] * weights.shape[2]),
            inputs=[
                weights,
                tensors[op.inputs[2]],
                zero_points,
                dequantized,
                has_zero_points,
            ],
            device=device,
        )
        op.attrs["_cublas"].gemm(
            tensors[op.inputs[0]].ptr,
            dequantized.ptr,
            op.attrs["_output_2d"].ptr,
            op.attrs["_rows"],
            N,
            K,
            wp.get_stream(device).cuda_stream,
            2 if dtype == wp.float16 else 14,
        )
        return
    if "_tile_gemm_kernel" in op.attrs:
        wp.launch_tiled(
            op.attrs["_tile_gemm_kernel"],
            dim=op.attrs["_tile_gemm_dim"],
            inputs=[
                tensors[op.inputs[0]].reshape((op.attrs["_rows"], K)),
                op.attrs["_tile_gemm_weights"],
                tensors[op.inputs[2]],
                op.attrs["_output_2d"],
            ],
            block_dim=128,
            device=device,
        )
        return
    if device.is_cuda:
        wp.launch(
            op.attrs["_matmul_kernel"],
            dim=op.attrs["_rows"] * N * op.attrs["_reduction_width"],
            inputs=[
                tensors[op.inputs[0]].reshape((op.attrs["_rows"], K)),
                tensors[op.inputs[1]],
                tensors[op.inputs[2]],
                zero_points,
                op.attrs["_output_2d"],
                has_zero_points,
            ],
            block_dim=128,
            device=device,
        )
        return
    wp.launch(
        op.attrs["_matmul_kernel"],
        dim=(op.attrs["_rows"], N),
        inputs=[
            tensors[op.inputs[0]].reshape((op.attrs["_rows"], K)),
            tensors[op.inputs[1]],
            tensors[op.inputs[2]],
            zero_points,
            op.attrs["_output_2d"],
            has_zero_points,
        ],
        device=device,
    )


def _exec_causal_conv_with_state(op, tensors, shapes, device):
    x = tensors[op.inputs[0]]
    past = tensors[op.inputs[3]] if op.attrs["_has_past"] else op.attrs["_past"]
    if op.attrs["_share_state"]:
        tensors[op.outputs[1]] = past
    wp.launch(
        op.attrs["_kernel"],
        dim=x.shape,
        inputs=[
            x,
            tensors[op.inputs[1]],
            op.attrs["_bias"],
            past,
            tensors[op.outputs[0]],
            op.attrs["_kernel_size"],
            op.attrs["_has_bias"],
            op.attrs["_silu"],
        ],
        device=device,
    )
    if op.attrs["_kernel_size"] > 1:
        if op.attrs["_share_state"]:
            wp.launch(
                op.attrs["_inplace_state_kernel"],
                dim=(x.shape[0], x.shape[1]),
                inputs=[x, past],
                device=device,
            )
        else:
            wp.launch(
                op.attrs["_state_kernel"],
                dim=tensors[op.outputs[1]].shape,
                inputs=[x, past, tensors[op.outputs[1]]],
                device=device,
            )


def _exec_linear_attention(op, tensors, shapes, device):
    batch = op.attrs["_batch"]
    sequence_length = op.attrs["_sequence_length"]
    query_heads = op.attrs["_query_heads"]
    key_heads = op.attrs["_key_heads"]
    value_heads = op.attrs["_value_heads"]
    key_size = op.attrs["_key_size"]
    value_size = op.attrs["_value_size"]
    past = tensors[op.inputs[3]] if op.attrs["_has_past"] else op.attrs["_past"]
    if op.attrs["_share_state"]:
        tensors[op.outputs[1]] = past
    decay = tensors[op.inputs[4]] if op.attrs["_needs_decay"] else op.attrs["_decay"]
    beta = tensors[op.inputs[5]] if op.attrs["_needs_beta"] else op.attrs["_beta"]
    wp.launch_tiled(
        op.attrs["_kernel"],
        dim=batch * value_heads * op.attrs["_value_blocks"],
        inputs=[
            tensors[op.inputs[0]].reshape(
                (batch * sequence_length, query_heads * key_size)
            ),
            tensors[op.inputs[1]].reshape(
                (batch * sequence_length, key_heads * key_size)
            ),
            tensors[op.inputs[2]].reshape(
                (batch * sequence_length, value_heads * value_size)
            ),
            past.reshape((batch * value_heads * key_size, value_size)),
            decay.reshape((int(np.prod(decay.shape[:-1])), decay.shape[-1])),
            beta.reshape((int(np.prod(beta.shape[:-1])), beta.shape[-1])),
            tensors[op.outputs[0]].reshape(
                (batch * sequence_length, max(query_heads, value_heads) * value_size)
            ),
            tensors[op.outputs[1]].reshape(
                (batch * value_heads * key_size, value_size)
            ),
            sequence_length,
            query_heads,
            key_heads,
            value_heads,
            False,
            op.attrs["_needs_decay"],
            op.attrs["_decay_per_key"],
            op.attrs["_needs_beta"],
            op.attrs["_beta_per_head"],
            op.attrs["_scale"],
        ],
        block_dim=op.attrs["_block_dim"],
        device=device,
    )


def _exec_simplified_layer_normalization(op, tensors, shapes, device):
    wp.launch_tiled(
        op.attrs["_rms_norm_kernels"][0],
        dim=op.attrs["_rows"],
        inputs=[
            tensors[op.inputs[0]].reshape((op.attrs["_rows"], op.attrs["_width"])),
            tensors[op.inputs[1]],
            op.attrs["_output_2d"],
            float(op.attrs.get("epsilon", 1.0e-5)),
            float(op.attrs.get("_scale_offset", 0.0)),
        ],
        block_dim=op.attrs["_tile_width"],
        device=device,
    )


def _exec_skip_simplified_layer_normalization(op, tensors, shapes, device):
    shape_2d = (op.attrs["_rows"], op.attrs["_width"])
    wp.launch_tiled(
        op.attrs["_rms_norm_kernels"][1],
        dim=op.attrs["_rows"],
        inputs=[
            tensors[op.inputs[0]].reshape(shape_2d),
            tensors[op.inputs[1]].reshape(shape_2d),
            tensors[op.inputs[2]],
            op.attrs["_output_2d"],
            op.attrs["_residual_2d"],
            float(op.attrs.get("epsilon", 1.0e-5)),
            float(op.attrs.get("_scale_offset", 0.0)),
        ],
        block_dim=op.attrs["_tile_width"],
        device=device,
    )


def _exec_swiglu(op, tensors, shapes, device):
    wp.launch(
        op.attrs["_kernel"],
        dim=op.attrs["_shape_2d"],
        inputs=[
            tensors[op.inputs[0]].reshape(op.attrs["_shape_2d"]),
            tensors[op.inputs[1]].reshape(op.attrs["_shape_2d"]),
            op.attrs["_output_2d"],
        ],
        device=device,
    )


def _exec_group_query_attention(op, tensors, shapes, device):
    batch = op.attrs["_batch"]
    sequence_length = op.attrs["_sequence_length"]
    past_length = op.attrs["_past_length"]
    total_length = op.attrs["_total_length"]
    head_size = op.attrs["_head_size"]
    query_heads = int(op.attrs["num_heads"])
    kv_heads = int(op.attrs["kv_num_heads"])
    share_cache = bool(op.attrs.get("_share_cache", False))
    if share_cache:
        present_key = tensors[op.inputs[3]]
        present_value = tensors[op.inputs[4]]
        tensors[op.outputs[1]] = present_key
        tensors[op.outputs[2]] = present_value
    else:
        present_key = tensors[op.outputs[1]]
        present_value = tensors[op.outputs[2]]
    if past_length and not share_cache:
        wp.launch(
            _gqa_copy_past_fp16_kernel,
            dim=(batch, kv_heads, past_length, head_size),
            inputs=[
                tensors[op.inputs[3]],
                tensors[op.inputs[4]],
                present_key,
                present_value,
            ],
            device=device,
        )
    wp.launch(
        _gqa_prepare_fp16_kernel,
        dim=(batch, query_heads, sequence_length, head_size),
        inputs=[
            tensors[op.inputs[0]],
            tensors[op.inputs[1]],
            tensors[op.inputs[2]],
            tensors[op.inputs[5]],
            op.attrs["_cos_cache"],
            op.attrs["_sin_cache"],
            op.attrs["_rotated_query"],
            present_key,
            present_value,
            query_heads,
            kv_heads,
            sequence_length,
            past_length,
            head_size,
            share_cache,
            op.attrs["_do_rotary"],
        ],
        device=device,
    )
    wp.launch_tiled(
        op.attrs["_attention_kernel"],
        dim=batch * query_heads * sequence_length,
        inputs=[
            op.attrs["_rotated_query"].reshape(
                (batch * query_heads * sequence_length, head_size)
            ),
            present_key.reshape((batch * kv_heads * total_length, head_size)),
            present_value.reshape((batch * kv_heads * total_length, head_size)),
            tensors[op.inputs[5]],
            tensors[op.outputs[0]].reshape(
                (batch * sequence_length, query_heads * head_size)
            ),
            query_heads,
            kv_heads,
            sequence_length,
            total_length,
            float(op.attrs.get("scale", head_size**-0.5)),
            0,
        ],
        block_dim=op.attrs["_attention_block_dim"],
        device=device,
    )


def _exec_squeeze(op, tensors, shapes, device):
    src = tensors[op.inputs[0]]
    out_shape = op.attrs["_out_shape"]
    tensors[op.outputs[0]] = src.reshape(out_shape)
    shapes[op.outputs[0]] = out_shape


def _exec_lstm(op, tensors, shapes, device):
    cache = op.attrs["_cache"]
    input_size: int = cache["input_size"]
    hidden_size: int = cache["hidden_size"]
    batch: int = cache["batch"]
    layout: int = cache["layout"]

    X = tensors[op.inputs[0]]
    if layout == 0:
        x_t = X.reshape((batch, input_size))
    else:
        x_t = X.reshape((batch, input_size))

    if len(op.inputs) > 5 and op.inputs[5] and op.inputs[5] in tensors:
        h_prev = tensors[op.inputs[5]].reshape((batch, hidden_size))
    else:
        if "h_prev_zero" not in cache:
            cache["h_prev_zero"] = wp.zeros(
                (batch, hidden_size), dtype=cache["dtype"], device=device
            )
        h_prev = cache["h_prev_zero"]
    if len(op.inputs) > 6 and op.inputs[6] and op.inputs[6] in tensors:
        c_prev = tensors[op.inputs[6]].reshape((batch, hidden_size))
    else:
        if "c_prev_zero" not in cache:
            cache["c_prev_zero"] = wp.zeros(
                (batch, hidden_size), dtype=cache["dtype"], device=device
            )
        c_prev = cache["c_prev_zero"]

    gates = cache["gates"]
    h_out = cache["h_out"]
    c_out = cache["c_out"]

    wp.launch(
        cache["gates_kernel"],
        dim=(batch, 4 * hidden_size),
        inputs=[x_t, h_prev, cache["W"], cache["R"], gates, input_size, hidden_size],
        device=device,
    )
    wp.launch(
        cache["cell_kernel"],
        dim=(batch, hidden_size),
        inputs=[gates, c_prev, cache["Bx"], cache["Bh"], h_out, c_out, hidden_size],
        device=device,
    )


def reuse_operation_outputs(
    layer: dict, tensors: dict, pool: dict, op_type: str | None = None
) -> None:
    """Alias same-role operation outputs across sequential model layers."""
    for role, value in layer.items():
        if isinstance(value, Operation) and (
            op_type is None or value.op_type == op_type
        ):
            for output_index, name in enumerate(value.outputs):
                if name and name in tensors:
                    output = tensors[name]
                    key = (
                        "operation",
                        role,
                        output_index,
                        tuple(output.shape),
                        output.dtype,
                    )
                    shared = pool.setdefault(key, output)
                    tensors[name] = shared
                    if output_index == 0 and "_output_2d" in value.attrs:
                        value.attrs["_output_2d"] = shared.reshape(
                            value.attrs["_output_2d"].shape
                        )
                    if (
                        output_index == 0
                        and shared is not output
                        and "_nvfp4_output" in value.attrs
                    ):
                        padded = value.attrs["_nvfp4_output"]
                        if shared.capacity < padded.capacity:
                            raise ValueError(
                                "shared output cannot hold NVFP4 row padding"
                            )
                        value.attrs["_nvfp4_output"] = wp.array(
                            ptr=shared.ptr,
                            dtype=shared.dtype,
                            shape=padded.shape,
                            capacity=shared.capacity,
                            device=shared.device,
                            copy=False,
                        )
                    if output_index == 3 and "_residual_2d" in value.attrs:
                        value.attrs["_residual_2d"] = shared.reshape(
                            value.attrs["_residual_2d"].shape
                        )


def reuse_linear_outputs(layer: dict, tensors: dict, pool: dict) -> None:
    """Alias same-role Linear outputs across sequential model layers."""
    reuse_operation_outputs(layer, tensors, pool, "Linear")


_ENCODER_DTYPES = (wp.float16, wp.bfloat16, wp.float32)


class EncoderLayerPlan:
    """One fixed-shape PyTorch-compatible post-norm TransformerEncoderLayer."""

    def __init__(
        self,
        x: wp.array,
        valid: wp.array,
        weights: dict[str, wp.array],
        prefix: str,
        heads: int,
        *,
        epsilon: float = 1.0e-5,
        cublas=None,
    ):
        if x.ndim != 3 or valid.shape != x.shape[:2]:
            raise ValueError(
                "encoder input must be [batch, sequence, hidden] with a matching mask"
            )
        if x.dtype not in _ENCODER_DTYPES or valid.dtype != wp.bool:
            raise TypeError("encoder requires FP16/BF16/FP32 input and a boolean mask")
        batch, sequence, hidden = x.shape
        if heads <= 0 or hidden % heads:
            raise ValueError("hidden size must be divisible by the positive head count")
        self.device = x.device
        self.dtype = x.dtype
        self.batch, self.sequence, self.hidden = batch, sequence, hidden
        self.heads, self.head_size = heads, hidden // heads
        self.valid = valid
        self.input = x
        self.output = wp.empty_like(x)
        self._attention_heads = wp.empty(
            (batch, heads, sequence, self.head_size), dtype=x.dtype, device=x.device
        )
        self._query = wp.empty_like(self._attention_heads)
        self._key = wp.empty_like(self._attention_heads)
        self._value = wp.empty_like(self._attention_heads)
        self._attention_flat = wp.empty(
            (batch * sequence, hidden), dtype=x.dtype, device=x.device
        )
        self._norm1 = wp.empty(
            (batch * sequence, hidden), dtype=x.dtype, device=x.device
        )
        self._norm2 = self.output.reshape((batch * sequence, hidden))
        self._epsilon = float(epsilon)
        self._weights = weights
        self._prefix = prefix
        self._tensors = {"x": x.reshape((batch * sequence, hidden))}
        self._shapes = {"x": (batch * sequence, hidden)}
        self._tensors["attention_flat"] = self._attention_flat
        self._shapes["attention_flat"] = self._attention_flat.shape
        self._tensors["norm1"] = self._norm1
        self._shapes["norm1"] = self._norm1.shape
        self._ops = []

        def linear(name, source, weight):
            op = Operation("Linear", [source, weight], [name])
            self._tensors[weight] = weights[weight]
            self._shapes[weight] = weights[weight].shape
            plan_linear(op, self._tensors, self._shapes, self.device, cublas)
            self._ops.append(op)
            return op

        p = prefix
        self._qkv = linear("qkv", "x", f"{p}.self_attn.in_proj_weight")
        self._out = linear(
            "attention_projection", "attention_flat", f"{p}.self_attn.out_proj.weight"
        )
        self._ff1 = linear("ff1", "norm1", f"{p}.linear1.weight")
        self._ff2 = linear("ff2", "ff1", f"{p}.linear2.weight")
        kernels = _encoder_kernels(x.dtype, self.head_size)
        (
            self._add_bias,
            self._bias_gelu,
            self._residual_norm,
            self._split,
            self._merge,
            self._attention,
        ) = kernels

    def _execute(self, op):
        execute_operations([op], self._tensors, self._shapes, self.device)

    def execute(self):
        p = self._prefix
        self._execute(self._qkv)
        qkv = self._tensors["qkv"]
        wp.launch(
            self._add_bias,
            dim=qkv.shape,
            inputs=[qkv, self._weights[f"{p}.self_attn.in_proj_bias"]],
            device=self.device,
        )
        wp.launch(
            self._split,
            dim=self._query.shape,
            inputs=[qkv, self._query, self._key, self._value],
            device=self.device,
        )
        wp.launch_tiled(
            self._attention,
            dim=self.batch * self.heads * self.sequence,
            inputs=[
                self._query,
                self._key,
                self._value,
                self.valid,
                self._attention_heads,
                wp.float32(1.0 / math.sqrt(self.head_size)),
            ],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            self._merge,
            dim=self._attention_heads.shape,
            inputs=[self._attention_heads, self._attention_flat],
            device=self.device,
        )
        self._execute(self._out)
        wp.launch(
            self._residual_norm,
            dim=self.batch * self.sequence,
            inputs=[
                self._tensors["attention_projection"],
                self._tensors["x"],
                self._weights[f"{p}.self_attn.out_proj.bias"],
                self._weights[f"{p}.norm1.weight"],
                self._weights[f"{p}.norm1.bias"],
                self._norm1,
                wp.float32(self._epsilon),
            ],
            device=self.device,
        )
        self._execute(self._ff1)
        wp.launch(
            self._bias_gelu,
            dim=self._tensors["ff1"].shape,
            inputs=[self._tensors["ff1"], self._weights[f"{p}.linear1.bias"]],
            device=self.device,
        )
        self._execute(self._ff2)
        wp.launch(
            self._residual_norm,
            dim=self.batch * self.sequence,
            inputs=[
                self._tensors["ff2"],
                self._norm1,
                self._weights[f"{p}.linear2.bias"],
                self._weights[f"{p}.norm2.weight"],
                self._weights[f"{p}.norm2.bias"],
                self._norm2,
                wp.float32(self._epsilon),
            ],
            device=self.device,
        )
        return self.output


class EncoderStackPlan:
    """A fixed-buffer stack of post-norm encoder layers."""

    def __init__(
        self, x, valid, weights, prefix, layers, heads, *, epsilon=1.0e-5, cublas=None
    ):
        if layers <= 0:
            raise ValueError("encoder stack requires at least one layer")
        self.layers = []
        current = x
        for index in range(layers):
            layer = EncoderLayerPlan(
                current,
                valid,
                weights,
                f"{prefix}.layers.{index}",
                heads,
                epsilon=epsilon,
                cublas=cublas,
            )
            self.layers.append(layer)
            current = layer.output
        self.output = current

    def execute(self):
        for layer in self.layers:
            layer.execute()
        return self.output


class BidirectionalGQAPlan:
    """Fixed-buffer full or symmetric-sliding grouped-query attention.

    Q/K/V projections stay separate so callers retain the optimized Linear path.
    Inputs use [batch, heads, sequence, head_size] and may have different
    query and key sequence lengths for cross-attention.
    """

    def __init__(
        self,
        query,
        key,
        value,
        *,
        query_valid=None,
        key_valid=None,
        window=None,
        scale=None,
    ):
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("attention Q/K/V must be rank-four arrays")
        if key.shape != value.shape:
            raise ValueError("attention K/V shapes must match")
        if any(item.dtype != query.dtype for item in (key, value)) or any(
            item.device != query.device for item in (key, value)
        ):
            raise ValueError("attention Q/K/V must share dtype and device")
        if query.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("attention requires FP16, BF16, or FP32 tensors")
        batch, query_heads, query_length, head_size = query.shape
        if (
            key.shape[0] != batch
            or key.shape[3] != head_size
            or key.shape[1] <= 0
            or query_heads % key.shape[1]
        ):
            raise ValueError("attention head geometry is incompatible")
        if window is not None and int(window) <= 0:
            raise ValueError("attention window must be positive")
        if window is not None and query_length != key.shape[2]:
            raise ValueError("sliding attention requires equal Q/K sequence lengths")
        self.query = query
        self.key = key
        self.value = value
        self.output = wp.empty_like(query)
        self.query_valid = self._mask(query_valid, batch, query_length, query)
        self.key_valid = self._mask(key_valid, batch, key.shape[2], query)
        self.window = int(window or 0)
        self.scale = float(head_size**-0.5 if scale is None else scale)
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("attention scale must be finite and positive")
        self._block_dim, self._kernel = _get_bidirectional_gqa_attention_kernel(
            head_size, query.dtype
        )

    @staticmethod
    def _mask(mask, batch, sequence, like):
        if mask is None:
            return wp.ones((batch, sequence), dtype=wp.bool, device=like.device)
        if (
            mask.shape != (batch, sequence)
            or mask.dtype != wp.bool
            or mask.device != like.device
        ):
            raise ValueError("attention masks must be matching boolean arrays")
        return mask

    def execute(self):
        wp.launch_tiled(
            self._kernel,
            dim=(self.query.shape[0], self.query.shape[1], self.query.shape[2]),
            inputs=[
                self.query,
                self.key,
                self.value,
                self.query_valid,
                self.key_valid,
                self.output,
                wp.float32(self.scale),
                self.window,
            ],
            block_dim=self._block_dim,
            device=self.query.device,
        )
        return self.output


class FixedKVAttentionPlan(BidirectionalGQAPlan):
    """Graph-safe cross-attention whose projected condition K/V stay fixed."""

    def __init__(
        self, query, key, value, *, query_valid=None, key_valid=None, scale=None
    ):
        super().__init__(
            query,
            key,
            value,
            query_valid=query_valid,
            key_valid=key_valid,
            scale=scale,
        )


class AdaptiveRMSNormPlan:
    """Graph-safe RMSNorm followed by broadcast timestep shift and scale."""

    def __init__(
        self,
        x,
        weight,
        scale_shift_table,
        timestep_modulation,
        *,
        shift_index,
        scale_index,
        epsilon=1.0e-6,
    ):
        if x.ndim != 3 or scale_shift_table.shape != (
            1,
            timestep_modulation.shape[1],
            x.shape[2],
        ):
            raise ValueError("adaptive RMSNorm modulation geometry is incompatible")
        if timestep_modulation.shape[0] != x.shape[0] or (
            timestep_modulation.shape[2] != x.shape[2]
        ):
            raise ValueError("adaptive RMSNorm timestep shape is incompatible")
        if not 0 <= shift_index < timestep_modulation.shape[1] or not (
            0 <= scale_index < timestep_modulation.shape[1]
        ):
            raise IndexError("adaptive RMSNorm modulation index is out of range")
        if any(
            value.dtype != x.dtype or value.device != x.device
            for value in (weight, scale_shift_table, timestep_modulation)
        ):
            raise ValueError("adaptive RMSNorm tensors must share dtype and device")
        self.input = x
        self.scale_shift_table = scale_shift_table
        self.timestep_modulation = timestep_modulation
        self.shift_index = int(shift_index)
        self.scale_index = int(scale_index)
        self._tensors = {"x": x, "weight": weight}
        self._shapes = {"x": x.shape, "weight": weight.shape}
        self._norm = Operation(
            "SimplifiedLayerNormalization",
            ["x", "weight"],
            ["normalized"],
            {"epsilon": float(epsilon)},
        )
        plan_rms_norm(self._norm, self._tensors, self._shapes, x.device)
        self.output = wp.empty_like(x)

    def execute(self):
        execute_operations(
            (self._norm,), self._tensors, self._shapes, self.input.device
        )
        wp.launch(
            _adaptive_rms_modulation_kernel,
            dim=self.output.shape,
            inputs=[
                self._tensors["normalized"],
                self.scale_shift_table,
                self.timestep_modulation,
                self.output,
                self.shift_index,
                self.scale_index,
            ],
            device=self.input.device,
        )
        return self.output


class ModulatedResidualPlan:
    """Graph-safe residual addition with an optional AdaLN gate."""

    def __init__(
        self,
        residual,
        branch,
        *,
        scale_shift_table=None,
        timestep_modulation=None,
        gate_index=0,
    ):
        if (
            residual.shape != branch.shape
            or residual.dtype != branch.dtype
            or (residual.device != branch.device)
        ):
            raise ValueError("residual and branch tensors must match")
        self.residual = residual
        self.branch = branch
        self.use_gate = scale_shift_table is not None
        if self.use_gate:
            if timestep_modulation is None or scale_shift_table.shape != (
                1,
                timestep_modulation.shape[1],
                residual.shape[2],
            ):
                raise ValueError("residual modulation geometry is incompatible")
            if (
                timestep_modulation.shape[0] != residual.shape[0]
                or timestep_modulation.shape[2] != residual.shape[2]
                or not 0 <= gate_index < timestep_modulation.shape[1]
            ):
                raise ValueError("residual timestep modulation is incompatible")
            if any(
                value.dtype != residual.dtype or value.device != residual.device
                for value in (scale_shift_table, timestep_modulation)
            ):
                raise ValueError("residual modulation tensors must match the branch")
        else:
            scale_shift_table = wp.empty(
                (1, 1, residual.shape[2]),
                dtype=residual.dtype,
                device=residual.device,
            )
            timestep_modulation = wp.empty_like(scale_shift_table)
        self.scale_shift_table = scale_shift_table
        self.timestep_modulation = timestep_modulation
        self.gate_index = int(gate_index)
        self.output = wp.empty_like(residual)

    def execute(self):
        wp.launch(
            _modulated_residual_kernel,
            dim=self.output.shape,
            inputs=[
                self.residual,
                self.branch,
                self.scale_shift_table,
                self.timestep_modulation,
                self.output,
                self.gate_index,
                self.use_gate,
            ],
            device=self.residual.device,
        )
        return self.output


class SpatialPatchPackPlan:
    """Graph-safe NCHW to spatial-token patch packing."""

    def __init__(self, x, patch_size):
        if x.ndim != 4 or x.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("spatial patch input must be rank-four FP16/BF16/FP32")
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError("spatial patch size must be positive")
        if x.shape[2] % self.patch_size or x.shape[3] % self.patch_size:
            raise ValueError("spatial dimensions must be divisible by patch size")
        self.input = x
        sequence = (x.shape[2] // self.patch_size) * (x.shape[3] // self.patch_size)
        channels = x.shape[1] * self.patch_size * self.patch_size
        self.output = wp.empty(
            (x.shape[0], sequence, channels), dtype=x.dtype, device=x.device
        )
        self._kernel = _spatial_diffusion_kernels(x.dtype)[0]

    def execute(self):
        wp.launch(
            self._kernel,
            dim=self.output.shape,
            inputs=[self.input, self.output, self.patch_size],
            device=self.input.device,
        )
        return self.output


class SpatialPatchUnpackPlan:
    """Graph-safe spatial-token patches to NCHW unpacking."""

    def __init__(self, x, height, width, patch_size):
        if x.ndim != 3 or x.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("spatial patch input must be rank-three FP16/BF16/FP32")
        self.patch_size = int(patch_size)
        self.height, self.width = int(height), int(width)
        if min(self.patch_size, self.height, self.width) <= 0:
            raise ValueError("spatial patch geometry must be positive")
        patch_area = self.patch_size * self.patch_size
        if (
            self.height % self.patch_size
            or self.width % self.patch_size
            or x.shape[1]
            != (self.height // self.patch_size) * (self.width // self.patch_size)
            or x.shape[2] % patch_area
        ):
            raise ValueError("packed spatial-token geometry is inconsistent")
        self.input = x
        self.output = wp.empty(
            (x.shape[0], x.shape[2] // patch_area, self.height, self.width),
            dtype=x.dtype,
            device=x.device,
        )
        self._kernel = _spatial_diffusion_kernels(x.dtype)[1]

    def execute(self):
        wp.launch(
            self._kernel,
            dim=self.output.shape,
            inputs=[self.input, self.output, self.patch_size],
            device=self.input.device,
        )
        return self.output


class ChannelAffinePlan:
    """Graph-safe per-channel affine transform for NCHW tensors."""

    def __init__(self, x, scale, bias):
        if x.ndim != 4 or x.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("channel affine input must be rank-four FP16/BF16/FP32")
        if (
            scale.shape != (x.shape[1],)
            or bias.shape != scale.shape
            or scale.dtype != wp.float32
            or bias.dtype != wp.float32
            or scale.device != x.device
            or bias.device != x.device
        ):
            raise ValueError(
                "channel affine scale/bias must be matching device FP32 vectors"
            )
        self.input, self.scale, self.bias = x, scale, bias
        self.output = wp.empty_like(x)
        self._kernel = _spatial_diffusion_kernels(x.dtype)[2]

    def execute(self):
        wp.launch(
            self._kernel,
            dim=self.output.shape,
            inputs=[self.input, self.scale, self.bias, self.output],
            device=self.input.device,
        )
        return self.output


def seeded_normal(shape, *, seed=0, dtype=wp.bfloat16, device=None):
    """Create device-resident deterministic independent standard-normal noise."""
    shape = tuple(int(value) for value in shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("normal noise shape must be positive")
    if dtype not in (wp.float16, wp.bfloat16, wp.float32):
        raise TypeError("normal noise requires FP16, BF16, or FP32 output")
    output = wp.empty(shape, dtype=dtype, device=device)
    wp.launch(
        _seeded_normal_kernel(dtype),
        dim=output.size,
        inputs=[output.flatten(), int(seed)],
        device=output.device,
    )
    return output


class TrueCFGPlan:
    """Fused graph-safe true CFG with per-token positive-norm rescaling."""

    def __init__(self, positive, negative, scale=4.0):
        if (
            positive.ndim != 3
            or negative.shape != positive.shape
            or negative.dtype != positive.dtype
            or negative.device != positive.device
        ):
            raise ValueError("CFG predictions must be matching rank-three arrays")
        if positive.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("CFG predictions must use FP16, BF16, or FP32")
        if not math.isfinite(scale):
            raise ValueError("CFG scale must be finite")
        self.positive, self.negative = positive, negative
        self.scale = float(scale)
        self.output = wp.empty_like(positive)
        self._kernel = _true_cfg_kernel(positive.dtype, positive.shape[2])

    def execute(self):
        wp.launch_tiled(
            self._kernel,
            dim=self.positive.shape[0] * self.positive.shape[1],
            inputs=[
                self.positive,
                self.negative,
                self.output,
                wp.float32(self.scale),
            ],
            block_dim=min(128, self.positive.shape[2]),
            device=self.positive.device,
        )
        return self.output


class FlowEulerPlan:
    """In-place graph-safe deterministic flow-matching Euler update."""

    def __init__(self, sample, velocity, sigma, next_sigma):
        if (
            sample.ndim != 3
            or velocity.shape != sample.shape
            or velocity.dtype != sample.dtype
            or velocity.device != sample.device
        ):
            raise ValueError(
                "flow sample and velocity must be matching rank-three arrays"
            )
        if sample.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("flow Euler update requires FP16/BF16/FP32 samples")
        if (
            sigma.shape != (sample.shape[0],)
            or next_sigma.shape != sigma.shape
            or sigma.dtype != wp.float32
            or next_sigma.dtype != wp.float32
            or sigma.device != sample.device
            or next_sigma.device != sample.device
        ):
            raise ValueError("flow sigmas must be matching device FP32 batch vectors")
        self.sample, self.velocity = sample, velocity
        self.sigma, self.next_sigma = sigma, next_sigma
        self._kernel = _spatial_diffusion_kernels(sample.dtype)[3]

    def execute(self):
        wp.launch(
            self._kernel,
            dim=self.sample.shape,
            inputs=[self.sample, self.velocity, self.sigma, self.next_sigma],
            device=self.sample.device,
        )
        return self.sample


def flow_match_euler_schedule(
    steps,
    image_sequence_length,
    *,
    base_sequence_length=256,
    maximum_sequence_length=4096,
    base_shift=0.5,
    maximum_shift=1.15,
    terminal_shift=None,
    time_shift_type="exponential",
):
    """Build the deterministic dynamic-shift FlowMatch Euler sigma schedule."""
    steps = int(steps)
    image_sequence_length = int(image_sequence_length)
    base_sequence_length = int(base_sequence_length)
    maximum_sequence_length = int(maximum_sequence_length)
    if steps <= 0 or image_sequence_length <= 0:
        raise ValueError("flow steps and image sequence length must be positive")
    if (
        base_sequence_length <= 0
        or maximum_sequence_length <= base_sequence_length
        or maximum_shift < base_shift
    ):
        raise ValueError("invalid dynamic flow-shift geometry")
    slope = (maximum_shift - base_shift) / (
        maximum_sequence_length - base_sequence_length
    )
    mu = base_shift + (image_sequence_length - base_sequence_length) * slope
    sigmas = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float64)
    odds = 1.0 / sigmas - 1.0
    if time_shift_type == "exponential":
        factor = math.exp(mu)
    elif time_shift_type == "linear":
        factor = mu
        if factor <= 0.0:
            raise ValueError("linear dynamic flow shift must be positive")
    else:
        raise ValueError("flow time_shift_type must be exponential or linear")
    sigmas = factor / (factor + odds)
    if terminal_shift is not None:
        terminal_shift = float(terminal_shift)
        if not 0.0 <= terminal_shift < 1.0:
            raise ValueError("flow terminal shift must be in [0, 1)")
        scale = (1.0 - sigmas[-1]) / (1.0 - terminal_shift)
        if scale <= 0.0:
            raise ValueError("flow schedule cannot be stretched to its terminal")
        sigmas = 1.0 - (1.0 - sigmas) / scale
    return np.concatenate((sigmas, np.zeros(1))).astype(np.float32)


def resolve_rope_parameters(
    base_parameters: Mapping[str, object],
    scaling: Mapping[str, object] | None,
    native_context: int,
    target_context: int,
) -> dict[str, object]:
    """Resolve an explicit RoPE override and validate its supported context."""
    parameters = dict(base_parameters)
    if scaling is None:
        if target_context > native_context:
            raise ValueError(
                "cache_capacity exceeds the model's native context; enable YaRN explicitly"
            )
        return parameters

    parameters.update(scaling)
    rope_type = str(scaling.get("rope_type", scaling.get("type", "yarn")))
    if rope_type != "yarn":
        raise ValueError("rope_scaling currently supports only YaRN")
    parameters["rope_type"] = rope_type
    original = int(parameters.get("original_max_position_embeddings", native_context))
    if original <= 0:
        raise ValueError("YaRN original_max_position_embeddings must be positive")
    factor = float(parameters.get("factor", max(1.0, target_context / original)))
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("YaRN factor must be at least 1")
    if target_context > original * factor + 1.0e-6 * original:
        raise ValueError(
            "cache_capacity exceeds the context covered by the YaRN factor"
        )
    beta_fast = float(parameters.get("beta_fast", 32.0))
    beta_slow = float(parameters.get("beta_slow", 1.0))
    if (
        not math.isfinite(beta_fast)
        or not math.isfinite(beta_slow)
        or beta_fast <= beta_slow
        or beta_slow <= 0.0
    ):
        raise ValueError("YaRN requires finite beta_fast > beta_slow > 0")
    parameters.update(
        {
            "factor": factor,
            "original_max_position_embeddings": original,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
        }
    )
    return parameters


def rotary_cache_values(
    length: int, rotary_dim: int, parameters: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    """Build FP32 cosine/sine tables for default RoPE or static YaRN."""
    if length <= 0 or rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError(
            "rotary cache length and even rotary dimension must be positive"
        )
    theta = float(parameters.get("rope_theta", 10000.0))
    if theta <= 1.0:
        raise ValueError("rope_theta must be greater than 1")
    dimensions = np.arange(0, rotary_dim, 2, dtype=np.float32)
    position_frequencies = theta ** (dimensions / rotary_dim)
    attention_factor = 1.0
    rope_type = str(parameters.get("rope_type", "default"))
    if rope_type == "default":
        inverse_frequencies = 1.0 / position_frequencies
    elif rope_type == "yarn":
        factor = float(parameters["factor"])
        original = int(parameters["original_max_position_embeddings"])
        beta_fast = float(parameters.get("beta_fast", 32.0))
        beta_slow = float(parameters.get("beta_slow", 1.0))

        def correction_dimension(rotations: float) -> float:
            return (
                rotary_dim
                * math.log(original / (rotations * 2.0 * math.pi))
                / (2.0 * math.log(theta))
            )

        low = correction_dimension(beta_fast)
        high = correction_dimension(beta_slow)
        if bool(parameters.get("truncate", True)):
            low, high = math.floor(low), math.ceil(high)
        low = max(low, 0.0)
        high = min(high, rotary_dim - 1.0)
        if low == high:
            high += 0.001
        ramp = np.clip(
            (np.arange(rotary_dim // 2, dtype=np.float32) - low) / (high - low),
            0.0,
            1.0,
        )
        extrapolation = 1.0 - ramp
        inverse_frequencies = (1.0 / (factor * position_frequencies)) * (
            1.0 - extrapolation
        ) + (1.0 / position_frequencies) * extrapolation
        attention_factor = float(
            parameters.get(
                "attention_factor",
                1.0 if factor <= 1.0 else 0.1 * math.log(factor) + 1.0,
            )
        )
        if not math.isfinite(attention_factor) or attention_factor <= 0.0:
            raise ValueError("YaRN attention_factor must be positive")
    else:
        raise ValueError(f"Unsupported rope_type '{rope_type}'")

    positions = np.arange(length, dtype=np.float32)[:, None]
    angles = positions * inverse_frequencies[None, :]
    return (
        attention_factor * np.cos(angles),
        attention_factor * np.sin(angles),
    )


_OP_DISPATCH: dict[str, Any] = {
    "_BatchNormalizationRelu": _exec_batch_normalization,
    "_RmsNormalization": _exec_rms_normalization,
    "_SwiGLU": _exec_swiglu,
    "Add": _exec_binary,
    "BatchNormalization": _exec_batch_normalization,
    "Cast": _exec_cast,
    "CausalConvWithState": _exec_causal_conv_with_state,
    "Constant": _exec_constant,
    "Div": _exec_binary,
    "Elu": _exec_elu,
    "Gemm": _exec_gemm,
    "Gather": _exec_gather,
    "GatherBlockQuantized": _exec_gather_block_quantized,
    "GroupQueryAttention": _exec_group_query_attention,
    "LSTM": _exec_lstm,
    "Linear": _exec_linear,
    "LinearAttention": _exec_linear_attention,
    "LpNormalization": _exec_lp_normalization,
    "MatMulNBits": _exec_matmul_nbits,
    "Mul": _exec_binary,
    "ReduceMean": _exec_reduce_mean,
    "ReduceMax": _exec_reduce_max,
    "ReduceSum": _exec_reduce_sum,
    "Range": _exec_static,
    "Relu": _exec_unary,
    "Reshape": _exec_reshape,
    "RotaryEmbedding": _exec_rotary_embedding,
    "Shape": _exec_static,
    "Sigmoid": _exec_unary,
    "Sqrt": _exec_unary,
    "Softplus": _exec_unary,
    "SimplifiedLayerNormalization": _exec_simplified_layer_normalization,
    "Squeeze": _exec_squeeze,
    "Sub": _exec_binary,
    "SkipSimplifiedLayerNormalization": _exec_skip_simplified_layer_normalization,
    "Slice": _exec_static,
    "Split": _exec_split,
    "Tanh": _exec_unary,
    "Tile": _exec_tile,
    "Transpose": _exec_transpose,
    "Unsqueeze": _exec_squeeze,
    "Where": _exec_where,
}


def _spatial_pair(value, label):
    if isinstance(value, int):
        result = (value, value)
    else:
        result = tuple(int(item) for item in value)
    if len(result) != 2 or any(item <= 0 for item in result):
        raise ValueError(f"{label} must contain two positive integers")
    return result


def _spatial_padding(value):
    if isinstance(value, int):
        result = (value, value, value, value)
    else:
        result = tuple(int(item) for item in value)
        if len(result) == 2:
            result = (result[0], result[0], result[1], result[1])
    if len(result) != 4 or any(item < 0 for item in result):
        raise ValueError("Conv2D padding must be nonnegative (top,bottom,left,right)")
    return result


def conv2d_output_shape(
    height,
    width,
    kernel_shape,
    *,
    stride=1,
    padding=0,
    dilation=1,
):
    """Return the channels-last Conv2D output height and width."""
    height, width = int(height), int(width)
    kernel_y, kernel_x = _spatial_pair(kernel_shape, "kernel shape")
    stride_y, stride_x = _spatial_pair(stride, "stride")
    dilation_y, dilation_x = _spatial_pair(dilation, "dilation")
    top, bottom, left, right = _spatial_padding(padding)
    if min(height, width) <= 0:
        raise ValueError("Conv2D input dimensions must be positive")
    output_y = (height + top + bottom - dilation_y * (kernel_y - 1) - 1) // stride_y + 1
    output_x = (width + left + right - dilation_x * (kernel_x - 1) - 1) // stride_x + 1
    if min(output_y, output_x) <= 0:
        raise ValueError("Conv2D geometry produces an empty output")
    return output_y, output_x


class Conv2dPlan:
    """Fixed-shape NHWC Conv2D with tensor-core interiors and FP32 accumulation."""

    def __init__(
        self,
        x,
        weight,
        bias=None,
        *,
        stride=1,
        padding=0,
        dilation=1,
        tensor_cores=True,
    ):
        if not isinstance(tensor_cores, bool):
            raise TypeError("tensor_cores must be boolean")
        if x.ndim != 4 or weight.ndim != 4:
            raise ValueError("Conv2D input and weight must be rank four")
        if x.dtype != weight.dtype or x.device != weight.device:
            raise ValueError("Conv2D input and weight must share dtype and device")
        if x.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("Conv2D requires FP16, BF16, or FP32 tensors")
        out_channels, in_channels, kernel_y, kernel_x = weight.shape
        if x.shape[3] != in_channels:
            raise ValueError("Conv2D OIHW weight channels do not match NHWC input")
        if bias is not None and (
            bias.shape != (out_channels,)
            or bias.dtype != x.dtype
            or bias.device != x.device
        ):
            raise ValueError(
                "Conv2D bias must match output channels, dtype, and device"
            )
        self.input = x
        self.stride = _spatial_pair(stride, "stride")
        self.padding = _spatial_padding(padding)
        self.dilation = _spatial_pair(dilation, "dilation")
        output_y, output_x = conv2d_output_shape(
            x.shape[1],
            x.shape[2],
            (kernel_y, kernel_x),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self.output = wp.empty(
            (x.shape[0], output_y, output_x, out_channels),
            dtype=x.dtype,
            device=x.device,
        )
        self.bias = (
            bias
            if bias is not None
            else wp.zeros(out_channels, dtype=x.dtype, device=x.device)
        )
        self._use_bias = bias is not None
        self._kernel = _channels_last_2d_kernels(x.dtype)[0]
        self._use_mma = False
        if (
            tensor_cores
            and x.device.is_cuda
            and self.stride == (1, 1)
            and self.dilation == (1, 1)
            and x.dtype in (wp.float16, wp.bfloat16)
            and in_channels % 16 == 0
            and out_channels % 32 == 0
        ):
            tile_m = 16
            top, _, left, _ = self.padding
            first_x_tile = (left + tile_m - 1) // tile_m
            maximum_start = x.shape[2] - 1 + left - (kernel_x - 1) - (tile_m - 1)
            last_x_tile = min(
                output_x // tile_m,
                maximum_start // tile_m + 1,
            )
            first_y = min(output_y, top)
            last_y = min(output_y, x.shape[1] + top - kernel_y + 1)
            if last_x_tile > first_x_tile and last_y > first_y:
                self._use_mma = True
                self._first_x_tile = first_x_tile
                self._x_tiles = last_x_tile - first_x_tile
                self._interior_x_begin = first_x_tile * tile_m
                self._interior_x_end = last_x_tile * tile_m
                self._interior_y_begin = first_y
                self._interior_y_end = last_y
                self._boundary_count = (
                    first_y * output_x
                    + (output_y - last_y) * output_x
                    + (last_y - first_y)
                    * (self._interior_x_begin + output_x - self._interior_x_end)
                )
                self._packed_weight = wp.empty(
                    (kernel_y, kernel_x, out_channels, in_channels),
                    dtype=x.dtype,
                    device=x.device,
                )
                (
                    self._pack_kernel,
                    self._mma_kernel,
                    self._boundary_kernel,
                ) = _conv2d_mma_kernels(x.dtype, kernel_y, kernel_x, tile_m, 32)
                wp.launch(
                    self._pack_kernel,
                    dim=weight.shape,
                    inputs=[weight, self._packed_weight],
                    device=x.device,
                )
        self.weight = None if self._use_mma else weight

    @property
    def uses_tensor_cores(self):
        return self._use_mma

    def execute(self):
        if self._use_mma:
            wp.launch_tiled(
                self._mma_kernel,
                dim=(
                    self._x_tiles,
                    self._interior_y_end - self._interior_y_begin,
                    self.output.shape[0] * (self.output.shape[3] // 32),
                ),
                inputs=[
                    self.input,
                    self._packed_weight,
                    self.bias,
                    self.output,
                    self._first_x_tile,
                    self._interior_y_begin,
                    self.padding[0],
                    self.padding[2],
                    self._use_bias,
                ],
                block_dim=128,
                device=self.input.device,
            )
            if self._boundary_count:
                wp.launch(
                    self._boundary_kernel,
                    dim=(
                        self.output.shape[0],
                        self._boundary_count,
                        self.output.shape[3],
                    ),
                    inputs=[
                        self.input,
                        self._packed_weight,
                        self.bias,
                        self.output,
                        self._interior_x_begin,
                        self._interior_x_end,
                        self._interior_y_begin,
                        self._interior_y_end,
                        self.padding[0],
                        self.padding[2],
                        self._use_bias,
                    ],
                    device=self.input.device,
                )
            return self.output
        wp.launch(
            self._kernel,
            dim=self.output.shape,
            inputs=[
                self.input,
                self.weight,
                self.bias,
                self.output,
                self.stride[0],
                self.stride[1],
                self.padding[0],
                self.padding[2],
                self.dilation[0],
                self.dilation[1],
                self._use_bias,
            ],
            device=self.input.device,
        )
        return self.output


class ClampPlan:
    """Graph-safe elementwise clamp for contiguous floating tensors."""

    def __init__(self, x, minimum, maximum):
        if x.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("clamp input must use FP16, BF16, or FP32")
        if not x.is_contiguous:
            raise ValueError("clamp input must be contiguous")
        self.minimum, self.maximum = float(minimum), float(maximum)
        if (
            not math.isfinite(self.minimum)
            or not math.isfinite(self.maximum)
            or self.minimum > self.maximum
        ):
            raise ValueError("clamp bounds must be finite and ordered")
        self.input = x
        self.output = wp.empty_like(x)
        self._kernel = _clamp_kernel_for_dtype(x.dtype)

    def execute(self):
        wp.launch(
            self._kernel,
            dim=self.input.size,
            inputs=[
                self.input.flatten(),
                self.output.flatten(),
                wp.float32(self.minimum),
                wp.float32(self.maximum),
            ],
            device=self.input.device,
        )
        return self.output


class OverlapTileBlendPlan:
    """Graph-safe in-place linear overlap blend for cropped NCHW tile canvases."""

    def __init__(
        self,
        tile,
        canvas,
        origin_y,
        origin_x,
        overlap_y,
        overlap_x,
        target_height,
        target_width,
    ):
        if tile.ndim != 4 or canvas.ndim != 4:
            raise ValueError("overlap tile and canvas must be rank-four NCHW arrays")
        if (
            tile.shape[:2] != canvas.shape[:2]
            or tile.dtype != canvas.dtype
            or tile.device != canvas.device
        ):
            raise ValueError("overlap tile and canvas batch/channels must match")
        if tile.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("overlap tile blend requires FP16, BF16, or FP32")
        values = (origin_y, origin_x, overlap_y, overlap_x, target_height, target_width)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("overlap tile blend geometry must use integers")
        if min(origin_y, origin_x, overlap_y, overlap_x) < 0:
            raise ValueError("overlap tile origins and extents must be nonnegative")
        if (
            target_height <= 0
            or target_width <= 0
            or target_height > canvas.shape[2]
            or target_width > canvas.shape[3]
            or origin_y >= target_height
            or origin_x >= target_width
        ):
            raise ValueError("overlap tile target geometry is outside the canvas")
        self.tile, self.canvas = tile, canvas
        self.origin_y, self.origin_x = origin_y, origin_x
        self.overlap_y, self.overlap_x = overlap_y, overlap_x
        self.target_height, self.target_width = target_height, target_width

    @property
    def output(self):
        return self.canvas

    def execute(self):
        wp.launch(
            _overlap_tile_blend_kernel,
            dim=self.tile.shape,
            inputs=[
                self.tile,
                self.canvas,
                self.origin_y,
                self.origin_x,
                self.overlap_y,
                self.overlap_x,
                self.target_height,
                self.target_width,
            ],
            device=self.tile.device,
        )
        return self.canvas


class SpatialRMSNormPlan:
    """Graph-safe channels-last spatial RMSNorm with optional fused SiLU."""

    def __init__(self, x, gamma, *, epsilon=1.0e-12, silu=False):
        if x.ndim != 4 or x.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("spatial RMSNorm input must be rank-four FP16/BF16/FP32")
        if (
            gamma.size != x.shape[3]
            or gamma.dtype != x.dtype
            or gamma.device != x.device
        ):
            raise ValueError("spatial RMSNorm scale must match the input channels")
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("spatial RMSNorm epsilon must be finite and nonnegative")
        self.input = x
        self.gamma = gamma.flatten()
        self.epsilon = float(epsilon)
        self.silu = bool(silu)
        self.output = wp.empty_like(x)
        self._kernel = _spatial_vae_kernels(x.dtype, x.shape[3])[0]

    def execute(self):
        wp.launch_tiled(
            self._kernel,
            dim=self.input.shape[0] * self.input.shape[1] * self.input.shape[2],
            inputs=[
                self.input,
                self.gamma,
                self.output,
                wp.float32(self.epsilon),
                int(self.silu),
            ],
            block_dim=min(128, self.input.shape[3]),
            device=self.input.device,
        )
        return self.output


class NearestUpsample2dPlan:
    """Graph-safe integer nearest-neighbor upsampling for NHWC tensors."""

    def __init__(self, x, scale=2):
        if x.ndim != 4 or x.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("nearest upsample input must be rank-four FP16/BF16/FP32")
        self.scale = int(scale)
        if self.scale <= 0:
            raise ValueError("nearest upsample scale must be positive")
        self.input = x
        self.output = wp.empty(
            (x.shape[0], x.shape[1] * self.scale, x.shape[2] * self.scale, x.shape[3]),
            dtype=x.dtype,
            device=x.device,
        )
        self._kernel = _spatial_vae_kernels(x.dtype, x.shape[3])[1]

    def execute(self):
        wp.launch(
            self._kernel,
            dim=self.output.shape,
            inputs=[self.input, self.output, self.scale],
            device=self.input.device,
        )
        return self.output


class ResidualAddPlan:
    """Graph-safe FP32-accumulating residual addition for NHWC tensors."""

    def __init__(self, left, right):
        if (
            left.ndim != 4
            or right.shape != left.shape
            or right.dtype != left.dtype
            or right.device != left.device
        ):
            raise ValueError("residual operands must be matching rank-four arrays")
        self.left, self.right = left, right
        self.output = wp.empty_like(left)
        self._kernel = _spatial_vae_kernels(left.dtype, left.shape[3])[2]

    def execute(self):
        wp.launch(
            self._kernel,
            dim=self.output.shape,
            inputs=[self.left, self.right, self.output],
            device=self.left.device,
        )
        return self.output


class _SpatialQKVPlan:
    def __init__(self, packed, channels):
        if packed.ndim != 4 or packed.shape[3] != channels * 3:
            raise ValueError("packed spatial QKV geometry is inconsistent")
        self.input = packed
        shape = (packed.shape[0], 1, packed.shape[1] * packed.shape[2], channels)
        self.query = wp.empty(shape, dtype=packed.dtype, device=packed.device)
        self.key = wp.empty_like(self.query)
        self.value = wp.empty_like(self.query)
        self._kernel = _spatial_vae_kernels(packed.dtype, channels)[3]

    def execute(self):
        wp.launch(
            self._kernel,
            dim=(self.query.shape[0], self.query.shape[2], self.query.shape[3]),
            inputs=[self.input, self.query, self.key, self.value],
            device=self.input.device,
        )


class _SpatialAttentionMergePlan:
    def __init__(self, x, height, width):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != height * width:
            raise ValueError("spatial attention output geometry is inconsistent")
        self.input = x
        self.output = wp.empty(
            (x.shape[0], height, width, x.shape[3]),
            dtype=x.dtype,
            device=x.device,
        )
        self._kernel = _spatial_vae_kernels(x.dtype, x.shape[3])[4]

    def execute(self):
        wp.launch(
            self._kernel,
            dim=(self.input.shape[0], self.input.shape[2], self.input.shape[3]),
            inputs=[self.input, self.output],
            device=self.input.device,
        )
        return self.output


class SpatialSelfAttentionPlan:
    """Graph-safe one-head spatial attention composed from shared operators."""

    def __init__(
        self,
        x,
        norm_weight,
        qkv_weight,
        qkv_bias,
        projection_weight,
        projection_bias,
        *,
        epsilon=1.0e-12,
    ):
        channels = x.shape[3]
        if qkv_weight.shape != (channels * 3, channels, 1, 1):
            raise ValueError("spatial attention QKV weight shape is inconsistent")
        if projection_weight.shape != (channels, channels, 1, 1):
            raise ValueError("spatial attention projection shape is inconsistent")
        self.input = x
        self.norm = SpatialRMSNormPlan(x, norm_weight, epsilon=epsilon)
        self.qkv_projection = Conv2dPlan(
            self.norm.output, qkv_weight, qkv_bias, tensor_cores=True
        )
        self.qkv = _SpatialQKVPlan(self.qkv_projection.output, channels)
        self.attention = BidirectionalGQAPlan(
            self.qkv.query, self.qkv.key, self.qkv.value
        )
        self.merge = _SpatialAttentionMergePlan(
            self.attention.output, x.shape[1], x.shape[2]
        )
        self.projection = Conv2dPlan(
            self.merge.output, projection_weight, projection_bias, tensor_cores=True
        )
        self.residual = ResidualAddPlan(x, self.projection.output)
        self.output = self.residual.output

    def execute(self):
        self.norm.execute()
        self.qkv_projection.execute()
        self.qkv.execute()
        self.attention.execute()
        self.merge.execute()
        self.projection.execute()
        return self.residual.execute()


def conv1d_output_length(
    length: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    transposed: bool = False,
    output_padding: int = 0,
) -> int:
    """Return the PyTorch-compatible Conv1D output length."""
    if min(length, kernel_size, stride, dilation) <= 0 or padding < 0:
        raise ValueError("invalid Conv1D geometry")
    if output_padding < 0 or output_padding >= stride:
        raise ValueError("output_padding must be smaller than stride")
    if transposed:
        return (
            (length - 1) * stride
            - 2 * padding
            + dilation * (kernel_size - 1)
            + output_padding
            + 1
        )
    if output_padding:
        raise ValueError("output_padding is valid only for transposed convolution")
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class Conv1dPlan:
    """Fixed-shape, graph-safe channels-last Conv1D or ConvTranspose1D."""

    def __init__(
        self,
        x,
        weight,
        bias=None,
        *,
        stride=1,
        padding=0,
        dilation=1,
        transposed=False,
        output_padding=0,
    ):
        if x.ndim != 3 or weight.ndim != 3:
            raise ValueError("Conv1D input and weight must be rank three")
        if x.dtype != weight.dtype or x.device != weight.device:
            raise ValueError("Conv1D input and weight must share dtype and device")
        if x.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("Conv1D requires FP16, BF16, or FP32 tensors")
        in_channels = weight.shape[0] if transposed else weight.shape[1]
        out_channels = weight.shape[1] if transposed else weight.shape[0]
        if x.shape[2] != in_channels:
            raise ValueError("Conv1D weight channels do not match the input")
        if bias is not None and (
            bias.shape != (out_channels,)
            or bias.dtype != x.dtype
            or bias.device != x.device
        ):
            raise ValueError(
                "Conv1D bias must match output channels, dtype, and device"
            )
        self.input = x
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.transposed = bool(transposed)
        output_length = conv1d_output_length(
            x.shape[1],
            weight.shape[2],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            transposed=self.transposed,
            output_padding=int(output_padding),
        )
        if output_length <= 0:
            raise ValueError("Conv1D geometry produces an empty output")
        self.output = wp.empty(
            (x.shape[0], output_length, out_channels),
            dtype=x.dtype,
            device=x.device,
        )
        self.bias = (
            bias
            if bias is not None
            else wp.zeros(out_channels, dtype=x.dtype, device=x.device)
        )
        self._use_bias = bias is not None
        kernels = _channels_last_1d_kernels(x.dtype)
        self._kernel = kernels[1 if self.transposed else 0]
        self._use_mma = False
        if (
            x.device.is_cuda
            and not self.transposed
            and self.stride == 1
            and x.dtype in (wp.float16, wp.bfloat16)
            and x.shape[2] % 16 == 0
            and out_channels % 32 == 0
        ):
            tile_m = 16
            first_tile = (self.padding + tile_m - 1) // tile_m
            maximum_start = (
                x.shape[1]
                - 1
                + self.padding
                - (weight.shape[2] - 1) * self.dilation
                - (tile_m - 1)
            )
            last_tile = min(
                output_length // tile_m,
                maximum_start // tile_m + 1,
            )
            if last_tile > first_tile:
                self._use_mma = True
                self._first_tile = first_tile
                self._interior_begin = first_tile * tile_m
                self._interior_end = last_tile * tile_m
                self._boundary_count = (
                    self._interior_begin + output_length - self._interior_end
                )
                self._packed_weight = wp.empty(
                    (weight.shape[2], out_channels, in_channels),
                    dtype=x.dtype,
                    device=x.device,
                )
                mma_kernels = _conv1d_mma_kernels(x.dtype, weight.shape[2], tile_m, 32)
                self._pack_kernel, self._mma_kernel, self._boundary_kernel = mma_kernels
                wp.launch(
                    self._pack_kernel,
                    dim=weight.shape,
                    inputs=[weight, self._packed_weight],
                    device=x.device,
                )
        self._use_transpose_mma = False
        if (
            x.device.is_cuda
            and self.transposed
            and self.dilation == 1
            and x.dtype in (wp.float16, wp.bfloat16)
            and x.shape[2] % 16 == 0
            and out_channels % 32 == 0
        ):
            self._use_transpose_mma = True
            residue_rows = (output_length + self.stride - 1) // self.stride
            self._transpose_rows = ((residue_rows + 15) // 16) * 16
            self._packed_weight = wp.empty(
                (weight.shape[2], out_channels, in_channels),
                dtype=x.dtype,
                device=x.device,
            )
            self._padded_input = wp.zeros(
                (x.shape[0], self._transpose_rows + 2, in_channels),
                dtype=x.dtype,
                device=x.device,
            )
            self._transpose_scratch = wp.empty(
                (x.shape[0], self.stride, self._transpose_rows, out_channels),
                dtype=x.dtype,
                device=x.device,
            )
            transpose_kernels = _conv_transpose1d_mma_kernels(
                x.dtype, weight.shape[2], 16, 32
            )
            (
                self._pack_kernel,
                self._pack_input_kernel,
                self._transpose_mma_kernel,
                self._unpack_kernel,
            ) = transpose_kernels
            wp.launch(
                self._pack_kernel,
                dim=weight.shape,
                inputs=[weight, self._packed_weight],
                device=x.device,
            )
        self.weight = None if (self._use_mma or self._use_transpose_mma) else weight

    def execute(self):
        if self._use_transpose_mma:
            wp.launch(
                self._pack_input_kernel,
                dim=self.input.shape,
                inputs=[self.input, self._padded_input],
                device=self.input.device,
            )
            wp.launch_tiled(
                self._transpose_mma_kernel,
                dim=(
                    self._transpose_rows // 16,
                    self.output.shape[2] // 32,
                    self.output.shape[0] * self.stride,
                ),
                inputs=[
                    self._padded_input,
                    self._packed_weight,
                    self.bias,
                    self._transpose_scratch,
                    self.stride,
                    self.padding,
                    self._use_bias,
                ],
                block_dim=128,
                device=self.input.device,
            )
            wp.launch(
                self._unpack_kernel,
                dim=self.output.shape,
                inputs=[self._transpose_scratch, self.output],
                device=self.input.device,
            )
            return self.output
        if self._use_mma:
            wp.launch_tiled(
                self._mma_kernel,
                dim=(
                    (self._interior_end - self._interior_begin) // 16,
                    self.output.shape[2] // 32,
                    self.output.shape[0],
                ),
                inputs=[
                    self.input,
                    self._packed_weight,
                    self.bias,
                    self.output,
                    self._first_tile,
                    self.padding,
                    self.dilation,
                    self._use_bias,
                ],
                block_dim=128,
                device=self.input.device,
            )
            if self._boundary_count:
                wp.launch(
                    self._boundary_kernel,
                    dim=(
                        self.output.shape[0],
                        self._boundary_count,
                        self.output.shape[2],
                    ),
                    inputs=[
                        self.input,
                        self._packed_weight,
                        self.bias,
                        self.output,
                        self._interior_begin,
                        self._interior_end,
                        self.padding,
                        self.dilation,
                        self._use_bias,
                    ],
                    device=self.input.device,
                )
            return self.output
        wp.launch(
            self._kernel,
            dim=self.output.shape,
            inputs=[
                self.input,
                self.weight,
                self.bias,
                self.output,
                self.stride,
                self.padding,
                self.dilation,
                self._use_bias,
            ],
            device=self.input.device,
        )
        return self.output


class Snake1dPlan:
    """Fixed-shape Oobleck Snake activation with channel parameters."""

    def __init__(self, x, alpha, beta, *, logscale=True):
        if x.ndim != 3 or alpha.shape != (x.shape[2],) or beta.shape != alpha.shape:
            raise ValueError("Snake parameters must match the channels-last input")
        if any(
            value.dtype != x.dtype or value.device != x.device
            for value in (alpha, beta)
        ):
            raise ValueError("Snake input and parameters must share dtype and device")
        self.input = x
        self.alpha = alpha
        self.beta = beta
        self.output = wp.empty_like(x)
        self.logscale = bool(logscale)
        self._kernel = _channels_last_1d_kernels(x.dtype)[2]

    def execute(self):
        wp.launch(
            self._kernel,
            dim=self.output.shape,
            inputs=[
                self.input,
                self.alpha,
                self.beta,
                self.output,
                self.logscale,
            ],
            device=self.input.device,
        )
        return self.output


class BiasedLinearPlan:
    """Fixed-buffer dense projection with fused bias and optional activation."""

    def __init__(self, x, weight, bias, *, activation=None, cublas=None):
        if x.ndim < 2 or weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError("biased linear geometry is incompatible")
        if x.shape[-1] != weight.shape[1]:
            raise ValueError("biased linear input width does not match its weight")
        if any(
            value.dtype != x.dtype or value.device != x.device
            for value in (weight, bias)
        ):
            raise ValueError("biased linear tensors must share dtype and device")
        activations = {None: 0, "silu": 1, "gelu_tanh": 2}
        if activation not in activations:
            raise ValueError("activation must be None, 'silu', or 'gelu_tanh'")
        self.input = x
        self.bias = bias
        self.activation = activations[activation]
        rows = int(np.prod(x.shape[:-1]))
        self._tensors = {"x": x.reshape((rows, x.shape[-1])), "weight": weight}
        self._shapes = {name: value.shape for name, value in self._tensors.items()}
        self._operation = Operation("Linear", ["x", "weight"], ["projected"])
        plan_linear(
            self._operation, self._tensors, self._shapes, x.device, cublas=cublas
        )
        self.output = self._tensors["projected"].reshape(
            (*x.shape[:-1], weight.shape[0])
        )

    def execute(self):
        execute_operations(
            (self._operation,), self._tensors, self._shapes, self.input.device
        )
        projected = self._tensors["projected"]
        wp.launch(
            _bias_activation_kernel,
            dim=projected.shape,
            inputs=[projected, self.bias, projected, self.activation],
            device=self.input.device,
        )
        return self.output


class LayerNormPlan:
    """Efficient affine-free last-axis LayerNorm with fixed buffers."""

    def __init__(self, x, *, epsilon=1.0e-6):
        if x.ndim < 2 or x.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("LayerNorm requires a rank-two-or-higher floating tensor")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("LayerNorm epsilon must be finite and positive")
        self.input = x
        self.epsilon = float(epsilon)
        self.output = wp.empty_like(x)
        self.rows = int(np.prod(x.shape[:-1]))
        self.width = x.shape[-1]
        self.tile_width, self.kernel = _get_layer_norm_kernel(self.width, x.dtype)

    def execute(self):
        wp.launch_tiled(
            self.kernel,
            dim=self.rows,
            inputs=[
                self.input.reshape((self.rows, self.width)),
                self.output.reshape((self.rows, self.width)),
                wp.float32(self.epsilon),
            ],
            block_dim=self.tile_width,
            device=self.input.device,
        )
        return self.output


class RMSNormPlan:
    """Fixed-buffer last-axis RMSNorm for arbitrary leading dimensions."""

    def __init__(self, x, weight, *, epsilon=1.0e-6):
        self.input = x
        self._tensors = {"x": x, "weight": weight}
        self._shapes = {name: value.shape for name, value in self._tensors.items()}
        self._operation = Operation(
            "SimplifiedLayerNormalization",
            ["x", "weight"],
            ["normalized"],
            {"epsilon": float(epsilon)},
        )
        plan_rms_norm(self._operation, self._tensors, self._shapes, x.device)
        self.output = self._tensors["normalized"]

    def execute(self):
        execute_operations(
            (self._operation,), self._tensors, self._shapes, self.input.device
        )
        return self.output


class AdaptiveLayerNormPlan:
    """Affine-free LayerNorm followed by batch-broadcast shift and scale."""

    def __init__(self, x, modulation, *, shift_index, scale_index, epsilon=1.0e-6):
        if x.ndim != 3 or modulation.ndim != 3:
            raise ValueError("adaptive LayerNorm requires rank-three tensors")
        if modulation.shape[0] != x.shape[0] or modulation.shape[2] != x.shape[2]:
            raise ValueError("adaptive LayerNorm modulation geometry is incompatible")
        if (
            not 0 <= shift_index < modulation.shape[1]
            or not 0 <= scale_index < modulation.shape[1]
        ):
            raise IndexError("adaptive LayerNorm modulation index is out of range")
        if modulation.dtype != x.dtype or modulation.device != x.device:
            raise ValueError("adaptive LayerNorm tensors must share dtype and device")
        self.norm = LayerNormPlan(x, epsilon=epsilon)
        self.modulation = modulation
        self.shift_index = int(shift_index)
        self.scale_index = int(scale_index)
        self.output = wp.empty_like(x)

    def execute(self):
        self.norm.execute()
        wp.launch(
            _adaptive_layer_norm_kernel,
            dim=self.output.shape,
            inputs=[
                self.norm.output,
                self.modulation,
                self.output,
                self.shift_index,
                self.scale_index,
            ],
            device=self.output.device,
        )
        return self.output


class BroadcastGatedResidualPlan:
    """Residual addition gated by one batch-broadcast modulation vector."""

    def __init__(self, residual, branch, modulation, *, gate_index):
        if residual.shape != branch.shape or residual.ndim != 3:
            raise ValueError(
                "gated residual branches must be matching rank-three tensors"
            )
        if (
            modulation.shape[0] != residual.shape[0]
            or modulation.shape[2] != residual.shape[2]
        ):
            raise ValueError("gated residual modulation geometry is incompatible")
        if not 0 <= gate_index < modulation.shape[1]:
            raise IndexError("gated residual modulation index is out of range")
        if any(
            value.dtype != residual.dtype or value.device != residual.device
            for value in (branch, modulation)
        ):
            raise ValueError("gated residual tensors must share dtype and device")
        self.residual, self.branch, self.modulation = residual, branch, modulation
        self.gate_index = int(gate_index)
        self.output = wp.empty_like(residual)

    def execute(self):
        wp.launch(
            _broadcast_gated_residual_kernel,
            dim=self.output.shape,
            inputs=[
                self.residual,
                self.branch,
                self.modulation,
                self.output,
                self.gate_index,
            ],
            device=self.residual.device,
        )
        return self.output


class SinusoidalEmbeddingPlan:
    """Graph-safe diffusion sinusoidal embedding from device-side FP32 values."""

    def __init__(
        self,
        values,
        width,
        *,
        dtype=wp.bfloat16,
        maximum_period=10000.0,
        scale=1.0,
        frequency_shift=1.0,
        flip_sin_cos=False,
    ):
        if values.ndim != 1 or values.dtype != wp.float32:
            raise TypeError("sinusoidal embedding values must be a rank-one FP32 array")
        if width <= 0 or maximum_period <= 0.0 or width // 2 <= frequency_shift:
            raise ValueError("sinusoidal embedding geometry is invalid")
        self.values = values
        self.maximum_period = float(maximum_period)
        self.scale = float(scale)
        self.frequency_shift = float(frequency_shift)
        self.flip_sin_cos = bool(flip_sin_cos)
        self.output = wp.empty(
            (values.shape[0], int(width)), dtype=dtype, device=values.device
        )

    def execute(self):
        wp.launch(
            _sinusoidal_embedding_kernel,
            dim=self.output.shape,
            inputs=[
                self.values,
                self.output,
                wp.float32(self.maximum_period),
                wp.float32(self.scale),
                wp.float32(self.frequency_shift),
                self.flip_sin_cos,
            ],
            device=self.values.device,
        )
        return self.output


class AttentionHeadsPlan:
    """Convert packed [B,S,H*D] activations into [B,H,S,D]."""

    def __init__(self, x, heads):
        if x.ndim != 3 or heads <= 0 or x.shape[2] % heads:
            raise ValueError("packed attention head geometry is incompatible")
        self.input = x
        self.output = wp.empty(
            (x.shape[0], int(heads), x.shape[1], x.shape[2] // int(heads)),
            dtype=x.dtype,
            device=x.device,
        )

    def execute(self):
        wp.launch(
            _split_attention_heads_kernel,
            dim=self.output.shape,
            inputs=[self.input, self.output],
            device=self.input.device,
        )
        return self.output


class AttentionMergePlan:
    """Convert [B,H,S,D] attention heads into packed [B,S,H*D]."""

    def __init__(self, x):
        if x.ndim != 4:
            raise ValueError("attention merge input must be rank four")
        self.input = x
        self.output = wp.empty(
            (x.shape[0], x.shape[2], x.shape[1] * x.shape[3]),
            dtype=x.dtype,
            device=x.device,
        )

    def execute(self):
        wp.launch(
            _merge_attention_heads_kernel,
            dim=self.input.shape,
            inputs=[self.input, self.output],
            device=self.input.device,
        )
        return self.output


class RotaryCachePlan:
    """Apply adjacent-pair RoPE using one cos/sin row per sequence token."""

    def __init__(self, x, cosine, sine):
        if x.ndim != 4 or x.shape[3] % 2:
            raise ValueError("rotary input must have an even head width")
        if cosine.shape != (x.shape[2], x.shape[3] // 2) or sine.shape != cosine.shape:
            raise ValueError("rotary cache geometry is incompatible")
        if any(
            value.dtype != x.dtype or value.device != x.device
            for value in (cosine, sine)
        ):
            raise ValueError("rotary tensors must share dtype and device")
        self.input, self.cosine, self.sine = x, cosine, sine
        self.output = wp.empty_like(x)

    def execute(self):
        wp.launch(
            _rotary_cache_kernel,
            dim=self.output.shape,
            inputs=[self.input, self.cosine, self.sine, self.output],
            device=self.input.device,
        )
        return self.output


class JointBidirectionalAttentionPlan:
    """Joint non-causal attention over two fixed [B,H,S,D] streams."""

    def __init__(
        self, first_qkv, second_qkv, *, first_valid=None, second_valid=None, scale=None
    ):
        first_q, first_k, first_v = first_qkv
        second_q, second_k, second_v = second_qkv
        if first_q.shape != first_k.shape or first_q.shape != first_v.shape:
            raise ValueError("first joint-attention Q/K/V shapes must match")
        if second_q.shape != second_k.shape or second_q.shape != second_v.shape:
            raise ValueError("second joint-attention Q/K/V shapes must match")
        if (
            first_q.shape[:2] != second_q.shape[:2]
            or first_q.shape[3] != second_q.shape[3]
        ):
            raise ValueError("joint-attention stream geometry is incompatible")
        if any(
            value.dtype != first_q.dtype or value.device != first_q.device
            for value in (first_k, first_v, second_q, second_k, second_v)
        ):
            raise ValueError("joint-attention tensors must share dtype and device")
        batch, heads, first_length, width = first_q.shape
        second_length = second_q.shape[2]
        shape = (batch, heads, first_length + second_length, width)
        self.first_qkv, self.second_qkv = first_qkv, second_qkv
        self.joint_q = wp.empty(shape, dtype=first_q.dtype, device=first_q.device)
        self.joint_k = wp.empty_like(self.joint_q)
        self.joint_v = wp.empty_like(self.joint_q)
        self.first_valid = self._valid(first_valid, batch, first_length, first_q)
        self.second_valid = self._valid(second_valid, batch, second_length, first_q)
        self.joint_valid = wp.empty(
            (batch, first_length + second_length), dtype=wp.bool, device=first_q.device
        )
        self.attention = BidirectionalGQAPlan(
            self.joint_q,
            self.joint_k,
            self.joint_v,
            key_valid=self.joint_valid,
            scale=scale,
        )
        self.first_output = wp.empty_like(first_q)
        self.second_output = wp.empty_like(second_q)

    @staticmethod
    def _valid(value, batch, sequence, like):
        if value is None:
            return wp.ones((batch, sequence), dtype=wp.bool, device=like.device)
        if (
            value.shape != (batch, sequence)
            or value.dtype != wp.bool
            or value.device != like.device
        ):
            raise ValueError("joint-attention validity mask is incompatible")
        return value

    def execute(self):
        for first, second, joint in zip(
            self.first_qkv, self.second_qkv, (self.joint_q, self.joint_k, self.joint_v)
        ):
            wp.launch(
                _concatenate_attention_streams_kernel,
                dim=joint.shape,
                inputs=[first, second, joint],
                device=joint.device,
            )
        wp.launch(
            _concatenate_validity_kernel,
            dim=self.joint_valid.shape,
            inputs=[self.first_valid, self.second_valid, self.joint_valid],
            device=self.joint_valid.device,
        )
        self.attention.execute()
        wp.launch(
            _split_attention_streams_kernel,
            dim=self.attention.output.shape,
            inputs=[self.attention.output, self.first_output, self.second_output],
            device=self.attention.output.device,
        )
        return self.first_output, self.second_output


class SequenceSlicePlan:
    """Graph-safe contiguous rank-three sequence slice."""

    def __init__(self, x, start, length):
        if x.ndim != 3 or start < 0 or length <= 0 or start + length > x.shape[1]:
            raise ValueError("sequence slice geometry is invalid")
        self.input = x
        self.start = int(start)
        self.output = wp.empty(
            (x.shape[0], int(length), x.shape[2]), dtype=x.dtype, device=x.device
        )

    def execute(self):
        wp.launch(
            _sequence_slice_kernel,
            dim=self.output.shape,
            inputs=[self.input, self.output, self.start],
            device=self.input.device,
        )
        return self.output


def multi_axis_rotary_cache_values(coordinates, axes, theta=10000.0):
    """Build adjacent-pair RoPE caches for explicit multi-axis coordinates."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    axes = tuple(int(value) for value in axes)
    if coordinates.ndim != 2 or coordinates.shape[1] != len(axes):
        raise ValueError("rotary coordinates must have one column per axis")
    if not axes or any(value <= 0 or value % 2 for value in axes):
        raise ValueError("rotary axis widths must be positive and even")
    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError("rotary theta must be finite and positive")
    angles = []
    for index, width in enumerate(axes):
        inverse = 1.0 / np.power(
            theta, np.arange(0, width, 2, dtype=np.float32) / width
        )
        angles.append(coordinates[:, index : index + 1] * inverse[None])
    values = np.concatenate(angles, axis=1)
    return np.cos(values).astype(np.float32), np.sin(values).astype(np.float32)


class ElementwiseActivationPlan:
    """Fixed-buffer SiLU or tanh-approximate GELU activation."""

    def __init__(self, x, activation):
        activations = {"silu": 1, "gelu_tanh": 2}
        if x.ndim < 2 or activation not in activations:
            raise ValueError(
                "activation requires a rank-two-or-higher tensor and known kind"
            )
        self.input = x
        self.activation = activations[activation]
        self.output = wp.empty_like(x)
        self.bias = wp.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
        self.rows = int(np.prod(x.shape[:-1]))
        self.width = x.shape[-1]

    def execute(self):
        wp.launch(
            _bias_activation_kernel,
            dim=(self.rows, self.width),
            inputs=[
                self.input.reshape((self.rows, self.width)),
                self.bias,
                self.output.reshape((self.rows, self.width)),
                self.activation,
            ],
            device=self.input.device,
        )
        return self.output
