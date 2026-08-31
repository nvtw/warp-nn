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

from warp_nn.runtime.formats.gguf import BlockQuantizedTensor

from warp_nn.runtime.kernels import (
    _GEMM_CONFIG,
    _GEMM_TRANSB_TILED_KERNEL,
    _gather_block_quantized_int8_kernel,
    _get_grouped_decode_linear_kernel,
    _get_bidirectional_gqa_attention_kernel,
    _get_linear_tiled_kernel,
    _get_prefill_mma_linear_kernel,
    _get_partitioned_gqa_attention_kernels,
    _get_linear_vector_kernel,
    _get_q8_grouped_decode_linear_kernel,
    _get_q8_prefill_mma_linear_kernel,
    _get_matmul_int8_q8_kernel,
    _get_quantize_activation_int8_kernel,
    _get_rms_norm_kernels,
    _get_swiglu_kernel,
    _gqa_copy_past_fp16_kernel,
    _gqa_prepare_fp16_kernel,
    _linear_kernel,
    _channels_last_1d_kernels,
    _conv1d_mma_kernels,
    _conv_transpose1d_mma_kernels,
    _adaptive_rms_modulation_kernel,
    _encoder_kernels,
    _modulated_residual_kernel,
    _quantize_activation_int8_kernel,
)
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
        head_size, dtype, partitions, rows_per_group, heads_per_group
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
    sequence_length = output.shape[0]
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
    q8_activation_cache=None,
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
    if isinstance(weight, BlockQuantizedTensor):
        if not device.is_cuda or dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Q8_0 Linear requires CUDA FP16/BF16 activations")
        if inner % 32:
            raise ValueError("Q8_0 Linear requires an inner width divisible by 32")
        output = wp.empty((rows, columns), dtype=dtype, device=device)
        tensors[op.outputs[0]] = output
        shapes[op.outputs[0]] = output.shape
        op.attrs.update({"_rows": rows, "_columns": columns, "_inner": inner})
        blocks = inner // 32
        cache_key = (op.inputs[0], rows, inner, dtype)
        cached_activation = (
            q8_activation_cache.get(cache_key)
            if q8_activation_cache is not None
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
            if q8_activation_cache is not None:
                q8_activation_cache[cache_key] = (
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
    if grouped_decode:
        op.attrs["_kernel"] = _get_grouped_decode_linear_kernel(dtype)
        op.attrs["_grouped_decode_kernel"] = True
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
