# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Format-independent operator execution for preplanned inference graphs."""

from __future__ import annotations

from typing import Any, Iterable

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from warp_nn.runtime.kernels import (
    _GEMM_CONFIG,
    _GEMM_TRANSB_TILED_KERNEL,
    _gather_block_quantized_int8_kernel,
    _gqa_copy_past_fp16_kernel,
    _gqa_prepare_fp16_kernel,
    _quantize_activation_int8_kernel,
)
from warp_nn.utils.ops import resolve_dim


@dataclass
class Operation:
    """A preplanned operation whose private attributes hold launch state."""

    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


def execute_operations(
    operations: Iterable[Operation], tensors: dict[str, wp.array], shapes: dict[str, tuple[int, ...]], device
) -> None:
    """Launch a preplanned operation sequence on the current Warp stream."""
    for operation in operations:
        try:
            dispatch = _OP_DISPATCH[operation.op_type]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported operation '{operation.op_type}'") from exc
        dispatch(operation, tensors, shapes, device)


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
    wp.launch(op.attrs["_kernel"], dim=shape, inputs=[x.reshape(shape), out.reshape(shape), alpha], device=device)


def _exec_unary(op, tensors, shapes, device):
    operation = {"Relu": 0, "Tanh": 1, "Sqrt": 2, "Sigmoid": 3, "Softplus": 4}[op.op_type]
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
        inputs=[tensors[op.inputs[0]].reshape((size,)), tensors[op.outputs[0]].reshape((size,))],
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
            inputs=[weights, tensors[op.inputs[2]], zero_points, dequantized, has_zero_points],
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
            tensors[op.inputs[0]].reshape((batch * sequence_length, query_heads * key_size)),
            tensors[op.inputs[1]].reshape((batch * sequence_length, key_heads * key_size)),
            tensors[op.inputs[2]].reshape((batch * sequence_length, value_heads * value_size)),
            past.reshape((batch * value_heads * key_size, value_size)),
            decay.reshape((int(np.prod(decay.shape[:-1])), decay.shape[-1])),
            beta.reshape((int(np.prod(beta.shape[:-1])), beta.shape[-1])),
            tensors[op.outputs[0]].reshape((batch * sequence_length, max(query_heads, value_heads) * value_size)),
            tensors[op.outputs[1]].reshape((batch * value_heads * key_size, value_size)),
            sequence_length,
            query_heads,
            key_heads,
            value_heads,
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
            inputs=[tensors[op.inputs[3]], tensors[op.inputs[4]], present_key, present_value],
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
            op.attrs["_rotated_query"].reshape((batch * query_heads * sequence_length, head_size)),
            present_key.reshape((batch * kv_heads * total_length, head_size)),
            present_value.reshape((batch * kv_heads * total_length, head_size)),
            tensors[op.inputs[5]],
            tensors[op.outputs[0]].reshape((batch * sequence_length, query_heads * head_size)),
            query_heads,
            kv_heads,
            sequence_length,
            total_length,
            float(op.attrs.get("scale", head_size**-0.5)),
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
            cache["h_prev_zero"] = wp.zeros((batch, hidden_size), dtype=cache["dtype"], device=device)
        h_prev = cache["h_prev_zero"]
    if len(op.inputs) > 6 and op.inputs[6] and op.inputs[6] in tensors:
        c_prev = tensors[op.inputs[6]].reshape((batch, hidden_size))
    else:
        if "c_prev_zero" not in cache:
            cache["c_prev_zero"] = wp.zeros((batch, hidden_size), dtype=cache["dtype"], device=device)
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
