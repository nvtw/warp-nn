# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Exact fixed-buffer Qwen transformer training composition."""

import warp as wp

from .bridges import add_fp32_gradients, cast_from_float32, cast_to_float32
from .mlp import LoRASwiGLUPlan
from .primitives import residual_forward
from .qk import QKTransformPlan


class QwenLoRATransformerBlockPlan:
    """Compose one Qwen pre-norm attention/MLP layer.

    Full-attention plans use Qwen's packed ``[Q, sigmoid-gate]`` projection;
    Gated Delta plans implement the same fixed-buffer forward/backward interface.
    Norm parameters stay frozen and every Linear is trained through LoRA.
    """

    def __init__(
        self,
        attention,
        mlp: LoRASwiGLUPlan,
        *,
        input_norm_weight: wp.array,
        post_attention_norm_weight: wp.array,
        epsilon: float,
        centered_norm_scales: bool,
    ):
        if attention.adapters is not mlp.adapters:
            raise ValueError("Qwen attention and MLP must share one adapter collection")
        if hasattr(attention, "packed_query_gate") and (
            not attention.packed_query_gate or attention.gate_name is not None
        ):
            raise ValueError("Qwen attention requires one packed Q/gate projection")
        if (
            attention.rows != mlp.rows
            or attention.hidden != mlp.hidden
            or attention.dtype != mlp.dtype
            or attention.device != mlp.device
        ):
            raise ValueError("Qwen attention and MLP geometry must match")
        rows, hidden = attention.rows, attention.hidden
        weights = (input_norm_weight, post_attention_norm_weight)
        for weight in weights:
            if (
                not isinstance(weight, wp.array)
                or weight.shape != (hidden,)
                or weight.dtype != attention.dtype
                or weight.device != attention.device
                or not weight.is_contiguous
            ):
                raise ValueError("Qwen norm weights must match hidden dtype and device")

        self.attention = attention
        self.mlp = mlp
        self.adapters = attention.adapters
        self.device = attention.device
        self.dtype = attention.dtype
        self.rows = rows
        self.hidden = hidden
        self.weights = weights
        norm_options = dict(
            batch=1,
            heads=1,
            sequence=rows,
            head_size=hidden,
            dtype=self.dtype,
            rotary_dim=0,
            epsilon=epsilon,
            weight_offset=1.0 if centered_norm_scales else 0.0,
            device=self.device,
        )
        self.input_norm = QKTransformPlan(**norm_options)
        self.post_attention_norm = QKTransformPlan(**norm_options)

        self.shape = (rows, hidden)
        self.shape4 = (1, 1, rows, hidden)
        self.attention_residual = wp.empty(
            self.shape, dtype=self.dtype, device=self.device
        )
        self.output = wp.empty(self.shape, dtype=self.dtype, device=self.device)
        self.grad_output_fp32 = wp.empty(
            self.shape, dtype=wp.float32, device=self.device
        )
        self.mlp_output_grad = wp.empty(
            self.shape, dtype=self.dtype, device=self.device
        )
        self.attention_output_grad = wp.empty(
            self.shape, dtype=self.dtype, device=self.device
        )
        self.residual_grad = wp.empty(self.shape, dtype=wp.float32, device=self.device)
        self.input_grad = wp.empty(self.shape, dtype=wp.float32, device=self.device)

    def forward(
        self, x: wp.array, lengths: wp.array, positions=None, cosine=None, sine=None
    ) -> wp.array:
        """Execute the exact Qwen attention layer."""
        normalized = self.input_norm.forward(
            x.reshape(self.shape4), self.weights[0]
        ).reshape(self.shape)
        attention_output = self.attention.forward(
            normalized, lengths, positions, cosine, sine
        )
        residual_forward(x, attention_output, self.attention_residual)
        mlp_input = self.post_attention_norm.forward(
            self.attention_residual.reshape(self.shape4), self.weights[1]
        ).reshape(self.shape)
        residual_forward(
            self.attention_residual, self.mlp.forward(mlp_input), self.output
        )
        return self.output

    def backward(
        self,
        x: wp.array,
        lengths: wp.array,
        grad_output: wp.array,
        positions=None,
        cosine=None,
        sine=None,
        *,
        accumulate: bool = False,
    ) -> wp.array:
        """Reverse the Qwen layer and return its fixed FP32 input gradient."""
        cast_to_float32(grad_output, self.grad_output_fp32)
        mlp_input_grad = self.mlp.backward(
            self.post_attention_norm.output.reshape(self.shape),
            grad_output,
            accumulate=accumulate,
        )
        residual_from_mlp = self.post_attention_norm.backward(
            self.attention_residual.reshape(self.shape4),
            self.weights[1],
            mlp_input_grad.reshape(self.shape4),
        )
        add_fp32_gradients(
            self.grad_output_fp32,
            residual_from_mlp.reshape(self.shape),
            self.residual_grad,
        )
        cast_from_float32(self.residual_grad, self.attention_output_grad)
        normalized_grad = self.attention.backward(
            self.input_norm.output.reshape(self.shape),
            lengths,
            self.attention_output_grad,
            positions,
            cosine,
            sine,
            accumulate=accumulate,
        )
        attention_input_grad = self.input_norm.backward(
            x.reshape(self.shape4),
            self.weights[0],
            normalized_grad.reshape(self.shape4),
        )
        add_fp32_gradients(
            self.residual_grad,
            attention_input_grad.reshape(self.shape),
            self.input_grad,
        )
        return self.input_grad
