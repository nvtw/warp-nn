# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Exact fixed-buffer Muse Glimmer transformer-layer training composition."""

from functools import lru_cache

import warp as wp

from .bridges import add_fp32_gradients, cast_from_float32, cast_to_float32
from .gqa import GQALoRAAttentionPlan
from .mlp import LoRASwiGLUPlan
from .qk import QKTransformPlan


@lru_cache(maxsize=None)
def _residual_kernel(dtype: type):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        left: wp.array1d(dtype=DTYPE),
        right: wp.array1d(dtype=DTYPE),
        output: wp.array1d(dtype=DTYPE),
    ):
        index = wp.tid()
        output[index] = DTYPE(wp.float32(left[index]) + wp.float32(right[index]))

    kernel.module.options["enable_backward"] = False
    return kernel


class MuseLoRATransformerBlockPlan:
    """Compose one exact Muse sandwich-norm attention/MLP layer.

    Norm weights stay frozen, while every Linear in the supplied attention and
    MLP plans is trained through LoRA. All saved state and gradient bridges are
    fixed at construction and the complete forward/backward is graph-capturable.
    """

    def __init__(
        self,
        attention: GQALoRAAttentionPlan,
        mlp: LoRASwiGLUPlan,
        *,
        input_norm_weight: wp.array,
        post_attention_norm_weight: wp.array,
        feedforward_norm_weight: wp.array,
        post_feedforward_norm_weight: wp.array,
        rms_epsilon: float,
        post_epsilon: float,
        centered_norm_scales: bool,
    ):
        if attention.adapters is not mlp.adapters:
            raise ValueError("Muse attention and MLP must share one adapter collection")
        if (
            attention.rows != mlp.rows
            or attention.hidden != mlp.hidden
            or attention.dtype != mlp.dtype
            or attention.device != mlp.device
        ):
            raise ValueError("Muse attention and MLP geometry must match")
        rows, hidden = attention.rows, attention.hidden
        weights = (
            input_norm_weight,
            post_attention_norm_weight,
            feedforward_norm_weight,
            post_feedforward_norm_weight,
        )
        for weight in weights:
            if (
                not isinstance(weight, wp.array)
                or weight.shape != (hidden,)
                or weight.dtype != attention.dtype
                or weight.device != attention.device
                or not weight.is_contiguous
            ):
                raise ValueError("Muse norm weights must match hidden dtype and device")

        self.attention = attention
        self.mlp = mlp
        self.adapters = attention.adapters
        self.device = attention.device
        self.dtype = attention.dtype
        self.rows = rows
        self.hidden = hidden
        self.weights = weights
        offset = 1.0 if centered_norm_scales else 0.0
        norm_options = dict(
            batch=1,
            heads=1,
            sequence=rows,
            head_size=hidden,
            dtype=self.dtype,
            rotary_dim=0,
            weight_offset=offset,
            device=self.device,
        )
        self.input_norm = QKTransformPlan(epsilon=rms_epsilon, **norm_options)
        self.post_attention_norm = QKTransformPlan(epsilon=post_epsilon, **norm_options)
        self.feedforward_norm = QKTransformPlan(epsilon=rms_epsilon, **norm_options)
        self.post_feedforward_norm = QKTransformPlan(
            epsilon=post_epsilon, **norm_options
        )

        shape = (rows, hidden)
        shape4 = (1, 1, rows, hidden)
        self.shape = shape
        self.shape4 = shape4
        self.attention_residual = wp.empty(shape, dtype=self.dtype, device=self.device)
        self.output = wp.empty(shape, dtype=self.dtype, device=self.device)
        self.grad_output_fp32 = wp.empty(shape, dtype=wp.float32, device=self.device)
        self.mlp_output_grad = wp.empty(shape, dtype=self.dtype, device=self.device)
        self.attention_output_grad = wp.empty(
            shape, dtype=self.dtype, device=self.device
        )
        self.residual_grad = wp.empty(shape, dtype=wp.float32, device=self.device)
        self.input_grad = wp.empty(shape, dtype=wp.float32, device=self.device)

    def forward(
        self, x: wp.array, lengths: wp.array, positions=None, cosine=None, sine=None
    ) -> wp.array:
        """Execute the exact Muse layer into the fixed output buffer."""
        input_norm = self.input_norm.forward(
            x.reshape(self.shape4), self.weights[0]
        ).reshape(self.shape)
        attention_output = self.attention.forward(
            input_norm, lengths, positions, cosine, sine
        )
        post_attention = self.post_attention_norm.forward(
            attention_output.reshape(self.shape4), self.weights[1]
        ).reshape(self.shape)
        wp.launch(
            _residual_kernel(self.dtype),
            dim=self.output.size,
            inputs=[x.flatten(), post_attention.flatten()],
            outputs=[self.attention_residual.flatten()],
            device=self.device,
        )
        feedforward_input = self.feedforward_norm.forward(
            self.attention_residual.reshape(self.shape4), self.weights[2]
        ).reshape(self.shape)
        mlp_output = self.mlp.forward(feedforward_input)
        post_feedforward = self.post_feedforward_norm.forward(
            mlp_output.reshape(self.shape4), self.weights[3]
        ).reshape(self.shape)
        wp.launch(
            _residual_kernel(self.dtype),
            dim=self.output.size,
            inputs=[self.attention_residual.flatten(), post_feedforward.flatten()],
            outputs=[self.output.flatten()],
            device=self.device,
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
        """Reverse the exact Muse layer and return fixed FP32 input gradients."""
        cast_to_float32(grad_output, self.grad_output_fp32)
        mlp_output_grad = self.post_feedforward_norm.backward(
            self.mlp.output.reshape(self.shape4),
            self.weights[3],
            self.grad_output_fp32.reshape(self.shape4),
        )
        cast_from_float32(mlp_output_grad.reshape(self.shape), self.mlp_output_grad)
        feedforward_grad = self.mlp.backward(
            self.feedforward_norm.output.reshape(self.shape),
            self.mlp_output_grad,
            accumulate=accumulate,
        )
        feedforward_residual_grad = self.feedforward_norm.backward(
            self.attention_residual.reshape(self.shape4),
            self.weights[2],
            feedforward_grad.reshape(self.shape4),
        )
        add_fp32_gradients(
            self.grad_output_fp32,
            feedforward_residual_grad.reshape(self.shape),
            self.residual_grad,
        )
        attention_output_grad = self.post_attention_norm.backward(
            self.attention.output.reshape(self.shape4),
            self.weights[1],
            self.residual_grad.reshape(self.shape4),
        )
        cast_from_float32(
            attention_output_grad.reshape(self.shape), self.attention_output_grad
        )
        normalized_input_grad = self.attention.backward(
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
            normalized_input_grad.reshape(self.shape4),
        )
        add_fp32_gradients(
            self.residual_grad,
            attention_input_grad.reshape(self.shape),
            self.input_grad,
        )
        return self.input_grad
