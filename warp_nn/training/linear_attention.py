# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""LoRA composition for Qwen Gated DeltaNet linear attention."""

from functools import lru_cache

import warp as wp

from .adapters import LoRAAdapterCollection
from .bridges import merge_heads, split_heads
from .gated_delta import GatedDeltaInputPlan
from .gated_delta_rule import GatedDeltaRulePlan
from .gated_norm import GatedRMSNormPlan


@lru_cache(maxsize=None)
def _sum_four_kernel(dtype: type):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        first: wp.array1d(dtype=DTYPE),
        second: wp.array1d(dtype=DTYPE),
        third: wp.array1d(dtype=DTYPE),
        fourth: wp.array1d(dtype=DTYPE),
        output: wp.array1d(dtype=wp.float32),
    ):
        index = wp.tid()
        output[index] = (
            wp.float32(first[index])
            + wp.float32(second[index])
            + wp.float32(third[index])
            + wp.float32(fourth[index])
        )

    kernel.module.options["enable_backward"] = False
    return kernel


class QwenGatedDeltaLoRAAttentionPlan:
    """Compose Qwen's five recurrent-attention projections and exact reverse."""

    def __init__(
        self,
        adapters: LoRAAdapterCollection,
        *,
        qkv: str,
        gate: str,
        decay: str,
        beta: str,
        output: str,
        inputs: GatedDeltaInputPlan,
        rule: GatedDeltaRulePlan,
        gated_norm: GatedRMSNormPlan,
        conv_weight: wp.array,
        conv_state: wp.array,
        a_log: wp.array,
        dt_bias: wp.array,
        recurrent_state: wp.array,
        norm_weight: wp.array,
    ):
        names = (qkv, gate, decay, beta, output)
        if len(set(names)) != len(names) or any(
            name not in adapters.targets for name in names
        ):
            raise ValueError(
                "Qwen Gated Delta projection names must be distinct adapter targets"
            )
        targets = tuple(adapters.targets[name] for name in names)
        dtype = targets[0].weight.dtype
        device = targets[0].weight.device
        rows = inputs.rows
        hidden = targets[0].weight.shape[1]
        if any(
            target.weight.dtype != dtype
            or target.weight.device != device
            or target.plan.rows != rows
            for target in targets
        ):
            raise ValueError(
                "Qwen Gated Delta targets must share dtype, device, and rows"
            )
        width = inputs.value_heads * inputs.value_size
        expected = (
            (inputs.conv_width, hidden),
            (width, hidden),
            (inputs.value_heads, hidden),
            (inputs.value_heads, hidden),
            (hidden, width),
        )
        actual = tuple(target.weight.shape for target in targets)
        if actual != expected:
            raise ValueError(
                f"Qwen Gated Delta projection shapes must be {expected}, got {actual}"
            )
        if (
            rule.batch != inputs.batch
            or rule.sequence != inputs.sequence
            or rule.key_heads != inputs.key_heads
            or rule.value_heads != inputs.value_heads
            or rule.key_size != inputs.key_size
            or rule.value_size != inputs.value_size
            or rule.dtype != dtype
            or rule.device != device
        ):
            raise ValueError("Gated Delta input and recurrence plans must match")
        norm_rows = inputs.batch * inputs.value_heads * inputs.sequence
        if (
            gated_norm.rows != norm_rows
            or gated_norm.width != inputs.value_size
            or gated_norm.dtype != dtype
            or gated_norm.device != device
        ):
            raise ValueError("Qwen gated normalization geometry must match recurrence")

        frozen = (
            (conv_weight, (inputs.conv_width, inputs.kernel_size), dtype),
            (
                conv_state,
                (
                    inputs.batch,
                    inputs.conv_width,
                    inputs.kernel_size - 1,
                ),
                dtype,
            ),
            (a_log, (inputs.value_heads,), dtype),
            (dt_bias, (inputs.value_heads,), dtype),
            (
                recurrent_state,
                (
                    inputs.batch,
                    inputs.value_heads,
                    inputs.key_size,
                    inputs.value_size,
                ),
                wp.float32,
            ),
            (norm_weight, (inputs.value_size,), dtype),
        )
        for array, shape, expected_dtype in frozen:
            if (
                not isinstance(array, wp.array)
                or array.shape != shape
                or array.dtype != expected_dtype
                or array.device != device
                or not array.is_contiguous
            ):
                raise ValueError(
                    f"Qwen Gated Delta frozen input must be contiguous {shape} "
                    f"{expected_dtype} on {device}"
                )

        self.adapters = adapters
        self.names = names
        self.inputs = inputs
        self.rule = rule
        self.gated_norm = gated_norm
        self.conv_weight = conv_weight
        self.conv_state = conv_state
        self.a_log = a_log
        self.dt_bias = dt_bias
        self.recurrent_state = recurrent_state
        self.norm_weight = norm_weight
        self.device, self.dtype = device, dtype
        self.batch, self.sequence = inputs.batch, inputs.sequence
        self.rows, self.hidden = rows, hidden
        self.value_heads, self.value_size = inputs.value_heads, inputs.value_size
        head_shape = (
            self.batch,
            self.value_heads,
            self.sequence,
            self.value_size,
        )
        packed_shape = (rows, width)
        self.gate_heads = wp.empty(head_shape, dtype=dtype, device=device)
        self.gate_grad_heads = wp.empty(head_shape, dtype=dtype, device=device)
        self.gate_grad_packed = wp.empty(packed_shape, dtype=dtype, device=device)
        self.gated_packed = wp.empty(packed_shape, dtype=dtype, device=device)
        self.input_grad = wp.empty((rows, hidden), dtype=wp.float32, device=device)

    @property
    def output(self) -> wp.array:
        return self.adapters.targets[self.names[4]].plan.output

    @property
    def conv_state_grad(self) -> wp.array:
        return self.inputs.state_grad

    @property
    def recurrent_state_grad(self) -> wp.array:
        return self.rule.past_grad

    def forward(
        self, x, lengths, positions=None, cosine=None, sine=None, *, segment_bounds=None
    ):
        del positions, cosine, sine
        qkv_name, gate_name, decay_name, beta_name, output_name = self.names
        qkv = self.adapters.forward(qkv_name, x)
        gate = self.adapters.forward(gate_name, x)
        decay = self.adapters.forward(decay_name, x)
        beta = self.adapters.forward(beta_name, x)
        query, key, value, decay_values, beta_values = self.inputs.forward(
            qkv,
            decay,
            beta,
            self.conv_weight,
            self.conv_state,
            self.a_log,
            self.dt_bias,
            segment_bounds=segment_bounds,
        )
        core, _ = self.rule.forward(
            query,
            key,
            value,
            decay_values,
            beta_values,
            lengths,
            self.recurrent_state,
            segment_bounds=segment_bounds,
        )
        split_heads(gate, self.gate_heads)
        gated = self.gated_norm.forward(
            core.reshape(
                (
                    self.batch * self.value_heads * self.sequence,
                    self.value_size,
                )
            ),
            self.gate_heads.reshape(
                (
                    self.batch * self.value_heads * self.sequence,
                    self.value_size,
                )
            ),
            self.norm_weight,
        )
        merge_heads(gated.reshape(self.gate_heads.shape), self.gated_packed)
        return self.adapters.forward(output_name, self.gated_packed)

    def backward(
        self,
        x,
        lengths,
        grad_output,
        positions=None,
        cosine=None,
        sine=None,
        *,
        segment_bounds=None,
        accumulate: bool = False,
    ):
        del positions, cosine, sine
        qkv_name, gate_name, decay_name, beta_name, output_name = self.names
        gated_grad = self.adapters.backward(
            output_name,
            self.gated_packed,
            grad_output,
            accumulate=accumulate,
        )
        split_heads(gated_grad, self.gate_grad_heads)
        core_grad, gate_grad = self.gated_norm.backward(
            self.rule.output.reshape(
                (
                    self.batch * self.value_heads * self.sequence,
                    self.value_size,
                )
            ),
            self.gate_heads.reshape(
                (
                    self.batch * self.value_heads * self.sequence,
                    self.value_size,
                )
            ),
            self.norm_weight,
            self.gate_grad_heads.reshape(
                (
                    self.batch * self.value_heads * self.sequence,
                    self.value_size,
                )
            ),
        )
        merge_heads(gate_grad.reshape(self.gate_heads.shape), self.gate_grad_packed)
        q_grad, k_grad, v_grad, decay_grad, beta_grad, _ = self.rule.backward(
            self.inputs.query,
            self.inputs.key,
            self.inputs.value,
            self.inputs.decay,
            self.inputs.beta,
            lengths,
            self.recurrent_state,
            core_grad.reshape(self.rule.output.shape),
            segment_bounds=segment_bounds,
        )
        qkv = self.adapters.targets[qkv_name].plan.output
        decay = self.adapters.targets[decay_name].plan.output
        qkv_grad, decay_projection_grad, beta_projection_grad, _ = self.inputs.backward(
            qkv,
            decay,
            self.conv_weight,
            self.a_log,
            self.dt_bias,
            q_grad,
            k_grad,
            v_grad,
            decay_grad,
            beta_grad,
            segment_bounds=segment_bounds,
        )
        input_gradients = (
            self.adapters.backward(qkv_name, x, qkv_grad, accumulate=accumulate),
            self.adapters.backward(
                gate_name,
                x,
                self.gate_grad_packed,
                accumulate=accumulate,
            ),
            self.adapters.backward(
                decay_name,
                x,
                decay_projection_grad,
                accumulate=accumulate,
            ),
            self.adapters.backward(
                beta_name,
                x,
                beta_projection_grad,
                accumulate=accumulate,
            ),
        )
        wp.launch(
            _sum_four_kernel(self.dtype),
            dim=self.input_grad.size,
            inputs=[gradient.flatten() for gradient in input_gradients],
            outputs=[self.input_grad.flatten()],
            device=self.device,
        )
        return self.input_grad
