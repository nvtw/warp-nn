# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Fixed-buffer preprocessing and gradients for Gated DeltaNet training."""

from dataclasses import dataclass
from functools import lru_cache
import math

import warp as wp


_STORAGE_DTYPES = (wp.float32, wp.float16, wp.bfloat16)


@dataclass(frozen=True)
class _Kernels:
    convolution: object
    normalize: object
    split_value: object
    prepare_gates: object
    normalize_backward: object
    convolution_backward: object
    state_backward: object
    gates_backward: object


@lru_cache(maxsize=None)
def _kernels(dtype: type, key_size: int, value_size: int, kernel_size: int):
    DTYPE = dtype
    KEY_SIZE = key_size
    VALUE_SIZE = value_size
    KERNEL_SIZE = kernel_size

    @wp.func
    def sigmoid(value: wp.float32):
        if value >= wp.float32(0.0):
            return wp.float32(1.0) / (wp.float32(1.0) + wp.exp(-value))
        exponential = wp.exp(value)
        return exponential / (wp.float32(1.0) + exponential)

    @wp.func
    def square(value: DTYPE):
        result = wp.float32(value)
        return result * result

    @wp.func
    def normalize_value(value: DTYPE, inverse: wp.float32):
        return DTYPE(wp.float32(value) * inverse)

    @wp.func
    def normalize_fp32(value: DTYPE, inverse: wp.float32):
        return wp.float32(value) * inverse

    @wp.func
    def gradient_value(
        value: DTYPE, gradient: wp.float32, inverse: wp.float32, dot: wp.float32
    ):
        normalized = wp.float32(value) * inverse
        return inverse * (gradient - normalized * dot)

    @wp.kernel(enable_backward=False, module="unique")
    def convolution(
        x: wp.array3d(dtype=DTYPE),
        weight: wp.array2d(dtype=DTYPE),
        state: wp.array3d(dtype=DTYPE),
        preactivation: wp.array3d(dtype=wp.float32),
        output: wp.array3d(dtype=DTYPE),
    ):
        batch, token, channel = wp.tid()
        total = wp.float32(0.0)
        for kernel_index in range(KERNEL_SIZE):
            source = token + kernel_index - (KERNEL_SIZE - 1)
            value = (
                wp.float32(state[batch, channel, source + KERNEL_SIZE - 1])
                if source < 0
                else wp.float32(x[batch, source, channel])
            )
            total += value * wp.float32(weight[channel, kernel_index])
        preactivation[batch, token, channel] = total
        output[batch, token, channel] = DTYPE(total * sigmoid(total))

    @wp.kernel(enable_backward=False, module="unique")
    def normalize(
        x: wp.array3d(dtype=DTYPE),
        output: wp.array4d(dtype=DTYPE),
        inverse_norm: wp.array3d(dtype=wp.float32),
        offset: int,
        epsilon: wp.float32,
    ):
        item = wp.tid()
        token = item % output.shape[2]
        head_item = item // output.shape[2]
        head = head_item % output.shape[1]
        batch = head_item // output.shape[1]
        column = offset + head * KEY_SIZE
        values = wp.tile_load(x[batch, token], shape=(KEY_SIZE,), offset=(column,))
        squares = wp.tile_map(square, values)
        norm = wp.sqrt(wp.tile_extract(wp.tile_sum(squares), 0) + epsilon)
        inverse = wp.float32(1.0) / wp.max(norm, wp.float32(1.0e-12))
        inverse_norm[batch, head, token] = inverse
        wp.tile_store(
            output[batch, head, token],
            wp.tile_map(normalize_value, values, inverse),
            offset=(0,),
        )

    @wp.kernel(enable_backward=False, module="unique")
    def split_value(
        x: wp.array3d(dtype=DTYPE), output: wp.array4d(dtype=DTYPE), offset: int
    ):
        batch, head, token, column = wp.tid()
        output[batch, head, token, column] = x[
            batch, token, offset + head * VALUE_SIZE + column
        ]

    @wp.kernel(enable_backward=False, module="unique")
    def prepare_gates(
        a: wp.array3d(dtype=DTYPE),
        b: wp.array3d(dtype=DTYPE),
        a_log: wp.array1d(dtype=DTYPE),
        dt_bias: wp.array1d(dtype=DTYPE),
        a_is_decay: bool,
        decay: wp.array3d(dtype=wp.float32),
        beta: wp.array3d(dtype=wp.float32),
    ):
        batch, token, head = wp.tid()
        beta[batch, token, head] = sigmoid(wp.float32(b[batch, token, head]))
        dt = wp.float32(a[batch, token, head]) + wp.float32(dt_bias[head])
        softplus = wp.max(dt, wp.float32(0.0)) + wp.log(
            wp.float32(1.0) + wp.exp(-wp.abs(dt))
        )
        parameter = wp.float32(a_log[head])
        coefficient = parameter if a_is_decay else -wp.exp(parameter)
        decay[batch, token, head] = wp.exp(coefficient * softplus)

    @wp.kernel(enable_backward=False, module="unique")
    def normalize_backward(
        x: wp.array3d(dtype=DTYPE),
        gradient: wp.array4d(dtype=wp.float32),
        inverse_norm: wp.array3d(dtype=wp.float32),
        output: wp.array4d(dtype=wp.float32),
        offset: int,
    ):
        item = wp.tid()
        token = item % gradient.shape[2]
        head_item = item // gradient.shape[2]
        head = head_item % gradient.shape[1]
        batch = head_item // gradient.shape[1]
        column = offset + head * KEY_SIZE
        values = wp.tile_load(x[batch, token], shape=(KEY_SIZE,), offset=(column,))
        grad = wp.tile_load(gradient[batch, head, token], shape=(KEY_SIZE,))
        inverse = inverse_norm[batch, head, token]
        normalized = wp.tile_map(normalize_fp32, values, inverse)
        dot = wp.tile_extract(wp.tile_sum(wp.tile_map(wp.mul, grad, normalized)), 0)
        wp.tile_store(
            output[batch, head, token],
            wp.tile_map(gradient_value, values, grad, inverse, dot),
        )

    @wp.func
    def conv_gradient(
        channel: int,
        batch: int,
        token: int,
        key_width: int,
        q_grad: wp.array4d(dtype=wp.float32),
        k_grad: wp.array4d(dtype=wp.float32),
        v_grad: wp.array4d(dtype=wp.float32),
    ):
        if channel < key_width:
            head = channel // KEY_SIZE
            return q_grad[batch, head, token, channel % KEY_SIZE]
        if channel < 2 * key_width:
            local = channel - key_width
            head = local // KEY_SIZE
            return k_grad[batch, head, token, local % KEY_SIZE]
        local = channel - 2 * key_width
        head = local // VALUE_SIZE
        return v_grad[batch, head, token, local % VALUE_SIZE]

    @wp.kernel(enable_backward=False, module="unique")
    def convolution_backward(
        weight: wp.array2d(dtype=DTYPE),
        preactivation: wp.array3d(dtype=wp.float32),
        q_grad: wp.array4d(dtype=wp.float32),
        k_grad: wp.array4d(dtype=wp.float32),
        v_grad: wp.array4d(dtype=wp.float32),
        output: wp.array3d(dtype=DTYPE),
    ):
        batch, token, channel = wp.tid()
        total = wp.float32(0.0)
        key_width = q_grad.shape[1] * KEY_SIZE
        for kernel_index in range(KERNEL_SIZE):
            output_token = token + KERNEL_SIZE - 1 - kernel_index
            if output_token < output.shape[1]:
                value = preactivation[batch, output_token, channel]
                activation = sigmoid(value)
                derivative = activation * (
                    wp.float32(1.0) + value * (wp.float32(1.0) - activation)
                )
                total += (
                    conv_gradient(
                        channel,
                        batch,
                        output_token,
                        key_width,
                        q_grad,
                        k_grad,
                        v_grad,
                    )
                    * derivative
                    * wp.float32(weight[channel, kernel_index])
                )
        output[batch, token, channel] = DTYPE(total)

    @wp.kernel(enable_backward=False, module="unique")
    def state_backward(
        weight: wp.array2d(dtype=DTYPE),
        preactivation: wp.array3d(dtype=wp.float32),
        q_grad: wp.array4d(dtype=wp.float32),
        k_grad: wp.array4d(dtype=wp.float32),
        v_grad: wp.array4d(dtype=wp.float32),
        output: wp.array3d(dtype=wp.float32),
    ):
        batch, channel, state_index = wp.tid()
        total = wp.float32(0.0)
        key_width = q_grad.shape[1] * KEY_SIZE
        for token in range(KERNEL_SIZE - 1):
            if token <= state_index and token < preactivation.shape[1]:
                kernel_index = state_index - token
                value = preactivation[batch, token, channel]
                activation = sigmoid(value)
                derivative = activation * (
                    wp.float32(1.0) + value * (wp.float32(1.0) - activation)
                )
                total += (
                    conv_gradient(
                        channel, batch, token, key_width, q_grad, k_grad, v_grad
                    )
                    * derivative
                    * wp.float32(weight[channel, kernel_index])
                )
        output[batch, channel, state_index] = total

    @wp.kernel(enable_backward=False, module="unique")
    def gates_backward(
        a: wp.array3d(dtype=DTYPE),
        a_log: wp.array1d(dtype=DTYPE),
        dt_bias: wp.array1d(dtype=DTYPE),
        a_is_decay: bool,
        decay: wp.array3d(dtype=wp.float32),
        beta: wp.array3d(dtype=wp.float32),
        decay_grad: wp.array3d(dtype=wp.float32),
        beta_grad: wp.array3d(dtype=wp.float32),
        a_grad: wp.array3d(dtype=DTYPE),
        b_grad: wp.array3d(dtype=DTYPE),
    ):
        batch, token, head = wp.tid()
        dt = wp.float32(a[batch, token, head]) + wp.float32(dt_bias[head])
        parameter = wp.float32(a_log[head])
        coefficient = parameter if a_is_decay else -wp.exp(parameter)
        a_grad[batch, token, head] = DTYPE(
            decay_grad[batch, token, head]
            * decay[batch, token, head]
            * coefficient
            * sigmoid(dt)
        )
        beta_value = beta[batch, token, head]
        b_grad[batch, token, head] = DTYPE(
            beta_grad[batch, token, head] * beta_value * (wp.float32(1.0) - beta_value)
        )

    result = _Kernels(
        convolution,
        normalize,
        split_value,
        prepare_gates,
        normalize_backward,
        convolution_backward,
        state_backward,
        gates_backward,
    )
    for kernel in result.__dict__.values():
        kernel.module.options["enable_backward"] = False
    return result


class GatedDeltaInputPlan:
    """Prepare Q/K/V and recurrent gates, with an explicit fixed-buffer reverse."""

    def __init__(
        self,
        batch: int,
        sequence: int,
        key_heads: int,
        value_heads: int,
        key_size: int,
        value_size: int,
        kernel_size: int,
        dtype: type,
        *,
        epsilon: float = 1.0e-6,
        a_is_decay: bool = False,
        device=None,
    ):
        if min(batch, sequence, key_heads, value_heads, key_size, value_size) <= 0:
            raise ValueError("Gated DeltaNet dimensions must be positive")
        if kernel_size < 2:
            raise ValueError("causal convolution kernel size must be at least two")
        if value_heads % key_heads:
            raise ValueError("value heads must be divisible by key heads")
        if dtype not in _STORAGE_DTYPES:
            raise TypeError("Gated DeltaNet storage must use FP32, FP16, or BF16")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("normalization epsilon must be finite and positive")
        self.batch, self.sequence = batch, sequence
        self.key_heads, self.value_heads = key_heads, value_heads
        self.key_size, self.value_size = key_size, value_size
        self.kernel_size, self.dtype = kernel_size, dtype
        self.epsilon, self.a_is_decay = float(epsilon), bool(a_is_decay)
        self.device = wp.get_device(device)
        self.rows = batch * sequence
        self.key_width = key_heads * key_size
        self.conv_width = 2 * self.key_width + value_heads * value_size
        self._kernels = _kernels(dtype, key_size, value_size, kernel_size)
        conv_shape = (batch, sequence, self.conv_width)
        key_shape = (batch, key_heads, sequence, key_size)
        value_shape = (batch, value_heads, sequence, value_size)
        gate_shape = (batch, sequence, value_heads)
        self.preactivation = wp.empty(conv_shape, dtype=wp.float32, device=self.device)
        self.convolved = wp.empty(conv_shape, dtype=dtype, device=self.device)
        self.query = wp.empty(key_shape, dtype=dtype, device=self.device)
        self.key = wp.empty(key_shape, dtype=dtype, device=self.device)
        self.value = wp.empty(value_shape, dtype=dtype, device=self.device)
        self.query_inverse = wp.empty(
            key_shape[:3], dtype=wp.float32, device=self.device
        )
        self.key_inverse = wp.empty(key_shape[:3], dtype=wp.float32, device=self.device)
        self.decay = wp.empty(gate_shape, dtype=wp.float32, device=self.device)
        self.beta = wp.empty(gate_shape, dtype=wp.float32, device=self.device)
        self.query_raw_grad = wp.empty(key_shape, dtype=wp.float32, device=self.device)
        self.key_raw_grad = wp.empty(key_shape, dtype=wp.float32, device=self.device)
        self.qkv_grad = wp.empty(conv_shape, dtype=dtype, device=self.device)
        self.a_grad = wp.empty(gate_shape, dtype=dtype, device=self.device)
        self.b_grad = wp.empty(gate_shape, dtype=dtype, device=self.device)
        self.state_grad = wp.empty(
            (batch, self.conv_width, kernel_size - 1),
            dtype=wp.float32,
            device=self.device,
        )

    def forward(self, qkv, a, b, conv_weight, conv_state, a_log, dt_bias):
        """Run convolution, Q/K normalization, V split, and gate preparation."""
        self._validate(qkv, a, b, conv_weight, conv_state, a_log, dt_bias)
        qkv3 = qkv.reshape((self.batch, self.sequence, self.conv_width))
        gate3 = (self.batch, self.sequence, self.value_heads)
        wp.launch(
            self._kernels.convolution,
            dim=qkv3.shape,
            inputs=[qkv3, conv_weight, conv_state],
            outputs=[self.preactivation, self.convolved],
            device=self.device,
        )
        items = self.batch * self.key_heads * self.sequence
        for output, inverse, offset in (
            (self.query, self.query_inverse, 0),
            (self.key, self.key_inverse, self.key_width),
        ):
            wp.launch_tiled(
                self._kernels.normalize,
                dim=items,
                inputs=[self.convolved, output, inverse, offset, self.epsilon],
                block_dim=min(512, max(32, 1 << (self.key_size - 1).bit_length())),
                device=self.device,
            )
        wp.launch(
            self._kernels.split_value,
            dim=self.value.shape,
            inputs=[self.convolved, self.value, 2 * self.key_width],
            device=self.device,
        )
        wp.launch(
            self._kernels.prepare_gates,
            dim=gate3,
            inputs=[
                a.reshape(gate3),
                b.reshape(gate3),
                a_log,
                dt_bias,
                self.a_is_decay,
            ],
            outputs=[self.decay, self.beta],
            device=self.device,
        )
        return self.query, self.key, self.value, self.decay, self.beta

    def backward(
        self,
        qkv,
        a,
        conv_weight,
        a_log,
        dt_bias,
        query_grad,
        key_grad,
        value_grad,
        decay_grad,
        beta_grad,
    ):
        """Return low-precision QKV/A/B gradients and FP32 prefix-state gradients."""
        self._validate_backward(
            qkv,
            a,
            conv_weight,
            a_log,
            dt_bias,
            query_grad,
            key_grad,
            value_grad,
            decay_grad,
            beta_grad,
        )
        qkv3 = qkv.reshape((self.batch, self.sequence, self.conv_width))
        items = self.batch * self.key_heads * self.sequence
        for gradient, inverse, output, offset in (
            (query_grad, self.query_inverse, self.query_raw_grad, 0),
            (key_grad, self.key_inverse, self.key_raw_grad, self.key_width),
        ):
            wp.launch_tiled(
                self._kernels.normalize_backward,
                dim=items,
                inputs=[self.convolved, gradient, inverse, output, offset],
                block_dim=min(512, max(32, 1 << (self.key_size - 1).bit_length())),
                device=self.device,
            )
        wp.launch(
            self._kernels.convolution_backward,
            dim=qkv3.shape,
            inputs=[
                conv_weight,
                self.preactivation,
                self.query_raw_grad,
                self.key_raw_grad,
                value_grad,
            ],
            outputs=[self.qkv_grad],
            device=self.device,
        )
        wp.launch(
            self._kernels.state_backward,
            dim=self.state_grad.shape,
            inputs=[
                conv_weight,
                self.preactivation,
                self.query_raw_grad,
                self.key_raw_grad,
                value_grad,
            ],
            outputs=[self.state_grad],
            device=self.device,
        )
        gate3 = (self.batch, self.sequence, self.value_heads)
        wp.launch(
            self._kernels.gates_backward,
            dim=gate3,
            inputs=[
                a.reshape(gate3),
                a_log,
                dt_bias,
                self.a_is_decay,
                self.decay,
                self.beta,
                decay_grad,
                beta_grad,
            ],
            outputs=[self.a_grad, self.b_grad],
            device=self.device,
        )
        return (
            self.qkv_grad.reshape((self.rows, self.conv_width)),
            self.a_grad.reshape((self.rows, self.value_heads)),
            self.b_grad.reshape((self.rows, self.value_heads)),
            self.state_grad,
        )

    def _validate(self, qkv, a, b, weight, state, a_log, dt_bias):
        expected = (
            (qkv, (self.rows, self.conv_width)),
            (a, (self.rows, self.value_heads)),
            (b, (self.rows, self.value_heads)),
            (weight, (self.conv_width, self.kernel_size)),
            (state, (self.batch, self.conv_width, self.kernel_size - 1)),
            (a_log, (self.value_heads,)),
            (dt_bias, (self.value_heads,)),
        )
        for value, shape in expected:
            if (
                not isinstance(value, wp.array)
                or value.shape != shape
                or value.dtype != self.dtype
                or value.device != self.device
                or not value.is_contiguous
            ):
                raise ValueError(
                    f"Gated DeltaNet input must be contiguous {shape} {self.dtype} on {self.device}"
                )

    def _validate_backward(
        self,
        qkv,
        a,
        weight,
        a_log,
        dt_bias,
        query_grad,
        key_grad,
        value_grad,
        decay_grad,
        beta_grad,
    ):
        expected = (
            (qkv, (self.rows, self.conv_width), self.dtype),
            (a, (self.rows, self.value_heads), self.dtype),
            (weight, (self.conv_width, self.kernel_size), self.dtype),
            (a_log, (self.value_heads,), self.dtype),
            (dt_bias, (self.value_heads,), self.dtype),
            (query_grad, self.query.shape, wp.float32),
            (key_grad, self.key.shape, wp.float32),
            (value_grad, self.value.shape, wp.float32),
            (decay_grad, self.decay.shape, wp.float32),
            (beta_grad, self.beta.shape, wp.float32),
        )
        for value, shape, dtype in expected:
            if (
                not isinstance(value, wp.array)
                or value.shape != shape
                or value.dtype != dtype
                or value.device != self.device
                or not value.is_contiguous
            ):
                raise ValueError(
                    f"Gated DeltaNet backward input must be contiguous {shape} "
                    f"{dtype} on {self.device}"
                )
