# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.gated_delta import GatedDeltaInputPlan


def _reference(qkv, a, b, weight, state, a_log, dt_bias, *, epsilon=1.0e-6):
    batch, sequence, width = qkv.shape
    kernel_size = weight.shape[1]
    convolved = np.empty_like(qkv)
    for batch_index in range(batch):
        for token in range(sequence):
            for channel in range(width):
                total = 0.0
                for kernel in range(kernel_size):
                    source = token + kernel - kernel_size + 1
                    value = (
                        state[batch_index, channel, source + kernel_size - 1]
                        if source < 0
                        else qkv[batch_index, source, channel]
                    )
                    total += value * weight[channel, kernel]
                convolved[batch_index, token, channel] = total / (1.0 + np.exp(-total))
    key_size = 2
    key_width = 2
    query_raw = convolved[:, :, :key_width]
    key_raw = convolved[:, :, key_width : 2 * key_width]
    value = convolved[:, :, 2 * key_width :].reshape(batch, sequence, 2, 2)
    query_norm = np.sqrt(
        np.sum(query_raw * query_raw, axis=-1, keepdims=True) + epsilon
    )
    key_norm = np.sqrt(np.sum(key_raw * key_raw, axis=-1, keepdims=True) + epsilon)
    query = (
        (query_raw / query_norm)
        .reshape(batch, sequence, 1, key_size)
        .transpose(0, 2, 1, 3)
    )
    key = (
        (key_raw / key_norm).reshape(batch, sequence, 1, key_size).transpose(0, 2, 1, 3)
    )
    value = value.transpose(0, 2, 1, 3)
    softplus = np.logaddexp(0.0, a + dt_bias)
    decay = np.exp(-np.exp(a_log) * softplus)
    beta = 1.0 / (1.0 + np.exp(-b))
    return query, key, value, decay, beta


def test_gated_delta_input_forward_and_backward_finite_difference_cpu():
    rng = np.random.default_rng(127)
    batch, sequence, key_heads, value_heads = 1, 3, 1, 2
    key_size = value_size = 2
    kernel_size = 3
    width = 2 * key_heads * key_size + value_heads * value_size
    qkv = rng.normal(0.0, 0.3, (batch, sequence, width)).astype(np.float32)
    a = rng.normal(0.0, 0.2, (batch, sequence, value_heads)).astype(np.float32)
    b = rng.normal(0.0, 0.2, a.shape).astype(np.float32)
    weight = rng.normal(0.0, 0.2, (width, kernel_size)).astype(np.float32)
    state = rng.normal(0.0, 0.2, (batch, width, kernel_size - 1)).astype(np.float32)
    a_log = rng.normal(-0.5, 0.1, value_heads).astype(np.float32)
    dt_bias = rng.normal(0.0, 0.1, value_heads).astype(np.float32)
    gradients = (
        rng.normal(0.0, 0.2, (batch, key_heads, sequence, key_size)).astype(np.float32),
        rng.normal(0.0, 0.2, (batch, key_heads, sequence, key_size)).astype(np.float32),
        rng.normal(0.0, 0.2, (batch, value_heads, sequence, value_size)).astype(
            np.float32
        ),
        rng.normal(0.0, 0.2, (batch, sequence, value_heads)).astype(np.float32),
        rng.normal(0.0, 0.2, (batch, sequence, value_heads)).astype(np.float32),
    )

    def array(value):
        return wp.array(value, dtype=wp.float32, device="cpu")

    plan = GatedDeltaInputPlan(
        batch,
        sequence,
        key_heads,
        value_heads,
        key_size,
        value_size,
        kernel_size,
        wp.float32,
        device="cpu",
    )
    inputs = tuple(
        map(
            array,
            (
                qkv.reshape(sequence, width),
                a.reshape(sequence, value_heads),
                b.reshape(sequence, value_heads),
                weight,
                state,
                a_log,
                dt_bias,
            ),
        )
    )
    outputs = plan.forward(*inputs)
    reference = _reference(qkv, a, b, weight, state, a_log, dt_bias)
    for actual, expected in zip(outputs, reference):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-5, atol=2.0e-6)
    grad_arrays = tuple(map(array, gradients))
    qkv_grad, a_grad, b_grad, state_grad = plan.backward(
        inputs[0], inputs[1], inputs[3], inputs[5], inputs[6], *grad_arrays
    )

    def loss(qkv_value, a_value, b_value, state_value):
        values = _reference(
            qkv_value, a_value, b_value, weight, state_value, a_log, dt_bias
        )
        return sum(
            float(np.sum(value * gradient))
            for value, gradient in zip(values, gradients)
        )

    def finite_difference(value, function, epsilon=1.0e-3):
        result = np.empty_like(value)
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + epsilon
            positive = function()
            value[index] = original - epsilon
            negative = function()
            value[index] = original
            result[index] = (positive - negative) / (2.0 * epsilon)
        return result

    numerical_qkv = finite_difference(qkv, lambda: loss(qkv, a, b, state))
    numerical_a = finite_difference(a, lambda: loss(qkv, a, b, state))
    numerical_b = finite_difference(b, lambda: loss(qkv, a, b, state))
    numerical_state = finite_difference(state, lambda: loss(qkv, a, b, state))
    np.testing.assert_allclose(
        qkv_grad.numpy().reshape(qkv.shape), numerical_qkv, rtol=2.0e-3, atol=2.0e-4
    )
    np.testing.assert_allclose(
        a_grad.numpy().reshape(a.shape), numerical_a, rtol=2.0e-3, atol=2.0e-4
    )
    np.testing.assert_allclose(
        b_grad.numpy().reshape(b.shape), numerical_b, rtol=2.0e-3, atol=2.0e-4
    )
    np.testing.assert_allclose(
        state_grad.numpy(), numerical_state, rtol=2.0e-3, atol=2.0e-4
    )
    with pytest.raises(ValueError, match="backward input"):
        plan.backward(
            inputs[0],
            inputs[1],
            inputs[3],
            inputs[5],
            inputs[6],
            *grad_arrays[:-1],
            wp.empty((1, sequence, value_heads), dtype=wp.float16, device="cpu"),
        )


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


def test_segmented_gated_delta_input_matches_two_isolated_examples_cpu():
    rng = np.random.default_rng(211)
    batch, sequence, split = 1, 4, 2
    key_heads, value_heads, key_size, value_size, kernel_size = 1, 2, 2, 2, 3
    width = 2 * key_heads * key_size + value_heads * value_size

    def array(value):
        return wp.array(value, dtype=wp.float32, device="cpu")

    qkv_np = rng.normal(0.0, 0.2, (sequence, width)).astype(np.float32)
    a_np = rng.normal(0.0, 0.2, (sequence, value_heads)).astype(np.float32)
    b_np = rng.normal(0.0, 0.2, (sequence, value_heads)).astype(np.float32)
    weight = array(rng.normal(0.0, 0.2, (width, kernel_size)))
    state = array(rng.normal(0.0, 0.2, (batch, width, kernel_size - 1)))
    zero_state = wp.zeros_like(state)
    a_log = array(rng.normal(-0.5, 0.1, value_heads))
    dt_bias = array(rng.normal(0.0, 0.1, value_heads))
    qkv, a, b = array(qkv_np), array(a_np), array(b_np)
    bounds_np = np.empty((batch, sequence, 2), dtype=np.int32)
    bounds_np[:, :split] = (0, split)
    bounds_np[:, split:] = (split, sequence)
    bounds = wp.array(bounds_np, dtype=wp.int32, device="cpu")

    packed = GatedDeltaInputPlan(
        batch,
        sequence,
        key_heads,
        value_heads,
        key_size,
        value_size,
        kernel_size,
        wp.float32,
        device="cpu",
    )
    packed_outputs = packed.forward(
        qkv, a, b, weight, state, a_log, dt_bias, segment_bounds=bounds
    )
    isolated_outputs = []
    isolated_plans = []
    for start, end, initial in ((0, split, state), (split, sequence, zero_state)):
        plan = GatedDeltaInputPlan(
            batch,
            end - start,
            key_heads,
            value_heads,
            key_size,
            value_size,
            kernel_size,
            wp.float32,
            device="cpu",
        )
        outputs = plan.forward(
            array(qkv_np[start:end]),
            array(a_np[start:end]),
            array(b_np[start:end]),
            weight,
            initial,
            a_log,
            dt_bias,
        )
        isolated_plans.append(plan)
        isolated_outputs.append(outputs)
    for output_index, actual in enumerate(packed_outputs):
        axis = 2 if output_index < 3 else 1
        expected = np.concatenate(
            [outputs[output_index].numpy() for outputs in isolated_outputs], axis=axis
        )
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-5, atol=2.0e-6)

    gradients = (
        rng.normal(0.0, 0.2, packed.query.shape).astype(np.float32),
        rng.normal(0.0, 0.2, packed.key.shape).astype(np.float32),
        rng.normal(0.0, 0.2, packed.value.shape).astype(np.float32),
        rng.normal(0.0, 0.2, packed.decay.shape).astype(np.float32),
        rng.normal(0.0, 0.2, packed.beta.shape).astype(np.float32),
    )
    packed_gradients = packed.backward(
        qkv,
        a,
        weight,
        a_log,
        dt_bias,
        *(array(value) for value in gradients),
        segment_bounds=bounds,
    )
    isolated_gradients = []
    for index, (start, end) in enumerate(((0, split), (split, sequence))):
        plan = isolated_plans[index]
        sliced = (
            gradients[0][:, :, start:end],
            gradients[1][:, :, start:end],
            gradients[2][:, :, start:end],
            gradients[3][:, start:end],
            gradients[4][:, start:end],
        )
        isolated_gradients.append(
            plan.backward(
                array(qkv_np[start:end]),
                array(a_np[start:end]),
                weight,
                a_log,
                dt_bias,
                *(array(value) for value in sliced),
            )
        )
    for index in range(3):
        expected = np.concatenate(
            [gradients_[index].numpy() for gradients_ in isolated_gradients], axis=0
        )
        np.testing.assert_allclose(
            packed_gradients[index].numpy(), expected, rtol=2.0e-5, atol=2.0e-6
        )
    np.testing.assert_allclose(
        packed_gradients[3].numpy(),
        isolated_gradients[0][3].numpy(),
        rtol=2.0e-5,
        atol=2.0e-6,
    )


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_gated_delta_input_bfloat16_cuda_graph_replay():
    device = CUDA_DEVICES[0]
    rng = np.random.default_rng(131)
    plan = GatedDeltaInputPlan(1, 4, 1, 2, 8, 8, 4, wp.bfloat16, device=device)
    width = plan.conv_width

    def low(shape):
        return wp.array(rng.normal(0.0, 0.2, shape), dtype=wp.bfloat16, device=device)

    qkv = low((4, width))
    a = low((4, 2))
    b = low((4, 2))
    weight = low((width, 4))
    state = low((1, width, 3))
    a_log = low((2,))
    dt_bias = low((2,))
    grads = (
        wp.array(rng.normal(0.0, 0.2, (1, 1, 4, 8)), dtype=wp.float32, device=device),
        wp.array(rng.normal(0.0, 0.2, (1, 1, 4, 8)), dtype=wp.float32, device=device),
        wp.array(rng.normal(0.0, 0.2, (1, 2, 4, 8)), dtype=wp.float32, device=device),
        wp.array(rng.normal(0.0, 0.2, (1, 4, 2)), dtype=wp.float32, device=device),
        wp.array(rng.normal(0.0, 0.2, (1, 4, 2)), dtype=wp.float32, device=device),
    )

    def execute():
        plan.forward(qkv, a, b, weight, state, a_log, dt_bias)
        plan.backward(qkv, a, weight, a_log, dt_bias, *grads)

    execute()
    wp.synchronize_device(device)
    references = tuple(
        array.numpy().copy()
        for array in (
            plan.query,
            plan.decay,
            plan.qkv_grad,
            plan.a_grad,
            plan.state_grad,
        )
    )
    wp.capture_begin(device=device)
    try:
        execute()
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)
    for array, reference in zip(
        (plan.query, plan.decay, plan.qkv_grad, plan.a_grad, plan.state_grad),
        references,
    ):
        np.testing.assert_array_equal(array.numpy(), reference)
