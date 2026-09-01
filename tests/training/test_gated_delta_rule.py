# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.gated_delta_rule import GatedDeltaRulePlan


def _trace(query, key, value, decay, beta, lengths, past, scale, segment_bounds=None):
    batch, key_heads, sequence, _ = query.shape
    value_heads, value_size = value.shape[1], value.shape[3]
    state = past.astype(np.float32).copy()
    output = np.zeros((batch, value_heads, sequence, value_size), dtype=np.float32)
    transformed = np.zeros_like(output)
    history = np.empty(
        (batch, value_heads, sequence + 1, state.shape[2], value_size),
        dtype=np.float32,
    )
    history[:, :, 0] = state
    for batch_index in range(batch):
        for token in range(sequence):
            if token < int(lengths[batch_index]):
                if (
                    segment_bounds is not None
                    and token > 0
                    and int(segment_bounds[batch_index, token, 0]) == token
                ):
                    state[batch_index] = 0.0
                    history[batch_index, :, token] = 0.0
                for value_head in range(value_heads):
                    key_head = value_head * key_heads // value_heads
                    q = query[batch_index, key_head, token].astype(np.float32)
                    k = key[batch_index, key_head, token].astype(np.float32)
                    v = value[batch_index, value_head, token].astype(np.float32)
                    d = decay[batch_index, token, value_head]
                    bt = beta[batch_index, token, value_head]
                    delta = bt * (v - d * (k @ state[batch_index, value_head]))
                    state[batch_index, value_head] = d * state[
                        batch_index, value_head
                    ] + np.outer(k, delta)
                    transformed[batch_index, value_head, token] = delta
                    output[batch_index, value_head, token] = scale * (
                        q @ state[batch_index, value_head]
                    )
            history[batch_index, :, token + 1] = state[batch_index]
    return output, transformed, state, history


def _reference(
    query, key, value, decay, beta, lengths, past, scale, segment_bounds=None
):
    output, transformed, state, _ = _trace(
        query, key, value, decay, beta, lengths, past, scale, segment_bounds
    )
    return output, transformed, state


def _backward_reference(
    query,
    key,
    value,
    decay,
    beta,
    lengths,
    past,
    scale,
    output_grad,
    present_grad,
    segment_bounds=None,
):
    _, transformed, _, history = _trace(
        query, key, value, decay, beta, lengths, past, scale, segment_bounds
    )
    query_grad = np.zeros_like(query, dtype=np.float32)
    key_grad = np.zeros_like(key, dtype=np.float32)
    value_grad = np.zeros_like(value, dtype=np.float32)
    decay_grad = np.zeros_like(decay, dtype=np.float32)
    beta_grad = np.zeros_like(beta, dtype=np.float32)
    past_grad = np.empty_like(past, dtype=np.float32)
    value_heads = value.shape[1]
    key_heads = key.shape[1]
    for batch_index in range(query.shape[0]):
        for value_head in range(value_heads):
            key_head = value_head * key_heads // value_heads
            state_gradient = present_grad[batch_index, value_head].copy()
            for token in range(int(lengths[batch_index]) - 1, -1, -1):
                q = query[batch_index, key_head, token].astype(np.float32)
                k = key[batch_index, key_head, token].astype(np.float32)
                v = value[batch_index, value_head, token].astype(np.float32)
                previous = history[batch_index, value_head, token]
                current = history[batch_index, value_head, token + 1]
                delta = transformed[batch_index, value_head, token]
                d = decay[batch_index, token, value_head]
                bt = beta[batch_index, token, value_head]
                gradient = output_grad[batch_index, value_head, token]
                query_grad[batch_index, key_head, token] += scale * (current @ gradient)
                state_gradient += scale * np.outer(q, gradient)
                decay_grad[batch_index, token, value_head] += np.sum(
                    state_gradient * previous
                )
                key_grad[batch_index, key_head, token] += state_gradient @ delta
                delta_gradient = k @ state_gradient
                retrieved = k @ previous
                residual = v - d * retrieved
                beta_grad[batch_index, token, value_head] += np.dot(
                    delta_gradient, residual
                )
                value_grad[batch_index, value_head, token] = bt * delta_gradient
                decay_grad[batch_index, token, value_head] += np.dot(
                    delta_gradient, -bt * retrieved
                )
                retrieved_gradient = -bt * d * delta_gradient
                key_grad[batch_index, key_head, token] += previous @ retrieved_gradient
                state_gradient = d * state_gradient + np.outer(k, retrieved_gradient)
                if (
                    segment_bounds is not None
                    and token > 0
                    and int(segment_bounds[batch_index, token, 0]) == token
                ):
                    state_gradient.fill(0.0)
            past_grad[batch_index, value_head] = state_gradient
    return query_grad, key_grad, value_grad, decay_grad, beta_grad, past_grad


def _inputs(device, dtype, *, key_size, value_size, sequence=19):
    rng = np.random.default_rng(137)
    batch, key_heads, value_heads = 2, 1, 2

    def low(shape):
        return wp.array(rng.normal(0.0, 0.15, shape), dtype=dtype, device=device)

    query = low((batch, key_heads, sequence, key_size))
    key = low((batch, key_heads, sequence, key_size))
    value = low((batch, value_heads, sequence, value_size))
    decay = wp.array(
        rng.uniform(0.88, 0.99, (batch, sequence, value_heads)),
        dtype=wp.float32,
        device=device,
    )
    beta = wp.array(
        rng.uniform(0.05, 0.4, (batch, sequence, value_heads)),
        dtype=wp.float32,
        device=device,
    )
    lengths = wp.array([sequence, sequence - 5], dtype=wp.int32, device=device)
    past = wp.array(
        rng.normal(0.0, 0.04, (batch, value_heads, key_size, value_size)),
        dtype=wp.float32,
        device=device,
    )
    return query, key, value, decay, beta, lengths, past


def _numpy(array):
    return array.numpy().astype(np.float32)


def test_gated_delta_rule_cpu_matches_recurrence_and_padding():
    inputs = _inputs("cpu", wp.float32, key_size=4, value_size=3)
    plan = GatedDeltaRulePlan(2, 19, 1, 2, 4, 3, wp.float32, device="cpu")
    output, present = plan.forward(*inputs)
    reference = _reference(
        *(_numpy(value) for value in inputs[:5]),
        inputs[5].numpy(),
        _numpy(inputs[6]),
        plan.scale,
    )
    np.testing.assert_allclose(output.numpy(), reference[0], rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(
        plan.transformed.numpy(), reference[1], rtol=2.0e-5, atol=2.0e-6
    )
    np.testing.assert_allclose(present.numpy(), reference[2], rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_array_equal(output.numpy()[1, :, 14:], 0.0)
    assert not plan.uses_tensor_cores
    rng = np.random.default_rng(139)
    output_gradient = rng.normal(0.0, 0.2, output.shape).astype(np.float32)
    present_gradient = rng.normal(0.0, 0.2, present.shape).astype(np.float32)
    gradients = plan.backward(
        *inputs,
        wp.array(output_gradient, dtype=wp.float32, device="cpu"),
        present_grad=wp.array(present_gradient, dtype=wp.float32, device="cpu"),
    )
    expected_gradients = _backward_reference(
        *(_numpy(value) for value in inputs[:5]),
        inputs[5].numpy(),
        _numpy(inputs[6]),
        plan.scale,
        output_gradient,
        present_gradient,
    )
    for actual, expected in zip(gradients, expected_gradients):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=3.0e-5, atol=3.0e-6)
    accumulated = plan.backward(
        *inputs,
        wp.array(output_gradient, dtype=wp.float32, device="cpu"),
        present_grad=wp.array(present_gradient, dtype=wp.float32, device="cpu"),
        accumulate=True,
    )
    for actual, expected in zip(accumulated, expected_gradients):
        np.testing.assert_allclose(
            actual.numpy(), 2.0 * expected, rtol=3.0e-5, atol=3.0e-6
        )


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_gated_delta_rule_segmented_gpu_matches_isolated_recurrences():
    device = CUDA_DEVICES[0]
    inputs = _inputs(device, wp.bfloat16, key_size=4, value_size=8, sequence=7)
    plan = GatedDeltaRulePlan(2, 7, 1, 2, 4, 8, wp.bfloat16, device=device)
    assert not plan._partials_in_state_workspace
    bounds_np = np.empty((2, 7, 2), dtype=np.int32)
    bounds_np[0, :3] = (0, 3)
    bounds_np[0, 3:] = (3, 7)
    bounds_np[1, :2] = (0, 2)
    bounds_np[1, 2:6] = (2, 6)
    bounds_np[1, 6] = (6, 6)
    bounds = wp.array(bounds_np, dtype=wp.int32, device=device)
    output, present = plan.forward(*inputs, segment_bounds=bounds)
    rng = np.random.default_rng(151)
    output_gradient_np = rng.normal(0.0, 0.1, output.shape).astype(np.float32)
    present_gradient_np = rng.normal(0.0, 0.1, present.shape).astype(np.float32)
    output_gradient = wp.array(output_gradient_np, dtype=wp.float32, device=device)
    present_gradient = wp.array(present_gradient_np, dtype=wp.float32, device=device)
    gradients = plan.backward(
        *inputs,
        output_gradient,
        segment_bounds=bounds,
        present_grad=present_gradient,
    )
    values = tuple(_numpy(value) for value in inputs[:5])
    lengths_np = inputs[5].numpy()
    past_np = _numpy(inputs[6])
    expected = _reference(*values, lengths_np, past_np, plan.scale, bounds_np)
    expected_gradients = _backward_reference(
        *values,
        lengths_np,
        past_np,
        plan.scale,
        output_gradient_np,
        present_gradient_np,
        bounds_np,
    )
    np.testing.assert_allclose(_numpy(output), expected[0], rtol=3.0e-2, atol=5.0e-3)
    np.testing.assert_allclose(_numpy(present), expected[2], rtol=3.0e-2, atol=5.0e-3)
    for actual, reference in zip(gradients, expected_gradients):
        np.testing.assert_allclose(_numpy(actual), reference, rtol=1.0e-1, atol=1.5e-2)
    snapshot = tuple(_numpy(gradient) for gradient in gradients)
    repeated = plan.backward(
        *inputs,
        output_gradient,
        segment_bounds=bounds,
        present_grad=present_gradient,
    )
    for actual, expected in zip(repeated, snapshot):
        np.testing.assert_array_equal(_numpy(actual), expected)

    wp.capture_begin(device=device)
    try:
        plan.backward(
            *inputs,
            output_gradient,
            segment_bounds=bounds,
            present_grad=present_gradient,
        )
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise
    wp.capture_launch(graph)
    captured = tuple(_numpy(gradient) for gradient in gradients)
    wp.capture_launch(graph)
    for actual, expected in zip(gradients, captured):
        np.testing.assert_array_equal(_numpy(actual), expected)


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_gated_delta_rule_segmented_chunkwise_matches_isolated_recurrences():
    device = CUDA_DEVICES[0]
    if device.arch < 80:
        pytest.skip("BF16 tensor cores require SM80")
    sequence = 32
    inputs = _inputs(
        device, wp.bfloat16, key_size=128, value_size=128, sequence=sequence
    )
    plan = GatedDeltaRulePlan(2, sequence, 1, 2, 128, 128, wp.bfloat16, device=device)
    bounds_np = np.empty((2, sequence, 2), dtype=np.int32)
    for row, (length, cuts) in enumerate(((32, (0, 7, 21, 32)), (27, (0, 9, 20, 27)))):
        for start, end in zip(cuts, cuts[1:]):
            bounds_np[row, start:end] = (start, end)
        for token in range(length, sequence):
            bounds_np[row, token] = (token, token)
    bounds = wp.array(bounds_np, dtype=wp.int32, device=device)
    output, present = plan.forward(*inputs, segment_bounds=bounds)
    rng = np.random.default_rng(157)
    output_gradient_np = rng.normal(0.0, 0.1, output.shape).astype(np.float32)
    present_gradient_np = rng.normal(0.0, 0.1, present.shape).astype(np.float32)
    gradients = plan.backward(
        *inputs,
        wp.array(output_gradient_np, dtype=wp.float32, device=device),
        segment_bounds=bounds,
        present_grad=wp.array(present_gradient_np, dtype=wp.float32, device=device),
    )
    values = tuple(_numpy(value) for value in inputs[:5])
    lengths_np = inputs[5].numpy()
    past_np = _numpy(inputs[6])
    expected = _reference(*values, lengths_np, past_np, plan.scale, bounds_np)
    expected_gradients = _backward_reference(
        *values,
        lengths_np,
        past_np,
        plan.scale,
        output_gradient_np,
        present_gradient_np,
        bounds_np,
    )
    np.testing.assert_allclose(_numpy(output), expected[0], rtol=2.0e-2, atol=4.0e-3)
    np.testing.assert_allclose(_numpy(present), expected[2], rtol=2.0e-2, atol=4.0e-3)
    for actual, reference in zip(gradients, expected_gradients):
        np.testing.assert_allclose(_numpy(actual), reference, rtol=1.0e-1, atol=1.5e-2)


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_gated_delta_rule_bfloat16_chunkwise_matches_recurrence_and_graph():
    device = CUDA_DEVICES[0]
    if device.arch < 80:
        pytest.skip("BF16 tensor cores require SM80")
    inputs = _inputs(device, wp.bfloat16, key_size=128, value_size=128, sequence=32)
    plan = GatedDeltaRulePlan(2, 32, 1, 2, 128, 128, wp.bfloat16, device=device)
    assert plan.uses_tensor_cores
    output, present = plan.forward(*inputs)
    wp.synchronize_device(device)
    reference = _reference(
        *(_numpy(value) for value in inputs[:5]),
        inputs[5].numpy(),
        _numpy(inputs[6]),
        plan.scale,
    )
    np.testing.assert_allclose(_numpy(output), reference[0], rtol=2.0e-2, atol=4.0e-3)
    np.testing.assert_allclose(
        _numpy(plan.transformed), reference[1], rtol=2.0e-2, atol=4.0e-3
    )
    np.testing.assert_allclose(_numpy(present), reference[2], rtol=2.0e-2, atol=4.0e-3)

    rng = np.random.default_rng(149)
    output_gradient_np = rng.normal(0.0, 0.1, output.shape).astype(np.float32)
    present_gradient_np = rng.normal(0.0, 0.1, present.shape).astype(np.float32)
    output_gradient = wp.array(output_gradient_np, dtype=wp.float32, device=device)
    present_gradient = wp.array(present_gradient_np, dtype=wp.float32, device=device)
    gradients = plan.backward(*inputs, output_gradient, present_grad=present_gradient)
    expected_gradients = _backward_reference(
        *(_numpy(value) for value in inputs[:5]),
        inputs[5].numpy(),
        _numpy(inputs[6]),
        plan.scale,
        output_gradient_np,
        present_gradient_np,
    )
    for actual, expected in zip(gradients, expected_gradients):
        np.testing.assert_allclose(_numpy(actual), expected, rtol=1.0e-1, atol=1.5e-2)

    gradient_snapshot = tuple(_numpy(gradient) for gradient in gradients)
    repeated_gradients = plan.backward(
        *inputs,
        output_gradient,
        present_grad=present_gradient,
    )
    for actual, expected in zip(repeated_gradients, gradient_snapshot):
        np.testing.assert_array_equal(_numpy(actual), expected)

    def execute():
        plan.forward(*inputs)
        plan.backward(*inputs, output_gradient, present_grad=present_gradient)

    wp.capture_begin(device=device)
    try:
        execute()
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise
    wp.capture_launch(graph)
    first = _numpy(plan.output)
    first_gradient = _numpy(plan.query_grad)
    inputs[2].fill_(0.0)
    wp.capture_launch(graph)
    second = _numpy(plan.output)
    second_gradient = _numpy(plan.query_grad)
    assert np.max(np.abs(first - second)) > 1.0e-3
    assert np.max(np.abs(first_gradient - second_gradient)) > 1.0e-3
    zero_reference = _reference(
        _numpy(inputs[0]),
        _numpy(inputs[1]),
        _numpy(inputs[2]),
        _numpy(inputs[3]),
        _numpy(inputs[4]),
        inputs[5].numpy(),
        _numpy(inputs[6]),
        plan.scale,
    )
    np.testing.assert_allclose(second, zero_reference[0], rtol=2.0e-2, atol=4.0e-3)
