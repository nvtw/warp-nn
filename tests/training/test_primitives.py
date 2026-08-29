# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training import CrossEntropyPlan, EmbeddingPlan, TransformerPrimitivePlan


@wp.kernel
def _sum_2d(values: wp.array2d(dtype=wp.float32), loss: wp.array1d(dtype=wp.float32)):
    row, column = wp.tid()
    wp.atomic_add(loss, 0, values[row, column])


def _array(values, *, requires_grad=False, dtype=wp.float32):
    return wp.array(
        np.asarray(values), dtype=dtype, device="cpu", requires_grad=requires_grad
    )


def _sum_gradient(operation, differentiated):
    loss = wp.zeros(1, dtype=wp.float32, device="cpu", requires_grad=True)
    tape = wp.Tape()
    with tape:
        output = operation()
        wp.launch(
            _sum_2d, dim=output.shape, inputs=[output], outputs=[loss], device="cpu"
        )
    tape.backward(loss)
    return output.numpy(), [array.grad.numpy() for array in differentiated]


def test_residual_and_channel_scale_are_differentiable():
    x_values = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]], dtype=np.float32)
    residual_values = np.array([[1.0, 2.0, -3.0], [-1.0, 0.5, 4.0]], dtype=np.float32)
    plan = TransformerPrimitivePlan(2, 3, rotary_dim=2, device="cpu")

    x = _array(x_values, requires_grad=True)
    residual = _array(residual_values, requires_grad=True)
    output, gradients = _sum_gradient(lambda: plan.residual(x, residual), [x, residual])
    np.testing.assert_allclose(output, x_values + residual_values)
    np.testing.assert_allclose(gradients[0], np.ones_like(x_values))
    np.testing.assert_allclose(gradients[1], np.ones_like(residual_values))

    x = _array(x_values, requires_grad=True)
    scale_values = np.array([0.25, -2.0, 3.0], dtype=np.float32)
    scale = _array(scale_values, requires_grad=True)
    output, gradients = _sum_gradient(lambda: plan.scale(x, scale), [x, scale])
    np.testing.assert_allclose(output, x_values * scale_values)
    np.testing.assert_allclose(
        gradients[0], np.broadcast_to(scale_values, x_values.shape)
    )
    np.testing.assert_allclose(gradients[1], x_values.sum(axis=0))


def test_swiglu_and_sigmoid_gate_are_differentiable():
    x_values = np.array([[-2.0, -0.25, 0.5], [1.0, 2.0, 4.0]], dtype=np.float32)
    other_values = np.array([[0.5, 2.0, -1.0], [3.0, -0.5, 0.25]], dtype=np.float32)
    sigmoid = 1.0 / (1.0 + np.exp(-x_values))
    plan = TransformerPrimitivePlan(2, 3, rotary_dim=2, device="cpu")

    gate = _array(x_values, requires_grad=True)
    up = _array(other_values, requires_grad=True)
    output, gradients = _sum_gradient(lambda: plan.swiglu(gate, up), [gate, up])
    np.testing.assert_allclose(output, x_values * sigmoid * other_values, rtol=1.0e-6)
    np.testing.assert_allclose(
        gradients[0],
        other_values * (sigmoid + x_values * sigmoid * (1.0 - sigmoid)),
        rtol=1.0e-5,
    )
    np.testing.assert_allclose(gradients[1], x_values * sigmoid, rtol=1.0e-6)

    values = _array(other_values, requires_grad=True)
    gate = _array(x_values, requires_grad=True)
    output, gradients = _sum_gradient(
        lambda: plan.sigmoid_gate(values, gate), [values, gate]
    )
    np.testing.assert_allclose(output, other_values * sigmoid, rtol=1.0e-6)
    np.testing.assert_allclose(gradients[0], sigmoid, rtol=1.0e-6)
    np.testing.assert_allclose(
        gradients[1], other_values * sigmoid * (1.0 - sigmoid), rtol=1.0e-5
    )


def test_rope_and_rms_norm_are_differentiable():
    x_values = np.array(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -3.0]], dtype=np.float32
    )
    angles = np.array([[0.2, -0.4], [0.7, 0.1]], dtype=np.float32)
    cosine_values = np.cos(angles).astype(np.float32)
    sine_values = np.sin(angles).astype(np.float32)
    plan = TransformerPrimitivePlan(2, 4, epsilon=1.0e-5, device="cpu")

    x = _array(x_values, requires_grad=True)
    cosine = _array(cosine_values)
    sine = _array(sine_values)
    output, gradients = _sum_gradient(lambda: plan.rope(x, cosine, sine), [x])
    expected = np.concatenate(
        [
            x_values[:, :2] * cosine_values - x_values[:, 2:] * sine_values,
            x_values[:, 2:] * cosine_values + x_values[:, :2] * sine_values,
        ],
        axis=1,
    )
    expected_gradient = np.concatenate(
        [cosine_values + sine_values, cosine_values - sine_values], axis=1
    )
    np.testing.assert_allclose(output, expected, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        gradients[0], expected_gradient, rtol=1.0e-6, atol=1.0e-6
    )

    weight_values = np.array([0.5, 1.0, -1.5, 2.0], dtype=np.float32)
    x = _array(x_values, requires_grad=True)
    weight = _array(weight_values, requires_grad=True)
    output, gradients = _sum_gradient(lambda: plan.rms_norm(x, weight), [x, weight])
    mean_square = np.mean(x_values * x_values, axis=1, keepdims=True)
    inverse_rms = 1.0 / np.sqrt(mean_square + 1.0e-5)
    expected = x_values * inverse_rms * weight_values
    weighted_sum = np.sum(x_values * weight_values, axis=1, keepdims=True)
    expected_x_gradient = weight_values * inverse_rms - x_values * weighted_sum * (
        inverse_rms**3
    ) / np.float32(x_values.shape[1])
    np.testing.assert_allclose(output, expected, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(
        gradients[0], expected_x_gradient, rtol=2.0e-5, atol=2.0e-6
    )
    np.testing.assert_allclose(
        gradients[1], np.sum(x_values * inverse_rms, axis=0), rtol=1.0e-5
    )


def test_embedding_gather_accumulates_repeated_token_gradients():
    table_values = np.arange(20, dtype=np.float32).reshape(5, 4) / 10.0
    token_values = np.array([3, 1, 3], dtype=np.int32)
    table = _array(table_values, requires_grad=True)
    token_ids = _array(token_values, dtype=wp.int32)
    plan = EmbeddingPlan(3, 5, 4, device="cpu")
    output, gradients = _sum_gradient(lambda: plan(table, token_ids), [table])
    np.testing.assert_allclose(output, table_values[token_values])
    expected_gradient = np.zeros_like(table_values)
    expected_gradient[1] = 1.0
    expected_gradient[3] = 2.0
    np.testing.assert_allclose(gradients[0], expected_gradient)


@pytest.mark.parametrize("dtype,atol", [(wp.float16, 4.0e-3), (wp.bfloat16, 3.0e-2)])
def test_low_precision_plans_forward_and_tape(dtype, atol):
    x_values = np.array(
        [[1.0, -2.0, 0.5, 3.0], [-0.25, 1.5, 2.0, -1.0]], dtype=np.float32
    )
    other_values = np.array(
        [[0.5, 1.0, -1.0, 2.0], [3.0, -0.5, 0.25, 1.0]], dtype=np.float32
    )
    scale_values = np.array([0.5, -1.0, 1.5, 2.0], dtype=np.float32)
    angles = np.array([[0.2, -0.4], [0.7, 0.1]], dtype=np.float32)
    plan = TransformerPrimitivePlan(2, 4, dtype=dtype, device="cpu")
    x = _array(x_values, dtype=dtype)
    other = _array(other_values, dtype=dtype)
    scale = _array(scale_values, dtype=dtype)
    cosine = _array(np.cos(angles), dtype=dtype)
    sine = _array(np.sin(angles), dtype=dtype)
    x_q, other_q, scale_q = (
        x.numpy().astype(np.float32),
        other.numpy().astype(np.float32),
        scale.numpy().astype(np.float32),
    )
    cosine_q, sine_q = (
        cosine.numpy().astype(np.float32),
        sine.numpy().astype(np.float32),
    )

    np.testing.assert_allclose(
        plan.residual(x, other).numpy(), x_q + other_q, atol=atol, rtol=atol
    )
    np.testing.assert_allclose(
        plan.scale(x, scale).numpy(), x_q * scale_q, atol=atol, rtol=atol
    )
    sigmoid = 1.0 / (1.0 + np.exp(-x_q))
    np.testing.assert_allclose(
        plan.swiglu(x, other).numpy(), x_q * sigmoid * other_q, atol=atol, rtol=atol
    )
    np.testing.assert_allclose(
        plan.sigmoid_gate(other, x).numpy(), other_q * sigmoid, atol=atol, rtol=atol
    )
    rope_reference = np.concatenate(
        [
            x_q[:, :2] * cosine_q - x_q[:, 2:] * sine_q,
            x_q[:, 2:] * cosine_q + x_q[:, :2] * sine_q,
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        plan.rope(x, cosine, sine).numpy(), rope_reference, atol=atol, rtol=atol
    )

    x = _array(x_values, dtype=dtype, requires_grad=True)
    weight = _array(scale_values, dtype=dtype, requires_grad=True)
    tape = wp.Tape()
    with tape:
        output = plan.rms_norm(x, weight)
    tape.backward(grads={output: wp.ones_like(output)})
    x_q, weight_q = x.numpy().astype(np.float32), weight.numpy().astype(np.float32)
    inverse_rms = 1.0 / np.sqrt(
        np.mean(x_q * x_q, axis=1, keepdims=True) + plan.epsilon
    )
    weighted_sum = np.sum(x_q * weight_q, axis=1, keepdims=True)
    x_gradient = (
        weight_q * inverse_rms - x_q * weighted_sum * inverse_rms**3 / x_q.shape[1]
    )
    np.testing.assert_allclose(x.grad.numpy(), x_gradient, atol=atol, rtol=atol)
    np.testing.assert_allclose(
        weight.grad.numpy(), np.sum(x_q * inverse_rms, axis=0), atol=atol, rtol=atol
    )

    table = _array(
        np.arange(20, dtype=np.float32).reshape(5, 4) / 10,
        dtype=dtype,
        requires_grad=True,
    )
    token_ids = _array([3, 1, 3], dtype=wp.int32)
    embedding = EmbeddingPlan(3, 5, 4, dtype=dtype, device="cpu")
    tape = wp.Tape()
    with tape:
        gathered = embedding(table, token_ids)
    tape.backward(grads={gathered: wp.ones_like(gathered)})
    expected = np.zeros((5, 4), dtype=np.float32)
    expected[1] = 1.0
    expected[3] = 2.0
    np.testing.assert_array_equal(table.grad.numpy(), expected)


def test_cross_entropy_is_stable_and_has_exact_mean_gradient():
    logits_values = np.array(
        [[10000.0, 9999.0, 9998.0], [-10000.0, -9997.0, -9999.0], [2.0, 3.0, 4.0]],
        dtype=np.float32,
    )
    target_values = np.array([0, 2, -100], dtype=np.int32)
    logits = _array(logits_values)
    targets = _array(target_values, dtype=wp.int32)
    plan = CrossEntropyPlan(3, 3, device="cpu")

    loss = plan.forward(logits, targets).numpy()[0]
    gradient = plan.backward(logits, targets).numpy()
    shifted = logits_values[:2] - np.max(logits_values[:2], axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    expected_loss = np.mean(-np.log(probabilities[[0, 1], [0, 2]]))
    expected_gradient = np.zeros_like(logits_values)
    expected_gradient[:2] = probabilities
    expected_gradient[0, 0] -= 1.0
    expected_gradient[1, 2] -= 1.0
    expected_gradient[:2] /= 2.0
    np.testing.assert_allclose(loss, expected_loss, rtol=1.0e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1.0e-6, atol=1.0e-7)


def test_cross_entropy_all_ignored_is_zero():
    logits = _array([[1.0, 2.0], [3.0, 4.0]])
    targets = _array([-100, -100], dtype=wp.int32)
    plan = CrossEntropyPlan(2, 2, device="cpu")
    np.testing.assert_array_equal(plan.forward(logits, targets).numpy(), [0.0])
    np.testing.assert_array_equal(
        plan.backward(logits, targets).numpy(), np.zeros((2, 2), dtype=np.float32)
    )
