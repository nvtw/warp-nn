# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import warp as wp

from warp_nn.training.primitives import TransformerPrimitivePlan
from warp_nn.training.step import LoRALinearTrainingPlan


@wp.kernel
def _sum_2d(values: wp.array2d(dtype=wp.float16), loss: wp.array1d[wp.float32]):
    row, column = wp.tid()
    wp.atomic_add(loss, 0, wp.float32(values[row, column]))


def _array(values, dtype):
    return wp.array(np.asarray(values, dtype=np.float32), dtype=dtype, device="cpu")


def test_lora_linear_plan_fixed_buffers_and_accumulation_cpu():
    rng = np.random.default_rng(91)
    rows, inner, columns, rank = 3, 5, 4, 2
    dtype = wp.float16
    x = _array(rng.normal(size=(rows, inner)), dtype)
    weight = _array(rng.normal(size=(columns, inner)), dtype)
    lora_a = _array(rng.normal(size=(rank, inner)), dtype)
    lora_b = _array(rng.normal(size=(columns, rank)), dtype)
    grad_output = _array(rng.normal(size=(rows, columns)), dtype)
    plan = LoRALinearTrainingPlan(
        rows, inner, columns, rank, dtype, train_base=True, device="cpu"
    )
    pointers = (
        plan.output.ptr,
        plan.output.grad.ptr,
        plan.hidden.ptr,
        plan.grad_input.ptr,
        plan.grad_a.ptr,
        plan.grad_b.ptr,
        plan.grad_weight.ptr,
    )

    plan.forward(x, weight, lora_a, lora_b, scale=0.25)
    plan.backward(x, weight, lora_a, lora_b, grad_output, scale=0.25)
    first_a = plan.grad_a.numpy().copy()
    first_b = plan.grad_b.numpy().copy()
    first_weight = plan.grad_weight.numpy().copy()
    first_input = plan.grad_input.numpy().copy()
    plan.backward(x, weight, lora_a, lora_b, grad_output, scale=0.25, accumulate=True)
    np.testing.assert_allclose(plan.grad_a.numpy(), 2.0 * first_a, atol=2.0e-5)
    np.testing.assert_allclose(plan.grad_b.numpy(), 2.0 * first_b, atol=2.0e-5)
    np.testing.assert_allclose(
        plan.grad_weight.numpy(), 2.0 * first_weight, atol=2.0e-5
    )
    np.testing.assert_allclose(plan.grad_input.numpy(), first_input, atol=2.0e-3)
    assert pointers == (
        plan.output.ptr,
        plan.output.grad.ptr,
        plan.hidden.ptr,
        plan.grad_input.ptr,
        plan.grad_a.ptr,
        plan.grad_b.ptr,
        plan.grad_weight.ptr,
    )

    plan.output.grad.fill_(1.0)
    plan.zero_grad()
    np.testing.assert_array_equal(
        plan.output.grad.numpy(), np.zeros_like(plan.output.numpy())
    )
    np.testing.assert_array_equal(plan.grad_a.numpy(), np.zeros_like(first_a))
    np.testing.assert_array_equal(plan.grad_b.numpy(), np.zeros_like(first_b))
    np.testing.assert_array_equal(plan.grad_weight.numpy(), np.zeros_like(first_weight))


def test_lora_linear_output_is_tape_boundary_cpu():
    rng = np.random.default_rng(19)
    rows, inner, columns, rank = 2, 3, 4, 2
    dtype = wp.float16
    x = _array(rng.normal(size=(rows, inner)), dtype)
    weight = _array(rng.normal(size=(columns, inner)), dtype)
    lora_a = _array(rng.normal(size=(rank, inner)), dtype)
    lora_b = _array(rng.normal(size=(columns, rank)), dtype)
    gate = wp.array(
        np.zeros((rows, columns), dtype=np.float32),
        dtype=dtype,
        device="cpu",
        requires_grad=True,
    )
    plan = LoRALinearTrainingPlan(rows, inner, columns, rank, dtype, device="cpu")
    primitive = TransformerPrimitivePlan(rows, columns, dtype=dtype, device="cpu")
    loss = wp.zeros(1, dtype=wp.float32, device="cpu", requires_grad=True)

    plan.forward(x, weight, lora_a, lora_b, scale=0.25)
    tape = wp.Tape()
    with tape:
        downstream = primitive.sigmoid_gate(plan.output, gate)
        wp.launch(
            _sum_2d,
            dim=downstream.shape,
            inputs=[downstream],
            outputs=[loss],
            device="cpu",
        )
    tape.backward(loss)

    boundary_gradient = plan.output.grad.numpy()
    assert np.isfinite(boundary_gradient).all()
    assert np.any(boundary_gradient != 0.0)
    plan.backward(x, weight, lora_a, lora_b, plan.output.grad, scale=0.25)
    assert np.isfinite(plan.grad_input.numpy()).all()
