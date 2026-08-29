# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.optimizer import AdamWPlan


def _adamw_reference(value, gradient, first, second, step, **config):
    gradient = gradient / (config["loss_scale"] * config["accumulation_steps"])
    first = config["beta1"] * first + (1.0 - config["beta1"]) * gradient
    second = config["beta2"] * second + (1.0 - config["beta2"]) * gradient**2
    first_hat = first / (1.0 - config["beta1"] ** step)
    second_hat = second / (1.0 - config["beta2"] ** step)
    update = first_hat / (np.sqrt(second_hat) + config["epsilon"])
    value = value - config["learning_rate"] * (update + config["weight_decay"] * value)
    return value, first, second


def test_adamw_fixed_state_loss_unscale_and_zero_cpu():
    parameter_np = np.array([[0.5, -1.0], [2.0, -0.25]], dtype=np.float32)
    gradient_np = np.array([[8.0, -4.0], [2.0, 6.0]], dtype=np.float32)
    parameter = wp.array(parameter_np, dtype=wp.float32, device="cpu")
    gradient = wp.array(gradient_np, dtype=wp.float32, device="cpu")
    config = {
        "learning_rate": 0.05,
        "beta1": 0.8,
        "beta2": 0.9,
        "epsilon": 1.0e-6,
        "weight_decay": 0.1,
        "loss_scale": 4.0,
        "accumulation_steps": 2,
    }
    plan = AdamWPlan([parameter], [gradient], **config)
    parameter_ptr = parameter.ptr
    state_ptrs = (
        plan.masters[0].ptr,
        plan.first_moments[0].ptr,
        plan.second_moments[0].ptr,
    )
    expected, first, second = _adamw_reference(
        parameter_np,
        gradient_np,
        np.zeros_like(parameter_np),
        np.zeros_like(parameter_np),
        1,
        **config,
    )

    plan.step()
    np.testing.assert_allclose(parameter.numpy(), expected, atol=2.0e-6, rtol=2.0e-6)
    np.testing.assert_allclose(
        plan.masters[0].numpy(), expected.reshape(-1), atol=2.0e-6
    )
    expected, first, second = _adamw_reference(
        expected, gradient_np, first, second, 2, **config
    )
    plan.step()
    np.testing.assert_allclose(parameter.numpy(), expected, atol=2.0e-6, rtol=2.0e-6)
    np.testing.assert_array_equal(plan.step_count.numpy(), [2])
    assert parameter.ptr == parameter_ptr
    assert (
        plan.masters[0].ptr,
        plan.first_moments[0].ptr,
        plan.second_moments[0].ptr,
    ) == state_ptrs

    plan.zero_grad()
    np.testing.assert_array_equal(gradient.numpy(), np.zeros_like(gradient_np))


def test_adamw_bfloat16_parameter_storage_cpu():
    parameter = wp.array([0.5, -1.0], dtype=wp.bfloat16, device="cpu")
    gradient = wp.array([2.0, -3.0], dtype=wp.float32, device="cpu")
    initial = parameter.numpy().astype(np.float32)
    plan = AdamWPlan(
        [parameter],
        [gradient],
        learning_rate=0.1,
        beta1=0.0,
        beta2=0.0,
        epsilon=1.0e-6,
    )

    plan.step()
    expected = initial - 0.1 * gradient.numpy() / (np.abs(gradient.numpy()) + 1.0e-6)
    np.testing.assert_allclose(parameter.numpy(), expected, atol=4.0e-3, rtol=4.0e-3)


def test_adamw_bfloat16_sub_ulp_updates_accumulate_in_master_cpu():
    parameter = wp.array([1.0], dtype=wp.bfloat16, device="cpu")
    gradient = wp.array([1.0], dtype=wp.float32, device="cpu")
    plan = AdamWPlan(
        [parameter], [gradient], learning_rate=1.0e-3, beta1=0.0, beta2=0.0, epsilon=1.0
    )
    master_ptr = plan.masters[0].ptr

    plan.step()
    np.testing.assert_array_equal(parameter.numpy(), [1.0])
    np.testing.assert_allclose(plan.masters[0].numpy(), [0.9995], atol=1.0e-7)

    for _ in range(7):
        plan.step()
    assert plan.masters[0].ptr == master_ptr
    np.testing.assert_allclose(plan.masters[0].numpy(), [0.996], atol=5.0e-7)
    assert parameter.numpy()[0] < 1.0


def test_adamw_rejects_empty_or_noncontiguous_buffers_cpu():
    empty_parameter = wp.empty(0, dtype=wp.bfloat16, device="cpu")
    empty_gradient = wp.empty(0, dtype=wp.float32, device="cpu")
    with pytest.raises(ValueError, match="non-empty"):
        AdamWPlan([empty_parameter], [empty_gradient])

    parameter_base = wp.zeros((2, 4), dtype=wp.bfloat16, device="cpu")
    gradient_base = wp.zeros((2, 4), dtype=wp.float32, device="cpu")
    contiguous_parameter = wp.zeros((2, 2), dtype=wp.bfloat16, device="cpu")
    contiguous_gradient = wp.zeros((2, 2), dtype=wp.float32, device="cpu")
    with pytest.raises(ValueError, match="contiguous"):
        AdamWPlan([parameter_base[:, ::2]], [contiguous_gradient])
    with pytest.raises(ValueError, match="contiguous"):
        AdamWPlan([contiguous_parameter], [gradient_base[:, ::2]])
