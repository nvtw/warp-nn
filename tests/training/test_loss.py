# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training import LowPrecisionCrossEntropyPlan


def _reference(logits, targets, ignore_index=-100, reduction="mean"):
    valid = targets != ignore_index
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    losses = np.zeros(logits.shape[0], dtype=np.float32)
    rows = np.flatnonzero(valid)
    losses[rows] = -np.log(probabilities[rows, targets[rows]])
    gradient = probabilities
    gradient[rows, targets[rows]] -= 1.0
    gradient[~valid] = 0.0
    if reduction == "mean" and len(rows):
        return np.mean(losses[rows]), gradient / np.float32(len(rows))
    return np.sum(losses), gradient


@pytest.mark.parametrize("dtype,atol", [(wp.float16, 8.0e-4), (wp.bfloat16, 7.0e-3)])
def test_low_precision_cross_entropy_matches_reference(dtype, atol):
    rng = np.random.default_rng(17)
    values = rng.normal(size=(4, 513)).astype(np.float32) * 2.0
    targets_np = np.array([0, 255, -100, 512], dtype=np.int32)
    logits = wp.array(values, dtype=dtype, device="cpu")
    targets = wp.array(targets_np, dtype=wp.int32, device="cpu")
    quantized = np.asarray(logits.numpy(), dtype=np.float32)
    expected_loss, expected_gradient = _reference(quantized, targets_np)
    plan = LowPrecisionCrossEntropyPlan(
        4, 513, dtype=dtype, ignore_index=-100, device="cpu"
    )

    actual_loss = float(plan.forward(logits, targets).numpy()[0])
    actual_gradient = np.asarray(
        plan.backward(logits, targets).numpy(), dtype=np.float32
    )

    np.testing.assert_allclose(actual_loss, expected_loss, atol=2.0e-6, rtol=2.0e-6)
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=atol, rtol=atol)
    assert int(plan.valid_count.numpy()[0]) == 3
    assert plan.partitions == 3


def test_low_precision_cross_entropy_sum_and_in_place_gradient():
    values = np.array([[4.0, -1.0, 0.5], [-3.0, 2.0, 1.0]], dtype=np.float32)
    targets_np = np.array([2, 1], dtype=np.int32)
    logits = wp.array(values, dtype=wp.float16, device="cpu")
    targets = wp.array(targets_np, dtype=wp.int32, device="cpu")
    expected_loss, expected_gradient = _reference(values, targets_np, reduction="sum")
    plan = LowPrecisionCrossEntropyPlan(
        2, 3, dtype=wp.float16, in_place=True, device="cpu"
    )

    actual_loss = float(plan.forward(logits, targets, reduction="sum").numpy()[0])
    gradient = plan.backward(logits, targets, reduction="sum", loss_scale=2.0)

    assert gradient.ptr == logits.ptr
    assert plan.gradient is None
    np.testing.assert_allclose(actual_loss, expected_loss, atol=2.0e-6)
    np.testing.assert_allclose(
        np.asarray(gradient.numpy(), dtype=np.float32),
        2.0 * expected_gradient,
        atol=8.0e-4,
        rtol=8.0e-4,
    )


def test_low_precision_cross_entropy_logit_scale_and_softcap_gradient():
    raw = np.array([[4.0, -2.0, 0.5], [-1.0, 2.5, 0.25]], dtype=np.float32)
    targets_np = np.array([2, 0], dtype=np.int32)
    multiplier, cap = 0.75, 2.0
    transformed = cap * np.tanh(raw * multiplier / cap)
    expected_loss, expected_gradient = _reference(transformed, targets_np)
    expected_gradient *= multiplier * (1.0 - (transformed / cap) ** 2)
    logits = wp.array(raw, dtype=wp.float16, device="cpu")
    targets = wp.array(targets_np, dtype=wp.int32, device="cpu")
    plan = LowPrecisionCrossEntropyPlan(
        2,
        3,
        dtype=wp.float16,
        in_place=True,
        logit_multiplier=multiplier,
        softcap=cap,
        device="cpu",
    )

    actual_loss = float(plan.forward(logits, targets).numpy()[0])
    actual_gradient = np.asarray(
        plan.backward(logits, targets).numpy(), dtype=np.float32
    )

    # Reference the low-precision transformed logits used by the loss kernels.
    stored = np.asarray(
        wp.array(transformed, dtype=wp.float16, device="cpu").numpy(),
        dtype=np.float32,
    )
    expected_loss, expected_gradient = _reference(stored, targets_np)
    expected_gradient *= multiplier * (1.0 - (stored / cap) ** 2)
    np.testing.assert_allclose(actual_loss, expected_loss, atol=2.0e-5)
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=8.0e-4)


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_low_precision_cross_entropy_cuda_graph_replay():
    device = CUDA_DEVICES[0]
    rng = np.random.default_rng(23)
    values = rng.normal(size=(8, 1025)).astype(np.float32)
    targets_np = np.arange(8, dtype=np.int32) * 127
    logits = wp.array(values, dtype=wp.bfloat16, device=device)
    targets = wp.array(targets_np, dtype=wp.int32, device=device)
    plan = LowPrecisionCrossEntropyPlan(8, 1025, dtype=wp.bfloat16, device=device)
    plan.forward(logits, targets)
    plan.backward(logits, targets)
    expected_loss = plan.loss.numpy().copy()
    expected_gradient = plan.gradient.numpy().copy()

    wp.capture_begin(device=device)
    try:
        plan.forward(logits, targets)
        plan.backward(logits, targets)
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)

    np.testing.assert_array_equal(plan.loss.numpy(), expected_loss)
    np.testing.assert_array_equal(plan.gradient.numpy(), expected_gradient)
