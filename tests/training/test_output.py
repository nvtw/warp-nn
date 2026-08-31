# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training import CausalLMOutputPlan


def _fixture(device, dtype=wp.bfloat16):
    rng = np.random.default_rng(37)
    rows, hidden, classes = 3, 8, 17
    x = wp.array(
        rng.normal(size=(rows, hidden)).astype(np.float32),
        dtype=dtype,
        device=device,
    )
    norm_weight = wp.array(
        rng.uniform(0.5, 1.5, size=hidden).astype(np.float32),
        dtype=dtype,
        device=device,
    )
    head = wp.array(
        rng.normal(size=(classes, hidden)).astype(np.float32) / 3.0,
        dtype=dtype,
        device=device,
    )
    targets = wp.array(
        np.array([2, 11, 16], dtype=np.int32), dtype=wp.int32, device=device
    )
    return CausalLMOutputPlan(rows, norm_weight, head), x, targets


def _reference(plan, x, targets):
    x_np = np.asarray(x.numpy(), dtype=np.float32)
    weight = np.asarray(plan.norm_weight.numpy(), dtype=np.float32)
    head = np.asarray(plan.lm_head.numpy(), dtype=np.float32)
    inverse = 1.0 / np.sqrt(np.mean(x_np * x_np, axis=1, keepdims=True) + 1.0e-6)
    normalized = x_np * inverse * weight
    # Match the stored low-precision boundary before the frozen head.
    normalized = np.asarray(
        wp.array(normalized, dtype=plan.dtype, device="cpu").numpy(),
        dtype=np.float32,
    )
    logits = normalized @ head.T
    logits = np.asarray(
        wp.array(logits, dtype=plan.dtype, device="cpu").numpy(), dtype=np.float32
    )
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    target_np = targets.numpy()
    loss = -np.log(probabilities[np.arange(len(target_np)), target_np]).mean()
    probabilities[np.arange(len(target_np)), target_np] -= 1.0
    dnorm = (probabilities / len(target_np)) @ head
    # The head gradient is stored low precision before norm backward.
    dnorm = np.asarray(
        wp.array(dnorm, dtype=plan.dtype, device="cpu").numpy(), dtype=np.float32
    )
    weighted = dnorm * weight
    dot = np.sum(weighted * x_np, axis=1, keepdims=True)
    dx = weighted * inverse - x_np * inverse**3 * dot / x_np.shape[1]
    return loss, dx


@pytest.mark.parametrize(
    "dtype,atol", [(wp.float16, 4.0e-3), (wp.bfloat16, 3.0e-2)]
)
def test_causal_lm_output_matches_reference(dtype, atol):
    plan, x, targets = _fixture("cpu", dtype)
    expected_loss, expected_gradient = _reference(plan, x, targets)

    actual_loss = float(plan.forward(x, targets).numpy()[0])
    actual_gradient = np.asarray(plan.backward(x, targets).numpy(), dtype=np.float32)

    np.testing.assert_allclose(actual_loss, expected_loss, atol=2.0e-5, rtol=2.0e-5)
    np.testing.assert_allclose(
        actual_gradient, expected_gradient, atol=atol, rtol=atol
    )
    assert plan.loss_plan.gradient is None
    assert plan.logits.ptr == plan.loss_plan.backward(plan.logits, targets).ptr


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_causal_lm_output_cuda_graph_replay():
    plan, x, targets = _fixture(CUDA_DEVICES[0])
    plan.forward(x, targets)
    plan.backward(x, targets)
    expected_loss = plan.loss.numpy().copy()
    expected_gradient = plan.input_grad.numpy().copy()

    wp.capture_begin(device=plan.device)
    try:
        plan.forward(x, targets)
        plan.backward(x, targets)
        graph = wp.capture_end(device=plan.device)
    except Exception:
        wp.capture_end(device=plan.device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)

    np.testing.assert_array_equal(plan.loss.numpy(), expected_loss)
    np.testing.assert_array_equal(plan.input_grad.numpy(), expected_gradient)
