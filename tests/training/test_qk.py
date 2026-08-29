# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.qk import QKTransformPlan


def _reference(x, weight, positions, cosine, sine, epsilon, scale, grad=None):
    inverse = 1.0 / np.sqrt(np.mean(x * x, axis=-1) + epsilon)
    normalized = x * inverse[..., None] * weight * scale
    rotary_dim = 2 * cosine.shape[1] if cosine is not None else 0
    output = normalized.copy()
    if rotary_dim:
        half = rotary_dim // 2
        for batch in range(x.shape[0]):
            for token in range(x.shape[2]):
                c = cosine[positions[batch, token]]
                s = sine[positions[batch, token]]
                first = normalized[batch, :, token, :half].copy()
                second = normalized[batch, :, token, half:rotary_dim].copy()
                output[batch, :, token, :half] = first * c - second * s
                output[batch, :, token, half:rotary_dim] = second * c + first * s
    if grad is None:
        return output, inverse

    normalized_grad = grad.copy()
    if rotary_dim:
        half = rotary_dim // 2
        for batch in range(x.shape[0]):
            for token in range(x.shape[2]):
                c = cosine[positions[batch, token]]
                s = sine[positions[batch, token]]
                first = grad[batch, :, token, :half].copy()
                second = grad[batch, :, token, half:rotary_dim].copy()
                normalized_grad[batch, :, token, :half] = first * c + second * s
                normalized_grad[batch, :, token, half:rotary_dim] = (
                    second * c - first * s
                )
    normalized_grad *= scale
    dot = np.sum(normalized_grad * weight * x, axis=-1)
    input_grad = normalized_grad * weight * inverse[..., None] - (
        x * inverse[..., None] ** 3 * dot[..., None] / x.shape[-1]
    )
    return output, input_grad


@pytest.mark.parametrize("rotary_dim", [0, 6])
def test_qk_transform_forward_backward_reference_cpu(rotary_dim):
    rng = np.random.default_rng(43 + rotary_dim)
    shape = (1, 2, 3, 8)
    x_np = rng.normal(0.0, 0.3, shape).astype(np.float32)
    weight_np = rng.normal(1.0, 0.1, shape[-1]).astype(np.float32)
    grad_np = rng.normal(0.0, 0.2, shape).astype(np.float32)
    positions_np = np.array([[0, 2, 4]], dtype=np.int64)
    angles = rng.normal(0.0, 0.4, (5, max(1, rotary_dim // 2))).astype(np.float32)
    cosine_np, sine_np = np.cos(angles), np.sin(angles)
    plan = QKTransformPlan(
        1,
        2,
        3,
        8,
        wp.bfloat16,
        rotary_dim=rotary_dim,
        epsilon=1.0e-5,
        scale=0.75,
        device="cpu",
    )
    x = wp.array(x_np, dtype=wp.bfloat16, device="cpu")
    weight = wp.array(weight_np, dtype=wp.bfloat16, device="cpu")
    grad = wp.array(grad_np, dtype=wp.float32, device="cpu")
    kwargs = {}
    if rotary_dim:
        kwargs = {
            "positions": wp.array(positions_np, dtype=wp.int64, device="cpu"),
            "cosine": wp.array(cosine_np, dtype=wp.bfloat16, device="cpu"),
            "sine": wp.array(sine_np, dtype=wp.bfloat16, device="cpu"),
        }
    output = plan.forward(x, weight, **kwargs)
    input_grad = plan.backward(x, weight, grad, **kwargs)

    x_actual, weight_actual = x.numpy(), weight.numpy()
    cosine_actual = kwargs["cosine"].numpy() if rotary_dim else None
    sine_actual = kwargs["sine"].numpy() if rotary_dim else None
    expected_output, expected_grad = _reference(
        x_actual,
        weight_actual,
        positions_np,
        cosine_actual,
        sine_actual,
        1.0e-5,
        0.75,
        grad_np,
    )
    np.testing.assert_allclose(output.numpy(), expected_output, atol=8e-3, rtol=8e-3)
    np.testing.assert_allclose(input_grad.numpy(), expected_grad, atol=5e-3, rtol=3e-3)


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_qk_transform_cuda_graph_replay():
    device = CUDA_DEVICES[0]
    rng = np.random.default_rng(61)
    shape = (1, 2, 4, 8)
    plan = QKTransformPlan(1, 2, 4, 8, wp.bfloat16, device=device)
    x = wp.array(rng.normal(size=shape), dtype=wp.bfloat16, device=device)
    weight = wp.ones(8, dtype=wp.bfloat16, device=device)
    positions = wp.array(np.arange(4, dtype=np.int64)[None], device=device)
    angles = rng.normal(size=(4, 4)).astype(np.float32)
    cosine = wp.array(np.cos(angles), dtype=wp.bfloat16, device=device)
    sine = wp.array(np.sin(angles), dtype=wp.bfloat16, device=device)
    grad = wp.array(rng.normal(size=shape), dtype=wp.float32, device=device)
    plan.forward(x, weight, positions, cosine, sine)
    plan.backward(x, weight, grad, positions, cosine, sine)
    references = plan.output.numpy().copy(), plan.input_grad.numpy().copy()

    wp.capture_begin(device=device)
    try:
        plan.forward(x, weight, positions, cosine, sine)
        plan.backward(x, weight, grad, positions, cosine, sine)
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)
    np.testing.assert_array_equal(plan.output.numpy(), references[0])
    np.testing.assert_array_equal(plan.input_grad.numpy(), references[1])
