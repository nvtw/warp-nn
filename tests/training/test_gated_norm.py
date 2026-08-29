# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.gated_norm import GatedRMSNormPlan


def _reference(x, gate, scale, epsilon):
    inverse = 1.0 / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + epsilon)
    silu = gate / (1.0 + np.exp(-gate))
    return x * inverse * scale * silu


def test_gated_rms_norm_forward_and_backward_finite_difference_cpu():
    rng = np.random.default_rng(167)
    x = rng.normal(0.0, 0.4, (3, 4)).astype(np.float32)
    gate = rng.normal(0.0, 0.3, x.shape).astype(np.float32)
    scale = rng.normal(1.0, 0.1, 4).astype(np.float32)
    output_grad = rng.normal(0.0, 0.2, x.shape).astype(np.float32)
    arrays = tuple(
        wp.array(value, dtype=wp.float32, device="cpu")
        for value in (x, gate, scale, output_grad)
    )
    plan = GatedRMSNormPlan(3, 4, wp.float32, epsilon=1.0e-5, device="cpu")
    output = plan.forward(*arrays[:3])
    np.testing.assert_allclose(
        output.numpy(), _reference(x, gate, scale, plan.epsilon), rtol=2.0e-6
    )
    input_grad, gate_grad = plan.backward(*arrays)

    def loss():
        return float(np.sum(_reference(x, gate, scale, plan.epsilon) * output_grad))

    def finite_difference(value):
        result = np.empty_like(value)
        epsilon = 1.0e-3
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + epsilon
            positive = loss()
            value[index] = original - epsilon
            negative = loss()
            value[index] = original
            result[index] = (positive - negative) / (2.0 * epsilon)
        return result

    np.testing.assert_allclose(
        input_grad.numpy(), finite_difference(x), rtol=5.0e-4, atol=5.0e-5
    )
    np.testing.assert_allclose(
        gate_grad.numpy(), finite_difference(gate), rtol=5.0e-4, atol=5.0e-5
    )


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_gated_rms_norm_bfloat16_cuda_graph_replay():
    device = CUDA_DEVICES[0]
    if device.arch < 80:
        pytest.skip("BF16 requires SM80")
    rng = np.random.default_rng(173)

    def low(value):
        return wp.array(value, dtype=wp.bfloat16, device=device)

    x = low(rng.normal(0.0, 0.3, (8, 128)))
    gate = low(rng.normal(0.0, 0.2, (8, 128)))
    scale = low(rng.normal(1.0, 0.05, 128))
    output_grad = low(rng.normal(0.0, 0.2, (8, 128)))
    plan = GatedRMSNormPlan(8, 128, wp.bfloat16, epsilon=1.0e-6, device=device)

    def execute():
        plan.forward(x, gate, scale)
        plan.backward(x, gate, scale, output_grad)

    execute()
    wp.synchronize_device(device)
    x_np = x.numpy().astype(np.float32)
    gate_np = gate.numpy().astype(np.float32)
    scale_np = scale.numpy().astype(np.float32)
    expected = _reference(x_np, gate_np, scale_np, plan.epsilon)
    np.testing.assert_allclose(
        plan.output.numpy().astype(np.float32),
        expected,
        rtol=1.0e-2,
        atol=2.0e-3,
    )
    wp.capture_begin(device=device)
    try:
        execute()
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise
    wp.capture_launch(graph)
    first = plan.output.numpy().copy()
    gate.fill_(0.0)
    wp.capture_launch(graph)
    second = plan.output.numpy()
    np.testing.assert_array_equal(second, 0.0)
    assert np.max(np.abs(first.astype(np.float32))) > 1.0e-3
    assert np.isfinite(plan.input_grad.numpy()).all()
    assert np.isfinite(plan.gate_grad.numpy()).all()
