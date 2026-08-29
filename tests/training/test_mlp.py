# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.adapters import LoRAAdapterCollection, LoRAAdapterConfig
from warp_nn.training.mlp import LoRASwiGLUPlan


def _fixture(device):
    rng = np.random.default_rng(71)
    shapes = {"gate": (12, 8), "up": (12, 8), "down": (8, 12)}
    weights = {
        name: wp.array(
            rng.normal(0.0, 0.2, shape).astype(np.float32),
            dtype=wp.bfloat16,
            device=device,
        )
        for name, shape in shapes.items()
    }
    adapters = LoRAAdapterCollection(
        weights,
        rows=4,
        configs=LoRAAdapterConfig(rank=2, alpha=4.0),
        seed=3,
        use_cublas=False,
    )
    plan = LoRASwiGLUPlan(adapters, gate="gate", up="up", down="down")
    x = wp.array(
        rng.normal(0.0, 0.2, (4, 8)).astype(np.float32),
        dtype=wp.bfloat16,
        device=device,
    )
    grad_output = wp.array(
        rng.normal(0.0, 0.2, (4, 8)).astype(np.float32),
        dtype=wp.bfloat16,
        device=device,
    )
    return adapters, plan, x, grad_output


def test_lora_swiglu_fixed_buffers_and_accumulation_cpu():
    adapters, plan, x, grad_output = _fixture("cpu")
    arrays = (
        plan.output,
        plan.activated,
        plan.gate_grad,
        plan.up_grad,
        plan.input_grad,
        *adapters.named_gradients.values(),
    )
    pointers = tuple(array.ptr for array in arrays)

    plan.forward(x)
    plan.backward(x, grad_output)
    first = {
        name: gradient.numpy().copy()
        for name, gradient in adapters.named_gradients.items()
    }
    plan.backward(x, grad_output, accumulate=True)

    assert tuple(array.ptr for array in arrays) == pointers
    assert np.isfinite(plan.output.numpy()).all()
    assert np.isfinite(plan.input_grad.numpy()).all()
    for name, gradient in adapters.named_gradients.items():
        np.testing.assert_allclose(gradient.numpy(), 2.0 * first[name])


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_lora_swiglu_cuda_graph_replay():
    _, plan, x, grad_output = _fixture(CUDA_DEVICES[0])
    plan.forward(x)
    plan.backward(x, grad_output)
    references = plan.output.numpy().copy(), plan.input_grad.numpy().copy()

    wp.capture_begin(device=plan.device)
    try:
        plan.forward(x)
        plan.backward(x, grad_output)
        graph = wp.capture_end(device=plan.device)
    except Exception:
        wp.capture_end(device=plan.device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)
    np.testing.assert_array_equal(plan.output.numpy(), references[0])
    np.testing.assert_array_equal(plan.input_grad.numpy(), references[1])
