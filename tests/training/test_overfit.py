# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.bridges import cast_from_float32, cast_to_float32
from warp_nn.training.optimizer import AdamWPlan
from warp_nn.training.primitives import CrossEntropyPlan
from warp_nn.training.step import LoRALinearTrainingPlan


def _array(values, dtype, device):
    return wp.array(np.asarray(values), dtype=dtype, device=device)


def _overfit_identity(device, steps, *, capture=False):
    size = 4
    dtype = wp.bfloat16
    rng = np.random.default_rng(7)
    x = _array(np.eye(size, dtype=np.float32), dtype, device)
    weight = _array(np.zeros((size, size), dtype=np.float32), dtype, device)
    lora_a = _array(0.1 * rng.normal(size=(size, size)), dtype, device)
    lora_b = _array(np.zeros((size, size), dtype=np.float32), dtype, device)
    targets = _array(np.arange(size, dtype=np.int32), wp.int32, device)
    logits = wp.empty((size, size), dtype=wp.float32, device=device)
    grad_output = wp.empty((size, size), dtype=dtype, device=device)
    linear = LoRALinearTrainingPlan(size, size, size, size, dtype, device=device)
    cross_entropy = CrossEntropyPlan(size, size, device=device)
    optimizer = AdamWPlan(
        [lora_a, lora_b],
        [linear.grad_a, linear.grad_b],
        learning_rate=0.1,
        weight_decay=0.0,
    )
    arrays = (
        x,
        weight,
        lora_a,
        lora_b,
        logits,
        grad_output,
        linear.output,
        linear.hidden,
        linear.grad_input,
        linear.grad_hidden,
        linear.grad_a,
        linear.grad_b,
        cross_entropy.logsumexp,
        cross_entropy.losses,
        cross_entropy.valid,
        cross_entropy.loss,
        cross_entropy.maximum,
        cross_entropy.valid_count,
        cross_entropy.gradient,
        optimizer.step_count,
        *optimizer.masters,
        *optimizer.first_moments,
        *optimizer.second_moments,
    )
    pointers = tuple(array.ptr for array in arrays)

    linear.forward(x, weight, lora_a, lora_b, scale=1.0)
    cast_to_float32(linear.output, logits)
    initial_loss = float(cross_entropy.forward(logits, targets).numpy()[0])

    def train_step():
        gradient = cross_entropy.backward(logits, targets)
        cast_from_float32(gradient, grad_output)
        linear.backward(x, weight, lora_a, lora_b, grad_output, scale=1.0)
        optimizer.step()
        linear.forward(x, weight, lora_a, lora_b, scale=1.0)
        cast_to_float32(linear.output, logits)
        cross_entropy.forward(logits, targets)

    if capture:
        train_step()  # Compile every kernel before capture.
        wp.synchronize_device(device)
        wp.capture_begin(device=device)
        try:
            train_step()
            graph = wp.capture_end(device=device)
        except Exception:
            wp.capture_end(device=device)
            raise
        for _ in range(steps - 1):
            wp.capture_launch(graph)
    else:
        for _ in range(steps):
            train_step()

    final_loss = float(cross_entropy.loss.numpy()[0])
    predictions = np.argmax(logits.numpy(), axis=1)
    assert linear.grad_weight is None
    np.testing.assert_array_equal(weight.numpy(), np.zeros((size, size)))
    np.testing.assert_array_equal(predictions, np.arange(size))
    assert final_loss < 0.1 * initial_loss
    assert tuple(array.ptr for array in arrays) == pointers
    assert all(master.dtype == wp.float32 for master in optimizer.masters)
    for array in (
        logits,
        cross_entropy.loss,
        cross_entropy.gradient,
        lora_a,
        lora_b,
        *optimizer.masters,
        *optimizer.first_moments,
        *optimizer.second_moments,
    ):
        assert np.isfinite(array.numpy()).all()


def test_lora_identity_guaranteed_overfit_cpu():
    _overfit_identity("cpu", steps=120)


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_lora_identity_guaranteed_overfit_cuda():
    _overfit_identity(CUDA_DEVICES[0], steps=80, capture=True)
