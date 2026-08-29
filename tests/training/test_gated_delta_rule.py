# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.gated_delta_rule import GatedDeltaRulePlan


def _reference(query, key, value, decay, beta, lengths, past, scale):
    batch, key_heads, sequence, _ = query.shape
    value_heads, value_size = value.shape[1], value.shape[3]
    state = past.astype(np.float32).copy()
    output = np.zeros((batch, value_heads, sequence, value_size), dtype=np.float32)
    transformed = np.zeros_like(output)
    for batch_index in range(batch):
        for token in range(int(lengths[batch_index])):
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
    return output, transformed, state


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


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


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

    wp.capture_begin(device=device)
    try:
        plan.forward(*inputs)
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise
    wp.capture_launch(graph)
    first = _numpy(plan.output)
    inputs[2].fill_(0.0)
    wp.capture_launch(graph)
    second = _numpy(plan.output)
    assert np.max(np.abs(first - second)) > 1.0e-3
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
