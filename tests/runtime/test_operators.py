# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime._cublas import try_create_cublas
from warp_nn.runtime.kernels import (
    _causal_conv_rows_kernel,
    _get_gated_rms_norm_kernel,
    _get_linear_attention_kernel,
    _prepare_gated_delta_kernel,
    _update_conv_rows_state_kernel,
)
from warp_nn.runtime.operators import Operation, execute_operations, plan_linear


@pytest.mark.parametrize(("device", "rows"), [("cpu", 3), ("cuda:0", 3), ("cuda:0", 32)])
def test_linear_operation(device, rows):
    if not is_device_available(device):
        pytest.skip(f"Device {device} is not available")
    dtype = wp.bfloat16 if device.startswith("cuda") else wp.float32
    rng = np.random.default_rng(13)
    x_np = rng.normal(size=(rows, 37)).astype(np.float32)
    weight_np = rng.normal(size=(41, 37)).astype(np.float32)
    tensors = {
        "x": wp.array(x_np, dtype=dtype, device=device),
        "weight": wp.array(weight_np, dtype=dtype, device=device),
    }
    shapes = {name: tuple(value.shape) for name, value in tensors.items()}
    operation = Operation("Linear", ["x", "weight"], ["output"])
    plan_linear(operation, tensors, shapes, wp.get_device(device))

    execute_operations([operation], tensors, shapes, wp.get_device(device))

    np.testing.assert_allclose(tensors["output"].numpy(), x_np @ weight_np.T, atol=0.2, rtol=0.02)


def test_linear_operation_cublas():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    cublas = try_create_cublas()
    if cublas is None:
        pytest.skip("cuBLAS is not available")
    rng = np.random.default_rng(17)
    x_np = rng.normal(size=(5, 32)).astype(np.float32)
    weight_np = rng.normal(size=(48, 32)).astype(np.float32)
    tensors = {
        "x": wp.array(x_np, dtype=wp.bfloat16, device="cuda:0"),
        "weight": wp.array(weight_np, dtype=wp.bfloat16, device="cuda:0"),
    }
    shapes = {name: tuple(value.shape) for name, value in tensors.items()}
    operation = Operation("Linear", ["x", "weight"], ["output"])
    plan_linear(operation, tensors, shapes, wp.get_device("cuda:0"), cublas=cublas)

    execute_operations([operation], tensors, shapes, wp.get_device("cuda:0"))

    np.testing.assert_allclose(tensors["output"].numpy(), x_np @ weight_np.T, atol=0.2, rtol=0.02)


def test_gated_rms_norm_bfloat16():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(19)
    x = wp.array(rng.normal(size=(3, 8)).astype(np.float32), dtype=wp.bfloat16, device="cuda:0")
    gate = wp.array(rng.normal(size=(3, 8)).astype(np.float32), dtype=wp.bfloat16, device="cuda:0")
    scale = wp.array(rng.normal(size=8).astype(np.float32), dtype=wp.bfloat16, device="cuda:0")
    output = wp.empty_like(x)
    tile_width, kernel = _get_gated_rms_norm_kernel(8, wp.bfloat16)

    wp.launch_tiled(kernel, dim=3, inputs=[x, gate, scale, output, 1.0e-6], block_dim=tile_width, device="cuda:0")

    x_np = x.numpy().astype(np.float32)
    gate_np = gate.numpy().astype(np.float32)
    expected = x_np / np.sqrt(np.mean(x_np * x_np, axis=1, keepdims=True) + 1.0e-6)
    expected *= scale.numpy().astype(np.float32)
    expected *= gate_np / (1.0 + np.exp(-gate_np))
    np.testing.assert_allclose(output.numpy(), expected, atol=0.04, rtol=0.02)


def test_mixed_state_linear_attention_bfloat16():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(23)
    rows, query_heads, key_heads, value_heads, width = 2, 2, 2, 4, 8
    q = wp.array(rng.normal(size=(rows, query_heads * width)).astype(np.float32), dtype=wp.bfloat16, device="cuda:0")
    k = wp.array(rng.normal(size=(rows, key_heads * width)).astype(np.float32), dtype=wp.bfloat16, device="cuda:0")
    v = wp.array(rng.normal(size=(rows, value_heads * width)).astype(np.float32), dtype=wp.bfloat16, device="cuda:0")
    past = wp.array((0.05 * rng.normal(size=(value_heads * width, width))).astype(np.float32), device="cuda:0")
    decay = wp.array(rng.uniform(-0.2, -0.01, size=(rows, value_heads)).astype(np.float32), device="cuda:0")
    beta = wp.array(rng.uniform(0.1, 0.9, size=(rows, value_heads)).astype(np.float32), device="cuda:0")
    output = wp.empty((rows, value_heads * width), dtype=wp.bfloat16, device="cuda:0")
    present = wp.empty_like(past)
    kernel = _get_linear_attention_kernel(width, width, wp.bfloat16, wp.float32)

    wp.launch_tiled(
        kernel,
        dim=value_heads,
        inputs=[
            q,
            k,
            v,
            past,
            decay,
            beta,
            output,
            present,
            rows,
            query_heads,
            key_heads,
            value_heads,
            True,
            False,
            True,
            True,
            width**-0.5,
        ],
        block_dim=32,
        device="cuda:0",
    )

    q_np, k_np, v_np = (array.numpy().astype(np.float32) for array in (q, k, v))
    state = past.numpy().reshape(value_heads, width, width).astype(np.float32)
    expected = np.empty((rows, value_heads * width), dtype=np.float32)
    decay_np, beta_np = decay.numpy(), beta.numpy()
    for row in range(rows):
        for value_head in range(value_heads):
            key_head = value_head * key_heads // value_heads
            key_vector = k_np[row, key_head * width : (key_head + 1) * width]
            value_vector = v_np[row, value_head * width : (value_head + 1) * width]
            state[value_head] *= np.exp(decay_np[row, value_head])
            delta = beta_np[row, value_head] * (value_vector - key_vector @ state[value_head])
            state[value_head] += np.outer(key_vector, delta)
            query_head = value_head * query_heads // value_heads
            query_vector = q_np[row, query_head * width : (query_head + 1) * width]
            expected[row, value_head * width : (value_head + 1) * width] = (
                width**-0.5 * query_vector @ state[value_head]
            )
    np.testing.assert_allclose(output.numpy(), expected, atol=0.08, rtol=0.03)
    np.testing.assert_allclose(present.numpy(), state.reshape(value_heads * width, width), atol=2.0e-4, rtol=2.0e-4)


def test_gated_delta_preparation_and_row_causal_conv():
    rows, channels, heads, kernel_size = 2, 3, 2, 3
    x_np = np.arange(rows * channels, dtype=np.float32).reshape(rows, channels) / 5.0
    weight_np = np.arange(channels * kernel_size, dtype=np.float32).reshape(channels, 1, kernel_size) / 10.0
    state_np = np.full((channels, kernel_size - 1), 0.25, dtype=np.float32)
    x = wp.array(x_np, device="cpu")
    weight = wp.array(weight_np, device="cpu")
    state = wp.array(state_np, device="cpu")
    output = wp.empty_like(x)
    a = wp.array(np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32), device="cpu")
    b = wp.array(np.array([[-0.5, 0.2], [0.7, -0.1]], dtype=np.float32), device="cpu")
    a_log = wp.array(np.array([0.0, 0.5], dtype=np.float32), device="cpu")
    dt_bias = wp.array(np.array([0.2, -0.3], dtype=np.float32), device="cpu")
    decay = wp.empty((rows, heads), dtype=wp.float32, device="cpu")
    beta = wp.empty_like(decay)

    wp.launch(_causal_conv_rows_kernel, dim=(rows, channels), inputs=[x, weight, state, output], device="cpu")
    wp.launch(_update_conv_rows_state_kernel, dim=channels, inputs=[x, state], device="cpu")
    wp.launch(_prepare_gated_delta_kernel, dim=(rows, heads), inputs=[a, b, a_log, dt_bias, decay, beta], device="cpu")

    padded = np.concatenate((state_np.T, x_np), axis=0)
    expected_conv = np.empty_like(x_np)
    for row in range(rows):
        for channel in range(channels):
            total = padded[row : row + kernel_size, channel] @ weight_np[channel, 0]
            expected_conv[row, channel] = total / (1.0 + np.exp(-total))
    np.testing.assert_allclose(output.numpy(), expected_conv, atol=1.0e-6)
    np.testing.assert_array_equal(state.numpy(), padded[-(kernel_size - 1) :].T)
    a_np, b_np = a.numpy(), b.numpy()
    expected_beta = 1.0 / (1.0 + np.exp(-b_np))
    expected_decay = -np.exp(a_log.numpy()) * np.logaddexp(0.0, a_np + dt_bias.numpy())
    np.testing.assert_allclose(beta.numpy(), expected_beta, atol=1.0e-6)
    np.testing.assert_allclose(decay.numpy(), expected_decay, atol=1.0e-6)
