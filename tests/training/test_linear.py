# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.linear import (
    _get_linear_kernels,
    _get_tiled_linear_kernels,
    linear_backward,
    linear_forward,
    lora_backward,
    lora_forward,
)


def _array(values, dtype):
    return wp.array(np.asarray(values, dtype=np.float32), dtype=dtype, device="cpu")


def _numpy(array):
    return np.asarray(array.numpy(), dtype=np.float32)


@pytest.mark.parametrize("dtype,atol", [(wp.float16, 4.0e-3), (wp.bfloat16, 3.0e-2)])
def test_linear_forward_backward_cpu(dtype, atol):
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((3, 5), dtype=np.float32)
    weight_np = rng.standard_normal((4, 5), dtype=np.float32)
    grad_output_np = rng.standard_normal((3, 4), dtype=np.float32)
    x = _array(x_np, dtype)
    weight = _array(weight_np, dtype)
    grad_output = _array(grad_output_np, dtype)
    output = wp.empty((3, 4), dtype=dtype, device="cpu")
    grad_input = wp.empty((3, 5), dtype=dtype, device="cpu")
    grad_weight = wp.empty((4, 5), dtype=wp.float32, device="cpu")

    linear_forward(x, weight, output)
    linear_backward(x, weight, grad_output, grad_input, grad_weight)

    x_q, weight_q, grad_output_q = _numpy(x), _numpy(weight), _numpy(grad_output)
    np.testing.assert_allclose(_numpy(output), x_q @ weight_q.T, atol=atol, rtol=atol)
    np.testing.assert_allclose(
        _numpy(grad_input), grad_output_q @ weight_q, atol=atol, rtol=atol
    )
    np.testing.assert_allclose(
        _numpy(grad_weight), grad_output_q.T @ x_q, atol=2.0e-5, rtol=2.0e-5
    )
    linear_backward(x, weight, grad_output, grad_input, grad_weight, accumulate=True)
    np.testing.assert_allclose(
        _numpy(grad_weight),
        2.0 * grad_output_q.T @ x_q,
        atol=3.0e-5,
        rtol=3.0e-5,
    )


@pytest.mark.parametrize("dtype,atol", [(wp.float16, 5.0e-3), (wp.bfloat16, 4.0e-2)])
def test_lora_forward_backward_cpu(dtype, atol):
    rng = np.random.default_rng(19)
    rows, columns, inner, rank = 3, 4, 5, 2
    x = _array(rng.standard_normal((rows, inner), dtype=np.float32), dtype)
    weight = _array(rng.standard_normal((columns, inner), dtype=np.float32), dtype)
    lora_a = _array(rng.standard_normal((rank, inner), dtype=np.float32), dtype)
    lora_b = _array(rng.standard_normal((columns, rank), dtype=np.float32), dtype)
    grad_output = _array(rng.standard_normal((rows, columns), dtype=np.float32), dtype)
    hidden = wp.empty((rows, rank), dtype=wp.float32, device="cpu")
    output = wp.empty((rows, columns), dtype=dtype, device="cpu")
    grad_hidden = wp.empty_like(hidden)
    grad_input = wp.empty_like(x)
    grad_a = wp.empty((rank, inner), dtype=wp.float32, device="cpu")
    grad_b = wp.empty((columns, rank), dtype=wp.float32, device="cpu")
    grad_weight = wp.empty((columns, inner), dtype=wp.float32, device="cpu")
    scale = 0.375

    lora_forward(x, weight, lora_a, lora_b, hidden, output, scale)
    lora_backward(
        x,
        weight,
        lora_a,
        lora_b,
        hidden,
        grad_output,
        grad_hidden,
        grad_input,
        grad_a,
        grad_b,
        scale,
        grad_weight,
    )

    x_q, weight_q = _numpy(x), _numpy(weight)
    a_q, b_q, dy_q = _numpy(lora_a), _numpy(lora_b), _numpy(grad_output)
    hidden_ref = x_q @ a_q.T
    grad_hidden_ref = scale * (dy_q @ b_q)
    np.testing.assert_allclose(_numpy(hidden), hidden_ref, atol=2.0e-5, rtol=2.0e-5)
    np.testing.assert_allclose(
        _numpy(output),
        x_q @ weight_q.T + scale * hidden_ref @ b_q.T,
        atol=atol,
        rtol=atol,
    )
    np.testing.assert_allclose(
        _numpy(grad_hidden), grad_hidden_ref, atol=2.0e-5, rtol=2.0e-5
    )
    np.testing.assert_allclose(
        _numpy(grad_input),
        dy_q @ weight_q + grad_hidden_ref @ a_q,
        atol=atol,
        rtol=atol,
    )
    np.testing.assert_allclose(
        _numpy(grad_a), grad_hidden_ref.T @ x_q, atol=2.0e-5, rtol=2.0e-5
    )
    np.testing.assert_allclose(
        _numpy(grad_b), scale * dy_q.T @ hidden_ref, atol=2.0e-5, rtol=2.0e-5
    )
    np.testing.assert_allclose(
        _numpy(grad_weight), dy_q.T @ x_q, atol=2.0e-5, rtol=2.0e-5
    )
    lora_backward(
        x,
        weight,
        lora_a,
        lora_b,
        hidden,
        grad_output,
        grad_hidden,
        grad_input,
        grad_a,
        grad_b,
        scale,
        grad_weight,
        accumulate=True,
    )
    np.testing.assert_allclose(
        _numpy(grad_input),
        dy_q @ weight_q + grad_hidden_ref @ a_q,
        atol=atol,
        rtol=atol,
    )
    np.testing.assert_allclose(
        _numpy(grad_a),
        2.0 * grad_hidden_ref.T @ x_q,
        atol=3.0e-5,
        rtol=3.0e-5,
    )
    np.testing.assert_allclose(
        _numpy(grad_b),
        2.0 * scale * dy_q.T @ hidden_ref,
        atol=3.0e-5,
        rtol=3.0e-5,
    )
    np.testing.assert_allclose(
        _numpy(grad_weight),
        2.0 * dy_q.T @ x_q,
        atol=3.0e-5,
        rtol=3.0e-5,
    )


def test_kernels_are_cached_by_storage_dtype():
    assert _get_linear_kernels(wp.float16) is _get_linear_kernels(wp.float16)
    assert _get_linear_kernels(wp.float16) is not _get_linear_kernels(wp.bfloat16)


def test_linear_rejects_non_fp32_weight_gradient():
    assert _get_tiled_linear_kernels(wp.float16) is _get_tiled_linear_kernels(
        wp.float16
    )
    assert _get_tiled_linear_kernels(wp.float16) is not _get_tiled_linear_kernels(
        wp.bfloat16
    )
    x = wp.zeros((2, 3), dtype=wp.float16, device="cpu")
    weight = wp.zeros((4, 3), dtype=wp.float16, device="cpu")
    grad_output = wp.zeros((2, 4), dtype=wp.float16, device="cpu")
    grad_input = wp.empty_like(x)
    grad_weight = wp.empty_like(weight)
    with pytest.raises(TypeError, match="grad_weight"):
        linear_backward(x, weight, grad_output, grad_input, grad_weight)
