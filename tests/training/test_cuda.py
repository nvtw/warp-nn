# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small permanent CUDA checks for the allocation-free training path."""

import numpy as np
import pytest

import warp as wp

from warp_nn.training import (
    CrossEntropyPlan,
    TransformerPrimitivePlan,
    gqa_attention_backward,
    gqa_attention_forward,
    linear_backward,
    linear_forward,
)
from warp_nn.runtime._cublas import try_create_cublas
from warp_nn.training.linear import _use_tiled
from warp_nn.training.optimizer import AdamWPlan
from warp_nn.training.step import LoRALinearTrainingPlan


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]
pytestmark = pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")


@wp.kernel
def _sum_2d(values: wp.array2d(dtype=wp.float32), loss: wp.array1d(dtype=wp.float32)):
    row, column = wp.tid()
    wp.atomic_add(loss, 0, values[row, column])


def _array(values, dtype, device):
    return wp.array(np.asarray(values), dtype=dtype, device=device)


def _numpy(array):
    return np.asarray(array.numpy(), dtype=np.float32)


def _gqa_reference(query, key, value, lengths, scale, window, segment_bounds=None):
    output = np.zeros_like(query, dtype=np.float32)
    lse = np.full(query.shape[:3], -np.finfo(np.float32).max, dtype=np.float32)
    heads_per_kv = query.shape[1] // key.shape[1]
    for batch, length in enumerate(lengths):
        for head in range(query.shape[1]):
            kv_head = head // heads_per_kv
            for token in range(int(length)):
                first = max(0, token + 1 - window) if window else 0
                if segment_bounds is not None:
                    first = max(first, int(segment_bounds[batch, token, 0]))
                keys = slice(first, token + 1)
                scores = key[batch, kv_head, keys] @ query[batch, head, token] * scale
                maximum = np.max(scores)
                exponentials = np.exp(scores - maximum)
                probabilities = exponentials / np.sum(exponentials)
                output[batch, head, token] = probabilities @ value[batch, kv_head, keys]
                lse[batch, head, token] = maximum + np.log(np.sum(exponentials))
    return output, lse


def _gqa_backward_reference(
    query, key, value, output_grad, lengths, scale, window, segment_bounds=None
):
    query_grad = np.zeros_like(query, dtype=np.float32)
    key_grad = np.zeros_like(key, dtype=np.float32)
    value_grad = np.zeros_like(value, dtype=np.float32)
    heads_per_kv = query.shape[1] // key.shape[1]
    for batch, length in enumerate(lengths):
        for head in range(query.shape[1]):
            kv_head = head // heads_per_kv
            for token in range(int(length)):
                first = max(0, token + 1 - window) if window else 0
                if segment_bounds is not None:
                    first = max(first, int(segment_bounds[batch, token, 0]))
                keys = slice(first, token + 1)
                scores = key[batch, kv_head, keys] @ query[batch, head, token] * scale
                probabilities = np.exp(scores - np.max(scores))
                probabilities /= np.sum(probabilities)
                probability_grad = (
                    value[batch, kv_head, keys] @ output_grad[batch, head, token]
                )
                score_grad = probabilities * (
                    probability_grad - probabilities @ probability_grad
                )
                query_grad[batch, head, token] += (
                    score_grad @ key[batch, kv_head, keys] * scale
                )
                key_grad[batch, kv_head, keys] += (
                    np.outer(score_grad, query[batch, head, token]) * scale
                )
                value_grad[batch, kv_head, keys] += np.outer(
                    probabilities, output_grad[batch, head, token]
                )
    return query_grad, key_grad, value_grad


def _gqa_buffers(device, seed=17):
    rng = np.random.default_rng(seed)
    query_shape = (1, 4, 4, 8)
    kv_shape = (1, 2, 4, 8)
    query = _array(rng.normal(size=query_shape).astype(np.float32), wp.bfloat16, device)
    key = _array(rng.normal(size=kv_shape).astype(np.float32), wp.bfloat16, device)
    value = _array(rng.normal(size=kv_shape).astype(np.float32), wp.bfloat16, device)
    lengths = _array(np.array([3], dtype=np.int32), wp.int32, device)
    output = wp.empty(query_shape, dtype=wp.bfloat16, device=device)
    lse = wp.empty(query_shape[:3], dtype=wp.float32, device=device)
    accumulator = wp.empty(query_shape, dtype=wp.float32, device=device)
    return query, key, value, lengths, output, lse, accumulator


def test_cuda_bfloat16_linear_forward_backward():
    device = CUDA_DEVICES[0]
    rng = np.random.default_rng(5)
    x = _array(rng.normal(size=(3, 5)).astype(np.float32), wp.bfloat16, device)
    weight = _array(rng.normal(size=(4, 5)).astype(np.float32), wp.bfloat16, device)
    grad_output = _array(
        rng.normal(size=(3, 4)).astype(np.float32), wp.bfloat16, device
    )
    output = wp.empty((3, 4), dtype=wp.bfloat16, device=device)
    grad_input = wp.empty_like(x)
    grad_weight = wp.empty((4, 5), dtype=wp.float32, device=device)

    linear_forward(x, weight, output)
    linear_backward(x, weight, grad_output, grad_input, grad_weight)

    x_ref, weight_ref, grad_ref = _numpy(x), _numpy(weight), _numpy(grad_output)
    np.testing.assert_allclose(
        _numpy(output), x_ref @ weight_ref.T, atol=4e-2, rtol=4e-2
    )
    np.testing.assert_allclose(
        _numpy(grad_input), grad_ref @ weight_ref, atol=4e-2, rtol=4e-2
    )
    np.testing.assert_allclose(
        _numpy(grad_weight), grad_ref.T @ x_ref, atol=3e-5, rtol=3e-5
    )


@pytest.mark.parametrize("rows", [32, 64])
def test_cuda_bfloat16_native_regular_right_linear_graph_replay(rows):
    device = CUDA_DEVICES[0]
    if device.arch < 80:
        pytest.skip("BF16 native Linear requires SM80 or newer")

    columns, inner = 96, 64
    rng = np.random.default_rng(113 + rows)
    x = wp.zeros((rows, inner), dtype=wp.bfloat16, device=device)
    weight = _array(rng.normal(0.0, 0.2, (columns, inner)), wp.bfloat16, device)
    grad_output = _array(rng.normal(0.0, 0.2, (rows, columns)), wp.bfloat16, device)
    grad_input = wp.empty_like(x)

    linear_backward(x, weight, grad_output, grad_input)
    expected = _numpy(grad_output) @ _numpy(weight)
    np.testing.assert_allclose(_numpy(grad_input), expected, atol=0.15, rtol=0.02)
    reference = _numpy(grad_input).copy()

    with wp.ScopedCapture(device) as capture:
        linear_backward(x, weight, grad_output, grad_input)
    wp.capture_launch(capture.graph)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(_numpy(grad_input), reference)


def test_cuda_bfloat16_tiled_linear_tails_and_graph_replay():
    device = CUDA_DEVICES[0]
    if device.arch < 80:
        pytest.skip("BF16 tiled Linear requires SM80 or newer")

    rows, columns, inner = 33, 35, 37
    rng = np.random.default_rng(37)
    x = _array(rng.normal(size=(rows, inner)).astype(np.float32), wp.bfloat16, device)
    weight = _array(
        rng.normal(size=(columns, inner)).astype(np.float32), wp.bfloat16, device
    )
    grad_output = _array(
        rng.normal(size=(rows, columns)).astype(np.float32), wp.bfloat16, device
    )
    output = wp.empty((rows, columns), dtype=wp.bfloat16, device=device)
    grad_input = wp.empty((rows, inner), dtype=wp.bfloat16, device=device)
    grad_weight = wp.empty((columns, inner), dtype=wp.float32, device=device)

    assert _use_tiled(rows, columns, inner, wp.bfloat16, device, x, weight, output)
    assert _use_tiled(
        rows, inner, columns, wp.bfloat16, device, grad_output, weight, grad_input
    )
    assert _use_tiled(
        columns, inner, rows, wp.bfloat16, device, grad_output, x, grad_weight
    )

    # Warm compilation also establishes the fixed-buffer replay reference.
    linear_forward(x, weight, output)
    linear_backward(x, weight, grad_output, grad_input, grad_weight)
    x_ref, weight_ref, grad_ref = _numpy(x), _numpy(weight), _numpy(grad_output)
    np.testing.assert_allclose(
        _numpy(output), x_ref @ weight_ref.T, atol=1.2e-1, rtol=3.0e-2
    )
    np.testing.assert_allclose(
        _numpy(grad_input), grad_ref @ weight_ref, atol=1.2e-1, rtol=3.0e-2
    )
    np.testing.assert_allclose(
        _numpy(grad_weight), grad_ref.T @ x_ref, atol=3.0e-3, rtol=3.0e-3
    )
    output_reference = _numpy(output).copy()
    grad_input_reference = _numpy(grad_input).copy()
    grad_weight_reference = _numpy(grad_weight).copy()

    wp.capture_begin(device=device)
    try:
        linear_forward(x, weight, output)
        linear_backward(x, weight, grad_output, grad_input, grad_weight)
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise

    wp.capture_launch(graph)
    wp.capture_launch(graph)
    np.testing.assert_array_equal(_numpy(output), output_reference)
    np.testing.assert_array_equal(_numpy(grad_input), grad_input_reference)
    np.testing.assert_array_equal(_numpy(grad_weight), grad_weight_reference)


def test_cuda_bfloat16_tiled_lora_tails():
    device = CUDA_DEVICES[0]
    if device.arch < 80:
        pytest.skip("BF16 tiled LoRA requires SM80 or newer")

    rows, columns, inner, rank = 33, 35, 37, 2
    scale = 0.25
    rng = np.random.default_rng(97)
    x = _array(rng.normal(0.0, 0.1, (rows, inner)), wp.bfloat16, device)
    weight = _array(rng.normal(0.0, 0.1, (columns, inner)), wp.bfloat16, device)
    lora_a = _array(rng.normal(0.0, 0.1, (rank, inner)), wp.bfloat16, device)
    lora_b = _array(rng.normal(0.0, 0.1, (columns, rank)), wp.bfloat16, device)
    grad_output = _array(rng.normal(0.0, 0.1, (rows, columns)), wp.bfloat16, device)
    plan = LoRALinearTrainingPlan(
        rows, inner, columns, rank, wp.bfloat16, device=device
    )
    # Exercise the non-split boundary-masked fallback explicitly.
    plan.forward_matmul_splits = 1
    plan.backward_matmul_splits = 1

    plan.forward(x, weight, lora_a, lora_b, scale=scale)
    plan.backward(x, weight, lora_a, lora_b, grad_output, scale=scale)
    x_np, weight_np = _numpy(x), _numpy(weight)
    a_np, b_np, dy_np = _numpy(lora_a), _numpy(lora_b), _numpy(grad_output)
    hidden_reference = x_np @ a_np.T
    grad_hidden_reference = scale * dy_np @ b_np
    np.testing.assert_allclose(
        _numpy(plan.hidden), hidden_reference, atol=0.01, rtol=0.01
    )
    np.testing.assert_allclose(
        _numpy(plan.output),
        x_np @ weight_np.T + scale * hidden_reference @ b_np.T,
        atol=0.08,
        rtol=0.03,
    )
    np.testing.assert_allclose(
        _numpy(plan.grad_hidden), grad_hidden_reference, atol=0.01, rtol=0.01
    )
    np.testing.assert_allclose(
        _numpy(plan.grad_input),
        dy_np @ weight_np + grad_hidden_reference @ a_np,
        atol=0.08,
        rtol=0.03,
    )


def test_cuda_bfloat16_base_linear_split_k_and_graph_replay():
    device = CUDA_DEVICES[0]
    if device.arch < 80:
        pytest.skip("BF16 split-K Linear requires SM80 or newer")

    rows, columns, inner = 64, 256, 1024
    rng = np.random.default_rng(127)
    x = _array(rng.normal(0.0, 0.05, (rows, inner)), wp.bfloat16, device)
    weight = _array(rng.normal(0.0, 0.05, (columns, inner)), wp.bfloat16, device)
    grad_output = _array(rng.normal(0.0, 0.05, (rows, columns)), wp.bfloat16, device)
    output = wp.empty((rows, columns), dtype=wp.bfloat16, device=device)
    grad_input = wp.empty_like(x)
    forward_workspace = wp.empty((2 * rows, columns), dtype=wp.float32, device=device)
    backward_workspace = wp.empty((4 * rows, inner), dtype=wp.float32, device=device)

    linear_forward(x, weight, output, matmul_workspace=forward_workspace)
    linear_backward(
        x,
        weight,
        grad_output,
        grad_input,
        matmul_workspace=backward_workspace,
    )
    output_reference = _numpy(x) @ _numpy(weight).T
    grad_input_reference = _numpy(grad_output) @ _numpy(weight)
    np.testing.assert_allclose(_numpy(output), output_reference, atol=0.08, rtol=0.02)
    np.testing.assert_allclose(
        _numpy(grad_input), grad_input_reference, atol=0.08, rtol=0.02
    )
    assert (
        np.linalg.norm(_numpy(output) - output_reference)
        / np.linalg.norm(output_reference)
        < 0.006
    )
    assert (
        np.linalg.norm(_numpy(grad_input) - grad_input_reference)
        / np.linalg.norm(grad_input_reference)
        < 0.006
    )
    references = (_numpy(output).copy(), _numpy(grad_input).copy())

    with wp.ScopedCapture(device) as capture:
        linear_forward(x, weight, output, matmul_workspace=forward_workspace)
        linear_backward(
            x,
            weight,
            grad_output,
            grad_input,
            matmul_workspace=backward_workspace,
        )
    wp.capture_launch(capture.graph)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(_numpy(output), references[0])
    np.testing.assert_array_equal(_numpy(grad_input), references[1])


@pytest.mark.parametrize("rank", [2, 16])
@pytest.mark.parametrize("use_cublas", [False, True])
def test_cuda_lora_split_k_tensor_cores_and_graph_replay(use_cublas, rank):
    device = CUDA_DEVICES[0]
    if device.arch < 80:
        pytest.skip("BF16 split-K LoRA requires SM80 or newer")
    cublas = try_create_cublas() if use_cublas else None
    if use_cublas and cublas is None:
        pytest.skip("cuBLAS is unavailable")

    rows, columns, inner = 32, 64, 64
    scale = 0.25
    rng = np.random.default_rng(109)
    x = _array(rng.normal(0.0, 0.2, (rows, inner)), wp.bfloat16, device)
    weight = _array(rng.normal(0.0, 0.2, (columns, inner)), wp.bfloat16, device)
    lora_a = _array(rng.normal(0.0, 0.2, (rank, inner)), wp.bfloat16, device)
    lora_b = _array(rng.normal(0.0, 0.2, (columns, rank)), wp.bfloat16, device)
    grad_output = _array(rng.normal(0.0, 0.2, (rows, columns)), wp.bfloat16, device)
    plan = LoRALinearTrainingPlan(
        rows, inner, columns, rank, wp.bfloat16, device=device, cublas=cublas
    )
    assert plan.forward_matmul_splits > 1
    assert plan.backward_matmul_splits > 1
    assert plan.matmul_workspace is not None
    pointers = (
        plan.output.ptr,
        plan.hidden.ptr,
        plan.grad_input.ptr,
        plan.grad_hidden.ptr,
        plan.grad_a.ptr,
        plan.grad_b.ptr,
        plan.matmul_workspace.ptr,
    )

    plan.forward(x, weight, lora_a, lora_b, scale=scale)
    plan.backward(x, weight, lora_a, lora_b, grad_output, scale=scale)
    x_np, weight_np = _numpy(x), _numpy(weight)
    a_np, b_np, dy_np = _numpy(lora_a), _numpy(lora_b), _numpy(grad_output)
    hidden_reference = x_np @ a_np.T
    output_reference = x_np @ weight_np.T + scale * hidden_reference @ b_np.T
    grad_hidden_reference = scale * (dy_np @ b_np)
    grad_input_reference = dy_np @ weight_np + grad_hidden_reference @ a_np
    np.testing.assert_allclose(
        _numpy(plan.hidden), hidden_reference, atol=2e-3, rtol=2e-3
    )
    np.testing.assert_allclose(
        _numpy(plan.output), output_reference, atol=6e-2, rtol=3e-2
    )
    np.testing.assert_allclose(
        _numpy(plan.grad_hidden), grad_hidden_reference, atol=2e-3, rtol=2e-3
    )
    np.testing.assert_allclose(
        _numpy(plan.grad_input), grad_input_reference, atol=6e-2, rtol=3e-2
    )
    np.testing.assert_allclose(
        _numpy(plan.grad_a), grad_hidden_reference.T @ x_np, atol=2e-3, rtol=2e-3
    )
    np.testing.assert_allclose(
        _numpy(plan.grad_b), scale * dy_np.T @ hidden_reference, atol=2e-3, rtol=2e-3
    )
    references = tuple(
        _numpy(array).copy()
        for array in (
            plan.output,
            plan.hidden,
            plan.grad_input,
            plan.grad_a,
            plan.grad_b,
        )
    )

    wp.capture_begin(device=device)
    try:
        plan.forward(x, weight, lora_a, lora_b, scale=scale)
        plan.backward(x, weight, lora_a, lora_b, grad_output, scale=scale)
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)
    for array, reference in zip(
        (plan.output, plan.hidden, plan.grad_input, plan.grad_a, plan.grad_b),
        references,
    ):
        np.testing.assert_array_equal(_numpy(array), reference)
    assert pointers == (
        plan.output.ptr,
        plan.hidden.ptr,
        plan.grad_input.ptr,
        plan.grad_hidden.ptr,
        plan.grad_a.ptr,
        plan.grad_b.ptr,
        plan.matmul_workspace.ptr,
    )


@pytest.mark.parametrize("window", [0, 2])
def test_cuda_bfloat16_gqa_full_and_sliding_reference(window):
    device = CUDA_DEVICES[0]
    query, key, value, lengths, output, lse, accumulator = _gqa_buffers(device)
    scale = 0.37
    gqa_attention_forward(
        query,
        key,
        value,
        lengths,
        output,
        lse,
        accumulator,
        scale=scale,
        window=window,
    )

    reference = _gqa_reference(
        _numpy(query), _numpy(key), _numpy(value), lengths.numpy(), scale, window
    )
    np.testing.assert_allclose(_numpy(output), reference[0], atol=2e-2, rtol=8e-3)
    np.testing.assert_allclose(_numpy(lse), reference[1], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize(
    "dtype,head_size",
    [(wp.float16, 8), (wp.bfloat16, 128), (wp.bfloat16, 256)],
)
@pytest.mark.parametrize("window", [0, 2])
def test_cuda_streaming_gqa_backward_reference_and_accumulate(dtype, head_size, window):
    device = CUDA_DEVICES[0]
    rng = np.random.default_rng(91 + head_size + window)
    batch, query_heads, kv_heads, sequence = 2, 4, 2, 4
    query_shape = (batch, query_heads, sequence, head_size)
    kv_shape = (batch, kv_heads, sequence, head_size)
    query = _array(rng.normal(0.0, 0.25, query_shape), dtype, device)
    key = _array(rng.normal(0.0, 0.25, kv_shape), dtype, device)
    value = _array(rng.normal(0.0, 0.25, kv_shape), dtype, device)
    output_grad = _array(rng.normal(0.0, 0.25, query_shape), dtype, device)
    lengths_np = np.array([4, 2], dtype=np.int32)
    lengths = _array(lengths_np, wp.int32, device)
    output = wp.empty(query_shape, dtype=dtype, device=device)
    lse = wp.empty(query_shape[:3], dtype=wp.float32, device=device)
    accumulator = wp.empty(query_shape, dtype=wp.float32, device=device)
    query_grad = wp.empty(query_shape, dtype=wp.float32, device=device)
    key_grad = wp.empty(kv_shape, dtype=wp.float32, device=device)
    value_grad = wp.empty(kv_shape, dtype=wp.float32, device=device)
    delta = wp.empty(query_shape[:3], dtype=wp.float32, device=device)
    scale = head_size**-0.5

    gqa_attention_forward(
        query,
        key,
        value,
        lengths,
        output,
        lse,
        accumulator,
        scale=scale,
        window=window,
    )
    gqa_attention_backward(
        query,
        key,
        value,
        output_grad,
        lengths,
        lse,
        query_grad,
        key_grad,
        value_grad,
        delta,
        scale=scale,
        window=window,
    )

    query_np, key_np, value_np = _numpy(query), _numpy(key), _numpy(value)
    output_grad_np = _numpy(output_grad)
    forward_reference = _gqa_reference(
        query_np, key_np, value_np, lengths_np, scale, window
    )
    backward_reference = _gqa_backward_reference(
        query_np, key_np, value_np, output_grad_np, lengths_np, scale, window
    )
    np.testing.assert_allclose(
        _numpy(output), forward_reference[0], atol=2e-2, rtol=8e-3
    )
    np.testing.assert_allclose(_numpy(lse), forward_reference[1], atol=5e-4, rtol=5e-4)
    gradient_atol = 2e-3 if dtype == wp.bfloat16 else 8e-4
    for actual, expected in zip(
        (_numpy(query_grad), _numpy(key_grad), _numpy(value_grad)),
        backward_reference,
    ):
        np.testing.assert_allclose(actual, expected, atol=gradient_atol, rtol=2e-3)

    gqa_attention_backward(
        query,
        key,
        value,
        output_grad,
        lengths,
        lse,
        query_grad,
        key_grad,
        value_grad,
        delta,
        scale=scale,
        window=window,
        accumulate=True,
    )
    for actual, expected in zip(
        (_numpy(query_grad), _numpy(key_grad), _numpy(value_grad)),
        backward_reference,
    ):
        np.testing.assert_allclose(
            actual, 2.0 * expected, atol=2.0 * gradient_atol, rtol=2e-3
        )


def test_cuda_segmented_flash_gqa_matches_isolated_examples():
    device = CUDA_DEVICES[0]
    rng = np.random.default_rng(313)
    batch, query_heads, kv_heads, sequence, head_size = 1, 4, 2, 18, 128
    query_shape = (batch, query_heads, sequence, head_size)
    kv_shape = (batch, kv_heads, sequence, head_size)
    query = _array(rng.normal(0.0, 0.2, query_shape), wp.bfloat16, device)
    key = _array(rng.normal(0.0, 0.2, kv_shape), wp.bfloat16, device)
    value = _array(rng.normal(0.0, 0.2, kv_shape), wp.bfloat16, device)
    output_grad = _array(rng.normal(0.0, 0.2, query_shape), wp.bfloat16, device)
    lengths_np = np.array([sequence], dtype=np.int32)
    lengths = _array(lengths_np, wp.int32, device)
    bounds_np = np.empty((batch, sequence, 2), dtype=np.int32)
    bounds_np[:, :7] = (0, 7)
    bounds_np[:, 7:] = (7, sequence)
    bounds = _array(bounds_np, wp.int32, device)
    output = wp.empty(query_shape, dtype=wp.bfloat16, device=device)
    lse = wp.empty(query_shape[:3], dtype=wp.float32, device=device)
    accumulator = wp.empty(query_shape, dtype=wp.float32, device=device)
    query_grad = wp.empty(query_shape, dtype=wp.float32, device=device)
    key_grad = wp.empty(kv_shape, dtype=wp.float32, device=device)
    value_grad = wp.empty(kv_shape, dtype=wp.float32, device=device)
    delta = wp.empty(query_shape[:3], dtype=wp.float32, device=device)
    scale = head_size**-0.5

    gqa_attention_forward(
        query,
        key,
        value,
        lengths,
        output,
        lse,
        accumulator,
        segment_bounds=bounds,
        scale=scale,
    )
    gqa_attention_backward(
        query,
        key,
        value,
        output_grad,
        lengths,
        lse,
        query_grad,
        key_grad,
        value_grad,
        delta,
        segment_bounds=bounds,
        scale=scale,
    )

    query_np, key_np, value_np = _numpy(query), _numpy(key), _numpy(value)
    forward = _gqa_reference(
        query_np, key_np, value_np, lengths_np, scale, 0, bounds_np
    )
    backward = _gqa_backward_reference(
        query_np,
        key_np,
        value_np,
        _numpy(output_grad),
        lengths_np,
        scale,
        0,
        bounds_np,
    )
    np.testing.assert_allclose(_numpy(output), forward[0], atol=2e-2, rtol=8e-3)
    np.testing.assert_allclose(_numpy(lse), forward[1], atol=5e-4, rtol=5e-4)
    for actual, expected in zip(
        (_numpy(query_grad), _numpy(key_grad), _numpy(value_grad)), backward
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_cuda_rms_tape_and_stable_cross_entropy():
    device = CUDA_DEVICES[0]
    x_values = np.array([[1.0, -2.0, 0.5], [-0.25, 1.5, 2.0]], dtype=np.float32)
    weight_values = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    x = wp.array(x_values, dtype=wp.float32, device=device, requires_grad=True)
    weight = wp.array(
        weight_values, dtype=wp.float32, device=device, requires_grad=True
    )
    plan = TransformerPrimitivePlan(2, 3, rotary_dim=2, epsilon=1e-5, device=device)
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        output = plan.rms_norm(x, weight)
        wp.launch(
            _sum_2d,
            dim=output.shape,
            inputs=[output],
            outputs=[loss],
            device=device,
        )
    tape.backward(loss)

    inverse_rms = 1.0 / np.sqrt(np.mean(x_values**2, axis=1, keepdims=True) + 1e-5)
    weighted_sum = np.sum(x_values * weight_values, axis=1, keepdims=True)
    expected_grad = weight_values * inverse_rms - (
        x_values * weighted_sum * inverse_rms**3 / x_values.shape[1]
    )
    assert np.all(np.isfinite(_numpy(x.grad)))
    np.testing.assert_allclose(_numpy(x.grad), expected_grad, atol=3e-5, rtol=3e-5)

    logits_np = np.array(
        [[10000.0, 9999.0, 9998.0], [-10000.0, -9997.0, -9999.0]],
        dtype=np.float32,
    )
    targets_np = np.array([0, 2], dtype=np.int32)
    logits = _array(logits_np, wp.float32, device)
    targets = _array(targets_np, wp.int32, device)
    cross_entropy = CrossEntropyPlan(2, 3, device=device)
    actual_loss = cross_entropy.forward(logits, targets).numpy()[0]
    actual_gradient = cross_entropy.backward(logits, targets).numpy()
    shifted = logits_np - np.max(logits_np, axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    expected_loss = np.mean(-np.log(probabilities[np.arange(2), targets_np]))
    probabilities[np.arange(2), targets_np] -= 1.0
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=2e-6)
    np.testing.assert_allclose(actual_gradient, probabilities / 2.0, atol=2e-7)


def test_cuda_graph_replays_linear_and_gqa_with_fixed_buffers():
    device = CUDA_DEVICES[0]
    x = _array(np.arange(12, dtype=np.float32).reshape(3, 4) / 7, wp.bfloat16, device)
    weight = _array(
        np.arange(20, dtype=np.float32).reshape(5, 4) / 11, wp.bfloat16, device
    )
    linear_output = wp.empty((3, 5), dtype=wp.bfloat16, device=device)
    query, key, value, lengths, output, lse, accumulator = _gqa_buffers(device, seed=29)
    output_grad = wp.ones(query.shape, dtype=wp.bfloat16, device=device)
    query_grad = wp.empty(query.shape, dtype=wp.float32, device=device)
    key_grad = wp.empty(key.shape, dtype=wp.float32, device=device)
    value_grad = wp.empty(value.shape, dtype=wp.float32, device=device)
    delta = wp.empty(query.shape[:3], dtype=wp.float32, device=device)

    # Compile and establish a deterministic reference before capture.
    linear_forward(x, weight, linear_output)
    gqa_attention_forward(
        query, key, value, lengths, output, lse, accumulator, window=2
    )
    gqa_attention_backward(
        query,
        key,
        value,
        output_grad,
        lengths,
        lse,
        query_grad,
        key_grad,
        value_grad,
        delta,
        window=2,
    )
    references = tuple(
        _numpy(array).copy()
        for array in (linear_output, output, query_grad, key_grad, value_grad, delta)
    )

    wp.capture_begin(device=device)
    try:
        linear_forward(x, weight, linear_output)
        gqa_attention_forward(
            query, key, value, lengths, output, lse, accumulator, window=2
        )
        gqa_attention_backward(
            query,
            key,
            value,
            output_grad,
            lengths,
            lse,
            query_grad,
            key_grad,
            value_grad,
            delta,
            window=2,
        )
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise

    wp.capture_launch(graph)
    wp.capture_launch(graph)
    for array, reference in zip(
        (linear_output, output, query_grad, key_grad, value_grad, delta), references
    ):
        np.testing.assert_array_equal(_numpy(array), reference)


def test_cuda_adamw_master_and_bfloat16_mirror_graph_replay():
    device = CUDA_DEVICES[0]
    # Compile the update kernels before capture without advancing the tested plan.
    warm_parameter = _array([0.0], wp.bfloat16, device)
    warm_gradient = _array([1.0], wp.float32, device)
    warm_plan = AdamWPlan([warm_parameter], [warm_gradient], beta1=0.0, beta2=0.0)
    warm_plan.step()
    wp.synchronize_device(device)

    initial = np.array([[0.5, -1.0], [2.0, -0.25]], dtype=np.float32)
    gradient_values = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    parameter = _array(initial, wp.bfloat16, device)
    gradient = _array(gradient_values, wp.float32, device)
    plan = AdamWPlan(
        [parameter],
        [gradient],
        learning_rate=0.125,
        beta1=0.0,
        beta2=0.0,
        epsilon=1.0e-8,
    )
    wp.synchronize_device(device)
    pointers = (
        parameter.ptr,
        gradient.ptr,
        plan.masters[0].ptr,
        plan.first_moments[0].ptr,
        plan.second_moments[0].ptr,
        plan.step_count.ptr,
    )

    wp.capture_begin(device=device)
    try:
        plan.step()
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)

    expected_master = initial - np.float32(0.25) * np.sign(gradient_values)
    np.testing.assert_array_equal(plan.step_count.numpy(), [2])
    np.testing.assert_array_equal(
        plan.masters[0].numpy().reshape(initial.shape), expected_master
    )
    np.testing.assert_allclose(
        parameter.numpy(), expected_master, atol=4.0e-3, rtol=0.0
    )
    assert pointers == (
        parameter.ptr,
        gradient.ptr,
        plan.masters[0].ptr,
        plan.first_moments[0].ptr,
        plan.second_moments[0].ptr,
        plan.step_count.ptr,
    )


def test_cuda_adamw_token_normalization_and_nonfinite_graph_replay():
    device = CUDA_DEVICES[0]
    parameter = _array([1.0], wp.bfloat16, device)
    gradient = _array([4.0], wp.float32, device)
    valid_count = _array([2], wp.int32, device)
    plan = AdamWPlan(
        [parameter],
        [gradient],
        learning_rate=0.1,
        beta1=0.0,
        beta2=0.0,
        loss_scale=2.0,
        normalize_by_valid_tokens=True,
    )
    plan.accumulate_valid_tokens(valid_count)

    # Warm an independent plan so compilation is not part of capture.
    warm_parameter = _array([0.0], wp.bfloat16, device)
    warm_gradient = _array([1.0], wp.float32, device)
    warm = AdamWPlan([warm_parameter], [warm_gradient])
    warm.step()
    wp.synchronize_device(device)

    wp.capture_begin(device=device)
    try:
        plan.step()
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)
    np.testing.assert_array_equal(plan.step_count.numpy(), [2])
    np.testing.assert_allclose(plan.masters[0].numpy(), [0.8], atol=2e-7)

    gradient.assign(np.array([np.nan], dtype=np.float32))
    wp.capture_launch(graph)
    np.testing.assert_array_equal(plan.step_count.numpy(), [2])
    np.testing.assert_allclose(plan.masters[0].numpy(), [0.8], atol=2e-7)
    np.testing.assert_array_equal(plan.all_finite.numpy(), [0])
