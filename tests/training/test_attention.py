# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from tests.utilities import is_device_available
from warp_nn.flash_attention import flash_gqa_cache_forward
from warp_nn.training.attention import (
    _attention_kernels,
    gqa_attention_backward,
    gqa_attention_forward,
)


def test_gqa_attention_streaming_factory_is_bounded():
    assert len(_attention_kernels(wp.float16, 8)) == 3


def _reference(query, key, value, output_grad, lengths, scale, window):
    output = np.zeros_like(query, dtype=np.float32)
    lse = np.full(query.shape[:3], -np.finfo(np.float32).max, dtype=np.float32)
    query_grad = np.zeros_like(query, dtype=np.float32)
    key_grad = np.zeros_like(key, dtype=np.float32)
    value_grad = np.zeros_like(value, dtype=np.float32)
    heads_per_kv = query.shape[1] // key.shape[1]
    for batch, length in enumerate(lengths):
        for query_head in range(query.shape[1]):
            kv_head = query_head // heads_per_kv
            for query_token in range(int(length)):
                first_key = max(0, query_token + 1 - window) if window else 0
                keys = slice(first_key, query_token + 1)
                scores = (
                    key[batch, kv_head, keys] @ query[batch, query_head, query_token]
                ) * scale
                maximum = np.max(scores)
                probabilities = np.exp(scores - maximum)
                probabilities /= np.sum(probabilities)
                lse[batch, query_head, query_token] = maximum + np.log(
                    np.sum(np.exp(scores - maximum))
                )
                output[batch, query_head, query_token] = (
                    probabilities @ value[batch, kv_head, keys]
                )
                probability_grad = (
                    value[batch, kv_head, keys]
                    @ output_grad[batch, query_head, query_token]
                )
                score_grad = probabilities * (
                    probability_grad - probabilities @ probability_grad
                )
                query_grad[batch, query_head, query_token] += (
                    score_grad @ key[batch, kv_head, keys] * scale
                )
                key_grad[batch, kv_head, keys] += (
                    np.outer(score_grad, query[batch, query_head, query_token]) * scale
                )
                value_grad[batch, kv_head, keys] += np.outer(
                    probabilities, output_grad[batch, query_head, query_token]
                )
    return output, lse, query_grad, key_grad, value_grad


@pytest.mark.parametrize("dtype", [wp.float16, wp.bfloat16])
@pytest.mark.parametrize("window", [0, 2])
def test_gqa_attention_forward_backward_cpu(dtype, window):
    rng = np.random.default_rng(73)
    batch, query_heads, kv_heads, sequence, head_size = 2, 4, 2, 4, 3
    shape = (batch, query_heads, sequence, head_size)
    kv_shape = (batch, kv_heads, sequence, head_size)
    query = wp.array(
        rng.normal(size=shape).astype(np.float32), dtype=dtype, device="cpu"
    )
    key = wp.array(
        rng.normal(size=kv_shape).astype(np.float32), dtype=dtype, device="cpu"
    )
    value = wp.array(
        rng.normal(size=kv_shape).astype(np.float32), dtype=dtype, device="cpu"
    )
    output_grad = wp.array(
        rng.normal(size=shape).astype(np.float32), dtype=dtype, device="cpu"
    )
    lengths_np = np.array([4, 3], dtype=np.int32)
    lengths = wp.array(lengths_np, device="cpu")
    output = wp.empty(shape, dtype=dtype, device="cpu")
    lse = wp.empty(shape[:3], dtype=wp.float32, device="cpu")
    accumulator = wp.empty(shape, dtype=wp.float32, device="cpu")
    query_grad = wp.empty(shape, dtype=wp.float32, device="cpu")
    key_grad = wp.empty(kv_shape, dtype=wp.float32, device="cpu")
    value_grad = wp.empty(kv_shape, dtype=wp.float32, device="cpu")
    delta = wp.empty(shape[:3], dtype=wp.float32, device="cpu")
    scale = 0.37

    gqa_attention_forward(
        query, key, value, lengths, output, lse, accumulator, scale=scale, window=window
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

    query_np = query.numpy().astype(np.float32)
    key_np = key.numpy().astype(np.float32)
    value_np = value.numpy().astype(np.float32)
    output_grad_np = output_grad.numpy().astype(np.float32)
    reference = _reference(
        query_np, key_np, value_np, output_grad_np, lengths_np, scale, window
    )
    np.testing.assert_allclose(
        output.numpy().astype(np.float32), reference[0], atol=1.5e-2, rtol=5e-3
    )
    np.testing.assert_allclose(lse.numpy(), reference[1], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(query_grad.numpy(), reference[2], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(key_grad.numpy(), reference[3], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(value_grad.numpy(), reference[4], atol=3e-5, rtol=3e-5)
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
    np.testing.assert_allclose(
        query_grad.numpy(), 2.0 * reference[2], atol=5e-5, rtol=5e-5
    )
    np.testing.assert_allclose(
        key_grad.numpy(), 2.0 * reference[3], atol=5e-5, rtol=5e-5
    )
    np.testing.assert_allclose(
        value_grad.numpy(), 2.0 * reference[4], atol=5e-5, rtol=5e-5
    )
    assert np.all(output.numpy()[1, :, 3] == 0)
    assert lse.size == batch * query_heads * sequence


def test_gqa_attention_rejects_non_divisible_heads():
    query = wp.zeros((1, 3, 2, 2), dtype=wp.float16, device="cpu")
    key = wp.zeros((1, 2, 2, 2), dtype=wp.float16, device="cpu")
    value = wp.zeros_like(key)
    lengths = wp.array(np.array([2], dtype=np.int32), device="cpu")
    output = wp.empty_like(query)
    lse = wp.empty((1, 3, 2), dtype=wp.float32, device="cpu")
    accumulator = wp.empty(query.shape, dtype=wp.float32, device="cpu")
    with pytest.raises(ValueError, match="divisible"):
        gqa_attention_forward(query, key, value, lengths, output, lse, accumulator)


@pytest.mark.parametrize("dtype", [wp.float16, wp.bfloat16])
def test_flash_gqa_cache_forward_matches_reference_and_replays(dtype):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(91)
    query_heads, kv_heads, rows, capacity, head_size = 6, 1, 17, 64, 256
    query_np = rng.normal(0.0, 0.2, (1, query_heads, rows, head_size)).astype(
        np.float32
    )
    key_np = rng.normal(0.0, 0.2, (1, kv_heads, capacity, head_size)).astype(np.float32)
    value_np = rng.normal(0.0, 0.2, (1, kv_heads, capacity, head_size)).astype(
        np.float32
    )
    positions_np = np.arange(37, 37 + rows, dtype=np.int32)
    query = wp.array(query_np, dtype=dtype, device="cuda:0")
    key = wp.array(key_np, dtype=dtype, device="cuda:0")
    value = wp.array(value_np, dtype=dtype, device="cuda:0")
    positions = wp.array(positions_np, device="cuda:0")
    length = wp.array(np.array([positions_np[-1]], dtype=np.int32), device="cuda:0")
    output = wp.empty_like(query)
    lse = wp.empty((1, query_heads, rows), dtype=wp.float32, device="cuda:0")
    workspace = wp.empty(query.shape, dtype=wp.float32, device="cuda:0")

    def launch():
        flash_gqa_cache_forward(
            query, key, value, length, positions, output, lse, workspace
        )

    launch()
    wp.synchronize_device("cuda:0")
    wp.capture_begin(device="cuda:0")
    launch()
    graph = wp.capture_end(device="cuda:0")
    wp.capture_launch(graph)
    wp.synchronize_device("cuda:0")

    reference = np.empty_like(query_np)
    scale = head_size**-0.5
    for head in range(query_heads):
        for row, position in enumerate(positions_np):
            scores = key_np[0, 0, : position + 1] @ query_np[0, head, row] * scale
            probabilities = np.exp(scores - np.max(scores))
            probabilities /= np.sum(probabilities)
            reference[0, head, row] = probabilities @ value_np[0, 0, : position + 1]
    np.testing.assert_allclose(
        output.numpy().astype(np.float32), reference, atol=8.0e-4, rtol=8.0e-3
    )
