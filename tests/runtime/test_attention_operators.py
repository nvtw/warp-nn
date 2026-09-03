# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.operators import BidirectionalGQAPlan, FixedKVAttentionPlan


def _reference(query, key, value, query_valid, key_valid, window=None):
    batch, query_heads, query_length, head_size = query.shape
    kv_heads = key.shape[1]
    output = np.zeros_like(query)
    for b in range(batch):
        for head in range(query_heads):
            kv_head = head // (query_heads // kv_heads)
            for q in range(query_length):
                if not query_valid[b, q]:
                    continue
                indices = [
                    k
                    for k in range(key.shape[2])
                    if key_valid[b, k] and (window is None or abs(q - k) <= window)
                ]
                if not indices:
                    continue
                scores = np.array(
                    [query[b, head, q] @ key[b, kv_head, k] for k in indices],
                    dtype=np.float32,
                ) / np.sqrt(head_size)
                weights = np.exp(scores - scores.max())
                weights /= weights.sum()
                for weight, k in zip(weights, indices, strict=True):
                    output[b, head, q] += weight * value[b, kv_head, k]
    return output


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("window", [None, 2])
def test_bidirectional_gqa_matches_cpu_reference(device, window):
    if not is_device_available(device):
        pytest.skip(f"{device} is unavailable")
    rng = np.random.default_rng(101)
    query = rng.normal(size=(2, 4, 7, 8)).astype(np.float32)
    key = rng.normal(size=(2, 2, 7, 8)).astype(np.float32)
    value = rng.normal(size=(2, 2, 7, 8)).astype(np.float32)
    query_valid = np.array([[True, True, True, True, True, False, False], [True] * 7])
    key_valid = np.array(
        [
            [True, True, True, True, False, False, False],
            [False, True, True, True, True, True, True],
        ]
    )
    plan = BidirectionalGQAPlan(
        wp.array(query, device=device),
        wp.array(key, device=device),
        wp.array(value, device=device),
        query_valid=wp.array(query_valid, device=device),
        key_valid=wp.array(key_valid, device=device),
        window=window,
    )
    actual = plan.execute().numpy()
    expected = _reference(query, key, value, query_valid, key_valid, window)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_fixed_kv_cross_attention_matches_reference_and_cuda_graph():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(103)
    query = rng.normal(size=(1, 6, 5, 16)).astype(np.float32)
    key = rng.normal(size=(1, 2, 9, 16)).astype(np.float32)
    value = rng.normal(size=(1, 2, 9, 16)).astype(np.float32)
    query_valid = np.ones((1, 5), dtype=np.bool_)
    key_valid = np.array([[True] * 7 + [False] * 2])
    plan = FixedKVAttentionPlan(
        wp.array(query, device="cuda:0"),
        wp.array(key, device="cuda:0"),
        wp.array(value, device="cuda:0"),
        query_valid=wp.array(query_valid, device="cuda:0"),
        key_valid=wp.array(key_valid, device="cuda:0"),
    )
    wp.capture_begin(device="cuda:0")
    plan.execute()
    graph = wp.capture_end(device="cuda:0")
    wp.capture_launch(graph)
    expected = _reference(query, key, value, query_valid, key_valid)
    np.testing.assert_allclose(plan.output.numpy(), expected, rtol=3.0e-5, atol=3.0e-5)


@pytest.mark.parametrize("dtype", [wp.float16, wp.bfloat16])
def test_attention_low_precision_matches_fp32_reference(dtype):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(107)
    query = rng.normal(0.0, 0.3, size=(1, 4, 6, 16)).astype(np.float32)
    key = rng.normal(0.0, 0.3, size=(1, 2, 6, 16)).astype(np.float32)
    value = rng.normal(0.0, 0.3, size=(1, 2, 6, 16)).astype(np.float32)
    valid = np.ones((1, 6), dtype=np.bool_)
    plan = BidirectionalGQAPlan(
        wp.array(query, dtype=dtype, device="cuda:0"),
        wp.array(key, dtype=dtype, device="cuda:0"),
        wp.array(value, dtype=dtype, device="cuda:0"),
        window=3,
    )
    expected = _reference(query, key, value, valid, valid, 3)
    np.testing.assert_allclose(plan.execute().numpy(), expected, rtol=0.02, atol=0.01)


@pytest.mark.parametrize("head_size", [128, 512])
def test_tiled_attention_matches_reference_across_query_tiles(head_size):
    """Exercise more than one query tile at production head width."""
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(109)
    query = rng.normal(0.0, 0.2, size=(1, 4, 37, head_size)).astype(np.float32)
    key = rng.normal(0.0, 0.2, size=(1, 2, 43, head_size)).astype(np.float32)
    value = rng.normal(0.0, 0.2, size=(1, 2, 43, head_size)).astype(np.float32)
    query_valid = np.array([[True] * 35 + [False] * 2])
    key_valid = np.array([[True] * 40 + [False] * 3])
    plan = BidirectionalGQAPlan(
        wp.array(query, dtype=wp.bfloat16, device="cuda:0"),
        wp.array(key, dtype=wp.bfloat16, device="cuda:0"),
        wp.array(value, dtype=wp.bfloat16, device="cuda:0"),
        query_valid=wp.array(query_valid, device="cuda:0"),
        key_valid=wp.array(key_valid, device="cuda:0"),
    )
    expected = _reference(query, key, value, query_valid, key_valid)
    np.testing.assert_allclose(plan.execute().numpy(), expected, rtol=0.025, atol=0.012)


def test_attention_rejects_incompatible_geometry():
    query = wp.zeros((1, 3, 4, 8), dtype=wp.float32, device="cpu")
    key = wp.zeros((1, 2, 4, 8), dtype=wp.float32, device="cpu")
    value = wp.zeros_like(key)
    with pytest.raises(ValueError, match="head geometry"):
        BidirectionalGQAPlan(query, key, value)
    with pytest.raises(ValueError, match="equal Q/K"):
        BidirectionalGQAPlan(
            wp.zeros((1, 4, 3, 8), dtype=wp.float32, device="cpu"),
            key,
            value,
            window=2,
        )
