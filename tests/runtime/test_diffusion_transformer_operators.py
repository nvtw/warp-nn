# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.operators import (
    AdaptiveLayerNormPlan,
    AttentionHeadsPlan,
    AttentionMergePlan,
    BiasedLinearPlan,
    BroadcastGatedResidualPlan,
    JointBidirectionalAttentionPlan,
    LayerNormPlan,
    RotaryCachePlan,
    SequenceSlicePlan,
    SinusoidalEmbeddingPlan,
    multi_axis_rotary_cache_values,
)


def test_multi_axis_rotary_cache_matches_explicit_frequencies():
    coordinates = np.array([[0, -1, 2], [1, 0, 3]], dtype=np.float32)
    cosine, sine = multi_axis_rotary_cache_values(coordinates, (2, 4, 2), 100.0)
    expected_angles = np.concatenate(
        (
            coordinates[:, 0:1],
            coordinates[:, 1:2]
            * (1.0 / np.power(100.0, np.arange(0, 4, 2) / 4))[None],
            coordinates[:, 2:3],
        ),
        axis=1,
    )
    np.testing.assert_allclose(cosine, np.cos(expected_angles), rtol=1e-6)
    np.testing.assert_allclose(sine, np.sin(expected_angles), rtol=1e-6)


@pytest.mark.skipif(not is_device_available("cuda:0"), reason="CUDA is unavailable")
def test_transformer_layout_plans_match_reference_and_replay_graph():
    rng = np.random.default_rng(317)
    values = rng.normal(0.0, 0.3, (1, 5, 6)).astype(np.float32)
    weight = rng.normal(0.0, 0.2, (6, 6)).astype(np.float32)
    bias = rng.normal(0.0, 0.1, 6).astype(np.float32)
    x = wp.array(values, dtype=wp.bfloat16, device="cuda:0")
    linear = BiasedLinearPlan(
        x,
        wp.array(weight, dtype=wp.bfloat16, device="cuda:0"),
        wp.array(bias, dtype=wp.bfloat16, device="cuda:0"),
        activation="gelu_tanh",
    )
    norm = LayerNormPlan(linear.output)
    sliced = SequenceSlicePlan(norm.output, 1, 3)

    linear.execute()
    norm.execute()
    sliced.execute()
    projected = values @ weight.T + bias
    expected = 0.5 * projected * (
        1.0
        + np.tanh(
            np.sqrt(2.0 / np.pi)
            * (projected + 0.044715 * projected * projected * projected)
        )
    )
    expected = (expected - expected.mean(axis=-1, keepdims=True)) / np.sqrt(
        expected.var(axis=-1, keepdims=True) + 1.0e-6
    )
    np.testing.assert_allclose(sliced.output.numpy(), expected[:, 1:4], rtol=0.04, atol=0.025)

    wp.synchronize()
    wp.capture_begin(device="cuda:0")
    linear.execute()
    norm.execute()
    sliced.execute()
    graph = wp.capture_end(device="cuda:0")
    replacement = rng.normal(0.0, 0.2, values.shape).astype(np.float32)
    x.assign(replacement)
    wp.capture_launch(graph)
    assert np.isfinite(sliced.output.numpy()).all()


@pytest.mark.skipif(not is_device_available("cuda:0"), reason="CUDA is unavailable")
def test_adaptive_rope_joint_attention_and_sinusoidal_plans():
    x_values = np.arange(24, dtype=np.float32).reshape(1, 2, 12) / 24.0
    x = wp.array(x_values, dtype=wp.bfloat16, device="cuda:0")
    heads = AttentionHeadsPlan(x, 2)
    heads.execute()
    cosine = wp.ones((2, 3), dtype=wp.bfloat16, device="cuda:0")
    sine = wp.zeros((2, 3), dtype=wp.bfloat16, device="cuda:0")
    rope = RotaryCachePlan(heads.output, cosine, sine)
    rope.execute()
    merge = AttentionMergePlan(rope.output)
    merge.execute()
    np.testing.assert_allclose(merge.output.numpy(), x_values, atol=0.004)

    first = rope.output
    second = wp.array(
        np.flip(heads.output.numpy(), axis=2).copy(),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    valid = wp.array([[True, False]], dtype=wp.bool, device="cuda:0")
    joint = JointBidirectionalAttentionPlan(
        (first, first, first),
        (second, second, second),
        first_valid=valid,
    )
    first_output, second_output = joint.execute()
    assert first_output.shape == first.shape
    assert second_output.shape == second.shape
    assert np.isfinite(first_output.numpy()).all()

    modulation = wp.array(
        np.full((1, 3, 12), 0.1, dtype=np.float32),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    adaptive = AdaptiveLayerNormPlan(x, modulation, shift_index=0, scale_index=1)
    adaptive.execute()
    residual = BroadcastGatedResidualPlan(
        x, adaptive.output, modulation, gate_index=2
    )
    residual.execute()
    assert np.isfinite(residual.output.numpy()).all()

    timesteps = wp.array([0.5], dtype=wp.float32, device="cuda:0")
    sinusoidal = SinusoidalEmbeddingPlan(
        timesteps,
        6,
        dtype=wp.bfloat16,
        scale=1000.0,
        frequency_shift=0.0,
        flip_sin_cos=True,
    )
    actual = sinusoidal.execute().numpy()
    frequencies = np.exp(-np.log(10000.0) * np.arange(3) / 3)
    angles = 500.0 * frequencies
    expected = np.concatenate((np.cos(angles), np.sin(angles)))[None]
    np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.005)
