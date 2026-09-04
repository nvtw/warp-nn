# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.operators import (
    ClampPlan,
    Conv2dPlan,
    NearestUpsample2dPlan,
    ResidualAddPlan,
    SpatialRMSNormPlan,
    SpatialSelfAttentionPlan,
    conv2d_output_shape,
)


def _reference(x, weight, bias, stride=(1, 1), padding=(0, 0, 0, 0)):
    batch, height, width, _ = x.shape
    out_channels, in_channels, kernel_y, kernel_x = weight.shape
    top, bottom, left, right = padding
    output_y, output_x = conv2d_output_shape(
        height,
        width,
        (kernel_y, kernel_x),
        stride=stride,
        padding=padding,
    )
    output = np.empty((batch, output_y, output_x, out_channels), dtype=np.float32)
    for b in range(batch):
        for y in range(output_y):
            for x_column in range(output_x):
                for out_channel in range(out_channels):
                    total = 0.0 if bias is None else float(bias[out_channel])
                    for ky in range(kernel_y):
                        source_y = y * stride[0] - top + ky
                        if not 0 <= source_y < height:
                            continue
                        for kx in range(kernel_x):
                            source_x = x_column * stride[1] - left + kx
                            if not 0 <= source_x < width:
                                continue
                            total += np.dot(
                                x[b, source_y, source_x],
                                weight[out_channel, :, ky, kx],
                            )
                    output[b, y, x_column, out_channel] = total
    return output


@pytest.mark.parametrize(
    ("stride", "padding"),
    [((1, 1), (1, 1, 1, 1)), ((2, 2), (0, 1, 0, 1))],
)
def test_conv2d_cpu_matches_reference(stride, padding):
    rng = np.random.default_rng(911)
    x = rng.normal(0.0, 0.2, size=(1, 7, 9, 3)).astype(np.float32)
    weight = rng.normal(0.0, 0.2, size=(5, 3, 3, 3)).astype(np.float32)
    bias = rng.normal(0.0, 0.1, size=5).astype(np.float32)
    plan = Conv2dPlan(
        wp.array(x, device="cpu"),
        wp.array(weight, device="cpu"),
        wp.array(bias, device="cpu"),
        stride=stride,
        padding=padding,
    )
    expected = _reference(x, weight, bias, stride, padding)
    np.testing.assert_allclose(plan.execute().numpy(), expected, atol=2.0e-6)


def test_conv2d_tensor_core_matches_reference_and_captures():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(912)
    x = rng.normal(0.0, 0.08, size=(1, 8, 48, 16)).astype(np.float32)
    weight = rng.normal(0.0, 0.08, size=(32, 16, 3, 3)).astype(np.float32)
    bias = rng.normal(0.0, 0.02, size=32).astype(np.float32)
    plan = Conv2dPlan(
        wp.array(x, dtype=wp.bfloat16, device="cuda:0"),
        wp.array(weight, dtype=wp.bfloat16, device="cuda:0"),
        wp.array(bias, dtype=wp.bfloat16, device="cuda:0"),
        padding=1,
    )
    assert plan.uses_tensor_cores
    fallback = Conv2dPlan(
        plan.input,
        wp.array(weight, dtype=wp.bfloat16, device="cuda:0"),
        wp.array(bias, dtype=wp.bfloat16, device="cuda:0"),
        padding=1,
        tensor_cores=False,
    )

    plan.execute()
    fallback.execute()
    wp.synchronize_device("cuda:0")
    wp.capture_begin(device="cuda:0")
    plan.execute()
    graph = wp.capture_end(device="cuda:0")
    wp.capture_launch(graph)

    np.testing.assert_allclose(
        plan.output.numpy(), fallback.output.numpy(), rtol=0.02, atol=0.012
    )
    expected = _reference(x, weight, bias, padding=(1, 1, 1, 1))
    np.testing.assert_allclose(plan.output.numpy(), expected, rtol=0.035, atol=0.018)


def test_conv2d_rejects_channel_mismatch():
    x = wp.zeros((1, 4, 4, 3), dtype=wp.float32, device="cpu")
    weight = wp.zeros((4, 2, 3, 3), dtype=wp.float32, device="cpu")
    with pytest.raises(ValueError, match="weight channels"):
        Conv2dPlan(x, weight)


def test_spatial_vae_primitives_cpu():
    rng = np.random.default_rng(913)
    values = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    gamma = rng.normal(size=4).astype(np.float32)
    x = wp.array(values, device="cpu")
    norm = SpatialRMSNormPlan(x, wp.array(gamma, device="cpu"), silu=True)
    normalized = values / np.sqrt(
        np.mean(values * values, axis=-1, keepdims=True) + 1.0e-12
    )
    normalized *= gamma
    expected_norm = normalized / (1.0 + np.exp(-normalized))
    np.testing.assert_allclose(norm.execute().numpy(), expected_norm, atol=2.0e-6)

    upsample = NearestUpsample2dPlan(x, 2)
    expected_up = np.repeat(np.repeat(values, 2, axis=1), 2, axis=2)
    np.testing.assert_array_equal(upsample.execute().numpy(), expected_up)

    residual = ResidualAddPlan(x, x)
    np.testing.assert_array_equal(residual.execute().numpy(), values * 2.0)


def test_spatial_self_attention_matches_reference_and_captures():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(914)
    batch, height, width, channels = 1, 2, 3, 32
    values = rng.normal(0.0, 0.08, size=(batch, height, width, channels)).astype(
        np.float32
    )
    gamma = rng.normal(1.0, 0.05, size=channels).astype(np.float32)
    qkv_weight = rng.normal(0.0, 0.08, size=(channels * 3, channels, 1, 1)).astype(
        np.float32
    )
    qkv_bias = rng.normal(0.0, 0.02, size=channels * 3).astype(np.float32)
    projection_weight = rng.normal(0.0, 0.08, size=(channels, channels, 1, 1)).astype(
        np.float32
    )
    projection_bias = rng.normal(0.0, 0.02, size=channels).astype(np.float32)
    arrays = [
        wp.array(value, dtype=wp.bfloat16, device="cuda:0")
        for value in (
            values,
            gamma,
            qkv_weight,
            qkv_bias,
            projection_weight,
            projection_bias,
        )
    ]
    plan = SpatialSelfAttentionPlan(*arrays)
    plan.execute()
    wp.synchronize_device("cuda:0")
    wp.capture_begin(device="cuda:0")
    plan.execute()
    graph = wp.capture_end(device="cuda:0")
    wp.capture_launch(graph)

    normalized = values / np.sqrt(
        np.mean(values * values, axis=-1, keepdims=True) + 1.0e-12
    )
    normalized *= gamma
    tokens = normalized.reshape(-1, channels)
    qkv = tokens @ qkv_weight[:, :, 0, 0].T + qkv_bias
    query, key, value = np.split(qkv, 3, axis=-1)
    scores = query @ key.T / np.sqrt(channels)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    attended = probabilities @ value
    projected = attended @ projection_weight[:, :, 0, 0].T + projection_bias
    expected = values + projected.reshape(values.shape)
    np.testing.assert_allclose(plan.output.numpy(), expected, rtol=0.055, atol=0.025)


def test_clamp_plan_preserves_shape_and_values():
    values = np.array([[-2.0, -0.5, 0.25, 3.0]], dtype=np.float32).reshape(1, 2, 2, 1)
    plan = ClampPlan(wp.array(values, device="cpu"), -1.0, 1.0)
    assert plan.output.shape == values.shape
    np.testing.assert_array_equal(plan.execute().numpy(), np.clip(values, -1.0, 1.0))


def test_depthwise_conv2d_matches_reference():
    rng = np.random.default_rng(915)
    x = rng.normal(size=(1, 5, 4, 3)).astype(np.float32)
    weight = rng.normal(size=(3, 1, 3, 3)).astype(np.float32)
    bias = rng.normal(size=3).astype(np.float32)
    plan = Conv2dPlan(
        wp.array(x, device="cpu"),
        wp.array(weight, device="cpu"),
        wp.array(bias, device="cpu"),
        padding=1,
        groups=3,
    )
    expected = np.empty_like(x)
    for row in range(x.shape[1]):
        for column in range(x.shape[2]):
            for channel in range(x.shape[3]):
                total = bias[channel]
                for ky in range(3):
                    for kx in range(3):
                        sy, sx = row - 1 + ky, column - 1 + kx
                        if 0 <= sy < x.shape[1] and 0 <= sx < x.shape[2]:
                            total += x[0, sy, sx, channel] * weight[channel, 0, ky, kx]
                expected[0, row, column, channel] = total
    np.testing.assert_allclose(
        plan.execute().numpy(), expected, rtol=2.0e-5, atol=2.0e-5
    )
