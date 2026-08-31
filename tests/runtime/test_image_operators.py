# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.operators import Conv2dPlan, conv2d_output_shape


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
