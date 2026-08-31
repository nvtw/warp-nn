# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.operators import Conv1dPlan, Snake1dPlan, conv1d_output_length


def _conv_reference(
    x, weight, bias, stride, padding, dilation, transposed, output_padding=0
):
    if transposed:
        length = conv1d_output_length(
            x.shape[1],
            weight.shape[2],
            stride=stride,
            padding=padding,
            dilation=dilation,
            transposed=True,
            output_padding=output_padding,
        )
        output = np.broadcast_to(bias, (x.shape[0], length, bias.size)).copy()
        for batch in range(x.shape[0]):
            for source in range(x.shape[1]):
                for kernel in range(weight.shape[2]):
                    target = source * stride - padding + kernel * dilation
                    if 0 <= target < length:
                        output[batch, target] += x[batch, source] @ weight[:, :, kernel]
        return output
    length = conv1d_output_length(
        x.shape[1], weight.shape[2], stride=stride, padding=padding, dilation=dilation
    )
    output = np.broadcast_to(bias, (x.shape[0], length, bias.size)).copy()
    for batch in range(x.shape[0]):
        for target in range(length):
            for kernel in range(weight.shape[2]):
                source = target * stride - padding + kernel * dilation
                if 0 <= source < x.shape[1]:
                    output[batch, target] += weight[:, :, kernel] @ x[batch, source]
    return output


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("transposed", [False, True])
def test_conv1d_plan_matches_reference(device, transposed):
    if not is_device_available(device):
        pytest.skip(f"{device} is unavailable")
    rng = np.random.default_rng(43)
    x = rng.normal(size=(2, 7, 3)).astype(np.float32)
    weight_shape = (3, 4, 3) if transposed else (4, 3, 3)
    weight = rng.normal(size=weight_shape).astype(np.float32)
    bias = rng.normal(size=4).astype(np.float32)
    plan = Conv1dPlan(
        wp.array(x, device=device),
        wp.array(weight, device=device),
        wp.array(bias, device=device),
        stride=2,
        padding=2,
        dilation=2,
        transposed=transposed,
        output_padding=1 if transposed else 0,
    )
    actual = plan.execute().numpy()
    expected = _conv_reference(
        x, weight, bias, 2, 2, 2, transposed, 1 if transposed else 0
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_snake1d_matches_reference_and_cuda_graph():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(47)
    x = rng.normal(size=(2, 9, 4)).astype(np.float32)
    alpha = rng.normal(size=4).astype(np.float32) * 0.2
    beta = rng.normal(size=4).astype(np.float32) * 0.2
    plan = Snake1dPlan(
        wp.array(x, device="cuda:0"),
        wp.array(alpha, device="cuda:0"),
        wp.array(beta, device="cuda:0"),
    )
    wp.capture_begin(device="cuda:0")
    plan.execute()
    graph = wp.capture_end(device="cuda:0")
    wp.capture_launch(graph)
    actual = plan.output.numpy()
    a = np.exp(alpha)
    b = np.exp(beta)
    expected = x + np.sin(x * a) ** 2 / (b + 1.0e-9)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_conv1d_mma_matches_reference_and_cuda_graph():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(53)
    x = rng.normal(size=(2, 37, 32)).astype(np.float16)
    weight = rng.normal(scale=0.1, size=(32, 32, 7)).astype(np.float16)
    bias = rng.normal(scale=0.1, size=32).astype(np.float16)
    plan = Conv1dPlan(
        wp.array(x, device="cuda:0"),
        wp.array(weight, device="cuda:0"),
        wp.array(bias, device="cuda:0"),
        padding=3,
    )
    assert plan._use_mma
    wp.synchronize_device("cuda:0")
    wp.capture_begin(device="cuda:0")
    plan.execute()
    graph = wp.capture_end(device="cuda:0")
    wp.capture_launch(graph)
    actual = plan.output.numpy().astype(np.float32)
    expected = _conv_reference(
        x.astype(np.float32),
        weight.astype(np.float32),
        bias.astype(np.float32),
        1,
        3,
        1,
        False,
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-3, atol=2.0e-3)


@pytest.mark.parametrize("stride", [2, 6, 10])
def test_conv_transpose1d_mma_matches_reference(stride):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(59 + stride)
    x = rng.normal(size=(2, 7, 32)).astype(np.float16)
    weight = rng.normal(scale=0.05, size=(32, 32, 2 * stride)).astype(np.float16)
    bias = rng.normal(scale=0.1, size=32).astype(np.float16)
    plan = Conv1dPlan(
        wp.array(x, device="cuda:0"),
        wp.array(weight, device="cuda:0"),
        wp.array(bias, device="cuda:0"),
        stride=stride,
        padding=(stride + 1) // 2,
        transposed=True,
    )
    assert plan._use_transpose_mma
    wp.synchronize_device("cuda:0")
    wp.capture_begin(device="cuda:0")
    plan.execute()
    graph = wp.capture_end(device="cuda:0")
    wp.capture_launch(graph)
    actual = plan.output.numpy().astype(np.float32)
    expected = _conv_reference(
        x.astype(np.float32),
        weight.astype(np.float32),
        bias.astype(np.float32),
        stride,
        (stride + 1) // 2,
        1,
        True,
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-3, atol=2.0e-3)
