# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime._cublas import try_create_cublas
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
