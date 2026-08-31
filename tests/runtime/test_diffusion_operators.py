# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.operators import AdaptiveRMSNormPlan, ModulatedResidualPlan


def test_adaptive_rms_and_gated_residual_match_reference_and_graph():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(127)
    x = rng.normal(0.0, 0.3, size=(2, 5, 16)).astype(np.float32)
    weight = rng.normal(1.0, 0.1, size=16).astype(np.float32)
    table = rng.normal(0.0, 0.1, size=(1, 6, 16)).astype(np.float32)
    timestep = rng.normal(0.0, 0.1, size=(2, 6, 16)).astype(np.float32)
    branch = rng.normal(0.0, 0.2, size=x.shape).astype(np.float32)
    arrays = [
        wp.array(value, dtype=wp.bfloat16, device="cuda:0")
        for value in (x, weight, table, timestep, branch)
    ]
    norm = AdaptiveRMSNormPlan(
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        shift_index=0,
        scale_index=1,
    )
    residual = ModulatedResidualPlan(
        arrays[0],
        arrays[4],
        scale_shift_table=arrays[2],
        timestep_modulation=arrays[3],
        gate_index=2,
    )
    wp.capture_begin(device="cuda:0")
    norm.execute()
    residual.execute()
    graph = wp.capture_end(device="cuda:0")
    wp.capture_launch(graph)

    normalized = x * weight / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1.0e-6)
    expected_norm = normalized * (1.0 + table[:, 1] + timestep[:, 1, None]) + (
        table[:, 0] + timestep[:, 0, None]
    )
    expected_residual = x + branch * (table[:, 2] + timestep[:, 2, None])
    np.testing.assert_allclose(norm.output.numpy(), expected_norm, rtol=0.04, atol=0.02)
    np.testing.assert_allclose(
        residual.output.numpy(), expected_residual, rtol=0.04, atol=0.02
    )


def test_plain_residual_does_not_require_modulation():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    left = np.arange(24, dtype=np.float32).reshape(1, 3, 8) * 0.01
    right = np.flip(left, axis=-1).copy()
    plan = ModulatedResidualPlan(
        wp.array(left, dtype=wp.float16, device="cuda:0"),
        wp.array(right, dtype=wp.float16, device="cuda:0"),
    )
    np.testing.assert_allclose(plan.execute().numpy(), left + right, atol=1.0e-3)
