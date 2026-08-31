# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.operators import (
    AdaptiveRMSNormPlan,
    ChannelAffinePlan,
    FlowEulerPlan,
    ModulatedResidualPlan,
    SpatialPatchPackPlan,
    SpatialPatchUnpackPlan,
    flow_match_euler_schedule,
)


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


def test_spatial_patch_roundtrip_and_channel_affine():
    x = np.arange(2 * 3 * 4 * 6, dtype=np.float32).reshape(2, 3, 4, 6)
    source = wp.array(x, device="cpu")
    pack = SpatialPatchPackPlan(source, 2)
    packed = pack.execute()
    unpack = SpatialPatchUnpackPlan(packed, 4, 6, 2)

    expected = x.reshape(2, 3, 2, 2, 3, 2).transpose(0, 2, 4, 1, 3, 5).reshape(2, 6, 12)
    np.testing.assert_array_equal(packed.numpy(), expected)
    np.testing.assert_array_equal(unpack.execute().numpy(), x)

    scale = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    bias = np.array([1.0, 2.0, -3.0], dtype=np.float32)
    affine = ChannelAffinePlan(
        source,
        wp.array(scale, device="cpu"),
        wp.array(bias, device="cpu"),
    )
    expected_affine = x * scale[None, :, None, None] + bias[None, :, None, None]
    np.testing.assert_array_equal(affine.execute().numpy(), expected_affine)


def test_flow_match_schedule_and_euler_update():
    schedule = flow_match_euler_schedule(
        8,
        6889,
        base_sequence_length=256,
        maximum_sequence_length=8192,
        base_shift=0.5,
        maximum_shift=0.9,
        terminal_shift=0.02,
    )
    assert schedule.shape == (9,)
    assert schedule[0] == 1.0
    assert schedule[-2] == pytest.approx(0.02, abs=1.0e-7)
    assert schedule[-1] == 0.0
    assert np.all(np.diff(schedule) < 0.0)

    sample_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.1
    velocity_np = np.full_like(sample_np, 0.25)
    sample = wp.array(sample_np, device="cpu")
    velocity = wp.array(velocity_np, device="cpu")
    sigma = wp.array(np.array([0.8, 0.6], dtype=np.float32), device="cpu")
    next_sigma = wp.array(np.array([0.5, 0.2], dtype=np.float32), device="cpu")
    plan = FlowEulerPlan(sample, velocity, sigma, next_sigma)
    expected = sample_np + (
        np.array([-0.3, -0.4], dtype=np.float32)[:, None, None] * velocity_np
    )
    np.testing.assert_allclose(plan.execute().numpy(), expected, atol=1.0e-6)


def test_spatial_diffusion_plans_capture_on_cuda():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(81)
    x = wp.array(
        rng.normal(size=(1, 4, 8, 8)).astype(np.float32),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    pack = SpatialPatchPackPlan(x, 2)
    unpack = SpatialPatchUnpackPlan(pack.output, 8, 8, 2)
    sigma = wp.array([0.7], dtype=wp.float32, device="cuda:0")
    next_sigma = wp.array([0.5], dtype=wp.float32, device="cuda:0")
    flow = FlowEulerPlan(pack.output, wp.zeros_like(pack.output), sigma, next_sigma)

    wp.capture_begin(device="cuda:0")
    pack.execute()
    flow.execute()
    unpack.execute()
    graph = wp.capture_end(device="cuda:0")
    wp.capture_launch(graph)

    np.testing.assert_allclose(unpack.output.numpy(), x.numpy(), atol=0.01)
