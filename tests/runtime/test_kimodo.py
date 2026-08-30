# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import warp as wp

from warp_nn.runtime.kimodo import (
    KimodoConfig,
    KimodoDiffusionPlan,
    KimodoStats,
    cosine_ddim_schedule,
)


def test_soma_config_and_cosine_schedule():
    config = KimodoConfig.soma_v1()
    assert config.motion_dim == 369
    assert config.body_dim == 364
    selected, alpha, previous = cosine_ddim_schedule(1000, 100)
    assert selected.shape == alpha.shape == previous.shape == (1000,)
    assert selected[0] == 0 and selected[-1] == 999
    assert previous[0] == 1.0
    np.testing.assert_allclose(previous[1:], alpha[:-1])


def test_motion_condition_root_and_ddim_cpu():
    config = KimodoConfig(33, 2, 30.0, 8, 16, 1, 2, text_dim=8, text_tokens=2)

    def zeros(size):
        return np.zeros(size, dtype=np.float32)

    def ones(size):
        return np.ones(size, dtype=np.float32)

    stats = KimodoStats(
        zeros(5),
        ones(5),
        zeros(4),
        ones(4),
        zeros(config.body_dim),
        ones(config.body_dim),
    )
    plan = KimodoDiffusionPlan(1, 3, config, stats, device="cpu")
    motion = np.arange(3 * config.motion_dim, dtype=np.float32).reshape(1, 3, -1) / 100
    observed = np.zeros_like(motion)
    observed[0, 1, 7] = -3.0
    mask = np.zeros_like(motion, dtype=bool)
    mask[0, 1, 7] = True
    plan.motion.assign(motion)
    plan.observed.assign(observed)
    plan.mask.assign(mask)
    plan.apply_conditions()
    conditioned = motion.copy()
    conditioned[mask] = observed[mask]
    np.testing.assert_array_equal(plan.conditioned.numpy(), conditioned)

    root = np.zeros((1, 3, 5), dtype=np.float32)
    root[0, :, 0] = [0, 1, 3]
    root[0, :, 2] = [0, -1, -1]
    angles = np.array([0.0, math.pi / 2, math.pi], dtype=np.float32)
    root[0, :, 3] = np.cos(angles)
    root[0, :, 4] = np.sin(angles)
    plan.lengths.assign(np.array([3], dtype=np.int32))
    local = plan.root_to_local(wp.array(root, dtype=wp.float32, device="cpu")).numpy()
    # Unit std is represented as sqrt(1 + epsilon), exactly as Kimodo Stats.
    scale = math.sqrt(1.0 + stats.epsilon)
    np.testing.assert_allclose(local[0, :2, 0], (math.pi / 2 * 30) / scale, rtol=1e-5)
    np.testing.assert_allclose(local[0, :, 1], [30, 60, 60] / scale, rtol=1e-5)
    np.testing.assert_allclose(local[0, :, 2], [-30, 0, 0] / scale, atol=1e-5)

    plan.motion.assign(np.full_like(motion, 0.5))
    plan.clean.assign(np.full_like(motion, 0.25))
    result = plan.step(0.6, 0.8).numpy()
    noise = (0.5 / math.sqrt(0.6) - 0.25) / math.sqrt(0.4 / 0.6)
    expected = 0.25 * math.sqrt(0.8) + math.sqrt(0.2) * noise
    np.testing.assert_allclose(result, expected, rtol=1e-6)
