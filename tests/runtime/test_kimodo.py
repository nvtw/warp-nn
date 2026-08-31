# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.kimodo.runner import (
    KimodoConfig,
    KimodoDenoiserPlan,
    KimodoDiffusionPlan,
    KimodoGenerationPlan,
    KimodoStats,
    cosine_ddim_schedule,
    decode_motion_features,
    load_kimodo_config,
    save_motion_npz,
)


def test_soma_config_and_cosine_schedule():
    config = KimodoConfig.soma_v1()
    assert config.motion_dim == 369
    assert config.body_dim == 364
    selected, alpha, previous = cosine_ddim_schedule(1000, 100)
    assert selected.shape == alpha.shape == previous.shape == (100,)
    assert selected[0] == 0 and selected[-1] == 999
    assert previous[0] == 1.0
    np.testing.assert_allclose(previous[1:], alpha[:-1])


def test_official_config_nested_stats_and_portable_decode(tmp_path):
    (tmp_path / "config.yaml").write_text(
        """num_base_steps: 1000
motion_mask_mode: concat
fps: 30
skeleton:
  _target_: kimodo.skeleton.SOMASkeleton30
llm_shape:
- 1
- 4096
latent_dim: 1024
ff_size: 2048
num_layers: 16
num_heads: 8
num_text_tokens_override: 50
input_first_heading_angle: true
""",
        encoding="utf-8",
    )
    config = load_kimodo_config(tmp_path / "config.yaml")
    assert config == KimodoConfig.soma_v1()

    stats_root = tmp_path / "stats" / "motion"
    widths = {"global_root": 5, "local_root": 4, "body": config.body_dim}
    for group, width in widths.items():
        folder = stats_root / group
        folder.mkdir(parents=True)
        np.save(folder / "mean.npy", np.zeros(width, dtype=np.float32))
        np.save(folder / "std.npy", np.ones(width, dtype=np.float32))
    stats = KimodoStats.load(tmp_path / "stats")
    features = np.zeros((2, config.motion_dim), dtype=np.float32)
    decoded = decode_motion_features(features, stats, config.joints)
    assert decoded["posed_joints"].shape == (2, config.joints, 3)
    assert decoded["global_rot_mats"].shape == (2, config.joints, 3, 3)
    output = tmp_path / "motion.npz"
    save_motion_npz(output, decoded, fps=config.fps)
    with np.load(output) as saved:
        assert saved["fps"] == 30 and "root_positions" in saved


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
    np.testing.assert_allclose(
        local[0, :, 1], np.array([30, 60, 60]) / scale, rtol=1e-5
    )
    np.testing.assert_allclose(
        local[0, :, 2], np.array([-30, 0, 0]) / scale, rtol=1e-5, atol=1e-5
    )

    plan.motion.assign(np.full_like(motion, 0.5))
    plan.clean.assign(np.full_like(motion, 0.25))
    result = plan.step(0.6, 0.8).numpy()
    noise = (0.5 / math.sqrt(0.6) - 0.25) / math.sqrt(0.4 / 0.6)
    expected = 0.25 * math.sqrt(0.8) + math.sqrt(0.2) * noise
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def _tiny_weights(config, device="cpu", dtype=wp.float32):
    rng = np.random.default_rng(5)
    arrays = {}

    def add(name, shape, scale=0.05):
        arrays[name] = wp.array(
            rng.normal(0, scale, shape).astype(np.float32),
            dtype=dtype,
            device=device,
        )

    stages = (
        ("root_model", config.motion_dim * 2, 5),
        ("body_model", config.body_dim + 4 + config.motion_dim, config.body_dim),
    )
    for stage, input_width, output_width in stages:
        projections = (
            ("embed_text", config.latent_dim, config.text_dim),
            ("input_linear", config.latent_dim, input_width),
            ("output_linear", output_width, config.latent_dim),
            ("linear_first_heading_angle", config.latent_dim, 2),
            ("embed_timestep.time_embed.0", config.latent_dim, config.latent_dim),
            ("embed_timestep.time_embed.2", config.latent_dim, config.latent_dim),
        )
        for name, out_width, in_width in projections:
            add(f"{stage}.{name}.weight", (out_width, in_width))
            add(f"{stage}.{name}.bias", (out_width,))
        for layer in range(config.layers):
            prefix = f"{stage}.seqTransEncoder.layers.{layer}"
            layer_projections = (
                ("self_attn.in_proj", 3 * config.latent_dim, config.latent_dim),
                ("self_attn.out_proj", config.latent_dim, config.latent_dim),
                ("linear1", config.feedforward_dim, config.latent_dim),
                ("linear2", config.latent_dim, config.feedforward_dim),
            )
            for name, out_width, in_width in layer_projections:
                if name == "self_attn.in_proj":
                    add(f"{prefix}.self_attn.in_proj_weight", (out_width, in_width))
                    add(f"{prefix}.self_attn.in_proj_bias", (out_width,))
                else:
                    add(f"{prefix}.{name}.weight", (out_width, in_width))
                    add(f"{prefix}.{name}.bias", (out_width,))
            for norm in ("norm1", "norm2"):
                arrays[f"{prefix}.{norm}.weight"] = wp.ones(
                    (config.latent_dim,), dtype=dtype, device=device
                )
                arrays[f"{prefix}.{norm}.bias"] = wp.zeros(
                    (config.latent_dim,), dtype=dtype, device=device
                )
    return arrays


def test_tiny_two_stage_denoiser_cpu_is_fixed_and_finite():
    config = KimodoConfig(33, 2, 30.0, 8, 16, 1, 2, text_dim=8, text_tokens=2)
    stats = KimodoStats(
        np.zeros(5),
        np.ones(5),
        np.zeros(4),
        np.ones(4),
        np.zeros(config.body_dim),
        np.ones(config.body_dim),
    )
    motion = wp.zeros((1, 3, config.motion_dim), dtype=wp.float32, device="cpu")
    mask = wp.zeros(motion.shape, dtype=wp.bool, device="cpu")
    valid = wp.array([[True, True, False]], dtype=wp.bool, device="cpu")
    text = wp.array(
        np.arange(config.text_tokens * config.text_dim, dtype=np.float32).reshape(
            1, config.text_tokens, config.text_dim
        )
        / 100,
        device="cpu",
    )
    timesteps = wp.array([7], dtype=wp.int32, device="cpu")
    heading = wp.array([0.2], dtype=wp.float32, device="cpu")
    plan = KimodoDenoiserPlan(
        motion,
        mask,
        valid,
        text,
        timesteps,
        heading,
        _tiny_weights(config),
        config,
        stats,
    )
    plan.lengths.assign(np.array([2], dtype=np.int32))
    pointers = (plan.output.ptr, plan.root_input.ptr, plan.body_input.ptr)
    first = plan.execute().numpy().copy()
    second = plan.execute().numpy().copy()
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert pointers == (plan.output.ptr, plan.root_input.ptr, plan.body_input.ptr)


@pytest.mark.skipif(not wp.get_cuda_devices(), reason="CUDA is unavailable")
def test_tiny_generation_cuda_graph_replays_with_new_guidance():
    device = wp.get_cuda_devices()[0]
    config = KimodoConfig(
        33,
        2,
        30.0,
        8,
        16,
        1,
        2,
        text_dim=8,
        text_tokens=2,
        diffusion_steps=8,
    )
    stats = KimodoStats(
        np.zeros(5),
        np.ones(5),
        np.zeros(4),
        np.ones(4),
        np.zeros(config.body_dim),
        np.ones(config.body_dim),
    )
    plan = KimodoGenerationPlan(
        1,
        3,
        config,
        stats,
        _tiny_weights(config, device, wp.bfloat16),
        dtype=wp.bfloat16,
        device=device,
    )
    text = np.zeros((1, config.text_tokens, config.text_dim), dtype=np.float32)
    plan.stage(text, [3], seed=11)
    first = plan.denoise(3, text_weight=1.5, constraint_weight=2.5).numpy()
    assert plan._graph is not None and np.isfinite(first).all()
    pointers = (plan.motion.ptr, plan.clean.ptr, plan.guidance_weights.ptr)
    plan.stage(text, [3], seed=19)
    second = plan.denoise(3, text_weight=0.5, constraint_weight=0.75).numpy()
    assert np.isfinite(second).all() and not np.array_equal(first, second)
    assert pointers == (plan.motion.ptr, plan.clean.ptr, plan.guidance_weights.ptr)
