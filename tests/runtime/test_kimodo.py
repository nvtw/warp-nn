# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import math

import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.kimodo.constraints import KimodoConstraints
from warp_nn.runtime.kimodo.runner import (
    _motion_kernels,
    _sampling_kernels,
    KimodoConfig,
    KimodoDenoiserPlan,
    KimodoDiffusionPlan,
    KimodoGenerationPlan,
    KimodoRunner,
    KimodoStats,
    cosine_ddim_schedule,
    decode_motion_features,
    load_kimodo_config,
    save_motion_npz,
)
from warp_nn.runtime.kimodo.viewer import write_motion_html
from warp_nn.runtime.kimodo.quadruped import _DOG_PAW_CHAINS, _stabilize_paws


def test_soma_config_and_cosine_schedule():
    config = KimodoConfig.soma_v1()
    assert config.motion_dim == 369
    assert config.body_dim == 364
    selected, alpha, previous = cosine_ddim_schedule(1000, 100)
    assert selected.shape == alpha.shape == previous.shape == (100,)
    assert selected[0] == 0 and selected[-1] == 999
    assert previous[0] == 1.0
    np.testing.assert_allclose(previous[1:], alpha[:-1])
    np.testing.assert_array_equal(
        cosine_ddim_schedule(1000, 15)[0],
        [0, 71, 143, 214, 285, 357, 428, 499, 571, 642, 714, 785, 856, 928, 999],
    )
    np.testing.assert_allclose(
        alpha[[0, 1, 49, 99]],
        [
            0.9999586939811707,
            0.9992788434028625,
            0.5016361474990845,
            2.4287349909002387e-9,
        ],
        rtol=1.0e-6,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        cosine_ddim_schedule(10, 11)


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
    features = np.zeros((1, 2, config.motion_dim), dtype=np.float32)
    decoded = decode_motion_features(features, stats, config.joints)
    assert decoded["posed_joints"].shape == (1, 2, config.joints, 3)
    assert decoded["posed_joints_from_positions"].shape == (1, 2, config.joints, 3)
    assert decoded["local_rot_mats"].shape == (1, 2, config.joints, 3, 3)
    assert decoded["global_rot_mats"].shape == (1, 2, config.joints, 3, 3)
    output = tmp_path / "motion.npz"
    save_motion_npz(output, decoded, fps=config.fps)
    with np.load(output) as saved:
        assert saved["fps"] == 30 and "root_positions" in saved
        assert saved["posed_joints"].shape == (2, config.joints, 3)


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


def test_sampling_broadcasts_one_prompt_embedding_to_batch():
    embedding = wp.array([[1, 2, 3, 4]], dtype=wp.float32, device="cpu")
    text = wp.zeros((3, 2, 4), dtype=wp.float32, device="cpu")
    stage_embedding = _sampling_kernels(wp.float32)[2]
    wp.launch(stage_embedding, dim=(3, 4), inputs=[embedding, text], device="cpu")
    expected = np.zeros((3, 2, 4), dtype=np.float32)
    expected[:, 0] = [1, 2, 3, 4]
    np.testing.assert_array_equal(text.numpy(), expected)


def test_kimodo_constraint_builder_composes_poses_joints_and_paths(tmp_path):
    joints, frames = 30, 8
    width = 12 * joints + 9
    source = np.arange(3 * width, dtype=np.float32).reshape(3, width) / 100
    constraints = KimodoConstraints.empty(frames, joints)
    constraints.start_pose(source, source_frame=1)
    constraints.end_pose(source)
    constraints.pose(3, source, source_frame=0, joints=[4, 7], rotations=False)
    constraints.pose([2, 5], source, source_frame=[0, 2])
    constraints.root_path([1, 4, 6], [[0, 0], [1, 2], [3, 5]], heading=[0, 0.5, 1])

    np.testing.assert_array_equal(constraints.observed[0, 0, :5], source[1, :5])
    np.testing.assert_array_equal(constraints.observed[0, -1, :5], source[-1, :5])
    assert constraints.mask[0, 0].sum() == 5 + joints * 3
    assert constraints.mask[0, 3].sum() == 5 + 2 * 3
    np.testing.assert_array_equal(constraints.observed[0, 2, :5], source[0, :5])
    np.testing.assert_array_equal(constraints.observed[0, 5, :5], source[2, :5])
    np.testing.assert_allclose(constraints.observed[0, [1, 4, 6], 0], [0, 1, 3])
    np.testing.assert_allclose(constraints.observed[0, [1, 4, 6], 2], [0, 2, 5])

    path = tmp_path / "motion.npz"
    np.savez(path, features=source)
    loaded = KimodoConstraints.empty(frames, joints)
    loaded.start_pose(path)
    np.testing.assert_array_equal(loaded.observed[0, 0, :5], source[-1, :5])

    effectors = KimodoConstraints.empty(frames, joints)
    effectors.end_effectors(2, source, ["LeftHand", "RightFoot"], source_frame=1)
    position_start = 5
    rotation_start = position_start + joints * 3
    assert effectors.mask[0, 2, position_start + 13 * 3 : position_start + 14 * 3].all()
    assert effectors.mask[0, 2, position_start + 15 * 3 : position_start + 16 * 3].all()
    assert effectors.mask[0, 2, rotation_start + 13 * 6 : rotation_start + 14 * 6].all()
    assert not effectors.mask[
        0, 2, rotation_start + 15 * 6 : rotation_start + 16 * 6
    ].any()

    cropped = constraints.crop(1, 7, prefix=2)
    assert cropped.frames == 8
    np.testing.assert_array_equal(cropped.mask[:, 2:], constraints.mask[:, 1:7])
    old_root = cropped.observed[..., [0, 2]].copy()
    cropped.translate_root([4, -2])
    selected = cropped.mask[..., [0, 2]]
    np.testing.assert_allclose(
        cropped.observed[..., [0, 2]][selected],
        (old_root + np.array([4, -2], np.float32))[selected],
    )

    spatial = KimodoConstraints.empty(frames, joints)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 2, 3, 3))
    positions = np.array(
        [[[2, 1, 4], [3, 2, 5]], [[6, 3, 8], [7, 4, 9]]], dtype=np.float32
    )
    spatial.spatial_targets(
        [2, 5],
        [4, 7],
        global_positions=positions,
        global_rotations=rotations,
        root_xz=[[1, 3], [5, 7]],
        root_y=[0.9, 1.0],
        heading=[0, np.pi / 2],
    )
    position_start = 5
    np.testing.assert_array_equal(
        spatial.observed[0, 2, position_start + 4 * 3 : position_start + 5 * 3],
        [1, 1, 1],
    )
    np.testing.assert_allclose(spatial.observed[0, 5, 3:5], [0, 1], atol=1e-6)

    archive = tmp_path / "constraints.npz"
    spatial.save(archive)
    restored = KimodoConstraints.load(archive)
    np.testing.assert_array_equal(restored.observed, spatial.observed)
    np.testing.assert_array_equal(restored.mask, spatial.mask)


def test_generation_final_projection_preserves_authored_values():
    config = KimodoConfig(21, 1, 30.0, 8, 16, 1, 2, diffusion_steps=10)
    plan = KimodoGenerationPlan.__new__(KimodoGenerationPlan)
    plan.batch, plan.branches, plan.config = 1, 1, config
    plan.device = wp.get_device("cpu")
    shape = (1, 2, config.motion_dim)
    plan.motion = wp.array(np.full(shape, 0.5, np.float32), device="cpu")
    plan.clean = wp.array(np.full(shape, 0.25, np.float32), device="cpu")
    observed = np.zeros(shape, np.float32)
    observed[0, 0, 3] = -1.75
    mask = np.zeros(shape, bool)
    mask[0, 0, 3] = True
    plan.observed = wp.array(observed, device="cpu")
    plan.mask = wp.array(mask, dtype=wp.bool, device="cpu")
    plan.guidance_weights = wp.empty((2,), dtype=wp.float32, device="cpu")
    plan.timesteps = wp.empty((1,), dtype=wp.int32, device="cpu")
    plan._enforce = _motion_kernels(wp.float32)[1]
    plan._ddim = _motion_kernels(wp.float32)[4]
    plan._graph = None
    plan._capture_ready = False
    plan._execute_denoiser = lambda: None

    result = plan.denoise(2).numpy()
    assert result[0, 0, 3] == observed[0, 0, 3]


def test_decode_projects_full_body_keyframe_but_not_sparse_joint():
    config = KimodoConfig.soma_v1()
    stats = KimodoStats(
        np.zeros(5, np.float32),
        np.ones(5, np.float32),
        np.zeros(4, np.float32),
        np.ones(4, np.float32),
        np.zeros(config.body_dim, np.float32),
        np.ones(config.body_dim, np.float32),
    )
    normalized = np.zeros((1, 3, config.motion_dim), dtype=np.float32)
    rotation_start = 5 + config.joints * 3
    normalized[..., rotation_start::6] = 1
    normalized[..., rotation_start + 4 :: 6] = 1
    source = decode_motion_features(normalized, stats, config.joints)
    source["posed_joints"][0, 1, :, 0] += 0.25
    constraints = KimodoConstraints.empty(3, config.joints)
    constraints.pose(0, source, source_frame=1)
    constraints.pose(2, source, source_frame=1, joints=[13])
    result = decode_motion_features(
        normalized, stats, config.joints, constraints=constraints
    )
    expected = (
        constraints.observed[0, 0, 5:rotation_start].reshape(config.joints, 3).copy()
    )
    expected[:, 0] += constraints.observed[0, 0, 0]
    expected[:, 2] += constraints.observed[0, 0, 2]
    np.testing.assert_array_equal(result["posed_joints"][0, 0], expected)
    assert not np.array_equal(result["posed_joints"][0, 2, 13], expected[13])

    # A decoded source must constrain the skeleton users actually see, not the
    # model's separate auxiliary position stream.
    visible = source["posed_joints"].copy()
    visible[0, 1, :, 0] += np.linspace(0, 0.1, config.joints)
    source["posed_joints"] = visible
    exact = KimodoConstraints.empty(3, config.joints)
    exact.start_pose(source, source_frame=1)
    projected = decode_motion_features(
        normalized, stats, config.joints, constraints=exact
    )
    np.testing.assert_allclose(
        projected["posed_joints"][0, 0], visible[0, 1], atol=1e-7
    )
    batched = decode_motion_features(
        np.repeat(normalized, 2, axis=0), stats, config.joints, constraints=exact
    )
    np.testing.assert_allclose(
        batched["posed_joints"][:, 0],
        np.repeat(visible[:, 1], 2, axis=0),
        atol=1e-7,
    )

    sparse = KimodoConstraints.empty(3, config.joints)
    hand_target = source["posed_joints"][0, 0, 13] + [-0.05, 0.05, 0]
    sparse.spatial_targets(
        2,
        13,
        global_positions=np.asarray(hand_target).reshape(1, 1, 3),
        global_rotations=np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3),
        root_xz=[0, 0],
    )
    corrected = decode_motion_features(
        normalized, stats, config.joints, constraints=sparse
    )
    np.testing.assert_allclose(
        corrected["posed_joints"][0, 2, 13], hand_target, atol=2e-5
    )
    chain = [11, 12, 13]
    before_lengths = np.linalg.norm(
        np.diff(source["posed_joints"][0, 0, chain], axis=0), axis=-1
    )
    after_lengths = np.linalg.norm(
        np.diff(corrected["posed_joints"][0, 2, chain], axis=0), axis=-1
    )
    np.testing.assert_allclose(after_lengths, before_lengths, atol=2e-6)


def test_prompt_sequence_uses_overlap_constraints_and_global_timeline():
    config = KimodoConfig(21, 1, 30.0, 8, 16, 1, 2)
    zeros = np.zeros
    stats = KimodoStats(
        zeros(5, np.float32),
        np.ones(5, np.float32),
        zeros(4, np.float32),
        np.ones(4, np.float32),
        zeros(config.body_dim, np.float32),
        np.ones(config.body_dim, np.float32),
    )
    runner = KimodoRunner.__new__(KimodoRunner)
    runner.config, runner.stats = config, stats
    calls = []

    def generate(prompt, count, **kwargs):
        calls.append((prompt, count, kwargs.get("constraints")))
        raw = np.zeros((1, count, config.motion_dim), dtype=np.float32)
        raw[0, :, 0] = np.arange(count) + 10 * (len(calls) - 1)
        raw[0, :, 3] = 1
        scale = np.sqrt(stats.std**2 + stats.epsilon)
        return (raw - stats.mean) / scale

    runner.generate = generate
    authored = KimodoConstraints.empty(13, joints=1)
    authored.root_path(8, [[7, 9]])
    result = runner.generate_sequence(
        ["walk", "wave"], [6, 7], transition_frames=2, constraints=authored
    )

    assert result.shape == (1, 13, config.motion_dim)
    assert [(prompt, count) for prompt, count, _ in calls] == [("walk", 6), ("wave", 9)]
    second = calls[1][2]
    assert second.mask[0, :2].sum() == 2 * (5 + config.joints * 3)
    assert second.mask[0, 4, 0] and second.mask[0, 4, 2]


def test_quadruped_paw_stabilization_preserves_limb_lengths():
    positions = np.zeros((80, 19, 3), dtype=np.float32)
    for chain_index, chain in enumerate(_DOG_PAW_CHAINS):
        for frame in range(len(positions)):
            base = np.array([chain_index, 0.7, frame * 0.01], dtype=np.float32)
            offsets = np.zeros((len(chain), 3), dtype=np.float32)
            offsets[:, 1] = -0.2 * np.arange(len(chain))
            positions[frame, list(chain)] = base + offsets
            positions[frame, chain[-1], 1] += 0.005 * (frame % 3)
    first_chain = list(_DOG_PAW_CHAINS[0])
    before = np.linalg.norm(np.diff(positions[:, first_chain], axis=1), axis=-1)
    corrected, contacts, paws = _stabilize_paws(positions)
    after = np.linalg.norm(np.diff(corrected[:, first_chain], axis=1), axis=-1)
    np.testing.assert_allclose(after, before, atol=2e-6)
    assert contacts.shape == (80, 4)
    np.testing.assert_array_equal(paws, [8, 12, 15, 18])
    original_slide = np.linalg.norm(np.diff(positions[:, paws[0]], axis=0), axis=-1)
    corrected_slide = np.linalg.norm(np.diff(corrected[:, paws[0]], axis=0), axis=-1)
    assert corrected_slide.min() < original_slide.min() * 0.1


def test_motion_html_is_single_file_with_exact_positions(tmp_path):
    frames = 3
    positions = np.arange(frames * 30 * 3, dtype=np.float32).reshape(frames, 30, 3)
    contacts = np.zeros((frames, 4), dtype=bool)
    contacts[1, 2] = True
    path = write_motion_html(
        tmp_path / "motion.html",
        {"posed_joints": positions, "foot_contacts": contacts},
        fps=30,
        prompt="A person waves </script>",
        seed=7,
        generation_seconds=1.25,
    )
    html = path.read_text(encoding="utf-8")
    assert "three@0.185.0" in html
    assert "joints.frustumCulled=bones.frustumCulled=false" in html
    assert "if(!ready){resetCamera();pose();ready=true" in html
    assert "loading.remove();requestAnimationFrame(()=>{resetCamera();pose()})" in html
    assert "playing=T>2" in html
    assert "A person waves" not in html
    encoded = html.split('const PAYLOAD="', 1)[1].split('";', 1)[0]
    payload = json.loads(base64.b64decode(encoded))
    recovered = np.frombuffer(
        base64.b64decode(payload["positions"]), dtype="<f4"
    ).reshape(frames, 30, 3)
    np.testing.assert_array_equal(recovered, positions)
    assert payload["meta"]["prompt"] == "A person waves </script>"
    assert payload["meta"]["ground_y"] == float(positions[..., 1].min())
    assert base64.b64decode(payload["contacts"])[6] == 1


def test_motion_html_accepts_generic_skeleton_without_contacts(tmp_path):
    positions = np.zeros((4, 3, 3), dtype=np.float32)
    path = write_motion_html(
        tmp_path / "quadruped.html",
        {
            "posed_joints": positions,
            "foot_contacts": np.zeros((4, 0), dtype=bool),
            "contact_joints": (),
            "joint_names": ("Root", "Spine", "Head"),
            "parents": (-1, 0, 1),
        },
        fps=30,
        prompt="A dog walks",
        seed=3,
        generation_seconds=0.5,
        label="Kimodo · Quadruped",
    )
    html = path.read_text(encoding="utf-8")
    encoded = html.split('const PAYLOAD="', 1)[1].split('";', 1)[0]
    payload = json.loads(base64.b64decode(encoded))
    assert payload["meta"]["parents"] == [-1, 0, 1]
    assert payload["meta"]["joint_names"] == ["Root", "Spine", "Head"]
    assert payload["meta"]["contact_joints"] == []
    assert payload["meta"]["label"] == "Kimodo · Quadruped"


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


def test_generation_plan_normalizes_observed_motion_cpu():
    config = KimodoConfig(33, 2, 30.0, 8, 16, 1, 2, text_dim=8, text_tokens=2)
    stats = KimodoStats(
        np.arange(5, dtype=np.float32),
        np.full(5, 2.0, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.arange(config.body_dim, dtype=np.float32),
        np.full(config.body_dim, 3.0, dtype=np.float32),
    )
    plan = KimodoGenerationPlan(
        1,
        3,
        config,
        stats,
        _tiny_weights(config),
        dtype=wp.float32,
        device="cpu",
    )
    observed = np.arange(3 * config.motion_dim, dtype=np.float32).reshape(
        plan.observed.shape
    )
    plan.stage(
        np.zeros((1, config.text_tokens, config.text_dim), dtype=np.float32),
        [3],
        observed=observed,
    )
    expected = (observed - stats.mean) / np.sqrt(stats.std**2 + stats.epsilon)
    np.testing.assert_allclose(plan.observed.numpy(), expected, rtol=1.0e-6)


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
