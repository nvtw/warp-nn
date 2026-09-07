# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import base64
import json

import numpy as np
import pytest

from warp_nn.runtime.kimodo.motion import (
    blend_motions,
    make_seamless_loop,
    retarget_soma30_motion,
)
from warp_nn.runtime.kimodo.constraints import SOMA30_PARENTS
from warp_nn.runtime.kimodo.runner import _SOMA30_NEUTRAL
from warp_nn.runtime.kimodo.viewer import write_motion_html
from warp_nn.runtime.skinning import (
    RiggedMesh,
    deform_rigged_mesh,
    load_rigged_mesh,
    normalize_skin_weights,
    save_rigged_mesh,
    skinning_transforms,
)


def _rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float32)


def _motion(frames, *, x_offset=0.0, yaw=0.0):
    positions = np.zeros((frames, 30, 3), dtype=np.float32)
    positions[..., 1] = np.arange(30, dtype=np.float32)[None] * 0.03
    positions[..., 0] += np.linspace(x_offset, x_offset + 1.0, frames)[:, None]
    rotations = np.broadcast_to(_rotation_y(yaw), (frames, 30, 3, 3)).copy()
    contacts = np.zeros((frames, 4), dtype=bool)
    contacts[::2, :2] = True
    return {
        "posed_joints": positions,
        "global_rot_mats": rotations,
        "foot_contacts": contacts,
    }


def test_skin_weight_normalization_is_sparse_and_deterministic():
    weights = np.asarray(((1, 1, 1, 1, 1), (0, -2, np.nan, 0, 0)), dtype=np.float32)
    first = normalize_skin_weights(weights)
    second = normalize_skin_weights(weights)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0)
    np.testing.assert_array_equal(first[0], (0.25, 0.25, 0.25, 0.25, 0))
    np.testing.assert_array_equal(first[1], (1, 0, 0, 0, 0))


def test_linear_blend_skinning_geometry_and_bind_pose():
    mesh = RiggedMesh(
        vertices=np.asarray(((0, 0, 0), (1, 0, 0), (2, 0, 0)), dtype=np.float32),
        faces=np.asarray(((0, 1, 2),), dtype=np.int32),
        rest_joints=np.asarray(((0, 0, 0), (1, 0, 0)), dtype=np.float32),
        parents=np.asarray((-1, 0), dtype=np.int32),
        weights=np.asarray(((1, 0), (0.5, 0.5), (0, 1)), dtype=np.float32),
    )
    identity = np.broadcast_to(np.eye(3), (2, 3, 3)).astype(np.float32)
    np.testing.assert_allclose(
        deform_rigged_mesh(mesh, mesh.rest_joints, identity), mesh.vertices
    )
    posed = mesh.rest_joints.copy()
    posed[1, 1] += 1
    deformed = deform_rigged_mesh(mesh, posed, identity)
    np.testing.assert_allclose(deformed, ((0, 0, 0), (1, 0.5, 0), (2, 1, 0)))
    transforms = skinning_transforms(
        mesh.rest_joints,
        np.stack((mesh.rest_joints, posed)),
        np.stack((identity, identity)),
    )
    assert transforms.shape == (2, 2, 3, 4)


def test_portable_rig_round_trip_and_top4_padding(tmp_path):
    mesh = RiggedMesh(
        vertices=np.asarray(((0, 0, 0), (1, 0, 0)), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
        rest_joints=np.asarray(((0, 0, 0), (1, 0, 0)), dtype=np.float32),
        parents=np.asarray((-1, 0), dtype=np.int32),
        weights=np.asarray(((1, 0), (0.25, 0.75)), dtype=np.float32),
        joint_names=("root", "tip"),
    )
    path = save_rigged_mesh(tmp_path / "rig.npz", mesh)
    recovered = load_rigged_mesh(path)
    np.testing.assert_array_equal(recovered.vertices, mesh.vertices)
    np.testing.assert_array_equal(recovered.weights, mesh.weights)
    assert recovered.joint_names == mesh.joint_names
    indices, weights = recovered.top4
    assert indices.shape == weights.shape == (2, 4)
    np.testing.assert_array_equal(indices[:, 2:], 0)
    np.testing.assert_array_equal(weights[:, 2:], 0)


def test_rigged_mesh_rejects_invalid_hierarchy():
    with pytest.raises(ValueError, match="precede"):
        RiggedMesh(
            np.zeros((1, 3)),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((2, 3)),
            np.asarray((-1, 1)),
            np.ones((1, 2)),
        )


def test_motion_blend_is_aligned_continuous_and_deterministic():
    first = _motion(20, yaw=0.4)
    second = _motion(24, x_offset=9.0, yaw=-0.7)
    blended = blend_motions(first, second, transition_frames=8)
    repeated = blend_motions(first, second, transition_frames=8)
    assert blended["posed_joints"].shape == (36, 30, 3)
    np.testing.assert_array_equal(blended["posed_joints"], repeated["posed_joints"])
    step = np.linalg.norm(np.diff(blended["root_positions"], axis=0), axis=1)
    assert step.max() < 0.2
    determinants = np.linalg.det(blended["global_rot_mats"])
    np.testing.assert_allclose(determinants, 1.0, atol=1.0e-5)


def test_seamless_loop_has_exact_pose_and_smooth_root_correction():
    loop = make_seamless_loop(_motion(40, yaw=0.65), blend_frames=10)
    np.testing.assert_array_equal(loop["posed_joints"][-1], loop["posed_joints"][0])
    np.testing.assert_array_equal(
        loop["global_rot_mats"][-1], loop["global_rot_mats"][0]
    )
    np.testing.assert_array_equal(loop["foot_contacts"][-1], loop["foot_contacts"][0])
    seam_speed = np.linalg.norm(loop["root_positions"][-2] - loop["root_positions"][-1])
    assert seam_speed < 0.15
    np.testing.assert_allclose(np.linalg.det(loop["global_rot_mats"]), 1.0, atol=1.0e-5)


def test_soma_retarget_preserves_target_bone_lengths_and_scales_travel():
    source = _motion(12)
    target = _SOMA30_NEUTRAL * 2.5 + np.asarray((3.0, 1.0, -2.0))
    result = retarget_soma30_motion(source, target)
    for joint in range(1, 30):
        parent = SOMA30_PARENTS[joint]
        expected = np.linalg.norm(target[joint] - target[parent])
        actual = np.linalg.norm(
            result["posed_joints"][:, joint] - result["posed_joints"][:, parent],
            axis=1,
        )
        np.testing.assert_allclose(actual, expected, atol=2.0e-6)
    np.testing.assert_allclose(result["root_positions"][0], target[0])
    np.testing.assert_allclose(
        result["root_positions"][-1] - result["root_positions"][0],
        (source["posed_joints"][-1, 0] - source["posed_joints"][0, 0]) * 2.5,
        atol=1.0e-5,
    )


def test_soma_retarget_preserves_model_bone_directions_and_arm_sides():
    positions = np.broadcast_to(_SOMA30_NEUTRAL, (2, 30, 3)).copy()
    rotations = np.broadcast_to(np.eye(3), (2, 30, 3, 3)).astype(np.float32).copy()
    motion = {
        "posed_joints": positions,
        "global_rot_mats": rotations,
        "foot_contacts": np.zeros((2, 4), dtype=bool),
    }
    # A deliberately incompatible target bind pose: arms hang and bend forward.
    target = _SOMA30_NEUTRAL.copy()
    target[11:16, 1] -= np.linspace(0.05, 0.35, 5)
    target[11:16, 2] += np.linspace(0.02, 0.30, 5)
    target[17:22, 1] -= np.linspace(0.05, 0.35, 5)
    target[17:22, 2] += np.linspace(0.02, 0.30, 5)

    result = retarget_soma30_motion(motion, target, stance_width=1.0)
    posed = result["posed_joints"]
    assert np.all(posed[:, 13, 0] > posed[:, 0, 0])
    assert np.all(posed[:, 19, 0] < posed[:, 0, 0])
    for joint in (11, 12, 13, 17, 18, 19):
        parent = SOMA30_PARENTS[joint]
        expected = _SOMA30_NEUTRAL[joint] - _SOMA30_NEUTRAL[parent]
        actual = posed[0, joint] - posed[0, parent]
        expected /= np.linalg.norm(expected)
        actual /= np.linalg.norm(actual)
        np.testing.assert_allclose(actual, expected, atol=2.0e-6)
        rotated_bind = result["global_rot_mats"][0, parent] @ (
            target[joint] - target[parent]
        )
        rotated_bind /= np.linalg.norm(rotated_bind)
        np.testing.assert_allclose(rotated_bind, expected, atol=2.0e-6)
    for joint in (7, 8, 9):
        parent = SOMA30_PARENTS[joint]
        expected = result["global_rot_mats"][0, parent] @ (
            target[joint] - target[parent]
        )
        np.testing.assert_allclose(
            posed[0, joint] - posed[0, parent], expected, atol=2.0e-6
        )


def test_soma_retarget_posture_controls_are_monotonic_and_validated():
    positions = np.broadcast_to(_SOMA30_NEUTRAL, (2, 30, 3)).copy()
    rotations = np.broadcast_to(np.eye(3), (2, 30, 3, 3)).astype(np.float32)
    motion = {
        "posed_joints": positions,
        "global_rot_mats": rotations,
        "foot_contacts": np.zeros((2, 4), dtype=bool),
    }
    wide = retarget_soma30_motion(motion, _SOMA30_NEUTRAL, stance_width=1.0)
    narrow = retarget_soma30_motion(motion, _SOMA30_NEUTRAL, stance_width=0.7)
    wide_hips = abs(wide["posed_joints"][0, 22, 0] - wide["posed_joints"][0, 26, 0])
    narrow_hips = abs(
        narrow["posed_joints"][0, 22, 0] - narrow["posed_joints"][0, 26, 0]
    )
    assert narrow_hips < wide_hips

    forward = positions.copy()
    forward[:, 4:7, 2] += 0.1
    motion["posed_joints"] = forward
    natural = retarget_soma30_motion(motion, _SOMA30_NEUTRAL, head_forward=1.0)
    straighter = retarget_soma30_motion(motion, _SOMA30_NEUTRAL, head_forward=0.5)
    assert abs(straighter["posed_joints"][0, 6, 2]) < abs(
        natural["posed_joints"][0, 6, 2]
    )
    with pytest.raises(ValueError, match="stance_width"):
        retarget_soma30_motion(motion, _SOMA30_NEUTRAL, stance_width=-0.1)


def test_kimodo_viewer_embeds_optional_skinned_mesh(tmp_path):
    motion = _motion(3)
    weights = np.zeros((3, 30), dtype=np.float32)
    weights[:, 0] = 1
    mesh = RiggedMesh(
        vertices=np.asarray(((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0))),
        faces=np.asarray(((0, 1, 2),)),
        rest_joints=_SOMA30_NEUTRAL,
        parents=SOMA30_PARENTS,
        weights=weights,
    )
    path = write_motion_html(
        tmp_path / "mesh.html",
        motion,
        fps=30,
        prompt="mesh",
        seed=1,
        generation_seconds=0.1,
        mesh=mesh,
    )
    html = path.read_text(encoding="utf-8")
    assert "skinUniforms" in html
    assert "rigMatrices[${J}]" in html
    assert "slerpQuaternions" in html
    assert "body.customDepthMaterial=depthMaterial" in html
    assert "depthMaterial.onBeforeCompile=applySkinning" in html
    assert "markerScale=facial?.42:1" in html
    assert "joints.visible=bones.visible=false" in html
    assert "Show or hide the skeleton" in html
    encoded = html.split('const PAYLOAD="', 1)[1].split('";', 1)[0]
    payload = json.loads(base64.b64decode(encoded))
    assert payload["meta"]["ground_y"] == pytest.approx(0.9887085)
    assert len(base64.b64decode(payload["mesh"]["vertices"])) == 3 * 3 * 4
    assert len(base64.b64decode(payload["mesh"]["joints"])) == 3 * 4 * 2
