# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Motion editing utilities for decoded Kimodo clips."""

from __future__ import annotations

import numpy as np

from .constraints import SOMA30_PARENTS
from .runner import _SOMA30_NEUTRAL


def _single_motion(motion):
    result = {}
    for name in ("posed_joints", "global_rot_mats", "foot_contacts"):
        if name not in motion:
            raise ValueError(f"motion is missing {name}")
        value = np.asarray(motion[name])
        if name != "foot_contacts" and value.ndim > (
            4 if name == "global_rot_mats" else 3
        ):
            if value.shape[0] != 1:
                raise ValueError("motion utilities operate on one clip at a time")
            value = value[0]
        elif name == "foot_contacts" and value.ndim == 3:
            if value.shape[0] != 1:
                raise ValueError("motion utilities operate on one clip at a time")
            value = value[0]
        result[name] = value
    posed = np.asarray(result["posed_joints"], dtype=np.float32)
    rotations = np.asarray(result["global_rot_mats"], dtype=np.float32)
    contacts = np.asarray(result["foot_contacts"], dtype=bool)
    if posed.ndim != 3 or posed.shape[1:] != (30, 3):
        raise ValueError("posed_joints must have shape [frames, 30, 3]")
    if rotations.shape != (len(posed), 30, 3, 3):
        raise ValueError("global_rot_mats must match posed_joints")
    if contacts.shape != (len(posed), 4):
        raise ValueError("foot_contacts must have shape [frames, 4]")
    if not np.isfinite(posed).all() or not np.isfinite(rotations).all():
        raise ValueError("motion must be finite")
    return posed, rotations, contacts


def _smoothstep(value):
    return value * value * (3.0 - 2.0 * value)


def _matrix_to_quaternion(matrix):
    """Convert rotation matrices to canonical ``[w, x, y, z]`` quaternions."""
    matrix = np.asarray(matrix, dtype=np.float64)
    flat = matrix.reshape(-1, 3, 3)
    result = np.empty((len(flat), 4), dtype=np.float64)
    for index, value in enumerate(flat):
        candidates = np.array(
            (
                1.0 + value[0, 0] + value[1, 1] + value[2, 2],
                1.0 + value[0, 0] - value[1, 1] - value[2, 2],
                1.0 - value[0, 0] + value[1, 1] - value[2, 2],
                1.0 - value[0, 0] - value[1, 1] + value[2, 2],
            )
        )
        choice = int(np.argmax(candidates))
        scale = 0.5 / np.sqrt(max(candidates[choice], 1.0e-15))
        if choice == 0:
            result[index] = (
                0.25 / scale,
                (value[2, 1] - value[1, 2]) * scale,
                (value[0, 2] - value[2, 0]) * scale,
                (value[1, 0] - value[0, 1]) * scale,
            )
        elif choice == 1:
            result[index] = (
                (value[2, 1] - value[1, 2]) * scale,
                0.25 / scale,
                (value[0, 1] + value[1, 0]) * scale,
                (value[0, 2] + value[2, 0]) * scale,
            )
        elif choice == 2:
            result[index] = (
                (value[0, 2] - value[2, 0]) * scale,
                (value[0, 1] + value[1, 0]) * scale,
                0.25 / scale,
                (value[1, 2] + value[2, 1]) * scale,
            )
        else:
            result[index] = (
                (value[1, 0] - value[0, 1]) * scale,
                (value[0, 2] + value[2, 0]) * scale,
                (value[1, 2] + value[2, 1]) * scale,
                0.25 / scale,
            )
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    result[result[:, 0] < 0] *= -1.0
    return result.reshape(*matrix.shape[:-2], 4)


def _quaternion_to_matrix(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-15)
    w, x, y, z = np.moveaxis(q, -1, 0)
    result = np.empty((*q.shape[:-1], 3, 3), dtype=np.float64)
    result[..., 0, 0] = 1 - 2 * (y * y + z * z)
    result[..., 0, 1] = 2 * (x * y - z * w)
    result[..., 0, 2] = 2 * (x * z + y * w)
    result[..., 1, 0] = 2 * (x * y + z * w)
    result[..., 1, 1] = 1 - 2 * (x * x + z * z)
    result[..., 1, 2] = 2 * (y * z - x * w)
    result[..., 2, 0] = 2 * (x * z - y * w)
    result[..., 2, 1] = 2 * (y * z + x * w)
    result[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return result.astype(np.float32)


def _slerp(first, second, amount):
    first = _matrix_to_quaternion(first)
    second = _matrix_to_quaternion(second)
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0, -second, second)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    amount = np.asarray(amount, dtype=np.float64)
    while amount.ndim < dot.ndim:
        amount = amount[..., None]
    angle = np.arccos(dot)
    sine = np.sin(angle)
    linear = sine < 1.0e-7
    left = np.empty_like(angle)
    right = np.empty_like(angle)
    np.divide(
        np.sin((1.0 - amount) * angle),
        sine,
        out=left,
        where=~linear,
    )
    np.divide(np.sin(amount * angle), sine, out=right, where=~linear)
    left[linear] = np.broadcast_to(1.0 - amount, angle.shape)[linear]
    right[linear] = np.broadcast_to(amount, angle.shape)[linear]
    return _quaternion_to_matrix(left * first + right * second)


def _yaw(rotation):
    # Project the local forward (+Z) direction onto the ground plane.
    return np.arctan2(rotation[..., 0, 2], rotation[..., 2, 2])


def _yaw_matrix(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float32,
    )


def _pack(posed, rotations, contacts):
    parents = SOMA30_PARENTS
    parent_rotations = rotations[:, parents].copy()
    parent_rotations[:, 0] = np.eye(3, dtype=np.float32)
    local = np.swapaxes(parent_rotations, -1, -2) @ rotations
    return {
        "posed_joints": np.ascontiguousarray(posed, dtype=np.float32),
        "root_positions": np.ascontiguousarray(posed[:, 0], dtype=np.float32),
        "global_rot_mats": np.ascontiguousarray(rotations, dtype=np.float32),
        "local_rot_mats": np.ascontiguousarray(local, dtype=np.float32),
        "foot_contacts": np.ascontiguousarray(contacts, dtype=bool),
    }


def blend_motions(first, second, *, transition_frames: int = 15):
    """Join two clips with root alignment and a smooth pose transition."""
    a_pos, a_rot, a_contact = _single_motion(first)
    b_pos, b_rot, b_contact = _single_motion(second)
    count = int(transition_frames)
    if count < 2 or count > min(len(a_pos), len(b_pos)):
        raise ValueError("transition_frames must be 2..min(clip lengths)")

    angle = float(_yaw(a_rot[-1, 0]) - _yaw(b_rot[0, 0]))
    alignment = _yaw_matrix(angle)
    anchor = b_pos[0, 0].copy()
    b_pos = np.einsum("ij,fkj->fki", alignment, b_pos - anchor)
    b_pos += a_pos[-1, 0]
    b_rot = np.einsum("ij,fkjl->fkil", alignment, b_rot)

    u = _smoothstep(np.linspace(0.0, 1.0, count, dtype=np.float32))
    mix_pos = (
        a_pos[-count:] * (1.0 - u[:, None, None]) + b_pos[:count] * u[:, None, None]
    )
    mix_rot = _slerp(a_rot[-count:], b_rot[:count], u[:, None])
    mix_contact = np.where(u[:, None] < 0.5, a_contact[-count:], b_contact[:count])
    return _pack(
        np.concatenate((a_pos[:-count], mix_pos, b_pos[count:])),
        np.concatenate((a_rot[:-count], mix_rot, b_rot[count:])),
        np.concatenate((a_contact[:-count], mix_contact, b_contact[count:])),
    )


def make_seamless_loop(motion, *, blend_frames: int = 15):
    """Close a clip into an in-place loop with an exactly matching end pose.

    Root translation and heading error are distributed smoothly over the full
    clip, avoiding one visible correction at the seam.  The final window then
    eases the articulated pose into frame zero with quaternion interpolation.
    """
    posed, rotations, contacts = _single_motion(motion)
    count = int(blend_frames)
    if count < 2 or count >= len(posed):
        raise ValueError("blend_frames must be 2..frames-1")
    posed = posed.copy()
    rotations = rotations.copy()
    contacts = contacts.copy()

    progress = _smoothstep(np.linspace(0.0, 1.0, len(posed), dtype=np.float32))
    root_delta = posed[-1, 0] - posed[0, 0]
    heading_delta = float(_yaw(rotations[-1, 0]) - _yaw(rotations[0, 0]))
    for frame, amount in enumerate(progress):
        correction = _yaw_matrix(-heading_delta * float(amount))
        root = posed[frame, 0].copy()
        posed[frame] = np.einsum("ij,kj->ki", correction, posed[frame] - root)
        root -= root_delta * amount
        posed[frame] += root
        rotations[frame] = np.einsum("ij,kjl->kil", correction, rotations[frame])

    start = len(posed) - count
    u = _smoothstep(np.linspace(0.0, 1.0, count, dtype=np.float32))
    posed[start:] = (
        posed[start:] * (1.0 - u[:, None, None]) + posed[0] * u[:, None, None]
    )
    rotations[start:] = _slerp(rotations[start:], rotations[0], u[:, None])
    contacts[start:] = np.where(u[:, None] < 0.5, contacts[start:], contacts[0])
    posed[-1] = posed[0]
    rotations[-1] = rotations[0]
    contacts[-1] = contacts[0]
    return _pack(posed, rotations, contacts)


def _rotation_between(source, target):
    """Return the minimum rotation mapping one nonzero vector onto another."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine < 1.0e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float32)
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(source)))] = 1.0
        axis = np.cross(source, axis)
        axis /= np.linalg.norm(axis)
        return (2.0 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)
    skew = np.asarray(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        )
    )
    return (np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))).astype(
        np.float32
    )


def _retarget_bind_alignments(source_rest, target_rest):
    """Map target bind-bone axes into the model's bind-bone axes."""
    children = [[] for _ in range(len(SOMA30_PARENTS))]
    for child, parent in enumerate(SOMA30_PARENTS[1:], 1):
        children[int(parent)].append(child)
    result = np.empty((30, 3, 3), dtype=np.float32)
    for joint in range(30):
        # The jaw is not a stable head-orientation axis, and the thumb is not
        # a stable hand axis across character meshes. Use the incoming neck
        # bone for the head and the middle-finger chain for each hand.
        if joint == 6:
            endpoint = int(SOMA30_PARENTS[joint])
            source = source_rest[joint] - source_rest[endpoint]
            target = target_rest[joint] - target_rest[endpoint]
        elif joint in (13, 19):
            endpoint = joint + 2
            source = source_rest[endpoint] - source_rest[joint]
            target = target_rest[endpoint] - target_rest[joint]
        elif children[joint]:
            endpoint = children[joint][0]
            source = source_rest[endpoint] - source_rest[joint]
            target = target_rest[endpoint] - target_rest[joint]
        else:
            parent = int(SOMA30_PARENTS[joint])
            source = source_rest[joint] - source_rest[parent]
            target = target_rest[joint] - target_rest[parent]
        result[joint] = _rotation_between(target, source)
    return result


def retarget_soma30_motion(
    motion,
    rest_joints,
    *,
    scale_root_motion: bool = True,
    stance_width: float = 1.0,
    head_forward: float = 0.5,
):
    """Retarget Kimodo motion onto a SOMA-30 bind skeleton.

    SkinTokens can predict weights for a mesh carrying this skeleton.  Bone
    lengths remain those of the target asset while Kimodo supplies directions
    and rotations. Bind-axis correction keeps an A-pose mesh compatible with
    Kimodo's T-pose skeleton. ``stance_width`` controls the lateral root-to-hip
    offset, and ``head_forward`` controls forward displacement in the neck chain.
    """
    source_posed, source_global, contacts = _single_motion(motion)
    rest = np.asarray(rest_joints, dtype=np.float32)
    if rest.shape != (30, 3) or not np.isfinite(rest).all():
        raise ValueError("rest_joints must be finite SOMA-30 positions")
    if not 0.0 <= stance_width <= 2.0:
        raise ValueError("stance_width must be between 0 and 2")
    if not 0.0 <= head_forward <= 2.0:
        raise ValueError("head_forward must be between 0 and 2")
    parents = SOMA30_PARENTS

    source_lengths = np.linalg.norm(
        _SOMA30_NEUTRAL[1:] - _SOMA30_NEUTRAL[parents[1:]], axis=1
    )
    target_lengths = np.linalg.norm(rest[1:] - rest[parents[1:]], axis=1)
    valid = source_lengths > 1.0e-5
    scale = float(np.median(target_lengths[valid] / source_lengths[valid]))
    if not scale_root_motion:
        scale = 1.0

    frames = len(source_posed)
    posed = np.empty((frames, 30, 3), dtype=np.float32)
    alignments = _retarget_bind_alignments(_SOMA30_NEUTRAL, rest)
    rotations = source_global @ alignments[None]
    travel = (source_posed[:, 0] - source_posed[0, 0]) * scale
    posed[:, 0] = rest[0] + travel
    for joint in range(1, 30):
        parent = int(parents[joint])
        if joint in (7, 8, 9):
            posed[:, joint] = posed[:, parent] + np.einsum(
                "fij,j->fi",
                rotations[:, parent],
                rest[joint] - rest[parent],
            )
            continue
        direction = source_posed[:, joint] - source_posed[:, parent]
        if parent == 0 and joint in (22, 26):
            direction[:, 0] *= stance_width
        if joint in (4, 5, 6):
            direction[:, 2] *= head_forward
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        fallback = _SOMA30_NEUTRAL[joint] - _SOMA30_NEUTRAL[parent]
        direction = np.divide(
            direction,
            norm,
            out=np.broadcast_to(
                fallback / np.linalg.norm(fallback), direction.shape
            ).copy(),
            where=norm > 1.0e-7,
        )
        length = np.linalg.norm(rest[joint] - rest[parent])
        posed[:, joint] = posed[:, parent] + direction * length
    return _pack(posed, rotations, contacts)


__all__ = ["blend_motions", "make_seamless_loop", "retarget_soma30_motion"]
