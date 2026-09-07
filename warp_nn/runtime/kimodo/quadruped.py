# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact PAN human-to-dog retargeting for Kimodo motion.

Only the released human encoder, dog skeleton encoder, and dog decoder are
loaded.  Training-only discriminators, reverse reconstruction, and cycle paths
are deliberately excluded.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

from ..formats.pytorch import load_pytorch_zip
from ..operators import (
    BiasedLinearPlan,
    Conv1dPlan,
    ElementwiseActivationPlan,
    LinearUpsample1dPlan,
    ReflectPad1dPlan,
)
from .runner import _rotation_between_vectors


_HUMAN_JOINTS = (
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
)
_SOMA_TO_HUMAN = np.asarray(
    (0, 22, 23, 24, 25, 26, 27, 28, 29, 1, 2, 3, 4, 6, 10, 11, 12, 13, 16, 17, 18, 19),
    dtype=np.int32,
)
_HUMAN_PARTS = ((3, 4, 7, 8, 1, 2, 5, 6), (9, 10, 11), (12, 13))
_DOG_PAW_CHAINS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15), (16, 17, 18))


def _parse_bvh_skeleton(path: str | Path):
    """Read the named, non-end-site hierarchy without parsing motion frames."""
    names: list[str] = []
    parents: list[int] = []
    offsets: list[tuple[float, float, float]] = []
    stack: list[int | None] = []
    pending: int | None = None
    current: int | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "MOTION":
            break
        fields = line.split()
        if fields and fields[0] in ("ROOT", "JOINT"):
            parent = next((value for value in reversed(stack) if value is not None), -1)
            names.append(" ".join(fields[1:]))
            parents.append(int(parent))
            offsets.append((0.0, 0.0, 0.0))
            pending = len(names) - 1
        elif line == "End Site":
            pending = None
        elif line == "{":
            stack.append(pending)
            current = pending
            pending = None
        elif line == "}":
            if stack:
                stack.pop()
            current = next(
                (value for value in reversed(stack) if value is not None), None
            )
        elif fields and fields[0] == "OFFSET" and current is not None:
            offsets[current] = tuple(float(value) for value in fields[1:4])
    return (
        tuple(names),
        np.asarray(parents, dtype=np.int32),
        np.asarray(offsets, dtype=np.float32),
    )


def _matrix_to_quaternion(matrix):
    """Vectorized rotation matrix to scalar-first quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    flat = matrix.reshape((-1, 3, 3))
    result = np.empty((flat.shape[0], 4), dtype=np.float64)
    for index, value in enumerate(flat):
        trace = np.trace(value)
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            result[index] = (
                0.25 * scale,
                (value[2, 1] - value[1, 2]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
                (value[1, 0] - value[0, 1]) / scale,
            )
        else:
            axis = int(np.argmax(np.diag(value)))
            other = (axis + 1) % 3
            last = (axis + 2) % 3
            scale = (
                math.sqrt(
                    max(
                        1.0
                        + value[axis, axis]
                        - value[other, other]
                        - value[last, last],
                        0.0,
                    )
                )
                * 2.0
            )
            q = np.empty(4)
            q[axis + 1] = 0.25 * scale
            q[0] = (value[last, other] - value[other, last]) / max(scale, 1.0e-12)
            q[other + 1] = (value[other, axis] + value[axis, other]) / max(
                scale, 1.0e-12
            )
            q[last + 1] = (value[last, axis] + value[axis, last]) / max(scale, 1.0e-12)
            result[index] = q
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1.0e-12)
    result[1:] *= np.where(
        np.sum(result[1:] * result[:-1], axis=-1, keepdims=True) < 0.0, -1.0, 1.0
    )
    result *= np.where(result[..., :1] < 0.0, -1.0, 1.0)
    return result.reshape((*matrix.shape[:-2], 4)).astype(np.float32)


def _quaternion_to_matrix(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float32)
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-12
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((*quaternion.shape[:-1], 3, 3))


def _smooth_contact_targets(positions):
    """Match PAN's block-height contacts and six-frame paw-lock easing."""
    paws = np.asarray([chain[-1] for chain in _DOG_PAW_CHAINS])
    targets = positions[:, paws].copy()
    contacts = np.zeros(targets.shape[:2], dtype=bool)
    for begin in range(0, len(positions), 40):
        end = min(begin + 40, len(positions))
        ground = targets[begin:end, :, 1].min(axis=0) + np.float32(0.015)
        contacts[begin:end] = targets[begin:end, :, 1] < ground

    changed = contacts.copy()
    radius = 6
    for paw in range(len(paws)):
        frame = 0
        while frame < len(positions):
            if not contacts[frame, paw]:
                frame += 1
                continue
            end = frame
            while end + 1 < len(positions) and contacts[end + 1, paw]:
                end += 1
            targets[frame : end + 1, paw] = targets[frame : end + 1, paw].mean(axis=0)
            frame = end + 1

        original = targets[:, paw].copy()
        for frame in range(len(positions)):
            if contacts[frame, paw]:
                continue
            left = next(
                (
                    frame - step
                    for step in range(1, radius + 1)
                    if frame - step >= 0 and contacts[frame - step, paw]
                ),
                None,
            )
            right = next(
                (
                    frame + step
                    for step in range(1, radius + 1)
                    if frame + step < len(positions) and contacts[frame + step, paw]
                ),
                None,
            )
            if left is None and right is None:
                continue

            def ease(distance):
                value = (distance + 1.0) / (radius + 1.0)
                return 2.0 * value**3 - 3.0 * value**2 + 1.0

            if left is not None:
                weight = ease(frame - left)
                left_value = (
                    original[frame] * (1.0 - weight) + targets[left, paw] * weight
                )
            if right is not None:
                weight = ease(right - frame)
                right_value = (
                    original[frame] * (1.0 - weight) + targets[right, paw] * weight
                )
            if left is not None and right is not None:
                weight = ease(frame - left) / max(
                    ease(frame - left) + ease(right - frame), 1.0e-8
                )
                targets[frame, paw] = right_value * (1.0 - weight) + left_value * weight
            else:
                targets[frame, paw] = left_value if left is not None else right_value
            changed[frame, paw] = True
    return paws, contacts, targets, changed


def _stabilize_paws(positions):
    """Lock contacting paws with short-chain FABRIK while preserving bone lengths."""
    corrected = positions.copy()
    paws, contacts, targets, changed = _smooth_contact_targets(positions)
    for paw, chain in enumerate(_DOG_PAW_CHAINS):
        for frame in np.flatnonzero(changed[:, paw]):
            points = corrected[frame, chain].copy()
            base = points[0].copy()
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
            target = targets[frame, paw]
            if np.linalg.norm(target - base) >= lengths.sum():
                direction = (target - base) / max(np.linalg.norm(target - base), 1.0e-8)
                for index, length in enumerate(lengths):
                    points[index + 1] = points[index] + direction * length
            else:
                for _ in range(8):
                    points[-1] = target
                    for index in range(len(points) - 2, -1, -1):
                        direction = points[index] - points[index + 1]
                        direction /= max(np.linalg.norm(direction), 1.0e-8)
                        points[index] = points[index + 1] + direction * lengths[index]
                    points[0] = base
                    for index, length in enumerate(lengths):
                        direction = points[index + 1] - points[index]
                        direction /= max(np.linalg.norm(direction), 1.0e-8)
                        points[index + 1] = points[index] + direction * length
            corrected[frame, chain] = points
    return corrected, contacts, paws


def _rotations_for_corrected_limbs(original, corrected, global_rotations, parents):
    """Update limb rotations to agree with FABRIK-corrected bone directions."""
    rotations = global_rotations.copy()
    changed = np.linalg.norm(corrected - original, axis=-1) > 1.0e-7
    for chain in _DOG_PAW_CHAINS:
        for frame in np.flatnonzero(np.any(changed[:, chain], axis=1)):
            for parent, child in zip(chain[:-1], chain[1:]):
                delta = _rotation_between_vectors(
                    original[frame, child] - original[frame, parent],
                    corrected[frame, child] - corrected[frame, parent],
                )
                rotations[frame, parent] = delta @ rotations[frame, parent]
    local = rotations.copy()
    for joint in range(1, len(parents)):
        local[:, joint] = (
            np.swapaxes(rotations[:, parents[joint]], -1, -2) @ rotations[:, joint]
        )
    return rotations, local


def _human_features(motion, parents, mean, std):
    positions = np.asarray(motion["posed_joints"], dtype=np.float32)
    source_global = np.asarray(motion["global_rot_mats"], dtype=np.float32)
    if positions.ndim == 4:
        if positions.shape[0] != 1:
            raise ValueError("quadruped retargeting currently accepts one motion")
        positions = positions[0]
        source_global = source_global[0]
    if source_global.shape != (*positions.shape[:2], 3, 3):
        raise ValueError("global_rot_mats must match posed_joints")
    mapped = positions[:, _SOMA_TO_HUMAN] * np.float32(100.0)
    frames, joints = mapped.shape[:2]
    prior_quaternions = mean[: joints * 4].reshape(joints, 4).astype(np.float32)
    prior_quaternions /= np.maximum(
        np.linalg.norm(prior_quaternions, axis=-1, keepdims=True), 1.0e-12
    )
    prior_local = _quaternion_to_matrix(prior_quaternions)
    prior_global = np.empty_like(prior_local)
    prior_global[0] = prior_local[0]
    for joint in range(1, joints):
        prior_global[joint] = prior_global[parents[joint]] @ prior_local[joint]

    # SOMA's root frame has Y up, whereas PAN's LAFAN root frame has Y forward.
    # Derive heading from anatomy rather than assuming either local-axis convention.
    lateral = positions[:, 22] - positions[:, 26]
    vertical = positions[:, 3] - positions[:, 0]
    forward = np.cross(lateral, vertical)
    forward[:, 1] = 0.0
    norm = np.linalg.norm(forward, axis=-1, keepdims=True)
    fallback = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    forward = np.where(norm > 1.0e-6, forward / np.maximum(norm, 1.0e-6), fallback)
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    cosine, sine = np.cos(-yaw), np.sin(-yaw)
    inverse_heading = np.zeros((frames, 3, 3), dtype=np.float32)
    inverse_heading[:, 0, 0] = cosine
    inverse_heading[:, 0, 2] = sine
    inverse_heading[:, 1, 1] = 1.0
    inverse_heading[:, 2, 0] = -sine
    inverse_heading[:, 2, 2] = cosine
    canonical_motion = inverse_heading[:, None] @ source_global[:, _SOMA_TO_HUMAN]
    canonical_global = canonical_motion @ prior_global[None]
    canonical_local = canonical_global.copy()
    for joint in range(1, joints):
        canonical_local[:, joint] = (
            np.swapaxes(canonical_global[:, parents[joint]], -1, -2)
            @ canonical_global[:, joint]
        )
    quaternions = _matrix_to_quaternion(canonical_local)

    delta = np.zeros((frames, 3), dtype=np.float32)
    delta[:-1] = mapped[1:, 0] - mapped[:-1, 0]
    delta[-1] = delta[-2] if frames > 1 else 0.0
    local_delta = np.einsum("tij,tj->ti", inverse_heading, delta)
    angular = np.zeros((frames, 1), dtype=np.float32)
    if frames > 1:
        angular[:-1, 0] = np.arctan2(
            np.sin(yaw[1:] - yaw[:-1]), np.cos(yaw[1:] - yaw[:-1])
        )
        angular[-1] = angular[-2]
    features = np.concatenate(
        (quaternions.reshape(frames, -1), local_delta, angular), axis=-1
    )
    normalized = (features - mean.astype(np.float32)) / std.astype(np.float32)
    # Cross-rig frame changes leak motion into quaternion components that are
    # effectively constant in PAN's training set. Project those dimensions back
    # onto the learned manifold and bound genuine components to its observed range.
    normalized[..., std < 0.04] = 0.0
    np.clip(normalized, -5.0, 5.0, out=normalized)
    return normalized[None], yaw


def _attention_mask():
    joints, parts = 23, len(_HUMAN_PARTS)
    allowed = np.zeros((parts + joints, parts + joints), dtype=bool)
    for index, members in enumerate(_HUMAN_PARTS):
        allowed[index, parts + np.asarray(members)] = True
        for first in members:
            allowed[parts + first, parts + np.asarray(members)] = True
    allowed[:, parts] = True
    allowed[:, parts + joints - 1] = True
    np.fill_diagonal(allowed, True)
    return allowed


@wp.kernel(enable_backward=False, module="unique")
def _pan_pack_tokens(
    projected: wp.array4d(dtype=wp.float32),
    part_tokens: wp.array3d(dtype=wp.float32),
    position: wp.array3d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32),
):
    row, token, channel = wp.tid()
    batch = row / projected.shape[1]
    frame = row % projected.shape[1]
    if token < part_tokens.shape[0]:
        output[row, token, channel] = part_tokens[token, 0, channel]
    else:
        joint = token - part_tokens.shape[0]
        output[row, token, channel] = (
            projected[batch, frame, joint, channel] * wp.sqrt(wp.float32(32.0))
            + position[joint, 0, channel]
        )


@wp.kernel(enable_backward=False, module="unique")
def _pan_attention(
    qkv: wp.array3d(dtype=wp.float32),
    allowed: wp.array2d(dtype=wp.bool),
    output: wp.array3d(dtype=wp.float32),
):
    row, query, channel = wp.tid()
    maximum = wp.float32(-3.402823466e38)
    for key in range(qkv.shape[1]):
        if allowed[query, key]:
            score = wp.float32(0.0)
            for inner in range(32):
                score += qkv[row, query, inner] * qkv[row, key, 32 + inner]
            maximum = wp.max(maximum, score * wp.float32(0.1767766952966369))
    total = wp.float32(0.0)
    value = wp.float32(0.0)
    for key in range(qkv.shape[1]):
        if allowed[query, key]:
            score = wp.float32(0.0)
            for inner in range(32):
                score += qkv[row, query, inner] * qkv[row, key, 32 + inner]
            probability = wp.exp(score * wp.float32(0.1767766952966369) - maximum)
            total += probability
            value += probability * qkv[row, key, 64 + channel]
    output[row, query, channel] = value / total


@wp.kernel(enable_backward=False, module="unique")
def _pan_affine_norm(
    x: wp.array3d(dtype=wp.float32),
    scale: wp.array1d(dtype=wp.float32),
    bias: wp.array1d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32),
):
    row, token, channel = wp.tid()
    mean = wp.float32(0.0)
    for index in range(32):
        mean += x[row, token, index]
    mean /= wp.float32(32.0)
    variance = wp.float32(0.0)
    for index in range(32):
        delta = x[row, token, index] - mean
        variance += delta * delta
    inverse = wp.float32(1.0) / wp.sqrt(
        variance / wp.float32(32.0) + wp.float32(1.0e-5)
    )
    output[row, token, channel] = (x[row, token, channel] - mean) * inverse * scale[
        channel
    ] + bias[channel]


@wp.kernel(enable_backward=False, module="unique")
def _pan_add(
    left: wp.array3d(dtype=wp.float32),
    right: wp.array3d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    output[i, j, k] = left[i, j, k] + right[i, j, k]


@wp.kernel(enable_backward=False, module="unique")
def _pan_select_parts(
    x: wp.array3d(dtype=wp.float32), output: wp.array3d(dtype=wp.float32)
):
    batch, frame, channel = wp.tid()
    part = channel / 32
    feature = channel % 32
    output[batch, frame, channel] = x[batch * output.shape[1] + frame, part, feature]


@wp.kernel(enable_backward=False, module="unique")
def _pan_skeleton_add(
    x: wp.array3d(dtype=wp.float32),
    skeleton: wp.array2d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32),
):
    batch, frame, channel = wp.tid()
    output[batch, frame, channel] = (
        x[batch, frame, channel] + skeleton[batch, channel] / 100.0
    )


class _ReflectConv:
    def __init__(self, x, weight, bias, *, stride=1):
        self.padding = ReflectPad1dPlan(x, weight.shape[2] // 2)
        self.conv = Conv1dPlan(self.padding.output, weight, bias, stride=stride)
        self.output = self.conv.output

    def execute(self):
        self.padding.execute()
        return self.conv.execute()


class PanQuadrupedPlan:
    """Fixed-frame CUDA-graphable PAN human-to-dog inference plan."""

    def __init__(self, frames, root, *, device=None):
        if frames < 16 or frames % 4:
            raise ValueError("PAN requires a frame count divisible by four")
        self.frames = int(frames)
        self.device = wp.get_device(device)
        root = Path(root)
        human = load_pytorch_zip(root / "human" / "ae.pth")
        dog = load_pytorch_zip(root / "dog" / "ae.pth")
        skeleton = load_pytorch_zip(root / "dog" / "skel_enc.pth")

        def array(value):
            return wp.array(
                np.ascontiguousarray(value, dtype=np.float32), device=self.device
            )

        def masked(source, key):
            return array(source[key] * source[key.replace("weight", "mask")])

        self.input = wp.empty((1, frames, 92), dtype=wp.float32, device=self.device)
        source = self.input.reshape((1, frames, 23, 4))
        first = BiasedLinearPlan(
            source,
            array(human["enc.joint_pos_encoder.linear1.weight"]),
            array(human["enc.joint_pos_encoder.linear1.bias"]),
            activation="relu",
        )
        second = BiasedLinearPlan(
            first.output,
            array(human["enc.joint_pos_encoder.linear2.weight"]),
            array(human["enc.joint_pos_encoder.linear2.bias"]),
        )
        self.source_projection = (first, second)
        self.tokens = wp.empty((frames, 26, 32), dtype=wp.float32, device=self.device)
        self.part_tokens = array(human["enc.parameter_part"])
        self.position = array(human["enc.joint_pos_encoder.pe"])
        self.allowed = wp.array(_attention_mask(), dtype=wp.bool, device=self.device)
        prefix = "enc.spatialTransEncoder.layers.0."
        self.qkv = BiasedLinearPlan(
            self.tokens,
            array(human[prefix + "self_attn.in_proj_weight"]),
            array(human[prefix + "self_attn.in_proj_bias"]),
        )
        self.attended = wp.empty_like(self.tokens)
        self.attention_output = BiasedLinearPlan(
            self.attended,
            array(human[prefix + "self_attn.out_proj.weight"]),
            array(human[prefix + "self_attn.out_proj.bias"]),
        )
        self.attention_residual = wp.empty_like(self.tokens)
        self.normalized = wp.empty_like(self.tokens)
        self.norm_scale = array(human[prefix + "norm1.weight"])
        self.norm_bias = array(human[prefix + "norm1.bias"])
        self.ff1 = BiasedLinearPlan(
            self.normalized,
            array(human[prefix + "linear1.weight"]),
            array(human[prefix + "linear1.bias"]),
            activation="gelu",
        )
        self.ff2 = BiasedLinearPlan(
            self.ff1.output,
            array(human[prefix + "linear2.weight"]),
            array(human[prefix + "linear2.bias"]),
        )
        self.ff_residual = wp.empty_like(self.tokens)
        self.transformer_output = wp.empty_like(self.tokens)
        self.norm2_scale = array(human[prefix + "norm2.weight"])
        self.norm2_bias = array(human[prefix + "norm2.bias"])
        self.parts = wp.empty((1, frames, 96), dtype=wp.float32, device=self.device)

        residual = _ReflectConv(
            self.input,
            masked(human, "enc.conv_residual.weight"),
            array(human["enc.conv_residual.bias"]),
            stride=2,
        )
        temporal1 = _ReflectConv(
            self.parts,
            masked(human, "enc.conv1.weight"),
            array(human["enc.conv1.bias"]),
            stride=2,
        )
        self.temporal_sum = wp.empty_like(residual.output)
        temporal_activation = ElementwiseActivationPlan(self.temporal_sum, "leaky_relu")
        temporal2 = _ReflectConv(
            temporal_activation.output,
            masked(human, "enc.conv2.weight"),
            array(human["enc.conv2.bias"]),
            stride=2,
        )
        latent_activation = ElementwiseActivationPlan(temporal2.output, "leaky_relu")
        self.encoder = (
            residual,
            temporal1,
            temporal_activation,
            temporal2,
            latent_activation,
        )

        offsets = _parse_bvh_skeleton(root / "metadata" / "dog_reference.bvh")[2][
            1:
        ].reshape(1, -1)
        offsets = array(offsets)
        skel1 = BiasedLinearPlan(
            offsets,
            masked(skeleton, "linear1.weight"),
            array(skeleton["linear1.bias"]),
            activation="leaky_relu",
        )
        skel2 = BiasedLinearPlan(
            skel1.output,
            masked(skeleton, "linear2.weight"),
            array(skeleton["linear2.bias"]),
            activation="leaky_relu",
        )
        skel3 = BiasedLinearPlan(
            skel2.output,
            masked(skeleton, "linear3.weight"),
            array(skeleton["linear3.bias"]),
            activation="leaky_relu",
        )
        self.skeleton_encoder = (skel1, skel2, skel3)
        self.conditioned = wp.empty_like(temporal2.output)

        up1 = LinearUpsample1dPlan(self.conditioned, 2)
        decode1 = _ReflectConv(
            up1.output,
            masked(dog, "dec.layers.0.1.weight"),
            array(dog["dec.layers.0.1.bias"]),
        )
        decode_activation = ElementwiseActivationPlan(decode1.output, "leaky_relu")
        up2 = LinearUpsample1dPlan(decode_activation.output, 2)
        decode2 = _ReflectConv(
            up2.output,
            masked(dog, "dec.layers.1.1.weight"),
            array(dog["dec.layers.1.1.bias"]),
        )
        self.decoder = (up1, decode1, decode_activation, up2, decode2)
        self.output = decode2.output
        self._graph = None

    def _execute(self):
        self.source_projection[0].execute()
        self.source_projection[1].execute()
        wp.launch(
            _pan_pack_tokens,
            dim=self.tokens.shape,
            inputs=[
                self.source_projection[1].output,
                self.part_tokens,
                self.position,
                self.tokens,
            ],
            device=self.device,
        )
        self.qkv.execute()
        wp.launch(
            _pan_attention,
            dim=self.attended.shape,
            inputs=[self.qkv.output, self.allowed, self.attended],
            device=self.device,
        )
        self.attention_output.execute()
        wp.launch(
            _pan_add,
            dim=self.attention_residual.shape,
            inputs=[self.tokens, self.attention_output.output, self.attention_residual],
            device=self.device,
        )
        wp.launch(
            _pan_affine_norm,
            dim=self.normalized.shape,
            inputs=[
                self.attention_residual,
                self.norm_scale,
                self.norm_bias,
                self.normalized,
            ],
            device=self.device,
        )
        self.ff1.execute()
        self.ff2.execute()
        wp.launch(
            _pan_add,
            dim=self.ff_residual.shape,
            inputs=[self.normalized, self.ff2.output, self.ff_residual],
            device=self.device,
        )
        wp.launch(
            _pan_affine_norm,
            dim=self.transformer_output.shape,
            inputs=[
                self.ff_residual,
                self.norm2_scale,
                self.norm2_bias,
                self.transformer_output,
            ],
            device=self.device,
        )
        wp.launch(
            _pan_select_parts,
            dim=self.parts.shape,
            inputs=[self.transformer_output, self.parts],
            device=self.device,
        )
        self.encoder[0].execute()
        self.encoder[1].execute()
        wp.launch(
            _pan_add,
            dim=self.temporal_sum.shape,
            inputs=[self.encoder[0].output, self.encoder[1].output, self.temporal_sum],
            device=self.device,
        )
        self.encoder[2].execute()
        self.encoder[3].execute()
        self.encoder[4].execute()
        for plan in self.skeleton_encoder:
            plan.execute()
        wp.launch(
            _pan_skeleton_add,
            dim=self.conditioned.shape,
            inputs=[
                self.encoder[4].output,
                self.skeleton_encoder[-1].output,
                self.conditioned,
            ],
            device=self.device,
        )
        for plan in self.decoder:
            plan.execute()

    def run(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.shape != self.input.shape:
            raise ValueError(f"PAN input must have shape {self.input.shape}")
        self.input.assign(features)
        if self._graph is None:
            self._execute()
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                self._execute()
            self._graph = capture.graph
        wp.capture_launch(self._graph)
        return self.output


class PanQuadrupedRetargeter:
    """Resident Kimodo-to-dog retargeter with one plan per frame count."""

    def __init__(self, root, *, device=None):
        self.root = Path(root)
        self.device = wp.get_device(device)
        metadata = self.root / "metadata"
        human_stats = np.load(metadata / "humstats.npz")
        dog_stats = np.load(metadata / "dogstats.npz")
        self.human_mean = human_stats["mean"]
        self.human_std = human_stats["std"]
        self.dog_mean = dog_stats["mean"].astype(np.float32)
        self.dog_std = dog_stats["std"].astype(np.float32)
        self.human_names, self.human_parents, _ = _parse_bvh_skeleton(
            metadata / "human_reference.bvh"
        )
        self.names, self.parents, self.offsets = _parse_bvh_skeleton(
            metadata / "dog_reference.bvh"
        )
        if self.human_names != _HUMAN_JOINTS or len(self.names) != 21:
            raise ValueError("PAN reference skeleton hierarchy is incompatible")
        self._plans = {}

    def retarget(self, motion):
        positions = np.asarray(motion["posed_joints"])
        frames = positions.shape[-3]
        usable = frames - frames % 4
        if usable < 16:
            raise ValueError("quadruped motion needs at least 16 frames")
        if positions.ndim == 4:
            sliced_positions = positions[:, :usable]
        elif positions.ndim == 3:
            sliced_positions = positions[:usable]
        else:
            raise ValueError("posed joints must be [frames,joints,3] or batched")
        if "global_rot_mats" not in motion:
            raise ValueError("quadruped retargeting requires Kimodo global_rot_mats")
        source_rotations = np.asarray(motion["global_rot_mats"])
        sliced_rotations = (
            source_rotations[:, :usable]
            if source_rotations.ndim == 5
            else source_rotations[:usable]
        )
        features, yaw = _human_features(
            {
                "posed_joints": sliced_positions,
                "global_rot_mats": sliced_rotations,
            },
            self.human_parents,
            self.human_mean,
            self.human_std,
        )
        plan = self._plans.get(usable)
        if plan is None:
            plan = PanQuadrupedPlan(usable, self.root, device=self.device)
            self._plans[usable] = plan
        normalized = plan.run(features).numpy()[0]
        decoded = normalized * self.dog_std + self.dog_mean
        quaternions = decoded[:, :84].reshape(usable, 21, 4)
        quaternions /= np.maximum(
            np.linalg.norm(quaternions, axis=-1, keepdims=True), 1.0e-12
        )
        local = _quaternion_to_matrix(quaternions)
        heading = np.zeros((usable, 3, 3), dtype=np.float32)
        heading[:, 0, 0] = np.cos(yaw)
        heading[:, 0, 2] = np.sin(yaw)
        heading[:, 1, 1] = 1.0
        heading[:, 2, 0] = -np.sin(yaw)
        heading[:, 2, 2] = np.cos(yaw)
        local[:, 0] = heading @ local[:, 0]
        velocity = np.einsum("tij,tj->ti", heading, decoded[:, 84:87])
        root = np.cumsum(velocity, axis=0)
        posed = np.empty((usable, 21, 3), dtype=np.float32)
        global_rotation = np.empty((usable, 21, 3, 3), dtype=np.float32)
        posed[:, 0] = root
        global_rotation[:, 0] = local[:, 0]
        for joint in range(1, 21):
            parent = int(self.parents[joint])
            global_rotation[:, joint] = global_rotation[:, parent] @ local[:, joint]
            posed[:, joint] = posed[:, parent] + np.einsum(
                "tij,j->ti", global_rotation[:, parent], self.offsets[joint]
            )
        posed *= np.float32(0.01)
        posed[..., 1] -= posed[..., 1].min()
        original_posed = posed
        posed, contacts, paws = _stabilize_paws(posed)
        global_rotation, local = _rotations_for_corrected_limbs(
            original_posed, posed, global_rotation, self.parents
        )
        return {
            "posed_joints": posed,
            "local_rot_mats": local,
            "global_rot_mats": global_rotation,
            "root_positions": posed[:, 0],
            "foot_contacts": contacts,
            "contact_joints": tuple(int(paw) for paw in paws),
            "joint_names": self.names,
            "parents": self.parents,
        }


__all__ = ["PanQuadrupedPlan", "PanQuadrupedRetargeter"]
