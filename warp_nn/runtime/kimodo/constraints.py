# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free authoring of Kimodo pose, joint, and path constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOMA30_JOINT_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumbEnd",
    "LeftHandMiddleEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumbEnd",
    "RightHandMiddleEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)
SOMA30_PARENTS = np.asarray(
    (
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        6,
        6,
        3,
        10,
        11,
        12,
        13,
        13,
        3,
        16,
        17,
        18,
        19,
        19,
        0,
        22,
        23,
        24,
        0,
        26,
        27,
        28,
    ),
    dtype=np.int32,
)
SOMA30_END_EFFECTORS = {
    "LeftFoot": ((24, 25), (24,)),
    "RightFoot": ((28, 29), (28,)),
    "LeftHand": ((13, 15), (13,)),
    "RightHand": ((19, 21), (19,)),
    "Hips": ((0,), (0,)),
}


@dataclass
class KimodoConstraints:
    """Composable constraints in Kimodo's unnormalized motion representation."""

    observed: np.ndarray
    mask: np.ndarray

    @classmethod
    def empty(cls, frames: int, joints: int = 30) -> "KimodoConstraints":
        if frames < 2 or joints < 1:
            raise ValueError("constraints require at least two frames and one joint")
        shape = (1, int(frames), 12 * int(joints) + 9)
        return cls(np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=bool))

    @classmethod
    def load(cls, path: str | Path) -> "KimodoConstraints":
        """Load warp-nn's compact, model-native constraint archive."""
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != {"observed", "mask"}:
                raise ValueError("constraint NPZ must contain only observed and mask")
            return cls(archive["observed"], archive["mask"])

    def __post_init__(self):
        self.observed = np.asarray(self.observed, dtype=np.float32)
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.observed.ndim != 3 or self.observed.shape[0] != 1:
            raise ValueError("observed motion must have shape [1, frames, features]")
        if self.mask.shape != self.observed.shape:
            raise ValueError("constraint mask must match observed motion")
        if self.observed.shape[2] < 21 or (self.observed.shape[2] - 9) % 12:
            raise ValueError("observed motion does not have a Kimodo feature width")
        if not np.isfinite(self.observed[self.mask]).all():
            raise ValueError("constrained values must be finite")

    @property
    def frames(self) -> int:
        return self.observed.shape[1]

    @property
    def joints(self) -> int:
        return (self.observed.shape[2] - 9) // 12

    def clear(self) -> None:
        self.observed.fill(0.0)
        self.mask.fill(False)

    def save(self, path: str | Path) -> None:
        """Save constraints without model weights or generated motion."""
        np.savez_compressed(Path(path), observed=self.observed, mask=self.mask)

    @property
    def constrained_frames(self) -> np.ndarray:
        """Sorted frames containing at least one authored value."""
        return np.flatnonzero(np.any(self.mask[0], axis=1))

    def _frames(self, frame_indices) -> np.ndarray:
        indices = np.asarray(frame_indices, dtype=np.int64)
        if indices.ndim == 0:
            indices = indices[None]
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.frames):
            raise ValueError(f"frame indices must be within 0..{self.frames - 1}")
        return indices

    def root_path(self, frame_indices, positions_xz, *, heading=None) -> None:
        """Constrain sparse waypoints or a dense root path in world-space X/Z."""
        frames = self._frames(frame_indices)
        positions = np.asarray(positions_xz, dtype=np.float32)
        if positions.shape != (len(frames), 2) or not np.isfinite(positions).all():
            raise ValueError(
                "positions_xz must have shape [waypoints, 2] and be finite"
            )
        self.observed[0, frames, 0] = positions[:, 0]
        self.observed[0, frames, 2] = positions[:, 1]
        self.mask[0, frames, 0] = True
        self.mask[0, frames, 2] = True
        if heading is not None:
            angles = np.asarray(heading, dtype=np.float32)
            if angles.ndim == 0:
                angles = np.full(len(frames), angles, dtype=np.float32)
            if angles.shape != (len(frames),) or not np.isfinite(angles).all():
                raise ValueError("heading must be one angle in radians per waypoint")
            self.observed[0, frames, 3] = np.cos(angles)
            self.observed[0, frames, 4] = np.sin(angles)
            self.mask[0, frames, 3:5] = True

    def overlay(self, other: "KimodoConstraints", *, frame_offset: int = 0) -> None:
        """Compose another constraint set at a temporal offset; newer values win."""
        if other.joints != self.joints:
            raise ValueError("constraint joint counts do not match")
        start, end = int(frame_offset), int(frame_offset) + other.frames
        if start < 0 or end > self.frames:
            raise ValueError("overlaid constraints do not fit the destination timeline")
        destination = self.observed[:, start:end]
        destination_mask = self.mask[:, start:end]
        destination[other.mask] = other.observed[other.mask]
        destination_mask |= other.mask

    def crop(self, start: int, end: int, *, prefix: int = 0) -> "KimodoConstraints":
        """Copy timeline ``[start, end)`` with optional empty prefix frames."""
        start, end, prefix = int(start), int(end), int(prefix)
        if start < 0 or end > self.frames or start >= end or prefix < 0:
            raise ValueError("invalid constraint crop")
        result = KimodoConstraints.empty(prefix + end - start, self.joints)
        result.observed[:, prefix:] = self.observed[:, start:end]
        result.mask[:, prefix:] = self.mask[:, start:end]
        return result

    def translate_root(self, xz) -> None:
        """Translate constrained root X/Z values without touching unconstrained slots."""
        delta = np.asarray(xz, dtype=np.float32)
        if delta.shape != (2,) or not np.isfinite(delta).all():
            raise ValueError("root translation must be a finite [x, z] vector")
        for column, value in zip((0, 2), delta):
            selected = self.mask[..., column]
            self.observed[..., column][selected] += value

    def pose(
        self,
        frame,
        motion,
        *,
        source_frame=-1,
        joints=None,
        positions: bool = True,
        rotations: bool = False,
    ) -> None:
        """Constrain one or more poses or selected joints from a decoded clip."""
        targets = self._frames(frame)
        features = _motion_features(motion, self.joints)
        sources = np.asarray(source_frame, dtype=np.int64)
        if sources.ndim == 0:
            sources = np.full(len(targets), sources, dtype=np.int64)
        if sources.shape != targets.shape:
            raise ValueError("source_frame must be one index per target frame")
        sources = np.where(sources < 0, sources + len(features), sources)
        if np.any(sources < 0) or np.any(sources >= len(features)):
            raise ValueError(f"source_frame must be within 0..{len(features) - 1}")
        selected = (
            np.arange(self.joints, dtype=np.int64)
            if joints is None
            else np.asarray(joints, dtype=np.int64)
        )
        if selected.ndim == 0:
            selected = selected[None]
        if (
            selected.ndim != 1
            or np.any(selected < 0)
            or np.any(selected >= self.joints)
        ):
            raise ValueError(f"joint indices must be within 0..{self.joints - 1}")

        for target, source in zip(targets, sources):
            # Root position and heading locate otherwise root-relative joint data.
            unset_root = ~self.mask[0, target, :5]
            self.observed[0, target, :5][unset_root] = features[source, :5][unset_root]
            self.mask[0, target, :5] = True
            if positions:
                start = 5
                columns = (start + selected[:, None] * 3 + np.arange(3)).reshape(-1)
                self.observed[0, target, columns] = features[source, columns]
                self.mask[0, target, columns] = True
            if rotations:
                start = 5 + self.joints * 3
                columns = (start + selected[:, None] * 6 + np.arange(6)).reshape(-1)
                self.observed[0, target, columns] = features[source, columns]
                self.mask[0, target, columns] = True
            elif joints is None:
                # Full-body conditioning uses positions only. Retaining source
                # orientations lets post-processing reproduce authored poses.
                start = 5 + self.joints * 3
                width = self.joints * 6
                self.observed[0, target, start : start + width] = features[
                    source, start : start + width
                ]

    def start_pose(self, motion, *, source_frame: int = -1) -> None:
        """Match frame zero to a source pose (the source end by default)."""
        self.pose(0, motion, source_frame=source_frame)

    def end_pose(self, motion, *, source_frame: int = -1) -> None:
        """Match the last frame to a source pose (the source end by default)."""
        self.pose(self.frames - 1, motion, source_frame=source_frame)

    def spatial_targets(
        self,
        frame_indices,
        joints,
        *,
        global_positions=None,
        global_rotations=None,
        root_xz=None,
        root_y=None,
        heading=None,
    ) -> None:
        """Author arbitrary world-space joint targets without a source clip.

        Global positions require the smoothed root X/Z reference used by
        Kimodo's representation. Rotations are 3x3 world-space matrices.
        """
        frames = self._frames(frame_indices)
        selected = np.asarray(joints, dtype=np.int64)
        if selected.ndim == 0:
            selected = selected[None]
        if (
            selected.ndim != 1
            or np.any(selected < 0)
            or np.any(selected >= self.joints)
        ):
            raise ValueError(f"joint indices must be within 0..{self.joints - 1}")
        count = len(frames)
        roots = None
        if root_xz is not None:
            roots = np.asarray(root_xz, dtype=np.float32)
            if roots.ndim == 1:
                roots = np.broadcast_to(roots, (count, 2))
            if roots.shape != (count, 2):
                raise ValueError("root_xz must have shape [frames, 2]")
            self.root_path(frames, roots, heading=heading)
        elif heading is not None:
            raise ValueError("heading requires root_xz")
        if root_y is not None:
            values = np.asarray(root_y, dtype=np.float32)
            if values.ndim == 0:
                values = np.full(count, values, dtype=np.float32)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError("root_y must be one finite value per frame")
            self.observed[0, frames, 1] = values
            self.mask[0, frames, 1] = True
        if global_positions is not None:
            if roots is None:
                raise ValueError("global_positions require root_xz")
            positions = np.asarray(global_positions, dtype=np.float32)
            expected = (count, len(selected), 3)
            if positions.shape != expected or not np.isfinite(positions).all():
                raise ValueError(f"global_positions must have shape {expected}")
            reference = np.zeros((count, 1, 3), dtype=np.float32)
            reference[..., 0] = roots[:, None, 0]
            reference[..., 2] = roots[:, None, 1]
            local = positions - reference
            start = 5
            columns = (start + selected[:, None] * 3 + np.arange(3)).reshape(-1)
            self.observed[0, frames[:, None], columns] = local.reshape(count, -1)
            self.mask[0, frames[:, None], columns] = True
        if global_rotations is not None:
            rotations = np.asarray(global_rotations, dtype=np.float32)
            expected = (count, len(selected), 3, 3)
            if rotations.shape != expected or not np.isfinite(rotations).all():
                raise ValueError(f"global_rotations must have shape {expected}")
            six = np.concatenate((rotations[..., 0], rotations[..., 1]), axis=-1)
            start = 5 + self.joints * 3
            columns = (start + selected[:, None] * 6 + np.arange(6)).reshape(-1)
            self.observed[0, frames[:, None], columns] = six.reshape(count, -1)
            self.mask[0, frames[:, None], columns] = True
        if global_positions is None and global_rotations is None and root_y is None:
            raise ValueError("provide at least one spatial target")

    def end_effectors(self, frame, motion, names, *, source_frame=-1) -> None:
        """Apply official SOMA hand/foot chain position and base-rotation masks."""
        lookup = {name.lower(): value for name, value in SOMA30_END_EFFECTORS.items()}
        position_joints, rotation_joints = [], []
        for name in names:
            if name.lower() not in lookup:
                choices = ", ".join(SOMA30_END_EFFECTORS)
                raise ValueError(
                    f"unknown end effector '{name}'; choose from {choices}"
                )
            positions, rotations = lookup[name.lower()]
            position_joints.extend(positions)
            rotation_joints.extend(rotations)
        if not position_joints:
            raise ValueError("provide at least one end effector")
        self.pose(
            frame,
            motion,
            source_frame=source_frame,
            joints=np.unique(position_joints),
            positions=True,
            rotations=False,
        )
        self.pose(
            frame,
            motion,
            source_frame=source_frame,
            joints=np.unique(rotation_joints),
            positions=False,
            rotations=True,
        )


def _motion_features(motion, joints: int) -> np.ndarray:
    if isinstance(motion, (str, Path)):
        with np.load(Path(motion), allow_pickle=False) as archive:
            if "features" not in archive:
                raise ValueError("Kimodo NPZ does not contain model-native 'features'")
            return _motion_features(
                {name: archive[name] for name in archive.files}, joints
            )
    elif isinstance(motion, dict) and "features" in motion:
        features = np.asarray(motion["features"], dtype=np.float32)
    else:
        features = np.asarray(motion, dtype=np.float32)
    if features.ndim == 3:
        if features.shape[0] != 1:
            raise ValueError("pose source must contain one motion")
        features = features[0]
    expected = 12 * joints + 9
    if features.ndim != 2 or features.shape[1] != expected:
        raise ValueError(f"pose source features must have shape [frames, {expected}]")
    if not np.isfinite(features).all():
        raise ValueError("pose source features must be finite")
    if isinstance(motion, dict) and all(
        name in motion
        for name in (
            "posed_joints",
            "smooth_root_pos",
            "global_root_heading",
            "global_rot_mats",
        )
    ):

        def unbatch(value, trailing):
            value = np.asarray(value, dtype=np.float32)
            return value[0] if value.ndim == trailing + 2 else value

        posed = unbatch(motion["posed_joints"], 2)
        smooth = unbatch(motion["smooth_root_pos"], 1)
        heading = unbatch(motion["global_root_heading"], 1)
        rotations = unbatch(motion["global_rot_mats"], 3)
        if posed.shape != (len(features), joints, 3):
            raise ValueError("pose source posed_joints do not match its features")
        rebuilt = features.copy()
        rebuilt[:, :3] = smooth
        rebuilt[:, 3:5] = heading
        local = posed.copy()
        local[..., 0] -= smooth[:, None, 0]
        local[..., 2] -= smooth[:, None, 2]
        position_end = 5 + joints * 3
        rebuilt[:, 5:position_end] = local.reshape(len(features), -1)
        rebuilt[:, position_end : position_end + joints * 6] = np.concatenate(
            (rotations[..., 0], rotations[..., 1]), axis=-1
        ).reshape(len(features), -1)
        features = rebuilt
    return features
