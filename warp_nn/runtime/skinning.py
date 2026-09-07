# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Portable rigged-mesh data and linear blend skinning.

The module intentionally contains no model policy.  A rig may come from
SkinTokens, an artist-authored GLB, or another auto-rigger; motion may come
from Kimodo or any source that can provide posed joints and global rotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def normalize_skin_weights(weights, *, max_influences: int | None = 4):
    """Return finite, nonnegative, normalized skin weights.

    When ``max_influences`` is set, ties are resolved by joint index so the
    result is deterministic across platforms.  Vertices without an influence
    are attached to joint zero rather than producing NaNs.
    """
    result = np.asarray(weights, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError("skin weights must have shape [vertices, joints]")
    if max_influences is not None and max_influences <= 0:
        raise ValueError("max_influences must be positive or None")
    result[~np.isfinite(result)] = 0.0
    np.maximum(result, 0.0, out=result)
    if max_influences is not None and max_influences < result.shape[1]:
        # Stable sorting makes equal weights prefer the lower joint index.
        keep = np.argsort(-result, axis=1, kind="stable")[:, :max_influences]
        sparse = np.zeros_like(result)
        rows = np.arange(result.shape[0])[:, None]
        sparse[rows, keep] = result[rows, keep]
        result = sparse
    totals = result.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= np.finfo(np.float32).tiny
    result[empty, 0] = 1.0
    totals[empty, 0] = 1.0
    result /= totals
    return result


@dataclass(frozen=True)
class RiggedMesh:
    """A triangle mesh, bind skeleton, and dense per-vertex joint weights."""

    vertices: np.ndarray
    faces: np.ndarray
    rest_joints: np.ndarray
    parents: np.ndarray
    weights: np.ndarray
    joint_names: tuple[str, ...] | None = None

    def __post_init__(self):
        vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        joints = np.ascontiguousarray(self.rest_joints, dtype=np.float32)
        parents = np.ascontiguousarray(self.parents, dtype=np.int32)
        weights = normalize_skin_weights(self.weights)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape [vertices, 3]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape [triangles, 3]")
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("faces contain an invalid vertex index")
        if joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError("rest_joints must have shape [joints, 3]")
        if parents.shape != (len(joints),):
            raise ValueError("parents must contain one entry per joint")
        if weights.shape != (len(vertices), len(joints)):
            raise ValueError("weights must have shape [vertices, joints]")
        if np.count_nonzero(parents == -1) != 1:
            raise ValueError("the skeleton must have exactly one root")
        for joint, parent in enumerate(parents):
            if parent >= joint or parent < -1:
                raise ValueError("parents must precede children in hierarchy order")
        if self.joint_names is not None and len(self.joint_names) != len(joints):
            raise ValueError("joint_names must contain one name per joint")
        for value, name in (
            (vertices, "vertices"),
            (joints, "rest_joints"),
            (weights, "weights"),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "rest_joints", joints)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "weights", weights)
        if self.joint_names is not None:
            object.__setattr__(self, "joint_names", tuple(map(str, self.joint_names)))

    @property
    def top4(self):
        """Return compact joint indices and weights for rendering/export."""
        count = min(4, self.weights.shape[1])
        indices = np.argsort(-self.weights, axis=1, kind="stable")[:, :count]
        rows = np.arange(len(self.vertices))[:, None]
        compact = self.weights[rows, indices]
        if count < 4:
            indices = np.pad(indices, ((0, 0), (0, 4 - count)))
            compact = np.pad(compact, ((0, 0), (0, 4 - count)))
        return indices.astype(np.uint16), compact


def skinning_transforms(rest_joints, posed_joints, global_rotations):
    """Build affine LBS transforms from bind and posed joint data.

    Bind rotations are identity, which matches SkinTokens' joint-position rig
    representation and Kimodo's decoded global rotations.
    """
    rest = np.asarray(rest_joints, dtype=np.float32)
    posed = np.asarray(posed_joints, dtype=np.float32)
    rotations = np.asarray(global_rotations, dtype=np.float32)
    if rest.ndim != 2 or rest.shape[1] != 3:
        raise ValueError("rest_joints must have shape [joints, 3]")
    if posed.shape[-2:] != rest.shape or rotations.shape[-3:] != (
        len(rest),
        3,
        3,
    ):
        raise ValueError("posed joints and rotations do not match the bind skeleton")
    if posed.shape[:-2] != rotations.shape[:-3]:
        raise ValueError("posed joints and rotations must have matching batch axes")
    translation = posed - np.einsum("...jik,jk->...ji", rotations, rest)
    transforms = np.zeros((*posed.shape[:-2], len(rest), 3, 4), dtype=np.float32)
    transforms[..., :3] = rotations
    transforms[..., 3] = translation
    return transforms


def linear_blend_skinning(vertices, weights, transforms):
    """Deform vertices with dense weights and affine ``[..., joints, 3, 4]`` transforms."""
    vertices = np.asarray(vertices, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    transforms = np.asarray(transforms, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [vertices, 3]")
    if weights.shape != (len(vertices), transforms.shape[-3]):
        raise ValueError("weights do not match vertices and transforms")
    if transforms.shape[-2:] != (3, 4):
        raise ValueError("transforms must end in [joints, 3, 4]")
    homogeneous = np.concatenate(
        (vertices, np.ones((len(vertices), 1), dtype=np.float32)), axis=1
    )
    moved = np.einsum("...jik,vk->...jvi", transforms, homogeneous)
    return np.sum(moved * weights.T[..., None], axis=-3).astype(np.float32, copy=False)


def deform_rigged_mesh(mesh: RiggedMesh, posed_joints, global_rotations):
    """Convenience wrapper that deforms ``mesh`` for one pose or a motion."""
    transforms = skinning_transforms(mesh.rest_joints, posed_joints, global_rotations)
    return linear_blend_skinning(mesh.vertices, mesh.weights, transforms)


def save_rigged_mesh(path: str | Path, mesh: RiggedMesh):
    """Save a portable rig without pickles or model-specific dependencies."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = np.asarray(mesh.joint_names or (), dtype=np.str_)
    np.savez_compressed(
        destination,
        vertices=mesh.vertices,
        faces=mesh.faces,
        rest_joints=mesh.rest_joints,
        parents=mesh.parents,
        weights=mesh.weights,
        joint_names=names,
    )
    return destination


def load_rigged_mesh(path: str | Path):
    """Load the portable rig format written by :func:`save_rigged_mesh`."""
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        names = tuple(str(value) for value in data["joint_names"])
        return RiggedMesh(
            vertices=data["vertices"],
            faces=data["faces"],
            rest_joints=data["rest_joints"],
            parents=data["parents"],
            weights=data["weights"],
            joint_names=names or None,
        )


__all__ = [
    "RiggedMesh",
    "deform_rigged_mesh",
    "linear_blend_skinning",
    "load_rigged_mesh",
    "normalize_skin_weights",
    "save_rigged_mesh",
    "skinning_transforms",
]
