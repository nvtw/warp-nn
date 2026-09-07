# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""SkinTokens mesh preprocessing, token grammar, and result postprocessing.

The neural runner is optional; these deterministic host operations mirror the
official MIT-licensed implementation without importing PyTorch, SciPy,
trimesh, or Blender.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..formats.mesh import TriangleMesh
from ..skinning import normalize_skin_weights


@dataclass(frozen=True)
class SkinTokensGeometry:
    """Normalized mesh and sampled point/normal conditioning for TokenRig."""

    mesh: TriangleMesh
    normalized_vertices: np.ndarray
    sampled_vertices: np.ndarray
    sampled_normals: np.ndarray
    world_to_model: np.ndarray
    model_to_world: np.ndarray


def _apply_transform(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def normalize_geometry(vertices, joints=None):
    """Return the official isotropic ``[-1, 1]`` transform and its inverse."""
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must have shape [vertices, 3]")
    points = (
        vertices
        if joints is None
        else np.concatenate((vertices, np.asarray(joints, dtype=np.float32)), axis=0)
    )
    lower, upper = points.min(axis=0), points.max(axis=0)
    extent = float(np.max(upper - lower))
    if not np.isfinite(extent) or extent <= 1.0e-12:
        raise ValueError("mesh bounds are degenerate")
    scale = 2.0 / extent
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] *= scale
    transform[:3, 3] = -(upper + lower) * (0.5 * scale)
    return transform, np.linalg.inv(transform).astype(np.float32)


def sample_surface(mesh: TriangleMesh, count: int, *, seed: int = 0):
    """Sample triangle surfaces using the official reflected-barycentric rule."""
    if count <= 0:
        raise ValueError("sample count must be positive")
    triangles = mesh.vertices[mesh.faces]
    edges0 = triangles[:, 1] - triangles[:, 0]
    edges1 = triangles[:, 2] - triangles[:, 0]
    weights = np.linalg.norm(np.cross(edges0, edges1), axis=-1)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 1.0e-20:
        raise ValueError("mesh has no nondegenerate triangle area")
    rng = np.random.RandomState(seed)
    # The official mixed sampler draws this permutation even when zero source
    # vertices are retained; preserving the RNG step gives exact seeded parity.
    rng.permutation(len(mesh.vertices))
    selected = np.searchsorted(np.cumsum(weights), rng.rand(count) * total)
    lengths = rng.rand(count, 2, 1)
    reflected = lengths.sum(axis=1)[:, 0] > 1.0
    lengths[reflected] -= 1.0
    lengths = np.abs(lengths)
    points = triangles[selected, 0] + (
        np.stack((edges0[selected], edges1[selected]), axis=1) * lengths
    ).sum(axis=1)
    _, face_normals = mesh.normals()
    return points.astype(np.float32), face_normals[selected]


def prepare_geometry(mesh: TriangleMesh, *, points: int = 8192, seed: int = 0):
    """Normalize and sample an unrigged mesh for the published TokenRig graphs."""
    world_to_model, model_to_world = normalize_geometry(mesh.vertices)
    normalized = _apply_transform(mesh.vertices, world_to_model).astype(np.float32)
    normalized_mesh = TriangleMesh(normalized, mesh.faces)
    sampled_vertices, sampled_normals = sample_surface(
        normalized_mesh, points, seed=seed
    )
    return SkinTokensGeometry(
        mesh=mesh,
        normalized_vertices=normalized,
        sampled_vertices=sampled_vertices,
        sampled_normals=sampled_normals,
        world_to_model=world_to_model,
        model_to_world=model_to_world,
    )


def transfer_skin_weights(
    vertices,
    sampled_vertices,
    sampled_weights,
    *,
    neighbors: int = 8,
    chunk_size: int = 512,
):
    """Transfer sampled weights to mesh vertices by inverse-distance k-NN.

    This exact, chunked NumPy implementation avoids SciPy and bounds temporary
    memory.  A coincident sample is copied exactly instead of divided by zero.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    samples = np.asarray(sampled_vertices, dtype=np.float32)
    weights = np.asarray(sampled_weights, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [vertices, 3]")
    if samples.ndim != 2 or samples.shape[1] != 3 or not len(samples):
        raise ValueError("sampled_vertices must have shape [samples, 3]")
    if weights.ndim != 2 or weights.shape[0] != len(samples):
        raise ValueError("sampled_weights must have shape [samples, joints]")
    if neighbors <= 0 or chunk_size <= 0:
        raise ValueError("neighbors and chunk_size must be positive")
    neighbors = min(int(neighbors), len(samples))
    output = np.empty((len(vertices), weights.shape[1]), dtype=np.float32)
    for start in range(0, len(vertices), chunk_size):
        current = vertices[start : start + chunk_size]
        squared = np.sum((current[:, None] - samples[None]) ** 2, axis=-1)
        indices = np.argpartition(squared, neighbors - 1, axis=1)[:, :neighbors]
        distances = np.take_along_axis(squared, indices, axis=1)
        exact = distances <= 1.0e-20
        factors = np.divide(
            1.0,
            np.sqrt(np.maximum(distances, 1.0e-20)),
            out=np.empty_like(distances),
        )
        factors /= factors.sum(axis=1, keepdims=True)
        if np.any(exact):
            factors[exact.any(axis=1)] = exact[exact.any(axis=1)] / exact[
                exact.any(axis=1)
            ].sum(axis=1, keepdims=True)
        output[start : start + len(current)] = np.einsum(
            "vk,vkj->vj", factors, weights[indices]
        )
    return normalize_skin_weights(output)


@dataclass(frozen=True)
class DecodedSkeleton:
    joints: np.ndarray
    parents: np.ndarray
    joint_names: tuple[str, ...]
    cls: str | None


class SkinTokensTokenizer:
    """Official 256-bin TokenRig skeleton grammar and SkinToken vocabulary."""

    num_discrete = 256
    branch = 256
    bos = 257
    switch = 258
    pad = 259
    spring = 260
    body = 261
    hand = 262
    cls_none = 263
    class_tokens = {"rignet": 264, "vroid": 265, "articulation": 266}
    skeleton_vocab = 267
    skin_codebook = 32768
    global_eos = 33035
    full_vocab = 33036
    tokens_per_skin = 4

    @staticmethod
    def discretize(value):
        value = np.asarray(value, dtype=np.float32)
        return np.clip(np.rint((value + 1.0) * 128.0), 0, 255).astype(np.int64)

    @staticmethod
    def undiscretize(value):
        return (np.asarray(value, dtype=np.float32) + 0.5) / 128.0 - 1.0

    def start_tokens(self, cls="articulation"):
        return np.asarray(
            (self.bos, self.class_tokens.get(cls, self.cls_none)), dtype=np.int64
        )

    def tokenize_skeleton(self, joints, parents, *, cls="articulation"):
        joints = np.asarray(joints, dtype=np.float32)
        parents = np.asarray(parents, dtype=np.int32)
        if joints.ndim != 2 or joints.shape[1] != 3 or parents.shape != (len(joints),):
            raise ValueError(
                "joints and parents must have shapes [joints,3] and [joints]"
            )
        if np.count_nonzero(parents == -1) != 1 or parents[0] != -1:
            raise ValueError("skeleton must have one root at index zero")
        if np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, len(parents))):
            raise ValueError("parents must precede children")
        discrete = self.discretize(joints)
        tokens = list(self.start_tokens(cls))
        previous = None
        for joint, parent in enumerate(parents):
            is_branch = joint > 0 and parent != previous
            if is_branch:
                tokens.append(self.branch)
                tokens.extend(discrete[parent])
            tokens.extend(discrete[joint])
            previous = joint
        tokens.append(self.switch)
        return np.asarray(tokens, dtype=np.int64)

    def decode_skeleton(self, tokens):
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if len(tokens) < 6 or tokens[0] != self.bos:
            raise ValueError("skeleton token stream must start with BOS")
        try:
            end = int(np.flatnonzero(tokens == self.switch)[0])
        except IndexError as exc:
            raise ValueError("skeleton token stream has no switch token") from exc
        stream = tokens[1:end]
        cls = None
        if len(stream) and stream[0] >= self.cls_none:
            inverse = {value: name for name, value in self.class_tokens.items()}
            cls = inverse.get(int(stream[0]))
            stream = stream[1:]
        joints, parent_points, last = [], [], None
        branch = False
        index = 0
        while index < len(stream):
            token = int(stream[index])
            if token in (self.spring, self.body, self.hand):
                index += 1
                continue
            if token == self.branch:
                branch = True
                last = None
                index += 1
                continue
            width = 6 if branch else 3
            values = stream[index : index + width]
            if len(values) != width or np.any(values >= self.num_discrete):
                raise ValueError("invalid or truncated skeleton coordinate tokens")
            if branch:
                parent_point = self.undiscretize(values[:3])
                point = self.undiscretize(values[3:])
            else:
                point = self.undiscretize(values)
                parent_point = point if not joints else joints[last]
            joints.append(point)
            parent_points.append(parent_point)
            last = len(joints) - 1
            branch = False
            index += width
        if not joints:
            raise ValueError("skeleton token stream contains no joints")
        names = tuple(f"bone_{index}" for index in range(len(joints)))
        return DecodedSkeleton(
            np.asarray(joints, dtype=np.float32),
            np.asarray(
                [-1]
                + [
                    int(
                        np.argmin(
                            np.sum(
                                (np.asarray(joints[:i]) - parent_points[i]) ** 2, axis=1
                            )
                        )
                    )
                    for i in range(1, len(joints))
                ],
                dtype=np.int32,
            ),
            names,
            cls,
        )

    def allowed_tokens(self, sequence):
        """Return legal next token IDs, including the fixed SkinToken phase."""
        sequence = np.asarray(sequence, dtype=np.int64).reshape(-1)
        switches = np.flatnonzero(sequence == self.switch)
        if len(switches):
            skeleton = self.decode_skeleton(sequence[: switches[0] + 1])
            generated = len(sequence) - switches[0] - 1
            expected = len(skeleton.joints) * self.tokens_per_skin
            if generated < expected:
                return np.arange(self.skeleton_vocab, self.global_eos, dtype=np.int64)
            return np.asarray((self.global_eos,), dtype=np.int64)
        state = "expect_bos"
        for token in sequence:
            token = int(token)
            if state == "expect_bos":
                if token != self.bos:
                    raise ValueError("token stream does not start with BOS")
                state = "class_or_part_or_joint"
            elif state == "class_or_part_or_joint":
                if token < self.num_discrete:
                    state = "joint_2"
                elif token == self.cls_none or token in self.class_tokens.values():
                    state = "part_or_joint"
                else:
                    state = "joint"
            elif state == "part_or_joint":
                if token < self.num_discrete:
                    state = "joint_2"
            elif state == "joint_2":
                state = "joint_3"
            elif state == "joint_3":
                state = "boundary"
            elif state == "boundary":
                state = "joint" if token >= self.num_discrete else "joint_2"
            elif state == "joint":
                state = "joint_2"
        coordinates = np.arange(self.num_discrete, dtype=np.int64)
        parts = np.asarray((self.spring, self.body, self.hand), dtype=np.int64)
        classes = np.asarray(
            (self.cls_none, *self.class_tokens.values()), dtype=np.int64
        )
        if state == "expect_bos":
            return np.asarray((self.bos,), dtype=np.int64)
        if state == "class_or_part_or_joint":
            return np.concatenate((coordinates, classes, parts))
        if state == "part_or_joint":
            return np.concatenate((coordinates, parts, (self.switch,)))
        if state in ("joint", "joint_2", "joint_3"):
            return coordinates
        return np.concatenate(
            (coordinates, parts, np.asarray((self.branch, self.switch)))
        )


__all__ = [
    "DecodedSkeleton",
    "SkinTokensGeometry",
    "SkinTokensTokenizer",
    "normalize_geometry",
    "prepare_geometry",
    "sample_surface",
    "transfer_skin_weights",
]
