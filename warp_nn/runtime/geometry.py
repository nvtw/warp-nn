# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Small dependency-free host geometry algorithms shared by model frontends."""

from __future__ import annotations

import numpy as np


def rotation_between_vectors(source, target) -> np.ndarray:
    """Return the minimum rotation taking one nonzero 3D vector onto another."""
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source = source / max(np.linalg.norm(source), 1.0e-8)
    target = target / max(np.linalg.norm(target), 1.0e-8)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine < 1.0e-7:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float32)
        basis = np.eye(3, dtype=np.float32)[np.argmin(np.abs(source))]
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return (2.0 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)
    x, y, z = cross
    skew = np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float32)
    return np.eye(3, dtype=np.float32) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def fabrik_chain(points, target, *, iterations: int = 16) -> np.ndarray:
    """Move one joint chain to a target while preserving every bone length."""
    result = np.asarray(points, dtype=np.float32).copy()
    target = np.asarray(target, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 2:
        raise ValueError("points must have shape [joints >= 2, 3]")
    if target.shape != (3,) or iterations < 1:
        raise ValueError("target must be a 3D point and iterations must be positive")
    base = result[0].copy()
    lengths = np.linalg.norm(np.diff(result, axis=0), axis=-1)
    distance = np.linalg.norm(target - base)
    if distance >= lengths.sum():
        direction = (target - base) / max(distance, 1.0e-8)
        for index, length in enumerate(lengths):
            result[index + 1] = result[index] + direction * length
        return result
    for _ in range(iterations):
        result[-1] = target
        for index in range(len(result) - 2, -1, -1):
            direction = result[index] - result[index + 1]
            direction /= max(np.linalg.norm(direction), 1.0e-8)
            result[index] = result[index + 1] + direction * lengths[index]
        result[0] = base
        for index, length in enumerate(lengths):
            direction = result[index + 1] - result[index]
            direction /= max(np.linalg.norm(direction), 1.0e-8)
            result[index + 1] = result[index] + direction * length
    return result


def farthest_point_indices(points, count: int) -> np.ndarray:
    """Select deterministic farthest points, starting at index zero.

    Distances stay FP32 to match neural point-cloud preprocessing. Stable
    ``argmax`` tie-breaking makes the result independent of GPU reductions.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or not len(points):
        raise ValueError("points must have shape [points, dimensions]")
    if not 1 <= count <= len(points):
        raise ValueError("sample count must be between one and the point count")
    selected = np.empty(count, dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float32)
    farthest = 0
    for index in range(count):
        selected[index] = farthest
        delta = points - points[farthest]
        candidate = np.sum(delta * delta, axis=1, dtype=np.float32)
        np.minimum(distances, candidate, out=distances)
        farthest = int(np.argmax(distances))
    return selected


__all__ = ["fabrik_chain", "farthest_point_indices", "rotation_between_vectors"]
