# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Small dependency-free host geometry algorithms shared by model frontends."""

from __future__ import annotations

import numpy as np


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


__all__ = ["farthest_point_indices"]
