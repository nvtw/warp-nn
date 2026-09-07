# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Adapters between Kimodo's SOMA-30 skeleton and unrigged character meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..formats.mesh import TriangleMesh, load_obj, obj_group_vertex_centers


MAKEHUMAN_SOMA30_GROUPS = (
    "joint-pelvis",
    "joint-spine-4",
    "joint-spine-3",
    "joint-spine-2",
    "joint-spine-1",
    "joint-neck",
    "joint-head",
    "joint-jaw",
    "joint-l-eye",
    "joint-r-eye",
    "joint-l-clavicle",
    "joint-l-shoulder",
    "joint-l-elbow",
    "joint-l-hand",
    "joint-l-finger-1-4",
    "joint-l-finger-3-4",
    "joint-r-clavicle",
    "joint-r-shoulder",
    "joint-r-elbow",
    "joint-r-hand",
    "joint-r-finger-1-4",
    "joint-r-finger-3-4",
    "joint-l-upper-leg",
    "joint-l-knee",
    "joint-l-ankle",
    "joint-l-foot-1",
    "joint-r-upper-leg",
    "joint-r-knee",
    "joint-r-ankle",
    "joint-r-foot-1",
)


def load_makehuman_soma30(path: str | Path):
    """Load MakeHuman's CC0 body and markers in warp-nn's meter-scale convention."""
    path = Path(path).expanduser()
    mesh = load_obj(path, groups="body")
    # MakeHuman's native OBJ coordinate unit is one decimeter.
    scale = np.float32(0.1)
    return (
        TriangleMesh(mesh.vertices * scale, mesh.faces),
        obj_group_vertex_centers(path, MAKEHUMAN_SOMA30_GROUPS) * scale,
    )


__all__ = ["MAKEHUMAN_SOMA30_GROUPS", "load_makehuman_soma30"]
