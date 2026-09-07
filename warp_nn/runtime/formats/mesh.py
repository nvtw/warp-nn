# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light triangle-mesh loading for neural geometry models."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TriangleMesh:
    """A finite indexed triangle mesh in one world-space coordinate system."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self):
        vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("vertices must have shape [vertices, 3]")
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError("faces must have shape [triangles, 3]")
        if not np.isfinite(vertices).all():
            raise ValueError("vertices must be finite")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("faces contain an invalid vertex index")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    def normals(self):
        """Return area-weighted vertex normals and unit face normals."""
        triangles = self.vertices[self.faces]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        lengths = np.linalg.norm(cross, axis=1, keepdims=True)
        face = cross / np.maximum(lengths, 1.0e-20)
        vertex = np.zeros_like(self.vertices)
        for corner in range(3):
            np.add.at(vertex, self.faces[:, corner], cross)
        vertex /= np.maximum(np.linalg.norm(vertex, axis=1, keepdims=True), 1.0e-20)
        return vertex, face.astype(np.float32, copy=False)


def _node_matrix(node):
    if "matrix" in node:
        value = np.asarray(node["matrix"], dtype=np.float64)
        if value.shape != (16,):
            raise ValueError("glTF node matrix must contain 16 values")
        return value.reshape(4, 4).T
    translation = np.asarray(node.get("translation", (0, 0, 0)), dtype=np.float64)
    scale = np.asarray(node.get("scale", (1, 1, 1)), dtype=np.float64)
    x, y, z, w = np.asarray(node.get("rotation", (0, 0, 0, 1)), dtype=np.float64)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-15:
        raise ValueError("glTF node has a zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation * scale[None]
    matrix[:3, 3] = translation
    return matrix


_COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _accessor(document, binary, index):
    accessor = document["accessors"][index]
    if accessor.get("sparse") is not None or "bufferView" not in accessor:
        raise NotImplementedError(
            "sparse or storage-free glTF accessors are unsupported"
        )
    view = document["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise NotImplementedError("GLB accessors must reference the embedded buffer")
    try:
        dtype = _COMPONENT_DTYPES[accessor["componentType"]]
        width = _TYPE_WIDTHS[accessor["type"]]
    except KeyError as exc:
        raise NotImplementedError("unsupported glTF accessor type") from exc
    count = int(accessor["count"])
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", dtype.itemsize * width))
    if stride < dtype.itemsize * width:
        raise ValueError("glTF accessor stride is too small")
    shape = (count,) if width == 1 else (count, width)
    strides = (stride,) if width == 1 else (stride, dtype.itemsize)
    try:
        result = np.ndarray(
            shape, dtype=dtype, buffer=binary, offset=offset, strides=strides
        ).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("glTF accessor exceeds its binary buffer") from exc
    if accessor.get("normalized", False):
        if dtype.kind not in "iu":
            raise ValueError("only integer glTF accessors may be normalized")
        info = np.iinfo(dtype)
        result = result.astype(np.float32)
        result = np.maximum(
            result / (info.max if dtype.kind == "u" else info.max), -1.0
        )
    return result


def load_glb(path: str | Path):
    """Load triangle geometry from a binary glTF, applying scene-node transforms."""
    data = Path(path).expanduser().read_bytes()
    if len(data) < 12:
        raise ValueError("GLB header is truncated")
    magic, version, length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or length != len(data):
        raise ValueError("file is not a valid glTF 2.0 binary")
    chunks = {}
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("GLB chunk header is truncated")
        size, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks[kind] = memoryview(data)[offset : offset + size]
        offset += size
    try:
        document = json.loads(bytes(chunks[0x4E4F534A]).decode("utf-8").rstrip(" \0"))
        binary = chunks[0x004E4942]
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("GLB must contain JSON and binary chunks") from exc

    nodes = document.get("nodes", ())
    roots = []
    scenes = document.get("scenes", ())
    if scenes:
        scene_index = int(document.get("scene", 0))
        roots = list(scenes[scene_index].get("nodes", ()))
    if not roots:
        children = {child for node in nodes for child in node.get("children", ())}
        roots = [index for index in range(len(nodes)) if index not in children]

    vertices_out, faces_out = [], []

    def visit(node_index, parent_matrix):
        node = nodes[node_index]
        world = parent_matrix @ _node_matrix(node)
        if "mesh" in node:
            mesh = document["meshes"][node["mesh"]]
            for primitive in mesh.get("primitives", ()):
                if int(primitive.get("mode", 4)) != 4:
                    raise NotImplementedError(
                        "only glTF triangle primitives are supported"
                    )
                attributes = primitive.get("attributes", {})
                if "POSITION" not in attributes:
                    raise ValueError("glTF primitive has no POSITION attribute")
                vertices = np.asarray(
                    _accessor(document, binary, attributes["POSITION"]),
                    dtype=np.float64,
                )
                homogeneous = np.concatenate(
                    (vertices, np.ones((len(vertices), 1))), axis=1
                )
                vertices = (homogeneous @ world.T)[:, :3]
                if "indices" in primitive:
                    indices = np.asarray(
                        _accessor(document, binary, primitive["indices"]),
                        dtype=np.int64,
                    ).reshape(-1)
                else:
                    indices = np.arange(len(vertices), dtype=np.int64)
                if len(indices) % 3:
                    raise ValueError(
                        "glTF triangle index count is not divisible by three"
                    )
                base = sum(len(item) for item in vertices_out)
                vertices_out.append(vertices.astype(np.float32))
                faces_out.append(indices.reshape(-1, 3) + base)
        for child in node.get("children", ()):
            visit(int(child), world)

    for root in roots:
        visit(int(root), np.eye(4, dtype=np.float64))
    if not vertices_out:
        raise ValueError("GLB scene contains no triangle mesh")
    return TriangleMesh(np.concatenate(vertices_out), np.concatenate(faces_out))


def load_obj(path: str | Path, *, groups: str | tuple[str, ...] | None = None):
    """Load triangulated OBJ geometry, optionally selecting named groups."""
    selected = (
        None if groups is None else {groups} if isinstance(groups, str) else set(groups)
    )
    if selected is not None and not selected:
        raise ValueError("OBJ group selection must not be empty")
    vertices, faces = [], []
    active_groups = set()
    for raw in (
        Path(path)
        .expanduser()
        .read_text(encoding="utf-8", errors="strict")
        .splitlines()
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "g":
            active_groups = set(fields[1:])
        elif (
            fields[0] == "f"
            and len(fields) >= 4
            and (selected is None or not selected.isdisjoint(active_groups))
        ):
            polygon = []
            for field in fields[1:]:
                index = int(field.split("/", 1)[0])
                polygon.append(index - 1 if index > 0 else len(vertices) + index)
            faces.extend(
                (polygon[0], polygon[i], polygon[i + 1])
                for i in range(1, len(polygon) - 1)
            )
    if selected is not None:
        if not faces:
            raise ValueError(f"OBJ contains no faces in groups {sorted(selected)}")
        used = np.unique(np.asarray(faces, dtype=np.int32))
        remap = np.full(len(vertices), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        vertices = np.asarray(vertices, dtype=np.float32)[used]
        faces = remap[np.asarray(faces, dtype=np.int32)]
    return TriangleMesh(vertices, faces)


def obj_group_vertex_centers(
    path: str | Path, groups: tuple[str, ...] | list[str]
) -> np.ndarray:
    """Return mean positions of the vertices referenced by named OBJ groups.

    This is useful for marker geometry embedded alongside a mesh. The OBJ is
    parsed once and the requested order is preserved.
    """
    requested = tuple(map(str, groups))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("OBJ group names must be nonempty and unique")
    wanted = set(requested)
    vertices = []
    referenced = {name: set() for name in requested}
    active_groups = set()
    for raw in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        fields = raw.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "g":
            active_groups = wanted.intersection(fields[1:])
        elif fields[0] == "f" and active_groups:
            indices = {
                (index - 1 if index > 0 else len(vertices) + index)
                for index in (int(field.split("/", 1)[0]) for field in fields[1:])
            }
            for name in active_groups:
                referenced[name].update(indices)
    missing = [name for name, indices in referenced.items() if not indices]
    if missing:
        raise ValueError(f"OBJ contains no faces in groups {missing}")
    points = np.asarray(vertices, dtype=np.float32)
    return np.asarray(
        [points[sorted(referenced[name])].mean(axis=0) for name in requested],
        dtype=np.float32,
    )


def load_triangle_mesh(path: str | Path, *, obj_groups=None):
    """Load a supported unrigged GLB or OBJ mesh."""
    path = Path(path).expanduser()
    if path.suffix.lower() == ".glb":
        return load_glb(path)
    if path.suffix.lower() == ".obj":
        return load_obj(path, groups=obj_groups)
    raise ValueError("unrigged mesh must be a .glb or .obj file")


__all__ = [
    "TriangleMesh",
    "load_glb",
    "load_obj",
    "load_triangle_mesh",
    "obj_group_vertex_centers",
]
