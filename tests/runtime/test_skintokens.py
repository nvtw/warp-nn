# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json
import struct

import numpy as np
import pytest

from warp_nn.runtime.formats.mesh import TriangleMesh, load_glb, load_obj
from warp_nn.runtime.skintokens.host import (
    SkinTokensTokenizer,
    normalize_geometry,
    prepare_geometry,
    sample_surface,
    transfer_skin_weights,
)


def _tiny_glb(path):
    positions = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype="<f4")
    indices = np.asarray((0, 1, 2), dtype="<u2")
    binary = positions.tobytes() + indices.tobytes()
    while len(binary) % 4:
        binary += b"\0"
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
            {
                "buffer": 0,
                "byteOffset": positions.nbytes,
                "byteLength": indices.nbytes,
            },
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "nodes": [{"mesh": 0, "translation": [2, 3, 4], "scale": [2, 2, 2]}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def test_dependency_light_mesh_loaders_apply_transform_and_triangulate(tmp_path):
    glb_path = tmp_path / "triangle.glb"
    _tiny_glb(glb_path)
    glb = load_glb(glb_path)
    np.testing.assert_allclose(glb.vertices, ((2, 3, 4), (4, 3, 4), (2, 5, 4)))
    np.testing.assert_array_equal(glb.faces, ((0, 1, 2),))

    obj_path = tmp_path / "quad.obj"
    obj_path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    obj = load_obj(obj_path)
    np.testing.assert_array_equal(obj.faces, ((0, 1, 2), (0, 2, 3)))


def test_official_normalization_and_surface_sampling_are_deterministic():
    mesh = TriangleMesh(
        np.asarray(((0, 0, 0), (2, 0, 0), (0, 1, 0)), dtype=np.float32),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    transform, inverse = normalize_geometry(mesh.vertices)
    normalized = np.c_[mesh.vertices, np.ones(3)] @ transform.T
    np.testing.assert_allclose(normalized[:, :3].min(axis=0), (-1, -0.5, 0))
    np.testing.assert_allclose(normalized[:, :3].max(axis=0), (1, 0.5, 0))
    np.testing.assert_allclose(transform @ inverse, np.eye(4), atol=1.0e-6)
    first, normals = sample_surface(mesh, 64, seed=7)
    second, _ = sample_surface(mesh, 64, seed=7)
    np.testing.assert_array_equal(first, second)
    assert np.all(first[:, 0] >= 0) and np.all(first[:, 1] >= 0)
    assert np.all(first[:, 0] / 2 + first[:, 1] <= 1.0 + 1.0e-6)
    np.testing.assert_allclose(normals, np.tile((0, 0, 1), (64, 1)))
    prepared = prepare_geometry(mesh, points=19, seed=3)
    assert prepared.sampled_vertices.shape == prepared.sampled_normals.shape == (19, 3)


def test_skin_weight_transfer_handles_exact_samples_and_normalizes():
    samples = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float32)
    weights = np.eye(3, dtype=np.float32)
    vertices = np.asarray(((0, 0, 0), (0.5, 0, 0)), dtype=np.float32)
    transferred = transfer_skin_weights(vertices, samples, weights, neighbors=2)
    np.testing.assert_array_equal(transferred[0], (1, 0, 0))
    np.testing.assert_allclose(transferred[1], (0.5, 0.5, 0), atol=1.0e-6)
    np.testing.assert_allclose(transferred.sum(axis=1), 1)


def test_official_skeleton_tokens_round_trip_and_skin_phase():
    tokenizer = SkinTokensTokenizer()
    joints = np.asarray(((0, 0, 0), (0, 0.5, 0), (-0.5, 0, 0)), dtype=np.float32)
    parents = np.asarray((-1, 0, 0), dtype=np.int32)
    tokens = tokenizer.tokenize_skeleton(joints, parents)
    np.testing.assert_array_equal(
        tokens,
        (257, 266, 128, 128, 128, 128, 192, 128, 256, 128, 128, 128, 64, 128, 128, 258),
    )
    decoded = tokenizer.decode_skeleton(tokens)
    np.testing.assert_array_equal(decoded.parents, parents)
    assert decoded.cls == "articulation"
    assert np.max(np.abs(decoded.joints - joints)) <= 1 / 256 + 1.0e-7
    allowed = tokenizer.allowed_tokens(tokens)
    assert allowed[0] == 267 and allowed[-1] == 33034 and len(allowed) == 32768
    complete = np.concatenate((tokens, np.full(12, 267, dtype=np.int64)))
    np.testing.assert_array_equal(tokenizer.allowed_tokens(complete), (33035,))
    with pytest.raises(ValueError, match="switch"):
        tokenizer.decode_skeleton(tokens[:-1])
