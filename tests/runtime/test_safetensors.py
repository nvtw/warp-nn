# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import json
import struct
from pathlib import Path

import numpy as np
import warp as wp

from warp_nn.runtime.safetensors import SafeTensorArchive


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header = {}
    data = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        begin = len(data)
        data.extend(payload)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, len(data)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def test_safetensor_archive_loads_single_file(tmp_path):
    values = np.array([[1.5, -2.0], [3.25, 4.0]], dtype=np.float32)
    _write_safetensors(tmp_path / "model.safetensors", {"weight": ("F32", values.shape, values.tobytes())})

    archive = SafeTensorArchive(tmp_path)
    loaded = archive.load("cpu")

    assert archive.names == ("weight",)
    assert archive.metadata("weight").dtype == wp.float32
    np.testing.assert_array_equal(loaded["weight"].numpy(), values)


def test_safetensor_archive_loads_shards_and_bfloat16_bits(tmp_path):
    bits = np.array([0x3F80, 0xC000], dtype=np.uint16)
    _write_safetensors(tmp_path / "part.safetensors", {"bf16": ("BF16", bits.shape, bits.tobytes())})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"bf16": "part.safetensors"}}), encoding="utf-8"
    )

    loaded = SafeTensorArchive(tmp_path).load("cpu")["bf16"]

    np.testing.assert_array_equal(loaded.numpy().view(np.uint16), bits)


def test_safetensor_archive_rejects_invalid_range(tmp_path):
    _write_safetensors(tmp_path / "model.safetensors", {"bad": ("F32", (2,), b"\0" * 4)})

    with pytest.raises(ValueError, match="Invalid data range"):
        SafeTensorArchive(tmp_path)
