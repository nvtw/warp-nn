# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import struct
import zlib

import numpy as np
import pytest

from warp_nn.runtime.formats.image import write_png_rgb8


def test_write_png_rgb8_roundtrip_payload(tmp_path):
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    path = tmp_path / "image.png"
    write_png_rgb8(path, image)
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    chunks = {}
    while offset < len(data):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        chunks.setdefault(kind, b"")
        chunks[kind] += data[offset + 8 : offset + 8 + size]
        offset += 12 + size
    width, height, depth, color, _, _, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"]
    )
    assert (width, height, depth, color, interlace) == (4, 3, 8, 2, 0)
    raw = zlib.decompress(chunks[b"IDAT"])
    decoded = np.frombuffer(
        b"".join(raw[row * 13 + 1 : (row + 1) * 13] for row in range(3)),
        dtype=np.uint8,
    ).reshape(image.shape)
    np.testing.assert_array_equal(decoded, image)


def test_write_png_rgb8_rejects_implicit_conversion(tmp_path):
    with pytest.raises(TypeError, match="uint8"):
        write_png_rgb8(tmp_path / "bad.png", np.zeros((2, 2, 3), dtype=np.float32))
