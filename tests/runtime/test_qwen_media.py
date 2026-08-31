# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import struct
import zlib

import numpy as np

from warp_nn.runtime.qwen.media import load_rgb_image


def _chunk(kind, payload):
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def test_load_indexed_8bit_png(tmp_path):
    path = tmp_path / "indexed.png"
    header = struct.pack(">IIBBBBB", 3, 2, 8, 3, 0, 0, 0)
    palette = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255))
    scanlines = b"\0\0\1\2\0\2\1\0"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"PLTE", palette)
        + _chunk(b"IDAT", zlib.compress(scanlines))
        + _chunk(b"IEND", b"")
    )
    image = load_rgb_image(path)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(
        image,
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[0, 0, 255], [0, 255, 0], [255, 0, 0]],
        ],
    )
