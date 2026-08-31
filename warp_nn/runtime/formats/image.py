# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free image file output."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def write_png_rgb8(path: str | Path, image) -> None:
    """Write one HWC uint8 RGB or RGBA image as a non-interlaced PNG."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4) or image.dtype != np.uint8:
        raise TypeError("PNG output requires an HWC uint8 RGB or RGBA array")
    height, width, channels = image.shape
    if min(height, width) <= 0:
        raise ValueError("PNG dimensions must be positive")
    image = np.ascontiguousarray(image)
    scanlines = b"".join(b"\0" + image[row].tobytes() for row in range(height))
    header = struct.pack(
        ">IIBBBBB",
        width,
        height,
        8,
        2 if channels == 3 else 6,
        0,
        0,
        0,
    )
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )
