# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free image file input and output."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distances = (
        abs(estimate - left),
        abs(estimate - above),
        abs(estimate - upper_left),
    )
    return (left, above, upper_left)[int(np.argmin(distances))]


def _load_png(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"'{path}' is not a PNG image")
    offset = 8
    payload = []
    palette = None
    width = height = color = None
    while offset + 12 <= len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        if offset + 12 + length > len(data):
            raise ValueError(f"PNG '{path}' is truncated")
        if kind == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", chunk)
            )
            if (
                depth != 8
                or color not in (0, 2, 3, 4, 6)
                or compression
                or filtering
                or interlace
            ):
                raise ValueError("only non-interlaced 8-bit PNG images are supported")
        elif kind == b"PLTE":
            if not chunk or len(chunk) % 3 or len(chunk) > 256 * 3:
                raise ValueError(f"PNG {path!r} has an invalid palette")
            palette = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, 3).copy()
        elif kind == b"IDAT":
            payload.append(chunk)
        elif kind == b"IEND":
            break
        offset += 12 + length
    if width is None or not payload:
        raise ValueError(f"PNG '{path}' has no image data")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    stride = width * channels
    raw = zlib.decompress(b"".join(payload))
    if len(raw) != height * (stride + 1):
        raise ValueError(f"PNG '{path}' has an unexpected data size")
    rows = np.empty((height, stride), dtype=np.uint8)
    cursor = 0
    previous = np.zeros(stride, dtype=np.uint8)
    for row in range(height):
        filter_type = raw[cursor]
        scan = np.frombuffer(
            raw, dtype=np.uint8, count=stride, offset=cursor + 1
        ).copy()
        cursor += stride + 1
        for index in range(stride):
            left = int(scan[index - channels]) if index >= channels else 0
            above = int(previous[index])
            upper = int(previous[index - channels]) if index >= channels else 0
            if filter_type == 1:
                scan[index] = (int(scan[index]) + left) & 255
            elif filter_type == 2:
                scan[index] = (int(scan[index]) + above) & 255
            elif filter_type == 3:
                scan[index] = (int(scan[index]) + ((left + above) // 2)) & 255
            elif filter_type == 4:
                scan[index] = (int(scan[index]) + _paeth(left, above, upper)) & 255
            elif filter_type != 0:
                raise ValueError(f"PNG '{path}' uses invalid filter {filter_type}")
        rows[row] = scan
        previous = scan
    pixels = rows.reshape(height, width, channels)
    if color == 3:
        if palette is None:
            raise ValueError(f"indexed PNG {path!r} has no palette")
        indices = pixels[..., 0]
        if int(indices.max(initial=0)) >= len(palette):
            raise ValueError(f"indexed PNG {path!r} references a missing color")
        return palette[indices].copy()
    if color in (0, 4):
        return np.repeat(pixels[..., :1], 3, axis=2)
    return pixels[..., :3].copy()


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load RGB from dependency-free PNG, PPM, NPY, or NPZ input."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".png":
        return _load_png(path)
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError("NPZ image must contain exactly one array")
            return np.asarray(archive[archive.files[0]])
    if suffix in (".ppm", ".pnm"):
        data = path.read_bytes()
        header, payload = data.split(b"\n", 1)
        if header.strip() != b"P6":
            raise ValueError("only binary P6 PPM is supported")
        tokens = []
        while len(tokens) < 3:
            line, payload = payload.split(b"\n", 1)
            if not line.startswith(b"#"):
                tokens.extend(line.split())
        width, height, maximum = map(int, tokens[:3])
        if maximum != 255 or len(payload) != width * height * 3:
            raise ValueError("PPM must use 8-bit RGB samples")
        return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
    raise ValueError("dependency-free image input supports PNG, PPM, NPY, and NPZ")


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
