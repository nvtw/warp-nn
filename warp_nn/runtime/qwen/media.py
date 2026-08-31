# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free image preprocessing shared by vision model runtimes."""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VisionInput:
    """Patchified media and its unmerged temporal/spatial token grid."""

    patches: np.ndarray
    grid_thw: tuple[int, int, int]

    @property
    def feature_count(self) -> int:
        t, h, w = self.grid_thw
        return t * h * w // 4


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = 32,
    minimum_pixels: int = 65_536,
    maximum_pixels: int = 16_777_216,
) -> tuple[int, int]:
    """Match Qwen's aspect-preserving, patch-grid-aligned resize policy."""
    if height <= 0 or width <= 0 or factor <= 0:
        raise ValueError("image dimensions and resize factor must be positive")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("image aspect ratio must not exceed 200")
    if minimum_pixels <= 0 or maximum_pixels < minimum_pixels:
        raise ValueError("invalid image pixel bounds")
    resized_h = max(factor, round(height / factor) * factor)
    resized_w = max(factor, round(width / factor) * factor)
    pixels = resized_h * resized_w
    if pixels > maximum_pixels:
        scale = math.sqrt(height * width / maximum_pixels)
        resized_h = max(factor, math.floor(height / scale / factor) * factor)
        resized_w = max(factor, math.floor(width / scale / factor) * factor)
    elif pixels < minimum_pixels:
        scale = math.sqrt(minimum_pixels / (height * width))
        resized_h = math.ceil(height * scale / factor) * factor
        resized_w = math.ceil(width * scale / factor) * factor
    return resized_h, resized_w


def _cubic(x: np.ndarray) -> np.ndarray:
    """PyTorch bicubic kernel (a=-0.75)."""
    absolute = np.abs(x)
    first = ((-0.75 + 2.0) * absolute - (-0.75 + 3.0)) * absolute * absolute + 1.0
    second = (
        (-0.75 * absolute - 5.0 * -0.75) * absolute + 8.0 * -0.75
    ) * absolute - 4.0 * -0.75
    return np.where(absolute <= 1.0, first, np.where(absolute < 2.0, second, 0.0))


def resize_bicubic(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an HWC RGB image with align_corners=False bicubic sampling."""
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if height <= 0 or width <= 0:
        raise ValueError("resize dimensions must be positive")
    source = source.astype(np.float32, copy=False)
    in_h, in_w = source.shape[:2]
    ys = (np.arange(height, dtype=np.float32) + 0.5) * in_h / height - 0.5
    xs = (np.arange(width, dtype=np.float32) + 0.5) * in_w / width - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    yi = y0[:, None] + np.arange(-1, 3, dtype=np.int64)[None, :]
    xi = x0[:, None] + np.arange(-1, 3, dtype=np.int64)[None, :]
    yw = _cubic(ys[:, None] - yi)
    xw = _cubic(xs[:, None] - xi)
    yw /= yw.sum(axis=1, keepdims=True)
    xw /= xw.sum(axis=1, keepdims=True)
    yi = np.clip(yi, 0, in_h - 1)
    xi = np.clip(xi, 0, in_w - 1)
    rows = np.sum(source[yi] * yw[:, :, None, None], axis=1)
    return np.sum(rows[:, xi] * xw[None, :, :, None], axis=2).astype(np.float32)


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
                or color not in (2, 6)
                or compression
                or filtering
                or interlace
            ):
                raise ValueError("only non-interlaced 8-bit RGB/RGBA PNG is supported")
        elif kind == b"IDAT":
            payload.append(chunk)
        elif kind == b"IEND":
            break
        offset += 12 + length
    if width is None or not payload:
        raise ValueError(f"PNG '{path}' has no image data")
    channels = 3 if color == 2 else 4
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
    return rows.reshape(height, width, channels)[..., :3].copy()


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


def preprocess_qwen_media(
    frames: np.ndarray | Sequence[np.ndarray],
    *,
    minimum_pixels: int = 65_536,
    maximum_pixels: int = 16_777_216,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> VisionInput:
    """Resize, normalize, and patchify image/video frames for Qwen vision."""
    array = np.asarray(frames)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] != 3 or array.shape[0] == 0:
        raise ValueError("media must have shape (frames, height, width, 3)")
    if array.dtype.kind not in "uif":
        raise TypeError("media pixels must be numeric")
    factor = patch_size * merge_size
    target_h, target_w = smart_resize(
        array.shape[1],
        array.shape[2],
        factor=factor,
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
    )
    resized = np.stack(
        [resize_bicubic(frame, target_h, target_w) for frame in array], axis=0
    )
    if resized.max(initial=0.0) <= 1.0:
        resized *= 255.0
    resized = resized / 127.5 - 1.0
    remainder = (-len(resized)) % temporal_patch_size
    if remainder:
        resized = np.concatenate(
            [resized, np.repeat(resized[-1:], remainder, axis=0)], axis=0
        )
    grid_t = len(resized) // temporal_patch_size
    grid_h = target_h // patch_size
    grid_w = target_w // patch_size
    patches = resized.transpose(0, 3, 1, 2).reshape(
        grid_t,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    patches = np.ascontiguousarray(
        patches.reshape(grid_t * grid_h * grid_w, -1), dtype=np.float32
    )
    return VisionInput(patches, (grid_t, grid_h, grid_w))


def qwen_vision_positions(grid_thw: tuple[int, int, int]) -> np.ndarray:
    """Return block-major temporal/height/width indices after spatial merge."""
    t, h, w = grid_thw
    if t <= 0 or h <= 0 or w <= 0 or h % 2 or w % 2:
        raise ValueError("vision grid must be positive with even height and width")
    height = np.arange(h, dtype=np.int64).reshape(h // 2, 2, 1, 1)
    height = np.broadcast_to(height, (h // 2, 2, w // 2, 2)).transpose(0, 2, 1, 3)
    width = np.arange(w, dtype=np.int64).reshape(1, 1, w // 2, 2)
    width = np.broadcast_to(width, (h // 2, 2, w // 2, 2)).transpose(0, 2, 1, 3)
    spatial_h = height.reshape(-1)
    spatial_w = width.reshape(-1)
    return np.stack(
        [
            np.repeat(np.arange(t, dtype=np.int64), h * w),
            np.tile(spatial_h, t),
            np.tile(spatial_w, t),
        ],
        axis=1,
    )
