# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free C-RADIO image encoder used by Nemotron Omni.

This module implements the still-image slice of the official model: dynamic
resolution preprocessing, C-RADIOv4-H, the v2 2x pixel unshuffle, and ``mlp1``.
It deliberately contains no video, audio, Transformers, Torch, or torchvision
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import warp as wp

from ..formats.safetensors import SafeTensorArchive
from ..operators import (
    AttentionHeadsPlan,
    AttentionMergePlan,
    BidirectionalGQAPlan,
    LayerNormPlan,
    Operation,
    RMSNormPlan,
    execute_operations,
    plan_linear,
)


_PREFIX = "vision_model.radio_model.model."
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class NemotronImage:
    """One normalized, channels-first image and its exact token geometry."""

    pixels: np.ndarray
    patch_grid: tuple[int, int]

    @property
    def tokens(self) -> int:
        return self.patch_grid[0] * self.patch_grid[1] // 4


def target_patch_grid(
    height: int,
    width: int,
    *,
    patch_size: int = 16,
    min_patches: int = 1024,
    max_patches: int = 13312,
    max_model_length: int = 16384,
) -> tuple[int, int]:
    """Return ``(height, width)`` in patches, matching the official tiler."""
    if height <= 0 or width <= 0 or patch_size <= 0:
        raise ValueError("image dimensions and patch size must be positive")
    available = max((max_model_length - 4) * 4, min_patches)
    available = min(available, max_patches) if max_patches > 0 else available
    # Python round is intentional: this is the expression in the official code.
    closest_h = round(height / patch_size + 0.5)
    closest_w = round(width / patch_size + 0.5)
    factor = min(math.sqrt(available / (closest_h * closest_w)), 1.0)
    target_h = max(1, math.floor(factor * closest_h))
    target_w = max(1, math.floor(factor * closest_w))
    if available > min_patches and target_h * target_w < min_patches:
        factor = math.sqrt(min_patches / (target_h * target_w))
        target_h = math.ceil(factor * target_h)
        target_w = math.ceil(factor * target_w)
    for axis in (0, 1):
        current, other = (target_h, target_w) if axis == 0 else (target_w, target_h)
        remainder = current % 2
        if remainder:
            current = (
                current + 1
                if (current + 1) * other <= available
                else max(2, current - 1)
            )
        if axis == 0:
            target_h = current
        else:
            target_w = current
    return target_h, target_w


def _cubic(value: np.ndarray) -> np.ndarray:
    """PyTorch bicubic convolution kernel (a=-0.75)."""
    x = np.abs(value)
    first = ((-0.75 + 2.0) * x - (-0.75 + 3.0)) * x * x + 1.0
    second = ((-0.75 * x + 5.0 * -0.75) * x - 8.0 * -0.75) * x + 4.0 * -0.75
    return np.where(x <= 1.0, first, np.where(x < 2.0, second, 0.0))


def _resize_weights(source: int, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Build antialiased bicubic indices/weights with half-pixel centers."""
    scale = source / target
    filter_scale = max(scale, 1.0)
    support = 2.0 * filter_scale
    taps = int(math.ceil(support) * 2 + 1)
    centers = (np.arange(target, dtype=np.float64) + 0.5) * scale - 0.5
    starts = np.ceil(centers - support).astype(np.int64)
    offsets = np.arange(taps, dtype=np.int64)
    raw = starts[:, None] + offsets[None, :]
    weights = _cubic((raw - centers[:, None]) / filter_scale)
    weights /= weights.sum(axis=1, keepdims=True)
    # Border coordinates are clamped by interpolate's edge padding. Duplicate
    # clamped taps are harmless and preserve their independently computed weights.
    return np.clip(raw, 0, source - 1), weights.astype(np.float32)


def _bicubic_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Channels-last antialiased bicubic resize, accumulated in FP32."""
    if image.shape[:2] == (height, width):
        return image.astype(np.float32, copy=False)
    yi, yw = _resize_weights(image.shape[0], height)
    xi, xw = _resize_weights(image.shape[1], width)
    source = image.astype(np.float32, copy=False)
    vertical = np.sum(source[yi] * yw[:, :, None, None], axis=1, dtype=np.float32)
    return np.sum(vertical[:, xi] * xw[None, :, :, None], axis=2, dtype=np.float32)


def preprocess_nemotron_image(
    image,
    *,
    patch_size: int = 16,
    min_patches: int = 1024,
    max_patches: int = 13312,
    max_model_length: int = 16384,
    mean: Sequence[float] = _MEAN,
    std: Sequence[float] = _STD,
    patch_grid: tuple[int, int] | None = None,
) -> NemotronImage:
    """Resize and normalize an RGB image exactly as Nemotron Omni expects."""
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("Nemotron images must be HxW RGB or RGBA arrays")
    pixels = pixels[..., :3]
    grid = patch_grid
    if grid is None:
        grid = target_patch_grid(
            pixels.shape[0],
            pixels.shape[1],
            patch_size=patch_size,
            min_patches=min_patches,
            max_patches=max_patches,
            max_model_length=max_model_length,
        )
    elif len(grid) != 2 or min(grid) <= 0 or grid[0] % 2 or grid[1] % 2:
        raise ValueError("explicit patch grid must contain two positive even values")
    resized = _bicubic_resize(pixels, grid[0] * patch_size, grid[1] * patch_size)
    normalized = (
        resized / np.float32(255.0) - np.asarray(mean, np.float32)
    ) / np.asarray(std, np.float32)
    return NemotronImage(
        np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32), grid
    )


def pixel_unshuffle_v2(values: np.ndarray) -> np.ndarray:
    """Reference implementation of the official ``pixel_shuffle(..., 0.5)``."""
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[0] % 2 or values.shape[1] % 2:
        raise ValueError("pixel unshuffle requires an even HxWxC array")
    h, w, channels = values.shape
    return (
        values.reshape(h // 2, 2, w // 2, 2, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(h // 2, w // 2, channels * 4)
    )


def vision_weight_names(depth: int, *, include_video: bool = False) -> list[str]:
    names = [
        _PREFIX + "patch_generator.embedder.weight",
        _PREFIX + "patch_generator.pos_embed",
        _PREFIX + "patch_generator.cls_token.token",
        "mlp1.0.weight",
        "mlp1.1.weight",
        "mlp1.3.weight",
    ]
    if include_video:
        names.append(_PREFIX + "patch_generator.video_embedder.weight")
    suffixes = (
        "norm1.weight",
        "norm1.bias",
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.proj.weight",
        "attn.proj.bias",
        "norm2.weight",
        "norm2.bias",
        "mlp.fc1.weight",
        "mlp.fc1.bias",
        "mlp.fc2.weight",
        "mlp.fc2.bias",
    )
    for index in range(depth):
        names.extend(_PREFIX + f"blocks.{index}.{suffix}" for suffix in suffixes)
    return names


@lru_cache(maxsize=None)
def _kernels(dtype: type, heads: int, head_size: int):
    DTYPE, HEADS, HEAD_SIZE = dtype, heads, head_size

    @wp.kernel(enable_backward=False, module="unique")
    def patchify(
        pixels: wp.array3d[wp.float32],
        patches: wp.array2d(dtype=DTYPE),
        patch_size: int,
    ):
        row, column = wp.tid()
        grid_w = pixels.shape[2] / patch_size
        py, px = row / grid_w, row % grid_w
        channel = column / (patch_size * patch_size)
        offset = column % (patch_size * patch_size)
        yy, xx = offset / patch_size, offset % patch_size
        patches[row, column] = DTYPE(
            pixels[channel, py * patch_size + yy, px * patch_size + xx]
        )

    @wp.kernel(enable_backward=False, module="unique")
    def position_prefix(
        projected: wp.array2d(dtype=DTYPE),
        position: wp.array2d(dtype=DTYPE),
        prefix: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        grid_h: int,
        grid_w: int,
    ):
        row, column = wp.tid()
        prefix_rows = prefix.shape[0]
        if row < prefix_rows:
            output[row, column] = prefix[row, column]
        else:
            patch = row - prefix_rows
            py, px = patch / grid_w, patch % grid_w
            max_dim = wp.max(grid_h, grid_w)
            # F.interpolate(..., bilinear, align_corners=False), followed by a
            # top-left crop when the requested aspect ratio is non-square.
            sy = (wp.float32(py) + 0.5) * wp.float32(128) / wp.float32(max_dim) - 0.5
            sx = (wp.float32(px) + 0.5) * wp.float32(128) / wp.float32(max_dim) - 0.5
            y0, x0 = wp.int32(wp.floor(sy)), wp.int32(wp.floor(sx))
            fy, fx = sy - wp.float32(y0), sx - wp.float32(x0)
            y0c, x0c = wp.clamp(y0, 0, 127), wp.clamp(x0, 0, 127)
            y1c, x1c = wp.clamp(y0 + 1, 0, 127), wp.clamp(x0 + 1, 0, 127)
            pos = (
                wp.float32(position[y0c * 128 + x0c, column]) * (1.0 - fy) * (1.0 - fx)
                + wp.float32(position[y0c * 128 + x1c, column]) * (1.0 - fy) * fx
                + wp.float32(position[y1c * 128 + x0c, column]) * fy * (1.0 - fx)
                + wp.float32(position[y1c * 128 + x1c, column]) * fy * fx
            )
            output[row, column] = DTYPE(wp.float32(projected[patch, column]) + pos)

    @wp.kernel(enable_backward=False, module="unique")
    def affine(
        x: wp.array2d(dtype=DTYPE),
        scale: wp.array1d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        x[row, column] = DTYPE(
            wp.float32(x[row, column]) * wp.float32(scale[column])
            + wp.float32(bias[column])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def split_qkv(
        x: wp.array2d(dtype=DTYPE),
        q: wp.array3d(dtype=DTYPE),
        k: wp.array3d(dtype=DTYPE),
        v: wp.array3d(dtype=DTYPE),
    ):
        token, head, column = wp.tid()
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        offset = head * HEAD_SIZE + column
        q[token, head, column] = x[token, offset]
        k[token, head, column] = x[token, HEADS * HEAD_SIZE + offset]
        v[token, head, column] = x[token, 2 * HEADS * HEAD_SIZE + offset]

    @wp.kernel(enable_backward=False, module="unique")
    def bias_residual(
        x: wp.array2d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        residual: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        output[row, column] = DTYPE(
            wp.float32(x[row, column])
            + wp.float32(bias[column])
            + wp.float32(residual[row, column])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def bias_gelu(x: wp.array2d(dtype=DTYPE), bias: wp.array1d(dtype=DTYPE)):
        row, column = wp.tid()
        value = wp.float32(x[row, column]) + wp.float32(bias[column])
        x[row, column] = DTYPE(value * 0.5 * (1.0 + wp.erf(value * 0.7071067811865476)))

    @wp.kernel(enable_backward=False, module="unique")
    def unshuffle(
        x: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        grid_w: int,
        prefix_rows: int,
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        row, column = wp.tid()
        out_w = grid_w / 2
        oy, ox = row / out_w, row % out_w
        source_offset, channel = column / x.shape[1], column % x.shape[1]
        sy, sx = source_offset / 2, source_offset % 2
        source_row = prefix_rows + (oy * 2 + sy) * grid_w + ox * 2 + sx
        output[row, column] = x[source_row, channel]

    @wp.kernel(enable_backward=False, module="unique")
    def square_relu(x: wp.array2d(dtype=DTYPE)):
        row, column = wp.tid()
        value = wp.max(wp.float32(x[row, column]), 0.0)
        x[row, column] = DTYPE(value * value)

    return (
        patchify,
        position_prefix,
        affine,
        split_qkv,
        bias_residual,
        bias_gelu,
        unshuffle,
        square_relu,
    )


class _Linear:
    def __init__(self, x, weight, cublas=None):
        self.tensors = {"x": x, "weight": weight}
        self.shapes = {name: value.shape for name, value in self.tensors.items()}
        self.op = Operation("Linear", ["x", "weight"], ["output"])
        plan_linear(self.op, self.tensors, self.shapes, x.device, cublas=cublas)
        self.output = self.tensors["output"]

    def execute(self):
        execute_operations((self.op,), self.tensors, self.shapes, self.output.device)
        return self.output


class _Block:
    def __init__(self, x, weights, prefix, heads, epsilon, cublas):
        self.x = x
        self.weights, self.prefix, self.epsilon = weights, prefix, epsilon
        self.norm1 = LayerNormPlan(x, epsilon=epsilon)
        self.qkv = _Linear(
            self.norm1.output, weights[prefix + "attn.qkv.weight"], cublas
        )
        rows, hidden = x.shape
        head_size = hidden // heads
        self.q = wp.empty((rows, heads, head_size), dtype=x.dtype, device=x.device)
        self.k = wp.empty_like(self.q)
        self.v = wp.empty_like(self.q)
        self.q_heads = AttentionHeadsPlan(self.q.reshape((1, rows, hidden)), heads)
        self.k_heads = AttentionHeadsPlan(self.k.reshape((1, rows, hidden)), heads)
        self.v_heads = AttentionHeadsPlan(self.v.reshape((1, rows, hidden)), heads)
        self.attention = BidirectionalGQAPlan(
            self.q_heads.output, self.k_heads.output, self.v_heads.output
        )
        self.merge = AttentionMergePlan(self.attention.output)
        self.projection = _Linear(
            self.merge.output.reshape((rows, hidden)),
            weights[prefix + "attn.proj.weight"],
            cublas,
        )
        self.after_attention = wp.empty_like(x)
        self.norm2 = LayerNormPlan(self.after_attention, epsilon=epsilon)
        self.fc1 = _Linear(
            self.norm2.output, weights[prefix + "mlp.fc1.weight"], cublas
        )
        self.fc2 = _Linear(self.fc1.output, weights[prefix + "mlp.fc2.weight"], cublas)
        self.output = wp.empty_like(x)

    def execute(self, kernels):
        _, _, affine, split_qkv, bias_residual, bias_gelu, _, _ = kernels
        p, w = self.prefix, self.weights
        self.norm1.execute()
        wp.launch(
            affine,
            dim=self.norm1.output.shape,
            inputs=[self.norm1.output, w[p + "norm1.weight"], w[p + "norm1.bias"]],
            device=self.x.device,
        )
        self.qkv.execute()
        wp.launch(
            split_qkv,
            dim=self.q.shape,
            inputs=[self.qkv.output, self.q, self.k, self.v],
            device=self.x.device,
        )
        self.q_heads.execute()
        self.k_heads.execute()
        self.v_heads.execute()
        self.attention.execute()
        self.merge.execute()
        self.projection.execute()
        wp.launch(
            bias_residual,
            dim=self.x.shape,
            inputs=[
                self.projection.output,
                w[p + "attn.proj.bias"],
                self.x,
                self.after_attention,
            ],
            device=self.x.device,
        )
        self.norm2.execute()
        wp.launch(
            affine,
            dim=self.norm2.output.shape,
            inputs=[self.norm2.output, w[p + "norm2.weight"], w[p + "norm2.bias"]],
            device=self.x.device,
        )
        self.fc1.execute()
        wp.launch(
            bias_gelu,
            dim=self.fc1.output.shape,
            inputs=[self.fc1.output, w[p + "mlp.fc1.bias"]],
            device=self.x.device,
        )
        self.fc2.execute()
        wp.launch(
            bias_residual,
            dim=self.output.shape,
            inputs=[
                self.fc2.output,
                w[p + "mlp.fc2.bias"],
                self.after_attention,
                self.output,
            ],
            device=self.x.device,
        )
        return self.output


class _VisionPlan:
    def __init__(
        self,
        encoder: "NemotronVisionEncoder",
        grid: tuple[int, int],
        temporal_patch_size: int = 1,
    ):
        self.encoder, self.grid = encoder, grid
        self.temporal_patch_size = temporal_patch_size
        h, w = grid
        rows = h * w
        hidden = encoder.hidden_size
        self.pixels = wp.empty(
            (
                3 * temporal_patch_size,
                h * encoder.patch_size,
                w * encoder.patch_size,
            ),
            dtype=wp.float32,
            device=encoder.device,
        )
        self.patches = wp.empty(
            (rows, 3 * temporal_patch_size * encoder.patch_size**2),
            dtype=encoder.dtype,
            device=encoder.device,
        )
        self.patch_projection = _Linear(
            self.patches,
            encoder.weights[
                _PREFIX
                + "patch_generator."
                + (
                    "embedder.weight"
                    if temporal_patch_size == 1
                    else "video_embedder.weight"
                )
            ],
            encoder.cublas,
        )
        self.hidden = wp.empty(
            (rows + encoder.prefix_tokens, hidden),
            dtype=encoder.dtype,
            device=encoder.device,
        )
        self.blocks = []
        current = self.hidden
        for index in range(encoder.depth):
            block = _Block(
                current,
                encoder.weights,
                _PREFIX + f"blocks.{index}.",
                encoder.heads,
                encoder.epsilon,
                encoder.cublas,
            )
            self.blocks.append(block)
            current = block.output
        self.features = current
        self.unshuffled = wp.empty(
            (rows // 4, hidden * 4), dtype=encoder.dtype, device=encoder.device
        )
        self.projector_norm = RMSNormPlan(
            self.unshuffled, encoder.weights["mlp1.0.weight"], epsilon=1.0e-5
        )
        self.projector_fc1 = _Linear(
            self.projector_norm.output, encoder.weights["mlp1.1.weight"], encoder.cublas
        )
        self.projector_fc2 = _Linear(
            self.projector_fc1.output, encoder.weights["mlp1.3.weight"], encoder.cublas
        )
        self.output = self.projector_fc2.output
        self.graph = None
        self._capture_ready = False

    def execute(self):
        kernels = self.encoder.kernels
        patchify, position_prefix, _, _, _, _, unshuffle, square_relu = kernels
        wp.launch(
            patchify,
            dim=self.patches.shape,
            inputs=[self.pixels, self.patches, self.encoder.patch_size],
            device=self.encoder.device,
        )
        self.patch_projection.execute()
        wp.launch(
            position_prefix,
            dim=self.hidden.shape,
            inputs=[
                self.patch_projection.output,
                self.encoder.weights[_PREFIX + "patch_generator.pos_embed"].reshape(
                    (16384, self.encoder.hidden_size)
                ),
                self.encoder.weights[_PREFIX + "patch_generator.cls_token.token"],
                self.hidden,
                self.grid[0],
                self.grid[1],
            ],
            device=self.encoder.device,
        )
        for block in self.blocks:
            block.execute(kernels)
        wp.launch(
            unshuffle,
            dim=self.unshuffled.shape,
            inputs=[
                self.features,
                self.unshuffled,
                self.grid[1],
                self.encoder.prefix_tokens,
            ],
            device=self.encoder.device,
        )
        self.projector_norm.execute()
        self.projector_fc1.execute()
        wp.launch(
            square_relu,
            dim=self.projector_fc1.output.shape,
            inputs=[self.projector_fc1.output],
            device=self.encoder.device,
        )
        self.projector_fc2.execute()
        return self.output

    def run(self):
        if not self.encoder.device.is_cuda:
            return self.execute()
        if not self._capture_ready:
            self._capture_ready = True
            return self.execute()
        if self.graph is None:
            wp.capture_begin(device=self.encoder.device)
            output = self.execute()
            self.graph = wp.capture_end(device=self.encoder.device)
            self.output = output
        wp.capture_launch(self.graph)
        return self.output


class NemotronVisionEncoder:
    """Lazy C-RADIOv4-H image encoder backed by an Omni safetensors model."""

    def __init__(self, path: str | Path, *, device=None, cublas=None):
        self.path = Path(path)
        document = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        vision = document["vision_config"]
        args = vision["args"]
        if (
            args.get("model") != "vit_huge_patch16_224"
            or document.get("ps_version") != "v2"
        ):
            raise ValueError(
                "Nemotron vision encoder requires C-RADIOv4-H with pixel shuffle v2"
            )
        self.patch_size = int(document["patch_size"])
        self.video_temporal_patch_size = int(
            document.get("video_temporal_patch_size", 2)
        )
        self.video_target_patches = int(vision.get("video_target_num_patches", 1024))
        self.depth, self.hidden_size, self.heads = (
            32,
            int(document["vit_hidden_size"]),
            16,
        )
        self.intermediate_size = 5120
        self.prefix_tokens = int(args.get("register_multiple", 10))
        self.output_size = int(document["llm_config"]["hidden_size"])
        self.epsilon = 1.0e-6
        self.min_patches = int(
            vision.get("min_num_patches", args.get("min_num_patches", 1024))
        )
        self.max_patches = int(
            vision.get("max_num_patches", args.get("max_num_patches", 13312))
        )
        self.device = wp.get_device(device)
        self.cublas = cublas
        archive = SafeTensorArchive(self.path)
        names = vision_weight_names(self.depth, include_video=True)
        missing = set(names) - set(archive.names)
        if missing:
            raise ValueError(
                f"Nemotron vision checkpoint is missing {sorted(missing)[:5]}"
            )
        self.dtype = archive.metadata(names[0]).dtype
        if self.dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Nemotron vision weights must be FP16 or BF16")
        self.weights = archive.load(self.device, names)
        if self.weights[_PREFIX + "patch_generator.pos_embed"].shape != (
            1,
            16384,
            self.hidden_size,
        ):
            raise ValueError("Nemotron C-RADIO position table must be 128x128")
        self.kernels = _kernels(self.dtype, self.heads, self.hidden_size // self.heads)
        self._plans: dict[tuple[tuple[int, int], int], _VisionPlan] = {}

    def preprocess(self, image) -> NemotronImage:
        return preprocess_nemotron_image(
            image,
            patch_size=self.patch_size,
            min_patches=self.min_patches,
            max_patches=self.max_patches,
        )

    def encode(self, image) -> wp.array:
        media = image if isinstance(image, NemotronImage) else self.preprocess(image)
        key = (media.patch_grid, 1)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._plans[key] = _VisionPlan(self, media.patch_grid)
        if media.pixels.shape != plan.pixels.shape:
            raise ValueError("preprocessed image does not match its patch grid")
        plan.pixels.assign(media.pixels)
        return plan.run()

    def encode_video(self, video) -> wp.array:
        """Encode temporally packed frames with the checkpoint video projection."""
        from .video import NemotronVideo

        if not isinstance(video, NemotronVideo):
            raise TypeError("encode_video expects a NemotronVideo")
        if video.temporal_patch_size != self.video_temporal_patch_size:
            raise ValueError("video temporal patch size does not match the checkpoint")
        grid = video.frames[0].patch_grid
        key = (grid, video.temporal_patch_size)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._plans[key] = _VisionPlan(self, grid, video.temporal_patch_size)
        output = wp.empty(
            (video.tokens, self.output_size),
            dtype=self.dtype,
            device=self.device,
        )
        for group in range(video.groups):
            begin = group * video.temporal_patch_size
            frames = list(video.frames[begin : begin + video.temporal_patch_size])
            while len(frames) < video.temporal_patch_size:
                frames.append(frames[-1])
            pixels = np.concatenate([frame.pixels for frame in frames], axis=0)
            plan.pixels.assign(pixels)
            group_output = plan.run()
            wp.copy(
                output.flatten(),
                group_output.flatten(),
                dest_offset=group * video.tokens_per_group * self.output_size,
                count=group_output.size,
            )
        return output
