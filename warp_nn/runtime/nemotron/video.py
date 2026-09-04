# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional video decoding and temporal prompt geometry for Nemotron Omni."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import warp as wp

from ..formats.media import VideoFrames, decode_video_frames
from ..kernels import _gather_rows_kernel
from .vision import NemotronImage, preprocess_nemotron_image


@dataclass(frozen=True)
class NemotronVideo:
    """Sampled, normalized frames and source timing for temporal C-RADIO."""

    frames: tuple[NemotronImage, ...]
    source_fps: float
    source_indices: tuple[int, ...]
    temporal_patch_size: int = 2

    def __post_init__(self):
        if not self.frames or len(self.frames) != len(self.source_indices):
            raise ValueError(
                "video frames and source indices must be non-empty and aligned"
            )
        if self.temporal_patch_size <= 0:
            raise ValueError("video temporal patch size must be positive")
        grid = self.frames[0].patch_grid
        if any(frame.patch_grid != grid for frame in self.frames):
            raise ValueError("all sampled video frames must share one spatial grid")

    @property
    def groups(self) -> int:
        return (
            len(self.frames) + self.temporal_patch_size - 1
        ) // self.temporal_patch_size

    @property
    def tokens_per_group(self) -> int:
        return self.frames[0].tokens

    @property
    def tokens(self) -> int:
        return self.groups * self.tokens_per_group

    def timestamp(self, index: int) -> float:
        frame_duration_ms = int(1000.0 / self.source_fps)
        return self.source_indices[index] * frame_duration_ms / 1000.0


def target_video_patch_grid(height: int, width: int, target: int = 1024):
    """Match the official aspect-preserving, pixel-unshuffle-aligned grid."""
    if height <= 0 or width <= 0 or target <= 0:
        raise ValueError("video dimensions and target patch count must be positive")
    aspect = width / height
    patch_h = max(round(np.sqrt(target / aspect)), 1)
    patch_w = max(round(np.sqrt(target * aspect)), 1)
    up_h, up_w = patch_h + (-patch_h % 2), patch_w + (-patch_w % 2)
    if up_h * up_w <= target:
        patch_h, patch_w = up_h, up_w
    else:
        patch_h, patch_w = max(2, patch_h - patch_h % 2), max(2, patch_w - patch_w % 2)
    return patch_h, patch_w


def preprocess_nemotron_video(
    source: str | Path | VideoFrames | Sequence[np.ndarray],
    *,
    fps: float = 1.0,
    max_frames: int = 128,
    temporal_patch_size: int = 2,
    target_patches: int = 1024,
) -> NemotronVideo:
    """Decode/sample a video and apply the shared C-RADIO image preprocessing."""
    if isinstance(source, (str, Path)):
        decoded = decode_video_frames(source, fps=fps, max_frames=max_frames)
    elif isinstance(source, VideoFrames):
        decoded = source
    else:
        raw = tuple(np.asarray(frame) for frame in source)
        if not raw:
            raise ValueError("video must contain at least one frame")
        decoded = VideoFrames(raw, float(fps), tuple(range(len(raw))))
    if target_patches <= 0:
        raise ValueError("video target patches must be positive")
    first = decoded.frames[0]
    grid = target_video_patch_grid(first.shape[0], first.shape[1], target_patches)
    frames = tuple(
        preprocess_nemotron_image(frame, patch_grid=grid) for frame in decoded.frames
    )
    return NemotronVideo(
        frames, decoded.source_fps, decoded.source_indices, temporal_patch_size
    )


def video_prompt_chunks(video: NemotronVideo) -> tuple[str, ...]:
    """Return the official timestamp labels and placeholder chunks per tubelet."""
    placeholder = "<img>" + "<image>" * video.tokens_per_group + "</img>"
    chunks = []
    for group in range(video.groups):
        labels = []
        begin = group * video.temporal_patch_size
        end = min(begin + video.temporal_patch_size, len(video.frames))
        for frame_index in range(begin, end):
            prefix = "Frame" if frame_index == begin else "frame"
            labels.append(
                f"{prefix} {frame_index + 1} sampled at "
                f"{video.timestamp(frame_index):.2f} seconds"
            )
        chunks.append(" and ".join(labels) + ": " + placeholder)
    return tuple(chunks)


@lru_cache(maxsize=None)
def _video_dissimilarity_kernel(dtype, hidden_size):
    DTYPE, HIDDEN_SIZE = dtype, hidden_size

    @wp.func
    def multiply(left: DTYPE, right: DTYPE):
        return wp.float32(DTYPE(left)) * wp.float32(DTYPE(right))

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        embeddings: wp.array2d(dtype=DTYPE),
        scores: wp.array1d(dtype=wp.float32),
        spatial_tokens: int,
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        row = wp.tid()
        if row < spatial_tokens:
            scores[row] = 255.0
        else:
            current = wp.tile_load(embeddings[row], shape=(HIDDEN_SIZE,))
            previous = wp.tile_load(
                embeddings[row - spatial_tokens], shape=(HIDDEN_SIZE,)
            )
            dot = wp.tile_extract(
                wp.tile_sum(wp.tile_map(multiply, current, previous)), 0
            )
            current_norm = wp.tile_extract(
                wp.tile_sum(wp.tile_map(multiply, current, current)), 0
            )
            previous_norm = wp.tile_extract(
                wp.tile_sum(wp.tile_map(multiply, previous, previous)), 0
            )
            cosine = dot / wp.sqrt(
                wp.max(current_norm * previous_norm, wp.float32(1.0e-20))
            )
            scores[row] = 1.0 - cosine

    kernel.module.options["enable_backward"] = False
    return min(1024, max(32, 1 << (hidden_size - 1).bit_length())), kernel


def prune_video_embeddings(
    embeddings,
    temporal_groups: int,
    spatial_tokens: int,
    pruning_rate: float,
):
    """Apply the official stable EVS ranking while retaining data on device."""
    if embeddings.ndim != 2 or embeddings.shape[0] != temporal_groups * spatial_tokens:
        raise ValueError("video embedding geometry does not match T*H*W")
    if not 0.0 <= pruning_rate < 1.0:
        raise ValueError("video pruning rate must be in [0, 1)")
    if pruning_rate == 0.0 or temporal_groups == 1:
        return embeddings, tuple(range(embeddings.shape[0]))
    scores = wp.empty(embeddings.shape[0], dtype=wp.float32, device=embeddings.device)
    block_dim, kernel = _video_dissimilarity_kernel(
        embeddings.dtype, embeddings.shape[1]
    )
    wp.launch_tiled(
        kernel,
        dim=embeddings.shape[0],
        inputs=[embeddings, scores, spatial_tokens],
        block_dim=block_dim,
        device=embeddings.device,
    )
    values = scores.numpy()
    keep = max(
        spatial_tokens,
        int(temporal_groups * spatial_tokens * (1.0 - pruning_rate)),
    )
    retained = np.sort(np.argsort(-values, kind="stable")[:keep]).astype(np.int64)
    indices = wp.array(retained[None, :], device=embeddings.device)
    compact = wp.empty(
        (1, keep, embeddings.shape[1]),
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    wp.launch(
        _gather_rows_kernel,
        dim=compact.shape,
        inputs=[embeddings, indices, compact],
        device=embeddings.device,
    )
    return compact.reshape((keep, embeddings.shape[1])), tuple(retained.tolist())
