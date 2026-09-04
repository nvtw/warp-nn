# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional compressed-media decoding through PyAV.

Importing :mod:`warp_nn.runtime` never imports PyAV.  Callers enter this module
only for compressed audio or video; PNG and PCM16 WAV retain their small
dependency-free loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _pyav():
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "compressed media requires the optional dependency: "
            "pip install warp-nn[media]"
        ) from exc
    return av


@dataclass(frozen=True)
class VideoFrames:
    """Sampled RGB frames plus their locations in the source stream."""

    frames: tuple[np.ndarray, ...]
    source_fps: float
    source_indices: tuple[int, ...]


def _sample_indices(total: int, desired: int) -> tuple[int, ...]:
    if total <= 0 or desired <= 0:
        raise ValueError("video frame counts must be positive")
    if desired >= total:
        return tuple(range(total))
    if desired == 1:
        return (0,)
    return tuple(
        int(value)
        for value in np.unique(
            np.rint(np.linspace(0, total - 1, desired)).astype(np.int64)
        )
    )


def decode_video_frames(
    path: str | Path, *, fps: float = 1.0, max_frames: int = 128
) -> VideoFrames:
    """Decode evenly sampled RGB frames without retaining the full video."""
    if not np.isfinite(fps) or fps <= 0.0 or max_frames <= 0:
        raise ValueError("video fps and max_frames must be positive")
    av = _pyav()
    with av.open(str(Path(path))) as container:
        if not container.streams.video:
            raise ValueError("media file contains no video stream")
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or stream.guessed_rate or 0.0)
        if not np.isfinite(source_fps) or source_fps <= 0.0:
            raise ValueError("video stream has no usable frame rate")
        total = int(stream.frames or 0)
        if total <= 0 and stream.duration is not None:
            total = max(
                1, round(float(stream.duration * stream.time_base) * source_fps)
            )
        if total <= 0:
            # Rare containers omit both frame count and duration. Decode once;
            # this fallback favors correctness over memory for malformed metadata.
            decoded = tuple(
                np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
                for frame in container.decode(stream)
            )
            if not decoded:
                raise ValueError("video stream contains no decodable frames")
            indices = _sample_indices(
                len(decoded),
                min(max_frames, max(1, int(len(decoded) / source_fps * fps))),
            )
            return VideoFrames(
                tuple(decoded[index] for index in indices), source_fps, indices
            )

        duration = total / source_fps
        desired = min(max_frames, max(1, int(duration * fps)))
        indices = _sample_indices(total, desired)
        targets = iter(indices)
        target = next(targets, None)
        frames = []
        for index, frame in enumerate(container.decode(stream)):
            if target is None:
                break
            if index < target:
                continue
            frames.append(
                np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            )
            target = next(targets, None)
        if len(frames) != len(indices):
            raise ValueError("video ended before its advertised frame count")
        return VideoFrames(tuple(frames), source_fps, indices)


def decode_audio_mono(path: str | Path, *, sample_rate: int = 16_000) -> np.ndarray:
    """Decode the first audio stream to contiguous mono FP32 samples."""
    if sample_rate <= 0:
        raise ValueError("audio sample rate must be positive")
    av = _pyav()
    with av.open(str(Path(path))) as container:
        if not container.streams.audio:
            raise ValueError("media file contains no audio stream")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        chunks = []
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                chunks.append(
                    np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
                )
        for converted in resampler.resample(None):
            chunks.append(
                np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
            )
    if not chunks:
        raise ValueError("audio stream contains no decodable samples")
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
