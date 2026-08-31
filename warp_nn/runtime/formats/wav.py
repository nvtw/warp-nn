# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free stereo PCM16 WAV input and output."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WavAudio:
    """Channels-last stereo samples in ``[-1, 1]`` and their sample rate."""

    samples: np.ndarray
    sample_rate: int


def _stereo_float_samples(samples) -> np.ndarray:
    values = np.asarray(samples)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("WAV audio must have channels-last shape [samples, 2]")
    if values.shape[0] == 0:
        raise ValueError("WAV audio must contain at least one sample")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("WAV input samples must use a floating dtype")
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("WAV input samples must be finite")
    return values


def float_to_pcm16(samples, *, normalize: bool = False) -> np.ndarray:
    """Quantize stereo floats to PCM16 with explicit clipping or normalization.

    With ``normalize=False`` (the default), values outside ``[-1, 1]`` are
    clipped and in-range loudness is unchanged. With ``normalize=True``, both
    channels are scaled together so their absolute peak is one before clipping.
    """
    values = _stereo_float_samples(samples)
    if normalize:
        peak = float(np.max(np.abs(values)))
        if peak > 0.0:
            values = values / peak
    values = np.clip(values, -1.0, 1.0)
    scaled = np.where(values < 0.0, values * 32768.0, values * 32767.0)
    return np.rint(scaled).astype("<i2")


def write_wav_pcm16(
    path: str | Path,
    samples,
    sample_rate: int,
    *,
    normalize: bool = False,
) -> None:
    """Write channels-last stereo floats as an uncompressed PCM16 WAV file."""
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("WAV sample_rate must be positive")
    pcm = float_to_pcm16(samples, normalize=normalize)
    with wave.open(str(Path(path)), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes(order="C"))


def read_wav_pcm16(path: str | Path) -> WavAudio:
    """Read an uncompressed stereo PCM16 WAV into channels-last FP32 samples."""
    with wave.open(str(Path(path)), "rb") as stream:
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        compression = stream.getcomptype()
        sample_rate = stream.getframerate()
        frames = stream.getnframes()
        if channels != 2:
            raise ValueError(f"WAV input must be stereo, found {channels} channels")
        if width != 2 or compression != "NONE":
            raise ValueError("WAV input must be uncompressed 16-bit PCM")
        raw = stream.readframes(frames)
    pcm = np.frombuffer(raw, dtype="<i2").reshape((-1, 2))
    values = pcm.astype(np.float32)
    values = np.where(values < 0.0, values / 32768.0, values / 32767.0)
    return WavAudio(np.ascontiguousarray(values, dtype=np.float32), sample_rate)
