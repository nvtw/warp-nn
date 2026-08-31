# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import wave

import numpy as np
import pytest

from warp_nn.runtime.formats.wav import (
    float_to_pcm16,
    read_wav_pcm16,
    write_wav_pcm16,
)


def test_pcm16_clipping_uses_full_signed_range():
    samples = np.array(
        [[-2.0, -1.0], [-0.5, 0.0], [0.5, 1.0], [2.0, 0.25]],
        dtype=np.float32,
    )
    actual = float_to_pcm16(samples)
    np.testing.assert_array_equal(
        actual,
        [[-32768, -32768], [-16384, 0], [16384, 32767], [32767, 8192]],
    )


def test_pcm16_peak_normalization_is_opt_in():
    samples = np.array([[-0.25, 0.5], [0.125, -0.5]], dtype=np.float32)
    unchanged = float_to_pcm16(samples)
    normalized = float_to_pcm16(samples, normalize=True)
    assert unchanged[0, 1] == 16384
    assert normalized[0, 1] == 32767
    assert normalized[1, 1] == -32768


def test_stereo_pcm16_wav_round_trip_and_header(tmp_path):
    samples = np.array(
        [[-1.0, 1.0], [-0.125, 0.25], [0.0, 0.0], [0.75, -0.5]],
        dtype=np.float32,
    )
    path = tmp_path / "audio.wav"
    write_wav_pcm16(path, samples, 48_000)
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 2
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == 48_000
        assert stream.getnframes() == len(samples)
        assert stream.getcomptype() == "NONE"
    decoded = read_wav_pcm16(path)
    assert decoded.sample_rate == 48_000
    assert decoded.samples.dtype == np.float32
    assert decoded.samples.flags.c_contiguous
    np.testing.assert_allclose(decoded.samples, samples, atol=1.0 / 32767.0)


def test_wav_boundary_rejects_ambiguous_or_invalid_audio(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        float_to_pcm16(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(TypeError, match="floating"):
        float_to_pcm16(np.zeros((2, 2), dtype=np.int16))
    with pytest.raises(ValueError, match="finite"):
        float_to_pcm16(np.array([[np.nan, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        write_wav_pcm16(tmp_path / "bad.wav", np.zeros((1, 2), dtype=np.float32), 0)

    mono = tmp_path / "mono.wav"
    with wave.open(str(mono), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(np.zeros(4, dtype="<i2").tobytes())
    with pytest.raises(ValueError, match="stereo"):
        read_wav_pcm16(mono)
