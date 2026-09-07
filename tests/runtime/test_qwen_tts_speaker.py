# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import wave

import numpy as np
import pytest

from warp_nn.runtime.qwen.tts_speaker import (
    Qwen3TTSSpeakerConfig,
    qwen3_tts_log_mel,
    qwen3_tts_log_mel_from_wav,
    qwen3_tts_mel_filter_bank,
    qwen3_tts_speaker_weight_names,
)


def test_qwen3_tts_speaker_config_and_weight_manifest(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"speaker_encoder_config": {"enc_dim": 2048, "sample_rate": 24000}}),
        encoding="utf-8",
    )
    config = Qwen3TTSSpeakerConfig.from_path(path)
    assert config.embedding_dim == 2048
    assert config.channels == (512, 512, 512, 512, 1536)
    names = qwen3_tts_speaker_weight_names(config)
    assert len(names) == 76
    assert len(names) == len(set(names))
    assert names[0] == "speaker_encoder.blocks.0.conv.weight"
    assert names[-1] == "speaker_encoder.fc.bias"
    assert "speaker_encoder.blocks.3.res2net_block.blocks.6.conv.weight" in names


def test_qwen3_tts_log_mel_is_finite_and_frequency_selective():
    filters = qwen3_tts_mel_filter_bank()
    assert filters.shape == (128, 513)
    assert filters.dtype == np.float32
    assert np.all(filters >= 0.0)

    time = np.arange(24000, dtype=np.float32) / 24000.0
    features = qwen3_tts_log_mel(np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32))
    assert features.shape == (93, 128)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert float(features.max() - features.min()) > 10.0


def test_qwen3_tts_wav_boundary_accepts_mono_and_rejects_wrong_rate(tmp_path):
    time = np.arange(2400, dtype=np.float32) / 24000.0
    samples = np.sin(2.0 * np.pi * 220.0 * time)

    def write(path, rate):
        pcm = np.rint(samples * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(pcm.tobytes())

    good = tmp_path / "good.wav"
    write(good, 24000)
    assert qwen3_tts_log_mel_from_wav(good).shape == (9, 128)

    wrong = tmp_path / "wrong.wav"
    write(wrong, 16000)
    with pytest.raises(ValueError, match="24000 Hz"):
        qwen3_tts_log_mel_from_wav(wrong)


@pytest.mark.parametrize(
    "waveform,exception",
    [
        (np.zeros((2, 20), dtype=np.float32), TypeError),
        (np.zeros(700, dtype=np.int16), TypeError),
        (np.full(700, np.nan, dtype=np.float32), ValueError),
        (np.zeros(300, dtype=np.float32), ValueError),
    ],
)
def test_qwen3_tts_log_mel_rejects_invalid_audio(waveform, exception):
    with pytest.raises(exception):
        qwen3_tts_log_mel(waveform)
