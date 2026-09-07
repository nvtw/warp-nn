# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Qwen3-TTS reference-audio frontend and ECAPA speaker encoder.

The public preprocessing functions intentionally use only NumPy.  The model
plan keeps fixed-shape execution on Warp so it can be captured together with
the rest of TTS inference.  Checkpoint-specific topology and tensor names stay
here; the expensive convolutions reuse the generic runtime Conv1D operator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import warp as wp

from ..formats.wav import read_wav_pcm16
from ..operators import Conv1dPlan


@dataclass(frozen=True)
class Qwen3TTSSpeakerConfig:
    """Normalized fields of ``speaker_encoder_config``."""

    mel_dim: int = 128
    embedding_dim: int = 1024
    channels: tuple[int, ...] = (512, 512, 512, 512, 1536)
    kernel_sizes: tuple[int, ...] = (5, 3, 3, 3, 1)
    dilations: tuple[int, ...] = (1, 2, 3, 4, 1)
    attention_channels: int = 128
    res2net_scale: int = 8
    se_channels: int = 128
    sample_rate: int = 24_000

    @classmethod
    def from_document(cls, document: dict) -> "Qwen3TTSSpeakerConfig":
        values = document.get("speaker_encoder_config", document)
        config = cls(
            mel_dim=int(values.get("mel_dim", 128)),
            embedding_dim=int(values.get("enc_dim", 1024)),
            channels=tuple(
                int(x) for x in values.get("enc_channels", (512, 512, 512, 512, 1536))
            ),
            kernel_sizes=tuple(
                int(x) for x in values.get("enc_kernel_sizes", (5, 3, 3, 3, 1))
            ),
            dilations=tuple(
                int(x) for x in values.get("enc_dilations", (1, 2, 3, 4, 1))
            ),
            attention_channels=int(values.get("enc_attention_channels", 128)),
            res2net_scale=int(values.get("enc_res2net_scale", 8)),
            se_channels=int(values.get("enc_se_channels", 128)),
            sample_rate=int(values.get("sample_rate", 24_000)),
        )
        config.validate()
        return config

    @classmethod
    def from_path(cls, path: str | Path) -> "Qwen3TTSSpeakerConfig":
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        return cls.from_document(json.loads(path.read_text(encoding="utf-8")))

    def validate(self) -> None:
        if (
            len(self.channels) != 5
            or len(self.kernel_sizes) != 5
            or len(self.dilations) != 5
        ):
            raise ValueError("Qwen3-TTS speaker encoder requires five channel stages")
        if (
            min(
                self.mel_dim,
                self.embedding_dim,
                self.attention_channels,
                self.se_channels,
                self.sample_rate,
            )
            <= 0
        ):
            raise ValueError("Qwen3-TTS speaker dimensions must be positive")
        if any(
            value <= 0 for value in self.channels + self.kernel_sizes + self.dilations
        ):
            raise ValueError("Qwen3-TTS speaker stage dimensions must be positive")
        if self.channels[:4] != (512, 512, 512, 512) or self.channels[4] != sum(
            self.channels[1:4]
        ):
            raise ValueError("unsupported Qwen3-TTS ECAPA channel topology")
        if self.channels[0] % self.res2net_scale:
            raise ValueError("ECAPA channels must be divisible by the Res2Net scale")
        if any(kernel % 2 != 1 for kernel in self.kernel_sizes):
            raise ValueError("ECAPA same-padding kernels must be odd")


def qwen3_tts_mel_filter_bank(
    *,
    sample_rate: int = 24_000,
    n_fft: int = 1024,
    num_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = 12_000.0,
) -> np.ndarray:
    """Return librosa-compatible Slaney mel filters used by Qwen3-TTS."""
    if sample_rate <= 0 or n_fft <= 0 or num_mels <= 0:
        raise ValueError("mel filter-bank dimensions must be positive")
    nyquist = sample_rate / 2.0
    fmax = nyquist if fmax is None else float(fmax)
    if not 0.0 <= fmin < fmax <= nyquist:
        raise ValueError("mel frequency range must lie within Nyquist")

    def hz_to_mel(frequencies):
        frequencies = np.asarray(frequencies, dtype=np.float64)
        mels = frequencies / (200.0 / 3.0)
        logarithmic = frequencies >= 1000.0
        mels[logarithmic] = 15.0 + np.log(frequencies[logarithmic] / 1000.0) / (
            np.log(6.4) / 27.0
        )
        return mels

    def mel_to_hz(mels):
        mels = np.asarray(mels, dtype=np.float64)
        frequencies = (200.0 / 3.0) * mels
        logarithmic = mels >= 15.0
        frequencies[logarithmic] = 1000.0 * np.exp(
            (np.log(6.4) / 27.0) * (mels[logarithmic] - 15.0)
        )
        return frequencies

    fft_frequencies = np.linspace(0.0, nyquist, n_fft // 2 + 1)
    mel_edges = np.linspace(hz_to_mel([fmin])[0], hz_to_mel([fmax])[0], num_mels + 2)
    hz_edges = mel_to_hz(mel_edges)
    ramps = hz_edges[:, None] - fft_frequencies[None, :]
    weights = np.maximum(
        0.0,
        np.minimum(
            -ramps[:-2] / np.diff(hz_edges)[:-1, None],
            ramps[2:] / np.diff(hz_edges)[1:, None],
        ),
    )
    weights *= (2.0 / (hz_edges[2:] - hz_edges[:-2]))[:, None]
    return np.ascontiguousarray(weights, dtype=np.float32)


def qwen3_tts_log_mel(
    waveform: np.ndarray,
    *,
    sample_rate: int = 24_000,
    n_fft: int = 1024,
    hop_size: int = 256,
    win_size: int = 1024,
    num_mels: int = 128,
) -> np.ndarray:
    """Compute the official Qwen3-TTS log-magnitude mel representation.

    The returned channels-last array has shape ``[frames, num_mels]`` and is
    directly consumable by :class:`Qwen3TTSSpeakerEncoderPlan`.
    """
    value = np.asarray(waveform)
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.floating):
        raise TypeError("reference audio must be a mono floating-point array")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("reference audio must contain only finite samples")
    padding = (n_fft - hop_size) // 2
    if value.size <= padding:
        raise ValueError(f"reference audio must contain more than {padding} samples")
    if sample_rate != 24_000:
        raise ValueError("Qwen3-TTS speaker audio must be resampled to 24000 Hz")
    if not 0 < win_size <= n_fft or hop_size <= 0:
        raise ValueError("invalid STFT geometry")

    padded = np.pad(value, (padding, padding), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_size]
    window = np.hanning(win_size + 1)[:-1].astype(np.float32)
    if win_size != n_fft:
        expanded = np.zeros(n_fft, dtype=np.float32)
        offset = (n_fft - win_size) // 2
        expanded[offset : offset + win_size] = window
        window = expanded
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=-1)
    magnitude = np.sqrt(
        spectrum.real * spectrum.real + spectrum.imag * spectrum.imag + 1.0e-9
    )
    filters = qwen3_tts_mel_filter_bank(
        sample_rate=sample_rate, n_fft=n_fft, num_mels=num_mels
    )
    features = np.log(np.maximum(magnitude @ filters.T, 1.0e-5))
    return np.ascontiguousarray(features, dtype=np.float32)


def qwen3_tts_log_mel_from_wav(path: str | Path) -> np.ndarray:
    """Read a 24-kHz PCM16 WAV and return its Qwen3-TTS speaker features."""
    audio = read_wav_pcm16(path)
    if audio.sample_rate != 24_000:
        raise ValueError("Qwen3-TTS reference WAV must use a 24000 Hz sample rate")
    mono = audio.samples.mean(axis=1, dtype=np.float32)
    return qwen3_tts_log_mel(mono, sample_rate=audio.sample_rate)


def qwen3_tts_speaker_weight_names(config: Qwen3TTSSpeakerConfig) -> tuple[str, ...]:
    """Return every checkpoint tensor required by the native speaker encoder."""
    names: list[str] = []

    def convolution(prefix: str) -> None:
        names.extend((f"{prefix}.weight", f"{prefix}.bias"))

    convolution("speaker_encoder.blocks.0.conv")
    for stage in range(1, 4):
        base = f"speaker_encoder.blocks.{stage}"
        convolution(f"{base}.tdnn1.conv")
        for block in range(config.res2net_scale - 1):
            convolution(f"{base}.res2net_block.blocks.{block}.conv")
        convolution(f"{base}.tdnn2.conv")
        convolution(f"{base}.se_block.conv1")
        convolution(f"{base}.se_block.conv2")
    convolution("speaker_encoder.mfa.conv")
    convolution("speaker_encoder.asp.tdnn.conv")
    convolution("speaker_encoder.asp.conv")
    convolution("speaker_encoder.fc")
    return tuple(names)


@wp.kernel(enable_backward=False, module="unique")
def _reflect_pad_1d(
    x: wp.array3d(dtype=wp.bfloat16),
    output: wp.array3d(dtype=wp.bfloat16),
    channel_offset: int,
    source_channels: int,
    padding: int,
    residual: wp.array3d(dtype=wp.bfloat16),
    add_residual: bool,
):
    batch, position, channel = wp.tid()
    source = position - padding
    if source < 0:
        source = -source
    elif source >= x.shape[1]:
        source = 2 * x.shape[1] - source - 2
    value = wp.float32(x[batch, source, channel_offset + channel])
    if add_residual:
        value += wp.float32(residual[batch, source, channel])
    output[batch, position, channel] = wp.bfloat16(value)


@wp.kernel(enable_backward=False, module="unique")
def _relu_3d(x: wp.array3d(dtype=wp.bfloat16)):
    batch, position, channel = wp.tid()
    x[batch, position, channel] = wp.bfloat16(
        wp.max(wp.float32(x[batch, position, channel]), 0.0)
    )


@wp.kernel(enable_backward=False, module="unique")
def _copy_channel_slice(
    x: wp.array3d(dtype=wp.bfloat16),
    output: wp.array3d(dtype=wp.bfloat16),
    offset: int,
):
    batch, position, channel = wp.tid()
    output[batch, position, offset + channel] = x[batch, position, channel]


@wp.kernel(enable_backward=False, module="unique")
def _mean_channels(
    x: wp.array3d(dtype=wp.bfloat16), output: wp.array3d(dtype=wp.bfloat16)
):
    batch, channel = wp.tid()
    total = float(0.0)
    for position in range(x.shape[1]):
        total += wp.float32(x[batch, position, channel])
    output[batch, 0, channel] = wp.bfloat16(total / float(x.shape[1]))


@wp.kernel(enable_backward=False, module="unique")
def _sigmoid_gate_residual(
    gate: wp.array3d(dtype=wp.bfloat16),
    hidden: wp.array3d(dtype=wp.bfloat16),
    residual: wp.array3d(dtype=wp.bfloat16),
    output: wp.array3d(dtype=wp.bfloat16),
):
    batch, position, channel = wp.tid()
    g = 1.0 / (1.0 + wp.exp(-wp.float32(gate[batch, 0, channel])))
    value = wp.float32(hidden[batch, position, channel]) * g + wp.float32(
        residual[batch, position, channel]
    )
    output[batch, position, channel] = wp.bfloat16(value)


@wp.kernel(enable_backward=False, module="unique")
def _pack_mfa(
    x1: wp.array3d(dtype=wp.bfloat16),
    x2: wp.array3d(dtype=wp.bfloat16),
    x3: wp.array3d(dtype=wp.bfloat16),
    output: wp.array3d(dtype=wp.bfloat16),
):
    batch, position, channel = wp.tid()
    width = x1.shape[2]
    if channel < width:
        output[batch, position, channel] = x1[batch, position, channel]
    elif channel < 2 * width:
        output[batch, position, channel] = x2[batch, position, channel - width]
    else:
        output[batch, position, channel] = x3[batch, position, channel - 2 * width]


@wp.kernel(enable_backward=False, module="unique")
def _pack_pooling_context(
    x: wp.array3d(dtype=wp.bfloat16), output: wp.array3d(dtype=wp.bfloat16)
):
    batch, channel = wp.tid()
    total = float(0.0)
    squared = float(0.0)
    for position in range(x.shape[1]):
        value = wp.float32(x[batch, position, channel])
        total += value
        squared += value * value
    mean = total / float(x.shape[1])
    variance = wp.max(squared / float(x.shape[1]) - mean * mean, 1.0e-12)
    standard_deviation = wp.sqrt(variance)
    for position in range(x.shape[1]):
        output[batch, position, channel] = x[batch, position, channel]
        output[batch, position, x.shape[2] + channel] = wp.bfloat16(mean)
        output[batch, position, 2 * x.shape[2] + channel] = wp.bfloat16(
            standard_deviation
        )


@wp.kernel(enable_backward=False, module="unique")
def _tanh_3d(x: wp.array3d(dtype=wp.bfloat16)):
    batch, position, channel = wp.tid()
    x[batch, position, channel] = wp.bfloat16(
        wp.tanh(wp.float32(x[batch, position, channel]))
    )


@wp.kernel(enable_backward=False, module="unique")
def _attentive_statistics(
    x: wp.array3d(dtype=wp.bfloat16),
    logits: wp.array3d(dtype=wp.bfloat16),
    output: wp.array3d(dtype=wp.bfloat16),
):
    batch, channel = wp.tid()
    maximum = float(-3.402823466e38)
    for position in range(x.shape[1]):
        maximum = wp.max(maximum, wp.float32(logits[batch, position, channel]))
    denominator = float(0.0)
    mean = float(0.0)
    for position in range(x.shape[1]):
        weight = wp.exp(wp.float32(logits[batch, position, channel]) - maximum)
        denominator += weight
        mean += weight * wp.float32(x[batch, position, channel])
    mean /= denominator
    variance = float(0.0)
    for position in range(x.shape[1]):
        weight = (
            wp.exp(wp.float32(logits[batch, position, channel]) - maximum) / denominator
        )
        delta = wp.float32(x[batch, position, channel]) - mean
        variance += weight * delta * delta
    output[batch, 0, channel] = wp.bfloat16(mean)
    output[batch, 0, x.shape[2] + channel] = wp.bfloat16(
        wp.sqrt(wp.max(variance, 1.0e-12))
    )


class _ReflectTDNNPlan:
    def __init__(
        self, x, weight, bias, *, dilation=1, channel_offset=0, residual=None, relu=True
    ):
        kernel_size = weight.shape[2]
        padding = dilation * (kernel_size - 1) // 2
        if padding >= x.shape[1]:
            raise ValueError(
                "reference audio is too short for ECAPA reflection padding"
            )
        self.input = x
        self.padding = padding
        self.channel_offset = int(channel_offset)
        self.residual = residual if residual is not None else x
        self.add_residual = residual is not None
        self.relu = bool(relu)
        channels = weight.shape[1]
        self.padded = wp.empty(
            (x.shape[0], x.shape[1] + 2 * padding, channels),
            dtype=x.dtype,
            device=x.device,
        )
        self.convolution = Conv1dPlan(self.padded, weight, bias, dilation=dilation)
        self.output = self.convolution.output

    def execute(self):
        wp.launch(
            _reflect_pad_1d,
            dim=self.padded.shape,
            inputs=[
                self.input,
                self.padded,
                self.channel_offset,
                self.padded.shape[2],
                self.padding,
                self.residual,
                self.add_residual,
            ],
            device=self.input.device,
        )
        self.convolution.execute()
        if self.relu:
            wp.launch(
                _relu_3d,
                dim=self.output.shape,
                inputs=[self.output],
                device=self.input.device,
            )
        return self.output


class _ECAPAStagePlan:
    def __init__(
        self,
        x,
        weights: Mapping[str, wp.array],
        prefix: str,
        *,
        scale: int,
        kernel_size: int,
        dilation: int,
    ):
        self.input = x
        self.tdnn1 = _ReflectTDNNPlan(
            x,
            weights[f"{prefix}.tdnn1.conv.weight"],
            weights[f"{prefix}.tdnn1.conv.bias"],
        )
        width = x.shape[2] // scale
        self.res2_output = wp.empty_like(x)
        self.res2_blocks = []
        for index in range(scale - 1):
            self.res2_blocks.append(
                _ReflectTDNNPlan(
                    self.tdnn1.output,
                    weights[f"{prefix}.res2net_block.blocks.{index}.conv.weight"],
                    weights[f"{prefix}.res2net_block.blocks.{index}.conv.bias"],
                    dilation=dilation,
                    channel_offset=(index + 1) * width,
                    residual=None if index == 0 else self.res2_blocks[-1].output,
                )
            )
        self.tdnn2 = _ReflectTDNNPlan(
            self.res2_output,
            weights[f"{prefix}.tdnn2.conv.weight"],
            weights[f"{prefix}.tdnn2.conv.bias"],
        )
        self.mean = wp.empty(
            (x.shape[0], 1, x.shape[2]), dtype=x.dtype, device=x.device
        )
        self.se1 = Conv1dPlan(
            self.mean,
            weights[f"{prefix}.se_block.conv1.weight"],
            weights[f"{prefix}.se_block.conv1.bias"],
        )
        self.se2 = Conv1dPlan(
            self.se1.output,
            weights[f"{prefix}.se_block.conv2.weight"],
            weights[f"{prefix}.se_block.conv2.bias"],
        )
        self.output = wp.empty_like(x)

    def execute(self):
        self.tdnn1.execute()
        width = self.input.shape[2] // (len(self.res2_blocks) + 1)
        wp.launch(
            _copy_channel_slice,
            dim=(self.input.shape[0], self.input.shape[1], width),
            inputs=[self.tdnn1.output, self.res2_output, 0],
            device=self.input.device,
        )
        for index, block in enumerate(self.res2_blocks):
            block.execute()
            wp.launch(
                _copy_channel_slice,
                dim=block.output.shape,
                inputs=[block.output, self.res2_output, (index + 1) * width],
                device=self.input.device,
            )
        self.tdnn2.execute()
        wp.launch(
            _mean_channels,
            dim=(self.input.shape[0], self.input.shape[2]),
            inputs=[self.tdnn2.output, self.mean],
            device=self.input.device,
        )
        self.se1.execute()
        wp.launch(
            _relu_3d,
            dim=self.se1.output.shape,
            inputs=[self.se1.output],
            device=self.input.device,
        )
        self.se2.execute()
        wp.launch(
            _sigmoid_gate_residual,
            dim=self.output.shape,
            inputs=[self.se2.output, self.tdnn2.output, self.input, self.output],
            device=self.input.device,
        )
        return self.output


class Qwen3TTSSpeakerEncoderPlan:
    """Fixed-shape, graph-safe native ECAPA-TDNN speaker encoder plan."""

    def __init__(
        self,
        features,
        weights: Mapping[str, wp.array],
        config: Qwen3TTSSpeakerConfig | None = None,
    ):
        self.config = config or Qwen3TTSSpeakerConfig()
        self.config.validate()
        if (
            features.ndim != 3
            or features.shape[2] != self.config.mel_dim
            or features.dtype != wp.bfloat16
        ):
            raise TypeError(
                "speaker features must be rank-three BF16 [batch, frames, 128]"
            )
        missing = set(qwen3_tts_speaker_weight_names(self.config)).difference(weights)
        if missing:
            raise KeyError(
                f"speaker encoder is missing checkpoint tensor '{min(missing)}'"
            )
        self.input = features
        self.initial = _ReflectTDNNPlan(
            features,
            weights["speaker_encoder.blocks.0.conv.weight"],
            weights["speaker_encoder.blocks.0.conv.bias"],
        )
        self.stages = []
        current = self.initial.output
        for index in range(1, 4):
            stage = _ECAPAStagePlan(
                current,
                weights,
                f"speaker_encoder.blocks.{index}",
                scale=self.config.res2net_scale,
                kernel_size=self.config.kernel_sizes[index],
                dilation=self.config.dilations[index],
            )
            self.stages.append(stage)
            current = stage.output
        self.mfa_input = wp.empty(
            (features.shape[0], features.shape[1], self.config.channels[-1]),
            dtype=features.dtype,
            device=features.device,
        )
        self.mfa = _ReflectTDNNPlan(
            self.mfa_input,
            weights["speaker_encoder.mfa.conv.weight"],
            weights["speaker_encoder.mfa.conv.bias"],
        )
        self.pool_context = wp.empty(
            (features.shape[0], features.shape[1], 3 * self.config.channels[-1]),
            dtype=features.dtype,
            device=features.device,
        )
        self.asp_tdnn = Conv1dPlan(
            self.pool_context,
            weights["speaker_encoder.asp.tdnn.conv.weight"],
            weights["speaker_encoder.asp.tdnn.conv.bias"],
        )
        self.asp_conv = Conv1dPlan(
            self.asp_tdnn.output,
            weights["speaker_encoder.asp.conv.weight"],
            weights["speaker_encoder.asp.conv.bias"],
        )
        self.pooled = wp.empty(
            (features.shape[0], 1, 2 * self.config.channels[-1]),
            dtype=features.dtype,
            device=features.device,
        )
        self.fc = Conv1dPlan(
            self.pooled,
            weights["speaker_encoder.fc.weight"],
            weights["speaker_encoder.fc.bias"],
        )
        self.output = self.fc.output.reshape(
            (features.shape[0], self.config.embedding_dim)
        )

    def execute(self):
        self.initial.execute()
        for stage in self.stages:
            stage.execute()
        wp.launch(
            _pack_mfa,
            dim=self.mfa_input.shape,
            inputs=[
                self.stages[0].output,
                self.stages[1].output,
                self.stages[2].output,
                self.mfa_input,
            ],
            device=self.input.device,
        )
        self.mfa.execute()
        wp.launch(
            _pack_pooling_context,
            dim=(self.input.shape[0], self.config.channels[-1]),
            inputs=[self.mfa.output, self.pool_context],
            device=self.input.device,
        )
        self.asp_tdnn.execute()
        wp.launch(
            _relu_3d,
            dim=self.asp_tdnn.output.shape,
            inputs=[self.asp_tdnn.output],
            device=self.input.device,
        )
        wp.launch(
            _tanh_3d,
            dim=self.asp_tdnn.output.shape,
            inputs=[self.asp_tdnn.output],
            device=self.input.device,
        )
        self.asp_conv.execute()
        wp.launch(
            _attentive_statistics,
            dim=(self.input.shape[0], self.config.channels[-1]),
            inputs=[self.mfa.output, self.asp_conv.output, self.pooled],
            device=self.input.device,
        )
        self.fc.execute()
        return self.output
