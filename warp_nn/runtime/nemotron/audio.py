# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free Nemotron Omni audio preprocessing and model layout.

The checkpoint uses the Parakeet Fast Conformer encoder followed by an
RMSNorm--MLP projection into the language-model embedding width.  This module
keeps checkpoint-specific names and shape rules out of the generic operators.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import warp as wp

from ..formats.wav import read_wav_pcm16
from ..formats.safetensors import SafeTensorArchive
from ..operators import (
    AttentionHeadsPlan,
    AttentionMergePlan,
    Conv1dPlan,
    Conv2dPlan,
    LayerNormPlan,
    Operation,
    RMSNormPlan,
    RelativeBidirectionalAttentionPlan,
    execute_operations,
    plan_linear,
)


_LOG_ZERO_GUARD = 2.0**-24
_FEATURE_EPSILON = 1.0e-5


@dataclass(frozen=True)
class NemotronAudioConfig:
    """Normalized audio fields from an Omni ``config.json``."""

    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    conv_kernel_size: int
    subsampling_conv_channels: int
    subsampling_conv_kernel_size: int
    subsampling_conv_stride: int
    subsampling_factor: int
    num_mel_bins: int
    projection_hidden_size: int
    llm_hidden_size: int
    sampling_rate: int = 16_000
    attention_bias: bool = False
    convolution_bias: bool = False
    projection_bias: bool = False
    scale_input: bool = False
    layer_norm_epsilon: float = 1.0e-5

    @classmethod
    def from_document(cls, document: dict) -> "NemotronAudioConfig":
        """Read and validate the nested sound and language configuration."""
        try:
            sound = document["sound_config"]
            llm = document["llm_config"]
            config = cls(
                hidden_size=int(sound["hidden_size"]),
                num_attention_heads=int(sound["num_attention_heads"]),
                num_hidden_layers=int(sound["num_hidden_layers"]),
                intermediate_size=int(sound["intermediate_size"]),
                conv_kernel_size=int(sound.get("conv_kernel_size", 9)),
                subsampling_conv_channels=int(sound["subsampling_conv_channels"]),
                subsampling_conv_kernel_size=int(
                    sound.get("subsampling_conv_kernel_size", 3)
                ),
                subsampling_conv_stride=int(sound.get("subsampling_conv_stride", 2)),
                subsampling_factor=int(sound.get("subsampling_factor", 8)),
                num_mel_bins=int(sound.get("num_mel_bins", 128)),
                projection_hidden_size=int(sound["projection_hidden_size"]),
                llm_hidden_size=int(llm["hidden_size"]),
                sampling_rate=int(sound.get("sampling_rate", 16_000)),
                attention_bias=bool(sound.get("attention_bias", False)),
                convolution_bias=bool(sound.get("convolution_bias", False)),
                projection_bias=bool(sound.get("projection_bias", False)),
                scale_input=bool(sound.get("scale_input", False)),
                layer_norm_epsilon=float(sound.get("layer_norm_epsilon", 1.0e-5)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid Nemotron Omni sound configuration") from exc
        config.validate()
        return config

    @classmethod
    def from_path(cls, path: str | Path) -> "NemotronAudioConfig":
        path = Path(path)
        config_path = path / "config.json" if path.is_dir() else path
        return cls.from_document(json.loads(config_path.read_text(encoding="utf-8")))

    @property
    def subsampling_layers(self) -> int:
        return 3

    def validate(self) -> None:
        integers = (
            self.hidden_size,
            self.num_attention_heads,
            self.num_hidden_layers,
            self.intermediate_size,
            self.conv_kernel_size,
            self.subsampling_conv_channels,
            self.subsampling_conv_kernel_size,
            self.subsampling_conv_stride,
            self.subsampling_factor,
            self.num_mel_bins,
            self.projection_hidden_size,
            self.llm_hidden_size,
            self.sampling_rate,
        )
        if any(value <= 0 for value in integers):
            raise ValueError("Nemotron audio dimensions must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("Parakeet hidden size must be divisible by its heads")
        if self.subsampling_factor & (self.subsampling_factor - 1):
            raise ValueError("Parakeet subsampling factor must be a power of two")
        if self.subsampling_conv_stride**3 != self.subsampling_factor:
            raise ValueError(
                "Parakeet subsampling factor must match its convolution stack"
            )
        if self.num_mel_bins % self.subsampling_factor:
            raise ValueError(
                "Parakeet mel bins must be divisible by its subsampling factor"
            )
        if self.conv_kernel_size % 2 != 1:
            raise ValueError("Parakeet Conformer convolution kernel must be odd")
        if self.attention_bias or self.convolution_bias or self.projection_bias:
            raise ValueError(
                "This Nemotron audio implementation requires bias-free attention, "
                "Conformer convolution, and sound projection"
            )
        if not math.isfinite(self.layer_norm_epsilon) or self.layer_norm_epsilon <= 0:
            raise ValueError("Nemotron audio normalization epsilon must be positive")


@dataclass(frozen=True)
class ParakeetFeatures:
    """Padded normalized log-mel frames and their boolean validity mask."""

    input_features: np.ndarray
    attention_mask: np.ndarray


def parakeet_subsampled_length(length: int, config: NemotronAudioConfig) -> int:
    """Return the temporal size after Parakeet's strided Conv2D stack."""
    length = int(length)
    if length < 0:
        raise ValueError("audio feature length cannot be negative")
    padding = (config.subsampling_conv_kernel_size - 1) // 2
    for _ in range(config.subsampling_layers):
        length = (
            length + 2 * padding - config.subsampling_conv_kernel_size
        ) // config.subsampling_conv_stride + 1
    return length


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    # Slaney's piecewise mel scale, matching librosa's default.
    frequencies = np.asarray(frequencies, dtype=np.float64)
    mel = frequencies / (200.0 / 3.0)
    logarithmic = frequencies >= 1000.0
    mel[logarithmic] = 15.0 + np.log(frequencies[logarithmic] / 1000.0) / (
        np.log(6.4) / 27.0
    )
    return mel


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    frequencies = (200.0 / 3.0) * mels
    logarithmic = mels >= 15.0
    frequencies[logarithmic] = 1000.0 * np.exp(
        (np.log(6.4) / 27.0) * (mels[logarithmic] - 15.0)
    )
    return frequencies


def parakeet_mel_filter_bank(
    *, sample_rate: int = 16_000, n_fft: int = 512, num_mel_bins: int = 128
) -> np.ndarray:
    """Construct librosa-compatible Slaney-normalized triangular mel filters."""
    if sample_rate <= 0 or n_fft <= 0 or num_mel_bins <= 0:
        raise ValueError("mel filter-bank dimensions must be positive")
    fft_frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    mel_edges = np.linspace(
        _hz_to_mel(np.array([0.0]))[0],
        _hz_to_mel(np.array([sample_rate / 2.0]))[0],
        num_mel_bins + 2,
    )
    hz_edges = _mel_to_hz(mel_edges)
    ramps = hz_edges[:, None] - fft_frequencies[None, :]
    filters = np.maximum(
        0.0,
        np.minimum(
            -ramps[:-2] / np.diff(hz_edges)[:-1, None],
            ramps[2:] / np.diff(hz_edges)[1:, None],
        ),
    )
    filters *= (2.0 / (hz_edges[2:] - hz_edges[:-2]))[:, None]
    return np.ascontiguousarray(filters, dtype=np.float32)


def preprocess_parakeet_audio(
    waveforms: np.ndarray | Sequence[np.ndarray],
    *,
    sample_rate: int = 16_000,
    num_mel_bins: int = 128,
    hop_length: int = 160,
    n_fft: int = 512,
    win_length: int = 400,
    preemphasis: float = 0.97,
) -> ParakeetFeatures:
    """Convert one or more mono waveforms to normalized Parakeet log-mel input.

    The STFT, Slaney mel filters, log guard, mask, and sample-variance
    normalization follow the official Transformers Parakeet feature extractor.
    """
    if sample_rate != 16_000:
        raise ValueError("Nemotron Omni audio must be sampled at 16000 Hz")
    if isinstance(waveforms, np.ndarray) and waveforms.ndim == 1:
        clips = [waveforms]
    elif isinstance(waveforms, np.ndarray) and waveforms.ndim == 2:
        clips = [row for row in waveforms]
    else:
        clips = list(waveforms)
    if not clips:
        raise ValueError("at least one audio waveform is required")
    values = []
    for clip in clips:
        value = np.asarray(clip)
        if value.ndim != 1:
            raise ValueError("Parakeet audio waveforms must be mono rank-one arrays")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError("Parakeet audio waveforms must use a floating dtype")
        value = np.asarray(value, dtype=np.float32)
        if value.size < 2 * hop_length or not np.isfinite(value).all():
            raise ValueError(
                "Parakeet waveforms must be finite and at least 20 ms long"
            )
        values.append(value)

    maximum = max(value.size for value in values)
    padded = np.zeros((len(values), maximum), dtype=np.float32)
    lengths = np.asarray([value.size for value in values], dtype=np.int64)
    for index, value in enumerate(values):
        padded[index, : value.size] = value
    emphasized = np.concatenate(
        [padded[:, :1], padded[:, 1:] - preemphasis * padded[:, :-1]], axis=1
    )
    emphasized[np.arange(maximum)[None, :] >= lengths[:, None]] = 0.0

    centered = np.pad(emphasized, ((0, 0), (n_fft // 2, n_fft // 2)))
    frames = np.lib.stride_tricks.sliding_window_view(centered, n_fft, axis=1)
    frames = frames[:, ::hop_length]
    window = np.zeros(n_fft, dtype=np.float32)
    offset = (n_fft - win_length) // 2
    window[offset : offset + win_length] = np.hanning(win_length).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=-1).astype(np.complex64)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(
        np.float32
    )
    filters = parakeet_mel_filter_bank(
        sample_rate=sample_rate, n_fft=n_fft, num_mel_bins=num_mel_bins
    )
    features = np.log(power @ filters.T + _LOG_ZERO_GUARD).astype(np.float32)

    feature_lengths = lengths // hop_length
    mask = np.arange(features.shape[1])[None, :] < feature_lengths[:, None]
    masked = features * mask[:, :, None]
    mean = masked.sum(axis=1) / feature_lengths[:, None]
    centered_features = features - mean[:, None, :]
    variance = (centered_features * centered_features * mask[:, :, None]).sum(
        axis=1
    ) / (feature_lengths - 1)[:, None]
    features = centered_features / (np.sqrt(variance)[:, None, :] + _FEATURE_EPSILON)
    features *= mask[:, :, None]
    return ParakeetFeatures(
        np.ascontiguousarray(features, dtype=np.float32),
        np.ascontiguousarray(mask, dtype=np.bool_),
    )


def preprocess_parakeet_wav(path: str | Path) -> ParakeetFeatures:
    """Read a shared-format PCM16 WAV and prepare its mono 16-kHz features."""
    audio = read_wav_pcm16(path)
    if audio.sample_rate != 16_000:
        raise ValueError("Nemotron Omni WAV input must use a 16000 Hz sample rate")
    mono = audio.samples.mean(axis=1, dtype=np.float32)
    return preprocess_parakeet_audio(mono, sample_rate=audio.sample_rate)


def parakeet_weight_names(config: NemotronAudioConfig) -> tuple[str, ...]:
    """Return every inference tensor required by the audio slice."""
    prefix = "sound_encoder.encoder."
    names = []
    convolution_indices = [0]
    for layer in range(1, config.subsampling_layers):
        convolution_indices.extend((3 * layer - 1, 3 * layer))
    for index in convolution_indices:
        names.extend(
            (
                f"{prefix}subsampling.layers.{index}.weight",
                f"{prefix}subsampling.layers.{index}.bias",
            )
        )
    names.extend(
        (f"{prefix}subsampling.linear.weight", f"{prefix}subsampling.linear.bias")
    )
    for index in range(config.num_hidden_layers):
        block = f"{prefix}layers.{index}."
        for feed_forward in ("feed_forward1", "feed_forward2"):
            names.extend(
                f"{block}{feed_forward}.{linear}.weight"
                for linear in ("linear1", "linear2")
            )
        names.extend(
            f"{block}self_attn.{name}"
            for name in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "relative_k_proj.weight",
                "bias_u",
                "bias_v",
            )
        )
        names.extend(
            f"{block}conv.{name}"
            for name in (
                "pointwise_conv1.weight",
                "depthwise_conv.weight",
                "norm.weight",
                "norm.bias",
                "norm.running_mean",
                "norm.running_var",
                "pointwise_conv2.weight",
            )
        )
        for norm in (
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ):
            names.extend((f"{block}{norm}.weight", f"{block}{norm}.bias"))
    names.extend(
        (
            "sound_projection.norm.weight",
            "sound_projection.linear1.weight",
            "sound_projection.linear2.weight",
        )
    )
    return tuple(names)


REQUIRED_SHARED_OPERATOR_APIS = (
    "GroupedConv2dPlan(x, weight, bias, stride, padding, groups)",
    "GroupedConv1dPlan(x, weight, bias, stride, padding, groups)",
    "RelativeBidirectionalAttentionPlan(query, key, value, relative_key, bias_u, bias_v, valid)",
)
"""Minimal generic primitives still needed for the exact Parakeet encoder."""


@wp.kernel(enable_backward=False, module="unique")
def _relu_squared(x: wp.array2d(dtype=wp.bfloat16)):
    row, column = wp.tid()
    value = wp.max(wp.float32(x[row, column]), 0.0)
    x[row, column] = wp.bfloat16(value * value)


class SoundProjectionPlan:
    """Fixed-shape exact Nemotron sound projection for BF16 encoder output."""

    def __init__(self, x, weights, *, epsilon=1.0e-5, cublas=None):
        if x.ndim != 3 or x.dtype != wp.bfloat16:
            raise TypeError("Nemotron sound projection requires rank-three BF16 input")
        self.input = x
        self.norm = RMSNormPlan(
            x, weights["sound_projection.norm.weight"], epsilon=epsilon
        )
        rows, hidden = x.shape[0] * x.shape[1], x.shape[2]
        self._tensors = {
            "normalized": self.norm.output.reshape((rows, hidden)),
            "linear1.weight": weights["sound_projection.linear1.weight"],
            "linear2.weight": weights["sound_projection.linear2.weight"],
        }
        self._shapes = {name: value.shape for name, value in self._tensors.items()}
        self._linear1 = Operation(
            "Linear", ["normalized", "linear1.weight"], ["hidden"]
        )
        plan_linear(self._linear1, self._tensors, self._shapes, x.device, cublas=cublas)
        self._linear2 = Operation("Linear", ["hidden", "linear2.weight"], ["projected"])
        plan_linear(self._linear2, self._tensors, self._shapes, x.device, cublas=cublas)
        self.output = self._tensors["projected"].reshape((x.shape[0], x.shape[1], -1))

    def execute(self):
        self.norm.execute()
        execute_operations(
            (self._linear1,), self._tensors, self._shapes, self.input.device
        )
        hidden = self._tensors["hidden"]
        wp.launch(
            _relu_squared, dim=hidden.shape, inputs=[hidden], device=self.input.device
        )
        execute_operations(
            (self._linear2,), self._tensors, self._shapes, self.input.device
        )
        return self.output


@lru_cache(maxsize=None)
def _audio_kernels(dtype):
    """Create the small elementwise kernels around reusable dense/conv plans."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def upload_features(
        source: wp.array3d(dtype=wp.float32),
        valid: wp.array2d(dtype=wp.bool),
        output: wp.array4d(dtype=DTYPE),
    ):
        batch, token, mel = wp.tid()
        value = wp.where(valid[batch, token], source[batch, token, mel], 0.0)
        output[batch, token, mel, 0] = DTYPE(value)

    @wp.kernel(enable_backward=False, module="unique")
    def relu_mask_4d(values: wp.array4d(dtype=DTYPE), valid: wp.array2d(dtype=wp.bool)):
        batch, token, frequency, channel = wp.tid()
        value = wp.float32(values[batch, token, frequency, channel])
        values[batch, token, frequency, channel] = wp.where(
            valid[batch, token], DTYPE(wp.max(value, 0.0)), DTYPE(0.0)
        )

    @wp.kernel(enable_backward=False, module="unique")
    def flatten_subsampling(
        values: wp.array4d(dtype=DTYPE), output: wp.array3d(dtype=DTYPE)
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        batch, token, column = wp.tid()
        frequency = column % values.shape[2]
        channel = column // values.shape[2]
        output[batch, token, column] = values[batch, token, frequency, channel]

    @wp.kernel(enable_backward=False, module="unique")
    def bias_mask_3d(
        values: wp.array3d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        valid: wp.array2d(dtype=wp.bool),
    ):
        batch, token, channel = wp.tid()
        value = wp.float32(values[batch, token, channel]) + wp.float32(bias[channel])
        values[batch, token, channel] = wp.where(
            valid[batch, token], DTYPE(value), DTYPE(0.0)
        )

    @wp.kernel(enable_backward=False, module="unique")
    def affine_3d(
        values: wp.array3d(dtype=DTYPE),
        weight: wp.array1d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
    ):
        batch, token, channel = wp.tid()
        values[batch, token, channel] = DTYPE(
            wp.float32(values[batch, token, channel]) * wp.float32(weight[channel])
            + wp.float32(bias[channel])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def silu_3d(values: wp.array3d(dtype=DTYPE)):
        batch, token, channel = wp.tid()
        value = wp.float32(values[batch, token, channel])
        values[batch, token, channel] = DTYPE(value / (1.0 + wp.exp(-value)))

    @wp.kernel(enable_backward=False, module="unique")
    def scaled_residual(
        residual: wp.array3d(dtype=DTYPE),
        branch: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        scale: wp.float32,
    ):
        batch, token, channel = wp.tid()
        output[batch, token, channel] = DTYPE(
            wp.float32(residual[batch, token, channel])
            + scale * wp.float32(branch[batch, token, channel])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def glu_mask(
        values: wp.array3d(dtype=DTYPE),
        valid: wp.array2d(dtype=wp.bool),
        output: wp.array3d(dtype=DTYPE),
    ):
        batch, token, channel = wp.tid()
        left = wp.float32(values[batch, token, channel])
        right = wp.float32(values[batch, token, channel + output.shape[2]])
        gated = left / (1.0 + wp.exp(-right))
        output[batch, token, channel] = wp.where(
            valid[batch, token], DTYPE(gated), DTYPE(0.0)
        )

    @wp.kernel(enable_backward=False, module="unique")
    def batch_norm_silu_mask(
        values: wp.array3d(dtype=DTYPE),
        weight: wp.array1d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        running_mean: wp.array1d(dtype=DTYPE),
        running_var: wp.array1d(dtype=DTYPE),
        valid: wp.array2d(dtype=wp.bool),
        epsilon: wp.float32,
    ):
        batch, token, channel = wp.tid()
        value = (
            wp.float32(values[batch, token, channel])
            - wp.float32(running_mean[channel])
        ) / wp.sqrt(wp.float32(running_var[channel]) + epsilon) * wp.float32(
            weight[channel]
        ) + wp.float32(bias[channel])
        activated = value / (1.0 + wp.exp(-value))
        values[batch, token, channel] = wp.where(
            valid[batch, token], DTYPE(activated), DTYPE(0.0)
        )

    @wp.kernel(enable_backward=False, module="unique")
    def mask_3d(values: wp.array3d(dtype=DTYPE), valid: wp.array2d(dtype=wp.bool)):
        batch, token, channel = wp.tid()
        if not valid[batch, token]:
            values[batch, token, channel] = DTYPE(0.0)

    return (
        upload_features,
        relu_mask_4d,
        flatten_subsampling,
        bias_mask_3d,
        affine_3d,
        silu_3d,
        scaled_residual,
        glu_mask,
        batch_norm_silu_mask,
        mask_3d,
    )


class _Linear3d:
    """Thin rank-three view over the shared optimized rank-two Linear planner."""

    def __init__(self, x, weight, cublas=None):
        rows, hidden = x.shape[0] * x.shape[1], x.shape[2]
        self.tensors = {"x": x.reshape((rows, hidden)), "weight": weight}
        self.shapes = {name: value.shape for name, value in self.tensors.items()}
        self.operation = Operation("Linear", ["x", "weight"], ["output"])
        plan_linear(self.operation, self.tensors, self.shapes, x.device, cublas=cublas)
        self.output = self.tensors["output"].reshape(
            (x.shape[0], x.shape[1], weight.shape[0])
        )

    def execute(self):
        execute_operations(
            (self.operation,), self.tensors, self.shapes, self.output.device
        )
        return self.output


class _AffineLayerNorm:
    def __init__(self, x, weight, bias, epsilon):
        self.norm = LayerNormPlan(x, epsilon=epsilon)
        self.weight, self.bias = weight, bias
        self.output = self.norm.output

    def execute(self, kernel):
        self.norm.execute()
        wp.launch(
            kernel,
            dim=self.output.shape,
            inputs=[self.output, self.weight, self.bias],
            device=self.output.device,
        )
        return self.output


def _relative_sinusoid(sequence: int, hidden: int) -> np.ndarray:
    positions = np.arange(sequence - 1, -sequence, -1, dtype=np.float32)
    dimensions = np.arange(0, hidden, 2, dtype=np.float32)
    inverse = np.power(np.float32(10_000.0), -dimensions / hidden)
    angles = positions[:, None] * inverse[None, :]
    output = np.empty((1, 2 * sequence - 1, hidden), dtype=np.float32)
    output[0, :, 0::2] = np.sin(angles)
    output[0, :, 1::2] = np.cos(angles)
    return output


class _ConformerBlock:
    def __init__(self, x, valid, relative, weights, prefix, config, cublas):
        self.x, self.valid, self.weights, self.prefix = x, valid, weights, prefix
        epsilon = config.layer_norm_epsilon
        heads, hidden = config.num_attention_heads, config.hidden_size

        self.norm_ff1 = _AffineLayerNorm(
            x,
            weights[prefix + "norm_feed_forward1.weight"],
            weights[prefix + "norm_feed_forward1.bias"],
            epsilon,
        )
        self.ff1_up = _Linear3d(
            self.norm_ff1.output,
            weights[prefix + "feed_forward1.linear1.weight"],
            cublas,
        )
        self.ff1_down = _Linear3d(
            self.ff1_up.output, weights[prefix + "feed_forward1.linear2.weight"], cublas
        )
        self.after_ff1 = wp.empty_like(x)

        self.norm_attention = _AffineLayerNorm(
            self.after_ff1,
            weights[prefix + "norm_self_att.weight"],
            weights[prefix + "norm_self_att.bias"],
            epsilon,
        )
        normalized = self.norm_attention.output
        self.q = _Linear3d(
            normalized, weights[prefix + "self_attn.q_proj.weight"], cublas
        )
        self.k = _Linear3d(
            normalized, weights[prefix + "self_attn.k_proj.weight"], cublas
        )
        self.v = _Linear3d(
            normalized, weights[prefix + "self_attn.v_proj.weight"], cublas
        )
        self.q_heads = AttentionHeadsPlan(self.q.output, heads)
        self.k_heads = AttentionHeadsPlan(self.k.output, heads)
        self.v_heads = AttentionHeadsPlan(self.v.output, heads)
        self.relative_projection = _Linear3d(
            relative, weights[prefix + "self_attn.relative_k_proj.weight"], cublas
        )
        self.relative_heads = AttentionHeadsPlan(self.relative_projection.output, heads)
        self.attention = RelativeBidirectionalAttentionPlan(
            self.q_heads.output,
            self.k_heads.output,
            self.v_heads.output,
            self.relative_heads.output,
            weights[prefix + "self_attn.bias_u"],
            weights[prefix + "self_attn.bias_v"],
            valid=valid,
        )
        self.merge = AttentionMergePlan(self.attention.output)
        self.attention_output = _Linear3d(
            self.merge.output, weights[prefix + "self_attn.o_proj.weight"], cublas
        )
        self.after_attention = wp.empty_like(x)

        self.norm_conv = _AffineLayerNorm(
            self.after_attention,
            weights[prefix + "norm_conv.weight"],
            weights[prefix + "norm_conv.bias"],
            epsilon,
        )
        self.conv_pointwise1 = Conv1dPlan(
            self.norm_conv.output,
            weights[prefix + "conv.pointwise_conv1.weight"],
        )
        self.glu = wp.empty_like(x)
        self.conv_depthwise = Conv1dPlan(
            self.glu,
            weights[prefix + "conv.depthwise_conv.weight"],
            padding=config.conv_kernel_size // 2,
            groups=hidden,
        )
        self.conv_pointwise2 = Conv1dPlan(
            self.conv_depthwise.output,
            weights[prefix + "conv.pointwise_conv2.weight"],
        )
        self.after_conv = wp.empty_like(x)

        self.norm_ff2 = _AffineLayerNorm(
            self.after_conv,
            weights[prefix + "norm_feed_forward2.weight"],
            weights[prefix + "norm_feed_forward2.bias"],
            epsilon,
        )
        self.ff2_up = _Linear3d(
            self.norm_ff2.output,
            weights[prefix + "feed_forward2.linear1.weight"],
            cublas,
        )
        self.ff2_down = _Linear3d(
            self.ff2_up.output, weights[prefix + "feed_forward2.linear2.weight"], cublas
        )
        self.after_ff2 = wp.empty_like(x)
        self.norm_out = _AffineLayerNorm(
            self.after_ff2,
            weights[prefix + "norm_out.weight"],
            weights[prefix + "norm_out.bias"],
            epsilon,
        )
        self.output = self.norm_out.output

    def prepare_relative(self):
        self.relative_projection.execute()
        self.relative_heads.execute()

    def execute(self, kernels, epsilon):
        _, _, _, _, affine, silu, residual, glu, batch_norm_silu, mask = kernels
        p, w = self.prefix, self.weights
        self.norm_ff1.execute(affine)
        self.ff1_up.execute()
        wp.launch(silu, dim=self.ff1_up.output.shape, inputs=[self.ff1_up.output])
        self.ff1_down.execute()
        wp.launch(
            residual,
            dim=self.after_ff1.shape,
            inputs=[self.x, self.ff1_down.output, self.after_ff1, wp.float32(0.5)],
        )

        self.norm_attention.execute(affine)
        self.q.execute()
        self.k.execute()
        self.v.execute()
        self.q_heads.execute()
        self.k_heads.execute()
        self.v_heads.execute()
        self.attention.execute()
        self.merge.execute()
        self.attention_output.execute()
        wp.launch(
            residual,
            dim=self.after_attention.shape,
            inputs=[
                self.after_ff1,
                self.attention_output.output,
                self.after_attention,
                wp.float32(1.0),
            ],
        )

        self.norm_conv.execute(affine)
        self.conv_pointwise1.execute()
        wp.launch(
            glu,
            dim=self.glu.shape,
            inputs=[self.conv_pointwise1.output, self.valid, self.glu],
        )
        self.conv_depthwise.execute()
        wp.launch(
            batch_norm_silu,
            dim=self.conv_depthwise.output.shape,
            inputs=[
                self.conv_depthwise.output,
                w[p + "conv.norm.weight"],
                w[p + "conv.norm.bias"],
                w[p + "conv.norm.running_mean"],
                w[p + "conv.norm.running_var"],
                self.valid,
                wp.float32(epsilon),
            ],
        )
        self.conv_pointwise2.execute()
        wp.launch(
            mask,
            dim=self.conv_pointwise2.output.shape,
            inputs=[self.conv_pointwise2.output, self.valid],
        )
        wp.launch(
            residual,
            dim=self.after_conv.shape,
            inputs=[
                self.after_attention,
                self.conv_pointwise2.output,
                self.after_conv,
                wp.float32(1.0),
            ],
        )

        self.norm_ff2.execute(affine)
        self.ff2_up.execute()
        wp.launch(silu, dim=self.ff2_up.output.shape, inputs=[self.ff2_up.output])
        self.ff2_down.execute()
        wp.launch(
            residual,
            dim=self.after_ff2.shape,
            inputs=[
                self.after_conv,
                self.ff2_down.output,
                self.after_ff2,
                wp.float32(0.5),
            ],
        )
        self.norm_out.execute(affine)
        return self.output


class _AudioPlan:
    def __init__(self, encoder: "NemotronAudioEncoder", shape):
        batch, frames, mel_bins = shape
        config = encoder.config
        self.encoder = encoder
        self.features = wp.empty(shape, dtype=wp.float32, device=encoder.device)
        self.masks = []
        length = frames
        self.masks.append(
            wp.empty((batch, length), dtype=wp.bool, device=encoder.device)
        )
        for _ in range(config.subsampling_layers):
            length = (
                length + config.subsampling_conv_stride - 1
            ) // config.subsampling_conv_stride
            self.masks.append(
                wp.empty((batch, length), dtype=wp.bool, device=encoder.device)
            )

        self.image = wp.empty(
            (batch, frames, mel_bins, 1), dtype=encoder.dtype, device=encoder.device
        )
        p = "sound_encoder.encoder.subsampling."
        w = encoder.weights
        self.conv0 = Conv2dPlan(
            self.image,
            w[p + "layers.0.weight"],
            w[p + "layers.0.bias"],
            stride=config.subsampling_conv_stride,
            padding=config.subsampling_conv_kernel_size // 2,
        )
        self.conv1_depthwise = Conv2dPlan(
            self.conv0.output,
            w[p + "layers.2.weight"],
            w[p + "layers.2.bias"],
            stride=config.subsampling_conv_stride,
            padding=config.subsampling_conv_kernel_size // 2,
            groups=config.subsampling_conv_channels,
        )
        self.conv1_pointwise = Conv2dPlan(
            self.conv1_depthwise.output,
            w[p + "layers.3.weight"],
            w[p + "layers.3.bias"],
        )
        self.conv2_depthwise = Conv2dPlan(
            self.conv1_pointwise.output,
            w[p + "layers.5.weight"],
            w[p + "layers.5.bias"],
            stride=config.subsampling_conv_stride,
            padding=config.subsampling_conv_kernel_size // 2,
            groups=config.subsampling_conv_channels,
        )
        self.conv2_pointwise = Conv2dPlan(
            self.conv2_depthwise.output,
            w[p + "layers.6.weight"],
            w[p + "layers.6.bias"],
        )
        final = self.conv2_pointwise.output
        flattened_width = final.shape[2] * final.shape[3]
        self.flattened = wp.empty(
            (batch, final.shape[1], flattened_width),
            dtype=encoder.dtype,
            device=encoder.device,
        )
        self.subsampling_linear = _Linear3d(
            self.flattened, w[p + "linear.weight"], encoder.cublas
        )

        relative = wp.array(
            _relative_sinusoid(final.shape[1], config.hidden_size),
            dtype=encoder.dtype,
            device=encoder.device,
        )
        self.blocks = []
        current = self.subsampling_linear.output
        for index in range(config.num_hidden_layers):
            block = _ConformerBlock(
                current,
                self.masks[-1],
                relative,
                w,
                f"sound_encoder.encoder.layers.{index}.",
                config,
                encoder.cublas,
            )
            block.prepare_relative()
            self.blocks.append(block)
            current = block.output
        self.projection = SoundProjectionPlan(
            current,
            w,
            epsilon=config.layer_norm_epsilon,
            cublas=encoder.cublas,
        )
        self.output = self.projection.output
        self.graph = None
        self._capture_ready = False

    def execute(self):
        kernels = self.encoder.kernels
        upload, relu4, flatten, bias3, *_ = kernels
        wp.launch(
            upload,
            dim=self.features.shape,
            inputs=[self.features, self.masks[0], self.image],
            device=self.encoder.device,
        )
        self.conv0.execute()
        wp.launch(
            relu4,
            dim=self.conv0.output.shape,
            inputs=[self.conv0.output, self.masks[1]],
        )
        self.conv1_depthwise.execute()
        self.conv1_pointwise.execute()
        wp.launch(
            relu4,
            dim=self.conv1_pointwise.output.shape,
            inputs=[self.conv1_pointwise.output, self.masks[2]],
        )
        self.conv2_depthwise.execute()
        self.conv2_pointwise.execute()
        wp.launch(
            relu4,
            dim=self.conv2_pointwise.output.shape,
            inputs=[self.conv2_pointwise.output, self.masks[3]],
        )
        wp.launch(
            flatten,
            dim=self.flattened.shape,
            inputs=[self.conv2_pointwise.output, self.flattened],
        )
        self.subsampling_linear.execute()
        wp.launch(
            bias3,
            dim=self.subsampling_linear.output.shape,
            inputs=[
                self.subsampling_linear.output,
                self.encoder.weights["sound_encoder.encoder.subsampling.linear.bias"],
                self.masks[-1],
            ],
        )
        for block in self.blocks:
            block.execute(kernels, self.encoder.config.layer_norm_epsilon)
        self.projection.execute()
        return self.output

    def run(self):
        if not self.encoder.device.is_cuda:
            return self.execute()
        if not self._capture_ready:
            self._capture_ready = True
            return self.execute()
        if self.graph is None:
            wp.capture_begin(device=self.encoder.device)
            self.output = self.execute()
            self.graph = wp.capture_end(device=self.encoder.device)
        wp.capture_launch(self.graph)
        return self.output


class NemotronAudioEncoder:
    """Lazy graph-captured Parakeet encoder and projection for Nemotron Omni."""

    def __init__(self, path: str | Path, *, device=None, cublas=None):
        self.path = Path(path)
        self.config = NemotronAudioConfig.from_path(self.path)
        self.device = wp.get_device(device)
        self.cublas = cublas
        archive = SafeTensorArchive(self.path)
        names = parakeet_weight_names(self.config)
        missing = set(names) - set(archive.names)
        if missing:
            raise ValueError(
                f"Nemotron audio checkpoint is missing {sorted(missing)[:5]}"
            )
        self.dtype = archive.metadata(names[0]).dtype
        if self.dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Nemotron audio weights must be FP16 or BF16")
        self.weights = archive.load(self.device, names)
        self.kernels = _audio_kernels(self.dtype)
        self._plans = {}

    def preprocess(self, audio) -> ParakeetFeatures:
        if isinstance(audio, (str, Path)):
            return preprocess_parakeet_wav(audio)
        return preprocess_parakeet_audio(audio, sample_rate=self.config.sampling_rate)

    @staticmethod
    def _subsample_masks(mask, config):
        masks = [np.ascontiguousarray(mask, dtype=np.bool_)]
        lengths = masks[0].sum(axis=1, dtype=np.int64)
        for _ in range(config.subsampling_layers):
            lengths = (
                lengths + config.subsampling_conv_stride - 1
            ) // config.subsampling_conv_stride
            width = (
                masks[-1].shape[1] + config.subsampling_conv_stride - 1
            ) // config.subsampling_conv_stride
            masks.append(np.arange(width)[None, :] < lengths[:, None])
        return masks

    def encode(self, audio) -> wp.array:
        features = (
            audio if isinstance(audio, ParakeetFeatures) else self.preprocess(audio)
        )
        plan = self._plans.get(features.input_features.shape)
        if plan is None:
            plan = self._plans[features.input_features.shape] = _AudioPlan(
                self, features.input_features.shape
            )
        masks = self._subsample_masks(features.attention_mask, self.config)
        if any(
            source.shape != target.shape for source, target in zip(masks, plan.masks)
        ):
            raise ValueError("audio validity masks do not match the encoder plan")
        plan.features.assign(features.input_features)
        for source, target in zip(masks, plan.masks):
            target.assign(source)
        return plan.run()
