# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free ACE-Step 1.5 Oobleck VAE decoding."""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import warp as wp

from warp_nn.runtime.formats.safetensors import SafeTensorArchive
from warp_nn.runtime.operators import Conv1dPlan, Snake1dPlan
from warp_nn.runtime.weights import load_cast_weights


@dataclass(frozen=True)
class OobleckVAEConfig:
    """Shape and sampling metadata for an ``AutoencoderOobleck`` checkpoint."""

    encoder_hidden_size: int = 128
    downsampling_ratios: tuple[int, ...] = (2, 4, 4, 6, 10)
    channel_multiples: tuple[int, ...] = (1, 2, 4, 8, 16)
    decoder_channels: int = 128
    decoder_input_channels: int = 64
    audio_channels: int = 2
    sampling_rate: int = 48_000

    def __post_init__(self):
        if len(self.downsampling_ratios) != len(self.channel_multiples):
            raise ValueError(
                "Oobleck ratios and channel multiples must have equal length"
            )
        if not self.downsampling_ratios or any(
            x <= 0 for x in self.downsampling_ratios
        ):
            raise ValueError("Oobleck downsampling ratios must be positive")
        if any(x <= 0 for x in self.channel_multiples):
            raise ValueError("Oobleck channel multiples must be positive")
        if (
            min(
                self.encoder_hidden_size,
                self.decoder_channels,
                self.decoder_input_channels,
                self.audio_channels,
                self.sampling_rate,
            )
            <= 0
        ):
            raise ValueError("Oobleck dimensions and sampling rate must be positive")

    @property
    def hop_length(self) -> int:
        return math.prod(self.downsampling_ratios)

    @classmethod
    def from_dict(cls, config: dict) -> "OobleckVAEConfig":
        return cls(
            encoder_hidden_size=int(config.get("encoder_hidden_size", 128)),
            downsampling_ratios=tuple(
                config.get("downsampling_ratios", (2, 4, 4, 8, 8))
            ),
            channel_multiples=tuple(config.get("channel_multiples", (1, 2, 4, 8, 16))),
            decoder_channels=int(config.get("decoder_channels", 128)),
            decoder_input_channels=int(config.get("decoder_input_channels", 64)),
            audio_channels=int(config.get("audio_channels", 2)),
            sampling_rate=int(config.get("sampling_rate", 44_100)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "OobleckVAEConfig":
        path = Path(path)
        if path.is_dir():
            path /= "config.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@lru_cache(maxsize=None)
def _vae_kernels(dtype, source_dtype):
    DTYPE = dtype
    SOURCE_DTYPE = source_dtype

    @wp.kernel
    def add(
        lhs: wp.array3d(dtype=DTYPE),
        rhs: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
    ):
        i, j, k = wp.tid()
        output[i, j, k] = lhs[i, j, k] + rhs[i, j, k]

    @wp.kernel
    def weight_norm(
        vector: wp.array3d(dtype=SOURCE_DTYPE),
        norms: wp.array(dtype=wp.float32),
    ):
        channel = wp.tid()
        total = wp.float32(0.0)
        for inner in range(vector.shape[1]):
            for kernel in range(vector.shape[2]):
                value = wp.float32(vector[channel, inner, kernel])
                total += value * value
        norms[channel] = wp.sqrt(total)

    @wp.kernel
    def fuse_weight_norm(
        scale: wp.array3d(dtype=SOURCE_DTYPE),
        vector: wp.array3d(dtype=SOURCE_DTYPE),
        norms: wp.array(dtype=wp.float32),
        output: wp.array3d(dtype=DTYPE),
    ):
        channel, inner, kernel = wp.tid()
        multiplier = wp.float32(scale[channel, 0, 0]) / (
            norms[channel] + wp.float32(1.0e-9)
        )
        output[channel, inner, kernel] = DTYPE(
            wp.float32(vector[channel, inner, kernel]) * multiplier
        )

    return add, weight_norm, fuse_weight_norm


class _AddPlan:
    def __init__(self, lhs, rhs):
        if lhs.shape != rhs.shape:
            raise ValueError("Oobleck residual shapes do not match")
        self.lhs = lhs
        self.rhs = rhs
        self.output = wp.empty_like(lhs)
        self._kernel = _vae_kernels(lhs.dtype, lhs.dtype)[0]

    def execute(self):
        wp.launch(
            self._kernel,
            dim=self.output.shape,
            inputs=[self.lhs, self.rhs, self.output],
            device=self.output.device,
        )
        return self.output


class _ResidualUnitPlan:
    def __init__(self, x, weights, prefix, dilation):
        self._plans = []
        snake = Snake1dPlan(
            x, weights[f"{prefix}.snake1.alpha"], weights[f"{prefix}.snake1.beta"]
        )
        conv1 = Conv1dPlan(
            snake.output,
            weights[f"{prefix}.conv1.weight"],
            weights.get(f"{prefix}.conv1.bias"),
            padding=3 * dilation,
            dilation=dilation,
        )
        snake2 = Snake1dPlan(
            conv1.output,
            weights[f"{prefix}.snake2.alpha"],
            weights[f"{prefix}.snake2.beta"],
        )
        conv2 = Conv1dPlan(
            snake2.output,
            weights[f"{prefix}.conv2.weight"],
            weights.get(f"{prefix}.conv2.bias"),
        )
        add = _AddPlan(x, conv2.output)
        self._plans.extend((snake, conv1, snake2, conv2, add))
        self.output = add.output

    def execute(self):
        for plan in self._plans:
            plan.execute()
        return self.output


class _DecoderBlockPlan:
    def __init__(self, x, weights, prefix, stride):
        snake = Snake1dPlan(
            x, weights[f"{prefix}.snake1.alpha"], weights[f"{prefix}.snake1.beta"]
        )
        upsample = Conv1dPlan(
            snake.output,
            weights[f"{prefix}.conv_t1.weight"],
            weights.get(f"{prefix}.conv_t1.bias"),
            stride=stride,
            padding=math.ceil(stride / 2),
            transposed=True,
        )
        self._plans = [snake, upsample]
        output = upsample.output
        for index, dilation in enumerate((1, 3, 9), 1):
            unit = _ResidualUnitPlan(
                output, weights, f"{prefix}.res_unit{index}", dilation
            )
            self._plans.append(unit)
            output = unit.output
        self.output = output

    def execute(self):
        for plan in self._plans:
            plan.execute()
        return self.output


def _canonical_parameter_names(config: OobleckVAEConfig) -> tuple[str, ...]:
    names = ["decoder.conv1.weight", "decoder.conv1.bias"]
    for block in range(len(config.downsampling_ratios)):
        prefix = f"decoder.block.{block}"
        names.extend(
            (
                f"{prefix}.snake1.alpha",
                f"{prefix}.snake1.beta",
                f"{prefix}.conv_t1.weight",
                f"{prefix}.conv_t1.bias",
            )
        )
        for unit in range(1, 4):
            residual = f"{prefix}.res_unit{unit}"
            for layer in range(1, 3):
                names.extend(
                    (
                        f"{residual}.snake{layer}.alpha",
                        f"{residual}.snake{layer}.beta",
                        f"{residual}.conv{layer}.weight",
                        f"{residual}.conv{layer}.bias",
                    )
                )
    names.extend(
        ("decoder.snake1.alpha", "decoder.snake1.beta", "decoder.conv2.weight")
    )
    return tuple(names)


def _weight_norm_sources(archive, name):
    base = name[: -len(".weight")]
    candidates = (
        (
            f"{base}.parametrizations.weight.original0",
            f"{base}.parametrizations.weight.original1",
        ),
        (f"{base}.weight_g", f"{base}.weight_v"),
    )
    return next(
        (pair for pair in candidates if all(item in archive.names for item in pair)),
        None,
    )


def _load_decoder_weights(archive, config, device, dtype):
    weights = {}
    for name in _canonical_parameter_names(config):
        if name.endswith(".weight") and name not in archive.names:
            sources = _weight_norm_sources(archive, name)
            if sources is None:
                raise KeyError(f"Oobleck checkpoint has no weight for '{name}'")
            scale, vector = (
                archive.load(device, [source])[source] for source in sources
            )
            if scale.ndim == 1:
                scale = scale.reshape((scale.shape[0], 1, 1))
            output = wp.empty(vector.shape, dtype=dtype, device=device)
            norms = wp.empty(vector.shape[0], dtype=wp.float32, device=device)
            kernels = _vae_kernels(dtype, vector.dtype)
            wp.launch(
                kernels[1], dim=vector.shape[0], inputs=[vector, norms], device=device
            )
            wp.launch(
                kernels[2],
                dim=vector.shape,
                inputs=[scale, vector, norms, output],
                device=device,
            )
            weights[name] = output
        elif name in archive.names:
            value = load_cast_weights(archive, [name], device, dtype)[name]
            if name.endswith((".alpha", ".beta")):
                value = value.flatten()
            weights[name] = value
        elif not name.endswith(".bias"):
            raise KeyError(f"Oobleck checkpoint is missing '{name}'")
    if wp.get_device(device).is_cuda:
        wp.synchronize_stream(wp.get_stream(device))
    return weights


class OobleckVAEDecoder:
    """Fixed-length, CUDA-graph-safe ACE-Step Oobleck latent decoder.

    Arrays use channels-last ``[batch, frames, channels]`` storage internally.
    ``input`` is caller-owned and may be updated between calls with ``wp.copy``.
    """

    def __init__(
        self,
        config,
        weights,
        latent_frames,
        *,
        batch_size=1,
        device=None,
        dtype=wp.float16,
    ):
        self.config = config
        self.device = wp.get_device(device)
        self.input = wp.empty(
            (batch_size, latent_frames, config.decoder_input_channels),
            dtype=dtype,
            device=self.device,
        )
        multiples = (1, *config.channel_multiples)
        first = Conv1dPlan(
            self.input,
            weights["decoder.conv1.weight"],
            weights.get("decoder.conv1.bias"),
            padding=3,
        )
        self._plans = [first]
        output = first.output
        strides = tuple(reversed(config.downsampling_ratios))
        for index, stride in enumerate(strides):
            expected_channels = (
                config.decoder_channels * multiples[len(strides) - index]
            )
            if output.shape[2] != expected_channels:
                raise ValueError(
                    "Oobleck decoder channel layout does not match its config"
                )
            block = _DecoderBlockPlan(output, weights, f"decoder.block.{index}", stride)
            self._plans.append(block)
            output = block.output
        snake = Snake1dPlan(
            output, weights["decoder.snake1.alpha"], weights["decoder.snake1.beta"]
        )
        final = Conv1dPlan(snake.output, weights["decoder.conv2.weight"], padding=3)
        self._plans.extend((snake, final))
        self.output = final.output
        self.graph = None

    @classmethod
    def from_pretrained(
        cls,
        path,
        latent_frames,
        *,
        batch_size=1,
        device=None,
        dtype=wp.float16,
    ):
        path = Path(path)
        config = OobleckVAEConfig.from_file(path)
        archive_path = (
            path / "diffusion_pytorch_model.safetensors" if path.is_dir() else path
        )
        archive = SafeTensorArchive(archive_path)
        weights = _load_decoder_weights(archive, config, device, dtype)
        return cls(
            config,
            weights,
            latent_frames,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def execute(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            for plan in self._plans:
                plan.execute()
        return self.output

    def capture(self):
        """Capture repeated decode execution into one CUDA graph."""
        if not self.device.is_cuda:
            raise RuntimeError("CUDA graph capture requires a CUDA device")
        wp.capture_begin(device=self.device)
        for plan in self._plans:
            plan.execute()
        self.graph = wp.capture_end(device=self.device)
        return self.graph
