# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image-2512 still-image VAE decoder checkpoint preparation.

The released VAE is a causal video autoencoder, but text-to-image decoding has
exactly one temporal sample. In that case a causal OITHW convolution only uses
its final temporal plane. Temporal convolutions inside the two nominal 3D
upsamplers are never called for the first (and only) frame and are intentionally
absent from this decoder-only contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warp as wp


from ..weights import extract_temporal_conv2d_weight
from .runner import QwenImageVAEConfig


@dataclass(frozen=True)
class QwenImageVAEWeightSpec:
    """One selected checkpoint tensor and its still-image runtime shape."""

    name: str
    source_shape: tuple[int, ...]
    prepared_shape: tuple[int, ...]
    temporal_index: int | None = None

    @property
    def is_causal_conv_weight(self) -> bool:
        return self.temporal_index is not None


def _vector(specs, name, channels, *, source_rank=1):
    source_shape = (channels,) + (1,) * (source_rank - 1)
    specs.append(QwenImageVAEWeightSpec(name, source_shape, (channels,)))


def _causal_conv(specs, prefix, in_channels, out_channels, kernel=(3, 3, 3)):
    temporal, height, width = kernel
    specs.append(
        QwenImageVAEWeightSpec(
            f"{prefix}.weight",
            (out_channels, in_channels, temporal, height, width),
            (out_channels, in_channels, height, width),
            temporal - 1,
        )
    )
    _vector(specs, f"{prefix}.bias", out_channels)


def _conv2d(specs, prefix, in_channels, out_channels, kernel=1):
    specs.append(
        QwenImageVAEWeightSpec(
            f"{prefix}.weight",
            (out_channels, in_channels, kernel, kernel),
            (out_channels, in_channels, kernel, kernel),
        )
    )
    _vector(specs, f"{prefix}.bias", out_channels)


def _residual(specs, prefix, in_channels, out_channels):
    _vector(specs, f"{prefix}.norm1.gamma", in_channels, source_rank=4)
    _causal_conv(specs, f"{prefix}.conv1", in_channels, out_channels)
    _vector(specs, f"{prefix}.norm2.gamma", out_channels, source_rank=4)
    _causal_conv(specs, f"{prefix}.conv2", out_channels, out_channels)
    if in_channels != out_channels:
        _causal_conv(
            specs,
            f"{prefix}.conv_shortcut",
            in_channels,
            out_channels,
            kernel=(1, 1, 1),
        )


def _mid_block(specs, prefix, channels):
    _residual(specs, f"{prefix}.resnets.0", channels, channels)
    _vector(specs, f"{prefix}.attentions.0.norm.gamma", channels, source_rank=3)
    _conv2d(specs, f"{prefix}.attentions.0.to_qkv", channels, 3 * channels)
    _conv2d(specs, f"{prefix}.attentions.0.proj", channels, channels)
    _residual(specs, f"{prefix}.resnets.1", channels, channels)


def _qwen_image_vae_decoder_weight_specs(
    config: QwenImageVAEConfig,
) -> tuple[QwenImageVAEWeightSpec, ...]:
    """Build a still-image contract for private synthetic test geometry."""

    specs: list[QwenImageVAEWeightSpec] = []
    base = config.base_dim
    latent = config.latent_channels
    channels = tuple(base * multiplier for multiplier in config.dimension_multipliers)

    _causal_conv(specs, "post_quant_conv", latent, latent, kernel=(1, 1, 1))
    _causal_conv(specs, "decoder.conv_in", latent, channels[-1])
    _mid_block(specs, "decoder.mid_block", channels[-1])

    decoder_dims = (channels[-1], *channels[::-1])
    block_inputs = tuple(
        value if index == 0 else value // 2
        for index, value in enumerate(decoder_dims[:-1])
    )
    block_outputs = decoder_dims[1:]
    for block, (in_channels, out_channels) in enumerate(
        zip(block_inputs, block_outputs)
    ):
        for residual in range(config.residual_blocks + 1):
            _residual(
                specs,
                f"decoder.up_blocks.{block}.resnets.{residual}",
                in_channels if residual == 0 else out_channels,
                out_channels,
            )
        if block < len(block_outputs) - 1:
            # Both reference upsamplers apply this spatial convolution. The
            # time_conv in blocks 0/1 is not executed for the sole first frame.
            _conv2d(
                specs,
                f"decoder.up_blocks.{block}.upsamplers.0.resample.1",
                out_channels,
                out_channels // 2,
                kernel=3,
            )

    _vector(specs, "decoder.norm_out.gamma", channels[0], source_rank=4)
    _causal_conv(specs, "decoder.conv_out", channels[0], 3)
    return tuple(specs)


def qwen_image_2512_vae_decoder_weight_specs(
    config: QwenImageVAEConfig,
) -> tuple[QwenImageVAEWeightSpec, ...]:
    """Return the exact selected Diffusers state-dict contract for decoding."""

    expected = (96, (1, 2, 4, 4), 2, 16, (False, True, True))
    actual = (
        config.base_dim,
        config.dimension_multipliers,
        config.residual_blocks,
        config.latent_channels,
        config.temporal_downsample,
    )
    if actual != expected:
        raise ValueError("unsupported Qwen-Image-2512 VAE decoder geometry")
    return _qwen_image_vae_decoder_weight_specs(config)


def prepare_qwen_image_vae_decoder_weights(
    archive,
    specs: Iterable[QwenImageVAEWeightSpec],
    device=None,
) -> dict[str, object]:
    """Validate and prepare selected decoder tensors without rank-five arrays."""

    specs = tuple(specs)
    names = tuple(spec.name for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("duplicate Qwen-Image VAE decoder weight name")
    available = set(archive.names)
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"missing Qwen-Image VAE decoder weights: {missing}")
    for spec in specs:
        shape = tuple(int(value) for value in archive.metadata(spec.name).shape)
        if shape != spec.source_shape:
            raise ValueError(
                f"Qwen-Image VAE weight '{spec.name}' has shape {shape}, "
                f"expected {spec.source_shape}"
            )

    output = {}
    for spec in specs:
        source = archive.load(device, [spec.name], flatten=True)[spec.name]
        if spec.is_causal_conv_weight:
            prepared = extract_temporal_conv2d_weight(
                source, spec.source_shape, spec.temporal_index
            )
            if source.device.is_cuda:
                wp.synchronize_stream(wp.get_stream(source.device))
        else:
            prepared = source.reshape(spec.prepared_shape)
        output[spec.name] = prepared
    return output


def load_qwen_image_2512_vae_decoder_weights(archive, config, device=None):
    """Load the exact still-image decoder subset from a safetensors archive."""

    return prepare_qwen_image_vae_decoder_weights(
        archive, qwen_image_2512_vae_decoder_weight_specs(config), device
    )
