# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official-style overlap-tiling schedule for Qwen-Image VAE decoding."""

from dataclasses import dataclass

import warp as wp

from .vae import qwen_image_2512_vae_decoder_weight_specs
from .vae_decoder import _QwenImageVAEDecoderPlan


def _pair(value, label):
    values = (value, value) if isinstance(value, int) else tuple(value)
    if len(values) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in values
    ):
        raise ValueError(f"{label} must contain two positive integers")
    return values


@dataclass(frozen=True)
class QwenImageVAETilingConfig:
    """Latent-space geometry of the official practical tiling approximation."""

    tile: tuple[int, int] = (32, 32)
    stride: tuple[int, int] = (24, 24)

    @classmethod
    def create(cls, tile=32, stride=24):
        tile = _pair(tile, "VAE tile")
        stride = _pair(stride, "VAE tile stride")
        if any(step > extent for step, extent in zip(stride, tile)):
            raise ValueError("VAE tile stride cannot exceed its tile extent")
        return cls(tile, stride)

    def sample_geometry(self, scale):
        """Return sample-space ``(tile, stride, overlap)`` for a VAE scale."""
        if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
            raise ValueError("VAE spatial scale must be a positive integer")
        return tuple(
            tuple(value * scale for value in values)
            for values in (self.tile, self.stride, self.overlap)
        )

    @property
    def overlap(self):
        return tuple(extent - step for extent, step in zip(self.tile, self.stride))


def _tile_origins(length, stride):
    """Match Diffusers' ``range(0, length, stride)`` edge-tile schedule."""
    return tuple(range(0, int(length), int(stride)))


class _QwenImageVAETiledDecoderPlan:
    """Explicit opt-in official-style overlap-tiled VAE decoder.

    This reduces peak activation and attention cost but is an approximation:
    each latent tile is decoded independently and linearly blended over the
    overlap. Exact untiled decoding remains the default public mode.
    """

    approximate = True
    mode = "official_overlap_tiled"

    def __init__(
        self,
        config,
        weights,
        latent_height,
        latent_width,
        *,
        batch_size=1,
        tile=32,
        stride=24,
    ):
        # Imported lazily so metadata/checkpoint use does not require this
        # optional execution primitive during package discovery.
        from ..operators import OverlapTileBlendPlan

        self.config = config
        self.tiling = QwenImageVAETilingConfig.create(tile, stride)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        if min(int(batch_size), self.latent_height, self.latent_width) <= 0:
            raise ValueError("Qwen-Image tiled decode geometry must be positive")
        first_weight = weights["post_quant_conv.weight"]
        self.device, self.dtype = first_weight.device, first_weight.dtype
        self.input = wp.empty(
            (
                int(batch_size),
                config.latent_channels,
                self.latent_height,
                self.latent_width,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        scale = config.spatial_scale_factor
        sample_height = self.latent_height * scale
        sample_width = self.latent_width * scale
        self.output = wp.empty(
            (int(batch_size), 3, sample_height, sample_width),
            dtype=self.dtype,
            device=self.device,
        )
        tile_y, tile_x = self.tiling.tile
        stride_y, stride_x = self.tiling.stride
        overlap_y, overlap_x = self.tiling.overlap
        self._decoders = {}
        self._schedule = []
        for origin_y in _tile_origins(self.latent_height, stride_y):
            height = min(tile_y, self.latent_height - origin_y)
            for origin_x in _tile_origins(self.latent_width, stride_x):
                width = min(tile_x, self.latent_width - origin_x)
                shape = (height, width)
                decoder = self._decoders.get(shape)
                if decoder is None:
                    decoder = _QwenImageVAEDecoderPlan(
                        config,
                        weights,
                        height,
                        width,
                        batch_size=batch_size,
                    )
                    self._decoders[shape] = decoder
                blend = OverlapTileBlendPlan(
                    decoder.output,
                    self.output,
                    origin_y * scale,
                    origin_x * scale,
                    overlap_y * scale,
                    overlap_x * scale,
                    sample_height,
                    sample_width,
                )
                self._schedule.append((origin_y, origin_x, decoder, blend))
        self.graphs = None

    @property
    def tile_count(self):
        return len(self._schedule)

    @property
    def decoder_shape_count(self):
        return len(self._decoders)

    def execute(self):
        for origin_y, origin_x, decoder, blend in self._schedule:
            height, width = decoder.input.shape[2:]
            source = self.input[
                :,
                :,
                origin_y : origin_y + height,
                origin_x : origin_x + width,
            ]
            wp.copy(decoder.input, source)
            decoder.execute()
            blend.execute()
        return self.output

    def capture(self):
        """Capture each reusable tile-shape decoder after its own warmup."""
        if not self.device.is_cuda:
            raise RuntimeError("CUDA graph capture requires a CUDA device")
        self.graphs = tuple(decoder.capture() for decoder in self._decoders.values())
        return self.graphs


class QwenImage2512VAETiledDecoder(_QwenImageVAETiledDecoderPlan):
    """Strict Qwen-Image-2512 opt-in tiled decoder approximation."""

    def __init__(
        self,
        config,
        weights,
        latent_height,
        latent_width,
        *,
        batch_size=1,
        tile=32,
        stride=24,
    ):
        qwen_image_2512_vae_decoder_weight_specs(config)
        super().__init__(
            config,
            weights,
            latent_height,
            latent_width,
            batch_size=batch_size,
            tile=tile,
            stride=stride,
        )
