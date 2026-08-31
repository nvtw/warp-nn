# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact fixed-shape still-image execution for the Qwen-Image-2512 VAE."""

from pathlib import Path

import warp as wp

from ..formats.safetensors import SafeTensorArchive
from ..operators import (
    ChannelAffinePlan,
    ClampPlan,
    Conv2dPlan,
    NearestUpsample2dPlan,
    ResidualAddPlan,
    SpatialPatchPackPlan,
    SpatialPatchUnpackPlan,
    SpatialRMSNormPlan,
    SpatialSelfAttentionPlan,
)
from .runner import QwenImageVAEConfig
from .vae import (
    _qwen_image_vae_decoder_weight_specs,
    load_qwen_image_2512_vae_decoder_weights,
    qwen_image_2512_vae_decoder_weight_specs,
)


class _ResidualBlockPlan:
    def __init__(self, x, weights, prefix):
        shortcut_weight = weights.get(f"{prefix}.conv_shortcut.weight")
        self.shortcut = (
            Conv2dPlan(
                x,
                shortcut_weight,
                weights[f"{prefix}.conv_shortcut.bias"],
                tensor_cores=True,
            )
            if shortcut_weight is not None
            else None
        )
        residual = x if self.shortcut is None else self.shortcut.output
        self.norm1 = SpatialRMSNormPlan(
            x, weights[f"{prefix}.norm1.gamma"], epsilon=1.0e-12, silu=True
        )
        self.conv1 = Conv2dPlan(
            self.norm1.output,
            weights[f"{prefix}.conv1.weight"],
            weights[f"{prefix}.conv1.bias"],
            padding=1,
            tensor_cores=True,
        )
        self.norm2 = SpatialRMSNormPlan(
            self.conv1.output,
            weights[f"{prefix}.norm2.gamma"],
            epsilon=1.0e-12,
            silu=True,
        )
        self.conv2 = Conv2dPlan(
            self.norm2.output,
            weights[f"{prefix}.conv2.weight"],
            weights[f"{prefix}.conv2.bias"],
            padding=1,
            tensor_cores=True,
        )
        self.add = ResidualAddPlan(self.conv2.output, residual)
        self.output = self.add.output

    def execute(self):
        if self.shortcut is not None:
            self.shortcut.execute()
        self.norm1.execute()
        self.conv1.execute()
        self.norm2.execute()
        self.conv2.execute()
        return self.add.execute()


class _MidBlockPlan:
    def __init__(self, x, weights, prefix="decoder.mid_block"):
        self.first = _ResidualBlockPlan(x, weights, f"{prefix}.resnets.0")
        attention = f"{prefix}.attentions.0"
        self.attention = SpatialSelfAttentionPlan(
            self.first.output,
            weights[f"{attention}.norm.gamma"],
            weights[f"{attention}.to_qkv.weight"],
            weights[f"{attention}.to_qkv.bias"],
            weights[f"{attention}.proj.weight"],
            weights[f"{attention}.proj.bias"],
            epsilon=1.0e-12,
        )
        self.second = _ResidualBlockPlan(
            self.attention.output, weights, f"{prefix}.resnets.1"
        )
        self.output = self.second.output

    def execute(self):
        self.first.execute()
        self.attention.execute()
        return self.second.execute()


class _UpBlockPlan:
    def __init__(self, x, weights, prefix, residual_blocks, *, upsample):
        self.residuals = []
        output = x
        for index in range(residual_blocks + 1):
            residual = _ResidualBlockPlan(output, weights, f"{prefix}.resnets.{index}")
            self.residuals.append(residual)
            output = residual.output
        self.upsample = NearestUpsample2dPlan(output, 2) if upsample else None
        self.upsample_conv = (
            Conv2dPlan(
                self.upsample.output,
                weights[f"{prefix}.upsamplers.0.resample.1.weight"],
                weights[f"{prefix}.upsamplers.0.resample.1.bias"],
                padding=1,
                tensor_cores=True,
            )
            if upsample
            else None
        )
        self.output = (
            self.upsample_conv.output if self.upsample_conv is not None else output
        )

    def execute(self):
        for residual in self.residuals:
            residual.execute()
        if self.upsample is not None:
            self.upsample.execute()
            return self.upsample_conv.execute()
        return self.output


class _QwenImageVAEDecoderPlan:
    """Generic fixed-shape plan used by the strict public decoder and tests."""

    def __init__(self, config, weights, latent_height, latent_width, *, batch_size=1):
        if min(int(batch_size), int(latent_height), int(latent_width)) <= 0:
            raise ValueError("Qwen-Image VAE decode geometry must be positive")
        specs = _qwen_image_vae_decoder_weight_specs(config)
        missing = [spec.name for spec in specs if spec.name not in weights]
        if missing:
            raise KeyError(f"missing prepared Qwen-Image VAE weights: {missing}")
        first_weight = weights["post_quant_conv.weight"]
        self.device = first_weight.device
        self.dtype = first_weight.dtype
        if self.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("Qwen-Image VAE weights must be FP16, BF16, or FP32")
        for spec in specs:
            value = weights[spec.name]
            if (
                value.shape != spec.prepared_shape
                or value.dtype != self.dtype
                or value.device != self.device
            ):
                raise ValueError(
                    f"prepared Qwen-Image VAE weight '{spec.name}' is incompatible"
                )

        self.config = config
        batch_size = int(batch_size)
        latent_height, latent_width = int(latent_height), int(latent_width)
        self.input = wp.empty(
            (batch_size, config.latent_channels, latent_height, latent_width),
            dtype=self.dtype,
            device=self.device,
        )
        latent_scale = wp.array(config.latent_std, dtype=wp.float32, device=self.device)
        latent_bias = wp.array(config.latent_mean, dtype=wp.float32, device=self.device)
        self.denormalize = ChannelAffinePlan(self.input, latent_scale, latent_bias)
        self.pack_input = SpatialPatchPackPlan(self.denormalize.output, 1)
        nhwc = self.pack_input.output.reshape(
            (batch_size, latent_height, latent_width, config.latent_channels)
        )

        self.post_quant = Conv2dPlan(
            nhwc,
            weights["post_quant_conv.weight"],
            weights["post_quant_conv.bias"],
            tensor_cores=True,
        )
        self.conv_in = Conv2dPlan(
            self.post_quant.output,
            weights["decoder.conv_in.weight"],
            weights["decoder.conv_in.bias"],
            padding=1,
            tensor_cores=True,
        )
        self.mid = _MidBlockPlan(self.conv_in.output, weights)

        self.up_blocks = []
        output = self.mid.output
        stages = len(config.dimension_multipliers)
        for index in range(stages):
            block = _UpBlockPlan(
                output,
                weights,
                f"decoder.up_blocks.{index}",
                config.residual_blocks,
                upsample=index < stages - 1,
            )
            self.up_blocks.append(block)
            output = block.output

        self.norm_out = SpatialRMSNormPlan(
            output,
            weights["decoder.norm_out.gamma"],
            epsilon=1.0e-12,
            silu=True,
        )
        self.conv_out = Conv2dPlan(
            self.norm_out.output,
            weights["decoder.conv_out.weight"],
            weights["decoder.conv_out.bias"],
            padding=1,
            tensor_cores=True,
        )
        self.clamp = ClampPlan(self.conv_out.output, -1.0, 1.0)
        sample_height = latent_height * config.spatial_scale_factor
        sample_width = latent_width * config.spatial_scale_factor
        packed_output = self.clamp.output.reshape(
            (batch_size, sample_height * sample_width, 3)
        )
        self.unpack_output = SpatialPatchUnpackPlan(
            packed_output, sample_height, sample_width, 1
        )
        self.output = self.unpack_output.output
        self.graph = None

    def _execute_uncaptured(self):
        self.denormalize.execute()
        self.pack_input.execute()
        self.post_quant.execute()
        self.conv_in.execute()
        self.mid.execute()
        for block in self.up_blocks:
            block.execute()
        self.norm_out.execute()
        self.conv_out.execute()
        self.clamp.execute()
        return self.unpack_output.execute()

    def execute(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._execute_uncaptured()
        return self.output

    def capture(self):
        """Warm up all kernels, then capture fixed-shape decode into a CUDA graph."""
        if not self.device.is_cuda:
            raise RuntimeError("CUDA graph capture requires a CUDA device")
        self._execute_uncaptured()
        wp.synchronize_stream(wp.get_stream(self.device))
        wp.capture_begin(device=self.device)
        self._execute_uncaptured()
        self.graph = wp.capture_end(device=self.device)
        return self.graph


class QwenImage2512VAEDecoder(_QwenImageVAEDecoderPlan):
    """Exact fixed-shape, still-image-only Qwen-Image-2512 VAE decoder.

    ``input`` and ``output`` use NCHW. All learned spatial computation is NHWC.
    The caller may replace standardized latent values with ``wp.copy`` between
    executions; captured graphs retain the fixed buffers and geometry.
    """

    def __init__(self, config, weights, latent_height, latent_width, *, batch_size=1):
        qwen_image_2512_vae_decoder_weight_specs(config)
        super().__init__(
            config,
            weights,
            latent_height,
            latent_width,
            batch_size=batch_size,
        )

    @classmethod
    def from_pretrained(
        cls,
        path,
        latent_height,
        latent_width,
        *,
        batch_size=1,
        device=None,
    ):
        """Load a local Diffusers VAE directory or its safetensors file."""
        path = Path(path)
        directory = path if path.is_dir() else path.parent
        config = QwenImageVAEConfig.load(directory / "config.json")
        checkpoint = (
            directory / "diffusion_pytorch_model.safetensors" if path.is_dir() else path
        )
        archive = SafeTensorArchive(checkpoint)
        weights = load_qwen_image_2512_vae_decoder_weights(
            archive, config, device=device
        )
        return cls(
            config,
            weights,
            latent_height,
            latent_width,
            batch_size=batch_size,
        )
