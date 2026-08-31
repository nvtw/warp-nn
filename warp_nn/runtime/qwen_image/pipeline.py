# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Memory-staged, dependency-free Qwen-Image-2512 generation."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import warp as wp

from ...utils.device import parse_device
from .._cublas import try_create_cublas
from ..formats.safetensors import SafeTensorArchive
from ..operators import (
    FlowEulerPlan,
    SpatialPatchUnpackPlan,
    TrueCFGPlan,
    seeded_normal,
)
from .mmdit import QwenImageMMDiTPlan, load_qwen_image_transformer_weights
from .prompt import QwenImagePromptEncoder
from .runner import QwenImage2512Bundle
from .vae_decoder import QwenImage2512VAEDecoder


def _padded_conditioning(positive, negative):
    """Copy two independently encoded prompts into one fixed transformer shape."""
    if positive.shape[0] != 1 or negative.shape[0] != 1:
        raise ValueError("Qwen-Image currently supports one image at a time")
    if positive.shape[2] != negative.shape[2]:
        raise ValueError("positive and negative prompt widths must match")
    length = max(positive.shape[1], negative.shape[1])
    shape = (1, length, positive.shape[2])
    outputs = []
    masks = []
    for value in (positive, negative):
        output = wp.zeros(shape, dtype=value.dtype, device=value.device)
        wp.copy(output.flatten(), value.flatten(), count=value.size)
        mask = np.zeros((1, length), dtype=bool)
        mask[:, : value.shape[1]] = True
        outputs.append(output)
        masks.append(wp.array(mask, dtype=wp.bool, device=value.device))
    return outputs[0], masks[0], outputs[1], masks[1]


def qwen_image_to_rgb8(sample) -> np.ndarray:
    """Convert one exact VAE ``[-1, 1]`` NCHW result to HWC RGB8."""
    values = sample.numpy() if hasattr(sample, "numpy") else np.asarray(sample)
    if values.ndim != 4 or values.shape[0] != 1 or values.shape[1] != 3:
        raise ValueError("Qwen-Image output must have shape [1, 3, height, width]")
    values = np.transpose(values[0], (1, 2, 0)).astype(np.float32, copy=False)
    return np.rint(np.clip((values + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)


class QwenImage2512Pipeline:
    """Run official Qwen-Image-2512 while loading only one large stage at a time.

    The exact untiled VAE is the default. ``vae_tiling=True`` is an explicit,
    approximate memory-saving mode using the release's overlap-tile geometry.
    """

    def __init__(
        self,
        bundle: QwenImage2512Bundle | str | Path,
        *,
        dtype=wp.bfloat16,
        device=None,
        use_cublas=True,
    ):
        self.bundle = (
            bundle
            if isinstance(bundle, QwenImage2512Bundle)
            else QwenImage2512Bundle.inspect(bundle, require_weights=True)
        )
        if dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Qwen-Image inference requires FP16 or BF16")
        self.dtype = dtype
        self.device = parse_device(device)
        self.use_cublas = bool(use_cublas)

    def _finish_stage(self):
        if self.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(self.device))

    @staticmethod
    def _collect_released_stage():
        gc.collect()

    def encode_prompts(self, prompt, negative_prompt="", *, max_sequence_length=512):
        encoder = QwenImagePromptEncoder.from_pretrained(
            self.bundle.root,
            dtype=self.dtype,
            device=self.device,
            use_cublas=self.use_cublas,
        )
        positive = encoder.encode(prompt, max_sequence_length=max_sequence_length)
        negative = encoder.encode(
            negative_prompt, max_sequence_length=max_sequence_length
        )
        conditioning = _padded_conditioning(positive, negative)
        self._finish_stage()
        del encoder, positive, negative
        self._collect_released_stage()
        return conditioning

    def denoise(
        self,
        positive,
        positive_valid,
        negative,
        negative_valid,
        *,
        width,
        height,
        steps=50,
        true_cfg_scale=4.0,
        seed=0,
    ):
        latent_width, latent_height, sequence = self.bundle.latent_geometry(
            width, height
        )
        if not 1 <= int(steps) <= 1000:
            raise ValueError("Qwen-Image steps must be between 1 and 1000")
        if not np.isfinite(true_cfg_scale) or true_cfg_scale < 0.0:
            raise ValueError("Qwen-Image true CFG scale must be finite and nonnegative")
        sample = seeded_normal(
            (1, sequence, self.bundle.transformer.input_channels),
            seed=seed,
            dtype=self.dtype,
            device=self.device,
        )
        timestep = wp.zeros((1,), dtype=wp.float32, device=self.device)
        index_path = (
            self.bundle.root
            / "transformer"
            / "diffusion_pytorch_model.safetensors.index.json"
        )
        archive = SafeTensorArchive(index_path)
        weights = load_qwen_image_transformer_weights(
            archive, self.bundle.transformer, self.device, self.dtype
        )
        transformer_text = wp.empty_like(positive)
        transformer_text.assign(positive)
        transformer_text_valid = wp.empty_like(positive_valid)
        transformer_text_valid.assign(positive_valid)
        cublas = (
            try_create_cublas() if self.use_cublas and self.device.is_cuda else None
        )
        plan = QwenImageMMDiTPlan(
            sample,
            transformer_text,
            transformer_text_valid,
            timestep,
            weights,
            self.bundle.transformer,
            latent_height // self.bundle.transformer.patch_size,
            latent_width // self.bundle.transformer.patch_size,
            cublas=cublas,
        )
        positive_velocity = wp.empty_like(plan.output)
        cfg = TrueCFGPlan(positive_velocity, plan.output, true_cfg_scale)
        sigma = wp.zeros((1,), dtype=wp.float32, device=self.device)
        next_sigma = wp.zeros_like(sigma)
        flow = FlowEulerPlan(sample, cfg.output, sigma, next_sigma)
        schedule = self.bundle.scheduler.schedule(steps, sequence)
        for current, following in zip(schedule[:-1], schedule[1:]):
            timestep.assign(np.array([current], dtype=np.float32))
            plan.replay(text=positive, text_valid=positive_valid)
            wp.copy(positive_velocity, plan.output)
            plan.replay(text=negative, text_valid=negative_valid)
            cfg.execute()
            sigma.assign(np.array([current], dtype=np.float32))
            next_sigma.assign(np.array([following], dtype=np.float32))
            flow.execute()
        unpack = SpatialPatchUnpackPlan(
            sample,
            latent_height,
            latent_width,
            self.bundle.transformer.patch_size,
        )
        latent = unpack.execute()
        self._finish_stage()
        del (
            plan,
            weights,
            cublas,
            archive,
            positive_velocity,
            cfg,
            flow,
            unpack,
            transformer_text,
            transformer_text_valid,
        )
        self._collect_released_stage()
        return latent

    def decode(self, latent, *, vae_tiling=False, tile=32, stride=24):
        decoder = QwenImage2512VAEDecoder.from_pretrained(
            self.bundle.root / "vae",
            latent.shape[2],
            latent.shape[3],
            batch_size=latent.shape[0],
            device=self.device,
            tiling=vae_tiling,
            tile=tile,
            stride=stride,
        )
        decoder.input.assign(latent)
        output = decoder.execute()
        self._finish_stage()
        image = qwen_image_to_rgb8(output)
        del decoder, output
        self._collect_released_stage()
        return image

    def generate(
        self,
        prompt,
        *,
        negative_prompt=" ",
        width=1328,
        height=1328,
        steps=50,
        true_cfg_scale=4.0,
        seed=0,
        max_sequence_length=512,
        vae_tiling=False,
    ) -> np.ndarray:
        """Generate one HWC RGB8 image using exact weights and staged loading."""
        positive, positive_valid, negative, negative_valid = self.encode_prompts(
            prompt,
            negative_prompt,
            max_sequence_length=max_sequence_length,
        )
        latent = self.denoise(
            positive,
            positive_valid,
            negative,
            negative_valid,
            width=width,
            height=height,
            steps=steps,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
        )
        del positive, positive_valid, negative, negative_valid
        self._collect_released_stage()
        return self.decode(latent, vae_tiling=vae_tiling)
