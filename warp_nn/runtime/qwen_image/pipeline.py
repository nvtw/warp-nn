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


def _padded_conditioning(positive, negative, batch_size=1):
    """Copy two independently encoded prompts into one fixed transformer shape."""
    if positive.shape[0] != 1 or negative.shape[0] != 1:
        raise ValueError("Qwen-Image prompt encodings must have batch size one")
    if positive.shape[2] != negative.shape[2]:
        raise ValueError("positive and negative prompt widths must match")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("Qwen-Image batch size must be positive")
    length = max(positive.shape[1], negative.shape[1])
    shape = (batch_size, length, positive.shape[2])
    outputs = []
    masks = []
    for value in (positive, negative):
        output = wp.zeros(shape, dtype=value.dtype, device=value.device)
        for batch in range(batch_size):
            wp.copy(
                output.flatten(),
                value.flatten(),
                count=value.size,
                dest_offset=batch * length * value.shape[2],
            )
        mask = np.zeros((batch_size, length), dtype=bool)
        mask[:, : value.shape[1]] = True
        outputs.append(output)
        masks.append(wp.array(mask, dtype=wp.bool, device=value.device))
    return outputs[0], masks[0], outputs[1], masks[1]


def qwen_image_batch_to_rgb8(sample) -> np.ndarray:
    """Convert exact VAE results to batch-HWC RGB8."""
    values = sample.numpy() if hasattr(sample, "numpy") else np.asarray(sample)
    if values.ndim != 4 or values.shape[0] <= 0 or values.shape[1] != 3:
        raise ValueError("Qwen-Image output must have shape [batch, 3, height, width]")
    values = np.transpose(values, (0, 2, 3, 1)).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Qwen-Image output contains non-finite values")
    return np.rint(np.clip((values + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)


def qwen_image_to_rgb8(sample) -> np.ndarray:
    """Convert one exact VAE result to HWC RGB8."""
    images = qwen_image_batch_to_rgb8(sample)
    if images.shape[0] != 1:
        raise ValueError("Qwen-Image single output must have batch size one")
    return images[0]


def _batch_seeds(seed, batch_size):
    if np.isscalar(seed):
        seeds = np.arange(int(seed), int(seed) + int(batch_size), dtype=np.int64)
    else:
        seeds = np.asarray(seed)
    if seeds.shape != (int(batch_size),) or not np.issubdtype(seeds.dtype, np.integer):
        raise ValueError("Qwen-Image requires one integer seed per image")
    return seeds


class QwenImage2512Pipeline:
    """Run official Qwen-Image-2512 with staged or resident large weights.

    ``resident=True`` retains text and transformer weights between generations.
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
        resident=False,
    ):
        self.bundle = (
            bundle
            if isinstance(bundle, QwenImage2512Bundle)
            else QwenImage2512Bundle.inspect(bundle, require_weights=True)
        )
        if isinstance(bundle, QwenImage2512Bundle):
            self.bundle.require_weight_files()
        if dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Qwen-Image inference requires FP16 or BF16")
        self.dtype = dtype
        self.device = parse_device(device)
        self.use_cublas = bool(use_cublas)
        self.resident = bool(resident)
        self._prompt_encoder = None
        self._transformer_weights = None
        self._cublas = None

    def _finish_stage(self):
        if self.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(self.device))

    def _collect_released_stage(self):
        if not self.resident:
            gc.collect()

    def encode_prompts(
        self, prompt, negative_prompt="", *, max_sequence_length=512, batch_size=1
    ):
        encoder = self._prompt_encoder
        if encoder is None:
            encoder = QwenImagePromptEncoder.from_pretrained(
                self.bundle.root,
                dtype=self.dtype,
                device=self.device,
                use_cublas=self.use_cublas,
            )
            if self.resident:
                self._prompt_encoder = encoder
        positive = encoder.encode(prompt, max_sequence_length=max_sequence_length)
        # QwenEncoder reuses its fixed-length output. Own the positive result
        # before encoding an equal-length negative prompt into that buffer.
        owned_positive = wp.empty_like(positive)
        owned_positive.assign(positive)
        negative = encoder.encode(
            negative_prompt, max_sequence_length=max_sequence_length
        )
        conditioning = _padded_conditioning(
            owned_positive, negative, batch_size=batch_size
        )
        self._finish_stage()
        del positive, owned_positive, negative
        if not self.resident:
            del encoder
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
        progress=None,
    ):
        latent_width, latent_height, sequence = self.bundle.latent_geometry(
            width, height
        )
        if not 2 <= int(steps) <= 1000:
            raise ValueError("Qwen-Image steps must be between 2 and 1000")
        if not np.isfinite(true_cfg_scale) or true_cfg_scale < 0.0:
            raise ValueError("Qwen-Image true CFG scale must be finite and nonnegative")
        batch_size = positive.shape[0]
        if (
            negative.shape[0] != batch_size
            or positive_valid.shape != positive.shape[:2]
            or negative_valid.shape != negative.shape[:2]
        ):
            raise ValueError("Qwen-Image conditioning batches are incompatible")
        seeds = _batch_seeds(seed, batch_size)
        sample = seeded_normal(
            (batch_size, sequence, self.bundle.transformer.input_channels),
            seed=seeds,
            dtype=self.dtype,
            device=self.device,
        )
        timestep = wp.zeros((batch_size,), dtype=wp.float32, device=self.device)
        index_path = (
            self.bundle.root
            / "transformer"
            / "diffusion_pytorch_model.safetensors.index.json"
        )
        weights = self._transformer_weights
        if weights is None:
            archive = SafeTensorArchive(index_path)
            weights = load_qwen_image_transformer_weights(
                archive, self.bundle.transformer, self.device, self.dtype
            )
            if self.resident:
                self._transformer_weights = weights
        transformer_text = wp.empty_like(positive)
        transformer_text.assign(positive)
        transformer_text_valid = wp.empty_like(positive_valid)
        transformer_text_valid.assign(positive_valid)
        cublas = self._cublas
        if cublas is None and self.use_cublas and self.device.is_cuda:
            cublas = try_create_cublas()
            if self.resident:
                self._cublas = cublas
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
        use_cfg = true_cfg_scale > 1.0
        positive_velocity = wp.empty_like(plan.output) if use_cfg else None
        cfg = (
            TrueCFGPlan(positive_velocity, plan.output, true_cfg_scale)
            if use_cfg
            else None
        )
        sigma = wp.zeros((batch_size,), dtype=wp.float32, device=self.device)
        next_sigma = wp.zeros_like(sigma)
        flow = FlowEulerPlan(
            sample, cfg.output if cfg is not None else plan.output, sigma, next_sigma
        )
        schedule = self.bundle.scheduler.schedule(steps, sequence)
        total = len(schedule) - 1
        if progress is not None:
            progress(0, total)
        current_values = np.empty(batch_size, dtype=np.float32)
        following_values = np.empty(batch_size, dtype=np.float32)
        for index, (current, following) in enumerate(zip(schedule[:-1], schedule[1:])):
            current_values.fill(current)
            following_values.fill(following)
            timestep.assign(current_values)
            plan.replay(text=positive, text_valid=positive_valid)
            if cfg is not None:
                wp.copy(positive_velocity, plan.output)
                plan.replay(text=negative, text_valid=negative_valid)
                cfg.execute()
            sigma.assign(current_values)
            next_sigma.assign(following_values)
            flow.execute()
            if progress is not None:
                progress(index + 1, total)
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
            cublas,
            flow,
            unpack,
            transformer_text,
            transformer_text_valid,
        )
        if cfg is not None:
            del positive_velocity, cfg
        if not self.resident:
            del weights
        self._collect_released_stage()
        return latent

    def decode_batch(self, latent, *, vae_tiling=False, tile=32, stride=24):
        """Decode one or more latents with one bounded-memory VAE plan."""
        if latent.ndim != 4 or latent.shape[0] <= 0:
            raise ValueError("Qwen-Image latents must be a nonempty NCHW batch")
        decoder = QwenImage2512VAEDecoder.from_pretrained(
            self.bundle.root / "vae",
            latent.shape[2],
            latent.shape[3],
            batch_size=1,
            device=self.device,
            tiling=vae_tiling,
            tile=tile,
            stride=stride,
        )
        values_per_image = int(np.prod(latent.shape[1:]))
        images = []
        for batch in range(latent.shape[0]):
            wp.copy(
                decoder.input.flatten(),
                latent.flatten(),
                count=values_per_image,
                src_offset=batch * values_per_image,
            )
            output = decoder.execute()
            self._finish_stage()
            images.append(qwen_image_to_rgb8(output))
        del decoder, output
        self._collect_released_stage()
        return np.stack(images)

    def decode(self, latent, *, vae_tiling=False, tile=32, stride=24):
        """Decode one latent image while preserving the original HWC API."""
        images = self.decode_batch(
            latent, vae_tiling=vae_tiling, tile=tile, stride=stride
        )
        if images.shape[0] != 1:
            raise ValueError("decode() requires one image; use decode_batch()")
        return images[0]

    def generate_batch(
        self,
        prompt,
        *,
        num_images=1,
        negative_prompt=" ",
        width=1328,
        height=1328,
        steps=50,
        true_cfg_scale=4.0,
        seed=0,
        max_sequence_length=512,
        vae_tiling=False,
        progress=None,
    ) -> np.ndarray:
        """Generate num_images candidates as a batch of HWC RGB8 arrays."""
        num_images = int(num_images)
        if num_images <= 0:
            raise ValueError("num_images must be positive")
        seeds = _batch_seeds(seed, num_images)
        positive, positive_valid, negative, negative_valid = self.encode_prompts(
            prompt,
            negative_prompt,
            max_sequence_length=max_sequence_length,
            batch_size=num_images,
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
            seed=seeds,
            progress=progress,
        )
        del positive, positive_valid, negative, negative_valid
        self._collect_released_stage()
        images = self.decode_batch(latent, vae_tiling=vae_tiling)
        expected = (num_images, height, width, 3)
        if images.shape != expected:
            raise RuntimeError(
                f"Qwen-Image produced {images.shape}, expected {expected}"
            )
        return images

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
        progress=None,
    ) -> np.ndarray:
        """Generate one HWC RGB8 image using the shared batched path."""
        return self.generate_batch(
            prompt,
            num_images=1,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            max_sequence_length=max_sequence_length,
            vae_tiling=vae_tiling,
            progress=progress,
        )[0]
