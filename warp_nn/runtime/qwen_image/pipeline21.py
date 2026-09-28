# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Pure Warp text-to-image inference for Qwen-Image-2.1."""

import gc
from pathlib import Path

import numpy as np
import warp as wp

from ...utils.device import parse_device
from .._cublas import try_create_cublas
from ..operators import FlowEulerPlan, SpatialPatchUnpackPlan, seeded_normal
from .dit21 import QwenImage21DiTPlan, load_qwen_image_21_transformer_weights
from .prompt import QwenImage21PromptEncoder
from .runner import FlowMatchEulerConfig, QwenImage21Bundle
from .vae_decoder import QwenImage21VAEDecoder


def qwen_image_to_rgba8(sample):
    """Convert one decoded four-channel NCHW image to HWC RGBA8."""
    values = sample.numpy() if hasattr(sample, "numpy") else np.asarray(sample)
    if values.ndim != 4 or values.shape[0] != 1 or values.shape[1] != 4:
        raise ValueError("Qwen-Image-2.1 output must have shape [1, 4, H, W]")
    values = np.transpose(values[0], (1, 2, 0)).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Qwen-Image-2.1 output contains non-finite values")
    return np.rint(np.clip((values + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)


class QwenImage21Pipeline:
    """Single-image native inference with staged weights and cached text K/V."""

    def __init__(
        self,
        bundle: QwenImage21Bundle | str | Path,
        *,
        dtype=wp.bfloat16,
        device=None,
        use_cublas=True,
    ):
        self.bundle = (
            bundle
            if isinstance(bundle, QwenImage21Bundle)
            else QwenImage21Bundle.inspect(bundle, require_weights=True)
        )
        if isinstance(bundle, QwenImage21Bundle) and self.bundle.missing_weight_files():
            raise FileNotFoundError(self.bundle.missing_weight_files()[0])
        if dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Qwen-Image-2.1 inference requires FP16 or BF16")
        self.dtype = dtype
        self.device = parse_device(device)
        self.use_cublas = bool(use_cublas)
        self.scheduler = FlowMatchEulerConfig.load(
            self.bundle.root / "scheduler" / "scheduler_config.json"
        )

    def generate(
        self,
        prompt: str,
        *,
        width=1024,
        height=1024,
        steps=40,
        seed=0,
        max_sequence_length=512,
        progress=None,
    ) -> np.ndarray:
        """Generate one RGBA8 image from a text prompt."""
        latent_width, latent_height, sequence = self.bundle.latent_geometry(
            width, height
        )
        if not 2 <= int(steps) <= 1000:
            raise ValueError("Qwen-Image-2.1 steps must be between 2 and 1000")
        if not isinstance(seed, (int, np.integer)):
            raise TypeError("Qwen-Image-2.1 seed must be an integer")
        encoder = QwenImage21PromptEncoder.from_pretrained(
            self.bundle.root,
            dtype=self.dtype,
            device=self.device,
            use_cublas=self.use_cublas,
        )
        encoded = encoder.encode(prompt, max_sequence_length=max_sequence_length)
        text = wp.empty_like(encoded)
        text.assign(encoded)
        if self.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(self.device))
        del encoder, encoded
        gc.collect()

        sample = seeded_normal(
            (1, sequence, 64),
            seed=np.array([int(seed)], dtype=np.int64),
            dtype=self.dtype,
            device=self.device,
        )
        config, weights = load_qwen_image_21_transformer_weights(
            self.bundle.root / "transformer", self.device, self.dtype
        )
        cublas = (
            try_create_cublas() if self.use_cublas and self.device.is_cuda else None
        )
        timestep = wp.array(np.array([0.0, 0.0], dtype=np.float32), device=self.device)
        plan = QwenImage21DiTPlan(
            sample,
            text,
            timestep,
            weights,
            config,
            latent_height,
            latent_width,
            cublas=cublas,
        )
        plan.prefill()
        sigma = wp.zeros((1,), dtype=wp.float32, device=self.device)
        next_sigma = wp.zeros_like(sigma)
        flow = FlowEulerPlan(sample, plan.output, sigma, next_sigma)
        schedule = self.scheduler.schedule(steps, sequence)
        if progress is not None:
            progress(0, steps)
        for index, (current, following) in enumerate(zip(schedule[:-1], schedule[1:])):
            timestep.assign(np.array([current, 0.0], dtype=np.float32))
            plan.replay()
            sigma.assign(np.array([current], dtype=np.float32))
            next_sigma.assign(np.array([following], dtype=np.float32))
            flow.execute()
            if progress is not None:
                progress(index + 1, steps)
        unpack = SpatialPatchUnpackPlan(sample, latent_height, latent_width, 1)
        latent = unpack.execute()
        if self.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(self.device))
        del plan, weights, cublas, flow, unpack, text
        gc.collect()

        decoder = QwenImage21VAEDecoder.from_pretrained(
            self.bundle.root / "vae",
            latent_height,
            latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        decoder.input.assign(latent)
        output = decoder.execute()
        if self.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(self.device))
        image = qwen_image_to_rgba8(output)
        if image.shape != (height, width, 4):
            raise RuntimeError(
                f"Qwen-Image-2.1 produced unexpected shape {image.shape}"
            )
        return image
