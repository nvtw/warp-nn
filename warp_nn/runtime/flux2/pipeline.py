# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Pure Warp FLUX.2 Klein 4B text-to-image inference."""

import gc
import json
import math
from pathlib import Path

import numpy as np
import warp as wp

from ...utils.device import parse_device
from .._cublas import try_create_cublas
from ..operators import (
    ChannelAffinePlan,
    FlowEulerPlan,
    SpatialPatchPackPlan,
    SpatialPatchUnpackPlan,
    seeded_normal,
    warp_to_numpy_float32,
)
from .prompt import Flux2KleinPromptEncoder
from .transformer import Flux2KleinTransformerPlan, load_flux2_klein_weights
from .vae import Flux2KleinVAEDecoder, flux2_klein_latent_batch_norm


class Flux2KleinBundle:
    """Validate the self-contained Diffusers layout without duplicate weights."""

    def __init__(self, root, *, require_weights=True):
        self.root = Path(root)
        required = (
            "model_index.json",
            "transformer/config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/tokenizer.json",
            "vae/config.json",
            "scheduler/scheduler_config.json",
        )
        missing = [
            self.root / name for name in required if not (self.root / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(missing[0])
        model = json.loads((self.root / "model_index.json").read_text(encoding="utf-8"))
        if model.get("_class_name") != "Flux2KleinPipeline":
            raise ValueError("model bundle is not FLUX.2 Klein")
        if require_weights:
            weights = (
                "transformer/diffusion_pytorch_model.safetensors",
                "text_encoder/model-00001-of-00002.safetensors",
                "text_encoder/model-00002-of-00002.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            )
            missing = [
                self.root / name for name in weights if not (self.root / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(missing[0])

    @staticmethod
    def geometry(width, height):
        width, height = int(width), int(height)
        if min(width, height) < 64 or width % 16 or height % 16:
            raise ValueError(
                "FLUX.2 Klein dimensions must be multiples of 16 and at least 64"
            )
        return height // 16, width // 16


def flux2_klein_schedule(steps, image_tokens):
    """Official resolution-aware exponential time/SNR shift."""
    steps, image_tokens = int(steps), int(image_tokens)
    if not 1 <= steps <= 100 or image_tokens < 1:
        raise ValueError("invalid FLUX.2 Klein schedule geometry")
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_tokens > 4300:
        mu = a2 * image_tokens + b2
    else:
        m_200 = a2 * image_tokens + b2
        m_10 = a1 * image_tokens + b1
        mu = m_200 + (steps - 200) * (m_200 - m_10) / 190.0
    scale = math.exp(mu)
    values = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)
    return np.array(
        [scale * t / (1.0 - t + scale * t) for t in values], dtype=np.float32
    )


class Flux2KleinPipeline:
    """Staged Qwen3, graph-captured Klein transformer, and Warp VAE decode."""

    def __init__(self, root, *, dtype=wp.bfloat16, device=None, use_cublas=True):
        self.bundle = (
            root if isinstance(root, Flux2KleinBundle) else Flux2KleinBundle(root)
        )
        if dtype not in (wp.bfloat16, wp.float16):
            raise TypeError("FLUX.2 Klein transformer requires BF16 or FP16")
        self.dtype = dtype
        self.device = parse_device(device)
        self.use_cublas = bool(use_cublas)

    def generate(
        self, prompt, *, width=1024, height=1024, steps=4, seed=0, progress=None
    ):
        """Return one RGB8 NumPy image."""
        grid_height, grid_width = self.bundle.geometry(width, height)
        if not isinstance(seed, (int, np.integer)):
            raise TypeError("FLUX.2 seed must be an integer")
        schedule = flux2_klein_schedule(steps, grid_height * grid_width)
        encoder = Flux2KleinPromptEncoder(
            self.bundle.root,
            dtype=self.dtype,
            device=self.device,
            use_cublas=self.use_cublas,
        )
        encoded = encoder.encode(prompt)
        text = wp.empty_like(encoded)
        text.assign(encoded)
        if self.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(self.device))
        del encoder, encoded
        gc.collect()

        image = seeded_normal(
            (1, grid_height * grid_width, 128),
            seed=np.array([int(seed)], dtype=np.int64),
            dtype=self.dtype,
            device=self.device,
        )
        config, weights = load_flux2_klein_weights(
            self.bundle.root / "transformer", self.device, self.dtype
        )
        cublas = (
            try_create_cublas() if self.use_cublas and self.device.is_cuda else None
        )
        sigma = wp.array(np.array([1.0], dtype=np.float32), device=self.device)
        denoiser = Flux2KleinTransformerPlan(
            image, text, sigma, weights, config, grid_height, grid_width, cublas=cublas
        )
        next_sigma = wp.zeros_like(sigma)
        flow = FlowEulerPlan(image, denoiser.output, sigma, next_sigma)
        if progress is not None:
            progress(0, steps)
        for index, (current, following) in enumerate(zip(schedule[:-1], schedule[1:])):
            sigma.assign(np.array([current], dtype=np.float32))
            if index == 0:
                denoiser.execute()
            else:
                denoiser.replay()
            next_sigma.assign(np.array([following], dtype=np.float32))
            flow.execute()
            if progress is not None:
                progress(index + 1, steps)

        packed = SpatialPatchUnpackPlan(image, grid_height, grid_width, 1)
        packed.execute()
        vae_root = self.bundle.root / "vae"
        mean, std = flux2_klein_latent_batch_norm(vae_root)
        scale = wp.array(std, dtype=wp.float32, device=self.device)
        bias = wp.array(mean, dtype=wp.float32, device=self.device)
        affine = ChannelAffinePlan(packed.output, scale, bias)
        affine.execute()
        repack = SpatialPatchPackPlan(affine.output, 1)
        repack.execute()
        unpack = SpatialPatchUnpackPlan(repack.output, height // 8, width // 8, 2)
        unpack.execute()
        if self.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(self.device))
        del denoiser, weights, cublas, flow, text
        gc.collect()

        decoder = Flux2KleinVAEDecoder(
            vae_root, height // 8, width // 8, dtype=self.dtype, device=self.device
        )
        decoder.input.assign(unpack.output)
        sample = warp_to_numpy_float32(decoder.execute())
        values = np.transpose(sample[0], (1, 2, 0))
        if not np.isfinite(values).all():
            raise ValueError("FLUX.2 Klein VAE produced non-finite output")
        return np.rint(np.clip((values + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)
