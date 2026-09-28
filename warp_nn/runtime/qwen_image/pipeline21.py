# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image-2.1 adapter using the upstream implementation of its new model.

The 2.1 transformer, Qwen3-VL encoder, and VAE differ from the native 2512
pipeline. Keep this adapter small and use the official cached denoising path
until those operators have a validated Warp implementation.
"""

from __future__ import annotations

from pathlib import Path

from .runner import QwenImage21Bundle


class QwenImage21Pipeline:
    """Load a local 2.1 checkpoint and keep its weights resident across calls."""

    def __init__(
        self, bundle: QwenImage21Bundle | str | Path, *, device="cuda", compile=False
    ):
        self.bundle = (
            bundle
            if isinstance(bundle, QwenImage21Bundle)
            else QwenImage21Bundle.inspect(bundle, require_weights=True)
        )
        if isinstance(bundle, QwenImage21Bundle) and bundle.missing_weight_files():
            raise FileNotFoundError(bundle.missing_weight_files()[0])
        try:
            import torch
            from diffusers import QwenImage21Pipeline as DiffusersQwenImage21Pipeline
        except ImportError as error:
            raise ImportError(
                "Qwen-Image-2.1 requires warp-nn[qwen-image21]"
            ) from error
        self.device = torch.device(device)
        self._torch = torch
        self._pipeline = DiffusersQwenImage21Pipeline.from_pretrained(
            str(self.bundle.root), dtype=torch.bfloat16, local_files_only=True
        ).to(self.device)
        if compile:
            from diffusers.models.transformers.transformer_qwenimage21 import (
                QwenImage21FlexAttnProcessor,
            )

            self._pipeline.transformer.set_attn_processor(
                QwenImage21FlexAttnProcessor()
            )
            self._pipeline.transformer.compile()
        self._pipeline.set_progress_bar_config(disable=True)

    def generate(
        self,
        prompt: str,
        *,
        image=None,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int = 40,
        true_cfg_scale: float = 1.0,
        seed: int = 0,
    ):
        """Return one PIL image; its alpha channel is retained when present."""
        if image is None:
            width = 2048 if width is None else width
            height = 2048 if height is None else height
        if width is not None and height is not None:
            self.bundle.latent_geometry(width, height)
        if steps <= 0:
            raise ValueError("steps must be positive")
        if true_cfg_scale <= 0:
            raise ValueError("true_cfg_scale must be positive")
        generator = self._torch.Generator(device=self.device).manual_seed(seed)
        result = self._pipeline(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
            use_kv_cache=True,
        )
        return result.images[0]
