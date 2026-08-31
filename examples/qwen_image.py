# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Validate or run the dependency-free official Qwen-Image-2512 pipeline."""

from __future__ import annotations

import argparse
import json

from warp_nn.runtime.formats.image import write_png_rgb8
from warp_nn.runtime.qwen_image.checkpoint import QwenImageTransformerManifest
from warp_nn.runtime.qwen_image.mmdit import qwen_image_mmdit_workspace_bytes
from warp_nn.runtime.qwen_image.pipeline import QwenImage2512Pipeline
from warp_nn.runtime.qwen_image.runner import (
    QWEN_IMAGE_2512_RESOLUTIONS,
    QwenImage2512Bundle,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="official Qwen-Image-2512 bundle directory")
    parser.add_argument("--prompt", default="", help="image description")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--preset", choices=QWEN_IMAGE_2512_RESOLUTIONS, default="1:1")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cublas", action="store_true")
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="opt in to approximate overlap-tiled VAE decoding",
    )
    parser.add_argument("--output", default="qwen-image.png")
    parser.add_argument(
        "--check", action="store_true", help="validate metadata without loading tensors"
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    bundle = QwenImage2512Bundle.inspect(args.model, require_weights=not args.check)
    preset_width, preset_height = QWEN_IMAGE_2512_RESOLUTIONS[args.preset]
    width = preset_width if args.width is None else args.width
    height = preset_height if args.height is None else args.height
    latent_width, latent_height, sequence = bundle.latent_geometry(width, height)
    if args.check:
        print(
            json.dumps(
                {
                    "model": "Qwen-Image-2512",
                    "resolution": [width, height],
                    "latent": [latent_width, latent_height],
                    "image_tokens": sequence,
                    "bf16_mmdit_workspace_bytes_at_512_text_tokens": (
                        qwen_image_mmdit_workspace_bytes(
                            bundle.transformer, sequence, 512
                        )
                    ),
                    "transformer_parameters": QwenImageTransformerManifest.from_config(
                        bundle.transformer
                    ).parameter_count,
                    "text_checkpoint_bytes": bundle.text_encoder_index.total_size,
                    "missing_weight_files": len(bundle.missing_weight_files()),
                },
                indent=2,
            )
        )
        return 0
    if not args.prompt:
        raise ValueError("--prompt is required for generation")
    pipeline = QwenImage2512Pipeline(
        bundle, device=args.device, use_cublas=not args.no_cublas
    )
    image = pipeline.generate(
        args.prompt,
        negative_prompt=args.negative_prompt,
        width=width,
        height=height,
        steps=args.steps,
        true_cfg_scale=args.true_cfg_scale,
        seed=args.seed,
        vae_tiling=args.vae_tiling,
    )
    write_png_rgb8(args.output, image)
    mode = "approximate overlap-tiled VAE" if args.vae_tiling else "exact untiled VAE"
    print(f"Wrote {args.output} ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
