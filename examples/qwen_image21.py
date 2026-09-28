# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

r"""Generate or edit images with a local Qwen-Image-2.1 checkpoint.

Install the optional runtime with ``uv pip install -e '.[qwen-image21]'``.
Download the checkpoint with::

    hf download Qwen/Qwen-Image-2.1 --local-dir ~/Models/warp-nn/Qwen/Qwen-Image-2.1

Then run::

    python examples/qwen_image21.py ~/Models/warp-nn/Qwen/Qwen-Image-2.1 \
        --prompt "A glass bird on a wooden table" --output bird.png

Use ``--image input.png`` for editing. ``--repeat 3`` keeps the model resident
and reports warm image throughput; ``--compile`` may help repeated generations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from warp_nn.runtime.qwen_image import (
    QWEN_IMAGE_21_RESOLUTIONS,
    QwenImage21Bundle,
    QwenImage21Pipeline,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local Qwen-Image-2.1 bundle")
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        help="condition image; repeat for up to ten images",
    )
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--preset", choices=QWEN_IMAGE_21_RESOLUTIONS)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--repeat", type=int, default=1, help="reuse loaded weights for N outputs"
    )
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile the transformer (slow first run; useful with --repeat)",
    )
    parser.add_argument("--output", type=Path, default=Path("qwen-image21.png"))
    parser.add_argument(
        "--check", action="store_true", help="validate metadata without loading weights"
    )
    args = parser.parse_args(argv)

    bundle = QwenImage21Bundle.inspect(args.model, require_weights=not args.check)
    preset_width, preset_height = QWEN_IMAGE_21_RESOLUTIONS[args.preset or "1:1"]
    width = (
        args.width
        if args.width is not None
        else (None if args.image and not args.preset else preset_width)
    )
    height = (
        args.height
        if args.height is not None
        else (None if args.image and not args.preset else preset_height)
    )
    if args.check:
        latent_width, latent_height, tokens = bundle.latent_geometry(
            width or preset_width, height or preset_height
        )
        print(
            json.dumps(
                {
                    "model": "Qwen-Image-2.1",
                    "resolution": [width, height],
                    "latent": [latent_width, latent_height],
                    "image_tokens": tokens,
                    "indexed_transformer_and_text_bytes": sum(
                        index.total_size or 0
                        for index in (
                            bundle.transformer_index,
                            bundle.text_encoder_index,
                        )
                    ),
                    "missing_weight_files": len(bundle.missing_weight_files()),
                },
                indent=2,
            )
        )
        return 0
    if not args.prompt:
        parser.error("--prompt is required for generation")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.image and len(args.image) > 10:
        parser.error("at most ten --image inputs are supported")
    if width is not None and height is not None:
        bundle.latent_geometry(width, height)
    images = None
    if args.image:
        from PIL import Image

        images = []
        for path in args.image:
            with Image.open(path) as source:
                images.append(source.copy())
    print(f"Loading Qwen-Image-2.1 on {args.device}...", flush=True)
    started = time.perf_counter()
    pipeline = QwenImage21Pipeline(bundle, device=args.device, compile=args.compile)
    size = (
        f"{width}x{height}" if width and height else "the condition image aspect ratio"
    )
    print(
        f"Ready in {time.perf_counter() - started:.1f}s; generating {size}...",
        flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    timings = []
    for index in range(args.repeat):
        started = time.perf_counter()
        image = pipeline.generate(
            args.prompt,
            image=images,
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            steps=args.steps,
            true_cfg_scale=args.true_cfg_scale,
            seed=args.seed + index,
        )
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        output = (
            args.output
            if args.repeat == 1
            else args.output.with_name(
                f"{args.output.stem}-{index:03d}{args.output.suffix}"
            )
        )
        image.save(output)
        print(f"Wrote {output} ({image.mode}) in {elapsed:.1f}s")
    if len(timings) > 1:
        warm = timings[1:]
        print(f"Warm throughput: {len(warm) / sum(warm):.3f} images/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
