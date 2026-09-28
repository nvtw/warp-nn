# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate RGBA images with pure Warp Qwen-Image-2.1 inference.

Download the official model first::

    hf download Qwen/Qwen-Image-2.1 --local-dir ~/Models/warp-nn/Qwen/Qwen-Image-2.1
"""

import argparse
import time

from warp_nn.runtime.formats.image import write_png_rgb8
from warp_nn.runtime.qwen_image import QwenImage21Bundle, QwenImage21Pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="official Qwen-Image-2.1 bundle directory")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cublas", action="store_true")
    parser.add_argument("--output", default="qwen-image21.png")
    parser.add_argument("--check", action="store_true", help="validate metadata only")
    args = parser.parse_args(argv)
    bundle = QwenImage21Bundle.inspect(args.model, require_weights=not args.check)
    if args.check:
        print(
            {
                "model": "Qwen-Image-2.1",
                "resolution": (args.width, args.height),
                "latent": bundle.latent_geometry(args.width, args.height),
                "missing_weight_files": len(bundle.missing_weight_files()),
            }
        )
        return 0
    if not args.prompt:
        parser.error("--prompt is required for generation")
    pipeline = QwenImage21Pipeline(
        bundle, device=args.device, use_cublas=not args.no_cublas
    )
    print(
        f"Generating {args.width}x{args.height} RGBA image in {args.steps} "
        f"steps on {pipeline.device}...",
        flush=True,
    )
    start = time.perf_counter()
    image = pipeline.generate(
        args.prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
    )
    write_png_rgb8(args.output, image)
    print(f"Wrote {args.output} in {time.perf_counter() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
