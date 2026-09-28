# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate images with pure Warp FLUX.2 Klein 4B.

Download the Apache-2.0 model without duplicate transformer weights::

    HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 \
      hf download black-forest-labs/FLUX.2-klein-4B \
      --exclude 'flux-2-klein-4b.safetensors' --exclude '*.jpg' \
      --local-dir ~/Models/warp-nn/black-forest-labs/FLUX.2-klein-4B
"""

import argparse
import time

from warp_nn.runtime.flux2 import Flux2KleinBundle, Flux2KleinPipeline
from warp_nn.runtime.formats.image import write_png_rgb8


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="official FLUX.2 Klein 4B bundle directory")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cublas", action="store_true")
    parser.add_argument("--output", default="flux2-klein.png")
    parser.add_argument("--check", action="store_true", help="validate metadata only")
    args = parser.parse_args(argv)
    bundle = Flux2KleinBundle(args.model, require_weights=not args.check)
    geometry = bundle.geometry(args.width, args.height)
    if args.check:
        print(
            {
                "model": "FLUX.2-klein-4B",
                "resolution": (args.width, args.height),
                "image_token_grid": geometry,
            }
        )
        return 0
    if not args.prompt:
        parser.error("--prompt is required for generation")
    pipeline = Flux2KleinPipeline(
        bundle, device=args.device, use_cublas=not args.no_cublas
    )
    print(
        f"Generating {args.width}x{args.height} RGB image in {args.steps} steps on {pipeline.device}...",
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
