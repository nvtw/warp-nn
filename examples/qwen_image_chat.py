# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate successive Qwen-Image-2512 images in a resident terminal session.

Model: ``Qwen/Qwen-Image-2512`` from Hugging Face. Download it with::

    hf download Qwen/Qwen-Image-2512 --local-dir ~/Models/warp-nn/Qwen/Qwen-Image-2512
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    from examples.interactive import (
        TerminalProgress,
        open_output,
        output_path,
        parse_toggle,
    )
except ImportError:
    from interactive import TerminalProgress, open_output, output_path, parse_toggle

from warp_nn.runtime.formats.image import write_png_rgb8
from warp_nn.runtime.qwen_image.pipeline import QwenImage2512Pipeline
from warp_nn.runtime.qwen_image.runner import (
    QWEN_IMAGE_2512_RESOLUTIONS,
    QwenImage2512Bundle,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="official Qwen-Image-2512 bundle directory")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--preset", choices=QWEN_IMAGE_2512_RESOLUTIONS, default="1:1")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0, help="seed for the first image")
    parser.add_argument(
        "--num-images",
        type=int,
        default=4,
        help="candidates generated together per prompt (default: 4; maximum: 8)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cublas", action="store_true")
    parser.add_argument(
        "--resident",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep text and transformer weights on GPU between images (default: on)",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="opt in to approximate overlap-tiled VAE decoding",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen-image-output"),
        help="directory for timestamped PNG files",
    )
    parser.add_argument(
        "--auto-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open each finished image in the default viewer (default: on)",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show host-side denoising progress (default: on)",
    )
    return parser


def _help():
    print("Commands:")
    print("  /open [on|off]      toggle or set automatic opening")
    print("  /num-images [1-8]   show or set candidates per prompt")
    print("  /progress [on|off]  toggle or set the progress bar")
    print("  /help               show this help")
    print("  /quit (/exit)       close the session")


def _try_open(path: Path) -> None:
    try:
        opened = open_output(path)
    except OSError as error:
        print(f"Could not open {path}: {error}")
        return
    if not opened:
        print(f"No desktop opener was found; image is available at {path}")


def main(argv=None):
    args = _parser().parse_args(argv)
    bundle = QwenImage2512Bundle.inspect(args.model, require_weights=True)
    preset_width, preset_height = QWEN_IMAGE_2512_RESOLUTIONS[args.preset]
    width = preset_width if args.width is None else args.width
    height = preset_height if args.height is None else args.height
    bundle.latent_geometry(width, height)
    if not 1 <= args.num_images <= 8:
        raise ValueError("--num-images must be between 1 and 8")

    print("Preparing Qwen-Image pipeline...", flush=True)
    pipeline = QwenImage2512Pipeline(
        bundle,
        device=args.device,
        use_cublas=not args.no_cublas,
        resident=args.resident,
    )
    auto_open = args.auto_open
    show_progress = args.progress
    next_seed = args.seed
    num_images = args.num_images
    print(f"Ready on {pipeline.device}; the pipeline stays alive between prompts.")
    print(
        "Large weights remain on GPU; use --no-resident if memory is limited."
        if args.resident
        else "Large stages are reloaded as needed to limit GPU memory."
    )
    print(f"Images are written to {args.output_dir.expanduser().resolve()}.")
    _help()

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        command, _, value = prompt.partition(" ")
        command = command.lower()
        if command in ("/quit", "/exit") and not value:
            break
        if command == "/help" and not value:
            _help()
            continue
        if command == "/num-images":
            if not value:
                print(f"Candidates per prompt: {num_images}")
                continue
            try:
                requested = int(value)
            except ValueError:
                requested = 0
            if not 1 <= requested <= 8:
                print("Usage: /num-images N, where N is between 1 and 8")
                continue
            num_images = requested
            print(f"Candidates per prompt: {num_images}")
            continue
        if command in ("/open", "/progress"):
            current = auto_open if command == "/open" else show_progress
            try:
                enabled = parse_toggle(value, current)
            except ValueError as error:
                print(f"Usage: {command} [on|off] ({error})")
                continue
            if command == "/open":
                auto_open = enabled
                name = "Automatic opening"
            else:
                show_progress = enabled
                name = "Progress"
            print(f"{name} is {'on' if enabled else 'off'}.")
            continue
        if prompt.startswith("/"):
            print(f"Unknown command: {command}. Use /help for available commands.")
            continue

        seeds = list(range(next_seed, next_seed + num_images))
        progress = TerminalProgress("Denoising", enabled=show_progress)
        print(
            f"Generating {num_images} × {width}x{height} candidates "
            f"({args.steps} steps, seeds {seeds[0]}-{seeds[-1]})...",
            flush=True,
        )
        started = time.perf_counter()
        try:
            images = pipeline.generate_batch(
                prompt,
                num_images=num_images,
                negative_prompt=args.negative_prompt,
                width=width,
                height=height,
                steps=args.steps,
                true_cfg_scale=args.true_cfg_scale,
                seed=seeds,
                vae_tiling=args.vae_tiling,
                progress=progress,
            )
            destinations = []
            for image, seed in zip(images, seeds):
                destination = output_path(
                    args.output_dir, f"{prompt} seed {seed}", ".png"
                ).resolve()
                write_png_rgb8(destination, image)
                destinations.append(destination)
        except Exception as error:
            print(f"Generation failed: {error}")
            continue
        next_seed += num_images
        elapsed = time.perf_counter() - started
        print(f"Generated {num_images} candidates in {elapsed:.1f}s:")
        for seed, destination in zip(seeds, destinations):
            print(f"  seed {seed}: {destination}")
            if auto_open:
                _try_open(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
