# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate and preview Kimodo motions in a resident terminal session."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import warp as wp

try:
    from examples.interactive import (
        TerminalProgress,
        open_output,
        output_path,
        parse_toggle,
    )
except ImportError:
    from interactive import TerminalProgress, open_output, output_path, parse_toggle

from warp_nn.runtime import KimodoRunner, decode_motion_features, write_motion_html


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Kimodo checkpoint directory")
    parser.add_argument("text_model", type=Path, help="full Llama-3 8B checkpoint")
    parser.add_argument(
        "--text-adapter",
        action="append",
        default=[],
        type=Path,
        help="PEFT adapter; repeat for MNTP then supervised adapters",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--text-weight", type=float, default=2.0)
    parser.add_argument("--constraint-weight", type=float, default=2.0)
    parser.add_argument(
        "--cfg", choices=("nocfg", "regular", "separated"), default="separated"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cublas", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kimodo-output"),
        help="directory for timestamped HTML previews",
    )
    parser.add_argument(
        "--auto-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open each finished motion in the default browser (default: on)",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show denoising progress (default: on)",
    )
    return parser


def _help():
    print("Commands:")
    print("  /duration SECONDS   set motion duration (up to 10 seconds)")
    print("  /steps NUMBER       set diffusion steps")
    print("  /seed INTEGER       set the next generation seed")
    print("  /open [on|off]      toggle or set automatic browser opening")
    print("  /progress [on|off]  toggle or set progress display")
    print("  /help               show this help")
    print("  /quit (/exit)       close the session")


def _try_open(path: Path) -> None:
    try:
        opened = open_output(path)
    except OSError as error:
        print(f"Could not open {path}: {error}")
        return
    if not opened:
        print(f"No desktop opener was found; motion is available at {path}")


def _frames(duration: float, fps: float) -> int:
    frames = round(duration * fps)
    if not 2 <= frames <= 300:
        raise ValueError(f"duration must produce 2-300 frames at {fps:g} FPS")
    return frames


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    print("Preparing Kimodo and its LLM2Vec text encoder...", flush=True)
    runner = KimodoRunner(
        args.model,
        text_model_path=args.text_model,
        text_adapter_paths=args.text_adapter,
        dtype=wp.bfloat16,
        device=args.device,
        use_cublas=args.cublas,
    )
    duration = args.duration
    _frames(duration, runner.config.fps)
    steps = args.steps
    next_seed = args.seed
    auto_open = args.auto_open
    show_progress = args.progress
    print(f"Ready on {runner.device}; weights stay loaded between prompts.")
    print(f"Motions are written to {args.output_dir.expanduser().resolve()}.")
    _help()

    while True:
        try:
            prompt = input("Motion> ").strip()
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
        if command in ("/open", "/progress"):
            current = auto_open if command == "/open" else show_progress
            try:
                enabled = parse_toggle(value, current)
            except ValueError as error:
                print(f"Usage: {command} [on|off] ({error})")
                continue
            if command == "/open":
                auto_open = enabled
                label = "Automatic opening"
            else:
                show_progress = enabled
                label = "Progress"
            print(f"{label} is {'on' if enabled else 'off'}.")
            continue
        if command == "/duration":
            try:
                candidate = float(value)
                _frames(candidate, runner.config.fps)
            except ValueError:
                print("Usage: /duration SECONDS (approximately 0.07-10)")
                continue
            duration = candidate
            print(f"Duration is {duration:g}s.")
            continue
        if command == "/steps":
            try:
                candidate = int(value)
                if not 1 <= candidate <= runner.config.diffusion_steps:
                    raise ValueError
            except ValueError:
                print(f"Usage: /steps NUMBER (1-{runner.config.diffusion_steps})")
                continue
            steps = candidate
            print(f"Diffusion steps: {steps}.")
            continue
        if command == "/seed":
            try:
                next_seed = int(value)
            except ValueError:
                print("Usage: /seed INTEGER")
                continue
            print(f"Next seed is {next_seed}.")
            continue
        if prompt.startswith("/"):
            print(f"Unknown command: {command}. Use /help for available commands.")
            continue

        frames = _frames(duration, runner.config.fps)
        progress = TerminalProgress("Motion", enabled=show_progress)
        print(
            f"Generating {duration:g}s ({frames} frames, {steps} steps, "
            f"seed {next_seed})...",
            flush=True,
        )
        started = time.perf_counter()
        try:
            features = runner.generate(
                prompt,
                frames,
                denoising_steps=steps,
                cfg_type=args.cfg,
                text_weight=args.text_weight,
                constraint_weight=args.constraint_weight,
                heading=np.array([args.heading], dtype=np.float32),
                seed=next_seed,
                progress=progress,
            )
            wp.synchronize_device(runner.device)
            generation_seconds = time.perf_counter() - started
            motion = decode_motion_features(
                features, runner.stats, runner.config.joints
            )
            destination = output_path(args.output_dir, prompt, ".html").resolve()
            write_motion_html(
                destination,
                motion,
                fps=runner.config.fps,
                prompt=prompt,
                seed=next_seed,
                generation_seconds=generation_seconds,
            )
        except Exception as error:
            print(f"Generation failed: {error}")
            continue
        print(f"Kimodo> {destination} ({generation_seconds:.2f}s)")
        next_seed += 1
        if auto_open:
            _try_open(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
