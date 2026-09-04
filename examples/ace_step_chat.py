# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate successive ACE-Step 1.5 songs in a resident terminal session."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    from examples.ace_step import write_audio
    from examples.interactive import (
        TerminalProgress,
        open_output,
        output_path,
        parse_toggle,
    )
except ModuleNotFoundError:
    from ace_step import write_audio
    from interactive import TerminalProgress, open_output, output_path, parse_toggle

from warp_nn.runtime.ace_step.runner import AceStep15Bundle, AceStep15Pipeline


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="official ACE-Step 1.5 bundle directory")
    parser.add_argument(
        "--variant",
        default="acestep-v15-xl-sft",
        help="checkpoint variant (default: highest-quality XL-SFT)",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--lyrics", default="[Instrumental]")
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="seed for the first song")
    parser.add_argument("--lm-codes-strength", type=float, default=0.6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cublas", action="store_true")
    parser.add_argument("--no-planner", action="store_true")
    parser.add_argument("--normalize-output", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ace-step-output"),
        help="directory for timestamped WAV files",
    )
    parser.add_argument(
        "--auto-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="play each finished song in the default application (default: on)",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show host-side diffusion progress (default: on)",
    )
    return parser


def _help():
    print("Commands:")
    print("  /open [on|off]      toggle or set automatic playback")
    print("  /progress [on|off]  toggle or set the progress bar")
    print("  /lyrics TEXT        set lyrics for following songs")
    print("  /instrumental       use an instrumental prompt")
    print("  /duration SECONDS   set song duration")
    print("  /seed INTEGER       set the next generation seed")
    print("  /help               show this help")
    print("  /quit (/exit)       close the session")


def _try_open(path: Path) -> None:
    try:
        opened = open_output(path)
    except OSError as error:
        print(f"Could not open {path}: {error}")
        return
    if not opened:
        print(f"No desktop opener was found; song is available at {path}")


def main(argv=None):
    args = _parser().parse_args(argv)
    bundle = AceStep15Bundle.discover(
        args.model, variant=args.variant, validate_weights=True
    )
    print("Preparing ACE-Step pipeline...", flush=True)
    pipeline = AceStep15Pipeline(bundle)
    pipeline.load_generation_stack(device=args.device, use_cublas=not args.no_cublas)
    if bundle.planner_path is not None and not args.no_planner:
        pipeline.load_planner(device=args.device, use_cublas=not args.no_cublas)

    auto_open = args.auto_open
    show_progress = args.progress
    lyrics = args.lyrics
    duration = args.duration_seconds
    next_seed = args.seed
    print("Ready; model weights stay loaded between prompts.")
    print(f"Songs are written to {args.output_dir.expanduser().resolve()}.")
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
        if command in ("/open", "/progress"):
            current = auto_open if command == "/open" else show_progress
            try:
                enabled = parse_toggle(value, current)
            except ValueError as error:
                print(f"Usage: {command} [on|off] ({error})")
                continue
            if command == "/open":
                auto_open = enabled
                name = "Automatic playback"
            else:
                show_progress = enabled
                name = "Progress"
            print(f"{name} is {'on' if enabled else 'off'}.")
            continue
        if command == "/lyrics":
            lyrics = value or ""
            print("Lyrics updated." if lyrics else "Lyrics cleared.")
            continue
        if command == "/instrumental" and not value:
            lyrics = "[Instrumental]"
            print("Instrumental mode is on.")
            continue
        if command == "/duration":
            try:
                duration = float(value)
                if duration <= 0.0:
                    raise ValueError
            except ValueError:
                print("Usage: /duration SECONDS (must be positive)")
                continue
            print(f"Duration is {duration:g} seconds.")
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

        progress = TerminalProgress("Diffusion", enabled=show_progress)
        print(
            f"Generating {duration:g}s song ({args.variant}, seed {next_seed})...",
            flush=True,
        )
        started = time.perf_counter()
        try:
            audio, plan = pipeline.generate_music(
                prompt,
                lyrics,
                language=args.language,
                duration_seconds=duration,
                seed=next_seed,
                steps=args.steps,
                lm_codes_strength=args.lm_codes_strength,
                progress=progress,
            )
            destination = output_path(args.output_dir, prompt, ".wav")
            write_audio(
                destination,
                audio,
                bundle.vae.sampling_rate,
                normalize=args.normalize_output,
            )
        except Exception as error:
            print(f"Generation failed: {error}")
            continue
        next_seed += 1
        destination = destination.resolve()
        if plan is not None:
            print(f"Plan: {len(plan.audio_codes)} semantic codes")
        print(f"Output: {destination} ({time.perf_counter() - started:.1f}s)")
        if auto_open:
            _try_open(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
