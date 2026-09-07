# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate successive Qwen3-TTS utterances in a resident terminal session."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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

from warp_nn.runtime.formats.wav import write_wav_pcm16
from warp_nn.runtime.qwen.tts import Qwen3TTSPipeline, SUPPORTED_TTS_LANGUAGES


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Qwen3-TTS-12Hz-1.7B-Base directory")
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=None,
        help="24-kHz PCM16 WAV for speaker-identity (x-vector) conditioning",
    )
    parser.add_argument(
        "--language",
        choices=SUPPORTED_TTS_LANGUAGES,
        default="english",
    )
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cublas", action="store_true")
    parser.add_argument("--cache-capacity", type=int, default=4096)
    parser.add_argument("--prefill-chunk-size", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen-tts-output"),
        help="directory for timestamped WAV files",
    )
    parser.add_argument(
        "--auto-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="play each finished utterance (default: on)",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show acoustic-code generation progress (default: on)",
    )
    return parser


def _help():
    print("Commands:")
    print("  /open [on|off]       toggle or set automatic playback")
    print("  /progress [on|off]   toggle or set progress display")
    print("  /language NAME       set the spoken language")
    print("  /reference PATH      set a 24-kHz PCM16 speaker-reference WAV")
    print("  /multiline           enter paragraphs; finish with an empty line")
    print("  /file PATH           speak a UTF-8 text file")
    print("  /max-seconds NUMBER  set the duration safety ceiling")
    print("  /seed INTEGER        set the next generation seed")
    print("  /help                show this help")
    print("  /quit (/exit)        close the session")


def _try_open(path: Path) -> None:
    try:
        opened = open_output(path)
    except OSError as error:
        print(f"Could not open {path}: {error}")
        return
    if not opened:
        print(f"No desktop opener was found; audio is available at {path}")


def _read_paragraphs() -> str:
    print("Enter text; finish with an empty line:")
    lines = []
    while True:
        try:
            line = input("... ")
        except EOFError:
            break
        if not line:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.max_seconds <= 0.0:
        raise ValueError("--max-seconds must be positive")
    reference = (
        None if args.reference_audio is None else args.reference_audio.expanduser()
    )
    if reference is not None and not reference.is_file():
        raise FileNotFoundError(reference)

    print("Preparing Qwen3-TTS pipeline...", flush=True)
    pipeline = Qwen3TTSPipeline(
        args.model,
        device=args.device,
        cache_capacity=args.cache_capacity,
        prefill_chunk_size=args.prefill_chunk_size,
        use_cublas=not args.no_cublas,
    )
    language = args.language
    max_seconds = args.max_seconds
    next_seed = args.seed
    auto_open = args.auto_open
    show_progress = args.progress
    print(f"Ready on {pipeline.device}; weights stay loaded between prompts.")
    if reference is None:
        print("Tip: /reference PATH conditions the Base model on a speaker identity.")
    _help()

    while True:
        try:
            text = input("Text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        command, _, value = text.partition(" ")
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
                label = "Automatic playback"
            else:
                show_progress = enabled
                label = "Progress"
            print(f"{label} is {'on' if enabled else 'off'}.")
            continue
        if command == "/language":
            candidate = value.strip().lower()
            if candidate not in SUPPORTED_TTS_LANGUAGES:
                print("Languages: " + ", ".join(SUPPORTED_TTS_LANGUAGES))
                continue
            language = candidate
            print(f"Language is {language}.")
            continue
        if command == "/reference":
            candidate = Path(value).expanduser()
            if not value or not candidate.is_file():
                print("Usage: /reference PATH (file must exist)")
                continue
            reference = candidate
            print(f"Reference audio: {reference.resolve()}")
            continue
        if command == "/multiline" and not value:
            text = _read_paragraphs()
            if not text:
                continue
        elif command == "/file":
            candidate = Path(value).expanduser()
            try:
                text = candidate.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as error:
                print(f"Could not read {candidate}: {error}")
                continue
            if not text:
                print(f"Text file is empty: {candidate}")
                continue
        if command == "/max-seconds":
            try:
                candidate = float(value)
                if candidate <= 0.0:
                    raise ValueError
            except ValueError:
                print("Usage: /max-seconds NUMBER (must be positive)")
                continue
            max_seconds = candidate
            print(f"Maximum duration is {max_seconds:g}s.")
            continue
        if command == "/seed":
            try:
                next_seed = int(value)
            except ValueError:
                print("Usage: /seed INTEGER")
                continue
            print(f"Next seed is {next_seed}.")
            continue
        if text.startswith("/") and command not in ("/multiline", "/file"):
            print(f"Unknown command: {command}. Use /help for available commands.")
            continue

        progress = TerminalProgress("Speech", enabled=show_progress)
        print(f"Generating speech (seed {next_seed})...", flush=True)
        started = time.perf_counter()
        try:
            audio, codes = pipeline.generate(
                text,
                language=language,
                reference_audio=reference,
                max_seconds=max_seconds,
                seed=next_seed,
                progress=progress,
            )
            wp.synchronize_device(pipeline.device)
            destination = output_path(args.output_dir, text, ".wav")
            write_wav_pcm16(
                destination,
                audio.numpy()[0],
                pipeline.sample_rate,
            )
        except Exception as error:
            print(f"Generation failed: {error}")
            continue
        next_seed += 1
        destination = destination.resolve()
        duration = audio.shape[1] / pipeline.sample_rate
        print(
            f"Output: {destination} ({duration:.2f}s audio, "
            f"{len(codes)} frames, {time.perf_counter() - started:.2f}s)"
        )
        if auto_open:
            _try_open(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
