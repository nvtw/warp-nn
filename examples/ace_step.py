# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Validate or run the dependency-free ACE-Step 1.5 pipeline.

The ``--check`` path is useful while downloading: it validates the official
multi-component bundle without loading tensors. Generation runs the native,
dependency-free Turbo or XL-SFT pipeline when the complete bundle is present.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from warp_nn.runtime.ace_step.runner import (
    AceStep15Bundle,
    AceStep15Pipeline,
    format_dit_metadata,
)
from warp_nn.runtime.formats.wav import write_wav_pcm16


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="official ACE-Step 1.5 bundle directory")
    parser.add_argument("--prompt", default="", help="music description")
    parser.add_argument("--lyrics", default="", help="lyrics or instrumental note")
    parser.add_argument("--language", default="en", help="lyric language code")
    parser.add_argument("--metadata", default="", help="formatted ACE metadata")
    parser.add_argument("--instruction", default=None, help="optional DiT instruction")
    parser.add_argument("--variant", default="acestep-v15-turbo")
    parser.add_argument(
        "--device", default=None, help="Warp device, for example cuda:0"
    )
    parser.add_argument("--output", default="ace-step.wav")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=30.0,
        help="requested audio duration (converted to ceil(seconds * 25) latents)",
    )
    parser.add_argument("--seed", type=int, default=0, help="diffusion noise seed")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="diffusion steps (default: 8 for Turbo, 50 for XL-SFT)",
    )
    parser.add_argument("--no-cublas", action="store_true")
    parser.add_argument(
        "--normalize-output",
        action="store_true",
        help="peak-normalize before PCM16 conversion (off by default)",
    )
    parser.add_argument(
        "--no-planner",
        action="store_true",
        help="skip the optional 5 Hz LM plan (faster and lower-memory, but less structured)",
    )
    parser.add_argument(
        "--lm-codes-strength",
        type=float,
        default=0.6,
        help="fraction of diffusion steps guided by LM codes (default: 0.6)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate files/configs without loading tensors",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    bundle = AceStep15Bundle.discover(
        args.model, variant=args.variant, validate_weights=not args.check
    )
    if args.check:
        print(
            json.dumps(
                {
                    "variant": bundle.variant,
                    "sample_rate": bundle.vae.sampling_rate,
                    "samples_per_latent": bundle.vae.samples_per_latent,
                    "text_hidden_size": bundle.text.hidden_size,
                    "dit_hidden_size": bundle.dit.hidden_size,
                    "planner": bundle.planner_path is not None,
                },
                indent=2,
            )
        )
        return 0
    if not args.prompt:
        raise ValueError("--prompt is required for generation")
    pipeline = AceStep15Pipeline(bundle)
    pipeline.load_generation_stack(device=args.device, use_cublas=not args.no_cublas)
    plan = None
    if bundle.planner_path is not None and not args.no_planner:
        pipeline.load_planner(device=args.device, use_cublas=not args.no_cublas)
        plan = pipeline.plan_music(
            args.prompt,
            args.lyrics,
            duration_seconds=args.duration_seconds,
            seed=args.seed,
        )
        print(
            f"Planner: {len(plan.audio_codes)} semantic codes; "
            + ", ".join(f"{name}={value}" for name, value in plan.metadata.items())
        )
    metadata = args.metadata
    caption = (
        plan.metadata.get("caption", args.prompt) if plan is not None else args.prompt
    )
    if plan is not None and not metadata:
        metadata = format_dit_metadata(plan.metadata, args.duration_seconds)
    non_cover_conditioning = None
    if plan is None or args.instruction is None:
        token_options = {"languages": [args.language], "metadata": [metadata]}
        if args.instruction is not None:
            token_options["instructions"] = [args.instruction]
        conditioning = pipeline.prepare_conditioning(
            [caption], [args.lyrics], **token_options
        )
    else:
        # The official fallback retains caption, lyrics, and metadata while
        # replacing only an explicitly customized instruction.
        paired = pipeline.prepare_conditioning(
            [caption, caption],
            [args.lyrics, args.lyrics],
            languages=[args.language, args.language],
            metadata=[metadata, metadata],
            instructions=[
                args.instruction,
                "Fill the audio semantic mask based on the given conditions:",
            ],
        )
        conditioning = paired.row(0)
        non_cover_conditioning = paired.row(1)
    if not pipeline.ready:
        missing = ", ".join(pipeline.missing_components)
        raise RuntimeError(
            f"ACE-Step bundle and Qwen conditioning are ready; generation still needs {missing}"
        )
    generation_options = {
        "conditioning": conditioning,
        "duration_seconds": args.duration_seconds,
        "seed": args.seed,
        "steps": args.steps,
    }
    if plan is not None:
        generation_options["audio_codes"] = plan.audio_codes
        generation_options["audio_cover_strength"] = args.lm_codes_strength
        generation_options["non_cover_conditioning"] = non_cover_conditioning
    audio = pipeline.generate(**generation_options)
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    audio = np.asarray(audio)
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio[0]
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    write_wav_pcm16(
        args.output,
        audio,
        bundle.vae.sampling_rate,
        normalize=args.normalize_output,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
