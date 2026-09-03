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

from warp_nn.runtime.ace_step.runner import AceStep15Bundle, AceStep15Pipeline
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
        help="diffusion steps (default: 8 for Turbo, 30 for XL-SFT)",
    )
    parser.add_argument("--no-cublas", action="store_true")
    parser.add_argument(
        "--normalize-output",
        action="store_true",
        help="peak-normalize before PCM16 conversion (off by default)",
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
    token_options = {
        "languages": [args.language],
        "metadata": [args.metadata],
    }
    if args.instruction is not None:
        token_options["instructions"] = [args.instruction]
    conditioning = pipeline.prepare_conditioning(
        [args.prompt], [args.lyrics], **token_options
    )
    if not pipeline.ready:
        missing = ", ".join(pipeline.missing_components)
        raise RuntimeError(
            f"ACE-Step bundle and Qwen conditioning are ready; generation still needs {missing}"
        )
    audio = pipeline.generate(
        conditioning=conditioning,
        duration_seconds=args.duration_seconds,
        seed=args.seed,
        steps=args.steps,
    )
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    audio = np.asarray(audio)
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio[0]
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
