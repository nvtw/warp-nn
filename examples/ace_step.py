# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Validate or run the dependency-free ACE-Step 1.5 pipeline.

The ``--check`` path is useful while downloading: it validates the official
multi-component bundle without loading tensors. Generation remains deliberately
gated by ``AceStep15Pipeline.ready`` until the native DiT sampler is complete.
"""

from __future__ import annotations

import argparse
import json

from warp_nn.runtime.ace_step.runner import AceStep15Bundle, AceStep15Pipeline


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
    parser.add_argument("--no-cublas", action="store_true")
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
    pipeline.load_text_encoder(device=args.device, use_cublas=not args.no_cublas)
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
    pipeline.generate(conditioning=conditioning, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
