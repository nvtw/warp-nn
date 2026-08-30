# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate a Kimodo skeletal motion without PyTorch or Transformers."""

import argparse
from pathlib import Path

import numpy as np
import warp as wp

from warp_nn.runtime import KimodoRunner, decode_motion_features, save_motion_npz


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Kimodo checkpoint directory")
    parser.add_argument(
        "text_model",
        type=Path,
        help="full Meta-Llama-3-8B-Instruct base checkpoint directory",
    )
    parser.add_argument("prompt", help="motion description")
    parser.add_argument(
        "--text-adapter",
        action="append",
        default=[],
        type=Path,
        help="PEFT adapter; repeat for MNTP then supervised adapters",
    )
    parser.add_argument("--frames", type=int, default=150)
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
    parser.add_argument("--output", type=Path, default=Path("kimodo_motion.npz"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 2 <= args.frames <= 300:
        raise ValueError("released Kimodo models support 2-300 frames")
    runner = KimodoRunner(
        args.model,
        text_model_path=args.text_model,
        text_adapter_paths=args.text_adapter,
        dtype=wp.bfloat16,
        device=args.device,
        use_cublas=args.cublas,
    )
    features = runner.generate(
        args.prompt,
        args.frames,
        denoising_steps=args.steps,
        cfg_type=args.cfg,
        text_weight=args.text_weight,
        constraint_weight=args.constraint_weight,
        heading=np.array([args.heading], dtype=np.float32),
        seed=args.seed,
    )
    decoded = decode_motion_features(features, runner.stats, runner.config.joints)
    save_motion_npz(args.output, decoded, fps=runner.config.fps)
    print(f"Saved {args.frames} frames at {runner.config.fps:g} fps to {args.output}")


if __name__ == "__main__":
    main()
