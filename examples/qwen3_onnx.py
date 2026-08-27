# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run dependency-free greedy generation with a Qwen3 ONNX model."""

import argparse
from pathlib import Path

from warp_nn.runtime import Qwen3OnnxRunner, Qwen3Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="Directory containing model.onnx and tokenizer.json")
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-capacity", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tokenizer = Qwen3Tokenizer(args.model_dir)
    token_ids = tokenizer.encode_chat([{"role": "user", "content": args.prompt}])
    runner = Qwen3OnnxRunner(
        str(args.model_dir / "model.onnx"),
        device=args.device,
        cache_capacity=args.cache_capacity,
    )
    generated = runner.generate_greedy(token_ids, args.max_new_tokens, tokenizer.eos_token_id)
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
