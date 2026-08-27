# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat with a local Qwen3 ONNX model in a terminal."""

import argparse
from pathlib import Path

from warp_nn.runtime import Qwen3OnnxRunner, Qwen3Tokenizer, sample_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="Directory containing model.onnx and tokenizer.json")
    parser.add_argument("--system", help="Optional system message")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cache-capacity", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3 thinking mode")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tokenizer = Qwen3Tokenizer(args.model_dir)
    runner = Qwen3OnnxRunner(
        str(args.model_dir / "model.onnx"),
        device=args.device,
        cache_capacity=args.cache_capacity,
    )
    messages = [] if args.system is None else [{"role": "system", "content": args.system}]

    print("Enter /clear to forget the conversation or /exit to quit.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt == "/exit":
            break
        if prompt == "/clear":
            messages = [] if args.system is None else [{"role": "system", "content": args.system}]
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": prompt})
        token_ids = tokenizer.encode_chat(messages, enable_thinking=args.thinking)
        if len(token_ids) >= args.cache_capacity:
            messages.pop()
            print("The conversation no longer fits in the KV cache; use /clear or a larger --cache-capacity.")
            continue

        logits = runner.prefill(token_ids)
        generated = []
        for _ in range(min(args.max_new_tokens, args.cache_capacity - len(token_ids))):
            token_id = sample_token(logits, temperature=args.temperature)
            generated.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break
            logits = runner.decode(token_id)

        response = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"Qwen: {response}")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
