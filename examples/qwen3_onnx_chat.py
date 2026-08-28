# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat with a local Qwen3 ONNX model in a terminal."""

import argparse
import codecs
from pathlib import Path

from warp_nn.runtime import Qwen3OnnxRunner, Qwen3Tokenizer, sample_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="Directory containing model.onnx and tokenizer.json")
    parser.add_argument("--system", help="Optional system message")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cache-capacity", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3 thinking mode")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tokenizer = Qwen3Tokenizer(args.model_dir)
    runner = Qwen3OnnxRunner(
        str(args.model_dir / "model.onnx"),
        device=args.device,
        cache_capacity=args.cache_capacity,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    messages = [] if args.system is None else [{"role": "system", "content": args.system}]
    cached_ids = []

    print("Enter /clear to forget the conversation or /exit to quit.")
    print("The first response may spend a few minutes compiling Warp kernels with no GPU activity.")
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
            cached_ids.clear()
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": prompt})
        token_ids = tokenizer.encode_chat(messages, enable_thinking=args.thinking)
        if len(token_ids) >= args.cache_capacity:
            messages.pop()
            print("The conversation no longer fits in the KV cache; use /clear or a larger --cache-capacity.")
            continue

        print("Qwen: ", end="", flush=True)
        if cached_ids and token_ids[: len(cached_ids)] == cached_ids:
            logits = runner.append(token_ids[len(cached_ids) :])
        else:
            logits = runner.prefill(token_ids)
        cached_ids = list(token_ids)
        generated = []
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        generation_limit = min(args.max_new_tokens, args.cache_capacity - len(token_ids))
        for _ in range(generation_limit):
            token_id = (
                runner.sample_greedy(logits)
                if args.temperature <= 0.0
                else sample_token(logits, temperature=args.temperature)
            )
            generated.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break
            print(decoder.decode(tokenizer.token_bytes(token_id, skip_special_tokens=True)), end="", flush=True)
            logits = runner.decode(token_id)
            cached_ids.append(token_id)

        print(decoder.decode(b"", final=True))
        if not generated or generated[-1] != tokenizer.eos_token_id:
            print(f"[Stopped at the {generation_limit}-token limit; use --max-new-tokens or /clear.]")
        response = tokenizer.decode(generated, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
