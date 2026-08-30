# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe a local image with Qwen3.8 Vision and no Python media framework."""

import argparse
import codecs

from warp_nn.runtime import create_multimodal_processor, create_text_runner
from warp_nn.runtime.chat import is_eos_token, sample_runner_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Qwen3.8 model directory or text GGUF")
    parser.add_argument("image", help="PNG, PPM, NPY, or NPZ RGB image")
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--vision-path", help="Optional mmproj GGUF path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-capacity", type=int, default=4096)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-cublas", action="store_true")
    args = parser.parse_args()

    processor = create_multimodal_processor(args.model)
    prompt = processor.encode_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": args.image},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        enable_thinking=False,
    )
    runner = create_text_runner(
        args.model,
        device=args.device,
        cache_capacity=args.cache_capacity,
        prefill_chunk_size=args.prefill_chunk_size,
        use_cublas=not args.no_cublas,
        vision_path=args.vision_path,
    )
    logits = runner.prefill_multimodal(prompt)
    tokenizer = processor.tokenizer
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    generated = []
    for _ in range(args.max_new_tokens):
        token = sample_runner_token(
            runner,
            logits,
            temperature=args.temperature,
            top_p=1.0,
            top_k=0,
            presence_penalty=0.0,
            previous_tokens=generated,
        )
        if is_eos_token(tokenizer, token):
            break
        generated.append(token)
        print(
            decoder.decode(tokenizer.token_bytes(token, skip_special_tokens=True)),
            end="",
            flush=True,
        )
        logits = runner.decode(token)
    print(decoder.decode(b"", final=True))


if __name__ == "__main__":
    main()
