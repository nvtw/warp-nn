# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a supported local checkpoint through OpenAI Chat Completions."""

import argparse
from pathlib import Path

from warp_nn.runtime import (
    ChatCompletions,
    OpenAIHTTPServer,
    create_text_runner,
    create_tokenizer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing a supported ONNX or safetensors model",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-id",
        help="Model name exposed by the API; defaults to the directory name",
    )
    parser.add_argument(
        "--api-key", help="Optional bearer token; omitted means no authentication"
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--cache-capacity", type=int, default=4096)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="Opt-in Qwen continuous decode batching",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=2.0,
        help="Idle request coalescing window for continuous batching",
    )
    parser.add_argument("--yarn", action="store_true")
    parser.add_argument("--yarn-factor", type=float)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"))
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-cublas", action="store_true")
    args = parser.parse_args()

    tokenizer = create_tokenizer(args.model_dir)
    if args.yarn_factor is not None and (not args.yarn or args.yarn_factor < 1.0):
        parser.error("--yarn-factor requires --yarn and must be at least 1")
    rope_scaling = None
    if args.yarn:
        rope_scaling = {"rope_type": "yarn"}
        if args.yarn_factor is not None:
            rope_scaling["factor"] = args.yarn_factor
    runner = create_text_runner(
        args.model_dir,
        device=args.device,
        cache_capacity=args.cache_capacity,
        prefill_chunk_size=args.prefill_chunk_size,
        use_cublas=not args.no_cublas,
        **({"rope_scaling": rope_scaling} if rope_scaling else {}),
    )
    model_id = args.model_id or args.model_dir.name
    thinking = (
        tokenizer.default_enable_thinking if args.thinking is None else args.thinking
    )
    temperature = (
        args.temperature if args.temperature is not None else (1.0 if thinking else 0.7)
    )
    top_p = args.top_p if args.top_p is not None else (0.95 if thinking else 0.8)
    presence_penalty = (
        args.presence_penalty
        if args.presence_penalty is not None
        else (0.0 if thinking else 1.5)
    )
    if temperature < 0.0 or not 0.0 < top_p <= 1.0 or args.top_k < 0:
        parser.error("invalid sampling parameters")
    if not -2.0 <= presence_penalty <= 2.0:
        parser.error("--presence-penalty must be between -2 and 2")
    if args.reasoning_effort and not thinking:
        parser.error("--reasoning-effort requires thinking mode")
    if args.reasoning_effort and not tokenizer.supports_reasoning_effort:
        parser.error("this model's chat template does not support --reasoning-effort")
    backend = ChatCompletions(
        model_id,
        runner,
        tokenizer,
        args.max_new_tokens,
        thinking,
        temperature,
        top_p,
        args.top_k,
        presence_penalty,
        args.reasoning_effort,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    server = OpenAIHTTPServer((args.host, args.port), backend, args.api_key)
    print(f"Serving {model_id} at http://{args.host}:{args.port}/v1")
    print("The first request may compile Warp kernels before it starts generating.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
