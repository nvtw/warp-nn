# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a local Qwen checkpoint through OpenAI Chat Completions."""

import argparse
from pathlib import Path

from warp_nn.runtime import ChatCompletions, OpenAIHTTPServer, Qwen3OnnxRunner, Qwen3Tokenizer, Qwen35Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="Directory containing an ONNX or safetensors Qwen model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", help="Model name exposed by the API; defaults to the directory name")
    parser.add_argument("--api-key", help="Optional bearer token; omitted means no authentication")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--cache-capacity", type=int, default=4096)
    parser.add_argument("--prefill-chunk-size", type=int, default=16)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen thinking mode")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-cublas", action="store_true")
    args = parser.parse_args()

    tokenizer = Qwen3Tokenizer(args.model_dir)
    if (args.model_dir / "model.onnx").is_file():
        runner = Qwen3OnnxRunner(
            args.model_dir / "model.onnx",
            device=args.device,
            cache_capacity=args.cache_capacity,
            prefill_chunk_size=args.prefill_chunk_size,
            use_cublas=not args.no_cublas,
        )
    else:
        runner = Qwen35Runner(
            args.model_dir,
            device=args.device,
            cache_capacity=args.cache_capacity,
            prefill_chunk_size=args.prefill_chunk_size,
            use_cublas=not args.no_cublas,
        )
    model_id = args.model_id or args.model_dir.name
    temperature = args.temperature if args.temperature is not None else (1.0 if args.thinking else 0.7)
    top_p = args.top_p if args.top_p is not None else (0.95 if args.thinking else 0.8)
    backend = ChatCompletions(
        model_id,
        runner,
        tokenizer,
        args.max_new_tokens,
        args.thinking,
        temperature,
        top_p,
        args.top_k,
        args.presence_penalty,
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
