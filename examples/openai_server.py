# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a supported local checkpoint through OpenAI Chat Completions."""

import argparse
import socket
from pathlib import Path

from warp_nn.runtime import (
    ChatCompletions,
    OpenAIHTTPServer,
    create_text_runner,
    create_tokenizer,
)


def _lan_ipv4() -> str | None:
    """Return the preferred LAN address without sending network traffic."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 80))
            preferred = probe.getsockname()[0]
            if not preferred.startswith("127."):
                return preferred
    except OSError:
        pass
    try:
        candidates = {
            address[4][0]
            for address in socket.getaddrinfo(
                socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM
            )
        }
    except OSError:
        return None
    return next(
        (address for address in sorted(candidates) if not address.startswith("127.")),
        None,
    )


def _print_connection_info(server, model_id: str, host: str, api_key: str | None):
    port = server.server_port
    wildcard = host in ("", "0.0.0.0", "::")
    local_url = f"http://127.0.0.1:{port}/v1"
    print(f"Serving {model_id}", flush=True)
    print(f"  Local:  {local_url}", flush=True)
    remote_url = None
    if wildcard:
        address = _lan_ipv4()
        if address:
            remote_url = f"http://{address}:{port}/v1"
            print(f"  LAN:    {remote_url}", flush=True)
        else:
            print("  LAN:    bound to all interfaces; IP detection failed", flush=True)
    elif not host.startswith("127.") and host != "localhost":
        remote_url = f"http://{host}:{port}/v1"
        print(f"  LAN:    {remote_url}", flush=True)
    else:
        print(
            "  LAN:    disabled; use --host 0.0.0.0 to allow other computers",
            flush=True,
        )
    if remote_url:
        key_option = " --api-key YOUR_API_KEY" if api_key else ""
        print(
            "  Client: python openai_client.py"
            f" --url {remote_url} --model {model_id}{key_option}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Supported model checkpoint or directory",
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
        choices=(1, 2, 4, 8),
        default=1,
        help="Maximum simultaneous Qwen requests (adaptive B1/B2/B4/B8)",
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
    _print_connection_info(server, model_id, args.host, args.api_key)
    print(
        "  Authentication: "
        + ("bearer token required" if args.api_key else "disabled"),
        flush=True,
    )
    print(
        f"  Parallel requests: {args.max_batch_size} maximum; decode width adapts"
        " to active requests",
        flush=True,
    )
    print(
        "The first request may compile Warp kernels before it starts generating.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
