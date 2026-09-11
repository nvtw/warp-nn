# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a local Qwen3.8, Muse-Glimmer, or Nemotron-3-Nano checkpoint.

Use ``unsloth/Qwen3.8-27B-GGUF``,
``esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF``,
``unsloth/Muse-Glimmer-30B-GGUF``,
``nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16``, or
``nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`` on Hugging Face.
Download one with::

    hf download REPOSITORY --local-dir ~/Models/warp-nn/OWNER/NAME
"""

import argparse
import socket
import os
import secrets
import shlex
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
    wildcard = host in ("", "0.0.0.0")
    addresses = (
        [("This PC", "127.0.0.1")]
        if wildcard or host in ("localhost", "127.0.0.1")
        else []
    )
    if wildcard:
        address = _lan_ipv4()
        if address:
            addresses.append(("LAN", address))
        else:
            print(
                "LAN address detection failed; use this computer's LAN IPv4 address.",
                flush=True,
            )
    elif host not in ("localhost", "127.0.0.1"):
        addresses.append(("Bound address", host))
    print(f"Ready: {model_id}", flush=True)
    print(f"  Model ID: {model_id}", flush=True)
    print(
        f"  API key: {api_key or 'not required (use local if a client requires a value)'}",
        flush=True,
    )
    for label, address in addresses:
        base = f"http://{address}:{port}"
        print(f"  {label} browser: {base}/", flush=True)
        print(f"  {label} API base: {base}/v1", flush=True)
        print(
            f"  {label} Aider (run in your code repository on the client PC):",
            flush=True,
        )
        print(
            "    aider --model "
            + shlex.quote("openai/" + model_id)
            + " --openai-api-base "
            + shlex.quote(base + "/v1")
            + " --openai-api-key "
            + shlex.quote(api_key or "local"),
            flush=True,
        )
        print(
            "  Terminal chat: python examples/openai_client.py --url "
            + shlex.quote(base + "/v1")
            + " --model "
            + shlex.quote(model_id)
            + " --api-key "
            + shlex.quote(api_key or "local"),
            flush=True,
        )
    if host in ("localhost", "127.0.0.1"):
        print("  LAN disabled; use --host 0.0.0.0 to enable it.", flush=True)
    print(
        f"  LAN clients must reach TCP port {port}; allow it in your firewall if necessary.",
        flush=True,
    )
    print(
        "  Browser chat runs here; coding-agent file edits and commands run on the client PC.",
        flush=True,
    )
    print("  Aider setup: https://aider.chat/docs/llms/openai-compat.html", flush=True)
    print(
        "  This endpoint implements Chat Completions (/v1/chat/completions), not /v1/responses.",
        flush=True,
    )


def _local_qwen_paths():
    model_candidates = [
        Path.home() / ".lmstudio/models/unsloth/Qwen3.8-27B-GGUF",
        Path.home() / "Models/warp-nn/Qwen/Qwen3.8-27B-GGUF",
        Path.home() / "Models/warp-nn/unsloth/Qwen3.8-27B-GGUF",
    ]
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshots = cache / "models--z-lab--Qwen3.8-27B-DFlash2/snapshots"
    drafts = sorted(snapshots.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return next((p for p in model_candidates if p.is_dir()), None), next(
        (p for p in drafts if p.is_dir()), None
    )


def main(argv=None, *, qwen_dflash=False):
    parser = argparse.ArgumentParser(description=__doc__)
    default_model, default_draft = _local_qwen_paths() if qwen_dflash else (None, None)
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Supported model checkpoint or directory",
        **({"nargs": "?", "default": default_model} if qwen_dflash else {}),
    )
    parser.add_argument("--host", default="0.0.0.0" if qwen_dflash else "127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-id",
        default="qwen3.8-27b" if qwen_dflash else None,
        help="Model name exposed by the API; defaults to the directory name",
    )
    parser.add_argument(
        "--api-key", help="Optional bearer token; omitted means no authentication"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=16384 if qwen_dflash else 4096
    )
    parser.add_argument("--cache-capacity", type=int, default=32768)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
        help="Maximum simultaneous native-model requests (adaptive B1/B2/B4/B8)",
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
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "xhigh"),
        help="Qwen3.8 thinking depth (default: medium)",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-cublas", action="store_true")
    parser.add_argument(
        "--dflash-path",
        type=Path,
        default=default_draft,
        help="DFlash assistant checkpoint directory",
    )
    args = parser.parse_args(argv)
    if args.model_dir is None or not args.model_dir.is_dir():
        parser.error("provide an existing model directory")
    if qwen_dflash and args.dflash_path is None:
        parser.error("provide --dflash-path /path/to/Qwen3.8-27B-DFlash2")
    if args.dflash_path is not None and not args.dflash_path.is_dir():
        parser.error("--dflash-path must be an existing directory")
    if args.dflash_path is not None and args.max_batch_size != 1:
        parser.error(
            "DFlash currently requires --max-batch-size 1; requests are serialized"
        )
    if qwen_dflash and not args.api_key:
        args.api_key = secrets.token_urlsafe(24)
    print(
        f"Loading {args.model_dir}"
        + (f" with DFlash from {args.dflash_path}" if args.dflash_path else ""),
        flush=True,
    )

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
        **({"dflash_path": args.dflash_path} if args.dflash_path else {}),
    )
    model_id = args.model_id or args.model_dir.name
    thinking = (
        tokenizer.default_enable_thinking if args.thinking is None else args.thinking
    )
    from warp_nn.runtime.sampling import validate_sampling

    defaults = tokenizer.sampling_defaults(thinking)
    sampling = {
        name: getattr(args, name) if getattr(args, name) is not None else value
        for name, value in defaults.items()
    }
    temperature, top_p = sampling["temperature"], sampling["top_p"]
    top_k, presence_penalty = sampling["top_k"], sampling["presence_penalty"]
    try:
        validate_sampling(temperature, top_k, top_p, presence_penalty)
    except ValueError as error:
        parser.error(str(error))
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
        top_k,
        presence_penalty,
        args.reasoning_effort,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        use_dflash=args.dflash_path is not None,
    )
    server = OpenAIHTTPServer(
        (args.host, args.port),
        backend,
        args.api_key,
        chat_html=Path(__file__).with_name("openai_chat.html").read_bytes(),
    )
    _print_connection_info(server, model_id, args.host, args.api_key)
    print(
        f"  Acceleration: {'DFlash' if args.dflash_path else 'ordinary decoding'}; thinking={thinking}",
        flush=True,
    )
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
