# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat with a local Qwen ONNX or safetensors model in a terminal."""

import argparse
import codecs
import json
from pathlib import Path

import numpy as np

from warp_nn.runtime import Qwen3OnnxRunner, Qwen3Tokenizer, Qwen35Runner, sample_token
from warp_nn.runtime.chat import split_tool_prefix
from warp_nn.runtime.coding_tools import CodingTools


def _generate(
    runner,
    tokenizer,
    logits,
    limit,
    temperature,
    cached_ids,
    tool_marker=None,
    top_p=1.0,
    top_k=0,
    presence_penalty=0.0,
    rng=None,
):
    generated = []
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    tool_started = False
    for _ in range(limit):
        token_id = (
            runner.sample_greedy(logits)
            if temperature <= 0.0
            else sample_token(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                previous_tokens=generated,
                rng=rng,
            )
        )
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        text = decoder.decode(tokenizer.token_bytes(token_id, skip_special_tokens=True))
        if tool_started:
            pending += text
        elif tool_marker:
            text, pending, tool_started = split_tool_prefix(pending + text, tool_marker)
            print(text, end="", flush=True)
        else:
            print(text, end="", flush=True)
        logits = runner.decode(token_id)
        cached_ids.append(token_id)
    tail = decoder.decode(b"", final=True)
    if tool_started:
        pending += tail
    elif tool_marker:
        text, pending, tool_started = split_tool_prefix(pending + tail, tool_marker)
        print(text, end="", flush=True)
    else:
        print(tail, end="", flush=True)
    response = tokenizer.decode(generated, skip_special_tokens=True)
    text, calls = tokenizer.parse_tool_calls(response) if tool_marker else (response, [])
    if pending and not calls:
        print(pending, end="", flush=True)
    return generated, text, calls


def _show_tool_result(result):
    print(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="Directory containing an ONNX or safetensors Qwen model")
    parser.add_argument("--system", help="Optional system message")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cache-capacity", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=16)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3 thinking mode")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--coding-agent",
        "--tools",
        action="store_true",
        help="Enable file and available sandboxed-shell tools",
    )
    parser.add_argument("--trusted-folder", type=Path, default=Path.cwd(), help="Root allowed for coding tools")
    parser.add_argument("--unsafe-shell", action="store_true", help="Allow shell commands without containment")
    args = parser.parse_args()
    temperature = args.temperature if args.temperature is not None else (1.0 if args.thinking else 0.7)
    top_p = args.top_p if args.top_p is not None else (0.95 if args.thinking else 0.8)
    if temperature < 0.0 or not 0.0 < top_p <= 1.0 or args.top_k < 0:
        parser.error("invalid sampling parameters")
    if not -2.0 <= args.presence_penalty <= 2.0:
        parser.error("--presence-penalty must be between -2 and 2")
    rng = np.random.default_rng(args.seed)

    tokenizer = Qwen3Tokenizer(args.model_dir)
    runner_type = Qwen3OnnxRunner if (args.model_dir / "model.onnx").is_file() else Qwen35Runner
    model_path = args.model_dir / "model.onnx" if runner_type is Qwen3OnnxRunner else args.model_dir
    runner = runner_type(
        model_path,
        device=args.device,
        cache_capacity=args.cache_capacity,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    system = args.system
    if args.coding_agent and system is None:
        system = "You are a coding agent. Inspect relevant files, make focused changes, and verify your work."
    messages = [] if system is None else [{"role": "system", "content": system}]
    cached_ids = []
    if args.unsafe_shell and not args.coding_agent:
        parser.error("--unsafe-shell requires --tools")
    shell = "unsafe" if args.unsafe_shell else "sandbox"
    coding_tools = CodingTools(args.trusted_folder, shell=shell) if args.coding_agent else None

    print("Enter /clear to forget the conversation or /exit to quit.")
    print("The first response may spend a few minutes compiling Warp kernels with no GPU activity.")
    if coding_tools:
        print(f"Coding tools confined to trusted folder {coding_tools.root}.")
        if coding_tools.shell == "unsafe":
            print("Warning: shell commands are unsandboxed and can modify files outside that folder.")
        elif not coding_tools.shell_available:
            print("Sandboxed shell unavailable on this host; command execution is disabled.")
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
            messages = [] if system is None else [{"role": "system", "content": system}]
            cached_ids.clear()
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": prompt})
        for tool_round in range(8):
            token_ids = tokenizer.encode_chat(
                messages,
                enable_thinking=args.thinking,
                tools=coding_tools.schemas if coding_tools else None,
            )
            if len(token_ids) >= args.cache_capacity:
                if tool_round == 0:
                    messages.pop()
                print("The conversation no longer fits in the KV cache; use /clear or a larger --cache-capacity.")
                break
            print("Qwen: ", end="", flush=True)
            if cached_ids and token_ids[: len(cached_ids)] == cached_ids:
                logits = runner.append(token_ids[len(cached_ids) :])
            else:
                logits = runner.prefill(token_ids)
            cached_ids = list(token_ids)
            generation_limit = min(args.max_new_tokens, args.cache_capacity - len(token_ids))
            generated, response, calls = _generate(
                runner,
                tokenizer,
                logits,
                generation_limit,
                temperature,
                cached_ids,
                tool_marker=tokenizer.tool_call_start if coding_tools else None,
                top_p=top_p,
                top_k=args.top_k,
                presence_penalty=args.presence_penalty,
                rng=rng,
            )
            print()
            if not generated or generated[-1] != tokenizer.eos_token_id:
                print(f"[Stopped at the {generation_limit}-token limit; use --max-new-tokens or /clear.]")
            if not calls:
                history_response = tokenizer.generation_prefix(args.thinking) + response
                messages.append({"role": "assistant", "content": history_response})
                break
            tool_calls = []
            for index, call in enumerate(calls):
                call_id = f"call_{tool_round}_{index}"
                arguments = json.dumps(call["arguments"], ensure_ascii=False, separators=(",", ":"))
                tool_calls.append(
                    {"id": call_id, "type": "function", "function": {"name": call["name"], "arguments": arguments}}
                )
            history_response = tokenizer.generation_prefix(args.thinking) + response
            messages.append({"role": "assistant", "content": history_response, "tool_calls": tool_calls})
            for call, tool_call in zip(calls, tool_calls):
                print(f"[tool] {call['name']}({json.dumps(call['arguments'], ensure_ascii=False)})")
                result = coding_tools.execute(call["name"], call["arguments"])
                _show_tool_result(result)
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})
            if tool_round == 7:
                print("[Stopped after 8 tool rounds.]")


if __name__ == "__main__":
    main()
