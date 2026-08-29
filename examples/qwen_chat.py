# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat with a supported local ONNX or safetensors model in a terminal."""

import argparse
import codecs
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

from warp_nn.runtime import create_text_runner, create_tokenizer
from warp_nn.runtime.chat import (
    ChatEncodingCache,
    is_eos_token,
    sample_runner_token,
    split_tool_prefix,
)
from warp_nn.runtime.coding_tools import CodingTools


class _EscapeMonitor:
    """Watch Esc without blocking model generation or tool execution."""

    def __init__(self):
        self.cancelled = threading.Event()
        self._done = threading.Event()
        self._terminal_state = None
        self._thread = None

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        if os.name != "nt":
            import termios
            import tty

            self._terminal_state = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._done.set()
        if self._thread:
            self._thread.join(0.2)
        if self._terminal_state is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._terminal_state)

    def _watch(self):
        if os.name == "nt":
            import msvcrt

            while not self._done.wait(0.05):
                if msvcrt.kbhit() and msvcrt.getwch() == "\x1b":
                    self.cancelled.set()
                    return
        else:
            import select

            while not self._done.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if readable and sys.stdin.read(1) == "\x1b":
                    self.cancelled.set()
                    return


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
    cancelled=None,
):
    generated = []
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    tool_started = False
    stream_filter = (
        tokenizer.stream_filter() if hasattr(tokenizer, "stream_filter") else None
    )
    for _ in range(limit):
        if cancelled and cancelled():
            break
        token_id = sample_runner_token(
            runner,
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            previous_tokens=generated,
            rng=rng,
        )
        generated.append(token_id)
        if is_eos_token(tokenizer, token_id):
            break
        logits = runner.decode(token_id)
        cached_ids.append(token_id)
        text = decoder.decode(
            tokenizer.token_bytes(token_id, skip_special_tokens=stream_filter is None)
        )
        if stream_filter:
            text = stream_filter.feed(text)
        if tool_started:
            pending += text
        elif tool_marker:
            text, pending, tool_started = split_tool_prefix(pending + text, tool_marker)
            print(text, end="", flush=True)
        else:
            print(text, end="", flush=True)
    tail = decoder.decode(b"", final=True)
    if stream_filter:
        tail = stream_filter.feed(tail, final=True)
    if tool_started:
        pending += tail
    elif tool_marker:
        text, pending, tool_started = split_tool_prefix(pending + tail, tool_marker)
        print(text, end="", flush=True)
    else:
        print(tail, end="", flush=True)
    response = tokenizer.decode(generated, skip_special_tokens=True)
    text, calls = (
        tokenizer.parse_tool_calls(response) if tool_marker else (response, [])
    )
    if pending and not calls:
        print(pending, end="", flush=True)
    return generated, text, calls


def _show_tool_result(result):
    print(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing a supported ONNX or safetensors model",
    )
    parser.add_argument("--system", help="Optional system message")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cache-capacity", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument(
        "--weight-quantization",
        choices=("q8_0",),
        help="Opt-in projection-weight compression during model loading",
    )
    parser.add_argument(
        "--yarn",
        action="store_true",
        help="Explicitly extend RoPE to cache capacity with YaRN",
    )
    parser.add_argument(
        "--yarn-factor", type=float, help="Optional YaRN extension factor"
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--coding-agent",
        "--tools",
        action="store_true",
        help="Enable file and available sandboxed-shell tools",
    )
    parser.add_argument(
        "--trusted-folder",
        type=Path,
        default=Path.cwd(),
        help="Root allowed for coding tools",
    )
    parser.add_argument(
        "--unsafe-shell",
        action="store_true",
        help="Allow shell commands without containment",
    )
    args = parser.parse_args()
    tokenizer = create_tokenizer(args.model_dir)
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
    if args.yarn_factor is not None and (not args.yarn or args.yarn_factor < 1.0):
        parser.error("--yarn-factor requires --yarn and must be at least 1")
    if args.reasoning_effort and not thinking:
        parser.error("--reasoning-effort requires thinking mode")
    if args.reasoning_effort and not tokenizer.supports_reasoning_effort:
        parser.error("this model's chat template does not support --reasoning-effort")
    rng = np.random.default_rng(args.seed)

    rope_scaling = None
    if args.yarn:
        rope_scaling = {"rope_type": "yarn"}
        if args.yarn_factor is not None:
            rope_scaling["factor"] = args.yarn_factor
    runner_options = {"rope_scaling": rope_scaling} if rope_scaling else {}
    if args.weight_quantization:
        runner_options["weight_quantization"] = args.weight_quantization
    runner = create_text_runner(
        args.model_dir,
        device=args.device,
        cache_capacity=args.cache_capacity,
        prefill_chunk_size=args.prefill_chunk_size,
        **runner_options,
    )
    system = args.system
    if args.coding_agent and system is None:
        system = (
            "You are a coding agent. Use tools only when a request requires inspecting or changing the trusted "
            "workspace. Never use tools for conversation, general knowledge, translation, or creative writing."
        )
    messages = [] if system is None else [{"role": "system", "content": system}]
    cached_ids = []
    chat_encoder = ChatEncodingCache(tokenizer)
    if args.unsafe_shell and not args.coding_agent:
        parser.error("--unsafe-shell requires --tools")
    shell = "unsafe" if args.unsafe_shell else "sandbox"
    coding_tools = (
        CodingTools(args.trusted_folder, shell=shell) if args.coding_agent else None
    )

    print("Enter /clear to forget the conversation or /exit to quit.")
    print("Press Esc to stop a response and return to user input.")
    print(
        "The first response may spend a few minutes compiling Warp kernels with no GPU activity."
    )
    if coding_tools:
        print(f"Coding tools confined to trusted folder {coding_tools.root}.")
        if coding_tools.shell == "unsafe":
            print(
                "Warning: shell commands are unsandboxed and can modify files outside that folder."
            )
        elif not coding_tools.shell_available:
            print(
                "Sandboxed shell unavailable on this host; command execution is disabled."
            )
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
            chat_encoder.reset()
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": prompt})
        with _EscapeMonitor() as cancel:
            for tool_round in range(8):
                token_ids = chat_encoder.encode_chat(
                    messages,
                    enable_thinking=thinking,
                    tools=coding_tools.schemas if coding_tools else None,
                    reasoning_effort=args.reasoning_effort,
                )
                if len(token_ids) >= args.cache_capacity:
                    if tool_round == 0:
                        messages.pop()
                    print(
                        "The conversation no longer fits in the KV cache; use /clear or a larger --cache-capacity."
                    )
                    break
                print("Assistant: ", end="", flush=True)
                if cached_ids and token_ids[: len(cached_ids)] == cached_ids:
                    logits = runner.append(token_ids[len(cached_ids) :])
                else:
                    logits = runner.prefill(token_ids)
                cached_ids = list(token_ids)
                generation_limit = min(
                    args.max_new_tokens, args.cache_capacity - len(token_ids)
                )
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
                    presence_penalty=presence_penalty,
                    rng=rng,
                    cancelled=cancel.cancelled.is_set,
                )
                chat_encoder.extend_raw(generated)
                print()
                if cancel.cancelled.is_set():
                    messages.append(
                        {
                            "role": "assistant",
                            "content": tokenizer.generation_prefix(thinking) + response,
                        }
                    )
                    print("[Cancelled.]")
                    break
                if not generated or not is_eos_token(tokenizer, generated[-1]):
                    print(
                        f"[Stopped at the {generation_limit}-token limit; use --max-new-tokens or /clear.]"
                    )
                if not calls:
                    history_response = tokenizer.generation_prefix(thinking) + response
                    message = {"role": "assistant", "content": history_response}
                    if generated and is_eos_token(tokenizer, generated[-1]):
                        message["_raw_token_ids"] = list(generated)
                    messages.append(message)
                    break
                tool_calls = []
                for index, call in enumerate(calls):
                    call_id = f"call_{tool_round}_{index}"
                    arguments = json.dumps(
                        call["arguments"], ensure_ascii=False, separators=(",", ":")
                    )
                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": call["name"], "arguments": arguments},
                        }
                    )
                history_response = tokenizer.generation_prefix(thinking) + response
                assistant_index = len(messages)
                message = {
                    "role": "assistant",
                    "content": history_response,
                    "tool_calls": tool_calls,
                }
                if generated and is_eos_token(tokenizer, generated[-1]):
                    message["_raw_token_ids"] = list(generated)
                messages.append(message)
                for call, tool_call in zip(calls, tool_calls):
                    print(
                        f"[tool] {call['name']}({json.dumps(call['arguments'], ensure_ascii=False)})"
                    )
                    result = coding_tools.execute(
                        call["name"],
                        call["arguments"],
                        cancelled=cancel.cancelled.is_set,
                    )
                    if cancel.cancelled.is_set():
                        del messages[assistant_index:]
                        if response:
                            messages.append(
                                {"role": "assistant", "content": history_response}
                            )
                        cached_ids.clear()
                        chat_encoder.reset()
                        print("[Cancelled.]")
                        break
                    _show_tool_result(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result,
                        }
                    )
                if cancel.cancelled.is_set():
                    break
                if tool_round == 7:
                    print("[Stopped after 8 tool rounds.]")


if __name__ == "__main__":
    main()
