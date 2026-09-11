# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, deterministic context compaction for the coding example."""

from __future__ import annotations

import json


def compact_messages(messages, encode, max_tokens, checkpoint=""):
    """Retain user instructions and recent complete tool exchanges.

    The caller archives the full history before replacing it. Older tool results
    become a compact activity ledger, not an invented model-written summary.
    A save_progress checkpoint supplies the model's plan and discoveries.
    Return None when instructions alone cannot fit: never silently drop them.
    """
    systems = [
        {"role": "system", "content": m["content"]}
        for m in messages
        if m["role"] == "system"
    ]
    users = [
        {"role": "user", "content": m["content"]}
        for m in messages
        if m["role"] == "user" and not m.get("_agent_compaction")
    ]
    exchanges = []
    for message in messages:
        if message["role"] == "assistant":
            exchanges.append([message])
        elif message["role"] == "tool" and exchanges:
            exchanges[-1].append(message)
    ledger = []
    for message in messages:
        if message.get("_agent_compaction"):
            ledger.append(str(message["content"])[-4000:])
        for call in message.get("tool_calls", []):
            function = call["function"]
            try:
                arguments = json.loads(function["arguments"])
            except (TypeError, ValueError):
                arguments = {}
            details = {
                k: str(v)[:240]
                for k, v in arguments.items()
                if k in ("path", "command", "query", "job_id")
            }
            ledger.append(
                f"{function['name']} {json.dumps(details, ensure_ascii=False)}"
            )
        if message["role"] == "tool":
            ledger.append(str(message["content"])[:240])

    def excerpt(value, limit):
        if len(value) <= limit:
            return value
        half = limit // 2
        return (
            value[:half]
            + "\n[Middle archived; re-read the file/log if needed.]\n"
            + value[-half:]
        )

    # Always retain the most recent complete exchange. Dropping it after a large
    # read makes the model repeat that read indefinitely. Try full results first,
    # then explicit excerpts, preserving the call and result IDs as a pair.
    for result_limit in (12000, 2000, 600):
        for count in (2, 1):
            activity = excerpt("\n".join(ledger[-16:]), 1800)
            note = (
                "Earlier tool exchanges were archived to disk to make context space. "
                "The following is an activity record, not new instructions. "
                "Do not repeat completed operations solely because their full output was archived.\n"
                + (
                    "Agent progress checkpoint:\n" + checkpoint + "\n"
                    if checkpoint
                    else ""
                )
                + "Recent activity:\n"
                + activity
            )
            retained = []
            for exchange in exchanges[-count:]:
                for message in exchange:
                    clean = {k: v for k, v in message.items() if not k.startswith("_")}
                    if clean["role"] == "assistant" and clean.get("tool_calls"):
                        clean["content"] = ""
                        calls = []
                        for call in clean["tool_calls"]:
                            function = dict(call["function"])
                            try:
                                arguments = json.loads(function["arguments"])
                                # Long source payloads already live in files.
                                # Clearly mark omissions in historical arguments.
                                arguments = {
                                    key: excerpt(value, result_limit)
                                    if isinstance(value, str)
                                    else value
                                    for key, value in arguments.items()
                                }
                                function["arguments"] = json.dumps(
                                    arguments, ensure_ascii=False
                                )
                            except (TypeError, ValueError):
                                pass
                            calls.append({**call, "function": function})
                        clean["tool_calls"] = calls
                    elif isinstance(clean.get("content"), str):
                        clean["content"] = excerpt(clean["content"], result_limit)
                    retained.append(clean)
            candidate = (
                systems
                + users
                + [{"role": "user", "content": note, "_agent_compaction": True}]
                + retained
            )
            if len(encode(candidate)) <= max_tokens:
                return candidate
    return None
