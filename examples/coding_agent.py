# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen or Muse coding agent using the shared chat loop and sandboxed tools.

Create a dedicated workspace and run either supported checkpoint::

    python examples/coding_agent.py /path/to/Qwen3.8-27B-GGUF --trusted-folder ./agent-workspace --cache-capacity 16384 --reasoning-effort medium
    python examples/coding_agent.py /path/to/Muse-Glimmer-30B-GGUF --trusted-folder ./agent-workspace --cache-capacity 16384

Add --dflash-path /path/to/assistant for acceleration. Qwen also supports --mtp
and --no-thinking. All sampling defaults come from the selected model's tokenizer.

An unattended request can create, test, and repair files::

    python examples/coding_agent.py /path/to/Qwen3.8-27B-GGUF --trusted-folder ./agent-workspace --cache-capacity 16384 --prompt 'Write minesweeper.py with a tkinter GUI and a --self-test mode that tests its game logic without opening a window. Run its self-test and fix any errors.'

The sandbox requires Linux Landlock ABI >= 6 and libseccomp. It permits file
contents from the workspace and system runtime, writes within the workspace,
and no network or host display. It applies time, output, and per-process resource
limits; these are not aggregate container/cgroup quotas. Unsupported hosts fail
closed. Use a dedicated folder containing only files the agent may change.
See docs/coding-agent.md for the security boundary and validation details.
"""

if __package__:
    from .qwen_chat import main
else:
    from qwen_chat import main


if __name__ == "__main__":
    main(coding_agent=True)
