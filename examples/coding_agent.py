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

The example works in a retained disposable repository copy. It prints the session
path and a review patch; the original repository is never modified. Use
--resume-workspace SESSION to continue. File tools and commands share a Linux
Landlock/seccomp sandbox. Long commands expose job IDs and retained logs. Progress
checkpoints and automatic context compaction support longer tasks. Resource
monitoring is best effort, not a VM or strict cgroup quota.
See docs/coding-agent.md for tools, dependencies, limits and review instructions.
"""

if __package__:
    from .qwen_chat import main
else:
    from qwen_chat import main


if __name__ == "__main__":
    main(coding_agent=True)
