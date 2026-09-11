# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start Qwen3.8-27B with DFlash, browser chat and a LAN-accessible OpenAI API.

    .venv/bin/python examples/qwen_dflash_server.py

Finds existing Qwen GGUF and cached DFlash checkpoints in common local locations.
Override with a positional model directory and --dflash-path DIRECTORY. No downloads
are performed. Prints browser/API URLs, model ID, generated key and client commands.
Use --host 127.0.0.1 for local-only service. See examples/openai_server.md.
"""

if __package__:
    from .openai_server import main
else:
    from openai_server import main

if __name__ == "__main__":
    main(qwen_dflash=True)
