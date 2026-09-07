# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate dog motions from text in a resident interactive session.

This combines ``nvidia/Kimodo-SOMA-RP-v1.1`` with PAN's released LAFAN-to-dog
retargeter. Kimodo, ``meta-llama/Meta-Llama-3-8B-Instruct``, and both McGill-NLP
LLM2Vec adapters can be downloaded from Hugging Face with
``huggingface-cli download REPOSITORY --local-dir PATH``. Obtain PAN's compact
``lafan1dog`` model archive from its official repository:
https://github.com/hlcdyy/pan-motion-retargeting

Run this file with ``--help`` for the positional model paths and all options.
The generated result is a standalone HTML motion viewer; only Three.js is
loaded from the web when that page is opened.
"""

try:
    from examples.kimodo_chat import main
except ImportError:
    from kimodo_chat import main


if __name__ == "__main__":
    raise SystemExit(main(quadruped=True, description=__doc__))
