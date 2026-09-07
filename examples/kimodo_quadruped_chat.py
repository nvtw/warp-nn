# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate dog motions from text in a resident interactive session.

This combines ``nvidia/Kimodo-SOMA-RP-v1.1`` with PAN's released LAFAN-to-dog
retargeter. It also needs gated ``meta-llama/Meta-Llama-3-8B-Instruct`` and the
``McGill-NLP`` adapters ``LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`` and
``LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised``. Download each Hugging
Face repository with::

    hf download REPOSITORY --local-dir ~/Models/warp-nn/OWNER/NAME

Clone PAN's official repository, then download ``pretrained_lafan1dog.zip``
from the checkpoint link in its README and pass the extracted directory::

    git clone --depth 1 https://github.com/hlcdyy/pan-motion-retargeting.git

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
