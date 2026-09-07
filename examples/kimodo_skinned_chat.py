# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate Kimodo motion and preview it on a SkinTokens-rigged human mesh.

Motion models: ``nvidia/Kimodo-SOMA-RP-v1.1``, gated
``meta-llama/Meta-Llama-3-8B-Instruct``, and the ``McGill-NLP`` adapters
``LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`` and
``LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised``. Skinning source:
``VAST-AI/SkinTokens``.
Download each Hugging Face repository with::

    hf download REPOSITORY --local-dir ~/Models/warp-nn/OWNER/NAME

Convert SkinTokens with ``tools/export_skintokens_onnx.py``. The mesh must be a
CC0 MakeHuman male OBJ retaining its joint vertex groups; official assets are
at https://github.com/makehumancommunity/makehuman. The rig is cached after the
first run.
"""

try:
    from examples.kimodo_chat import main
except ImportError:
    from kimodo_chat import main


if __name__ == "__main__":
    raise SystemExit(main(skinned=True, description=__doc__))
