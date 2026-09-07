# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate Kimodo motion and preview it on a SkinTokens-rigged human mesh.

Motion models: nvidia/Kimodo-SOMA-RP-v1.1,
meta-llama/Meta-Llama-3-8B-Instruct, and the MNTP plus supervised
McGill-NLP LLM2Vec adapters from Hugging Face. Skinning model:
VAST-AI/SkinTokens. The integrated CC0 character is MakeHuman's
makehuman/data/3dobjs/base.obj; pass that file as the mesh argument.

Use huggingface-cli download REPOSITORY --local-dir PATH for model
repositories. SkinTokens runs only when the rig cache is absent; subsequent
motions and sessions reuse the portable cached rig.
"""

try:
    from examples.kimodo_chat import main
except ImportError:
    from kimodo_chat import main


if __name__ == "__main__":
    raise SystemExit(main(skinned=True, description=__doc__))
