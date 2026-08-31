# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image model assembly built from reusable runtime operators."""

from .checkpoint import (
    QwenImageTensorSpec,
    QwenImageTransformerManifest,
    TensorMetadataArchive,
)
from .prompt import (
    QWEN_IMAGE_PREFIX_TOKENS,
    QWEN_IMAGE_SYSTEM_PROMPT,
    format_qwen_image_prompt,
    tokenize_qwen_image_prompt,
)
from .runner import (
    QWEN_IMAGE_2512_RESOLUTIONS,
    FlowMatchEulerConfig,
    QwenImage2512Bundle,
    QwenImageTransformerConfig,
    QwenImageVAEConfig,
)

from .vae import (
    QwenImageVAEWeightSpec,
    load_qwen_image_2512_vae_decoder_weights,
    prepare_qwen_image_vae_decoder_weights,
    qwen_image_2512_vae_decoder_weight_specs,
)
from .vae_decoder import QwenImage2512VAEDecoder

__all__ = [
    "QWEN_IMAGE_PREFIX_TOKENS",
    "QWEN_IMAGE_SYSTEM_PROMPT",
    "format_qwen_image_prompt",
    "tokenize_qwen_image_prompt",
    "FlowMatchEulerConfig",
    "QWEN_IMAGE_2512_RESOLUTIONS",
    "QwenImage2512VAEDecoder",
    "QwenImage2512Bundle",
    "QwenImageTensorSpec",
    "QwenImageTransformerConfig",
    "QwenImageTransformerManifest",
    "QwenImageVAEConfig",
    "TensorMetadataArchive",
    "QwenImageVAEWeightSpec",
    "load_qwen_image_2512_vae_decoder_weights",
    "prepare_qwen_image_vae_decoder_weights",
    "qwen_image_2512_vae_decoder_weight_specs",
]
