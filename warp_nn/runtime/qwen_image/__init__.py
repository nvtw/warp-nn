# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image model assembly built from reusable runtime operators."""

from .checkpoint import (
    QwenImageTensorSpec,
    QwenImageTransformerManifest,
    TensorMetadataArchive,
)
from .mmdit import (
    QwenImageMMDiTLayerPlan,
    QwenImageMMDiTPlan,
    load_qwen_image_transformer_weights,
    qwen_image_mmdit_workspace_bytes,
    qwen_image_rotary_coordinates,
)
from .prompt import (
    QWEN_IMAGE_PREFIX_TOKENS,
    QWEN_IMAGE_SYSTEM_PROMPT,
    QwenImagePromptEncoder,
    format_qwen_image_prompt,
    tokenize_qwen_image_prompt,
)
from .pipeline import QwenImage2512Pipeline, qwen_image_to_rgb8
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
from .vae_tiling import QwenImage2512VAETiledDecoder, QwenImageVAETilingConfig

__all__ = [
    "QWEN_IMAGE_PREFIX_TOKENS",
    "QWEN_IMAGE_SYSTEM_PROMPT",
    "QwenImagePromptEncoder",
    "format_qwen_image_prompt",
    "tokenize_qwen_image_prompt",
    "FlowMatchEulerConfig",
    "QWEN_IMAGE_2512_RESOLUTIONS",
    "QwenImage2512VAEDecoder",
    "QwenImage2512Bundle",
    "QwenImage2512Pipeline",
    "QwenImageMMDiTLayerPlan",
    "QwenImage2512VAETiledDecoder",
    "QwenImageMMDiTPlan",
    "QwenImageTensorSpec",
    "QwenImageTransformerConfig",
    "QwenImageTransformerManifest",
    "QwenImageVAEConfig",
    "TensorMetadataArchive",
    "QwenImageVAEWeightSpec",
    "load_qwen_image_2512_vae_decoder_weights",
    "QwenImageVAETilingConfig",
    "load_qwen_image_transformer_weights",
    "prepare_qwen_image_vae_decoder_weights",
    "qwen_image_mmdit_workspace_bytes",
    "qwen_image_rotary_coordinates",
    "qwen_image_to_rgb8",
    "qwen_image_2512_vae_decoder_weight_specs",
]
