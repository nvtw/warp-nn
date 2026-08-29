# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Training-oriented neural-network operations."""

from .adapters import LoRAAdapterCollection, LoRAAdapterConfig
from .attention import gqa_attention_backward, gqa_attention_forward
from .bridges import (
    accumulate_fp32_gradient,
    add_fp32_gradients,
    cast_from_float32,
    cast_to_float32,
    merge_heads,
    split_heads,
)
from .checkpoint import LoRACheckpoint, load_lora_safetensors, save_lora_safetensors
from .linear import linear_backward, linear_forward, lora_backward, lora_forward
from .optimizer import AdamWPlan
from .primitives import CrossEntropyPlan, EmbeddingPlan, TransformerPrimitivePlan
from .step import LoRALinearTrainingPlan


__all__ = [
    "AdamWPlan",
    "CrossEntropyPlan",
    "EmbeddingPlan",
    "LoRAAdapterCollection",
    "LoRAAdapterConfig",
    "LoRACheckpoint",
    "LoRALinearTrainingPlan",
    "TransformerPrimitivePlan",
    "accumulate_fp32_gradient",
    "add_fp32_gradients",
    "cast_from_float32",
    "cast_to_float32",
    "gqa_attention_backward",
    "gqa_attention_forward",
    "linear_backward",
    "linear_forward",
    "load_lora_safetensors",
    "lora_backward",
    "lora_forward",
    "merge_heads",
    "save_lora_safetensors",
    "split_heads",
]
