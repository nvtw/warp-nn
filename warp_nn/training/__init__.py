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
from .checkpoint import (
    LoRACheckpoint,
    load_lora_safetensors,
    restore_lora_collection,
    save_lora_collection,
    save_lora_safetensors,
)
from .data import SFTBatch, SFTExample, prepare_packed_sft_batch, prepare_sft_batch
from .gated_delta import GatedDeltaInputPlan
from .gated_delta_rule import GatedDeltaRulePlan
from .gated_norm import GatedRMSNormPlan
from .gqa import GQALoRAAttentionPlan
from .linear_attention import QwenGatedDeltaLoRAAttentionPlan
from .loaders import (
    LoadedLoRATrainingModel,
    load_muse_gguf_lora_training_plan,
    load_qwen_gguf_lora_training_plan,
)
from .loss import LowPrecisionCrossEntropyPlan
from .linear import linear_backward, linear_forward, lora_backward, lora_forward
from .mlp import LoRASwiGLUPlan
from .model import CausalLMTrainingPlan
from .muse import MuseLoRATransformerBlockPlan, build_muse_lora_training_plan
from .optimizer import AdamWPlan
from .output import CausalLMOutputPlan
from .primitives import CrossEntropyPlan, EmbeddingPlan, TransformerPrimitivePlan
from .qk import QKTransformPlan
from .qwen import QwenLoRATransformerBlockPlan, build_qwen_lora_training_plan
from .step import LoRALinearTrainingPlan
from .stack import LoRATransformerStackPlan
from .trainer import LoRATrainer


__all__ = [
    "AdamWPlan",
    "CrossEntropyPlan",
    "CausalLMOutputPlan",
    "CausalLMTrainingPlan",
    "EmbeddingPlan",
    "GatedDeltaInputPlan",
    "GatedDeltaRulePlan",
    "GatedRMSNormPlan",
    "GQALoRAAttentionPlan",
    "LoRAAdapterCollection",
    "LoRAAdapterConfig",
    "LoRACheckpoint",
    "LoRALinearTrainingPlan",
    "LoRASwiGLUPlan",
    "LoRATransformerStackPlan",
    "LoRATrainer",
    "LoadedLoRATrainingModel",
    "LowPrecisionCrossEntropyPlan",
    "MuseLoRATransformerBlockPlan",
    "build_muse_lora_training_plan",
    "QKTransformPlan",
    "QwenLoRATransformerBlockPlan",
    "SFTBatch",
    "SFTExample",
    "build_qwen_lora_training_plan",
    "TransformerPrimitivePlan",
    "accumulate_fp32_gradient",
    "add_fp32_gradients",
    "QwenGatedDeltaLoRAAttentionPlan",
    "cast_from_float32",
    "cast_to_float32",
    "gqa_attention_backward",
    "gqa_attention_forward",
    "linear_backward",
    "linear_forward",
    "load_lora_safetensors",
    "load_muse_gguf_lora_training_plan",
    "load_qwen_gguf_lora_training_plan",
    "restore_lora_collection",
    "lora_backward",
    "lora_forward",
    "merge_heads",
    "prepare_sft_batch",
    "prepare_packed_sft_batch",
    "save_lora_safetensors",
    "save_lora_collection",
    "split_heads",
]
