# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-native model loading without inference caches or weight copies."""

from dataclasses import dataclass
from pathlib import Path

import warp as wp

from warp_nn.runtime.formats.gguf import GGUFArchive, find_gguf_files
from warp_nn.runtime.muse.glimmer import (
    _gguf_config as _muse_gguf_config,
    _gguf_weight_map as _muse_gguf_weight_map,
    _validate_config as _validate_muse_config,
    _weight_names as _muse_weight_names,
)
from warp_nn.runtime.operators import rotary_cache_values
from warp_nn.runtime.qwen.qwen35 import (
    _gguf_config as _qwen_gguf_config,
    _gguf_weight_map as _qwen_gguf_weight_map,
    _validate_config as _validate_qwen_config,
    _weight_names as _qwen_weight_names,
)
from warp_nn.runtime.quantization import load_native_weights
from warp_nn.runtime.weights import MappedWeightArchive
from warp_nn.utils.device import parse_device

from .adapters import LoRAAdapterConfig
from .model import CausalLMTrainingPlan
from .muse import build_muse_lora_training_plan
from .qwen import build_qwen_lora_training_plan


@dataclass(frozen=True)
class LoadedLoRATrainingModel:
    """A built model plus its fixed rotary caches and source configuration."""

    model: CausalLMTrainingPlan
    cosine: wp.array
    sine: wp.array
    config: dict


def _rotary_caches(config, sequence, rotary_dim, dtype, device):
    cosine, sine = rotary_cache_values(
        sequence, rotary_dim, config.get("rope_parameters", {})
    )
    return (
        wp.array(cosine, dtype=dtype, device=device),
        wp.array(sine, dtype=dtype, device=device),
    )


def load_qwen_gguf_lora_training_plan(
    path: str | Path,
    *,
    batch: int,
    sequence: int,
    adapter_config: LoRAAdapterConfig,
    device=None,
    seed: int = 0,
    optimizer_options=None,
    use_cublas: bool = True,
) -> LoadedLoRATrainingModel:
    """Load one unquantized Qwen 3.5 GGUF directly into a LoRA training plan."""
    device = parse_device(device)
    archive = GGUFArchive(find_gguf_files(path))
    if archive.metadata.get("general.architecture") != "qwen35":
        raise ValueError("GGUF checkpoint is not a Qwen 3.5 model")
    config = _qwen_gguf_config(archive.metadata)
    _validate_qwen_config(config)
    mapped = MappedWeightArchive(archive, _qwen_gguf_weight_map(config), archive.tensor)
    required = tuple(_qwen_weight_names(config))
    missing = set(required) - set(mapped.names)
    if missing:
        raise ValueError(f"training checkpoint is missing {sorted(missing)[:5]}")
    weights = load_native_weights(mapped, device, required, None)
    model = build_qwen_lora_training_plan(
        config,
        weights,
        batch=batch,
        sequence=sequence,
        adapter_config=adapter_config,
        centered_norm_scales=False,
        gguf_layout=True,
        ssm_a_is_decay=True,
        seed=seed,
        optimizer_options=optimizer_options,
        use_cublas=use_cublas,
    )
    rotary_dim = int(
        int(config["head_dim"])
        * float(config["rope_parameters"].get("partial_rotary_factor", 1.0))
    )
    cosine, sine = _rotary_caches(config, sequence, rotary_dim, model.dtype, device)
    return LoadedLoRATrainingModel(model, cosine, sine, config)


def load_muse_gguf_lora_training_plan(
    path: str | Path,
    *,
    batch: int,
    sequence: int,
    adapter_config: LoRAAdapterConfig,
    device=None,
    seed: int = 0,
    optimizer_options=None,
    use_cublas: bool = True,
) -> LoadedLoRATrainingModel:
    """Load one unquantized Muse Glimmer GGUF directly into a LoRA plan."""
    device = parse_device(device)
    archive = GGUFArchive(find_gguf_files(path))
    config = _muse_gguf_config(archive.metadata)
    _validate_muse_config(config)
    mapped = MappedWeightArchive(archive, _muse_gguf_weight_map(config), archive.tensor)
    required = tuple(_muse_weight_names(config))
    missing = set(required) - set(mapped.names)
    if missing:
        raise ValueError(f"training checkpoint is missing {sorted(missing)[:5]}")
    weights = load_native_weights(mapped, device, required, None)
    model = build_muse_lora_training_plan(
        config,
        weights,
        batch=batch,
        sequence=sequence,
        adapter_config=adapter_config,
        centered_norm_scales=False,
        seed=seed,
        optimizer_options=optimizer_options,
        use_cublas=use_cublas,
    )
    cosine, sine = _rotary_caches(
        config, sequence, int(config["head_dim"]), model.dtype, device
    )
    return LoadedLoRATrainingModel(model, cosine, sine, config)
