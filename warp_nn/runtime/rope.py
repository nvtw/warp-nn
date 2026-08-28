# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable rotary-frequency construction for native text runners."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


def resolve_rope_parameters(
    base_parameters: Mapping[str, object],
    scaling: Mapping[str, object] | None,
    native_context: int,
    target_context: int,
) -> dict[str, object]:
    """Resolve an explicit RoPE override and validate its supported context."""
    parameters = dict(base_parameters)
    if scaling is None:
        if target_context > native_context:
            raise ValueError(
                "cache_capacity exceeds the model's native context; enable YaRN explicitly"
            )
        return parameters

    parameters.update(scaling)
    rope_type = str(scaling.get("rope_type", scaling.get("type", "yarn")))
    if rope_type != "yarn":
        raise ValueError("rope_scaling currently supports only YaRN")
    parameters["rope_type"] = rope_type
    original = int(parameters.get("original_max_position_embeddings", native_context))
    if original <= 0:
        raise ValueError("YaRN original_max_position_embeddings must be positive")
    factor = float(parameters.get("factor", max(1.0, target_context / original)))
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("YaRN factor must be at least 1")
    if target_context > original * factor + 1.0e-6 * original:
        raise ValueError(
            "cache_capacity exceeds the context covered by the YaRN factor"
        )
    beta_fast = float(parameters.get("beta_fast", 32.0))
    beta_slow = float(parameters.get("beta_slow", 1.0))
    if (
        not math.isfinite(beta_fast)
        or not math.isfinite(beta_slow)
        or beta_fast <= beta_slow
        or beta_slow <= 0.0
    ):
        raise ValueError("YaRN requires finite beta_fast > beta_slow > 0")
    parameters.update(
        {
            "factor": factor,
            "original_max_position_embeddings": original,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
        }
    )
    return parameters


def rotary_cache_values(
    length: int, rotary_dim: int, parameters: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    """Build FP32 cosine/sine tables for default RoPE or static YaRN."""
    if length <= 0 or rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError(
            "rotary cache length and even rotary dimension must be positive"
        )
    theta = float(parameters.get("rope_theta", 10000.0))
    if theta <= 1.0:
        raise ValueError("rope_theta must be greater than 1")
    dimensions = np.arange(0, rotary_dim, 2, dtype=np.float32)
    position_frequencies = theta ** (dimensions / rotary_dim)
    attention_factor = 1.0
    rope_type = str(parameters.get("rope_type", "default"))
    if rope_type == "default":
        inverse_frequencies = 1.0 / position_frequencies
    elif rope_type == "yarn":
        factor = float(parameters["factor"])
        original = int(parameters["original_max_position_embeddings"])
        beta_fast = float(parameters.get("beta_fast", 32.0))
        beta_slow = float(parameters.get("beta_slow", 1.0))

        def correction_dimension(rotations: float) -> float:
            return (
                rotary_dim
                * math.log(original / (rotations * 2.0 * math.pi))
                / (2.0 * math.log(theta))
            )

        low = correction_dimension(beta_fast)
        high = correction_dimension(beta_slow)
        if bool(parameters.get("truncate", True)):
            low, high = math.floor(low), math.ceil(high)
        low = max(low, 0.0)
        high = min(high, rotary_dim - 1.0)
        if low == high:
            high += 0.001
        ramp = np.clip(
            (np.arange(rotary_dim // 2, dtype=np.float32) - low) / (high - low),
            0.0,
            1.0,
        )
        extrapolation = 1.0 - ramp
        inverse_frequencies = (1.0 / (factor * position_frequencies)) * (
            1.0 - extrapolation
        ) + (1.0 / position_frequencies) * extrapolation
        attention_factor = float(
            parameters.get(
                "attention_factor",
                1.0 if factor <= 1.0 else 0.1 * math.log(factor) + 1.0,
            )
        )
        if not math.isfinite(attention_factor) or attention_factor <= 0.0:
            raise ValueError("YaRN attention_factor must be positive")
    else:
        raise ValueError(f"Unsupported rope_type '{rope_type}'")

    positions = np.arange(length, dtype=np.float32)[:, None]
    angles = positions * inverse_frequencies[None, :]
    return (
        attention_factor * np.cos(angles),
        attention_factor * np.sin(angles),
    )
