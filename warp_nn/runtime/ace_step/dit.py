# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step 1.5 diffusion-transformer configuration and sampler foundations.

Only ACE-Step policy belongs here. Dense projections, RMSNorm, SwiGLU, RoPE and
attention execution remain shared runtime operations. The manifest follows the
official AceStepDiTModel and covers both turbo and XL checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np


_LAYER_TYPES = frozenset(("full_attention", "sliding_attention"))
TURBO_TIMESTEPS: dict[float, tuple[float, ...]] = {
    1.0: (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125),
    2.0: (
        1.0,
        0.9333333333333333,
        0.8571428571428571,
        0.7692307692307693,
        0.6666666666666666,
        0.5454545454545454,
        0.4,
        0.2222222222222222,
    ),
    3.0: (
        1.0,
        0.9545454545454546,
        0.9,
        0.8333333333333334,
        0.75,
        0.6428571428571429,
        0.5,
        0.3,
    ),
}
_VALID_TURBO_TIMESTEPS = tuple(
    sorted({value for schedule in TURBO_TIMESTEPS.values() for value in schedule})
)


@dataclass(frozen=True)
class AceStepDiTConfig:
    """Validated dimensions required by the 1.5 DiT."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    in_channels: int
    audio_acoustic_hidden_dim: int
    patch_size: int
    encoder_hidden_size: int
    layer_types: tuple[str, ...]
    sliding_window: int | None
    rope_theta: float
    rms_norm_eps: float
    attention_bias: bool
    model_version: str
    is_turbo: bool

    @classmethod
    def from_dict(cls, source: Mapping[str, object]) -> "AceStepDiTConfig":
        required = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "in_channels",
            "audio_acoustic_hidden_dim",
            "patch_size",
            "layer_types",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"ACE-Step DiT config is missing {missing}")
        layer_types = tuple(str(value) for value in source["layer_types"])
        layers = int(source["num_hidden_layers"])
        if len(layer_types) != layers:
            raise ValueError("layer_types must contain one entry per DiT layer")
        unknown = sorted(set(layer_types) - _LAYER_TYPES)
        if unknown:
            raise ValueError(f"unsupported ACE-Step attention types {unknown}")
        hidden = int(source["hidden_size"])
        heads = int(source["num_attention_heads"])
        kv_heads = int(source["num_key_value_heads"])
        head_dim = int(source["head_dim"])
        if min(hidden, heads, kv_heads, head_dim, layers) <= 0:
            raise ValueError("ACE-Step DiT dimensions must be positive")
        if heads % kv_heads or hidden != heads * head_dim:
            raise ValueError("ACE-Step attention head dimensions are inconsistent")
        if str(source.get("hidden_act", "silu")) != "silu":
            raise ValueError("ACE-Step DiT requires the SiLU-gated Qwen MLP")
        patch_size = int(source["patch_size"])
        in_channels = int(source["in_channels"])
        audio_channels = int(source["audio_acoustic_hidden_dim"])
        if min(patch_size, in_channels, audio_channels) <= 0:
            raise ValueError("ACE-Step patch and channel dimensions must be positive")
        if in_channels != 3 * audio_channels:
            raise ValueError(
                "in_channels must be three times audio_acoustic_hidden_dim"
            )
        use_sliding = bool(source.get("use_sliding_window", True))
        window = int(source.get("sliding_window", 128)) if use_sliding else None
        if window is not None and window <= 0:
            raise ValueError("sliding_window must be positive")
        if "sliding_attention" in layer_types and window is None:
            raise ValueError("sliding layers require use_sliding_window")
        return cls(
            hidden_size=hidden,
            intermediate_size=int(source["intermediate_size"]),
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            in_channels=in_channels,
            audio_acoustic_hidden_dim=audio_channels,
            patch_size=patch_size,
            encoder_hidden_size=int(source.get("encoder_hidden_size", hidden)),
            layer_types=layer_types,
            sliding_window=window,
            rope_theta=float(source.get("rope_theta", 1_000_000.0)),
            rms_norm_eps=float(source.get("rms_norm_eps", 1.0e-6)),
            attention_bias=bool(source.get("attention_bias", False)),
            model_version=str(source.get("model_version", "turbo")),
            is_turbo=bool(source.get("is_turbo", False)),
        )


@dataclass(frozen=True)
class AceStepDiTLayout:
    """Fixed shapes from which a graph-capturable execution plan is allocated."""

    batch: int
    latent_frames: int
    padded_frames: int
    patch_rows: int
    condition_tokens: int
    hidden_shape: tuple[int, int, int]
    condition_shape: tuple[int, int, int]
    self_kv_shape: tuple[int, int, int, int]
    cross_kv_shape: tuple[int, int, int, int]

    @classmethod
    def create(
        cls,
        config: AceStepDiTConfig,
        batch: int,
        latent_frames: int,
        condition_tokens: int,
    ) -> "AceStepDiTLayout":
        if min(batch, latent_frames, condition_tokens) <= 0:
            raise ValueError(
                "batch, latent_frames, and condition_tokens must be positive"
            )
        padded = (
            (latent_frames + config.patch_size - 1) // config.patch_size
        ) * config.patch_size
        rows = padded // config.patch_size
        return cls(
            batch=batch,
            latent_frames=latent_frames,
            padded_frames=padded,
            patch_rows=rows,
            condition_tokens=condition_tokens,
            hidden_shape=(batch, rows, config.hidden_size),
            condition_shape=(batch, condition_tokens, config.hidden_size),
            self_kv_shape=(batch, config.num_key_value_heads, rows, config.head_dim),
            cross_kv_shape=(
                batch,
                config.num_key_value_heads,
                condition_tokens,
                config.head_dim,
            ),
        )

    def attention_window(
        self, config: AceStepDiTConfig, layer_index: int
    ) -> int | None:
        if not 0 <= layer_index < config.num_hidden_layers:
            raise IndexError("ACE-Step layer index is out of range")
        return (
            config.sliding_window
            if config.layer_types[layer_index] == "sliding_attention"
            else None
        )


def dit_weight_names(config: AceStepDiTConfig, prefix: str = "decoder") -> list[str]:
    """Return the exact safetensors manifest for the official DiT submodule."""
    root = f"{prefix}." if prefix else ""
    names = [
        f"{root}scale_shift_table",
        f"{root}proj_in.1.weight",
        f"{root}proj_in.1.bias",
        f"{root}condition_embedder.weight",
        f"{root}condition_embedder.bias",
        f"{root}norm_out.weight",
        f"{root}proj_out.1.weight",
        f"{root}proj_out.1.bias",
    ]
    for time_prefix in ("time_embed", "time_embed_r"):
        for projection in ("linear_1", "linear_2", "time_proj"):
            names += [
                f"{root}{time_prefix}.{projection}.weight",
                f"{root}{time_prefix}.{projection}.bias",
            ]
    attention = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
    )
    if config.attention_bias:
        attention += ("q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias")
    for index in range(config.num_hidden_layers):
        layer = f"{root}layers.{index}."
        names += [
            f"{layer}scale_shift_table",
            f"{layer}self_attn_norm.weight",
            f"{layer}cross_attn_norm.weight",
            f"{layer}mlp_norm.weight",
            f"{layer}mlp.gate_proj.weight",
            f"{layer}mlp.up_proj.weight",
            f"{layer}mlp.down_proj.weight",
        ]
        for module in ("self_attn", "cross_attn"):
            names += [f"{layer}{module}.{suffix}" for suffix in attention]
    return names


def timestep_embedding(
    timestep: np.ndarray | Sequence[float] | float,
    dimensions: int = 256,
    *,
    scale: float = 1000.0,
    max_period: float = 10000.0,
) -> np.ndarray:
    """Compute the official ACE-Step sinusoidal timestep input in FP32."""
    if dimensions <= 0:
        raise ValueError("timestep embedding dimensions must be positive")
    values = np.asarray(timestep, dtype=np.float32).reshape(-1)
    half = dimensions // 2
    if half:
        frequencies = np.exp(
            -math.log(max_period) * np.arange(half, dtype=np.float32) / half
        )
        arguments = values[:, None] * np.float32(scale) * frequencies[None, :]
        result = np.concatenate((np.cos(arguments), np.sin(arguments)), axis=-1)
    else:
        result = np.empty((values.size, 0), dtype=np.float32)
    if dimensions % 2:
        result = np.pad(result, ((0, 0), (0, 1)))
    return result.astype(np.float32, copy=False)


def turbo_schedule(
    shift: float = 3.0,
    *,
    steps: int | None = None,
    timesteps: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Resolve the official distilled-turbo schedule."""
    if timesteps is not None:
        values = [float(value) for value in timesteps]
        while values and values[-1] == 0.0:
            values.pop()
        if not values:
            raise ValueError("custom timesteps must contain a non-zero value")
        return tuple(
            min(_VALID_TURBO_TIMESTEPS, key=lambda x: abs(x - value))
            for value in values[:20]
        )
    if steps is not None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        count = min(int(steps), 20)
        raw = tuple(1.0 - index / count for index in range(count))
        return (
            raw
            if shift == 1.0
            else tuple(shift * value / (1.0 + (shift - 1.0) * value) for value in raw)
        )
    nearest = min(TURBO_TIMESTEPS, key=lambda value: abs(value - shift))
    return TURBO_TIMESTEPS[nearest]


def flow_euler_step(
    latent: np.ndarray,
    velocity: np.ndarray,
    timestep: float,
    next_timestep: float = 0.0,
) -> np.ndarray:
    """Apply one ACE-Step flow ODE step, including final x0 reconstruction."""
    if latent.shape != velocity.shape:
        raise ValueError("latent and velocity shapes must match")
    if not 0.0 <= next_timestep <= timestep <= 1.0:
        raise ValueError("timesteps must descend within [0, 1]")
    return latent - velocity * (timestep - next_timestep)


def bidirectional_attention_mask(
    sequence: int, *, valid: np.ndarray | None = None, window: int | None = None
) -> np.ndarray:
    """Build the boolean geometry for full or sliding bidirectional attention."""
    if sequence <= 0 or (window is not None and window <= 0):
        raise ValueError("attention sequence and window must be positive")
    positions = np.arange(sequence)
    geometry = np.ones((sequence, sequence), dtype=np.bool_)
    if window is not None:
        geometry &= np.abs(positions[:, None] - positions[None, :]) <= window
    if valid is None:
        return geometry[None, None]
    valid = np.asarray(valid, dtype=np.bool_)
    if valid.ndim != 2 or valid.shape[1] != sequence:
        raise ValueError("valid must have shape [batch, sequence]")
    return geometry[None, None] & valid[:, None, None, :]


def split_adaln_modulation(values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Split a six-row modulation tensor in official shift/scale/gate order."""
    values = np.asarray(values)
    if values.ndim < 2 or values.shape[-2] != 6:
        raise ValueError("AdaLN modulation must have six parameter rows")
    return tuple(values[..., index : index + 1, :] for index in range(6))
