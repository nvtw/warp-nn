# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image-2512 bundle discovery and fixed-shape inference contracts.

This module deliberately contains no generic device kernels. Image packing,
convolution, attention, normalization, and diffusion updates belong in shared
runtime kernels/operators; this package owns only Qwen-specific configuration,
weight mapping, and model orchestration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..formats.safetensors import SafeTensorIndex, read_safetensors_index
from ..operators import flow_match_euler_schedule


QWEN_IMAGE_2512_RESOLUTIONS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Qwen-Image JSON file '{path}'") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Qwen-Image JSON object expected in '{path}'")
    return value


def _require(data: dict, fields: Sequence[str], label: str) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValueError(f"{label} config is missing {missing}")


def _positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class QwenImageTransformerConfig:
    """Validated geometry of the official 20B Qwen-Image MMDiT."""

    patch_size: int
    input_channels: int
    output_channels: int
    layers: int
    heads: int
    head_dim: int
    text_width: int
    rope_axes: tuple[int, int, int]
    guidance_embeds: bool

    @property
    def hidden_size(self) -> int:
        return self.heads * self.head_dim

    @classmethod
    def load(cls, path: str | Path) -> QwenImageTransformerConfig:
        data = _read_json(Path(path))
        _require(
            data,
            (
                "_class_name",
                "patch_size",
                "in_channels",
                "out_channels",
                "num_layers",
                "num_attention_heads",
                "attention_head_dim",
                "joint_attention_dim",
                "axes_dims_rope",
            ),
            "Qwen-Image transformer",
        )
        if data["_class_name"] != "QwenImageTransformer2DModel":
            raise ValueError("unsupported Qwen-Image transformer architecture")
        axes_value = data["axes_dims_rope"]
        if not isinstance(axes_value, list):
            raise ValueError("Qwen-Image RoPE axes must be a list")
        axes = tuple(int(value) for value in axes_value)
        head_dim = _positive_int(data["attention_head_dim"], "attention_head_dim")
        if len(axes) != 3 or any(value <= 0 or value % 2 for value in axes):
            raise ValueError(
                "Qwen-Image RoPE axes must contain three positive even widths"
            )
        if sum(axes) != head_dim:
            raise ValueError("Qwen-Image RoPE axes must span one attention head")
        return cls(
            patch_size=_positive_int(data["patch_size"], "patch_size"),
            input_channels=_positive_int(data["in_channels"], "in_channels"),
            output_channels=_positive_int(data["out_channels"], "out_channels"),
            layers=_positive_int(data["num_layers"], "num_layers"),
            heads=_positive_int(data["num_attention_heads"], "num_attention_heads"),
            head_dim=head_dim,
            text_width=_positive_int(
                data["joint_attention_dim"], "joint_attention_dim"
            ),
            rope_axes=axes,
            guidance_embeds=bool(data.get("guidance_embeds", False)),
        )


@dataclass(frozen=True)
class QwenImageVAEConfig:
    """Validated image geometry of Qwen-Image's Wan-derived causal VAE."""

    base_dim: int
    dimension_multipliers: tuple[int, ...]
    residual_blocks: int
    latent_channels: int
    temporal_downsample: tuple[bool, ...]
    latent_mean: tuple[float, ...]
    latent_std: tuple[float, ...]

    @property
    def spatial_scale_factor(self) -> int:
        return 2 ** len(self.temporal_downsample)

    @classmethod
    def load(cls, path: str | Path) -> QwenImageVAEConfig:
        data = _read_json(Path(path))
        _require(
            data,
            (
                "_class_name",
                "base_dim",
                "dim_mult",
                "num_res_blocks",
                "z_dim",
                "temperal_downsample",
                "latents_mean",
                "latents_std",
            ),
            "Qwen-Image VAE",
        )
        if data["_class_name"] != "AutoencoderKLQwenImage":
            raise ValueError("unsupported Qwen-Image VAE architecture")
        if not isinstance(data["dim_mult"], list) or not isinstance(
            data["temperal_downsample"], list
        ):
            raise ValueError("Qwen-Image VAE stage geometry must use lists")
        multipliers = tuple(int(value) for value in data["dim_mult"])
        temporal = tuple(bool(value) for value in data["temperal_downsample"])
        latent_channels = _positive_int(data["z_dim"], "z_dim")
        means = tuple(float(value) for value in data["latents_mean"])
        stds = tuple(float(value) for value in data["latents_std"])
        if not multipliers or any(value <= 0 for value in multipliers):
            raise ValueError("Qwen-Image VAE dimension multipliers must be positive")
        if len(temporal) != len(multipliers) - 1:
            raise ValueError("Qwen-Image VAE resampling stages do not match dim_mult")
        if len(means) != latent_channels or len(stds) != latent_channels:
            raise ValueError("Qwen-Image VAE latent statistics do not match z_dim")
        if any(value <= 0.0 for value in stds):
            raise ValueError(
                "Qwen-Image VAE latent standard deviations must be positive"
            )
        return cls(
            base_dim=_positive_int(data["base_dim"], "base_dim"),
            dimension_multipliers=multipliers,
            residual_blocks=_positive_int(data["num_res_blocks"], "num_res_blocks"),
            latent_channels=latent_channels,
            temporal_downsample=temporal,
            latent_mean=means,
            latent_std=stds,
        )


@dataclass(frozen=True)
class FlowMatchEulerConfig:
    """Qwen-Image's dynamic-shift flow-matching schedule contract."""

    training_steps: int
    base_sequence_length: int
    maximum_sequence_length: int
    base_shift: float
    maximum_shift: float
    terminal_shift: float
    dynamic_shifting: bool
    time_shift_type: str

    def schedule(self, steps: int, image_sequence_length: int):
        """Build the exact official inference sigma schedule."""
        if not self.dynamic_shifting:
            raise ValueError("Qwen-Image requires dynamic flow shifting")
        return flow_match_euler_schedule(
            steps,
            image_sequence_length,
            base_sequence_length=self.base_sequence_length,
            maximum_sequence_length=self.maximum_sequence_length,
            base_shift=self.base_shift,
            maximum_shift=self.maximum_shift,
            terminal_shift=self.terminal_shift,
            time_shift_type=self.time_shift_type,
        )

    @classmethod
    def load(cls, path: str | Path) -> FlowMatchEulerConfig:
        data = _read_json(Path(path))
        _require(
            data,
            (
                "_class_name",
                "num_train_timesteps",
                "base_image_seq_len",
                "max_image_seq_len",
                "base_shift",
                "max_shift",
                "shift_terminal",
                "use_dynamic_shifting",
                "time_shift_type",
            ),
            "Qwen-Image scheduler",
        )
        if data["_class_name"] != "FlowMatchEulerDiscreteScheduler":
            raise ValueError("unsupported Qwen-Image scheduler")
        base_length = _positive_int(data["base_image_seq_len"], "base_image_seq_len")
        maximum_length = _positive_int(data["max_image_seq_len"], "max_image_seq_len")
        base_shift = float(data["base_shift"])
        maximum_shift = float(data["max_shift"])
        terminal = float(data["shift_terminal"])
        shift_type = str(data["time_shift_type"])
        if maximum_length < base_length or not 0.0 < base_shift <= maximum_shift:
            raise ValueError("invalid Qwen-Image dynamic shift range")
        if not 0.0 <= terminal < 1.0:
            raise ValueError("invalid Qwen-Image terminal shift")
        if shift_type not in ("exponential", "linear"):
            raise ValueError("unsupported Qwen-Image time shift type")
        return cls(
            training_steps=_positive_int(
                data["num_train_timesteps"], "num_train_timesteps"
            ),
            base_sequence_length=base_length,
            maximum_sequence_length=maximum_length,
            base_shift=base_shift,
            maximum_shift=maximum_shift,
            terminal_shift=terminal,
            dynamic_shifting=bool(data["use_dynamic_shifting"]),
            time_shift_type=shift_type,
        )


@dataclass(frozen=True)
class QwenImage2512Bundle:
    """Official Diffusers bundle layout, validated without loading tensors."""

    root: Path
    transformer: QwenImageTransformerConfig
    vae: QwenImageVAEConfig
    scheduler: FlowMatchEulerConfig
    transformer_index: SafeTensorIndex
    text_encoder_index: SafeTensorIndex
    text_hidden_size: int
    text_layers: int

    @property
    def image_multiple(self) -> int:
        return self.vae.spatial_scale_factor * self.transformer.patch_size

    def latent_geometry(self, width: int, height: int) -> tuple[int, int, int]:
        width = _positive_int(width, "width")
        height = _positive_int(height, "height")
        if width % self.image_multiple or height % self.image_multiple:
            raise ValueError(
                f"Qwen-Image dimensions must be divisible by {self.image_multiple}"
            )
        latent_width = width // self.vae.spatial_scale_factor
        latent_height = height // self.vae.spatial_scale_factor
        sequence = (latent_width // self.transformer.patch_size) * (
            latent_height // self.transformer.patch_size
        )
        return latent_width, latent_height, sequence

    def missing_weight_files(self) -> tuple[Path, ...]:
        missing = list(self.transformer_index.missing_shards())
        missing.extend(self.text_encoder_index.missing_shards())
        vae_weight = self.root / "vae" / "diffusion_pytorch_model.safetensors"
        if not vae_weight.is_file():
            missing.append(vae_weight)
        return tuple(missing)

    @classmethod
    def inspect(
        cls, path: str | Path, *, require_weights: bool = False
    ) -> QwenImage2512Bundle:
        root = Path(path)
        model_index = _read_json(root / "model_index.json")
        expected = {
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": [
                "transformers",
                "Qwen2_5_VLForConditionalGeneration",
            ],
            "tokenizer": ["transformers", "Qwen2Tokenizer"],
            "transformer": ["diffusers", "QwenImageTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLQwenImage"],
        }
        if model_index.get("_class_name") != "QwenImagePipeline" or any(
            model_index.get(name) != value for name, value in expected.items()
        ):
            raise ValueError("bundle is not an official Qwen-Image-2512 pipeline")
        transformer = QwenImageTransformerConfig.load(
            root / "transformer" / "config.json"
        )
        vae = QwenImageVAEConfig.load(root / "vae" / "config.json")
        scheduler = FlowMatchEulerConfig.load(
            root / "scheduler" / "scheduler_config.json"
        )
        text = _read_json(root / "text_encoder" / "config.json")
        _require(
            text,
            ("model_type", "hidden_size", "num_hidden_layers"),
            "Qwen-Image text encoder",
        )
        if text["model_type"] != "qwen2_5_vl":
            raise ValueError("Qwen-Image text encoder must use Qwen2.5-VL")
        text_hidden = _positive_int(text["hidden_size"], "text hidden size")
        text_layers = _positive_int(text["num_hidden_layers"], "text layers")
        if text_hidden != transformer.text_width:
            raise ValueError("Qwen-Image text and transformer widths do not match")
        if (
            transformer.input_channels
            != vae.latent_channels * transformer.patch_size**2
        ):
            raise ValueError("Qwen-Image packed latent input width is inconsistent")
        if transformer.output_channels != vae.latent_channels:
            raise ValueError("Qwen-Image output and VAE latent widths do not match")
        transformer_index = read_safetensors_index(
            root / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
        )
        text_index = read_safetensors_index(
            root / "text_encoder" / "model.safetensors.index.json"
        )
        bundle = cls(
            root=root,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            transformer_index=transformer_index,
            text_encoder_index=text_index,
            text_hidden_size=text_hidden,
            text_layers=text_layers,
        )
        missing = bundle.missing_weight_files()
        if require_weights and missing:
            raise FileNotFoundError(
                f"Qwen-Image bundle is missing {len(missing)} weight file(s): "
                f"{missing[0]}"
            )
        return bundle
