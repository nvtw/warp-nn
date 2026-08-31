# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image transformer checkpoint names and metadata validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .runner import QwenImageTransformerConfig


class TensorMetadataArchive(Protocol):
    """The format-neutral archive surface needed for checkpoint validation."""

    @property
    def names(self) -> tuple[str, ...]: ...

    def metadata(self, name: str): ...


@dataclass(frozen=True)
class QwenImageTensorSpec:
    name: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


def _linear(specs, name: str, output: int, input: int, *, bias: bool = True):
    specs.append(QwenImageTensorSpec(f"{name}.weight", (output, input)))
    if bias:
        specs.append(QwenImageTensorSpec(f"{name}.bias", (output,)))


@dataclass(frozen=True)
class QwenImageTransformerManifest:
    """Exact Diffusers state-dict contract for a Qwen-Image MMDiT config."""

    tensors: tuple[QwenImageTensorSpec, ...]

    @classmethod
    def from_config(
        cls, config: QwenImageTransformerConfig
    ) -> QwenImageTransformerManifest:
        width = config.hidden_size
        if width <= 0 or config.layers <= 0:
            raise ValueError("Qwen-Image transformer geometry must be positive")

        specs: list[QwenImageTensorSpec] = []
        _linear(specs, "img_in", width, config.input_channels)
        specs.append(QwenImageTensorSpec("txt_norm.weight", (config.text_width,)))
        _linear(specs, "txt_in", width, config.text_width)
        _linear(specs, "time_text_embed.timestep_embedder.linear_1", width, 256)
        _linear(specs, "time_text_embed.timestep_embedder.linear_2", width, width)

        for layer in range(config.layers):
            prefix = f"transformer_blocks.{layer}"
            attention = f"{prefix}.attn"
            for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                _linear(specs, f"{attention}.{projection}", width, width)
            for projection in (
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_add_out",
            ):
                _linear(specs, f"{attention}.{projection}", width, width)
            for norm in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
                specs.append(
                    QwenImageTensorSpec(
                        f"{attention}.{norm}.weight", (config.head_dim,)
                    )
                )
            for stream in ("img", "txt"):
                _linear(specs, f"{prefix}.{stream}_mod.1", 6 * width, width)
                _linear(
                    specs,
                    f"{prefix}.{stream}_mlp.net.0.proj",
                    4 * width,
                    width,
                )
                _linear(
                    specs,
                    f"{prefix}.{stream}_mlp.net.2",
                    width,
                    4 * width,
                )

        _linear(specs, "norm_out.linear", 2 * width, width)
        _linear(
            specs,
            "proj_out",
            config.patch_size**2 * config.output_channels,
            width,
        )
        return cls(tuple(specs))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.tensors)

    @property
    def parameter_count(self) -> int:
        return sum(spec.elements for spec in self.tensors)

    @property
    def bfloat16_bytes(self) -> int:
        return 2 * self.parameter_count

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {spec.name: spec.shape for spec in self.tensors}

    def validate_archive(
        self, archive: TensorMetadataArchive, *, allow_extra: bool = False
    ) -> None:
        """Validate names, shapes, and BF16 storage without loading tensor data."""
        try:
            archive_names = tuple(archive.names)
        except (AttributeError, TypeError) as exc:
            raise TypeError("archive must expose an iterable 'names' property") from exc

        expected = set(self.names)
        present = set(archive_names)
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        if missing:
            raise ValueError(f"Qwen-Image checkpoint is missing tensor '{missing[0]}'")
        if extra and not allow_extra:
            raise ValueError(
                f"Qwen-Image checkpoint has unexpected tensor '{extra[0]}'"
            )

        for spec in self.tensors:
            try:
                metadata = archive.metadata(spec.name)
                shape = tuple(int(value) for value in metadata.shape)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise TypeError(
                    f"archive has invalid metadata for tensor '{spec.name}'"
                ) from exc
            if shape != spec.shape:
                raise ValueError(
                    f"Qwen-Image tensor '{spec.name}' has shape {shape}, "
                    f"expected {spec.shape}"
                )

            format_name = getattr(metadata, "format", None)
            if format_name is None:
                dtype = getattr(metadata, "dtype", None)
                format_name = getattr(dtype, "__name__", str(dtype))
            if str(format_name).upper() not in ("BF16", "BFLOAT16"):
                raise TypeError(
                    f"Qwen-Image tensor '{spec.name}' must use BF16 storage"
                )

            nbytes = getattr(metadata, "nbytes", None)
            if nbytes is not None and int(nbytes) != 2 * spec.elements:
                raise ValueError(
                    f"Qwen-Image tensor '{spec.name}' has inconsistent byte size"
                )
