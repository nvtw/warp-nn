# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step 1.5 diffusion-transformer configuration and sampler foundations.

Only ACE-Step policy belongs here. Dense projections, RMSNorm, SwiGLU, RoPE and
attention execution remain shared runtime operations. The manifest follows the
official AceStepDiTModel and covers both turbo and XL checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import warp as wp

from warp_nn.runtime.kernels import (
    _merge_attention_heads_kernel,
    _rotary_embedding_kernel_for_dtype,
    _split_attention_heads_kernel,
)
from warp_nn.runtime.formats.safetensors import SafeTensorArchive
from warp_nn.runtime.operators import (
    AdaptiveRMSNormPlan,
    BidirectionalGQAPlan,
    FixedKVAttentionPlan,
    ModulatedResidualPlan,
    Operation,
    Conv1dPlan,
    execute_operations,
    plan_linear,
    plan_rms_norm,
    plan_swiglu,
    rotary_cache_values,
)
from warp_nn.runtime.weights import load_cast_weights


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
    encoder_intermediate_size: int
    encoder_num_attention_heads: int
    encoder_num_key_value_heads: int
    layer_types: tuple[str, ...]
    sliding_window: int | None
    rope_theta: float
    rms_norm_eps: float
    attention_bias: bool
    model_version: str
    is_turbo: bool
    text_hidden_dim: int
    timbre_hidden_dim: int
    num_lyric_encoder_hidden_layers: int
    num_timbre_encoder_hidden_layers: int
    num_audio_decoder_hidden_layers: int
    timbre_fix_frame: int

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
        if heads % kv_heads:
            raise ValueError("ACE-Step attention head groups are inconsistent")
        if str(source.get("hidden_act", "silu")) != "silu":
            raise ValueError("ACE-Step DiT requires the SiLU-gated Qwen MLP")
        patch_size = int(source["patch_size"])
        in_channels = int(source["in_channels"])
        audio_channels = int(source.get("audio_acoustic_hidden_dim", in_channels // 3))
        text_width = int(source.get("text_hidden_dim", 1024))
        timbre_width = int(source.get("timbre_hidden_dim", 64))
        encoder_hidden = int(source.get("encoder_hidden_size", hidden))
        encoder_intermediate = int(
            source.get("encoder_intermediate_size", source["intermediate_size"])
        )
        encoder_heads = int(source.get("encoder_num_attention_heads", heads))
        encoder_kv_heads = int(source.get("encoder_num_key_value_heads", kv_heads))
        if (
            encoder_heads % encoder_kv_heads
            or encoder_hidden != encoder_heads * head_dim
        ):
            raise ValueError("ACE-Step condition encoder head geometry is inconsistent")
        lyric_layers = int(source.get("num_lyric_encoder_hidden_layers", 8))
        timbre_layers = int(source.get("num_timbre_encoder_hidden_layers", 4))
        audio_decoder_layers = int(
            source.get("num_audio_decoder_hidden_layers", layers)
        )
        timbre_frames = int(source.get("timbre_fix_frame", 750))
        if (
            min(
                patch_size,
                in_channels,
                audio_channels,
                text_width,
                timbre_width,
                lyric_layers,
                timbre_layers,
                audio_decoder_layers,
                timbre_frames,
            )
            <= 0
        ):
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
            encoder_hidden_size=encoder_hidden,
            encoder_intermediate_size=encoder_intermediate,
            encoder_num_attention_heads=encoder_heads,
            encoder_num_key_value_heads=encoder_kv_heads,
            layer_types=layer_types,
            sliding_window=window,
            rope_theta=float(source.get("rope_theta", 1_000_000.0)),
            rms_norm_eps=float(source.get("rms_norm_eps", 1.0e-6)),
            attention_bias=bool(source.get("attention_bias", False)),
            model_version=str(source.get("model_version", "turbo")),
            is_turbo=bool(
                source.get("is_turbo", source.get("model_version", "turbo") == "turbo")
            ),
            text_hidden_dim=text_width,
            timbre_hidden_dim=timbre_width,
            num_lyric_encoder_hidden_layers=lyric_layers,
            num_timbre_encoder_hidden_layers=timbre_layers,
            num_audio_decoder_hidden_layers=audio_decoder_layers,
            timbre_fix_frame=timbre_frames,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "AceStepDiTConfig":
        """Load one DiT config file (or component directory)."""
        path = Path(path)
        if path.is_dir():
            path /= "config.json"
        source = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise ValueError(f"ACE-Step JSON object expected in {path}")
        required = (
            "model_type",
            "model_version",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "in_channels",
            "text_hidden_dim",
            "num_lyric_encoder_hidden_layers",
            "num_timbre_encoder_hidden_layers",
            "num_audio_decoder_hidden_layers",
            "patch_size",
            "layer_types",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"ACE-Step DiT config is missing {missing}")
        if source.get("model_type") != "acestep":
            raise ValueError("ACE-Step DiT config has an incompatible model_type")
        return cls.from_dict(source)

    load = from_file

    # Compatibility names used by bundle discovery and older callers. Keeping
    # these as properties avoids maintaining a second config representation.
    @property
    def layers(self) -> int:
        return self.num_hidden_layers

    @property
    def query_heads(self) -> int:
        return self.num_attention_heads

    @property
    def kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def input_channels(self) -> int:
        return self.in_channels

    @property
    def text_hidden_size(self) -> int:
        return self.text_hidden_dim

    @property
    def lyric_layers(self) -> int:
        return self.num_lyric_encoder_hidden_layers

    @property
    def timbre_layers(self) -> int:
        return self.num_timbre_encoder_hidden_layers

    @property
    def audio_decoder_layers(self) -> int:
        return self.num_audio_decoder_hidden_layers

    @property
    def condition_encoder(self) -> "AceStepDiTConfig":
        """Return the condition-transformer geometry used by XL and turbo."""
        return replace(
            self,
            hidden_size=self.encoder_hidden_size,
            intermediate_size=self.encoder_intermediate_size,
            num_attention_heads=self.encoder_num_attention_heads,
            num_key_value_heads=self.encoder_num_key_value_heads,
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


class AceStepAttentionPlan:
    """Executable official ACE attention block with optimized projections.

    Self-attention applies per-head Q/K RMSNorm and split-half RoPE before the
    reusable bidirectional GQA core. Cross-attention projects and normalizes
    condition K/V once; repeated diffusion steps only recompute Q and output.
    """

    def __init__(
        self,
        hidden,
        weights,
        prefix,
        config,
        *,
        context=None,
        query_valid=None,
        key_valid=None,
        position_ids=None,
        cos_cache=None,
        sin_cache=None,
        layer_index=0,
        cublas=None,
    ):
        if hidden.ndim != 3 or hidden.shape[2] != config.hidden_size:
            raise ValueError("ACE attention hidden input has incompatible shape")
        if hidden.dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("ACE attention requires FP16 or BF16 input")
        self.hidden = hidden
        self.context = context
        self.cross_attention = context is not None
        if self.cross_attention:
            if (
                context.ndim != 3
                or context.shape[0] != hidden.shape[0]
                or context.shape[2] != config.hidden_size
                or context.dtype != hidden.dtype
                or context.device != hidden.device
            ):
                raise ValueError("ACE cross-attention context is incompatible")
        elif any(value is None for value in (position_ids, cos_cache, sin_cache)):
            raise ValueError("ACE self-attention requires positions and RoPE caches")
        if config.attention_bias:
            raise ValueError("biased ACE attention projections are not yet supported")
        if not 0 <= layer_index < config.num_hidden_layers:
            raise IndexError("ACE attention layer index is out of range")
        self.device = hidden.device
        self.config = config
        self.prefix = prefix
        self.weights = weights
        self.tensors = dict(weights)
        self.shapes = {name: tuple(value.shape) for name, value in weights.items()}
        batch, query_length, hidden_size = hidden.shape
        key_source = context if self.cross_attention else hidden
        key_length = key_source.shape[1]
        self.tensors["hidden"] = hidden.reshape((batch * query_length, hidden_size))
        self.shapes["hidden"] = self.tensors["hidden"].shape
        self.tensors["key_source"] = key_source.reshape(
            (batch * key_length, hidden_size)
        )
        self.shapes["key_source"] = self.tensors["key_source"].shape

        def linear(name, source, weight):
            operation = Operation("Linear", [source, weight], [name])
            plan_linear(
                operation,
                self.tensors,
                self.shapes,
                self.device,
                cublas=cublas,
            )
            return operation

        self.q_projection = linear("q_projected", "hidden", prefix + ".q_proj.weight")
        self.k_projection = linear(
            "k_projected", "key_source", prefix + ".k_proj.weight"
        )
        self.v_projection = linear(
            "v_projected", "key_source", prefix + ".v_proj.weight"
        )
        q_shape = (
            batch,
            config.num_attention_heads,
            query_length,
            config.head_dim,
        )
        kv_shape = (
            batch,
            config.num_key_value_heads,
            key_length,
            config.head_dim,
        )
        self.query = wp.empty(q_shape, dtype=hidden.dtype, device=self.device)
        self.key = wp.empty(kv_shape, dtype=hidden.dtype, device=self.device)
        self.value = wp.empty_like(self.key)
        self.tensors["query"] = self.query
        self.shapes["query"] = q_shape
        self.tensors["key"] = self.key
        self.shapes["key"] = kv_shape
        self.q_norm = Operation(
            "SimplifiedLayerNormalization",
            ["query", prefix + ".q_norm.weight"],
            ["query_norm"],
            {"epsilon": config.rms_norm_eps},
        )
        self.k_norm = Operation(
            "SimplifiedLayerNormalization",
            ["key", prefix + ".k_norm.weight"],
            ["key_norm"],
            {"epsilon": config.rms_norm_eps},
        )
        plan_rms_norm(self.q_norm, self.tensors, self.shapes, self.device)
        plan_rms_norm(self.k_norm, self.tensors, self.shapes, self.device)
        self.position_ids = position_ids
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        if self.cross_attention:
            attention_query = self.tensors["query_norm"]
            attention_key = self.tensors["key_norm"]
            self.attention = FixedKVAttentionPlan(
                attention_query,
                attention_key,
                self.value,
                query_valid=query_valid,
                key_valid=key_valid,
            )
        else:
            self.rotated_query = wp.empty_like(self.query)
            self.rotated_key = wp.empty_like(self.key)
            self._rotary = _rotary_embedding_kernel_for_dtype(hidden.dtype)
            window = (
                config.sliding_window
                if config.layer_types[layer_index] == "sliding_attention"
                else None
            )
            self.attention = BidirectionalGQAPlan(
                self.rotated_query,
                self.rotated_key,
                self.value,
                query_valid=query_valid,
                key_valid=key_valid,
                window=window,
            )
        attention_width = config.num_attention_heads * config.head_dim
        self.merged = wp.empty(
            (batch, query_length, attention_width),
            dtype=hidden.dtype,
            device=self.device,
        )
        self.tensors["merged"] = self.merged.reshape(
            (batch * query_length, attention_width)
        )
        self.shapes["merged"] = self.tensors["merged"].shape
        self.output_projection = linear("output", "merged", prefix + ".o_proj.weight")
        self.output = self.tensors["output"].reshape((batch, query_length, hidden_size))
        self._fixed_kv_ready = False

    def _execute(self, operation):
        execute_operations((operation,), self.tensors, self.shapes, self.device)

    def _prepare_kv(self):
        self._execute(self.k_projection)
        self._execute(self.v_projection)
        for projected, output, heads in (
            (self.tensors["k_projected"], self.key, self.config.num_key_value_heads),
            (self.tensors["v_projected"], self.value, self.config.num_key_value_heads),
        ):
            wp.launch(
                _split_attention_heads_kernel,
                dim=output.shape,
                inputs=[
                    projected.reshape(
                        (
                            self.hidden.shape[0],
                            output.shape[2],
                            heads * self.config.head_dim,
                        )
                    ),
                    output,
                ],
                device=self.device,
            )
        self._execute(self.k_norm)
        if not self.cross_attention:
            wp.launch(
                self._rotary,
                dim=self.rotated_key.shape,
                inputs=[
                    self.tensors["key_norm"],
                    self.position_ids,
                    self.cos_cache,
                    self.sin_cache,
                    self.rotated_key,
                    self.config.head_dim,
                    False,
                    False,
                ],
                device=self.device,
            )

    def prepare_fixed_kv(self, force=False):
        """Project condition K/V once before graph capture."""
        if not self.cross_attention:
            raise RuntimeError("fixed K/V preparation is only for cross-attention")
        if force or not self._fixed_kv_ready:
            self._prepare_kv()
            self._fixed_kv_ready = True

    def execute(self):
        self._execute(self.q_projection)
        wp.launch(
            _split_attention_heads_kernel,
            dim=self.query.shape,
            inputs=[
                self.tensors["q_projected"].reshape(
                    (
                        self.hidden.shape[0],
                        self.hidden.shape[1],
                        self.config.num_attention_heads * self.config.head_dim,
                    )
                ),
                self.query,
            ],
            device=self.device,
        )
        self._execute(self.q_norm)
        if self.cross_attention:
            self.prepare_fixed_kv()
        else:
            self._prepare_kv()
            wp.launch(
                self._rotary,
                dim=self.rotated_query.shape,
                inputs=[
                    self.tensors["query_norm"],
                    self.position_ids,
                    self.cos_cache,
                    self.sin_cache,
                    self.rotated_query,
                    self.config.head_dim,
                    False,
                    False,
                ],
                device=self.device,
            )
        attention_output = self.attention.execute()
        wp.launch(
            _merge_attention_heads_kernel,
            dim=attention_output.shape,
            inputs=[attention_output, self.merged],
            device=self.device,
        )
        self._execute(self.output_projection)
        return self.output


class AceStepDiTLayerPlan:
    """One exact ACE-Step AdaLN/self/cross/SwiGLU transformer layer."""

    def __init__(
        self,
        hidden,
        timestep_modulation,
        context,
        weights,
        config,
        layer_index,
        *,
        query_valid=None,
        context_valid=None,
        position_ids=None,
        cos_cache=None,
        sin_cache=None,
        cublas=None,
    ):
        if timestep_modulation.shape != (
            hidden.shape[0],
            6,
            config.hidden_size,
        ):
            raise ValueError("ACE layer timestep modulation must be [batch, 6, hidden]")
        prefix = f"decoder.layers.{layer_index}"
        table = weights[prefix + ".scale_shift_table"]
        self.device = hidden.device
        self.weights = weights
        self.self_norm = AdaptiveRMSNormPlan(
            hidden,
            weights[prefix + ".self_attn_norm.weight"],
            table,
            timestep_modulation,
            shift_index=0,
            scale_index=1,
            epsilon=config.rms_norm_eps,
        )
        self.self_attention = AceStepAttentionPlan(
            self.self_norm.output,
            weights,
            prefix + ".self_attn",
            config,
            query_valid=query_valid,
            key_valid=query_valid,
            position_ids=position_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            layer_index=layer_index,
            cublas=cublas,
        )
        self.self_residual = ModulatedResidualPlan(
            hidden,
            self.self_attention.output,
            scale_shift_table=table,
            timestep_modulation=timestep_modulation,
            gate_index=2,
        )
        self._cross_tensors = {
            "x": self.self_residual.output,
            "weight": weights[prefix + ".cross_attn_norm.weight"],
        }
        self._cross_shapes = {
            name: value.shape for name, value in self._cross_tensors.items()
        }
        self._cross_norm = Operation(
            "SimplifiedLayerNormalization",
            ["x", "weight"],
            ["normalized"],
            {"epsilon": config.rms_norm_eps},
        )
        plan_rms_norm(
            self._cross_norm,
            self._cross_tensors,
            self._cross_shapes,
            self.device,
        )
        self.cross_attention = AceStepAttentionPlan(
            self._cross_tensors["normalized"],
            weights,
            prefix + ".cross_attn",
            config,
            context=context,
            query_valid=query_valid,
            key_valid=context_valid,
            layer_index=layer_index,
            cublas=cublas,
        )
        self.cross_residual = ModulatedResidualPlan(
            self.self_residual.output, self.cross_attention.output
        )
        self.mlp_norm = AdaptiveRMSNormPlan(
            self.cross_residual.output,
            weights[prefix + ".mlp_norm.weight"],
            table,
            timestep_modulation,
            shift_index=3,
            scale_index=4,
            epsilon=config.rms_norm_eps,
        )
        rows = hidden.shape[0] * hidden.shape[1]
        self._mlp_tensors = dict(weights)
        self._mlp_tensors["x"] = self.mlp_norm.output.reshape(
            (rows, config.hidden_size)
        )
        self._mlp_shapes = {
            name: tuple(value.shape) for name, value in self._mlp_tensors.items()
        }

        def linear(name, source, weight):
            operation = Operation("Linear", [source, weight], [name])
            plan_linear(
                operation,
                self._mlp_tensors,
                self._mlp_shapes,
                self.device,
                cublas=cublas,
            )
            return operation

        self.gate = linear("gate", "x", prefix + ".mlp.gate_proj.weight")
        self.up = linear("up", "x", prefix + ".mlp.up_proj.weight")
        self.swiglu = Operation("_SwiGLU", ["gate", "up"], ["activated"])
        plan_swiglu(self.swiglu, self._mlp_tensors, self._mlp_shapes, self.device)
        self.down = linear("down", "activated", prefix + ".mlp.down_proj.weight")
        down = self._mlp_tensors["down"].reshape(hidden.shape)
        self.mlp_residual = ModulatedResidualPlan(
            self.cross_residual.output,
            down,
            scale_shift_table=table,
            timestep_modulation=timestep_modulation,
            gate_index=5,
        )
        self.output = self.mlp_residual.output

    def prepare_fixed_condition(self, force=False):
        """Project the condition K/V once for repeated diffusion steps."""
        self.cross_attention.prepare_fixed_kv(force=force)

    def execute(self):
        self.self_norm.execute()
        self.self_attention.execute()
        self.self_residual.execute()
        execute_operations(
            (self._cross_norm,),
            self._cross_tensors,
            self._cross_shapes,
            self.device,
        )
        self.cross_attention.execute()
        self.cross_residual.execute()
        self.mlp_norm.execute()
        execute_operations(
            (self.gate, self.up, self.swiglu, self.down),
            self._mlp_tensors,
            self._mlp_shapes,
            self.device,
        )
        self.mlp_residual.execute()
        return self.output


@lru_cache(maxsize=None)
def _ace_dit_kernels(dtype):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def pack_latents(
        hidden: wp.array3d(dtype=DTYPE),
        context: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
    ):
        batch, frame, channel = wp.tid()
        if frame >= hidden.shape[1]:
            output[batch, frame, channel] = DTYPE(0.0)
        elif channel < context.shape[2]:
            output[batch, frame, channel] = context[batch, frame, channel]
        else:
            output[batch, frame, channel] = hidden[
                batch, frame, channel - context.shape[2]
            ]

    @wp.kernel(enable_backward=False, module="unique")
    def bias(x: wp.array2d(dtype=DTYPE), b: wp.array1d(dtype=DTYPE)):
        row, column = wp.tid()
        x[row, column] = DTYPE(wp.float32(x[row, column]) + wp.float32(b[column]))

    @wp.kernel(enable_backward=False, module="unique")
    def silu(x: wp.array2d(dtype=DTYPE), output: wp.array2d(dtype=DTYPE)):
        row, column = wp.tid()
        value = wp.float32(x[row, column])
        output[row, column] = DTYPE(value / (wp.float32(1.0) + wp.exp(-value)))

    @wp.kernel(enable_backward=False, module="unique")
    def timestep_frequency(
        timestep: wp.array1d(dtype=DTYPE),
        timestep_r: wp.array1d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        difference: bool,
    ):
        batch, column = wp.tid()
        half = output.shape[1] / 2
        if column >= half * 2:
            output[batch, column] = DTYPE(0.0)
        else:
            frequency_column = column % half
            frequency = wp.exp(
                -wp.log(wp.float32(10000.0))
                * wp.float32(frequency_column)
                / wp.float32(half)
            )
            value = wp.float32(timestep[batch])
            if difference:
                value -= wp.float32(timestep_r[batch])
            angle = value * wp.float32(1000.0) * frequency
            output[batch, column] = DTYPE(
                wp.cos(angle) if column < half else wp.sin(angle)
            )

    @wp.kernel(enable_backward=False, module="unique")
    def combine(
        left: wp.array2d(dtype=DTYPE),
        right: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        output[row, column] = DTYPE(
            wp.float32(left[row, column]) + wp.float32(right[row, column])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def repeat_rows(x: wp.array2d(dtype=DTYPE), output: wp.array3d(dtype=DTYPE)):
        typed_zero = DTYPE(0.0)
        batch, row, column = wp.tid()
        output[batch, row, column] = DTYPE(
            wp.float32(x[batch, column]) + wp.float32(typed_zero)
        )

    @wp.kernel(enable_backward=False, module="unique")
    def crop(x: wp.array3d(dtype=DTYPE), output: wp.array3d(dtype=DTYPE)):
        typed_zero = DTYPE(0.0)
        batch, frame, channel = wp.tid()
        output[batch, frame, channel] = DTYPE(
            wp.float32(x[batch, frame, channel]) + wp.float32(typed_zero)
        )

    @wp.kernel(enable_backward=False, module="unique")
    def flow_step(
        latent: wp.array3d(dtype=DTYPE),
        velocity: wp.array3d(dtype=DTYPE),
        timestep: wp.array1d(dtype=DTYPE),
        next_timestep: wp.array1d(dtype=DTYPE),
    ):
        batch, frame, channel = wp.tid()
        dt = wp.float32(timestep[batch]) - wp.float32(next_timestep[batch])
        latent[batch, frame, channel] = DTYPE(
            wp.float32(latent[batch, frame, channel])
            - wp.float32(velocity[batch, frame, channel]) * dt
        )

    return (
        pack_latents,
        bias,
        silu,
        timestep_frequency,
        combine,
        repeat_rows,
        crop,
        flow_step,
    )


@lru_cache(maxsize=None)
def _apg_flow_kernels(dtype):
    """Create deterministic APG guidance fused with the flow update."""
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def update_momentum(
        prediction: wp.array3d(dtype=DTYPE),
        momentum: wp.array3d(dtype=DTYPE),
        active: wp.array1d(dtype=wp.int32),
    ):
        batch, frame, channel = wp.tid()
        if active[0] != 0:
            difference = wp.float32(prediction[batch, frame, channel]) - wp.float32(
                prediction[batch + momentum.shape[0], frame, channel]
            )
            momentum[batch, frame, channel] = DTYPE(
                difference
                - wp.float32(0.75) * wp.float32(momentum[batch, frame, channel])
            )

    @wp.kernel(enable_backward=False, module="unique")
    def statistics(
        prediction: wp.array3d(dtype=DTYPE),
        momentum: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=wp.float64),
    ):
        batch, channel = wp.tid()
        difference_norm = wp.float64(0.0)
        condition_norm = wp.float64(0.0)
        typed_zero = DTYPE(0.0)
        dot = wp.float64(0.0)
        for frame in range(momentum.shape[1]):
            difference = wp.float64(momentum[batch, frame, channel] + typed_zero)
            condition = wp.float64(prediction[batch, frame, channel])
            difference_norm += difference * difference
            condition_norm += condition * condition
            dot += difference * condition
        output[batch, channel, 0] = difference_norm
        output[batch, channel, 1] = condition_norm
        output[batch, channel, 2] = dot

    @wp.kernel(enable_backward=False, module="unique")
    def guide_and_step(
        latent: wp.array3d(dtype=DTYPE),
        prediction: wp.array3d(dtype=DTYPE),
        momentum: wp.array3d(dtype=DTYPE),
        stats: wp.array3d(dtype=wp.float64),
        timestep: wp.array1d(dtype=DTYPE),
        next_timestep: wp.array1d(dtype=DTYPE),
        guidance: wp.array1d(dtype=wp.float32),
    ):
        batch, frame, channel = wp.tid()
        scale = guidance[0]
        condition = wp.float32(prediction[batch, frame, channel])
        velocity = condition
        if scale != wp.float32(1.0):
            difference_norm = wp.sqrt(stats[batch, channel, 0])
            clip = wp.min(
                wp.float64(1.0),
                wp.float64(2.5) / wp.max(difference_norm, wp.float64(1.0e-20)),
            )
            projection = (
                clip
                * stats[batch, channel, 2]
                / wp.max(stats[batch, channel, 1], wp.float64(1.0e-20))
            )
            update = wp.float32(clip) * wp.float32(momentum[batch, frame, channel])
            update -= wp.float32(projection) * condition
            velocity += (scale - wp.float32(1.0)) * update
        delta = wp.float32(timestep[batch]) - wp.float32(next_timestep[batch])
        value = DTYPE(wp.float32(latent[batch, frame, channel]) - velocity * delta)
        latent[batch, frame, channel] = value
        latent[batch + momentum.shape[0], frame, channel] = value

    return update_momentum, statistics, guide_and_step


class _TimeEmbeddingPlan:
    def __init__(
        self, timestep, timestep_r, weights, prefix, hidden, difference, cublas
    ):
        self.device = timestep.device
        self.timestep = timestep
        self.timestep_r = timestep_r
        self.difference = difference
        self.weights = weights
        self.tensors = dict(weights)
        batch = timestep.shape[0]
        self.frequency = wp.empty(
            (batch, 256), dtype=timestep.dtype, device=self.device
        )
        self.tensors["frequency"] = self.frequency
        self.shapes = {name: tuple(value.shape) for name, value in self.tensors.items()}

        def linear(name, source, suffix):
            operation = Operation(
                "Linear", [source, prefix + suffix + ".weight"], [name]
            )
            plan_linear(
                operation, self.tensors, self.shapes, self.device, cublas=cublas
            )
            return operation

        self.linear1 = linear("linear1", "frequency", ".linear_1")
        self.first_activation = wp.empty_like(self.tensors["linear1"])
        self.tensors["first_activation"] = self.first_activation
        self.shapes["first_activation"] = self.first_activation.shape
        self.linear2 = linear("temb", "first_activation", ".linear_2")
        self.activated = wp.empty(
            (batch, hidden), dtype=timestep.dtype, device=self.device
        )
        self.tensors["activated"] = self.activated
        self.shapes["activated"] = self.activated.shape
        self.projection = linear("projection", "activated", ".time_proj")
        self.temb = self.tensors["temb"]
        self.modulation = self.tensors["projection"].reshape((batch, 6, hidden))
        self._kernels = _ace_dit_kernels(timestep.dtype)
        self.prefix = prefix

    def execute(self):
        wp.launch(
            self._kernels[3],
            dim=self.frequency.shape,
            inputs=[self.timestep, self.timestep_r, self.frequency, self.difference],
            device=self.device,
        )
        execute_operations((self.linear1,), self.tensors, self.shapes, self.device)
        wp.launch(
            self._kernels[1],
            dim=self.tensors["linear1"].shape,
            inputs=[
                self.tensors["linear1"],
                self.weights[self.prefix + ".linear_1.bias"],
            ],
            device=self.device,
        )
        wp.launch(
            self._kernels[2],
            dim=self.first_activation.shape,
            inputs=[self.tensors["linear1"], self.first_activation],
            device=self.device,
        )
        execute_operations((self.linear2,), self.tensors, self.shapes, self.device)
        wp.launch(
            self._kernels[1],
            dim=self.temb.shape,
            inputs=[self.temb, self.weights[self.prefix + ".linear_2.bias"]],
            device=self.device,
        )
        wp.launch(
            self._kernels[2],
            dim=self.activated.shape,
            inputs=[self.temb, self.activated],
            device=self.device,
        )
        execute_operations((self.projection,), self.tensors, self.shapes, self.device)
        wp.launch(
            self._kernels[1],
            dim=self.tensors["projection"].shape,
            inputs=[
                self.tensors["projection"],
                self.weights[self.prefix + ".time_proj.bias"],
            ],
            device=self.device,
        )


class AceStepDiTPlan:
    """Fixed-shape, graph-capturable ACE-Step 1.5 diffusion transformer."""

    def __init__(
        self,
        hidden,
        context_latents,
        condition,
        weights,
        config,
        *,
        condition_valid=None,
        cublas=None,
    ):
        if hidden.shape[:2] != context_latents.shape[:2] or (
            hidden.shape[2] != config.audio_acoustic_hidden_dim
            or context_latents.shape[2] + hidden.shape[2] != config.in_channels
        ):
            raise ValueError("ACE DiT latent and context shapes are incompatible")
        if condition.shape[0] != hidden.shape[0] or (
            condition.shape[2] != config.encoder_hidden_size
        ):
            raise ValueError("ACE DiT condition shape is incompatible")
        self.device = hidden.device
        self.dtype = hidden.dtype
        self.config = config
        self.hidden = hidden
        self.context_latents = context_latents
        self.weights = weights
        batch, frames, _ = hidden.shape
        padded = (
            (frames + config.patch_size - 1) // config.patch_size
        ) * config.patch_size
        self.packed = wp.empty(
            (batch, padded, config.in_channels), dtype=self.dtype, device=self.device
        )
        self.timestep = wp.ones(batch, dtype=self.dtype, device=self.device)
        self.timestep_r = wp.ones(batch, dtype=self.dtype, device=self.device)
        self.next_timestep = wp.zeros(batch, dtype=self.dtype, device=self.device)
        self._kernels = _ace_dit_kernels(self.dtype)
        self.proj_in = Conv1dPlan(
            self.packed,
            weights["decoder.proj_in.1.weight"],
            weights["decoder.proj_in.1.bias"],
            stride=config.patch_size,
        )
        condition_tensors = dict(weights)
        condition_tensors["condition"] = condition.reshape(
            (-1, config.encoder_hidden_size)
        )
        condition_shapes = {
            name: tuple(value.shape) for name, value in condition_tensors.items()
        }
        self._condition_tensors = condition_tensors
        self._condition_shapes = condition_shapes
        self.condition_projection = Operation(
            "Linear", ["condition", "decoder.condition_embedder.weight"], ["projected"]
        )
        plan_linear(
            self.condition_projection,
            condition_tensors,
            condition_shapes,
            self.device,
            cublas=cublas,
        )
        self.condition = condition_tensors["projected"].reshape(
            (batch, condition.shape[1], config.hidden_size)
        )
        self.time_t = _TimeEmbeddingPlan(
            self.timestep,
            self.timestep_r,
            weights,
            "decoder.time_embed",
            config.hidden_size,
            False,
            cublas,
        )
        self.time_r = _TimeEmbeddingPlan(
            self.timestep,
            self.timestep_r,
            weights,
            "decoder.time_embed_r",
            config.hidden_size,
            True,
            cublas,
        )
        self.temb = wp.empty(
            (batch, config.hidden_size), dtype=self.dtype, device=self.device
        )
        self.modulation = wp.empty(
            (batch, 6, config.hidden_size), dtype=self.dtype, device=self.device
        )
        positions = wp.array(
            np.broadcast_to(
                np.arange(self.proj_in.output.shape[1], dtype=np.int64),
                (batch, self.proj_in.output.shape[1]),
            ).copy(),
            device=self.device,
        )
        cos, sin = rotary_cache_values(
            self.proj_in.output.shape[1],
            config.head_dim,
            {"rope_theta": config.rope_theta},
        )
        cos_cache = wp.array(cos, dtype=self.dtype, device=self.device)
        sin_cache = wp.array(sin, dtype=self.dtype, device=self.device)
        self.layers = []
        current = self.proj_in.output
        for index in range(config.num_hidden_layers):
            layer = AceStepDiTLayerPlan(
                current,
                self.modulation,
                self.condition,
                weights,
                config,
                index,
                context_valid=condition_valid,
                position_ids=positions,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                cublas=cublas,
            )
            self.layers.append(layer)
            current = layer.output
        self.output_modulation = wp.empty(
            (batch, 2, config.hidden_size), dtype=self.dtype, device=self.device
        )
        self.output_norm = AdaptiveRMSNormPlan(
            current,
            weights["decoder.norm_out.weight"],
            weights["decoder.scale_shift_table"],
            self.output_modulation,
            shift_index=0,
            scale_index=1,
            epsilon=config.rms_norm_eps,
        )
        self.proj_out = Conv1dPlan(
            self.output_norm.output,
            weights["decoder.proj_out.1.weight"],
            weights["decoder.proj_out.1.bias"],
            stride=config.patch_size,
            transposed=True,
        )
        self.output = wp.empty(hidden.shape, dtype=self.dtype, device=self.device)
        self.graph = None

    def prepare_fixed_condition(self):
        execute_operations(
            (self.condition_projection,),
            self._condition_tensors,
            self._condition_shapes,
            self.device,
        )
        wp.launch(
            self._kernels[1],
            dim=self._condition_tensors["projected"].shape,
            inputs=[
                self._condition_tensors["projected"],
                self.weights["decoder.condition_embedder.bias"],
            ],
            device=self.device,
        )
        for layer in self.layers:
            layer.prepare_fixed_condition(force=True)

    def execute(self):
        wp.launch(
            self._kernels[0],
            dim=self.packed.shape,
            inputs=[self.hidden, self.context_latents, self.packed],
            device=self.device,
        )
        self.proj_in.execute()
        self.time_t.execute()
        self.time_r.execute()
        wp.launch(
            self._kernels[4],
            dim=self.temb.shape,
            inputs=[self.time_t.temb, self.time_r.temb, self.temb],
            device=self.device,
        )
        wp.launch(
            self._kernels[4],
            dim=self.modulation.reshape((self.modulation.shape[0], -1)).shape,
            inputs=[
                self.time_t.modulation.reshape((self.modulation.shape[0], -1)),
                self.time_r.modulation.reshape((self.modulation.shape[0], -1)),
                self.modulation.reshape((self.modulation.shape[0], -1)),
            ],
            device=self.device,
        )
        for layer in self.layers:
            layer.execute()
        wp.launch(
            self._kernels[5],
            dim=self.output_modulation.shape,
            inputs=[self.temb, self.output_modulation],
            device=self.device,
        )
        self.output_norm.execute()
        self.proj_out.execute()
        wp.launch(
            self._kernels[6],
            dim=self.output.shape,
            inputs=[self.proj_out.output, self.output],
            device=self.device,
        )
        return self.output

    def capture(self):
        self.prepare_fixed_condition()
        self.execute()
        wp.launch(
            self._kernels[7],
            dim=self.hidden.shape,
            inputs=[self.hidden, self.output, self.timestep, self.timestep],
            device=self.device,
        )
        wp.synchronize_stream(wp.get_stream(self.device))
        wp.capture_begin(device=self.device)
        self.execute()
        wp.launch(
            self._kernels[7],
            dim=self.hidden.shape,
            inputs=[
                self.hidden,
                self.output,
                self.timestep,
                self.next_timestep,
            ],
            device=self.device,
        )
        self.graph = wp.capture_end(device=self.device)
        return self.graph

    def diffusion_step(self, timestep, next_timestep):
        self.timestep.assign(np.full(self.timestep.shape, timestep, dtype=np.float32))
        self.timestep_r.assign(
            np.full(self.timestep_r.shape, timestep, dtype=np.float32)
        )
        self.next_timestep.assign(
            np.full(self.next_timestep.shape, next_timestep, dtype=np.float32)
        )
        if self.graph is None:
            self.capture()
        wp.capture_launch(self.graph)
        return self.hidden

    def run_schedule(self, schedule):
        """Run a descending turbo/base flow schedule through one captured graph."""
        values = tuple(float(value) for value in schedule)
        if (
            not values
            or any(not 0.0 < value <= 1.0 for value in values)
            or any(left <= right for left, right in zip(values, values[1:]))
        ):
            raise ValueError(
                "ACE diffusion schedule must be strictly descending in (0, 1]"
            )
        for index, value in enumerate(values):
            next_value = values[index + 1] if index + 1 < len(values) else 0.0
            self.diffusion_step(value, next_value)
        return self.hidden


class AceStepGuidedDiTPlan:
    """Batch-two XL-SFT DiT with official deterministic APG guidance."""

    def __init__(
        self,
        hidden,
        context_latents,
        condition,
        null_condition,
        weights,
        config,
        *,
        condition_valid=None,
        cublas=None,
        guidance_scale=7.0,
        guidance_start=0.0,
        guidance_end=1.0,
    ):
        if condition.shape != null_condition.shape:
            raise ValueError(
                "ACE conditional and null embeddings must have equal shapes"
            )
        if not 0.0 <= guidance_start <= guidance_end <= 1.0:
            raise ValueError("ACE guidance interval must be within [0, 1]")
        self.batch = hidden.shape[0]
        self.guidance_scale = float(guidance_scale)
        self.guidance_start = float(guidance_start)
        self.guidance_end = float(guidance_end)
        self.device = hidden.device
        self.dtype = hidden.dtype

        def paired(source, second=None):
            second = source if second is None else second
            output = wp.empty(
                (source.shape[0] * 2, *source.shape[1:]),
                dtype=source.dtype,
                device=source.device,
            )
            wp.copy(output.flatten(), source.flatten(), count=source.size)
            wp.copy(
                output.flatten(),
                second.flatten(),
                dest_offset=source.size,
                count=source.size,
            )
            return output

        model_hidden = paired(hidden)
        model_context = paired(context_latents)
        model_condition = paired(condition, null_condition)
        model_valid = paired(condition_valid) if condition_valid is not None else None
        self.plan = AceStepDiTPlan(
            model_hidden,
            model_context,
            model_condition,
            weights,
            config,
            condition_valid=model_valid,
            cublas=cublas,
        )
        self.hidden = self.plan.hidden[: self.batch]
        self.momentum = wp.zeros_like(hidden)
        self.statistics = wp.empty(
            (self.batch, hidden.shape[2], 3), dtype=wp.float64, device=self.device
        )
        self.active = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.guidance = wp.ones(1, dtype=wp.float32, device=self.device)
        self._kernels = _apg_flow_kernels(self.dtype)
        self.graph = None

    def _execute(self):
        self.plan.execute()
        wp.launch(
            self._kernels[0],
            dim=self.momentum.shape,
            inputs=[self.plan.output, self.momentum, self.active],
            device=self.device,
        )
        wp.launch(
            self._kernels[1],
            dim=(self.batch, self.momentum.shape[2]),
            inputs=[self.plan.output, self.momentum, self.statistics],
            device=self.device,
        )
        wp.launch(
            self._kernels[2],
            dim=self.momentum.shape,
            inputs=[
                self.plan.hidden,
                self.plan.output,
                self.momentum,
                self.statistics,
                self.plan.timestep,
                self.plan.next_timestep,
                self.guidance,
            ],
            device=self.device,
        )

    def capture(self):
        """Prepare fixed cross-attention state and capture one guided flow step."""
        self.plan.prepare_fixed_condition()
        ones = np.ones(self.plan.timestep.shape, dtype=np.float32)
        self.plan.timestep.assign(ones)
        self.plan.timestep_r.assign(ones)
        self.plan.next_timestep.assign(ones)
        self.active.zero_()
        self.guidance.fill_(1.0)
        self._execute()
        wp.synchronize_stream(wp.get_stream(self.device))
        wp.capture_begin(device=self.device)
        self._execute()
        self.graph = wp.capture_end(device=self.device)
        return self.graph

    def run_schedule(self, schedule):
        """Run checkpoint-native APG through one captured GPU graph."""
        values = tuple(float(value) for value in schedule)
        if (
            not values
            or any(not 0.0 < value <= 1.0 for value in values)
            or any(left <= right for left, right in zip(values, values[1:]))
        ):
            raise ValueError(
                "ACE diffusion schedule must be strictly descending in (0, 1]"
            )
        if self.graph is None:
            self.capture()
        count = len(values)
        for index, value in enumerate(values):
            next_value = values[index + 1] if index + 1 < count else 0.0
            active = (
                self.guidance_start <= value <= self.guidance_end
                and self.guidance_scale not in (0.0, 1.0)
            )
            scale = 1.0
            if active:
                scale = self.guidance_scale
            shape = self.plan.timestep.shape
            self.plan.timestep.assign(np.full(shape, value, dtype=np.float32))
            self.plan.timestep_r.assign(np.full(shape, value, dtype=np.float32))
            self.plan.next_timestep.assign(np.full(shape, next_value, dtype=np.float32))
            self.active.assign(np.array([int(active)], dtype=np.int32))
            self.guidance.assign(np.array([scale], dtype=np.float32))
            wp.capture_launch(self.graph)
        return self.hidden


def load_ace_dit_weights(path, config, device, dtype=wp.bfloat16):
    """Load exactly the official ACE DiT tensors from sharded safetensors."""
    archive = SafeTensorArchive(path)
    return load_cast_weights(archive, dit_weight_names(config), device, dtype)


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
        return flow_schedule(steps, shift=shift)
    nearest = min(TURBO_TIMESTEPS, key=lambda value: abs(value - shift))
    return TURBO_TIMESTEPS[nearest]


def flow_schedule(steps: int, *, shift: float = 1.0) -> tuple[float, ...]:
    """Build the official descending flow-matching schedule."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if shift <= 0.0:
        raise ValueError("shift must be positive")
    raw = tuple(1.0 - index / int(steps) for index in range(int(steps)))
    if shift == 1.0:
        return raw
    return tuple(shift * value / (1.0 + (shift - 1.0) * value) for value in raw)


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
