# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Official ACE-Step 1.5 caption/lyric condition encoder."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import warp as wp

from .._cublas import try_create_cublas
from ..formats.safetensors import SafeTensorArchive
from ..operators import (
    ModulatedResidualPlan,
    Operation,
    execute_operations,
    plan_linear,
    plan_rms_norm,
    plan_swiglu,
    rotary_cache_values,
)
from ..weights import load_cast_weights
from ...utils.device import parse_device
from .dit import AceStepAttentionPlan, AceStepDiTConfig


def condition_weight_names(config: AceStepDiTConfig) -> tuple[str, ...]:
    """Return the official caption, lyric, null, and timbre tensor contract."""
    names = [
        "encoder.text_projector.weight",
        "encoder.lyric_encoder.embed_tokens.weight",
        "encoder.lyric_encoder.embed_tokens.bias",
        "encoder.lyric_encoder.norm.weight",
        "encoder.timbre_encoder.embed_tokens.weight",
        "encoder.timbre_encoder.embed_tokens.bias",
        "encoder.timbre_encoder.norm.weight",
        "encoder.timbre_encoder.special_token",
        "null_condition_emb",
    ]
    for base, count in (
        ("encoder.lyric_encoder", config.num_lyric_encoder_hidden_layers),
        ("encoder.timbre_encoder", config.num_timbre_encoder_hidden_layers),
    ):
        for index in range(count):
            prefix = f"{base}.layers.{index}."
            names.extend(
                prefix + suffix
                for suffix in (
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                )
            )
    return tuple(names)


@lru_cache(maxsize=None)
def _condition_kernels(dtype):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def add_bias(
        values: wp.array3d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
    ):
        batch, token, channel = wp.tid()
        output[batch, token, channel] = DTYPE(
            wp.float32(values[batch, token, channel]) + wp.float32(bias[channel])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def pack(
        lyric: wp.array3d(dtype=DTYPE),
        timbre: wp.array3d(dtype=DTYPE),
        text: wp.array3d(dtype=DTYPE),
        order: wp.array2d(dtype=wp.int32),
        output: wp.array3d(dtype=DTYPE),
        lyric_length: int,
        timbre_length: int,
    ):
        batch, token, channel = wp.tid()
        source = order[batch, token]
        if source < lyric_length:
            output[batch, token, channel] = lyric[batch, source, channel]
        elif source < lyric_length + timbre_length:
            output[batch, token, channel] = timbre[
                batch, source - lyric_length, channel
            ]
        else:
            output[batch, token, channel] = text[
                batch, source - lyric_length - timbre_length, channel
            ]

    @wp.kernel(enable_backward=False, module="unique")
    def fill_null(
        null_embedding: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
    ):
        batch, token, channel = wp.tid()
        output[batch, token, channel] = null_embedding[0, 0, channel]

    @wp.kernel(enable_backward=False, module="unique")
    def copy_reference(
        source: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        source_batch: int,
    ):
        batch, frame, channel = wp.tid()
        output[batch, frame, channel] = source[batch % source_batch, frame, channel]

    return add_bias, pack, fill_null, copy_reference


class _EncoderLayerPlan:
    def __init__(
        self,
        hidden,
        valid,
        weights,
        config,
        base,
        index,
        position_ids,
        cos_cache,
        sin_cache,
        cublas,
    ):
        self.hidden = hidden
        self.device = hidden.device
        self.weights = weights
        prefix = f"{base}.layers.{index}"
        self._norm_tensors = dict(weights)
        self._norm_tensors["hidden"] = hidden
        self._norm_shapes = {
            name: tuple(value.shape) for name, value in self._norm_tensors.items()
        }
        self.input_norm = Operation(
            "SimplifiedLayerNormalization",
            ["hidden", prefix + ".input_layernorm.weight"],
            ["input_norm"],
            {"epsilon": config.rms_norm_eps},
        )
        plan_rms_norm(
            self.input_norm, self._norm_tensors, self._norm_shapes, self.device
        )
        normalized = self._norm_tensors["input_norm"]
        self.attention = AceStepAttentionPlan(
            normalized,
            weights,
            prefix + ".self_attn",
            config,
            query_valid=valid,
            key_valid=valid,
            position_ids=position_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            layer_index=index,
            cublas=cublas,
        )
        self.attention_residual = ModulatedResidualPlan(hidden, self.attention.output)
        self._mlp_tensors = dict(weights)
        self._mlp_tensors["hidden"] = self.attention_residual.output
        self._mlp_shapes = {
            name: tuple(value.shape) for name, value in self._mlp_tensors.items()
        }
        self.post_norm = Operation(
            "SimplifiedLayerNormalization",
            ["hidden", prefix + ".post_attention_layernorm.weight"],
            ["post_norm"],
            {"epsilon": config.rms_norm_eps},
        )
        plan_rms_norm(self.post_norm, self._mlp_tensors, self._mlp_shapes, self.device)
        post_norm = self._mlp_tensors["post_norm"]
        self._mlp_tensors["post_norm"] = post_norm.reshape((-1, post_norm.shape[-1]))
        self._mlp_shapes["post_norm"] = tuple(self._mlp_tensors["post_norm"].shape)

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

        self.gate = linear("gate", "post_norm", prefix + ".mlp.gate_proj.weight")
        self.up = linear("up", "post_norm", prefix + ".mlp.up_proj.weight")
        self.swiglu = Operation("_SwiGLU", ["gate", "up"], ["activated"])
        plan_swiglu(self.swiglu, self._mlp_tensors, self._mlp_shapes, self.device)
        self.down = linear("down", "activated", prefix + ".mlp.down_proj.weight")
        self.mlp_residual = ModulatedResidualPlan(
            self.attention_residual.output,
            self._mlp_tensors["down"].reshape(hidden.shape),
        )
        self.output = self.mlp_residual.output

    def execute(self):
        execute_operations(
            (self.input_norm,), self._norm_tensors, self._norm_shapes, self.device
        )
        self.attention.execute()
        self.attention_residual.execute()
        execute_operations(
            (self.post_norm, self.gate, self.up, self.swiglu, self.down),
            self._mlp_tensors,
            self._mlp_shapes,
            self.device,
        )
        self.mlp_residual.execute()
        return self.output


class _EncoderStackPlan:
    """Shared official ACE bidirectional transformer encoder stack."""

    def __init__(self, hidden, valid, weights, config, base, layers, cublas):
        length = hidden.shape[1]
        self.device = hidden.device
        self.layers = []
        positions = wp.array(
            np.broadcast_to(np.arange(length, dtype=np.int64), hidden.shape[:2]).copy(),
            device=self.device,
        )
        cos, sin = rotary_cache_values(
            length, config.head_dim, {"rope_theta": config.rope_theta}
        )
        cos_cache = wp.array(cos, dtype=hidden.dtype, device=self.device)
        sin_cache = wp.array(sin, dtype=hidden.dtype, device=self.device)
        current = hidden
        for index in range(layers):
            layer = _EncoderLayerPlan(
                current,
                valid,
                weights,
                config,
                base,
                index,
                positions,
                cos_cache,
                sin_cache,
                cublas,
            )
            self.layers.append(layer)
            current = layer.output
        self._tensors = {"hidden": current, "weight": weights[base + ".norm.weight"]}
        self._shapes = {
            name: tuple(value.shape) for name, value in self._tensors.items()
        }
        self.norm = Operation(
            "SimplifiedLayerNormalization",
            ["hidden", "weight"],
            ["output"],
            {"epsilon": config.rms_norm_eps},
        )
        plan_rms_norm(self.norm, self._tensors, self._shapes, self.device)
        self.output = self._tensors["output"]

    def execute(self):
        for layer in self.layers:
            layer.execute()
        execute_operations((self.norm,), self._tensors, self._shapes, self.device)
        return self.output


class AceStepConditionPlan:
    """Fixed-shape ACE 1.5 turbo condition plan.

    The current timbre boundary accepts one reference per sample, or one silence
    reference broadcast over the batch; packed multi-reference input is not yet
    supported. Masks are read once while the fixed packing plan is constructed.
    """

    def __init__(
        self,
        text_hidden,
        text_valid,
        lyric_embeddings,
        lyric_valid,
        reference_latents,
        weights,
        config,
        *,
        cublas=None,
    ):
        if text_hidden.ndim != 3 or lyric_embeddings.ndim != 3:
            raise ValueError("ACE text and lyric inputs must be rank three")
        if reference_latents is None or reference_latents.ndim != 3:
            raise ValueError("ACE reference/silence latents must be rank three")
        batch = text_hidden.shape[0]
        if (
            lyric_embeddings.shape[0] != batch
            or text_valid.shape != text_hidden.shape[:2]
            or lyric_valid.shape != lyric_embeddings.shape[:2]
            or reference_latents.shape[0] not in (1, batch)
        ):
            raise ValueError("ACE condition batch and mask shapes must match")
        if (
            text_hidden.shape[2] != config.text_hidden_dim
            or lyric_embeddings.shape[2] != config.text_hidden_dim
        ):
            raise ValueError(
                f"ACE Qwen text and lyric inputs must have width {config.text_hidden_dim}"
            )
        if (
            reference_latents.shape[1] < config.timbre_fix_frame
            or reference_latents.shape[2] != config.timbre_hidden_dim
        ):
            raise ValueError(
                f"ACE timbre input must provide at least {config.timbre_fix_frame} "
                f"frames of width {config.timbre_hidden_dim}"
            )
        if any(
            value.dtype != text_hidden.dtype or value.device != text_hidden.device
            for value in (lyric_embeddings, reference_latents)
        ):
            raise ValueError("ACE condition inputs must share dtype and device")
        if text_valid.dtype != wp.bool or lyric_valid.dtype != wp.bool:
            raise TypeError("ACE condition masks must be boolean")
        if (
            text_valid.device != text_hidden.device
            or lyric_valid.device != text_hidden.device
        ):
            raise ValueError("ACE condition masks must share the input device")
        self.device = text_hidden.device
        self.dtype = text_hidden.dtype
        self.config = config
        self.weights = weights
        self._kernels = _condition_kernels(self.dtype)

        self._text_tensors = dict(weights)
        self._text_tensors["input"] = text_hidden.reshape((-1, config.text_hidden_dim))
        self._text_shapes = {
            name: tuple(value.shape) for name, value in self._text_tensors.items()
        }
        self.text_projector = Operation(
            "Linear", ["input", "encoder.text_projector.weight"], ["output"]
        )
        plan_linear(
            self.text_projector,
            self._text_tensors,
            self._text_shapes,
            self.device,
            cublas=cublas,
        )
        self.text = self._text_tensors["output"].reshape(
            (batch, text_hidden.shape[1], config.encoder_hidden_size)
        )

        self._lyric_tensors = dict(weights)
        self._lyric_tensors["input"] = lyric_embeddings.reshape(
            (-1, config.text_hidden_dim)
        )
        self._lyric_shapes = {
            name: tuple(value.shape) for name, value in self._lyric_tensors.items()
        }
        self.lyric_projector = Operation(
            "Linear",
            ["input", "encoder.lyric_encoder.embed_tokens.weight"],
            ["projected"],
        )
        plan_linear(
            self.lyric_projector,
            self._lyric_tensors,
            self._lyric_shapes,
            self.device,
            cublas=cublas,
        )
        self.lyric = wp.empty(
            (batch, lyric_embeddings.shape[1], config.encoder_hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.lyric_stack = _EncoderStackPlan(
            self.lyric,
            lyric_valid,
            weights,
            config,
            "encoder.lyric_encoder",
            config.num_lyric_encoder_hidden_layers,
            cublas,
        )

        # Base/turbo ACE 1.5 does not prepend special_token. Encode a fixed
        # reference prefix and pool position zero; crop/broadcast stays on-device.
        self.reference_input = reference_latents
        self.reference = wp.empty(
            (batch, config.timbre_fix_frame, config.timbre_hidden_dim),
            dtype=self.dtype,
            device=self.device,
        )
        reference = self.reference
        self._timbre_tensors = dict(weights)
        self._timbre_tensors["input"] = reference.reshape(
            (-1, config.timbre_hidden_dim)
        )
        self._timbre_shapes = {
            name: tuple(value.shape) for name, value in self._timbre_tensors.items()
        }
        self.timbre_projector = Operation(
            "Linear",
            ["input", "encoder.timbre_encoder.embed_tokens.weight"],
            ["projected"],
        )
        plan_linear(
            self.timbre_projector,
            self._timbre_tensors,
            self._timbre_shapes,
            self.device,
            cublas=cublas,
        )
        self.timbre_hidden = wp.empty(
            (batch, config.timbre_fix_frame, config.encoder_hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        timbre_valid = wp.ones(
            (batch, config.timbre_fix_frame), dtype=wp.bool, device=self.device
        )
        self.timbre_stack = _EncoderStackPlan(
            self.timbre_hidden,
            timbre_valid,
            weights,
            config,
            "encoder.timbre_encoder",
            config.num_timbre_encoder_hidden_layers,
            cublas,
        )
        self.timbre = self.timbre_stack.output[:, :1, :]
        self.timbre_valid = wp.ones((batch, 1), dtype=wp.bool, device=self.device)

        masks = [
            lyric_valid.numpy().astype(bool),
            np.ones((batch, 1), dtype=bool),
            text_valid.numpy().astype(bool),
        ]
        combined_mask = np.concatenate(masks, axis=1)
        order = np.argsort(~combined_mask, axis=1, kind="stable").astype(np.int32)
        self.valid = wp.array(
            np.arange(combined_mask.shape[1])[None, :]
            < combined_mask.sum(axis=1)[:, None],
            dtype=wp.bool,
            device=self.device,
        )
        self.order = wp.array(order, dtype=wp.int32, device=self.device)
        self.output = wp.empty(
            (batch, combined_mask.shape[1], config.encoder_hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._null_output = wp.empty_like(self.output)

    def execute(self):
        execute_operations(
            (self.text_projector,), self._text_tensors, self._text_shapes, self.device
        )
        execute_operations(
            (self.lyric_projector,),
            self._lyric_tensors,
            self._lyric_shapes,
            self.device,
        )
        wp.launch(
            self._kernels[0],
            dim=self.lyric.shape,
            inputs=[
                self._lyric_tensors["projected"].reshape(self.lyric.shape),
                self.weights["encoder.lyric_encoder.embed_tokens.bias"],
                self.lyric,
            ],
            device=self.device,
        )
        self.lyric_stack.execute()
        wp.launch(
            self._kernels[3],
            dim=self.reference.shape,
            inputs=[
                self.reference_input,
                self.reference,
                self.reference_input.shape[0],
            ],
            device=self.device,
        )
        execute_operations(
            (self.timbre_projector,),
            self._timbre_tensors,
            self._timbre_shapes,
            self.device,
        )
        wp.launch(
            self._kernels[0],
            dim=self.timbre_hidden.shape,
            inputs=[
                self._timbre_tensors["projected"].reshape(self.timbre_hidden.shape),
                self.weights["encoder.timbre_encoder.embed_tokens.bias"],
                self.timbre_hidden,
            ],
            device=self.device,
        )
        self.timbre_stack.execute()
        wp.launch(
            self._kernels[1],
            dim=self.output.shape,
            inputs=[
                self.lyric_stack.output,
                self.timbre,
                self.text,
                self.order,
                self.output,
                self.lyric.shape[1],
                1,
            ],
            device=self.device,
        )
        return self.output, self.valid

    def null_condition(self):
        """Return the learned unconditional embedding expanded to this shape."""
        wp.launch(
            self._kernels[2],
            dim=self._null_output.shape,
            inputs=[self.weights["null_condition_emb"], self._null_output],
            device=self.device,
        )
        return self._null_output


class AceStepConditionEncoder:
    """Load exact ACE 1.5 turbo condition weights and build fixed-shape plans."""

    def __init__(
        self,
        path,
        config,
        *,
        dtype=wp.bfloat16,
        device=None,
        use_cublas=True,
    ):
        self.device = parse_device(device)
        self.dtype = dtype
        self.config = config
        if config.model_version != "turbo" or not config.is_turbo:
            raise ValueError(
                "ACE condition execution currently supports only ACE-Step 1.5 turbo"
            )
        if config.encoder_hidden_size != config.hidden_size:
            raise ValueError(
                "ACE turbo condition execution requires encoder and decoder hidden widths to match"
            )
        archive = SafeTensorArchive(Path(path))
        names = condition_weight_names(config)
        missing = set(names) - set(archive.names)
        if missing:
            raise ValueError(
                f"ACE condition checkpoint is missing {sorted(missing)[:5]}"
            )
        self.weights = load_cast_weights(archive, names, self.device, dtype)
        self.cublas = (
            try_create_cublas() if use_cublas and self.device.is_cuda else None
        )

    def plan(
        self,
        text_hidden,
        text_valid,
        lyric_embeddings,
        lyric_valid,
        reference_latents,
    ):
        return AceStepConditionPlan(
            text_hidden,
            text_valid,
            lyric_embeddings,
            lyric_valid,
            reference_latents,
            self.weights,
            self.config,
            cublas=self.cublas,
        )
