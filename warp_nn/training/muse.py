# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Exact fixed-buffer Muse Glimmer transformer-layer training composition."""

from collections.abc import Mapping

import warp as wp

from .adapters import LoRAAdapterCollection, LoRAAdapterConfig
from .bridges import add_fp32_gradients, cast_from_float32, cast_to_float32
from .gqa import GQALoRAAttentionPlan
from .mlp import LoRASwiGLUPlan
from .model import CausalLMTrainingPlan, require_weights
from .output import CausalLMOutputPlan
from .primitives import residual_forward
from .qk import QKTransformPlan
from .stack import LoRATransformerStackPlan


class MuseLoRATransformerBlockPlan:
    """Compose one exact Muse sandwich-norm attention/MLP layer.

    Norm weights stay frozen, while every Linear in the supplied attention and
    MLP plans is trained through LoRA. All saved state and gradient bridges are
    fixed at construction and the complete forward/backward is graph-capturable.
    """

    def __init__(
        self,
        attention: GQALoRAAttentionPlan,
        mlp: LoRASwiGLUPlan,
        *,
        input_norm_weight: wp.array,
        post_attention_norm_weight: wp.array,
        feedforward_norm_weight: wp.array,
        post_feedforward_norm_weight: wp.array,
        rms_epsilon: float,
        post_epsilon: float,
        centered_norm_scales: bool,
    ):
        if attention.adapters is not mlp.adapters:
            raise ValueError("Muse attention and MLP must share one adapter collection")
        if (
            attention.rows != mlp.rows
            or attention.hidden != mlp.hidden
            or attention.dtype != mlp.dtype
            or attention.device != mlp.device
        ):
            raise ValueError("Muse attention and MLP geometry must match")
        rows, hidden = attention.rows, attention.hidden
        weights = (
            input_norm_weight,
            post_attention_norm_weight,
            feedforward_norm_weight,
            post_feedforward_norm_weight,
        )
        for weight in weights:
            if (
                not isinstance(weight, wp.array)
                or weight.shape != (hidden,)
                or weight.dtype != attention.dtype
                or weight.device != attention.device
                or not weight.is_contiguous
            ):
                raise ValueError("Muse norm weights must match hidden dtype and device")

        self.attention = attention
        self.mlp = mlp
        self.adapters = attention.adapters
        self.device = attention.device
        self.dtype = attention.dtype
        self.rows = rows
        self.hidden = hidden
        self.weights = weights
        offset = 1.0 if centered_norm_scales else 0.0
        norm_options = dict(
            batch=1,
            heads=1,
            sequence=rows,
            head_size=hidden,
            dtype=self.dtype,
            rotary_dim=0,
            weight_offset=offset,
            device=self.device,
        )
        self.input_norm = QKTransformPlan(epsilon=rms_epsilon, **norm_options)
        self.post_attention_norm = QKTransformPlan(epsilon=post_epsilon, **norm_options)
        self.feedforward_norm = QKTransformPlan(epsilon=rms_epsilon, **norm_options)
        self.post_feedforward_norm = QKTransformPlan(
            epsilon=post_epsilon, **norm_options
        )

        shape = (rows, hidden)
        shape4 = (1, 1, rows, hidden)
        self.shape = shape
        self.shape4 = shape4
        self.attention_residual = wp.empty(shape, dtype=self.dtype, device=self.device)
        self.output = wp.empty(shape, dtype=self.dtype, device=self.device)
        self.grad_output_fp32 = wp.empty(shape, dtype=wp.float32, device=self.device)
        self.mlp_output_grad = wp.empty(shape, dtype=self.dtype, device=self.device)
        self.attention_output_grad = wp.empty(
            shape, dtype=self.dtype, device=self.device
        )
        self.residual_grad = wp.empty(shape, dtype=wp.float32, device=self.device)
        self.input_grad = wp.empty(shape, dtype=wp.float32, device=self.device)

    def forward(
        self, x: wp.array, lengths: wp.array, positions=None, cosine=None, sine=None
    ) -> wp.array:
        """Execute the exact Muse layer into the fixed output buffer."""
        input_norm = self.input_norm.forward(
            x.reshape(self.shape4), self.weights[0]
        ).reshape(self.shape)
        attention_output = self.attention.forward(
            input_norm, lengths, positions, cosine, sine
        )
        post_attention = self.post_attention_norm.forward(
            attention_output.reshape(self.shape4), self.weights[1]
        ).reshape(self.shape)
        residual_forward(x, post_attention, self.attention_residual)
        feedforward_input = self.feedforward_norm.forward(
            self.attention_residual.reshape(self.shape4), self.weights[2]
        ).reshape(self.shape)
        mlp_output = self.mlp.forward(feedforward_input)
        post_feedforward = self.post_feedforward_norm.forward(
            mlp_output.reshape(self.shape4), self.weights[3]
        ).reshape(self.shape)
        residual_forward(self.attention_residual, post_feedforward, self.output)
        return self.output

    def backward(
        self,
        x: wp.array,
        lengths: wp.array,
        grad_output: wp.array,
        positions=None,
        cosine=None,
        sine=None,
        *,
        accumulate: bool = False,
    ) -> wp.array:
        """Reverse the exact Muse layer and return fixed FP32 input gradients."""
        cast_to_float32(grad_output, self.grad_output_fp32)
        mlp_output_grad = self.post_feedforward_norm.backward(
            self.mlp.output.reshape(self.shape4),
            self.weights[3],
            self.grad_output_fp32.reshape(self.shape4),
        )
        cast_from_float32(mlp_output_grad.reshape(self.shape), self.mlp_output_grad)
        feedforward_grad = self.mlp.backward(
            self.feedforward_norm.output.reshape(self.shape),
            self.mlp_output_grad,
            accumulate=accumulate,
        )
        feedforward_residual_grad = self.feedforward_norm.backward(
            self.attention_residual.reshape(self.shape4),
            self.weights[2],
            feedforward_grad.reshape(self.shape4),
        )
        add_fp32_gradients(
            self.grad_output_fp32,
            feedforward_residual_grad.reshape(self.shape),
            self.residual_grad,
        )
        attention_output_grad = self.post_attention_norm.backward(
            self.attention.output.reshape(self.shape4),
            self.weights[1],
            self.residual_grad.reshape(self.shape4),
        )
        cast_from_float32(
            attention_output_grad.reshape(self.shape), self.attention_output_grad
        )
        normalized_input_grad = self.attention.backward(
            self.input_norm.output.reshape(self.shape),
            lengths,
            self.attention_output_grad,
            positions,
            cosine,
            sine,
            accumulate=accumulate,
        )
        attention_input_grad = self.input_norm.backward(
            x.reshape(self.shape4),
            self.weights[0],
            normalized_input_grad.reshape(self.shape4),
        )
        add_fp32_gradients(
            self.residual_grad,
            attention_input_grad.reshape(self.shape),
            self.input_grad,
        )
        return self.input_grad


def build_muse_lora_training_plan(
    config: Mapping[str, object],
    weights: Mapping[str, object],
    *,
    batch: int,
    sequence: int,
    adapter_config: LoRAAdapterConfig,
    centered_norm_scales: bool,
    seed: int = 0,
    optimizer_options: Mapping[str, object] | None = None,
    use_cublas: bool = True,
) -> CausalLMTrainingPlan:
    """Build an exact fixed-buffer Muse LoRA model from canonical weights."""
    if batch <= 0 or sequence <= 0:
        raise ValueError("batch and sequence must be positive")
    required_config = (
        "hidden_size",
        "num_hidden_layers",
        "layer_types",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "sliding_window",
        "qk_scale_factor",
        "rms_norm_eps",
        "post_norm_eps",
        "output_multiplier",
        "final_logit_softcapping",
    )
    missing_config = [name for name in required_config if name not in config]
    if missing_config:
        raise ValueError(f"Muse training config is missing {missing_config}")
    layers = int(config["num_hidden_layers"])
    layer_types = tuple(config["layer_types"])
    if layers <= 0 or len(layer_types) != layers:
        raise ValueError("Muse layer_types must match num_hidden_layers")
    if set(layer_types) - {"sliding_attention", "full_attention"}:
        raise ValueError("unsupported Muse attention layer type")
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_size = int(config["head_dim"])
    rows = batch * sequence
    root_names = (
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    )
    suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.gate_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    layer_names = tuple(
        f"model.language_model.layers.{index}.{suffix}"
        for index in range(layers)
        for suffix in suffixes
    )
    loaded = require_weights(weights, root_names + layer_names)
    projection_suffixes = suffixes[4:]
    projection_weights = {
        f"model.language_model.layers.{index}.{suffix}": loaded[
            f"model.language_model.layers.{index}.{suffix}"
        ]
        for index in range(layers)
        for suffix in projection_suffixes
    }
    adapters = LoRAAdapterCollection(
        projection_weights,
        rows,
        adapter_config,
        seed=seed,
        optimizer_options=optimizer_options,
        use_cublas=use_cublas,
    )
    dtype = adapters.targets[next(iter(adapters.targets))].weight.dtype
    device = adapters.device
    unit_head = wp.ones(head_size, dtype=dtype, device=device)
    blocks = []
    for index, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{index}."
        rotary_dim = head_size if layer_type == "sliding_attention" else 0
        attention = GQALoRAAttentionPlan(
            adapters,
            query=prefix + "self_attn.q_proj.weight",
            key=prefix + "self_attn.k_proj.weight",
            value=prefix + "self_attn.v_proj.weight",
            gate=prefix + "self_attn.gate_proj.weight",
            output=prefix + "self_attn.o_proj.weight",
            batch=batch,
            sequence=sequence,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_size=head_size,
            window=(
                int(config["sliding_window"])
                if layer_type == "sliding_attention"
                else 0
            ),
            query_transform=QKTransformPlan(
                batch,
                query_heads,
                sequence,
                head_size,
                dtype,
                rotary_dim=rotary_dim,
                scale=float(config["qk_scale_factor"]),
                device=device,
            ),
            key_transform=QKTransformPlan(
                batch,
                kv_heads,
                sequence,
                head_size,
                dtype,
                rotary_dim=rotary_dim,
                device=device,
            ),
            query_norm_weight=unit_head,
            key_norm_weight=unit_head,
        )
        mlp = LoRASwiGLUPlan(
            adapters,
            gate=prefix + "mlp.gate_proj.weight",
            up=prefix + "mlp.up_proj.weight",
            down=prefix + "mlp.down_proj.weight",
        )
        blocks.append(
            MuseLoRATransformerBlockPlan(
                attention,
                mlp,
                input_norm_weight=loaded[prefix + "input_layernorm.weight"],
                post_attention_norm_weight=loaded[
                    prefix + "post_attention_layernorm.weight"
                ],
                feedforward_norm_weight=loaded[
                    prefix + "pre_feedforward_layernorm.weight"
                ],
                post_feedforward_norm_weight=loaded[
                    prefix + "post_feedforward_layernorm.weight"
                ],
                rms_epsilon=float(config["rms_norm_eps"]),
                post_epsilon=float(config["post_norm_eps"]),
                centered_norm_scales=centered_norm_scales,
            )
        )
    stack = LoRATransformerStackPlan(blocks)
    output = CausalLMOutputPlan(
        rows,
        loaded["model.language_model.norm.weight"],
        loaded["lm_head.weight"],
        epsilon=float(config["rms_norm_eps"]),
        norm_weight_offset=1.0 if centered_norm_scales else 0.0,
        logit_multiplier=float(config["output_multiplier"]),
        softcap=float(config["final_logit_softcapping"]),
        cublas=adapters.cublas,
    )
    return CausalLMTrainingPlan(
        loaded["model.language_model.embed_tokens.weight"], stack, output
    )
