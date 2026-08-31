# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Exact fixed-buffer Qwen transformer training composition."""

from collections.abc import Mapping

import warp as wp

from .adapters import LoRAAdapterCollection, LoRAAdapterConfig
from .bridges import add_fp32_gradients, cast_from_float32, cast_to_float32
from .gated_delta import GatedDeltaInputPlan
from .gated_delta_rule import GatedDeltaRulePlan
from .gated_norm import GatedRMSNormPlan
from .gqa import GQALoRAAttentionPlan
from .linear_attention import QwenGatedDeltaLoRAAttentionPlan
from .mlp import LoRASwiGLUPlan
from .model import CausalLMTrainingPlan, require_weights
from .output import CausalLMOutputPlan
from .primitives import residual_forward
from .qk import QKTransformPlan
from .stack import LoRATransformerStackPlan


class QwenLoRATransformerBlockPlan:
    """Compose one Qwen pre-norm attention/MLP layer.

    Full-attention plans use Qwen's packed ``[Q, sigmoid-gate]`` projection;
    Gated Delta plans implement the same fixed-buffer forward/backward interface.
    Norm parameters stay frozen and every Linear is trained through LoRA.
    """

    def __init__(
        self,
        attention,
        mlp: LoRASwiGLUPlan,
        *,
        input_norm_weight: wp.array,
        post_attention_norm_weight: wp.array,
        epsilon: float,
        centered_norm_scales: bool,
    ):
        if attention.adapters is not mlp.adapters:
            raise ValueError("Qwen attention and MLP must share one adapter collection")
        if hasattr(attention, "packed_query_gate") and (
            not attention.packed_query_gate or attention.gate_name is not None
        ):
            raise ValueError("Qwen attention requires one packed Q/gate projection")
        if (
            attention.rows != mlp.rows
            or attention.hidden != mlp.hidden
            or attention.dtype != mlp.dtype
            or attention.device != mlp.device
        ):
            raise ValueError("Qwen attention and MLP geometry must match")
        rows, hidden = attention.rows, attention.hidden
        weights = (input_norm_weight, post_attention_norm_weight)
        for weight in weights:
            if (
                not isinstance(weight, wp.array)
                or weight.shape != (hidden,)
                or weight.dtype != attention.dtype
                or weight.device != attention.device
                or not weight.is_contiguous
            ):
                raise ValueError("Qwen norm weights must match hidden dtype and device")

        self.attention = attention
        self.mlp = mlp
        self.adapters = attention.adapters
        self.device = attention.device
        self.dtype = attention.dtype
        self.rows = rows
        self.hidden = hidden
        self.weights = weights
        norm_options = dict(
            batch=1,
            heads=1,
            sequence=rows,
            head_size=hidden,
            dtype=self.dtype,
            rotary_dim=0,
            epsilon=epsilon,
            weight_offset=1.0 if centered_norm_scales else 0.0,
            device=self.device,
        )
        self.input_norm = QKTransformPlan(**norm_options)
        self.post_attention_norm = QKTransformPlan(**norm_options)

        self.shape = (rows, hidden)
        self.shape4 = (1, 1, rows, hidden)
        self.attention_residual = wp.empty(
            self.shape, dtype=self.dtype, device=self.device
        )
        self.output = wp.empty(self.shape, dtype=self.dtype, device=self.device)
        self.grad_output_fp32 = wp.empty(
            self.shape, dtype=wp.float32, device=self.device
        )
        self.mlp_output_grad = wp.empty(
            self.shape, dtype=self.dtype, device=self.device
        )
        self.attention_output_grad = wp.empty(
            self.shape, dtype=self.dtype, device=self.device
        )
        self.residual_grad = wp.empty(self.shape, dtype=wp.float32, device=self.device)
        self.input_grad = wp.empty(self.shape, dtype=wp.float32, device=self.device)

    def forward(
        self, x: wp.array, lengths: wp.array, positions=None, cosine=None, sine=None
    ) -> wp.array:
        """Execute the exact Qwen attention layer."""
        normalized = self.input_norm.forward(
            x.reshape(self.shape4), self.weights[0]
        ).reshape(self.shape)
        attention_output = self.attention.forward(
            normalized, lengths, positions, cosine, sine
        )
        residual_forward(x, attention_output, self.attention_residual)
        mlp_input = self.post_attention_norm.forward(
            self.attention_residual.reshape(self.shape4), self.weights[1]
        ).reshape(self.shape)
        residual_forward(
            self.attention_residual, self.mlp.forward(mlp_input), self.output
        )
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
        """Reverse the Qwen layer and return its fixed FP32 input gradient."""
        cast_to_float32(grad_output, self.grad_output_fp32)
        mlp_input_grad = self.mlp.backward(
            self.post_attention_norm.output.reshape(self.shape),
            grad_output,
            accumulate=accumulate,
        )
        residual_from_mlp = self.post_attention_norm.backward(
            self.attention_residual.reshape(self.shape4),
            self.weights[1],
            mlp_input_grad.reshape(self.shape4),
        )
        add_fp32_gradients(
            self.grad_output_fp32,
            residual_from_mlp.reshape(self.shape),
            self.residual_grad,
        )
        cast_from_float32(self.residual_grad, self.attention_output_grad)
        normalized_grad = self.attention.backward(
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
            normalized_grad.reshape(self.shape4),
        )
        add_fp32_gradients(
            self.residual_grad,
            attention_input_grad.reshape(self.shape),
            self.input_grad,
        )
        return self.input_grad


def build_qwen_lora_training_plan(
    config: Mapping[str, object],
    weights: Mapping[str, object],
    *,
    batch: int,
    sequence: int,
    adapter_config: LoRAAdapterConfig,
    centered_norm_scales: bool,
    gguf_layout: bool = False,
    ssm_a_is_decay: bool = False,
    seed: int = 0,
    optimizer_options: Mapping[str, object] | None = None,
    use_cublas: bool = True,
) -> CausalLMTrainingPlan:
    """Build an exact Qwen 3.5 LoRA model from canonical runtime weights."""
    if batch <= 0 or sequence <= 0:
        raise ValueError("batch and sequence must be positive")
    required_config = (
        "hidden_size",
        "num_hidden_layers",
        "layer_types",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "rms_norm_eps",
        "rope_parameters",
    )
    missing_config = [name for name in required_config if name not in config]
    if missing_config:
        raise ValueError(f"Qwen training config is missing {missing_config}")
    layers = int(config["num_hidden_layers"])
    layer_types = tuple(config["layer_types"])
    if layers <= 0 or len(layer_types) != layers:
        raise ValueError("Qwen layer_types must match num_hidden_layers")
    if set(layer_types) - {"full_attention", "linear_attention"}:
        raise ValueError("unsupported Qwen attention layer type")
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_size = int(config["head_dim"])
    linear_key_heads = int(config["linear_num_key_heads"])
    linear_value_heads = int(config["linear_num_value_heads"])
    linear_key_size = int(config["linear_key_head_dim"])
    linear_value_size = int(config["linear_value_head_dim"])
    kernel_size = int(config["linear_conv_kernel_dim"])
    rotary_dim = int(
        head_size * float(config["rope_parameters"].get("partial_rotary_factor", 1.0))
    )
    rows = batch * sequence
    root_names = (
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    )
    common_suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    attention_suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
    )
    linear_suffixes = (
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.conv1d.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
    )
    layer_names = []
    projection_names = []
    for index, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{index}."
        layer_names.extend(prefix + suffix for suffix in common_suffixes)
        projection_names.extend(prefix + suffix for suffix in common_suffixes[2:])
        if layer_type == "full_attention":
            layer_names.extend(prefix + suffix for suffix in attention_suffixes)
            projection_names.extend(
                prefix + suffix
                for suffix in (
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                )
            )
        else:
            layer_names.extend(prefix + suffix for suffix in linear_suffixes)
            projection_names.extend(
                prefix + suffix
                for suffix in (
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.in_proj_z.weight",
                    "linear_attn.in_proj_a.weight",
                    "linear_attn.in_proj_b.weight",
                    "linear_attn.out_proj.weight",
                )
            )
    loaded = require_weights(weights, root_names + tuple(layer_names))
    adapters = LoRAAdapterCollection(
        {name: loaded[name] for name in projection_names},
        rows,
        adapter_config,
        seed=seed,
        optimizer_options=optimizer_options,
        use_cublas=use_cublas,
    )
    dtype = adapters.targets[next(iter(adapters.targets))].weight.dtype
    device = adapters.device
    blocks = []
    for index, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{index}."
        if layer_type == "full_attention":
            attention = GQALoRAAttentionPlan(
                adapters,
                query=prefix + "self_attn.q_proj.weight",
                key=prefix + "self_attn.k_proj.weight",
                value=prefix + "self_attn.v_proj.weight",
                output=prefix + "self_attn.o_proj.weight",
                packed_query_gate=True,
                batch=batch,
                sequence=sequence,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_size=head_size,
                query_transform=QKTransformPlan(
                    batch,
                    query_heads,
                    sequence,
                    head_size,
                    dtype,
                    rotary_dim=rotary_dim,
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
                query_norm_weight=loaded[prefix + "self_attn.q_norm.weight"],
                key_norm_weight=loaded[prefix + "self_attn.k_norm.weight"],
                interleaved_rope_weights=gguf_layout,
            )
        else:
            inputs = GatedDeltaInputPlan(
                batch,
                sequence,
                linear_key_heads,
                linear_value_heads,
                linear_key_size,
                linear_value_size,
                kernel_size,
                dtype,
                epsilon=float(config["rms_norm_eps"]),
                a_is_decay=ssm_a_is_decay,
                device=device,
            )
            rule = GatedDeltaRulePlan(
                batch,
                sequence,
                linear_key_heads,
                linear_value_heads,
                linear_key_size,
                linear_value_size,
                dtype,
                device=device,
            )
            conv_width = inputs.conv_width
            conv_weight = loaded[prefix + "linear_attn.conv1d.weight"]
            if conv_weight.shape == (conv_width, 1, kernel_size):
                conv_weight = conv_weight.reshape((conv_width, kernel_size))
            attention = QwenGatedDeltaLoRAAttentionPlan(
                adapters,
                qkv=prefix + "linear_attn.in_proj_qkv.weight",
                gate=prefix + "linear_attn.in_proj_z.weight",
                decay=prefix + "linear_attn.in_proj_a.weight",
                beta=prefix + "linear_attn.in_proj_b.weight",
                output=prefix + "linear_attn.out_proj.weight",
                inputs=inputs,
                rule=rule,
                gated_norm=GatedRMSNormPlan(
                    batch * linear_value_heads * sequence,
                    linear_value_size,
                    dtype,
                    epsilon=float(config["rms_norm_eps"]),
                    device=device,
                ),
                conv_weight=conv_weight,
                conv_state=wp.zeros(
                    (batch, conv_width, kernel_size - 1),
                    dtype=dtype,
                    device=device,
                ),
                a_log=loaded[prefix + "linear_attn.A_log"],
                dt_bias=loaded[prefix + "linear_attn.dt_bias"],
                recurrent_state=wp.zeros(
                    (
                        batch,
                        linear_value_heads,
                        linear_key_size,
                        linear_value_size,
                    ),
                    dtype=wp.float32,
                    device=device,
                ),
                norm_weight=loaded[prefix + "linear_attn.norm.weight"],
            )
        mlp = LoRASwiGLUPlan(
            adapters,
            gate=prefix + "mlp.gate_proj.weight",
            up=prefix + "mlp.up_proj.weight",
            down=prefix + "mlp.down_proj.weight",
        )
        blocks.append(
            QwenLoRATransformerBlockPlan(
                attention,
                mlp,
                input_norm_weight=loaded[prefix + "input_layernorm.weight"],
                post_attention_norm_weight=loaded[
                    prefix + "post_attention_layernorm.weight"
                ],
                epsilon=float(config["rms_norm_eps"]),
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
        cublas=adapters.cublas,
    )
    return CausalLMTrainingPlan(
        loaded["model.language_model.embed_tokens.weight"], stack, output
    )
