# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Fixed-shape Qwen-Image MMDiT assembly over shared runtime operators."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import warp as wp

from ..operators import (
    AdaptiveLayerNormPlan,
    AttentionHeadsPlan,
    AttentionMergePlan,
    BiasedLinearPlan,
    BroadcastGatedResidualPlan,
    ElementwiseActivationPlan,
    JointBidirectionalAttentionPlan,
    RMSNormPlan,
    RotaryCachePlan,
    SinusoidalEmbeddingPlan,
    multi_axis_rotary_cache_values,
)
from ..weights import load_cast_weights
from .checkpoint import QwenImageTransformerManifest
from .runner import QwenImageTransformerConfig


def qwen_image_rotary_coordinates(text_tokens, image_height, image_width):
    """Return official centered text/image temporal-height-width coordinates."""
    if min(text_tokens, image_height, image_width) <= 0:
        raise ValueError("Qwen-Image RoPE geometry must be positive")
    offset = max(image_height // 2, image_width // 2)
    text_position = np.arange(offset, offset + text_tokens, dtype=np.float32)
    text = np.stack((text_position, text_position, text_position), axis=1)
    rows = np.arange(
        -(image_height - image_height // 2), image_height // 2, dtype=np.float32
    )
    columns = np.arange(
        -(image_width - image_width // 2), image_width // 2, dtype=np.float32
    )
    image = np.empty((image_height * image_width, 3), dtype=np.float32)
    image[:, 0] = 0.0
    image[:, 1] = np.repeat(rows, image_width)
    image[:, 2] = np.tile(columns, image_height)
    return text, image


def qwen_image_mmdit_workspace_bytes(
    config, image_tokens, text_tokens, *, batch=1, element_bytes=2
):
    """Return fixed activation/mask storage, excluding inputs and model weights."""
    if min(image_tokens, text_tokens, batch, element_bytes) <= 0:
        raise ValueError("Qwen-Image workspace geometry must be positive")
    width = config.hidden_size
    joint = image_tokens + text_tokens
    output_width = config.patch_size**2 * config.output_channels
    persistent = (
        batch * image_tokens * width
        + batch * text_tokens * config.text_width
        + batch * text_tokens * width
        + batch * 256
        + 3 * batch * width
        + width
        + joint * config.head_dim
        + 2 * batch * width
        + 2 * batch * image_tokens * width
        + batch * image_tokens * output_width
    )
    layer_scratch = 27 * batch * joint * width + 12 * batch * width
    boolean_masks = batch * (3 * image_tokens + 2 * text_tokens)
    return element_bytes * (persistent + layer_scratch) + boolean_masks


def _validate_plan_weights(weights, config, dtype, device):
    expected = QwenImageTransformerManifest.from_config(config).shapes()
    missing = sorted(set(expected) - weights.keys())
    if missing:
        raise ValueError(f"Qwen-Image execution weights are missing '{missing[0]}'")
    for name, shape in expected.items():
        value = weights[name]
        if tuple(value.shape) != shape:
            raise ValueError(
                f"Qwen-Image execution weight '{name}' has shape {value.shape}, expected {shape}"
            )
        if value.dtype != dtype or value.device != device:
            raise ValueError(
                "Qwen-Image execution weights must match input dtype/device"
            )


def _array_nbytes(value):
    return int(np.prod(value.shape)) * wp.types.type_size_in_bytes(value.dtype)


class _LayerScratch:
    """Model-specific fixed buffers reused at identical points in every block."""

    def __init__(self):
        self.buffers = {}

    def bind(self, name, allocated):
        output = self.buffers.setdefault(name, allocated)
        if (
            output.shape != allocated.shape
            or output.dtype != allocated.dtype
            or output.device != allocated.device
        ):
            raise ValueError(f"Qwen-Image scratch buffer '{name}' is incompatible")
        return output

    @property
    def nbytes(self):
        return sum(_array_nbytes(value) for value in self.buffers.values())


def _bind_output(plan, scratch, name):
    plan.output = scratch.bind(name, plan.output)
    return plan.output


def _bind_linear_output(plan, scratch, name):
    output = scratch.bind(name, plan.output)
    plan._tensors["projected"] = output.reshape(plan._tensors["projected"].shape)
    plan.output = output
    return output


def _bind_rms_output(plan, scratch, name):
    plan.output = scratch.bind(name, plan.output)
    plan._tensors["normalized"] = plan.output
    return plan.output


def _bind_adaptive_output(plan, scratch, name):
    _bind_output(plan.norm, scratch, name + ".norm")
    return _bind_output(plan, scratch, name)


def _bind_joint_attention(plan, scratch):
    plan.joint_q = scratch.bind("joint.q", plan.joint_q)
    plan.joint_k = scratch.bind("joint.k", plan.joint_k)
    plan.joint_v = scratch.bind("joint.v", plan.joint_v)
    plan.second_valid = scratch.bind("joint.second_valid", plan.second_valid)
    plan.joint_valid = scratch.bind("joint.valid", plan.joint_valid)
    plan.attention.query = plan.joint_q
    plan.attention.key = plan.joint_k
    plan.attention.value = plan.joint_v
    plan.attention.key_valid = plan.joint_valid
    plan.attention.query_valid = scratch.bind(
        "joint.query_valid", plan.attention.query_valid
    )
    plan.attention.output = scratch.bind("joint.output", plan.attention.output)
    plan.first_output = scratch.bind("joint.first_output", plan.first_output)
    plan.second_output = scratch.bind("joint.second_output", plan.second_output)


class _QwenQKVPlan:
    def __init__(
        self,
        x,
        weights,
        prefix,
        heads,
        cosine,
        sine,
        *,
        added=False,
        scratch,
        scratch_prefix,
        cublas=None,
    ):
        stem = "add_{}_proj" if added else "to_{}"
        self.q = BiasedLinearPlan(
            x,
            weights[f"{prefix}.{stem.format('q')}.weight"],
            weights[f"{prefix}.{stem.format('q')}.bias"],
            cublas=cublas,
        )
        _bind_linear_output(self.q, scratch, scratch_prefix + ".q")
        self.k = BiasedLinearPlan(
            x,
            weights[f"{prefix}.{stem.format('k')}.weight"],
            weights[f"{prefix}.{stem.format('k')}.bias"],
            cublas=cublas,
        )
        _bind_linear_output(self.k, scratch, scratch_prefix + ".k")
        self.v = BiasedLinearPlan(
            x,
            weights[f"{prefix}.{stem.format('v')}.weight"],
            weights[f"{prefix}.{stem.format('v')}.bias"],
            cublas=cublas,
        )
        _bind_linear_output(self.v, scratch, scratch_prefix + ".v")
        self.q_heads = AttentionHeadsPlan(self.q.output, heads)
        self.k_heads = AttentionHeadsPlan(self.k.output, heads)
        self.v_heads = AttentionHeadsPlan(self.v.output, heads)
        _bind_output(self.q_heads, scratch, scratch_prefix + ".q_heads")
        _bind_output(self.k_heads, scratch, scratch_prefix + ".k_heads")
        _bind_output(self.v_heads, scratch, scratch_prefix + ".v_heads")
        norm_q = "norm_added_q" if added else "norm_q"
        norm_k = "norm_added_k" if added else "norm_k"
        self.q_norm = RMSNormPlan(
            self.q_heads.output, weights[f"{prefix}.{norm_q}.weight"], epsilon=1.0e-6
        )
        self.k_norm = RMSNormPlan(
            self.k_heads.output, weights[f"{prefix}.{norm_k}.weight"], epsilon=1.0e-6
        )
        _bind_rms_output(self.q_norm, scratch, scratch_prefix + ".q_norm")
        _bind_rms_output(self.k_norm, scratch, scratch_prefix + ".k_norm")
        self.q_rope = RotaryCachePlan(self.q_norm.output, cosine, sine)
        self.k_rope = RotaryCachePlan(self.k_norm.output, cosine, sine)
        _bind_output(self.q_rope, scratch, scratch_prefix + ".q_rope")
        _bind_output(self.k_rope, scratch, scratch_prefix + ".k_rope")

    @property
    def output(self):
        return self.q_rope.output, self.k_rope.output, self.v_heads.output

    def execute(self):
        self.q.execute()
        self.k.execute()
        self.v.execute()
        self.q_heads.execute()
        self.k_heads.execute()
        self.v_heads.execute()
        self.q_norm.execute()
        self.k_norm.execute()
        self.q_rope.execute()
        self.k_rope.execute()


class QwenImageMMDiTLayerPlan:
    """One graph-safe Qwen dual-stream joint-attention transformer block."""

    def __init__(
        self,
        image,
        text,
        timestep_embedding,
        text_valid,
        weights,
        config,
        layer,
        image_rope,
        text_rope,
        *,
        scratch,
        cublas=None,
    ):
        prefix = f"transformer_blocks.{layer}"
        self.image_modulation = BiasedLinearPlan(
            timestep_embedding,
            weights[f"{prefix}.img_mod.1.weight"],
            weights[f"{prefix}.img_mod.1.bias"],
            cublas=cublas,
        )
        _bind_linear_output(self.image_modulation, scratch, "image.modulation")
        self.text_modulation = BiasedLinearPlan(
            timestep_embedding,
            weights[f"{prefix}.txt_mod.1.weight"],
            weights[f"{prefix}.txt_mod.1.bias"],
            cublas=cublas,
        )
        _bind_linear_output(self.text_modulation, scratch, "text.modulation")
        batch, _, width = image.shape
        self.image_mod = self.image_modulation.output.reshape((batch, 6, width))
        self.text_mod = self.text_modulation.output.reshape((batch, 6, width))
        self.image_norm1 = AdaptiveLayerNormPlan(
            image, self.image_mod, shift_index=0, scale_index=1
        )
        self.text_norm1 = AdaptiveLayerNormPlan(
            text, self.text_mod, shift_index=0, scale_index=1
        )
        _bind_adaptive_output(self.image_norm1, scratch, "image.norm1")
        _bind_adaptive_output(self.text_norm1, scratch, "text.norm1")
        attention = f"{prefix}.attn"
        self.image_qkv = _QwenQKVPlan(
            self.image_norm1.output,
            weights,
            attention,
            config.heads,
            *image_rope,
            scratch=scratch,
            scratch_prefix="image.qkv",
            cublas=cublas,
        )
        self.text_qkv = _QwenQKVPlan(
            self.text_norm1.output,
            weights,
            attention,
            config.heads,
            *text_rope,
            scratch=scratch,
            scratch_prefix="text.qkv",
            added=True,
            cublas=cublas,
        )
        self.joint_attention = JointBidirectionalAttentionPlan(
            self.text_qkv.output,
            self.image_qkv.output,
            first_valid=text_valid,
        )
        _bind_joint_attention(self.joint_attention, scratch)
        self.image_merge = AttentionMergePlan(self.joint_attention.second_output)
        self.text_merge = AttentionMergePlan(self.joint_attention.first_output)
        _bind_output(self.image_merge, scratch, "image.merge")
        _bind_output(self.text_merge, scratch, "text.merge")
        self.image_attention_output = BiasedLinearPlan(
            self.image_merge.output,
            weights[f"{attention}.to_out.0.weight"],
            weights[f"{attention}.to_out.0.bias"],
            cublas=cublas,
        )
        _bind_linear_output(self.image_attention_output, scratch, "image.attention")
        self.text_attention_output = BiasedLinearPlan(
            self.text_merge.output,
            weights[f"{attention}.to_add_out.weight"],
            weights[f"{attention}.to_add_out.bias"],
            cublas=cublas,
        )
        _bind_linear_output(self.text_attention_output, scratch, "text.attention")
        self.image_attention_residual = BroadcastGatedResidualPlan(
            image,
            self.image_attention_output.output,
            self.image_mod,
            gate_index=2,
        )
        self.text_attention_residual = BroadcastGatedResidualPlan(
            text,
            self.text_attention_output.output,
            self.text_mod,
            gate_index=2,
        )
        _bind_output(self.image_attention_residual, scratch, "image.attention_residual")
        _bind_output(self.text_attention_residual, scratch, "text.attention_residual")
        self.image_norm2 = AdaptiveLayerNormPlan(
            self.image_attention_residual.output,
            self.image_mod,
            shift_index=3,
            scale_index=4,
        )
        self.text_norm2 = AdaptiveLayerNormPlan(
            self.text_attention_residual.output,
            self.text_mod,
            shift_index=3,
            scale_index=4,
        )
        _bind_adaptive_output(self.image_norm2, scratch, "image.norm2")
        _bind_adaptive_output(self.text_norm2, scratch, "text.norm2")
        self.image_mlp_up = BiasedLinearPlan(
            self.image_norm2.output,
            weights[f"{prefix}.img_mlp.net.0.proj.weight"],
            weights[f"{prefix}.img_mlp.net.0.proj.bias"],
            activation="gelu_tanh",
            cublas=cublas,
        )
        _bind_linear_output(self.image_mlp_up, scratch, "image.mlp_up")
        self.text_mlp_up = BiasedLinearPlan(
            self.text_norm2.output,
            weights[f"{prefix}.txt_mlp.net.0.proj.weight"],
            weights[f"{prefix}.txt_mlp.net.0.proj.bias"],
            activation="gelu_tanh",
            cublas=cublas,
        )
        _bind_linear_output(self.text_mlp_up, scratch, "text.mlp_up")
        self.image_mlp_down = BiasedLinearPlan(
            self.image_mlp_up.output,
            weights[f"{prefix}.img_mlp.net.2.weight"],
            weights[f"{prefix}.img_mlp.net.2.bias"],
            cublas=cublas,
        )
        _bind_linear_output(self.image_mlp_down, scratch, "image.mlp_down")
        self.text_mlp_down = BiasedLinearPlan(
            self.text_mlp_up.output,
            weights[f"{prefix}.txt_mlp.net.2.weight"],
            weights[f"{prefix}.txt_mlp.net.2.bias"],
            cublas=cublas,
        )
        _bind_linear_output(self.text_mlp_down, scratch, "text.mlp_down")
        self.image_residual = BroadcastGatedResidualPlan(
            self.image_attention_residual.output,
            self.image_mlp_down.output,
            self.image_mod,
            gate_index=5,
        )
        self.text_residual = BroadcastGatedResidualPlan(
            self.text_attention_residual.output,
            self.text_mlp_down.output,
            self.text_mod,
            gate_index=5,
        )
        self.image_residual.output = image
        self.text_residual.output = text
        self.image_output = self.image_residual.output
        self.text_output = self.text_residual.output

    def execute(self):
        self.image_modulation.execute()
        self.text_modulation.execute()
        self.image_norm1.execute()
        self.text_norm1.execute()
        self.image_qkv.execute()
        self.text_qkv.execute()
        self.joint_attention.execute()
        self.image_merge.execute()
        self.text_merge.execute()
        self.image_attention_output.execute()
        self.text_attention_output.execute()
        self.image_attention_residual.execute()
        self.text_attention_residual.execute()
        self.image_norm2.execute()
        self.text_norm2.execute()
        self.image_mlp_up.execute()
        self.text_mlp_up.execute()
        self.image_mlp_down.execute()
        self.text_mlp_down.execute()
        self.image_residual.execute()
        self.text_residual.execute()
        return self.image_output, self.text_output


class QwenImageMMDiTPlan:
    """Fixed-shape, graph-captured Qwen-Image transformer execution."""

    def __init__(
        self,
        image_tokens,
        text,
        text_valid,
        timestep,
        weights: Mapping[str, wp.array],
        config: QwenImageTransformerConfig,
        image_height,
        image_width,
        *,
        cublas=None,
    ):
        if image_tokens.ndim != 3 or text.ndim != 3:
            raise ValueError("Qwen-Image inputs must be rank-three token tensors")
        if image_tokens.shape[0] != text.shape[0] or (
            image_tokens.shape[2] != config.input_channels
            or text.shape[2] != config.text_width
        ):
            raise ValueError("Qwen-Image input geometry is incompatible")
        if image_tokens.shape[1] != image_height * image_width:
            raise ValueError("Qwen-Image token count does not match its image grid")
        if text_valid.shape != text.shape[:2] or text_valid.dtype != wp.bool:
            raise ValueError("Qwen-Image text validity mask is incompatible")
        if timestep.shape != (image_tokens.shape[0],) or timestep.dtype != wp.float32:
            raise ValueError("Qwen-Image timestep must be a batch FP32 vector")
        if any(
            value.device != image_tokens.device
            for value in (text, text_valid, timestep)
        ):
            raise ValueError("Qwen-Image inputs must share one device")
        if text.dtype != image_tokens.dtype or image_tokens.dtype not in (
            wp.float16,
            wp.bfloat16,
        ):
            raise TypeError("Qwen-Image execution requires matching FP16/BF16 inputs")
        if sum(config.rope_axes) != config.head_dim:
            raise ValueError("Qwen-Image RoPE axes must span one attention head")
        _validate_plan_weights(weights, config, image_tokens.dtype, image_tokens.device)

        self.image_tokens = image_tokens
        self.text = text
        self.text_valid = text_valid
        self.timestep = timestep
        self.weights = weights
        self.config = config
        self.device = image_tokens.device
        self.graph = None
        self.image_input = BiasedLinearPlan(
            image_tokens,
            weights["img_in.weight"],
            weights["img_in.bias"],
            cublas=cublas,
        )
        self.text_norm = RMSNormPlan(text, weights["txt_norm.weight"], epsilon=1.0e-6)
        self.text_input = BiasedLinearPlan(
            self.text_norm.output,
            weights["txt_in.weight"],
            weights["txt_in.bias"],
            cublas=cublas,
        )
        self.time_frequency = SinusoidalEmbeddingPlan(
            timestep,
            256,
            dtype=image_tokens.dtype,
            maximum_period=10000.0,
            scale=1000.0,
            frequency_shift=0.0,
            flip_sin_cos=True,
            quantize_input=True,
        )
        self.time_linear1 = BiasedLinearPlan(
            self.time_frequency.output,
            weights["time_text_embed.timestep_embedder.linear_1.weight"],
            weights["time_text_embed.timestep_embedder.linear_1.bias"],
            activation="silu",
            cublas=cublas,
        )
        self.time_linear2 = BiasedLinearPlan(
            self.time_linear1.output,
            weights["time_text_embed.timestep_embedder.linear_2.weight"],
            weights["time_text_embed.timestep_embedder.linear_2.bias"],
            cublas=cublas,
        )
        self.time_activation = ElementwiseActivationPlan(
            self.time_linear2.output, "silu"
        )
        text_coordinates, image_coordinates = qwen_image_rotary_coordinates(
            text.shape[1], int(image_height), int(image_width)
        )
        text_cos, text_sin = multi_axis_rotary_cache_values(
            text_coordinates, config.rope_axes
        )
        image_cos, image_sin = multi_axis_rotary_cache_values(
            image_coordinates, config.rope_axes
        )
        text_rope = (
            wp.array(text_cos, dtype=image_tokens.dtype, device=self.device),
            wp.array(text_sin, dtype=image_tokens.dtype, device=self.device),
        )
        image_rope = (
            wp.array(image_cos, dtype=image_tokens.dtype, device=self.device),
            wp.array(image_sin, dtype=image_tokens.dtype, device=self.device),
        )
        self.layers = []
        self.layer_scratch = _LayerScratch()
        image = self.image_input.output
        encoded_text = self.text_input.output
        for index in range(config.layers):
            layer = QwenImageMMDiTLayerPlan(
                image,
                encoded_text,
                self.time_activation.output,
                text_valid,
                weights,
                config,
                index,
                image_rope,
                text_rope,
                cublas=cublas,
                scratch=self.layer_scratch,
            )
            self.layers.append(layer)
            image, encoded_text = layer.image_output, layer.text_output
        self.final_modulation = BiasedLinearPlan(
            self.time_activation.output,
            weights["norm_out.linear.weight"],
            weights["norm_out.linear.bias"],
            cublas=cublas,
        )
        self.final_mod = self.final_modulation.output.reshape(
            (image.shape[0], 2, image.shape[2])
        )
        self.final_norm = AdaptiveLayerNormPlan(
            image, self.final_mod, shift_index=1, scale_index=0
        )
        self.projection = BiasedLinearPlan(
            self.final_norm.output,
            weights["proj_out.weight"],
            weights["proj_out.bias"],
            cublas=cublas,
        )
        self.output = self.projection.output
        self.layer_scratch_nbytes = self.layer_scratch.nbytes
        self.activation_workspace_nbytes = qwen_image_mmdit_workspace_bytes(
            config,
            image_tokens.shape[1],
            text.shape[1],
            batch=image_tokens.shape[0],
            element_bytes=wp.types.type_size_in_bytes(image_tokens.dtype),
        )

    def execute(self):
        self.image_input.execute()
        self.text_norm.execute()
        self.text_input.execute()
        self.time_frequency.execute()
        self.time_linear1.execute()
        self.time_linear2.execute()
        self.time_activation.execute()
        for layer in self.layers:
            layer.execute()
        self.final_modulation.execute()
        self.final_norm.execute()
        self.projection.execute()
        return self.output

    def capture(self):
        self.execute()
        wp.synchronize_stream(wp.get_stream(self.device))
        wp.capture_begin(device=self.device)
        self.execute()
        self.graph = wp.capture_end(device=self.device)
        return self.graph

    def replay(self, *, image_tokens=None, text=None, text_valid=None, timestep=None):
        for target, value in (
            (self.image_tokens, image_tokens),
            (self.text, text),
            (self.text_valid, text_valid),
            (self.timestep, timestep),
        ):
            if value is not None:
                target.assign(value)
        if self.graph is None:
            self.capture()
        wp.capture_launch(self.graph)
        return self.output


def load_qwen_image_transformer_weights(archive, config, device, dtype=wp.bfloat16):
    """Validate metadata, then stream transformer tensors into final device buffers."""
    manifest = QwenImageTransformerManifest.from_config(config)
    manifest.validate_archive(archive)
    return load_cast_weights(archive, manifest.names, device, dtype)
