# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Native FLUX.2 Klein transformer assembled from shared Warp operators."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import warp as wp

from ..formats.safetensors import SafeTensorArchive
from ..operators import (
    ActivationBufferPool,
    AdaptiveLayerNormPlan,
    AttentionHeadsPlan,
    AttentionMergePlan,
    BidirectionalGQAPlan,
    BroadcastGatedResidualPlan,
    ElementwiseActivationPlan,
    FusedSwiGLUPlan,
    JointBidirectionalAttentionPlan,
    LinearPlan,
    RMSNormPlan,
    RotaryCachePlan,
    SinusoidalEmbeddingPlan,
    multi_axis_rotary_cache_values,
    reuse_plan_output,
)
from ..weights import load_cast_weights


@lru_cache(maxsize=None)
def _kernels(dtype):
    D = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def concat(a: wp.array3d(dtype=D), b: wp.array3d(dtype=D), y: wp.array3d(dtype=D)):
        batch, token, channel = wp.tid()
        if token < a.shape[1]:
            y[batch, token, channel] = a[batch, token, channel]
        else:
            y[batch, token, channel] = b[batch, token - a.shape[1], channel]

    @wp.kernel(enable_backward=False, module="unique")
    def split_single(
        x: wp.array3d(dtype=D),
        q: wp.array4d(dtype=D),
        k: wp.array4d(dtype=D),
        v: wp.array4d(dtype=D),
        mlp: wp.array3d(dtype=D),
    ):
        batch, token, channel = wp.tid()
        width = q.shape[1] * q.shape[3]
        if channel < width:
            head = channel // q.shape[3]
            part = channel % q.shape[3]
            q[batch, head, token, part] = x[batch, token, channel]
            k[batch, head, token, part] = x[batch, token, width + channel]
            v[batch, head, token, part] = x[batch, token, 2 * width + channel]
        if channel < mlp.shape[2]:
            mlp[batch, token, channel] = x[batch, token, 3 * width + channel]

    @wp.kernel(enable_backward=False, module="unique")
    def concat_features(
        a: wp.array3d(dtype=D), b: wp.array3d(dtype=D), y: wp.array3d(dtype=D)
    ):
        batch, token, channel = wp.tid()
        if channel < a.shape[2]:
            y[batch, token, channel] = a[batch, token, channel]
        else:
            y[batch, token, channel] = b[batch, token, channel - a.shape[2]]

    @wp.kernel(enable_backward=False, module="unique")
    def slice_image(x: wp.array3d(dtype=D), y: wp.array3d(dtype=D), offset: int):
        batch, token, channel = wp.tid()
        y[batch, token, channel] = x[batch, token + offset, channel]

    return concat, split_single, concat_features, slice_image


class _QKV:
    def __init__(self, x, weights, prefix, heads, rope, cublas=None, *, text=False):
        stem = "add_{}_proj" if text else "to_{}"
        self.projections = [
            LinearPlan(
                x, weights[f"{prefix}.{stem.format(part)}.weight"], cublas=cublas
            )
            for part in ("q", "k", "v")
        ]
        self.heads = [AttentionHeadsPlan(p.output, heads) for p in self.projections]
        q_scale = "norm_added_q" if text else "norm_q"
        k_scale = "norm_added_k" if text else "norm_k"
        self.q_norm = RMSNormPlan(
            self.heads[0].output, weights[f"{prefix}.{q_scale}.weight"]
        )
        self.k_norm = RMSNormPlan(
            self.heads[1].output, weights[f"{prefix}.{k_scale}.weight"]
        )
        self.q_rope = RotaryCachePlan(self.q_norm.output, *rope)
        self.k_rope = RotaryCachePlan(self.k_norm.output, *rope)
        self.output = (self.q_rope.output, self.k_rope.output, self.heads[2].output)

    def execute(self):
        for plan in self.projections + self.heads:
            plan.execute()
        self.q_norm.execute()
        self.k_norm.execute()
        self.q_rope.execute()
        self.k_rope.execute()


class _FeedForward:
    def __init__(self, x, weights, prefix, cublas=None):
        self.first = LinearPlan(x, weights[f"{prefix}.linear_in.weight"], cublas=cublas)
        self.activation = FusedSwiGLUPlan(self.first.output)
        self.last = LinearPlan(
            self.activation.output,
            weights[f"{prefix}.linear_out.weight"],
            cublas=cublas,
        )
        self.output = self.last.output

    def execute(self):
        self.first.execute()
        self.activation.execute()
        return self.last.execute()


class _DoubleBlock:
    def __init__(
        self, image, text, weights, prefix, img_mod, txt_mod, heads, ropes, cublas=None
    ):
        self.img_norm = AdaptiveLayerNormPlan(
            image, img_mod, shift_index=0, scale_index=1
        )
        self.txt_norm = AdaptiveLayerNormPlan(
            text, txt_mod, shift_index=0, scale_index=1
        )
        self.img_qkv = _QKV(
            self.img_norm.output, weights, prefix + ".attn", heads, ropes[1], cublas
        )
        self.txt_qkv = _QKV(
            self.txt_norm.output,
            weights,
            prefix + ".attn",
            heads,
            ropes[0],
            cublas,
            text=True,
        )
        self.attn = JointBidirectionalAttentionPlan(
            self.txt_qkv.output, self.img_qkv.output
        )
        self.txt_merge = AttentionMergePlan(self.attn.first_output)
        self.img_merge = AttentionMergePlan(self.attn.second_output)
        self.txt_out = LinearPlan(
            self.txt_merge.output,
            weights[f"{prefix}.attn.to_add_out.weight"],
            cublas=cublas,
        )
        self.img_out = LinearPlan(
            self.img_merge.output,
            weights[f"{prefix}.attn.to_out.0.weight"],
            cublas=cublas,
        )
        self.txt_res1 = BroadcastGatedResidualPlan(
            text, self.txt_out.output, txt_mod, gate_index=2
        )
        self.img_res1 = BroadcastGatedResidualPlan(
            image, self.img_out.output, img_mod, gate_index=2
        )
        self.txt_norm2 = AdaptiveLayerNormPlan(
            self.txt_res1.output, txt_mod, shift_index=3, scale_index=4
        )
        self.img_norm2 = AdaptiveLayerNormPlan(
            self.img_res1.output, img_mod, shift_index=3, scale_index=4
        )
        self.txt_ff = _FeedForward(
            self.txt_norm2.output, weights, prefix + ".ff_context", cublas
        )
        self.img_ff = _FeedForward(
            self.img_norm2.output, weights, prefix + ".ff", cublas
        )
        self.txt_res2 = BroadcastGatedResidualPlan(
            self.txt_res1.output, self.txt_ff.output, txt_mod, gate_index=5
        )
        self.img_res2 = BroadcastGatedResidualPlan(
            self.img_res1.output, self.img_ff.output, img_mod, gate_index=5
        )
        self.output = (self.txt_res2.output, self.img_res2.output)

    def execute(self):
        self.img_norm.execute()
        self.txt_norm.execute()
        self.img_qkv.execute()
        self.txt_qkv.execute()
        self.attn.execute()
        self.txt_merge.execute()
        self.img_merge.execute()
        self.txt_out.execute()
        self.img_out.execute()
        self.txt_res1.execute()
        self.img_res1.execute()
        self.txt_norm2.execute()
        self.img_norm2.execute()
        self.txt_ff.execute()
        self.img_ff.execute()
        self.txt_res2.execute()
        self.img_res2.execute()


class _SingleBlock:
    def __init__(
        self,
        x,
        weights,
        prefix,
        modulation,
        heads,
        rope,
        cublas=None,
        *,
        scratch=None,
        index=0,
    ):
        self.norm = AdaptiveLayerNormPlan(x, modulation, shift_index=0, scale_index=1)
        if scratch is not None:
            reuse_plan_output(self.norm.norm, scratch, "single.norm.raw")
            reuse_plan_output(self.norm, scratch, "single.norm")
        self.proj = LinearPlan(
            self.norm.output,
            weights[f"{prefix}.attn.to_qkv_mlp_proj.weight"],
            cublas=cublas,
        )
        if scratch is not None:
            reuse_plan_output(self.proj, scratch, "single.proj")
        width = x.shape[2]
        mlp_width = (self.proj.output.shape[2] - 3 * width) // 2
        shape = (x.shape[0], heads, x.shape[1], width // heads)
        self.q = wp.empty(shape, dtype=x.dtype, device=x.device)
        self.k = wp.empty_like(self.q)
        self.v = wp.empty_like(self.q)
        self.mlp = wp.empty(
            (*x.shape[:2], 2 * mlp_width), dtype=x.dtype, device=x.device
        )
        if scratch is not None:
            self.q = scratch.bind("single.q", self.q)
            self.k = scratch.bind("single.k", self.k)
            self.v = scratch.bind("single.v", self.v)
            self.mlp = scratch.bind("single.mlp", self.mlp)
        self.q_norm = RMSNormPlan(self.q, weights[f"{prefix}.attn.norm_q.weight"])
        self.k_norm = RMSNormPlan(self.k, weights[f"{prefix}.attn.norm_k.weight"])
        if scratch is not None:
            reuse_plan_output(self.q_norm, scratch, "single.q_norm")
            reuse_plan_output(self.k_norm, scratch, "single.k_norm")
        self.q_rope = RotaryCachePlan(self.q_norm.output, *rope)
        self.k_rope = RotaryCachePlan(self.k_norm.output, *rope)
        if scratch is not None:
            reuse_plan_output(self.q_rope, scratch, "single.q_rope")
            reuse_plan_output(self.k_rope, scratch, "single.k_rope")
        self.attn = BidirectionalGQAPlan(self.q_rope.output, self.k_rope.output, self.v)
        if scratch is not None:
            reuse_plan_output(self.attn, scratch, "single.attn")
        self.merge = AttentionMergePlan(self.attn.output)
        if scratch is not None:
            reuse_plan_output(self.merge, scratch, "single.merge")
        self.activation = FusedSwiGLUPlan(self.mlp)
        if scratch is not None:
            reuse_plan_output(self.activation, scratch, "single.activated")
        self.activated = self.activation.output
        self.combined = wp.empty(
            (*x.shape[:2], width + mlp_width), dtype=x.dtype, device=x.device
        )
        if scratch is not None:
            self.combined = scratch.bind("single.combined", self.combined)
        self.out = LinearPlan(
            self.combined, weights[f"{prefix}.attn.to_out.weight"], cublas=cublas
        )
        if scratch is not None:
            reuse_plan_output(self.out, scratch, "single.out")
        self.residual = BroadcastGatedResidualPlan(
            x, self.out.output, modulation, gate_index=2
        )
        if scratch is not None:
            reuse_plan_output(self.residual, scratch, f"single.hidden.{index % 2}")
        self.output = self.residual.output

    def execute(self):
        self.norm.execute()
        self.proj.execute()
        wp.launch(
            _kernels(self.q.dtype)[1],
            dim=(
                self.proj.output.shape[0],
                self.proj.output.shape[1],
                max(self.q.shape[1] * self.q.shape[3], self.mlp.shape[2]),
            ),
            inputs=[self.proj.output, self.q, self.k, self.v, self.mlp],
            device=self.q.device,
        )
        self.q_norm.execute()
        self.k_norm.execute()
        self.q_rope.execute()
        self.k_rope.execute()
        self.attn.execute()
        self.merge.execute()
        self.activation.execute()
        wp.launch(
            _kernels(self.q.dtype)[2],
            dim=self.combined.shape,
            inputs=[self.merge.output, self.activated, self.combined],
            device=self.q.device,
        )
        self.out.execute()
        return self.residual.execute()


def _rotary(text_length, height, width, axes, theta, device):
    text = np.zeros((text_length, 4), dtype=np.float32)
    text[:, 3] = np.arange(text_length, dtype=np.float32)
    image = np.zeros((height * width, 4), dtype=np.float32)
    image[:, 1] = np.repeat(np.arange(height, dtype=np.float32), width)
    image[:, 2] = np.tile(np.arange(width, dtype=np.float32), height)

    def upload(coordinates):
        cos, sin = multi_axis_rotary_cache_values(coordinates, axes, theta=theta)
        # The reference rotates BF16 queries with FP32 trigonometric values.
        return wp.array(cos, dtype=wp.float32, device=device), wp.array(
            sin, dtype=wp.float32, device=device
        )

    return upload(text), upload(image), upload(np.concatenate((text, image)))


class Flux2KleinTransformerPlan:
    """Fixed-shape, graph-captured four-step Klein denoiser."""

    def __init__(
        self, image, text, sigma, weights, config, height, width, *, cublas=None
    ):
        if image.shape != (1, height * width, 128) or text.shape != (1, 512, 7680):
            raise ValueError("FLUX.2 Klein image/text geometry is incompatible")
        if sigma.shape != (1,) or sigma.dtype != wp.float32:
            raise ValueError("FLUX.2 sigma must be one FP32 value")
        self.image, self.text, self.sigma = image, text, sigma
        self.device = image.device
        self.graph = None
        self._warmed = False
        heads = int(config["num_attention_heads"])
        hidden = heads * int(config["attention_head_dim"])
        ropes = _rotary(
            text.shape[1],
            height,
            width,
            config["axes_dims_rope"],
            config["rope_theta"],
            image.device,
        )
        self.time_freq = SinusoidalEmbeddingPlan(
            sigma,
            256,
            dtype=image.dtype,
            scale=1000.0,
            frequency_shift=0.0,
            flip_sin_cos=True,
            quantize_input=True,
            quantize_scaled=True,
        )
        self.time_in = LinearPlan(
            self.time_freq.output,
            weights["time_guidance_embed.timestep_embedder.linear_1.weight"],
            cublas=cublas,
        )
        self.time_act = ElementwiseActivationPlan(self.time_in.output, "silu")
        self.time_out = LinearPlan(
            self.time_act.output,
            weights["time_guidance_embed.timestep_embedder.linear_2.weight"],
            cublas=cublas,
        )
        self.mod_act = ElementwiseActivationPlan(self.time_out.output, "silu")
        self.img_mod = LinearPlan(
            self.mod_act.output,
            weights["double_stream_modulation_img.linear.weight"],
            cublas=cublas,
        )
        self.txt_mod = LinearPlan(
            self.mod_act.output,
            weights["double_stream_modulation_txt.linear.weight"],
            cublas=cublas,
        )
        self.single_mod = LinearPlan(
            self.mod_act.output,
            weights["single_stream_modulation.linear.weight"],
            cublas=cublas,
        )
        img_mod = self.img_mod.output.reshape((1, 6, hidden))
        txt_mod = self.txt_mod.output.reshape((1, 6, hidden))
        single_mod = self.single_mod.output.reshape((1, 3, hidden))
        self.img_in = LinearPlan(image, weights["x_embedder.weight"], cublas=cublas)
        self.txt_in = LinearPlan(
            text, weights["context_embedder.weight"], cublas=cublas
        )
        condition, target = self.txt_in.output, self.img_in.output
        self.double = []
        for index in range(int(config["num_layers"])):
            block = _DoubleBlock(
                target,
                condition,
                weights,
                f"transformer_blocks.{index}",
                img_mod,
                txt_mod,
                heads,
                ropes[:2],
                cublas,
            )
            self.double.append(block)
            condition, target = block.output
        self.joint = wp.empty(
            (1, text.shape[1] + image.shape[1], hidden),
            dtype=image.dtype,
            device=image.device,
        )
        joint = self.joint
        self.single = []
        single_scratch = ActivationBufferPool()
        for index in range(int(config["num_single_layers"])):
            block = _SingleBlock(
                joint,
                weights,
                f"single_transformer_blocks.{index}",
                single_mod,
                heads,
                ropes[2],
                cublas,
                scratch=single_scratch,
                index=index,
            )
            self.single.append(block)
            joint = block.output
        self.image_only = wp.empty(
            (1, image.shape[1], hidden), dtype=image.dtype, device=image.device
        )
        self.final_act = ElementwiseActivationPlan(self.time_out.output, "silu")
        self.final_mod = LinearPlan(
            self.final_act.output, weights["norm_out.linear.weight"], cublas=cublas
        )
        self.final_norm = AdaptiveLayerNormPlan(
            self.image_only,
            self.final_mod.output.reshape((1, 2, hidden)),
            shift_index=1,
            scale_index=0,
        )
        self.projection = LinearPlan(
            self.final_norm.output, weights["proj_out.weight"], cublas=cublas
        )
        self.output = self.projection.output
        self._condition = condition
        self._target = target
        self._joint_end = joint

    def execute(self):
        for plan in (
            self.time_freq,
            self.time_in,
            self.time_act,
            self.time_out,
            self.mod_act,
            self.img_mod,
            self.txt_mod,
            self.single_mod,
            self.img_in,
            self.txt_in,
        ):
            plan.execute()
        for block in self.double:
            block.execute()
        wp.launch(
            _kernels(self.image.dtype)[0],
            dim=self.joint.shape,
            inputs=[self._condition, self._target, self.joint],
            device=self.device,
        )
        for block in self.single:
            block.execute()
        wp.launch(
            _kernels(self.image.dtype)[3],
            dim=self.image_only.shape,
            inputs=[self._joint_end, self.image_only, self.text.shape[1]],
            device=self.device,
        )
        self.final_act.execute()
        self.final_mod.execute()
        self.final_norm.execute()
        output = self.projection.execute()
        self._warmed = True
        return output

    def replay(self):
        if not self.device.is_cuda:
            return self.execute()
        if self.graph is None:
            if not self._warmed:
                self.execute()
            wp.synchronize_stream(wp.get_stream(self.device))
            wp.capture_begin(device=self.device)
            self.execute()
            self.graph = wp.capture_end(device=self.device)
        wp.capture_launch(self.graph)
        return self.output


def load_flux2_klein_weights(path, device, dtype=wp.bfloat16):
    """Load the Diffusers transformer once, without the duplicate native checkpoint."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("in_channels"),
        config.get("joint_attention_dim"),
        config.get("num_layers"),
        config.get("num_single_layers"),
    ) != (128, 7680, 5, 20):
        raise ValueError("unsupported FLUX.2 Klein transformer configuration")
    archive = SafeTensorArchive(path / "diffusion_pytorch_model.safetensors")
    names = {
        "x_embedder.weight",
        "context_embedder.weight",
        "proj_out.weight",
        "norm_out.linear.weight",
        "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.linear.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
    }
    for index in range(5):
        p = f"transformer_blocks.{index}"
        names.update(
            f"{p}.attn.{name}.weight"
            for name in (
                "to_q",
                "to_k",
                "to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "norm_q",
                "norm_k",
                "norm_added_q",
                "norm_added_k",
                "to_out.0",
                "to_add_out",
            )
        )
        names.update(
            f"{p}.{module}.{name}.weight"
            for module in ("ff", "ff_context")
            for name in ("linear_in", "linear_out")
        )
    for index in range(20):
        p = f"single_transformer_blocks.{index}.attn"
        names.update(
            f"{p}.{name}.weight"
            for name in ("to_qkv_mlp_proj", "norm_q", "norm_k", "to_out")
        )
    missing = names - set(archive.names)
    if missing:
        raise ValueError(f"FLUX.2 Klein transformer is missing {sorted(missing)[0]}")
    return config, load_cast_weights(archive, names, device, dtype)
