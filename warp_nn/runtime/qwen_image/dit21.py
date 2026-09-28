# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Native, cached text-to-image DiT for Qwen-Image-2.1."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import warp as wp

from ..kernels import _get_gqa_attention_kernel
from ..operators import (
    AttentionHeadsPlan,
    AttentionMergePlan,
    BidirectionalGQAPlan,
    ElementwiseActivationPlan,
    LayerNormPlan,
    Operation,
    RMSNormPlan,
    RotaryCachePlan,
    SinusoidalEmbeddingPlan,
    execute_operations,
    multi_axis_rotary_cache_values,
    plan_linear,
    plan_swiglu,
)
from ..weights import load_cast_weights
from .mmdit import _LayerScratch


@lru_cache(maxsize=None)
def _pointwise_kernels(dtype):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def one_plus(source: wp.array1d(dtype=DTYPE), output: wp.array1d(dtype=DTYPE)):
        channel = wp.tid()
        output[channel] = DTYPE(wp.float32(source[channel]) + 1.0)

    @wp.kernel(enable_backward=False, module="unique")
    def modulate(
        source: wp.array3d(dtype=DTYPE),
        modulation: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        row: int,
        scale_index: int,
    ):
        batch, token, channel = wp.tid()
        value = wp.float32(source[batch, token, channel])
        scale = wp.float32(modulation[row, scale_index, channel])
        output[batch, token, channel] = DTYPE(value * (1.0 + scale))

    @wp.kernel(enable_backward=False, module="unique")
    def residual(
        source: wp.array3d(dtype=DTYPE),
        branch: wp.array3d(dtype=DTYPE),
        modulation: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        row: int,
        gate_index: int,
    ):
        batch, token, channel = wp.tid()
        gate = wp.tanh(wp.float32(modulation[row, gate_index, channel]))
        output[batch, token, channel] = DTYPE(
            wp.float32(source[batch, token, channel])
            + gate * wp.float32(branch[batch, token, channel])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def join(
        prefix: wp.array4d(dtype=DTYPE),
        target: wp.array4d(dtype=DTYPE),
        output: wp.array4d(dtype=DTYPE),
    ):
        batch, head, token, channel = wp.tid()
        if token < prefix.shape[2]:
            output[batch, head, token, channel] = prefix[batch, head, token, channel]
        else:
            output[batch, head, token, channel] = target[
                batch, head, token - prefix.shape[2], channel
            ]

    return one_plus, modulate, residual, join


class _Linear:
    """Bias-free projection through the shared GEMM planner."""

    def __init__(self, x, weight, cublas=None):
        rows = int(np.prod(x.shape[:-1]))
        self.input = x
        self.tensors = {"x": x.reshape((rows, x.shape[-1])), "weight": weight}
        self.shapes = {name: value.shape for name, value in self.tensors.items()}
        self.op = Operation("Linear", ["x", "weight"], ["y"])
        plan_linear(self.op, self.tensors, self.shapes, x.device, cublas=cublas)
        self.output = self.tensors["y"].reshape((*x.shape[:-1], weight.shape[0]))

    def execute(self):
        execute_operations((self.op,), self.tensors, self.shapes, self.input.device)
        return self.output


class _SwiGLU:
    def __init__(self, gate, value):
        self.tensors = {"gate": gate, "value": value}
        self.shapes = {name: item.shape for name, item in self.tensors.items()}
        self.op = Operation("_SwiGLU", ["gate", "value"], ["output"])
        plan_swiglu(self.op, self.tensors, self.shapes, gate.device)
        self.output = self.tensors["output"]

    def execute(self):
        execute_operations((self.op,), self.tensors, self.shapes, self.output.device)
        return self.output


def _share(plan, scratch, name):
    """Bind a layer's activation to a fixed reusable scratch slot."""
    if scratch is None:
        return
    plan.output = scratch.bind(name, plan.output)
    if isinstance(plan, _Linear):
        plan.tensors["y"] = plan.output.reshape(plan.tensors["y"].shape)
    elif isinstance(plan, RMSNormPlan):
        plan._tensors["normalized"] = plan.output
        plan._operation.attrs["_output_2d"] = plan.output.reshape(
            plan._operation.attrs["_output_2d"].shape
        )
    elif isinstance(plan, _SwiGLU):
        plan.tensors["output"] = plan.output
        plan.op.attrs["_output_2d"] = plan.output.reshape(
            plan.op.attrs["_output_2d"].shape
        )


class _Modulation:
    def __init__(self, source, modulation, row, index):
        self.source, self.modulation = source, modulation
        self.row, self.index = row, index
        self.output = wp.empty_like(source)

    def execute(self):
        wp.launch(
            _pointwise_kernels(self.source.dtype)[1],
            dim=self.output.shape,
            inputs=[self.source, self.modulation, self.output, self.row, self.index],
            device=self.source.device,
        )
        return self.output


class _GatedResidual:
    def __init__(self, source, branch, modulation, row, index):
        self.source, self.branch, self.modulation = source, branch, modulation
        self.row, self.index = row, index
        self.output = wp.empty_like(source)

    def execute(self):
        wp.launch(
            _pointwise_kernels(self.source.dtype)[2],
            dim=self.output.shape,
            inputs=[
                self.source,
                self.branch,
                self.modulation,
                self.output,
                self.row,
                self.index,
            ],
            device=self.source.device,
        )
        return self.output


class _KVJoin:
    def __init__(self, prefix, target):
        self.prefix, self.target = prefix, target
        self.output = wp.empty(
            (
                target.shape[0],
                target.shape[1],
                prefix.shape[2] + target.shape[2],
                target.shape[3],
            ),
            dtype=target.dtype,
            device=target.device,
        )

    def execute(self):
        wp.launch(
            _pointwise_kernels(self.target.dtype)[3],
            dim=self.output.shape,
            inputs=[self.prefix, self.target, self.output],
            device=self.target.device,
        )
        return self.output


class _CausalAttention:
    def __init__(self, q, k, v):
        if q.shape != k.shape or k.shape != v.shape or q.shape[0] != 1:
            raise ValueError("2.1 prefix attention requires one matching Q/K/V batch")
        self.q, self.k, self.v = q, k, v
        _, self.heads, self.length, self.width = q.shape
        self.end = wp.array(
            np.array([self.length - 1], dtype=np.int32), device=q.device
        )
        self.output = wp.empty(
            (1, self.length, self.heads * self.width), dtype=q.dtype, device=q.device
        )
        self.block, self.kernel = _get_gqa_attention_kernel(self.width, q.dtype)

    def execute(self):
        wp.launch_tiled(
            self.kernel,
            dim=self.heads * self.length,
            inputs=[
                self.q.reshape((self.heads * self.length, self.width)),
                self.k.reshape((self.heads * self.length, self.width)),
                self.v.reshape((self.heads * self.length, self.width)),
                self.end,
                self.output.reshape((self.length, self.heads * self.width)),
                self.heads,
                self.heads,
                self.length,
                self.length,
                self.width**-0.5,
                0,
                self.length,
            ],
            block_dim=self.block,
            device=self.q.device,
        )
        return self.output


class _Block:
    def __init__(
        self,
        x,
        modulation,
        weights,
        prefix,
        heads,
        rope,
        *,
        row,
        cached=None,
        cublas=None,
        scratch=None,
    ):
        self.norm1 = LayerNormPlan(x)
        _share(self.norm1, scratch, "norm1")
        self.mod1 = _Modulation(self.norm1.output, modulation, row, 0)
        _share(self.mod1, scratch, "mod1")
        stem = prefix + ".attn."
        self.q = _Linear(self.mod1.output, weights[stem + "to_q.weight"], cublas)
        _share(self.q, scratch, "q")
        self.k = _Linear(self.mod1.output, weights[stem + "to_k.weight"], cublas)
        _share(self.k, scratch, "k")
        self.v = _Linear(self.mod1.output, weights[stem + "to_v.weight"], cublas)
        _share(self.v, scratch, "v")
        self.q_heads = AttentionHeadsPlan(self.q.output, heads)
        _share(self.q_heads, scratch, "q_heads")
        self.k_heads = AttentionHeadsPlan(self.k.output, heads)
        _share(self.k_heads, scratch, "k_heads")
        self.v_heads = AttentionHeadsPlan(self.v.output, heads)
        _share(self.v_heads, scratch, "v_heads")
        self.q_norm = RMSNormPlan(self.q_heads.output, weights[stem + "norm_q.weight"])
        _share(self.q_norm, scratch, "q_norm")
        self.k_norm = RMSNormPlan(self.k_heads.output, weights[stem + "norm_k.weight"])
        _share(self.k_norm, scratch, "k_norm")
        self.q_rope = RotaryCachePlan(self.q_norm.output, *rope)
        _share(self.q_rope, scratch, "q_rope")
        self.k_rope = RotaryCachePlan(self.k_norm.output, *rope)
        _share(self.k_rope, scratch, "k_rope")
        if cached is None:
            self.attention = _CausalAttention(
                self.q_rope.output, self.k_rope.output, self.v_heads.output
            )
            attention_output = self.attention.output
            self.k_join = self.v_join = self.merge = None
        else:
            self.k_join = _KVJoin(cached.k_rope.output, self.k_rope.output)
            _share(self.k_join, scratch, "k_join")
            self.v_join = _KVJoin(cached.v_heads.output, self.v_heads.output)
            _share(self.v_join, scratch, "v_join")
            self.attention = BidirectionalGQAPlan(
                self.q_rope.output, self.k_join.output, self.v_join.output
            )
            _share(self.attention, scratch, "attention")
            self.merge = AttentionMergePlan(self.attention.output)
            _share(self.merge, scratch, "merge")
            attention_output = self.merge.output
        self.attn_out = _Linear(
            attention_output, weights[stem + "to_out.0.weight"], cublas
        )
        _share(self.attn_out, scratch, "attn_out")
        self.residual1 = _GatedResidual(x, self.attn_out.output, modulation, row, 1)
        _share(self.residual1, scratch, "residual1")
        self.norm2 = LayerNormPlan(self.residual1.output)
        _share(self.norm2, scratch, "norm2")
        self.mod2 = _Modulation(self.norm2.output, modulation, row, 2)
        _share(self.mod2, scratch, "mod2")
        self.gate = _Linear(
            self.mod2.output, weights[prefix + ".img_mlp.gate_layer.weight"], cublas
        )
        _share(self.gate, scratch, "gate")
        self.up = _Linear(
            self.mod2.output, weights[prefix + ".img_mlp.proj.weight"], cublas
        )
        _share(self.up, scratch, "up")
        self.swiglu = _SwiGLU(self.gate.output, self.up.output)
        _share(self.swiglu, scratch, "swiglu")
        self.down = _Linear(
            self.swiglu.output, weights[prefix + ".img_mlp.out.weight"], cublas
        )
        _share(self.down, scratch, "down")
        self.residual2 = _GatedResidual(
            self.residual1.output, self.down.output, modulation, row, 3
        )
        _share(self.residual2, scratch, "hidden")
        self.output = self.residual2.output

    def execute(self):
        for plan in (
            self.norm1,
            self.mod1,
            self.q,
            self.k,
            self.v,
            self.q_heads,
            self.k_heads,
            self.v_heads,
            self.q_norm,
            self.k_norm,
            self.q_rope,
            self.k_rope,
        ):
            plan.execute()
        if self.k_join is not None:
            self.k_join.execute()
            self.v_join.execute()
        self.attention.execute()
        if self.merge is not None:
            self.merge.execute()
        for plan in (
            self.attn_out,
            self.residual1,
            self.norm2,
            self.mod2,
            self.gate,
            self.up,
            self.swiglu,
            self.down,
            self.residual2,
        ):
            plan.execute()
        return self.output


def _rope(text_length, height, width, axes, dtype, device):
    text = np.arange(text_length, dtype=np.float32)
    text_coords = np.stack((text, text, text), axis=1)
    rows = np.arange(-(height - height // 2), height // 2, dtype=np.float32)
    columns = np.arange(-(width - width // 2), width // 2, dtype=np.float32)
    image_coords = np.empty((height * width, 3), dtype=np.float32)
    image_coords[:, 0] = float(text_length)
    image_coords[:, 1] = np.repeat(rows, width)
    image_coords[:, 2] = np.tile(columns, height)
    result = []
    for coords in (text_coords, image_coords):
        cosine, sine = multi_axis_rotary_cache_values(coords, axes)
        result.append(
            (
                wp.array(cosine, dtype=dtype, device=device),
                wp.array(sine, dtype=dtype, device=device),
            )
        )
    return result


class QwenImage21DiTPlan:
    """A fixed prompt and image shape; prefill text once, then denoise with cached K/V."""

    def __init__(
        self,
        image_tokens,
        text,
        timesteps,
        weights,
        config,
        height,
        width,
        *,
        cublas=None,
    ):
        if (
            image_tokens.shape != (1, height * width, 64)
            or text.ndim != 3
            or text.shape[0] != 1
        ):
            raise ValueError("Qwen-Image-2.1 text/image geometry is incompatible")
        if timesteps.shape != (2,) or timesteps.dtype != wp.float32:
            raise ValueError("timesteps must hold current sigma and zero")
        if any(item.device != image_tokens.device for item in (text, timesteps)):
            raise ValueError("Qwen-Image-2.1 tensors must share one device")
        self.image_tokens, self.text, self.timesteps = image_tokens, text, timesteps
        self.device = image_tokens.device
        self.graph = None
        self.config = config
        heads = int(config["num_attention_heads"])
        width_hidden = heads * int(config["attention_head_dim"])
        axes = tuple(config["axes_dims_rope"])
        text_rope, image_rope = _rope(
            text.shape[1], height, width, axes, image_tokens.dtype, self.device
        )
        self.text_scale = wp.empty_like(weights["txt_in.text_norm.weight"])
        wp.launch(
            _pointwise_kernels(image_tokens.dtype)[0],
            dim=self.text_scale.shape,
            inputs=[weights["txt_in.text_norm.weight"], self.text_scale],
            device=self.device,
        )
        self.txt_norm = RMSNormPlan(text, self.text_scale)
        self.txt_in = _Linear(
            self.txt_norm.output, weights["txt_in.in_layer.weight"], cublas
        )
        self.txt_act = ElementwiseActivationPlan(self.txt_in.output, "gelu_tanh")
        self.txt_out = _Linear(
            self.txt_act.output, weights["txt_in.out_layer.weight"], cublas
        )
        self.img_in = _Linear(image_tokens, weights["img_in.weight"], cublas)
        self.time_freq = SinusoidalEmbeddingPlan(
            timesteps,
            256,
            dtype=image_tokens.dtype,
            scale=1000.0,
            maximum_period=10000.0,
            frequency_shift=0.0,
            flip_sin_cos=True,
            quantize_input=True,
        )
        self.time_in = _Linear(
            self.time_freq.output,
            weights["time_text_embed.timestep_embedder.linear_1.weight"],
            cublas,
        )
        self.time_act = ElementwiseActivationPlan(self.time_in.output, "silu")
        self.time_out = _Linear(
            self.time_act.output,
            weights["time_text_embed.timestep_embedder.linear_2.weight"],
            cublas,
        )
        self.mod_act = ElementwiseActivationPlan(self.time_out.output, "silu")
        self.mod_linear = _Linear(
            self.mod_act.output, weights["modulation.1.weight"], cublas
        )
        self.modulation = self.mod_linear.output.reshape((2, 4, width_hidden))
        self.prefix = []
        self.target = []
        self.layer_scratch = _LayerScratch()
        condition = self.txt_out.output
        image = self.img_in.output
        for index in range(int(config["num_layers"])):
            stem = f"transformer_blocks.{index}"
            prefix = _Block(
                condition,
                self.modulation,
                weights,
                stem,
                heads,
                text_rope,
                row=1,
                cublas=cublas,
            )
            target = _Block(
                image,
                self.modulation,
                weights,
                stem,
                heads,
                image_rope,
                row=0,
                cached=prefix,
                cublas=cublas,
                scratch=self.layer_scratch,
            )
            self.prefix.append(prefix)
            self.target.append(target)
            condition, image = prefix.output, target.output
        self.final_norm = LayerNormPlan(image)
        self.final_act = ElementwiseActivationPlan(self.time_out.output, "silu")
        self.final_scale = _Linear(
            self.final_act.output, weights["norm_out.linear.weight"], cublas
        )
        self.final_scaled = _Modulation(
            self.final_norm.output,
            self.final_scale.output.reshape((2, 1, width_hidden)),
            0,
            0,
        )
        self.projection = _Linear(
            self.final_scaled.output, weights["proj_out.weight"], cublas
        )
        self.output = self.projection.output

    def prefill(self):
        self.timesteps.assign(np.array([0.0, 0.0], dtype=np.float32))
        for plan in (
            self.txt_norm,
            self.txt_in,
            self.txt_act,
            self.txt_out,
            self.time_freq,
            self.time_in,
            self.time_act,
            self.time_out,
            self.mod_act,
            self.mod_linear,
        ):
            plan.execute()
        for layer in self.prefix:
            layer.execute()

    def execute(self):
        for plan in (
            self.img_in,
            self.time_freq,
            self.time_in,
            self.time_act,
            self.time_out,
            self.mod_act,
            self.mod_linear,
        ):
            plan.execute()
        for layer in self.target:
            layer.execute()
        for plan in (
            self.final_norm,
            self.final_act,
            self.final_scale,
            self.final_scaled,
            self.projection,
        ):
            plan.execute()
        return self.output

    def capture(self):
        self.execute()
        wp.synchronize_stream(wp.get_stream(self.device))
        wp.capture_begin(device=self.device)
        self.execute()
        self.graph = wp.capture_end(device=self.device)

    def replay(self):
        if self.graph is None:
            self.capture()
        wp.capture_launch(self.graph)
        return self.output


def load_qwen_image_21_transformer_weights(path, device, dtype=wp.bfloat16):
    """Stream the official 2.1 checkpoint into Warp arrays."""
    from ..formats.safetensors import SafeTensorArchive

    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    archive = SafeTensorArchive(path / "diffusion_pytorch_model.safetensors.index.json")
    heads = int(config["num_attention_heads"])
    width = heads * int(config["attention_head_dim"])
    names = {
        "img_in.weight",
        "modulation.1.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
        "txt_in.in_layer.weight",
        "txt_in.out_layer.weight",
        "txt_in.text_norm.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "time_text_embed.timestep_embedder.linear_2.weight",
    }
    for index in range(int(config["num_layers"])):
        stem = f"transformer_blocks.{index}"
        names.update(
            f"{stem}.attn.{part}.weight"
            for part in ("to_q", "to_k", "to_v", "to_out.0", "norm_q", "norm_k")
        )
        names.update(
            f"{stem}.img_mlp.{part}.weight" for part in ("gate_layer", "proj", "out")
        )
    missing = names - set(archive.names)
    if missing:
        raise ValueError(f"Qwen-Image-2.1 transformer is missing {sorted(missing)[0]}")
    if (
        config.get("in_channels") != 64
        or config.get("out_channels") != 64
        or width != 4096
    ):
        raise ValueError("unsupported Qwen-Image-2.1 transformer geometry")
    return config, load_cast_weights(archive, names, device, dtype)
