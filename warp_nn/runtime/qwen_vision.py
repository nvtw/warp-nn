# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 vision encoder and multimodal prompt preparation."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import warp as wp

from .gguf import GGUFArchive, MappedGGUFArchive
from .operators import Operation, execute_operations, plan_linear
from .safetensors import SafeTensorArchive
from .vision import VisionInput, preprocess_qwen_media, qwen_vision_positions


def _vision_weight_names(depth: int) -> list[str]:
    names = [
        "patch_embed.weight",
        "patch_embed.bias",
        "position_embedding.weight",
        "merger.norm.weight",
        "merger.norm.bias",
        "merger.fc1.weight",
        "merger.fc1.bias",
        "merger.fc2.weight",
        "merger.fc2.bias",
    ]
    suffixes = (
        "norm1.weight",
        "norm1.bias",
        "attention.qkv.weight",
        "attention.qkv.bias",
        "attention.proj.weight",
        "attention.proj.bias",
        "norm2.weight",
        "norm2.bias",
        "mlp.fc1.weight",
        "mlp.fc1.bias",
        "mlp.fc2.weight",
        "mlp.fc2.bias",
    )
    for index in range(depth):
        names.extend(f"blocks.{index}.{suffix}" for suffix in suffixes)
    return names


def _safetensor_map(depth: int) -> dict[str, str]:
    result = {
        "patch_embed.weight": "model.visual.patch_embed.proj.weight",
        "patch_embed.bias": "model.visual.patch_embed.proj.bias",
        "position_embedding.weight": "model.visual.pos_embed.weight",
        "merger.norm.weight": "model.visual.merger.norm.weight",
        "merger.norm.bias": "model.visual.merger.norm.bias",
        "merger.fc1.weight": "model.visual.merger.linear_fc1.weight",
        "merger.fc1.bias": "model.visual.merger.linear_fc1.bias",
        "merger.fc2.weight": "model.visual.merger.linear_fc2.weight",
        "merger.fc2.bias": "model.visual.merger.linear_fc2.bias",
    }
    aliases = {
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "attention.qkv.weight": "attn.qkv.weight",
        "attention.qkv.bias": "attn.qkv.bias",
        "attention.proj.weight": "attn.proj.weight",
        "attention.proj.bias": "attn.proj.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "mlp.fc1.weight": "mlp.linear_fc1.weight",
        "mlp.fc1.bias": "mlp.linear_fc1.bias",
        "mlp.fc2.weight": "mlp.linear_fc2.weight",
        "mlp.fc2.bias": "mlp.linear_fc2.bias",
    }
    for index in range(depth):
        for target, source in aliases.items():
            result[f"blocks.{index}.{target}"] = f"model.visual.blocks.{index}.{source}"
    return result


def _gguf_map(depth: int) -> dict[str, str]:
    result = {
        "patch_embed.weight.0": "v.patch_embd.weight",
        "patch_embed.weight.1": "v.patch_embd.weight.1",
        "patch_embed.bias": "v.patch_embd.bias",
        "position_embedding.weight": "v.position_embd.weight",
        "merger.norm.weight": "v.post_ln.weight",
        "merger.norm.bias": "v.post_ln.bias",
        "merger.fc1.weight": "mm.0.weight",
        "merger.fc1.bias": "mm.0.bias",
        "merger.fc2.weight": "mm.2.weight",
        "merger.fc2.bias": "mm.2.bias",
    }
    aliases = {
        "norm1.weight": "ln1.weight",
        "norm1.bias": "ln1.bias",
        "attention.qkv.weight": "attn_qkv.weight",
        "attention.qkv.bias": "attn_qkv.bias",
        "attention.proj.weight": "attn_out.weight",
        "attention.proj.bias": "attn_out.bias",
        "norm2.weight": "ln2.weight",
        "norm2.bias": "ln2.bias",
        "mlp.fc1.weight": "ffn_up.weight",
        "mlp.fc1.bias": "ffn_up.bias",
        "mlp.fc2.weight": "ffn_down.weight",
        "mlp.fc2.bias": "ffn_down.bias",
    }
    for index in range(depth):
        for target, source in aliases.items():
            result[f"blocks.{index}.{target}"] = f"v.blk.{index}.{source}"
    return result


@lru_cache(maxsize=None)
def _vision_kernels(dtype: type, heads: int, head_size: int, parameter_dtype: type):
    DTYPE, PARAM = dtype, parameter_dtype
    HEADS, HEAD_SIZE = heads, head_size

    @wp.kernel(enable_backward=False, module="unique")
    def cast_2d(source: wp.array2d[wp.float32], output: wp.array2d(dtype=DTYPE)):
        i, j = wp.tid()
        output[i, j] = DTYPE(source[i, j])

    @wp.kernel(enable_backward=False, module="unique")
    def cast_1d(source: wp.array1d[wp.float32], output: wp.array1d(dtype=DTYPE)):
        i = wp.tid()
        output[i] = DTYPE(source[i])

    @wp.kernel(enable_backward=False, module="unique")
    def pack_patch_weights(
        first: wp.array2d(dtype=DTYPE),
        second: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        channel = column / 512
        temporal = (column / 256) % 2
        source_column = channel * 256 + column % 256
        output[row, column] = (
            first[row, source_column] if temporal == 0 else second[row, source_column]
        )

    @wp.kernel(enable_backward=False, module="unique")
    def add_position(
        x: wp.array2d(dtype=DTYPE),
        position: wp.array2d(dtype=PARAM),
        indices: wp.array2d[wp.int64],
        grid_h: int,
        grid_w: int,
    ):
        row, column = wp.tid()
        y = (
            wp.float32(indices[row, 0])
            * wp.float32(47.0)
            / wp.float32(wp.max(grid_h - 1, 1))
        )
        z = (
            wp.float32(indices[row, 1])
            * wp.float32(47.0)
            / wp.float32(wp.max(grid_w - 1, 1))
        )
        y0, z0 = wp.int32(wp.floor(y)), wp.int32(wp.floor(z))
        y1, z1 = wp.min(y0 + 1, 47), wp.min(z0 + 1, 47)
        fy, fz = y - wp.float32(y0), z - wp.float32(z0)
        value = (
            wp.float32(position[y0 * 48 + z0, column])
            * (wp.float32(1.0) - fy)
            * (wp.float32(1.0) - fz)
            + wp.float32(position[y0 * 48 + z1, column]) * (wp.float32(1.0) - fy) * fz
            + wp.float32(position[y1 * 48 + z0, column]) * fy * (wp.float32(1.0) - fz)
            + wp.float32(position[y1 * 48 + z1, column]) * fy * fz
        )
        x[row, column] = DTYPE(wp.float32(x[row, column]) + value)

    @wp.kernel(enable_backward=False, module="unique")
    def layer_norm(
        x: wp.array2d(dtype=DTYPE),
        scale: wp.array1d(dtype=PARAM),
        bias: wp.array1d(dtype=PARAM),
        output: wp.array2d(dtype=DTYPE),
        epsilon: wp.float32,
    ):
        row = wp.tid()
        width = x.shape[1]
        mean = wp.float32(0.0)
        for column in range(width):
            mean += wp.float32(x[row, column])
        mean /= wp.float32(width)
        variance = wp.float32(0.0)
        for column in range(width):
            centered = wp.float32(x[row, column]) - mean
            variance += centered * centered
        inverse = wp.float32(1.0) / wp.sqrt(variance / wp.float32(width) + epsilon)
        for column in range(width):
            output[row, column] = DTYPE(
                (wp.float32(x[row, column]) - mean)
                * inverse
                * wp.float32(scale[column])
                + wp.float32(bias[column])
            )

    @wp.kernel(enable_backward=False, module="unique")
    def add_bias(x: wp.array2d(dtype=DTYPE), bias: wp.array1d(dtype=PARAM)):
        row, column = wp.tid()
        x[row, column] = DTYPE(wp.float32(x[row, column]) + wp.float32(bias[column]))

    @wp.kernel(enable_backward=False, module="unique")
    def split_qkv_rotary(
        packed: wp.array2d(dtype=DTYPE),
        positions: wp.array2d[wp.int64],
        inverse_frequency: wp.array1d[wp.float32],
        query: wp.array3d(dtype=DTYPE),
        key: wp.array3d(dtype=DTYPE),
        value: wp.array3d(dtype=DTYPE),
    ):
        token, head, column = wp.tid()
        offset = head * HEAD_SIZE + column
        half = HEAD_SIZE / 2
        frequency_column = column % half
        axis = frequency_column / inverse_frequency.shape[0]
        frequency = frequency_column % inverse_frequency.shape[0]
        angle = wp.float32(positions[token, axis]) * inverse_frequency[frequency]
        cosine, sine = wp.cos(angle), wp.sin(angle)
        partner_column = column + half if column < half else column - half
        sign = wp.float32(-1.0) if column < half else wp.float32(1.0)
        partner = head * HEAD_SIZE + partner_column
        query[token, head, column] = DTYPE(
            wp.float32(packed[token, offset]) * cosine
            + sign * wp.float32(packed[token, partner]) * sine
        )
        key_offset, key_partner = (
            HEADS * HEAD_SIZE + offset,
            HEADS * HEAD_SIZE + partner,
        )
        key[token, head, column] = DTYPE(
            wp.float32(packed[token, key_offset]) * cosine
            + sign * wp.float32(packed[token, key_partner]) * sine
        )
        value[token, head, column] = packed[token, 2 * HEADS * HEAD_SIZE + offset]

    @wp.func
    def dot(left: DTYPE, right: DTYPE):
        return wp.float32(left) * wp.float32(right)

    @wp.func
    def update(
        total: wp.float32,
        current: DTYPE,
        old_scale: wp.float32,
        probability: wp.float32,
    ):
        return total * old_scale + wp.float32(current) * probability

    @wp.func
    def normalize(total: wp.float32, denominator: wp.float32):
        return total / denominator

    @wp.kernel(enable_backward=False, module="unique")
    def packed_attention(
        query: wp.array3d(dtype=DTYPE),
        key: wp.array3d(dtype=DTYPE),
        value: wp.array3d(dtype=DTYPE),
        segments: wp.array1d[wp.int32],
        output: wp.array3d(dtype=DTYPE),
        scale: wp.float32,
    ):
        item = wp.tid()
        token, head = item % query.shape[0], item / query.shape[0]
        segment = segments[token]
        q = wp.tile_load(query[token, head], shape=(HEAD_SIZE,))
        accumulator = wp.tile_zeros(shape=(HEAD_SIZE,), dtype=wp.float32)
        maximum, denominator = wp.float32(-3.402823466e38), wp.float32(0.0)
        for source in range(query.shape[0]):
            if segments[source] == segment:
                k = wp.tile_load(key[source, head], shape=(HEAD_SIZE,))
                score = wp.tile_extract(wp.tile_sum(wp.tile_map(dot, q, k)), 0) * scale
                new_maximum = wp.max(maximum, score)
                old_scale, probability = (
                    wp.exp(maximum - new_maximum),
                    wp.exp(score - new_maximum),
                )
                denominator = denominator * old_scale + probability
                v = wp.tile_load(value[source, head], shape=(HEAD_SIZE,))
                accumulator = wp.tile_map(
                    update, accumulator, v, old_scale, probability
                )
                maximum = new_maximum
        wp.tile_store(
            output[token, head],
            wp.tile_astype(
                wp.tile_map(normalize, accumulator, denominator), dtype=DTYPE
            ),
        )

    @wp.kernel(enable_backward=False, module="unique")
    def merge_heads(x: wp.array3d(dtype=DTYPE), output: wp.array2d(dtype=DTYPE)):
        token, head, column = wp.tid()
        output[token, head * HEAD_SIZE + column] = x[token, head, column]

    @wp.kernel(enable_backward=False, module="unique")
    def bias_residual(
        branch: wp.array2d(dtype=DTYPE),
        bias: wp.array1d(dtype=PARAM),
        residual: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        output[row, column] = DTYPE(
            wp.float32(branch[row, column])
            + wp.float32(bias[column])
            + wp.float32(residual[row, column])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def bias_gelu(
        x: wp.array2d(dtype=DTYPE), bias: wp.array1d(dtype=PARAM), approximate: bool
    ):
        row, column = wp.tid()
        value = wp.float32(x[row, column]) + wp.float32(bias[column])
        if approximate:
            cubic = value * value * value
            value *= wp.float32(0.5) * (
                wp.float32(1.0)
                + wp.tanh(
                    wp.float32(0.7978845608028654)
                    * (value + wp.float32(0.044715) * cubic)
                )
            )
        else:
            value *= wp.float32(0.5) * (
                wp.float32(1.0) + wp.erf(value * wp.float32(0.7071067811865476))
            )
        x[row, column] = DTYPE(value)

    @wp.kernel(enable_backward=False, module="unique")
    def merge_patches(x: wp.array2d(dtype=DTYPE), output: wp.array2d(dtype=DTYPE)):
        row, column = wp.tid()
        width = x.shape[1]
        output[row, column] = x[row * 4 + column / width, column % width]

    return (
        cast_2d,
        cast_1d,
        pack_patch_weights,
        add_position,
        layer_norm,
        add_bias,
        split_qkv_rotary,
        packed_attention,
        merge_heads,
        bias_residual,
        bias_gelu,
        merge_patches,
    )


class _VisionPlan:
    """Fixed-grid Qwen vision graph with shared sequential layer buffers."""

    def __init__(self, encoder: "QwenVisionEncoder", grid_thw: tuple[int, int, int]):
        self.encoder = encoder
        self.device, self.dtype = encoder.device, encoder.dtype
        self.grid_thw = grid_thw
        t, grid_h, grid_w = grid_thw
        self.rows = t * grid_h * grid_w
        self.features = self.rows // 4
        hidden, intermediate = encoder.hidden_size, encoder.intermediate_size
        heads, head_size = encoder.heads, encoder.head_size
        self.kernels = _vision_kernels(
            self.dtype, heads, head_size, encoder.parameter_dtype
        )
        self.tensors = dict(encoder.weights)
        self.shapes = {name: tuple(value.shape) for name, value in self.tensors.items()}
        self.patches = wp.empty((self.rows, 1536), dtype=self.dtype, device=self.device)
        self.positions = wp.empty((self.rows, 2), dtype=wp.int64, device=self.device)
        self.segments = wp.empty(self.rows, dtype=wp.int32, device=self.device)
        self.tensors["patches"], self.shapes["patches"] = (
            self.patches,
            self.patches.shape,
        )

        self.hidden_slots = (
            wp.empty((self.rows, hidden), dtype=self.dtype, device=self.device),
            wp.empty((self.rows, hidden), dtype=self.dtype, device=self.device),
        )
        self.norm = wp.empty((self.rows, hidden), dtype=self.dtype, device=self.device)
        self.qkv = wp.empty(
            (self.rows, hidden * 3), dtype=self.dtype, device=self.device
        )
        self.query = wp.empty(
            (self.rows, heads, head_size), dtype=self.dtype, device=self.device
        )
        self.key = wp.empty_like(self.query)
        self.value = wp.empty_like(self.query)
        self.attention = wp.empty_like(self.query)
        self.merged_heads = wp.empty(
            (self.rows, hidden), dtype=self.dtype, device=self.device
        )
        self.branch = wp.empty_like(self.merged_heads)
        self.mlp = wp.empty(
            (self.rows, intermediate), dtype=self.dtype, device=self.device
        )

        self.patch = self._linear("hidden.0", "patches", "patch_embed.weight")
        self.tensors["hidden.0"] = self.hidden_slots[0]
        self.layers = []
        for index in range(encoder.depth):
            source = f"hidden.{index}"
            target = f"hidden.{index + 1}"
            self.tensors[source] = self.hidden_slots[index % 2]
            self.shapes[source] = self.hidden_slots[index % 2].shape
            prefix = f"blocks.{index}."
            self.tensors[f"norm1.{index}"] = self.norm
            self.shapes[f"norm1.{index}"] = self.norm.shape
            self.tensors[f"attention_merge.{index}"] = self.merged_heads
            self.shapes[f"attention_merge.{index}"] = self.merged_heads.shape
            self.tensors[f"norm2.{index}"] = self.norm
            self.shapes[f"norm2.{index}"] = self.norm.shape
            qkv = self._linear(
                f"qkv.{index}", f"norm1.{index}", prefix + "attention.qkv.weight"
            )
            projection = self._linear(
                f"attention_branch.{index}",
                f"attention_merge.{index}",
                prefix + "attention.proj.weight",
            )
            fc1 = self._linear(
                f"mlp.{index}", f"norm2.{index}", prefix + "mlp.fc1.weight"
            )
            fc2 = self._linear(
                f"mlp_branch.{index}", f"mlp.{index}", prefix + "mlp.fc2.weight"
            )
            self.tensors[f"norm1.{index}"] = self.norm
            self.shapes[f"norm1.{index}"] = self.norm.shape
            self.tensors[f"qkv.{index}"] = self.qkv
            self.tensors[f"attention_merge.{index}"] = self.merged_heads
            self.shapes[f"attention_merge.{index}"] = self.merged_heads.shape
            self.tensors[f"attention_branch.{index}"] = self.branch
            self.tensors[f"norm2.{index}"] = self.norm
            self.shapes[f"norm2.{index}"] = self.norm.shape
            self.tensors[f"mlp.{index}"] = self.mlp
            self.tensors[f"mlp_branch.{index}"] = self.branch
            self.tensors[target] = self.hidden_slots[(index + 1) % 2]
            self.shapes[target] = self.hidden_slots[(index + 1) % 2].shape
            self.layers.append((prefix, source, target, qkv, projection, fc1, fc2))

        final_hidden = f"hidden.{encoder.depth}"
        self.merger_norm = wp.empty(
            (self.rows, hidden), dtype=self.dtype, device=self.device
        )
        self.merged_patches = wp.empty(
            (self.features, hidden * 4), dtype=self.dtype, device=self.device
        )
        self.merger_hidden = wp.empty_like(self.merged_patches)
        self.tensors["merger.input"], self.shapes["merger.input"] = (
            self.merged_patches,
            self.merged_patches.shape,
        )
        self.merger_fc1 = self._linear(
            "merger.hidden", "merger.input", "merger.fc1.weight"
        )
        self.tensors["merger.hidden"] = self.merger_hidden
        self.output = wp.empty(
            (self.features, encoder.output_size), dtype=self.dtype, device=self.device
        )
        self.tensors["merger.activated"] = self.merger_hidden
        self.shapes["merger.activated"] = self.merger_hidden.shape
        self.merger_fc2 = self._linear(
            "vision.output", "merger.activated", "merger.fc2.weight"
        )
        self.tensors["vision.output"] = self.output
        self.final_hidden = self.tensors[final_hidden]
        self.graph = None
        self._capture_ready = False

    def _linear(self, output: str, x: str, weight: str) -> Operation:
        op = Operation("Linear", [x, weight], [output])
        plan_linear(
            op, self.tensors, self.shapes, self.device, cublas=self.encoder.cublas
        )
        return op

    def _execute_op(self, op: Operation) -> None:
        execute_operations((op,), self.tensors, self.shapes, self.device)

    def execute(self) -> wp.array:
        (
            _,
            _,
            _,
            add_position,
            layer_norm,
            add_bias,
            split_qkv,
            attention,
            merge_heads,
            bias_residual,
            bias_gelu,
            merge_patches,
        ) = self.kernels
        self._execute_op(self.patch)
        wp.launch(
            add_bias,
            dim=self.hidden_slots[0].shape,
            inputs=[self.hidden_slots[0], self.tensors["patch_embed.bias"]],
            device=self.device,
        )
        wp.launch(
            add_position,
            dim=self.hidden_slots[0].shape,
            inputs=[
                self.hidden_slots[0],
                self.tensors["position_embedding.weight"],
                self.positions,
                self.grid_thw[1],
                self.grid_thw[2],
            ],
            device=self.device,
        )
        for prefix, source, target, qkv, projection, fc1, fc2 in self.layers:
            residual = self.tensors[source]
            wp.launch(
                layer_norm,
                dim=self.rows,
                inputs=[
                    residual,
                    self.tensors[prefix + "norm1.weight"],
                    self.tensors[prefix + "norm1.bias"],
                    self.norm,
                    self.encoder.epsilon,
                ],
                device=self.device,
            )
            self._execute_op(qkv)
            wp.launch(
                add_bias,
                dim=self.qkv.shape,
                inputs=[self.qkv, self.tensors[prefix + "attention.qkv.bias"]],
                device=self.device,
            )
            wp.launch(
                split_qkv,
                dim=self.query.shape,
                inputs=[
                    self.qkv,
                    self.positions,
                    self.encoder.inverse_frequency,
                    self.query,
                    self.key,
                    self.value,
                ],
                device=self.device,
            )
            wp.launch_tiled(
                attention,
                dim=self.rows * self.encoder.heads,
                inputs=[
                    self.query,
                    self.key,
                    self.value,
                    self.segments,
                    self.attention,
                    self.encoder.attention_scale,
                ],
                block_dim=128,
                device=self.device,
            )
            wp.launch(
                merge_heads,
                dim=self.query.shape,
                inputs=[self.attention, self.merged_heads],
                device=self.device,
            )
            self._execute_op(projection)
            target_hidden = self.tensors[target]
            wp.launch(
                bias_residual,
                dim=target_hidden.shape,
                inputs=[
                    self.branch,
                    self.tensors[prefix + "attention.proj.bias"],
                    residual,
                    target_hidden,
                ],
                device=self.device,
            )
            wp.launch(
                layer_norm,
                dim=self.rows,
                inputs=[
                    target_hidden,
                    self.tensors[prefix + "norm2.weight"],
                    self.tensors[prefix + "norm2.bias"],
                    self.norm,
                    self.encoder.epsilon,
                ],
                device=self.device,
            )
            self._execute_op(fc1)
            wp.launch(
                bias_gelu,
                dim=self.mlp.shape,
                inputs=[self.mlp, self.tensors[prefix + "mlp.fc1.bias"], True],
                device=self.device,
            )
            self._execute_op(fc2)
            wp.launch(
                bias_residual,
                dim=target_hidden.shape,
                inputs=[
                    self.branch,
                    self.tensors[prefix + "mlp.fc2.bias"],
                    target_hidden,
                    target_hidden,
                ],
                device=self.device,
            )
        wp.launch(
            layer_norm,
            dim=self.rows,
            inputs=[
                self.final_hidden,
                self.tensors["merger.norm.weight"],
                self.tensors["merger.norm.bias"],
                self.merger_norm,
                self.encoder.epsilon,
            ],
            device=self.device,
        )
        wp.launch(
            merge_patches,
            dim=self.merged_patches.shape,
            inputs=[self.merger_norm, self.merged_patches],
            device=self.device,
        )
        self._execute_op(self.merger_fc1)
        wp.launch(
            bias_gelu,
            dim=self.merger_hidden.shape,
            inputs=[self.merger_hidden, self.tensors["merger.fc1.bias"], False],
            device=self.device,
        )
        self._execute_op(self.merger_fc2)
        wp.launch(
            add_bias,
            dim=self.output.shape,
            inputs=[self.output, self.tensors["merger.fc2.bias"]],
            device=self.device,
        )
        return self.output

    def run(self) -> wp.array:
        if not self.device.is_cuda:
            return self.execute()
        if not self._capture_ready:
            self._capture_ready = True
            return self.execute()
        if self.graph is None:
            wp.capture_begin(device=self.device)
            output = self.execute()
            self.graph = wp.capture_end(device=self.device)
            self.output = output
        wp.capture_launch(self.graph)
        return self.output

    def stage(self, media: VisionInput) -> None:
        if media.grid_thw != self.grid_thw or media.patches.shape != self.patches.shape:
            raise ValueError("vision input does not match this fixed-grid plan")
        self.patches.assign(media.patches)
        positions = qwen_vision_positions(self.grid_thw)
        self.positions.assign(positions[:, 1:])
        frame_tokens = self.grid_thw[1] * self.grid_thw[2]
        self.segments.assign(
            np.repeat(np.arange(self.grid_thw[0], dtype=np.int32), frame_tokens)
        )


class QwenVisionEncoder:
    """Lazy dependency-free Qwen3.8 vision encoder for safetensors or GGUF mmproj."""

    def __init__(
        self,
        path: str | Path,
        *,
        device=None,
        cublas=None,
        vision_path: str | Path | None = None,
    ):
        self.path = Path(path)
        directory = self.path if self.path.is_dir() else self.path.parent
        config_path = directory / "config.json"
        config_data = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        vision_config = config_data.get("vision_config", {})
        self.device = wp.get_device(device)
        self.cublas = cublas

        integrated = any(directory.glob("*.safetensors")) and any(
            name.startswith("model.visual.")
            for name in SafeTensorArchive(directory).names
        )
        if vision_path is not None:
            source = Path(vision_path)
            gguf = GGUFArchive(source)
            archive_kind = "gguf"
        elif integrated:
            source = directory
            gguf = None
            archive_kind = "safetensors"
        else:
            candidates = sorted(directory.glob("mmproj*.gguf"))
            if not candidates:
                raise FileNotFoundError(
                    "Qwen vision weights were not found; provide vision_path or place mmproj*.gguf beside the text model"
                )
            source, gguf, archive_kind = (
                candidates[0],
                GGUFArchive(candidates[0]),
                "gguf",
            )

        metadata = {} if gguf is None else gguf.metadata
        self.depth = int(
            vision_config.get("depth", metadata.get("clip.vision.block_count", 27))
        )
        self.hidden_size = int(
            vision_config.get(
                "hidden_size", metadata.get("clip.vision.embedding_length", 1152)
            )
        )
        self.intermediate_size = int(
            vision_config.get(
                "intermediate_size",
                metadata.get("clip.vision.feed_forward_length", 4304),
            )
        )
        self.heads = int(
            vision_config.get(
                "num_heads", metadata.get("clip.vision.attention.head_count", 16)
            )
        )
        self.output_size = int(
            vision_config.get(
                "out_hidden_size", metadata.get("clip.vision.projection_dim", 5120)
            )
        )
        self.head_size = self.hidden_size // self.heads
        self.epsilon = float(vision_config.get("layer_norm_eps", 1.0e-6))
        self.attention_scale = 1.0 / math.sqrt(self.head_size)
        if self.hidden_size % self.heads or self.head_size % 2:
            raise ValueError(
                "Qwen vision hidden size must divide into even attention heads"
            )

        if archive_kind == "safetensors":
            raw_archive = SafeTensorArchive(source)
            mapping = _safetensor_map(self.depth)

            class _MappedSafe:
                names = tuple(mapping)

                def metadata(self, name):
                    return raw_archive.metadata(mapping[name])

                def load(self, device=None, names=None):
                    selected = self.names if names is None else tuple(names)
                    loaded = raw_archive.load(
                        device, [mapping[name] for name in selected]
                    )
                    return {name: loaded[mapping[name]] for name in selected}

            archive = _MappedSafe()
            self.dtype = archive.metadata("patch_embed.weight").dtype
            selected = _vision_weight_names(self.depth)
        else:
            if (
                metadata.get("general.type") != "mmproj"
                or metadata.get("clip.projector_type") != "qwen3vl_merger"
            ):
                raise ValueError(
                    "GGUF vision checkpoint is not a Qwen3-VL merger mmproj"
                )
            archive = MappedGGUFArchive(gguf, _gguf_map(self.depth))
            self.dtype = archive.metadata("patch_embed.weight.0").dtype
            selected = [
                name
                for name in _vision_weight_names(self.depth)
                if name != "patch_embed.weight"
            ]
            selected += ["patch_embed.weight.0", "patch_embed.weight.1"]
        if self.dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Qwen vision projection weights must be FP16 or BF16")
        missing = set(selected) - set(archive.names)
        if missing:
            raise ValueError(f"Qwen vision checkpoint is missing {sorted(missing)[:5]}")
        loaded = archive.load(self.device, selected)
        kernels = _vision_kernels(self.dtype, self.heads, self.head_size, self.dtype)
        cast_2d, cast_1d, pack_patch = kernels[:3]
        weights = {}
        for name, value in loaded.items():
            if value.ndim > 2:
                value = value.reshape((value.shape[0], math.prod(value.shape[1:])))
            if value.dtype == self.dtype:
                weights[name] = value
                continue
            if value.dtype == wp.float32 and (
                value.ndim == 1 or name == "position_embedding.weight"
            ):
                weights[name] = value
                continue
            if value.dtype != wp.float32 or value.ndim not in (1, 2):
                raise TypeError(f"unsupported vision tensor dtype for '{name}'")
            converted = wp.empty(value.shape, dtype=self.dtype, device=self.device)
            wp.launch(
                cast_1d if value.ndim == 1 else cast_2d,
                dim=value.shape,
                inputs=[value, converted],
                device=self.device,
            )
            weights[name] = converted
        if archive_kind == "gguf":
            first, second = (
                weights.pop("patch_embed.weight.0"),
                weights.pop("patch_embed.weight.1"),
            )
            combined = wp.empty(
                (self.hidden_size, 1536), dtype=self.dtype, device=self.device
            )
            wp.launch(
                pack_patch,
                dim=combined.shape,
                inputs=[first, second, combined],
                device=self.device,
            )
            weights["patch_embed.weight"] = combined
        else:
            weights["patch_embed.weight"] = weights["patch_embed.weight"].reshape(
                (self.hidden_size, 1536)
            )
        parameter_dtypes = {
            value.dtype
            for name, value in weights.items()
            if value.ndim == 1 or name == "position_embedding.weight"
        }
        if len(parameter_dtypes) != 1:
            raise TypeError("Qwen vision semantic parameters must share one dtype")
        self.parameter_dtype = parameter_dtypes.pop()
        self.weights = weights
        frequency = np.arange(self.head_size // 4, dtype=np.float32)
        frequency = np.power(
            np.float32(10_000.0), -frequency / np.float32(self.head_size // 4)
        )
        self.inverse_frequency = wp.array(
            frequency, dtype=wp.float32, device=self.device
        )
        self._plans: dict[tuple[int, int, int], _VisionPlan] = {}

    def encode(
        self, media: VisionInput | np.ndarray | Sequence[np.ndarray]
    ) -> wp.array:
        """Encode preprocessed media or raw RGB arrays, returning GPU embeddings."""
        if not isinstance(media, VisionInput):
            media = preprocess_qwen_media(media)
        plan = self._plans.get(media.grid_thw)
        if plan is None:
            plan = self._plans[media.grid_thw] = _VisionPlan(self, media.grid_thw)
        plan.stage(media)
        return plan.run()


@dataclass(frozen=True)
class QwenMultimodalPrompt:
    """Tokenized chat plus GPU-ready media and exact three-axis RoPE positions."""

    token_ids: tuple[int, ...]
    media: tuple[VisionInput, ...]
    feature_starts: tuple[int, ...]
    rope_positions: np.ndarray
    rope_delta: int


class QwenMultimodalProcessor:
    """Transform OpenAI-style multimodal messages without Pillow or Transformers."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        pad = tokenizer.encode("<|image_pad|>")
        video_pad = tokenizer.encode("<|video_pad|>")
        if len(pad) != 1 or len(video_pad) != 1:
            raise ValueError("tokenizer does not define single image/video-pad tokens")
        self.image_pad_id = int(pad[0])
        self.video_pad_id = int(video_pad[0])

    def encode_chat(
        self, messages: Sequence[Mapping[str, object]], **kwargs
    ) -> QwenMultimodalPrompt:
        media = []
        transformed = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, Sequence) or isinstance(content, str):
                transformed.append(dict(message))
                continue
            pieces = []
            for item in content:
                if not isinstance(item, Mapping):
                    raise ValueError("multimodal content entries must be objects")
                kind = item.get("type")
                if kind == "text":
                    pieces.append(str(item.get("text", "")))
                elif kind in ("image", "image_url"):
                    source = item.get("image", item.get("image_url"))
                    if isinstance(source, Mapping):
                        source = source.get("url")
                    if isinstance(source, (str, Path)):
                        from .vision import load_rgb_image

                        source = load_rgb_image(source)
                    vision = preprocess_qwen_media(source)
                    media.append(vision)
                    pieces.append(
                        "<|vision_start|>"
                        + "<|image_pad|>" * vision.feature_count
                        + "<|vision_end|>"
                    )
                elif kind == "video":
                    vision = preprocess_qwen_media(
                        item.get("frames"),
                        minimum_pixels=4096,
                        maximum_pixels=25_165_824,
                    )
                    media.append(vision)
                    pieces.append(
                        "<|vision_start|>"
                        + "<|video_pad|>" * vision.feature_count
                        + "<|vision_end|>"
                    )
                else:
                    raise ValueError(f"unsupported multimodal content type '{kind}'")
            copied = dict(message)
            copied["content"] = "".join(pieces)
            transformed.append(copied)
        token_ids = tuple(
            self.tokenizer.encode(self.tokenizer.format_chat(transformed, **kwargs))
        )
        starts = []
        cursor = 0
        for vision in media:
            while cursor < len(token_ids) and token_ids[cursor] not in (
                self.image_pad_id,
                self.video_pad_id,
            ):
                cursor += 1
            if cursor + vision.feature_count > len(token_ids):
                raise ValueError(
                    "formatted prompt has fewer vision-pad tokens than features"
                )
            starts.append(cursor)
            cursor += vision.feature_count
        positions = np.empty((3, len(token_ids)), dtype=np.int64)
        token_cursor = logical = 0
        for start, vision in zip(starts, media, strict=True):
            count = start - token_cursor
            positions[:, token_cursor:start] = np.arange(
                logical, logical + count, dtype=np.int64
            )
            logical += count
            t, h, w = vision.grid_thw
            merged_h, merged_w = h // 2, w // 2
            local = np.stack(
                [
                    np.repeat(np.arange(t), merged_h * merged_w),
                    np.tile(np.repeat(np.arange(merged_h), merged_w), t),
                    np.tile(np.tile(np.arange(merged_w), merged_h), t),
                ]
            ).astype(np.int64)
            positions[:, start : start + vision.feature_count] = local + logical
            logical += max(t, merged_h, merged_w)
            token_cursor = start + vision.feature_count
        positions[:, token_cursor:] = np.arange(
            logical, logical + len(token_ids) - token_cursor, dtype=np.int64
        )
        logical += len(token_ids) - token_cursor
        return QwenMultimodalPrompt(
            token_ids, tuple(media), tuple(starts), positions, logical - len(token_ids)
        )
