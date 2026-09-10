# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Lean DFlash2 draft execution for Qwen 3.8."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

from warp_nn.runtime.formats.gguf import BlockQuantizedTensor, PackedQuantizedTensor
from warp_nn.runtime.formats.safetensors import SafeTensorArchive
from warp_nn.runtime.kernels import (
    _add_arrays_kernel,
    _concatenate_attention_streams_kernel,
    _gather_rows_kernel,
    _get_gather_q2_k_rows_kernel,
    _get_gather_q8_0_rows_kernel,
    _get_mrope_embedding_kernel,
    _get_top_k_kernels,
    _shift_append_heads_kernel,
    _shift_valid_kernel,
)
from warp_nn.runtime.operators import (
    AttentionHeadsPlan,
    AttentionMergePlan,
    BidirectionalGQAPlan,
    Operation,
    execute_operations,
    plan_linear,
    plan_rms_norm,
    plan_swiglu,
    rotary_cache_values,
)
from warp_nn.runtime.quantization import (
    dequantize_q3_k_weight,
    dequantize_nvfp4_weight,
    dequantize_q8_0_weight,
)


@wp.kernel(enable_backward=False, module="unique")
def _grouped_dynamic_conv_kernel(
    hidden: wp.array2d(dtype=wp.bfloat16),
    dynamic: wp.array2d(dtype=wp.bfloat16),
    base: wp.array3d(dtype=wp.bfloat16),
    output: wp.array2d(dtype=wp.bfloat16),
    phase: int,
    group_size: int,
):
    row, column = wp.tid()
    groups = hidden.shape[1] / group_size
    group = column / group_size
    value = wp.float32(0.0)
    for offset in range(2):
        source = row - offset
        if source >= 0:
            dynamic_column = (phase * 2 + offset) * groups + group
            coefficient = wp.float32(base[phase, offset, column]) + wp.float32(
                dynamic[row, dynamic_column]
            )
            value += coefficient * wp.float32(hidden[source, column])
    output[row, column] = wp.bfloat16(value)


@wp.kernel(enable_backward=False, module="unique")
def _candidate_scores_kernel(
    hidden: wp.array2d(dtype=wp.bfloat16),
    candidates: wp.array2d(dtype=wp.int32),
    unary: wp.array2d(dtype=wp.float32),
    anchor: wp.array2d(dtype=wp.int64),
    path: wp.array1d(dtype=wp.int32),
    predecessor_codebook: wp.array2d(dtype=wp.bfloat16),
    successor_codebook: wp.array2d(dtype=wp.bfloat16),
    scores: wp.array1d(dtype=wp.float32),
    position: int,
):
    candidate = wp.tid()
    predecessor = wp.int32(anchor[0, 0]) if position == 0 else path[position - 1]
    successor = candidates[position, candidate]
    score = unary[position, candidate]
    for rank in range(256):
        score += (
            wp.float32(predecessor_codebook[predecessor, rank])
            * wp.float32(hidden[position, rank])
            * wp.float32(successor_codebook[successor, rank])
        )
    scores[candidate] = score


@wp.kernel(enable_backward=False, module="unique")
def _select_candidate_kernel(
    scores: wp.array1d(dtype=wp.float32),
    candidates: wp.array2d(dtype=wp.int32),
    path: wp.array1d(dtype=wp.int32),
    position: int,
):
    best = wp.int32(0)
    score = scores[0]
    for candidate in range(1, 16):
        if scores[candidate] > score:
            best = candidate
            score = scores[candidate]
    path[position] = candidates[position, best]


class _Plan:
    def __init__(self, draft, rows: int):
        self.draft = draft
        self.rows = rows
        self.device = draft.device
        self.dtype = wp.bfloat16
        self.tensors = dict(draft.weights)
        self.shapes = {name: value.shape for name, value in self.tensors.items()}
        self.operations = {}
        self.graphs = {}
        self._capture_ready = False

    def register(self, name, tensor):
        self.tensors[name] = tensor
        self.shapes[name] = tensor.shape
        return name

    def linear(self, name, source, weight):
        operation = Operation("Linear", [source, weight], [name])
        output_scale = self.draft.linear_output_scales.get(weight)
        if output_scale is not None:
            operation.attrs["_output_scale"] = output_scale
        plan_linear(
            operation,
            self.tensors,
            self.shapes,
            self.device,
            cublas=self.draft.target.cublas,
        )
        self.operations[name] = operation
        return name

    def rms(self, name, source, weight):
        operation = Operation(
            "SimplifiedLayerNormalization",
            [source, weight],
            [name],
            {"epsilon": 1.0e-6},
        )
        plan_rms_norm(operation, self.tensors, self.shapes, self.device)
        self.operations[name] = operation
        return name

    def swiglu(self, name, gate, up):
        operation = Operation("_SwiGLU", [gate, up], [name])
        plan_swiglu(operation, self.tensors, self.shapes, self.device)
        self.operations[name] = operation
        return name

    def execute_op(self, name):
        execute_operations(
            (self.operations[name],), self.tensors, self.shapes, self.device
        )


class _ContextPlan(_Plan):
    def __init__(self, draft, rows: int):
        super().__init__(draft, rows)
        width = draft.hidden_size
        self.input = wp.empty(
            (rows, len(draft.target_layers) * width),
            dtype=self.dtype,
            device=self.device,
        )
        self.position_ids = wp.empty((3, rows), dtype=wp.int64, device=self.device)
        self.register("context.input", self.input)
        self.linear("context.fc", "context.input", "fc.weight")
        self.rms("context.hidden", "context.fc", "hidden_norm.weight")
        self.layers = []
        for index in range(draft.num_layers):
            prefix = f"layers.{index}.self_attn."
            layer = {"index": index}
            self.linear(
                f"context.{index}.k", "context.hidden", prefix + "k_proj.weight"
            )
            self.linear(
                f"context.{index}.v", "context.hidden", prefix + "v_proj.weight"
            )
            layer["k_heads"] = AttentionHeadsPlan(
                self.tensors[f"context.{index}.k"].reshape(
                    (1, rows, draft.kv_heads * draft.head_size)
                ),
                draft.kv_heads,
            )
            layer["v_heads"] = AttentionHeadsPlan(
                self.tensors[f"context.{index}.v"].reshape(
                    (1, rows, draft.kv_heads * draft.head_size)
                ),
                draft.kv_heads,
            )
            self.register(f"context.{index}.k_heads", layer["k_heads"].output)
            self.rms(
                f"context.{index}.k_norm",
                f"context.{index}.k_heads",
                prefix + "k_norm.weight",
            )
            layer["k_rotated"] = wp.empty_like(self.tensors[f"context.{index}.k_norm"])
            self.layers.append(layer)

    def execute(self):
        self.execute_op("context.fc")
        self.execute_op("context.hidden")
        rotary = _get_mrope_embedding_kernel(self.dtype)
        for layer in self.layers:
            index = layer["index"]
            self.execute_op(f"context.{index}.k")
            self.execute_op(f"context.{index}.v")
            layer["k_heads"].execute()
            layer["v_heads"].execute()
            self.execute_op(f"context.{index}.k_norm")
            wp.launch(
                rotary,
                dim=layer["k_rotated"].shape,
                inputs=[
                    self.tensors[f"context.{index}.k_norm"],
                    self.position_ids,
                    self.draft.cos_cache,
                    self.draft.sin_cache,
                    layer["k_rotated"],
                    self.draft.head_size,
                ],
                device=self.device,
            )
            for appended, cache, scratch in (
                (
                    layer["k_rotated"],
                    self.draft.key_caches[index],
                    self.draft.key_scratch[index],
                ),
                (
                    layer["v_heads"].output,
                    self.draft.value_caches[index],
                    self.draft.value_scratch[index],
                ),
            ):
                wp.launch(
                    _shift_append_heads_kernel,
                    dim=scratch.shape,
                    inputs=[cache, appended, scratch],
                    device=self.device,
                )
                wp.copy(cache, scratch)
        wp.launch(
            _shift_valid_kernel,
            dim=self.draft.cache_valid_scratch.shape,
            inputs=[
                self.draft.key_valid,
                self.draft.cache_valid_scratch,
                self.rows,
            ],
            device=self.device,
        )
        wp.copy(
            self.draft.key_valid.flatten(),
            self.draft.cache_valid_scratch.flatten(),
            count=self.draft.cache_capacity,
        )
        return self.input


class _ProposalPlan(_Plan):
    def __init__(self, draft):
        super().__init__(draft, draft.block_size)
        rows, width = self.rows, draft.hidden_size
        self.input_ids = wp.empty((1, rows), dtype=wp.int64, device=self.device)
        self.position_ids = wp.empty((3, rows), dtype=wp.int64, device=self.device)
        self.embedding = wp.empty(
            (1, rows, width), dtype=self.dtype, device=self.device
        )
        self.register("hidden.0", self.embedding.reshape((rows, width)))
        self.hidden = "hidden.0"
        self.layers = []
        for index in range(draft.num_layers):
            self._build_layer(index)
        self.rms("output", self.hidden, "norm.weight")
        self.output = self.tensors["output"].reshape((1, rows, width))
        self.register("candidate.hidden", self.tensors["output"][1:])
        self.linear("candidate.logits", "candidate.hidden", "lm_head.weight")
        self.linear(
            "candidate.projected",
            "candidate.hidden",
            "candidate_selector.hidden_projection.weight",
        )
        self.logits = self.tensors["candidate.logits"].reshape(
            (rows - 1, 1, draft.vocabulary)
        )
        self.top_k = 16
        self.top_k_kernels = _get_top_k_kernels(512, self.top_k, self.dtype)
        partials = (draft.vocabulary + 511) // 512
        merges = (partials + 15) // 16
        self.top_k_states = [
            (
                wp.empty(partials * self.top_k, dtype=wp.float32, device=self.device),
                wp.empty(partials * self.top_k, dtype=wp.int32, device=self.device),
                wp.empty(merges * self.top_k, dtype=wp.float32, device=self.device),
                wp.empty(merges * self.top_k, dtype=wp.int32, device=self.device),
            )
            for _ in range(rows - 1)
        ]
        self.candidate_values = wp.empty(
            (rows - 1, self.top_k), dtype=wp.float32, device=self.device
        )
        self.candidate_tokens = wp.empty(
            (rows - 1, self.top_k), dtype=wp.int32, device=self.device
        )
        self.candidate_scores = wp.empty(
            self.top_k, dtype=wp.float32, device=self.device
        )
        self.path = wp.empty(rows - 1, dtype=wp.int32, device=self.device)

    def _build_layer(self, index):
        prefix = f"layers.{index}."
        layer = {"index": index, "prefix": prefix, "input": self.hidden}
        self.rms(
            f"layer.{index}.attention_norm",
            self.hidden,
            prefix + "input_layernorm.weight",
        )
        self.linear(
            f"layer.{index}.attention_dynamic",
            f"layer.{index}.attention_norm",
            prefix + "attention_conv.kernel_projection.weight",
        )
        layer["attention_input"] = wp.empty(
            (self.rows, self.draft.hidden_size), dtype=self.dtype, device=self.device
        )
        self.register(f"layer.{index}.attention_input", layer["attention_input"])
        for name, weight in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
            self.linear(
                f"layer.{index}.{name}",
                f"layer.{index}.attention_input",
                prefix + f"self_attn.{weight}.weight",
            )
        layer["q_heads"] = AttentionHeadsPlan(
            self.tensors[f"layer.{index}.q"].reshape(
                (1, self.rows, self.draft.query_heads * self.draft.head_size)
            ),
            self.draft.query_heads,
        )
        layer["k_heads"] = AttentionHeadsPlan(
            self.tensors[f"layer.{index}.k"].reshape(
                (1, self.rows, self.draft.kv_heads * self.draft.head_size)
            ),
            self.draft.kv_heads,
        )
        layer["v_heads"] = AttentionHeadsPlan(
            self.tensors[f"layer.{index}.v"].reshape(
                (1, self.rows, self.draft.kv_heads * self.draft.head_size)
            ),
            self.draft.kv_heads,
        )
        self.register(f"layer.{index}.q_heads", layer["q_heads"].output)
        self.register(f"layer.{index}.k_heads", layer["k_heads"].output)
        self.rms(
            f"layer.{index}.q_norm",
            f"layer.{index}.q_heads",
            prefix + "self_attn.q_norm.weight",
        )
        self.rms(
            f"layer.{index}.k_norm",
            f"layer.{index}.k_heads",
            prefix + "self_attn.k_norm.weight",
        )
        layer["q_rotated"] = wp.empty_like(self.tensors[f"layer.{index}.q_norm"])
        layer["k_rotated"] = wp.empty_like(self.tensors[f"layer.{index}.k_norm"])
        key_length = self.draft.cache_capacity + self.rows
        layer["key"] = wp.empty(
            (1, self.draft.kv_heads, key_length, self.draft.head_size),
            dtype=self.dtype,
            device=self.device,
        )
        layer["value"] = wp.empty_like(layer["key"])
        layer["attention"] = BidirectionalGQAPlan(
            layer["q_rotated"],
            layer["key"],
            layer["value"],
            key_valid=self.draft.key_valid,
            window=self.draft.cache_capacity,
        )
        layer["merge"] = AttentionMergePlan(layer["attention"].output)
        self.register(
            f"layer.{index}.merged",
            layer["merge"].output.reshape(
                (self.rows, self.draft.query_heads * self.draft.head_size)
            ),
        )
        self.linear(
            f"layer.{index}.attention_output",
            f"layer.{index}.merged",
            prefix + "self_attn.o_proj.weight",
        )
        layer["attention_finished"] = wp.empty(
            (self.rows, self.draft.hidden_size), dtype=self.dtype, device=self.device
        )
        layer["attention_residual"] = wp.empty_like(layer["attention_finished"])
        self.register(f"layer.{index}.attention_residual", layer["attention_residual"])
        self.rms(
            f"layer.{index}.mlp_norm",
            f"layer.{index}.attention_residual",
            prefix + "post_attention_layernorm.weight",
        )
        self.linear(
            f"layer.{index}.mlp_dynamic",
            f"layer.{index}.mlp_norm",
            prefix + "mlp_conv.kernel_projection.weight",
        )
        layer["mlp_input"] = wp.empty_like(layer["attention_finished"])
        self.register(f"layer.{index}.mlp_input", layer["mlp_input"])
        self.linear(
            f"layer.{index}.gate",
            f"layer.{index}.mlp_input",
            prefix + "mlp.gate_proj.weight",
        )
        self.linear(
            f"layer.{index}.up",
            f"layer.{index}.mlp_input",
            prefix + "mlp.up_proj.weight",
        )
        self.swiglu(f"layer.{index}.swiglu", f"layer.{index}.gate", f"layer.{index}.up")
        self.linear(
            f"layer.{index}.down",
            f"layer.{index}.swiglu",
            prefix + "mlp.down_proj.weight",
        )
        layer["mlp_finished"] = wp.empty_like(layer["attention_finished"])
        layer["output"] = wp.empty_like(layer["attention_finished"])
        self.hidden = self.register(f"layer.{index}.output", layer["output"])
        self.layers.append(layer)

    def _gather_embeddings(self):
        weight = self.draft.target.weights["model.language_model.embed_tokens.weight"]
        if isinstance(weight, BlockQuantizedTensor):
            wp.launch(
                _get_gather_q8_0_rows_kernel(self.dtype),
                dim=self.embedding.shape,
                inputs=[weight.values, self.input_ids, weight.scales, self.embedding],
                device=self.device,
            )
        elif isinstance(weight, PackedQuantizedTensor):
            wp.launch(
                _get_gather_q2_k_rows_kernel(self.dtype),
                dim=self.embedding.shape,
                inputs=[weight.blocks, self.input_ids, self.embedding],
                device=self.device,
            )
        else:
            wp.launch(
                _gather_rows_kernel,
                dim=self.embedding.shape,
                inputs=[weight, self.input_ids, self.embedding],
                device=self.device,
            )

    def execute(self):
        self._gather_embeddings()
        rotary = _get_mrope_embedding_kernel(self.dtype)
        for layer in self.layers:
            index, prefix = layer["index"], layer["prefix"]
            self.execute_op(f"layer.{index}.attention_norm")
            self.execute_op(f"layer.{index}.attention_dynamic")
            wp.launch(
                _grouped_dynamic_conv_kernel,
                dim=layer["attention_input"].shape,
                inputs=[
                    self.tensors[f"layer.{index}.attention_norm"],
                    self.tensors[f"layer.{index}.attention_dynamic"],
                    self.draft.weights[prefix + "attention_conv.base_kernel"],
                    layer["attention_input"],
                    0,
                    self.draft.group_size,
                ],
                device=self.device,
            )
            for name in ("q", "k", "v"):
                self.execute_op(f"layer.{index}.{name}")
            layer["q_heads"].execute()
            layer["k_heads"].execute()
            layer["v_heads"].execute()
            self.execute_op(f"layer.{index}.q_norm")
            self.execute_op(f"layer.{index}.k_norm")
            for source, output in (
                (self.tensors[f"layer.{index}.q_norm"], layer["q_rotated"]),
                (self.tensors[f"layer.{index}.k_norm"], layer["k_rotated"]),
            ):
                wp.launch(
                    rotary,
                    dim=output.shape,
                    inputs=[
                        source,
                        self.position_ids,
                        self.draft.cos_cache,
                        self.draft.sin_cache,
                        output,
                        self.draft.head_size,
                    ],
                    device=self.device,
                )
            for first, second, output in (
                (self.draft.key_caches[index], layer["k_rotated"], layer["key"]),
                (
                    self.draft.value_caches[index],
                    layer["v_heads"].output,
                    layer["value"],
                ),
            ):
                wp.launch(
                    _concatenate_attention_streams_kernel,
                    dim=output.shape,
                    inputs=[first, second, output],
                    device=self.device,
                )
            layer["attention"].execute()
            layer["merge"].execute()
            self.execute_op(f"layer.{index}.attention_output")
            wp.launch(
                _grouped_dynamic_conv_kernel,
                dim=layer["attention_finished"].shape,
                inputs=[
                    self.tensors[f"layer.{index}.attention_output"],
                    self.tensors[f"layer.{index}.attention_dynamic"],
                    self.draft.weights[prefix + "attention_conv.base_kernel"],
                    layer["attention_finished"],
                    1,
                    self.draft.group_size,
                ],
                device=self.device,
            )
            wp.launch(
                _add_arrays_kernel,
                dim=layer["attention_residual"].size,
                inputs=[
                    self.tensors[layer["input"]].flatten(),
                    layer["attention_finished"].flatten(),
                    layer["attention_residual"].flatten(),
                ],
                device=self.device,
            )
            self.execute_op(f"layer.{index}.mlp_norm")
            self.execute_op(f"layer.{index}.mlp_dynamic")
            wp.launch(
                _grouped_dynamic_conv_kernel,
                dim=layer["mlp_input"].shape,
                inputs=[
                    self.tensors[f"layer.{index}.mlp_norm"],
                    self.tensors[f"layer.{index}.mlp_dynamic"],
                    self.draft.weights[prefix + "mlp_conv.base_kernel"],
                    layer["mlp_input"],
                    0,
                    self.draft.group_size,
                ],
                device=self.device,
            )
            for name in ("gate", "up", "swiglu", "down"):
                self.execute_op(f"layer.{index}.{name}")
            wp.launch(
                _grouped_dynamic_conv_kernel,
                dim=layer["mlp_finished"].shape,
                inputs=[
                    self.tensors[f"layer.{index}.down"],
                    self.tensors[f"layer.{index}.mlp_dynamic"],
                    self.draft.weights[prefix + "mlp_conv.base_kernel"],
                    layer["mlp_finished"],
                    1,
                    self.draft.group_size,
                ],
                device=self.device,
            )
            wp.launch(
                _add_arrays_kernel,
                dim=layer["output"].size,
                inputs=[
                    layer["attention_residual"].flatten(),
                    layer["mlp_finished"].flatten(),
                    layer["output"].flatten(),
                ],
                device=self.device,
            )
        self.execute_op("output")
        self.execute_op("candidate.logits")
        self.execute_op("candidate.projected")
        for row, state in enumerate(self.top_k_states):
            values, tokens, merge_values, merge_tokens = state
            partials = values.shape[0] // self.top_k
            wp.launch_tiled(
                self.top_k_kernels[0],
                dim=partials,
                inputs=[self.logits[row : row + 1], values, tokens],
                block_dim=256,
                device=self.device,
            )
            source_values, target_values = values, merge_values
            source_tokens, target_tokens = tokens, merge_tokens
            input_groups = partials
            while input_groups > 1:
                output_groups = (input_groups + 15) // 16
                wp.launch_tiled(
                    self.top_k_kernels[1],
                    dim=output_groups,
                    inputs=[
                        source_values,
                        source_tokens,
                        target_values,
                        target_tokens,
                        input_groups,
                    ],
                    block_dim=256,
                    device=self.device,
                )
                source_values, target_values = target_values, source_values
                source_tokens, target_tokens = target_tokens, source_tokens
                input_groups = output_groups
            wp.copy(
                self.candidate_values.flatten(),
                source_values,
                dest_offset=row * self.top_k,
                count=self.top_k,
            )
            wp.copy(
                self.candidate_tokens.flatten(),
                source_tokens,
                dest_offset=row * self.top_k,
                count=self.top_k,
            )
        projected = self.tensors["candidate.projected"]
        for position in range(self.rows - 1):
            wp.launch(
                _candidate_scores_kernel,
                dim=self.top_k,
                inputs=[
                    projected,
                    self.candidate_tokens,
                    self.candidate_values,
                    self.input_ids,
                    self.path,
                    self.draft.weights["candidate_selector.predecessor_codebook"],
                    self.draft.weights["candidate_selector.successor_codebook"],
                    self.candidate_scores,
                    position,
                ],
                device=self.device,
            )
            wp.launch(
                _select_candidate_kernel,
                dim=1,
                inputs=[
                    self.candidate_scores,
                    self.candidate_tokens,
                    self.path,
                    position,
                ],
                device=self.device,
            )
        return self.path


class DFlash2Draft:
    """DFlash2 backbone sharing Qwen embeddings, LM head, and CUDA stream."""

    def __init__(self, target, path: str | Path):
        self.target = target
        self.device = target.device
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        draft = config["dflash_config"]
        self.target_layers = tuple(int(index) for index in draft["target_layer_ids"])
        self.hidden_size = int(config["hidden_size"])
        self.query_heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.head_size = int(config["head_dim"])
        self.num_layers = int(config["num_hidden_layers"])
        self.block_size = int(draft["block_size"])
        self.mask_token = int(draft["mask_token_id"])
        self.group_size = int(draft["conv_group_size"])
        self.vocabulary = int(config["vocab_size"])
        self.cache_capacity = int(config["sliding_window"]) - 1
        expected = (5120, 32, 8, 128, 5, 8, 16)
        actual = (
            self.hidden_size,
            self.query_heads,
            self.kv_heads,
            self.head_size,
            self.num_layers,
            self.block_size,
            self.group_size,
        )
        if actual != expected or self.target_layers != target.dflash_target_layers:
            raise ValueError("Unsupported DFlash2 checkpoint geometry")
        checkpoint = (path / "model.safetensors").resolve()
        self.weights = SafeTensorArchive(checkpoint).load(self.device)
        self.linear_output_scales = dict(target.linear_output_scales)
        lm_head = target.weights["lm_head.weight"]
        if isinstance(lm_head, BlockQuantizedTensor):
            if lm_head.format == "Q8_0":
                lm_head = dequantize_q8_0_weight(lm_head)
            elif lm_head.format in ("NVFP4", "NVFP4_MMA"):
                lm_head = dequantize_nvfp4_weight(
                    lm_head,
                    wp.bfloat16,
                    self.linear_output_scales.pop("lm_head.weight", 1.0),
                )
        elif isinstance(lm_head, PackedQuantizedTensor) and lm_head.format == "Q3_K":
            lm_head = dequantize_q3_k_weight(lm_head)
        self.weights.update(
            {
                "model.language_model.embed_tokens.weight": target.weights[
                    "model.language_model.embed_tokens.weight"
                ],
                "lm_head.weight": lm_head,
            }
        )
        cache_shape = (1, self.kv_heads, self.cache_capacity, self.head_size)
        self.key_caches = [
            wp.zeros(cache_shape, dtype=wp.bfloat16, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.value_caches = [wp.zeros_like(cache) for cache in self.key_caches]
        self.key_scratch = [wp.empty_like(cache) for cache in self.key_caches]
        self.value_scratch = [wp.empty_like(cache) for cache in self.key_caches]
        valid = np.zeros((1, self.cache_capacity + self.block_size), dtype=np.bool_)
        valid[:, self.cache_capacity :] = True
        self.key_valid = wp.array(valid, device=self.device)
        self.cache_valid_scratch = wp.empty(
            (1, self.cache_capacity), dtype=wp.bool, device=self.device
        )
        cosine, sine = rotary_cache_values(
            target.cache_capacity + self.block_size,
            self.head_size,
            {"rope_theta": float(config["rope_parameters"]["rope_theta"])},
        )
        self.cos_cache = wp.array(cosine, dtype=wp.bfloat16, device=self.device)
        self.sin_cache = wp.array(sine, dtype=wp.bfloat16, device=self.device)
        self.context_plans = {}
        self.proposal_plan = _ProposalPlan(self)
        target._record_plan_storage(self.proposal_plan)
        self.sequence_length = 0

    def reset(self):
        for cache in (*self.key_caches, *self.value_caches):
            cache.zero_()
        valid = np.zeros((1, self.cache_capacity + self.block_size), dtype=np.bool_)
        valid[:, self.cache_capacity :] = True
        self.key_valid.assign(valid)
        self.sequence_length = 0

    def append_context(self, hidden, start: int):
        if start != self.sequence_length:
            raise RuntimeError("DFlash and target sequence states are out of sync")
        rows = hidden.shape[0]
        plan = self.context_plans.get(rows)
        if plan is None:
            plan = self.context_plans[rows] = _ContextPlan(self, rows)
            self.target._record_plan_storage(plan)
        wp.copy(plan.input.flatten(), hidden.flatten())
        positions = np.arange(start, start + rows, dtype=np.int64)
        plan.position_ids.assign(np.broadcast_to(positions, (3, rows)))
        self.target._run(plan)
        self.sequence_length += rows

    def forward(self, anchor: int):
        plan = self.proposal_plan
        plan.input_ids.assign(
            np.asarray(
                [[anchor] + [self.mask_token] * (self.block_size - 1)], dtype=np.int64
            )
        )
        positions = np.arange(
            self.sequence_length,
            self.sequence_length + self.block_size,
            dtype=np.int64,
        )
        plan.position_ids.assign(np.broadcast_to(positions, (3, self.block_size)))
        return self.target._run(plan)

    def propose(self, anchor: int) -> list[int]:
        path = self.forward(anchor)
        return [int(token) for token in path.numpy()]
