# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Official BF16 DFlash assistant for Muse Glimmer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

from warp_nn.runtime.formats.gguf import BlockQuantizedTensor
from warp_nn.runtime.formats.safetensors import SafeTensorArchive
from warp_nn.runtime.kernels import (
    _add_arrays_kernel,
    _concatenate_attention_streams_kernel,
    _gather_rows_kernel,
    _get_gather_q8_0_rows_kernel,
    _rotary_embedding_kernel_for_dtype,
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
        plan_linear(
            operation,
            self.tensors,
            self.shapes,
            self.device,
            cublas=self.draft.target.cublas,
        )
        operation.attrs["_sequence"] = (operation,)
        self.operations[name] = operation
        return name

    def rms(self, name, source, weight):
        operation = Operation(
            "SimplifiedLayerNormalization",
            [source, weight],
            [name],
            {"epsilon": self.draft.epsilon},
        )
        plan_rms_norm(operation, self.tensors, self.shapes, self.device)
        operation.attrs["_sequence"] = (operation,)
        self.operations[name] = operation
        return name

    def swiglu(self, name, gate, up):
        operation = Operation("_SwiGLU", [gate, up], [name])
        plan_swiglu(operation, self.tensors, self.shapes, self.device)
        operation.attrs["_sequence"] = (operation,)
        self.operations[name] = operation
        return name

    def execute_op(self, name):
        operation = self.operations[name]
        execute_operations(
            operation.attrs["_sequence"], self.tensors, self.shapes, self.device
        )

    def rotate(self, source, positions, output, heads):
        wp.launch(
            _rotary_embedding_kernel_for_dtype(self.dtype),
            dim=(1, heads, source.shape[2], self.draft.head_size),
            inputs=[
                source,
                positions,
                self.draft.cos_cache,
                self.draft.sin_cache,
                output,
                self.draft.head_size,
                False,
                False,
            ],
            device=self.device,
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
        self.position_ids = wp.empty((1, rows), dtype=wp.int64, device=self.device)
        self.register("context.input", self.input)
        self.linear("context.fc", "context.input", "encoder.fc.weight")
        self.rms("context.hidden", "context.fc", "encoder.output_norm_enc.weight")
        self.layers = []
        for index in range(draft.num_layers):
            prefix = f"layers.{index}.self_attn."
            layer = {}
            for name in ("k", "v"):
                self.linear(
                    f"context.{index}.{name}",
                    "context.hidden",
                    prefix + f"{name}_proj.weight",
                )
                layer[name] = AttentionHeadsPlan(
                    self.tensors[f"context.{index}.{name}"].reshape(
                        (1, rows, draft.kv_heads * draft.head_size)
                    ),
                    draft.kv_heads,
                )
            self.register(f"context.{index}.k_heads", layer["k"].output)
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
        for index, layer in enumerate(self.layers):
            self.execute_op(f"context.{index}.k")
            self.execute_op(f"context.{index}.v")
            layer["k"].execute()
            layer["v"].execute()
            self.execute_op(f"context.{index}.k_norm")
            normalized = self.tensors[f"context.{index}.k_norm"]
            self.rotate(
                normalized, self.position_ids, layer["k_rotated"], self.draft.kv_heads
            )
            for appended, cache, scratch in (
                (
                    layer["k_rotated"],
                    self.draft.key_caches[index],
                    self.draft.key_scratch[index],
                ),
                (
                    layer["v"].output,
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
            inputs=[self.draft.key_valid, self.draft.cache_valid_scratch, self.rows],
            device=self.device,
        )
        wp.copy(self.draft.key_valid, self.draft.cache_valid_scratch)
        return self.input


class _ProposalPlan(_Plan):
    def __init__(self, draft):
        super().__init__(draft, draft.block_size)
        rows, width = self.rows, draft.hidden_size
        self.input_ids = wp.empty((1, rows), dtype=wp.int64, device=self.device)
        self.position_ids = wp.empty((1, rows), dtype=wp.int64, device=self.device)
        self.embedding = wp.empty(
            (1, rows, width), dtype=self.dtype, device=self.device
        )
        self.hidden = self.register("hidden.0", self.embedding.reshape((rows, width)))
        self.layers = []
        for index in range(draft.num_layers):
            self._build_layer(index)
        self.rms("output", self.hidden, "norm.weight")
        self.linear("logits", "output", "lm_head.weight")
        self.logits = self.tensors["logits"].reshape((rows, 1, draft.vocabulary))[1:]

    def _build_layer(self, index):
        prefix = f"layers.{index}."
        layer = {"input": self.hidden}
        self.rms(
            f"layer.{index}.attention_norm",
            self.hidden,
            prefix + "input_layernorm.weight",
        )
        for name in ("q", "k", "v"):
            self.linear(
                f"layer.{index}.{name}",
                f"layer.{index}.attention_norm",
                prefix + f"self_attn.{name}_proj.weight",
            )
        layer["q"] = AttentionHeadsPlan(
            self.tensors[f"layer.{index}.q"].reshape(
                (1, self.rows, self.draft.query_heads * self.draft.head_size)
            ),
            self.draft.query_heads,
        )
        for name in ("k", "v"):
            layer[name] = AttentionHeadsPlan(
                self.tensors[f"layer.{index}.{name}"].reshape(
                    (1, self.rows, self.draft.kv_heads * self.draft.head_size)
                ),
                self.draft.kv_heads,
            )
        self.register(f"layer.{index}.q_heads", layer["q"].output)
        self.register(f"layer.{index}.k_heads", layer["k"].output)
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
        layer["attention_residual"] = wp.empty(
            (self.rows, self.draft.hidden_size), dtype=self.dtype, device=self.device
        )
        self.register(f"layer.{index}.attention_residual", layer["attention_residual"])
        self.rms(
            f"layer.{index}.mlp_norm",
            f"layer.{index}.attention_residual",
            prefix + "post_attention_layernorm.weight",
        )
        self.linear(
            f"layer.{index}.gate",
            f"layer.{index}.mlp_norm",
            prefix + "mlp.gate_proj.weight",
        )
        self.linear(
            f"layer.{index}.up",
            f"layer.{index}.mlp_norm",
            prefix + "mlp.up_proj.weight",
        )
        self.swiglu(f"layer.{index}.swiglu", f"layer.{index}.gate", f"layer.{index}.up")
        self.linear(
            f"layer.{index}.down",
            f"layer.{index}.swiglu",
            prefix + "mlp.down_proj.weight",
        )
        layer["output"] = wp.empty_like(layer["attention_residual"])
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
        else:
            wp.launch(
                _gather_rows_kernel,
                dim=self.embedding.shape,
                inputs=[weight, self.input_ids, self.embedding],
                device=self.device,
            )

    def execute(self):
        self._gather_embeddings()
        for index, layer in enumerate(self.layers):
            self.execute_op(f"layer.{index}.attention_norm")
            for name in ("q", "k", "v"):
                self.execute_op(f"layer.{index}.{name}")
                layer[name].execute()
            self.execute_op(f"layer.{index}.q_norm")
            self.execute_op(f"layer.{index}.k_norm")
            self.rotate(
                self.tensors[f"layer.{index}.q_norm"],
                self.position_ids,
                layer["q_rotated"],
                self.draft.query_heads,
            )
            self.rotate(
                self.tensors[f"layer.{index}.k_norm"],
                self.position_ids,
                layer["k_rotated"],
                self.draft.kv_heads,
            )
            for first, second, output in (
                (self.draft.key_caches[index], layer["k_rotated"], layer["key"]),
                (self.draft.value_caches[index], layer["v"].output, layer["value"]),
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
                _add_arrays_kernel,
                dim=layer["attention_residual"].size,
                inputs=[
                    self.tensors[layer["input"]].flatten(),
                    self.tensors[f"layer.{index}.attention_output"].flatten(),
                    layer["attention_residual"].flatten(),
                ],
                device=self.device,
            )
            for name in ("mlp_norm", "gate", "up", "swiglu", "down"):
                self.execute_op(f"layer.{index}.{name}")
            wp.launch(
                _add_arrays_kernel,
                dim=layer["output"].size,
                inputs=[
                    layer["attention_residual"].flatten(),
                    self.tensors[f"layer.{index}.down"].flatten(),
                    layer["output"].flatten(),
                ],
                device=self.device,
            )
        self.execute_op("output")
        self.execute_op("logits")
        return self.logits


class MuseDFlashDraft:
    """Muse Glimmer's official block-diffusion assistant."""

    def __init__(self, target, path: str | Path):
        self.target = target
        self.device = target.device
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        self.target_layers = tuple(int(index) for index in config["target_layer_ids"])
        self.hidden_size = int(config["hidden_size"])
        self.query_heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.head_size = int(config["head_dim"])
        self.num_layers = int(config["num_hidden_layers"])
        self.block_size = int(config["block_size"])
        self.mask_token = int(config["mask_token_id"])
        self.vocabulary = target.vocab_size
        self.epsilon = float(config["rms_norm_eps"])
        self.cache_capacity = int(config["sliding_window"]) - 1
        actual = (
            self.hidden_size,
            self.query_heads,
            self.kv_heads,
            self.head_size,
            self.num_layers,
            self.block_size,
        )
        if (
            actual != (6656, 32, 8, 128, 5, 16)
            or self.target_layers != target.dflash_target_layers
            or self.hidden_size != target.hidden_size
        ):
            raise ValueError("Unsupported Muse DFlash checkpoint geometry")
        checkpoint = (path / "model.safetensors").resolve()
        self.weights = SafeTensorArchive(checkpoint).load(self.device)
        self.weights["lm_head.weight"] = target.weights["lm_head.weight"]
        cache_shape = (1, self.kv_heads, self.cache_capacity, self.head_size)
        self.key_caches = [
            wp.zeros(cache_shape, dtype=self.dtype, device=self.device)
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
        self.cos_cache = wp.array(cosine, dtype=self.dtype, device=self.device)
        self.sin_cache = wp.array(sine, dtype=self.dtype, device=self.device)
        self.context_plans = {}
        self.proposal_plan = _ProposalPlan(self)
        target._record_plan_storage(self.proposal_plan)
        self.sequence_length = 0

    @property
    def dtype(self):
        return wp.bfloat16

    def reset(self):
        for cache in (*self.key_caches, *self.value_caches):
            cache.zero_()
        valid = np.zeros((1, self.cache_capacity + self.block_size), dtype=np.bool_)
        valid[:, self.cache_capacity :] = True
        self.key_valid.assign(valid)
        self.sequence_length = 0

    def append_context(self, hidden, start: int):
        if start != self.sequence_length:
            raise RuntimeError("Muse DFlash and target states are out of sync")
        rows = hidden.shape[0]
        plan = self.context_plans.get(rows)
        if plan is None:
            plan = self.context_plans[rows] = _ContextPlan(self, rows)
            self.target._record_plan_storage(plan)
        wp.copy(plan.input.flatten(), hidden.flatten())
        plan.position_ids.assign(
            np.arange(start, start + rows, dtype=np.int64)[None, :]
        )
        self.target._run(plan)
        self.sequence_length += rows

    def propose(self, anchor: int) -> list[int]:
        plan = self.proposal_plan
        plan.input_ids.assign(
            np.asarray(
                [[anchor] + [self.mask_token] * (self.block_size - 1)],
                dtype=np.int64,
            )
        )
        plan.position_ids.assign(
            np.arange(
                self.sequence_length,
                self.sequence_length + self.block_size,
                dtype=np.int64,
            )[None, :]
        )
        logits = self.target._run(plan)
        return [int(token) for token in self.target.sample_greedy_rows(logits)]
