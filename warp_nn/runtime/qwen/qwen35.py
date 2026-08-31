# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native text-only runner for Qwen 3.5-family Hugging Face checkpoints."""

from __future__ import annotations

import json
import weakref
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import warp as wp

from warp_nn.runtime._cublas import try_create_cublas
from warp_nn.runtime.batch_decode import NativeBatchDecoder
from warp_nn.runtime.formats.gguf import (
    BlockQuantizedTensor,
    GGUFArchive,
    find_gguf_files,
)
from warp_nn.runtime.kernels import (
    _append_head_cache_kernel,
    _get_append_head_cache_decode_batch_kernel,
    _causal_conv_rows_kernel,
    _get_causal_conv_decode_batch_kernels,
    _gather_rows_kernel,
    _cast_kernel_for_dtypes,
    _get_gated_rms_norm_kernel,
    _get_gated_delta_decode_batch_kernel,
    _get_gather_q8_0_rows_kernel,
    _get_mrope_embedding_kernel,
    _get_mrope_decode_batch_kernel,
    _get_gqa_attention_kernel,
    _get_greedy_argmax_kernels,
    _get_linear_attention_kernel,
    _get_lp_normalization_kernel,
    _linear_attention_value_blocks,
    _prepare_gated_delta_kernel,
    _reorder_heads_kernel,
    _reorder_heads_decode_batch_kernel,
    _reorder_interleaved_heads_kernel,
    _overlay_embedding_rows_kernel,
    _set_sequence_end,
    _sigmoid_gate_kernel,
    _split_last_axis_kernel,
    _unpack_gated_heads_kernel,
    _unpack_gated_heads_decode_batch_kernel,
    _update_conv_rows_state_kernel,
)
from warp_nn.runtime.operators import (
    Operation,
    _allocate_partitioned_gqa,
    _decode_attention_partitions,
    _launch_partitioned_gqa,
    execute_operations,
    plan_linear,
    plan_residual_rms_norm,
    plan_rms_norm,
    plan_swiglu,
    reuse_linear_outputs,
)
from warp_nn.runtime.weights import MappedWeightArchive
from warp_nn.runtime.quantization import (
    estimate_loaded_weight_bytes,
    load_native_weights,
    normalize_weight_quantization,
)
from warp_nn.runtime.operators import resolve_rope_parameters, rotary_cache_values
from warp_nn.runtime.formats.safetensors import SafeTensorArchive
from warp_nn.utils.device import parse_device
from warp_nn.runtime.autoregressive import AutoregressiveRunner


def _weight_names(config: dict) -> list[str]:
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    for index, layer_type in enumerate(config["layer_types"]):
        prefix = f"model.language_model.layers.{index}."
        names.extend(
            prefix + suffix
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        )
        if layer_type == "linear_attention":
            names.extend(
                prefix + "linear_attn." + suffix
                for suffix in (
                    "in_proj_qkv.weight",
                    "in_proj_z.weight",
                    "in_proj_a.weight",
                    "in_proj_b.weight",
                    "conv1d.weight",
                    "A_log",
                    "dt_bias",
                    "norm.weight",
                    "out_proj.weight",
                )
            )
        elif layer_type == "full_attention":
            names.extend(
                prefix + "self_attn." + suffix
                for suffix in (
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "q_norm.weight",
                    "k_norm.weight",
                    "o_proj.weight",
                )
            )
        else:
            raise ValueError(f"Unsupported Qwen 3.5 layer type '{layer_type}'")
    return names


def _gguf_weight_map(config: dict) -> dict[str, str]:
    names = {
        "model.language_model.embed_tokens.weight": "token_embd.weight",
        "model.language_model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    common = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    linear = {
        "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
        "linear_attn.in_proj_z.weight": "attn_gate.weight",
        "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
        "linear_attn.in_proj_b.weight": "ssm_beta.weight",
        "linear_attn.conv1d.weight": "ssm_conv1d.weight",
        "linear_attn.A_log": "ssm_a",
        "linear_attn.dt_bias": "ssm_dt.bias",
        "linear_attn.norm.weight": "ssm_norm.weight",
        "linear_attn.out_proj.weight": "ssm_out.weight",
    }
    attention = {
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
    }
    for index, layer_type in enumerate(config["layer_types"]):
        prefix = f"model.language_model.layers.{index}."
        suffixes = common | (linear if layer_type == "linear_attention" else attention)
        names.update(
            {
                prefix + target: f"blk.{index}.{source}"
                for target, source in suffixes.items()
            }
        )
    return names


def _gguf_config(metadata: dict) -> dict:
    """Build the text runtime configuration from self-contained Qwen GGUF metadata."""
    layers = int(metadata["qwen35.block_count"]) - int(
        metadata.get("qwen35.nextn_predict_layers", 0)
    )
    interval = int(metadata["qwen35.full_attention_interval"])
    head_dim = int(metadata["qwen35.attention.key_length"])
    linear_dim = int(metadata["qwen35.ssm.state_size"])
    linear_value_heads = int(metadata["qwen35.ssm.inner_size"]) // linear_dim
    return {
        "attention_bias": False,
        "attn_output_gate": True,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": int(metadata["qwen35.embedding_length"]),
        "intermediate_size": int(metadata["qwen35.feed_forward_length"]),
        "layer_types": [
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(layers)
        ],
        "linear_conv_kernel_dim": int(metadata["qwen35.ssm.conv_kernel"]),
        "linear_key_head_dim": linear_dim,
        "linear_num_key_heads": int(metadata["qwen35.ssm.group_count"]),
        "linear_num_value_heads": linear_value_heads,
        "linear_value_head_dim": linear_dim,
        "max_position_embeddings": int(metadata["qwen35.context_length"]),
        "num_attention_heads": int(metadata["qwen35.attention.head_count"]),
        "num_hidden_layers": layers,
        "num_key_value_heads": int(metadata["qwen35.attention.head_count_kv"]),
        "output_gate_type": "swish",
        "rms_norm_eps": float(metadata["qwen35.attention.layer_norm_rms_epsilon"]),
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": list(metadata["qwen35.rope.dimension_sections"][:3]),
            "partial_rotary_factor": float(metadata["qwen35.rope.dimension_count"])
            / head_dim,
            "rope_theta": float(metadata["qwen35.rope.freq_base"]),
            "rope_type": "default",
        },
        "vocab_size": len(metadata["tokenizer.ggml.tokens"]),
    }


def _validate_config(config: dict) -> None:
    required = (
        "hidden_size",
        "intermediate_size",
        "vocab_size",
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
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_parameters",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Qwen 3.5 config is missing {missing}")
    if len(config["layer_types"]) != int(config["num_hidden_layers"]):
        raise ValueError("Qwen 3.5 layer_types does not match num_hidden_layers")
    if int(config["num_attention_heads"]) % int(config["num_key_value_heads"]):
        raise ValueError("Qwen 3.5 query heads must be divisible by KV heads")
    if (
        config.get("attention_bias", False)
        or config.get("hidden_act", "silu") != "silu"
    ):
        raise ValueError("Only bias-free SiLU Qwen 3.5 text models are supported")
    if (
        not config.get("attn_output_gate", True)
        or config.get("output_gate_type", "swish") != "swish"
    ):
        raise ValueError("Only swish-gated Qwen attention output is supported")
    if config["rope_parameters"].get("rope_type", "default") != "default":
        raise ValueError("Only default Qwen rotary embeddings are supported")


class _Qwen35Plan:
    """Fixed-row execution plan sharing weights and recurrent state."""

    def __init__(
        self,
        runner: Qwen35Runner,
        rows: int,
        external_embeddings: bool = False,
        decode_batch: bool = False,
    ):
        self.runner = weakref.proxy(runner)
        self.rows = rows
        self.decode_batch = bool(decode_batch)
        self.mapped_state = bool(getattr(runner, "mapped_state", False))
        if self.decode_batch and (external_embeddings or rows not in (2, 4, 8)):
            raise ValueError("Qwen decode batches require 2, 4, or 8 text-only slots")
        self.device = runner.device
        self.dtype = runner.dtype
        self.config = runner.config
        self.tensors = dict(runner.weights)
        self.shapes = {name: tuple(value.shape) for name, value in self.tensors.items()}
        self.input_ids = wp.zeros((1, rows), dtype=wp.int64, device=self.device)
        self.position_ids = wp.zeros((1, rows), dtype=wp.int64, device=self.device)
        self.rope_position_ids = wp.zeros((3, rows), dtype=wp.int64, device=self.device)
        self.external_embeddings = external_embeddings
        if external_embeddings:
            self.visual_embeddings = wp.empty(
                (rows, runner.hidden_size), dtype=self.dtype, device=self.device
            )
            self.visual_indices = wp.empty(rows, dtype=wp.int32, device=self.device)
        self.embedding = wp.empty(
            (1, rows, runner.hidden_size), dtype=self.dtype, device=self.device
        )
        self.tensors["hidden.0"] = self.embedding.reshape((rows, runner.hidden_size))
        self.shapes["hidden.0"] = (rows, runner.hidden_size)
        self.layers = []
        self._layer_buffer_pool = {}
        self._build()
        self.graphs = {}

    def _linear(
        self, name: str, x: str, weight: str, q8_activation_cache=None
    ) -> Operation:
        op = Operation("Linear", [x, weight], [name])
        if self.decode_batch:
            op.attrs.update(_small_batch_decode=True, _small_batch_outputs_per_group=4)
        plan_linear(
            op,
            self.tensors,
            self.shapes,
            self.device,
            cublas=self.runner.cublas,
            q8_activation_cache=q8_activation_cache,
        )
        op.attrs["_sequence"] = (op,)
        return op

    def _rms(self, name: str, x: str, scale: str) -> Operation:
        op = Operation(
            "SimplifiedLayerNormalization",
            [x, scale],
            [name],
            {
                "epsilon": self.runner.epsilon,
                "_scale_offset": float(self.runner.centered_norm_scales),
            },
        )
        plan_rms_norm(op, self.tensors, self.shapes, self.device)
        op.attrs["_sequence"] = (op,)
        return op

    def _residual_rms(
        self, name: str, x: str, residual: str, scale: str, residual_name: str
    ) -> Operation:
        op = Operation(
            "SkipSimplifiedLayerNormalization",
            [x, residual, scale],
            [name, "", "", residual_name],
            {
                "epsilon": self.runner.epsilon,
                "_scale_offset": float(self.runner.centered_norm_scales),
            },
        )
        plan_residual_rms_norm(op, self.tensors, self.shapes, self.device)
        op.attrs["_sequence"] = (op,)
        return op

    def _swiglu(self, name: str, gate: str, up: str) -> Operation:
        op = Operation("_SwiGLU", [gate, up], [name])
        plan_swiglu(op, self.tensors, self.shapes, self.device)
        op.attrs["_sequence"] = (op,)
        return op

    def _build(self) -> None:
        hidden_name = "hidden.0"
        first_scale = "model.language_model.layers.0.input_layernorm.weight"
        normalized_name = "layer.0.input"
        self.first_norm = self._rms(normalized_name, hidden_name, first_scale)

        for index, layer_type in enumerate(self.config["layer_types"]):
            prefix = f"model.language_model.layers.{index}."
            layer = {"type": layer_type}
            if layer_type == "linear_attention":
                self._build_linear_attention(layer, index, prefix, normalized_name)
            else:
                self._build_full_attention(layer, index, prefix, normalized_name)

            token_output = layer["output"].outputs[0]
            mlp_input = f"layer.{index}.mlp_input"
            residual = f"layer.{index}.attention_residual"
            layer["post_norm"] = self._residual_rms(
                mlp_input,
                token_output,
                hidden_name,
                prefix + "post_attention_layernorm.weight",
                residual,
            )
            mlp_q8_cache = {}
            layer["mlp_gate"] = self._linear(
                f"layer.{index}.mlp_gate",
                mlp_input,
                prefix + "mlp.gate_proj.weight",
                mlp_q8_cache,
            )
            layer["mlp_up"] = self._linear(
                f"layer.{index}.mlp_up",
                mlp_input,
                prefix + "mlp.up_proj.weight",
                mlp_q8_cache,
            )
            layer["swiglu"] = self._swiglu(
                f"layer.{index}.mlp_hidden",
                layer["mlp_gate"].outputs[0],
                layer["mlp_up"].outputs[0],
            )
            layer["down"] = self._linear(
                f"layer.{index}.mlp_output",
                layer["swiglu"].outputs[0],
                prefix + "mlp.down_proj.weight",
            )
            if index + 1 < len(self.config["layer_types"]):
                next_scale = (
                    f"model.language_model.layers.{index + 1}.input_layernorm.weight"
                )
                normalized_name = f"layer.{index + 1}.input"
            else:
                next_scale = "model.language_model.norm.weight"
                normalized_name = "final.normalized"
            hidden_name = f"hidden.{index + 1}"
            layer["next_norm"] = self._residual_rms(
                normalized_name,
                layer["down"].outputs[0],
                residual,
                next_scale,
                hidden_name,
            )
            reuse_linear_outputs(layer, self.tensors, self._layer_buffer_pool)
            self.layers.append(layer)

        last_normalized = "final.last_normalized"
        if self.decode_batch:
            self.tensors[last_normalized] = self.tensors[normalized_name]
            self.shapes[last_normalized] = (self.rows, self.runner.hidden_size)
        else:
            self.tensors[last_normalized] = self.tensors[normalized_name][
                self.rows - 1 : self.rows
            ]
            self.shapes[last_normalized] = (1, self.runner.hidden_size)
        self.lm_head = self._linear("logits", last_normalized, "lm_head.weight")
        output_rows = self.rows if self.decode_batch else 1
        self.logits = self.tensors["logits"].reshape(
            (output_rows, 1, self.config["vocab_size"])
        )

    def _build_linear_attention(
        self, layer: dict, index: int, prefix: str, x: str
    ) -> None:
        attn = prefix + "linear_attn."
        projection_q8_cache = {}
        layer["qkv"] = self._linear(
            f"layer.{index}.qkv",
            x,
            attn + "in_proj_qkv.weight",
            projection_q8_cache,
        )
        layer["z"] = self._linear(
            f"layer.{index}.z", x, attn + "in_proj_z.weight", projection_q8_cache
        )
        layer["a"] = self._linear(
            f"layer.{index}.a", x, attn + "in_proj_a.weight", projection_q8_cache
        )
        layer["b"] = self._linear(
            f"layer.{index}.b", x, attn + "in_proj_b.weight", projection_q8_cache
        )
        conv_dim = (
            self.runner.linear_key_heads * self.runner.linear_key_size * 2
            + self.runner.linear_value_heads * self.runner.linear_value_size
        )
        layer["conv"] = wp.empty(
            (self.rows, conv_dim), dtype=self.dtype, device=self.device
        )
        widths = (
            self.runner.linear_key_heads * self.runner.linear_key_size,
            self.runner.linear_key_heads * self.runner.linear_key_size,
            self.runner.linear_value_heads * self.runner.linear_value_size,
        )
        layer["q"], layer["k"], layer["v"] = (
            wp.empty((self.rows, width), dtype=self.dtype, device=self.device)
            for width in widths
        )
        layer["q_norm"] = wp.empty_like(layer["q"])
        layer["k_norm"] = wp.empty_like(layer["k"])
        layer["decay"] = wp.empty(
            (self.rows, self.runner.linear_value_heads),
            dtype=wp.float32,
            device=self.device,
        )
        layer["beta"] = wp.empty_like(layer["decay"])
        layer["core"] = wp.empty(
            (self.rows, self.runner.linear_value_heads * self.runner.linear_value_size),
            dtype=self.dtype,
            device=self.device,
        )
        layer["gated"] = wp.empty_like(layer["core"])
        layer["lp_block"], layer["lp_kernel"] = _get_lp_normalization_kernel(
            self.runner.linear_key_size, self.dtype
        )
        layer["attention_kernel"] = _get_linear_attention_kernel(
            self.runner.linear_key_size,
            self.runner.linear_value_size,
            self.dtype,
            wp.float32,
            scalar_gated_delta=True,
        )
        if self.decode_batch:
            layer["attention_kernel_batch"] = _get_gated_delta_decode_batch_kernel(
                self.runner.linear_key_size,
                self.runner.linear_value_size,
                self.dtype,
                self.mapped_state,
            )
            layer["conv_batch_kernels"] = _get_causal_conv_decode_batch_kernels(
                self.mapped_state
            )
        scale_dtype = self.runner.weights[attn + "norm.weight"].dtype
        layer["gated_block"], layer["gated_kernel"] = _get_gated_rms_norm_kernel(
            self.runner.linear_value_size,
            self.dtype,
            scale_dtype=scale_dtype,
        )
        self.tensors[f"layer.{index}.gated"] = layer["gated"].reshape(
            (self.rows, self.runner.linear_value_heads * self.runner.linear_value_size)
        )
        self.shapes[f"layer.{index}.gated"] = tuple(
            self.tensors[f"layer.{index}.gated"].shape
        )
        layer["output"] = self._linear(
            f"layer.{index}.attention_output",
            f"layer.{index}.gated",
            attn + "out_proj.weight",
        )

    def _build_full_attention(
        self, layer: dict, index: int, prefix: str, x: str
    ) -> None:
        attn = prefix + "self_attn."
        attention_width = self.runner.query_heads * self.runner.head_size
        projection_q8_cache = {}
        layer["q_proj"] = self._linear(
            f"layer.{index}.q_projected",
            x,
            attn + "q_proj.weight",
            projection_q8_cache,
        )
        layer["k_proj"] = self._linear(
            f"layer.{index}.k_projected",
            x,
            attn + "k_proj.weight",
            projection_q8_cache,
        )
        layer["v_proj"] = self._linear(
            f"layer.{index}.v_projected",
            x,
            attn + "v_proj.weight",
            projection_q8_cache,
        )
        if self.decode_batch:
            layer["q_heads"] = wp.empty(
                (self.rows, self.runner.query_heads, 1, self.runner.head_size),
                dtype=self.dtype,
                device=self.device,
            )
            layer["k_heads"] = wp.empty(
                (self.rows, self.runner.kv_heads, 1, self.runner.head_size),
                dtype=self.dtype,
                device=self.device,
            )
            layer["v_heads"] = wp.empty_like(layer["k_heads"])
            layer["q"] = layer["q_heads"].reshape(
                (self.rows * self.runner.query_heads, self.runner.head_size)
            )
            layer["k"] = layer["k_heads"].reshape(
                (self.rows * self.runner.kv_heads, self.runner.head_size)
            )
            layer["v"] = layer["v_heads"].reshape(
                (self.rows * self.runner.kv_heads, self.runner.head_size)
            )
        else:
            layer["q"] = wp.empty(
                (self.runner.query_heads * self.rows, self.runner.head_size),
                dtype=self.dtype,
                device=self.device,
            )
            layer["k"] = wp.empty(
                (self.runner.kv_heads * self.rows, self.runner.head_size),
                dtype=self.dtype,
                device=self.device,
            )
            layer["v"] = wp.empty_like(layer["k"])
        layer["attention_gate"] = wp.empty(
            (self.rows, attention_width), dtype=self.dtype, device=self.device
        )
        self.tensors[f"layer.{index}.q"] = layer["q"]
        self.shapes[f"layer.{index}.q"] = tuple(layer["q"].shape)
        self.tensors[f"layer.{index}.k"] = layer["k"]
        self.shapes[f"layer.{index}.k"] = tuple(layer["k"].shape)
        layer["q_norm"] = self._rms(
            f"layer.{index}.q_norm", f"layer.{index}.q", attn + "q_norm.weight"
        )
        layer["k_norm"] = self._rms(
            f"layer.{index}.k_norm", f"layer.{index}.k", attn + "k_norm.weight"
        )
        if self.decode_batch:
            layer["q_rotated_heads"] = wp.empty_like(layer["q_heads"])
            layer["k_rotated_heads"] = wp.empty_like(layer["k_heads"])
            layer["q_rotated"] = layer["q_rotated_heads"].reshape(layer["q"].shape)
            layer["k_rotated"] = layer["k_rotated_heads"].reshape(layer["k"].shape)
        else:
            layer["q_rotated"] = wp.empty_like(layer["q"])
            layer["k_rotated"] = wp.empty_like(layer["k"])
        layer["core"] = wp.empty(
            (self.rows, attention_width), dtype=self.dtype, device=self.device
        )
        layer["gated"] = wp.empty_like(layer["core"])
        layer["attention_block"], layer["attention_kernel"] = _get_gqa_attention_kernel(
            self.runner.head_size, self.dtype
        )
        if not hasattr(self, "partitioned_attention"):
            self.attention_partitions = (
                _decode_attention_partitions(self.runner.head_size)
                if self.rows == 1 or self.decode_batch
                else 16
            )
            partitions = self.attention_partitions
            self.partitioned_attention = {
                partitions: _allocate_partitioned_gqa(
                    self.runner.query_heads,
                    self.runner.head_size,
                    self.dtype,
                    self.device,
                    partitions,
                    rows=self.rows,
                    kv_heads=self.runner.kv_heads,
                    mapped=self.mapped_state,
                )
            }
        layer["partitioned_attention"] = self.partitioned_attention
        if self.decode_batch:
            layer["append_cache_kernel"] = _get_append_head_cache_decode_batch_kernel(
                self.mapped_state
            )
        self.tensors[f"layer.{index}.gated"] = layer["gated"]
        self.shapes[f"layer.{index}.gated"] = tuple(layer["gated"].shape)
        layer["output"] = self._linear(
            f"layer.{index}.attention_output",
            f"layer.{index}.gated",
            attn + "o_proj.weight",
        )

    def _execute_op(self, op: Operation) -> None:
        execute_operations(
            op.attrs["_sequence"], self.tensors, self.shapes, self.device
        )

    def _execute_linear_attention_batch(self, layer: dict, index: int) -> None:
        for name in ("qkv", "z", "a", "b"):
            self._execute_op(layer[name])
        qkv = self.tensors[layer["qkv"].outputs[0]]
        wp.launch(
            layer["conv_batch_kernels"][0],
            dim=layer["conv"].shape,
            inputs=[
                qkv,
                self.runner.weights[
                    f"model.language_model.layers.{index}.linear_attn.conv1d.weight"
                ],
                self.runner.conv_states[index],
                self.runner.active,
                self.runner.slot_indices,
                layer["conv"],
            ],
            device=self.device,
        )
        offset = 0
        for output in (layer["q"], layer["k"], layer["v"]):
            wp.launch(
                _split_last_axis_kernel,
                dim=output.shape,
                inputs=[layer["conv"], output, offset],
                device=self.device,
            )
            offset += output.shape[1]
        wp.launch(
            layer["conv_batch_kernels"][1],
            dim=qkv.shape,
            inputs=[
                qkv,
                self.runner.conv_states[index],
                self.runner.active,
                self.runner.slot_indices,
            ],
            device=self.device,
        )
        for source, output in (
            (layer["q"], layer["q_norm"]),
            (layer["k"], layer["k_norm"]),
        ):
            wp.launch_tiled(
                layer["lp_kernel"],
                dim=self.rows * self.runner.linear_key_heads,
                inputs=[
                    source.reshape((-1, self.runner.linear_key_size)),
                    output.reshape((-1, self.runner.linear_key_size)),
                    1.0e-6,
                ],
                block_dim=layer["lp_block"],
                device=self.device,
            )
        wp.launch(
            _prepare_gated_delta_kernel,
            dim=layer["decay"].shape,
            inputs=[
                self.tensors[layer["a"].outputs[0]],
                self.tensors[layer["b"].outputs[0]],
                self.runner.weights[
                    f"model.language_model.layers.{index}.linear_attn.A_log"
                ],
                self.runner.weights[
                    f"model.language_model.layers.{index}.linear_attn.dt_bias"
                ],
                self.runner.ssm_a_is_decay,
                layer["decay"],
                layer["beta"],
            ],
            device=self.device,
        )
        wp.launch_tiled(
            layer["attention_kernel_batch"],
            dim=self.rows
            * self.runner.linear_value_heads
            * _linear_attention_value_blocks(self.runner.linear_value_size),
            inputs=[
                layer["q_norm"],
                layer["k_norm"],
                layer["v"],
                self.runner.recurrent_states[index],
                layer["decay"],
                layer["beta"],
                self.runner.active,
                self.runner.slot_indices,
                layer["core"],
                self.runner.linear_key_heads,
                self.runner.linear_key_heads,
                self.runner.linear_value_heads,
                self.runner.linear_key_size**-0.5,
            ],
            block_dim=32,
            device=self.device,
        )
        wp.launch_tiled(
            layer["gated_kernel"],
            dim=self.rows * self.runner.linear_value_heads,
            inputs=[
                layer["core"].reshape((-1, self.runner.linear_value_size)),
                self.tensors[layer["z"].outputs[0]].reshape(
                    (-1, self.runner.linear_value_size)
                ),
                self.runner.weights[
                    f"model.language_model.layers.{index}.linear_attn.norm.weight"
                ].reshape((1, self.runner.linear_value_size)),
                layer["gated"].reshape((-1, self.runner.linear_value_size)),
                self.runner.epsilon,
            ],
            block_dim=layer["gated_block"],
            device=self.device,
        )
        self._execute_op(layer["output"])

    def _execute_linear_attention(self, layer: dict, index: int) -> None:
        if self.decode_batch:
            self._execute_linear_attention_batch(layer, index)
            return
        for name in ("qkv", "z", "a", "b"):
            self._execute_op(layer[name])
        qkv = self.tensors[layer["qkv"].outputs[0]]
        wp.launch(
            _causal_conv_rows_kernel,
            dim=layer["conv"].shape,
            inputs=[
                qkv,
                self.runner.weights[
                    f"model.language_model.layers.{index}.linear_attn.conv1d.weight"
                ],
                self.runner.zero_bias,
                self.runner.conv_states[index],
                layer["conv"],
                False,
            ],
            device=self.device,
        )
        offset = 0
        for output in (layer["q"], layer["k"], layer["v"]):
            wp.launch(
                _split_last_axis_kernel,
                dim=output.shape,
                inputs=[layer["conv"], output, offset],
                device=self.device,
            )
            offset += output.shape[1]
        wp.launch(
            _update_conv_rows_state_kernel,
            dim=qkv.shape[1],
            inputs=[qkv, self.runner.conv_states[index]],
            device=self.device,
        )
        for source, output in (
            (layer["q"], layer["q_norm"]),
            (layer["k"], layer["k_norm"]),
        ):
            wp.launch_tiled(
                layer["lp_kernel"],
                dim=self.rows * self.runner.linear_key_heads,
                inputs=[
                    source.reshape((-1, self.runner.linear_key_size)),
                    output.reshape((-1, self.runner.linear_key_size)),
                    1.0e-6,
                ],
                block_dim=layer["lp_block"],
                device=self.device,
            )
        wp.launch(
            _prepare_gated_delta_kernel,
            dim=layer["decay"].shape,
            inputs=[
                self.tensors[layer["a"].outputs[0]],
                self.tensors[layer["b"].outputs[0]],
                self.runner.weights[
                    f"model.language_model.layers.{index}.linear_attn.A_log"
                ],
                self.runner.weights[
                    f"model.language_model.layers.{index}.linear_attn.dt_bias"
                ],
                self.runner.ssm_a_is_decay,
                layer["decay"],
                layer["beta"],
            ],
            device=self.device,
        )
        state = self.runner.recurrent_states[index]
        wp.launch_tiled(
            layer["attention_kernel"],
            dim=self.runner.linear_value_heads
            * _linear_attention_value_blocks(self.runner.linear_value_size),
            inputs=[
                layer["q_norm"],
                layer["k_norm"],
                layer["v"],
                state,
                layer["decay"],
                layer["beta"],
                layer["core"],
                state,
                self.rows,
                self.runner.linear_key_heads,
                self.runner.linear_key_heads,
                self.runner.linear_value_heads,
                self.runner.gguf_layout,
                True,
                False,
                True,
                True,
                self.runner.linear_key_size**-0.5,
            ],
            block_dim=32,
            device=self.device,
        )
        wp.launch_tiled(
            layer["gated_kernel"],
            dim=self.rows * self.runner.linear_value_heads,
            inputs=[
                layer["core"].reshape((-1, self.runner.linear_value_size)),
                self.tensors[layer["z"].outputs[0]].reshape(
                    (-1, self.runner.linear_value_size)
                ),
                self.runner.weights[
                    f"model.language_model.layers.{index}.linear_attn.norm.weight"
                ].reshape((1, self.runner.linear_value_size)),
                layer["gated"].reshape((-1, self.runner.linear_value_size)),
                self.runner.epsilon,
            ],
            block_dim=layer["gated_block"],
            device=self.device,
        )
        self._execute_op(layer["output"])

    def _execute_full_attention_batch(self, layer: dict, index: int) -> None:
        for name in ("q_proj", "k_proj", "v_proj"):
            self._execute_op(layer[name])
        wp.launch(
            _unpack_gated_heads_decode_batch_kernel,
            dim=(self.rows, self.runner.query_heads, self.runner.head_size),
            inputs=[
                self.tensors[layer["q_proj"].outputs[0]],
                layer["q_heads"],
                layer["attention_gate"],
                self.runner.head_size,
                self.runner.gguf_layout,
            ],
            device=self.device,
        )
        for projected, output, interleaved in (
            (layer["k_proj"], layer["k_heads"], self.runner.gguf_layout),
            (layer["v_proj"], layer["v_heads"], False),
        ):
            wp.launch(
                _reorder_heads_decode_batch_kernel,
                dim=(self.rows, self.runner.kv_heads, self.runner.head_size),
                inputs=[
                    self.tensors[projected.outputs[0]],
                    output,
                    self.runner.head_size,
                    interleaved,
                ],
                device=self.device,
            )
        self._execute_op(layer["q_norm"])
        self._execute_op(layer["k_norm"])
        rotary = _get_mrope_decode_batch_kernel(self.dtype)
        for source, output, heads in (
            (
                self.tensors[layer["q_norm"].outputs[0]].reshape(
                    (self.rows, self.runner.query_heads, 1, self.runner.head_size)
                ),
                layer["q_rotated_heads"],
                self.runner.query_heads,
            ),
            (
                self.tensors[layer["k_norm"].outputs[0]].reshape(
                    (self.rows, self.runner.kv_heads, 1, self.runner.head_size)
                ),
                layer["k_rotated_heads"],
                self.runner.kv_heads,
            ),
        ):
            wp.launch(
                rotary,
                dim=(self.rows, heads, self.runner.head_size),
                inputs=[
                    source,
                    self.rope_position_ids,
                    self.runner.cos_cache,
                    self.runner.sin_cache,
                    output,
                    self.runner.rotary_dim,
                ],
                device=self.device,
            )
        key_cache, value_cache = self.runner.kv_caches[index]
        for source, cache in (
            (layer["k_rotated_heads"], key_cache),
            (layer["v_heads"], value_cache),
        ):
            wp.launch(
                layer["append_cache_kernel"],
                dim=(self.rows, self.runner.kv_heads, self.runner.head_size),
                inputs=[
                    source,
                    self.runner.positions,
                    self.runner.active,
                    self.runner.slot_indices,
                    cache,
                    self.runner.cache_capacity,
                ],
                device=self.device,
            )
        _launch_partitioned_gqa(
            layer["partitioned_attention"][self.attention_partitions],
            layer["q_rotated"],
            key_cache,
            value_cache,
            self.runner.sequence_end,
            layer["core"],
            self.runner.query_heads,
            self.runner.kv_heads,
            self.runner.cache_capacity,
            self.runner.head_size**-0.5,
            0,
            self.device,
            sequence_length=1,
            slot_indices=(self.runner.slot_indices if self.mapped_state else None),
        )
        wp.launch(
            _sigmoid_gate_kernel,
            dim=layer["core"].shape,
            inputs=[layer["core"], layer["attention_gate"], layer["gated"]],
            device=self.device,
        )
        self._execute_op(layer["output"])

    def _execute_full_attention(self, layer: dict, index: int) -> None:
        if self.decode_batch:
            self._execute_full_attention_batch(layer, index)
            return
        for name in ("q_proj", "k_proj", "v_proj"):
            self._execute_op(layer[name])
        wp.launch(
            _unpack_gated_heads_kernel,
            dim=(self.rows, self.runner.query_heads, self.runner.head_size),
            inputs=[
                self.tensors[layer["q_proj"].outputs[0]],
                layer["q"],
                layer["attention_gate"],
                self.runner.head_size,
                self.runner.gguf_layout,
            ],
            device=self.device,
        )
        for projected, output, interleaved in (
            (layer["k_proj"], layer["k"], self.runner.gguf_layout),
            (layer["v_proj"], layer["v"], False),
        ):
            wp.launch(
                _reorder_interleaved_heads_kernel
                if interleaved
                else _reorder_heads_kernel,
                dim=(self.rows, self.runner.kv_heads, self.runner.head_size),
                inputs=[
                    self.tensors[projected.outputs[0]],
                    output,
                    self.runner.head_size,
                ],
                device=self.device,
            )
        self._execute_op(layer["q_norm"])
        self._execute_op(layer["k_norm"])
        rotary = _get_mrope_embedding_kernel(self.dtype)
        for source, output, heads in (
            (
                self.tensors[layer["q_norm"].outputs[0]],
                layer["q_rotated"],
                self.runner.query_heads,
            ),
            (
                self.tensors[layer["k_norm"].outputs[0]],
                layer["k_rotated"],
                self.runner.kv_heads,
            ),
        ):
            wp.launch(
                rotary,
                dim=(1, heads, self.rows, self.runner.head_size),
                inputs=[
                    source.reshape((1, heads, self.rows, self.runner.head_size)),
                    self.rope_position_ids,
                    self.runner.cos_cache,
                    self.runner.sin_cache,
                    output.reshape((1, heads, self.rows, self.runner.head_size)),
                    self.runner.rotary_dim,
                ],
                device=self.device,
            )
        key_cache, value_cache = self.runner.kv_caches[index]
        for source, cache in (
            (layer["k_rotated"], key_cache),
            (layer["v"], value_cache),
        ):
            wp.launch(
                _append_head_cache_kernel,
                dim=(self.runner.kv_heads, self.rows, self.runner.head_size),
                inputs=[
                    source,
                    self.position_ids,
                    cache,
                    self.runner.kv_heads,
                    self.runner.head_size,
                ],
                device=self.device,
            )
        if "partitioned_attention" in layer:
            _launch_partitioned_gqa(
                layer["partitioned_attention"][self.attention_partitions],
                layer["q_rotated"],
                key_cache,
                value_cache,
                self.runner.sequence_end,
                layer["core"],
                self.runner.query_heads,
                self.runner.kv_heads,
                self.runner.cache_capacity,
                self.runner.head_size**-0.5,
                0,
                self.device,
            )
        else:
            wp.launch_tiled(
                layer["attention_kernel"],
                dim=self.runner.query_heads * self.rows,
                inputs=[
                    layer["q_rotated"],
                    key_cache,
                    value_cache,
                    self.runner.sequence_end,
                    layer["core"],
                    self.runner.query_heads,
                    self.runner.kv_heads,
                    self.rows,
                    self.runner.cache_capacity,
                    self.runner.head_size**-0.5,
                    0,
                ],
                block_dim=layer["attention_block"],
                device=self.device,
            )
        wp.launch(
            _sigmoid_gate_kernel,
            dim=layer["core"].shape,
            inputs=[layer["core"], layer["attention_gate"], layer["gated"]],
            device=self.device,
        )
        self._execute_op(layer["output"])

    def execute(self) -> wp.array:
        """Execute the preallocated plan on its staged token buffers."""
        embedding_weight = self.runner.weights[
            "model.language_model.embed_tokens.weight"
        ]
        if isinstance(embedding_weight, BlockQuantizedTensor):
            wp.launch(
                _get_gather_q8_0_rows_kernel(self.dtype),
                dim=self.embedding.shape,
                inputs=[
                    embedding_weight.values,
                    self.input_ids,
                    embedding_weight.scales,
                    self.embedding,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _gather_rows_kernel,
                dim=self.embedding.shape,
                inputs=[embedding_weight, self.input_ids, self.embedding],
                device=self.device,
            )
        if self.external_embeddings:
            wp.launch(
                _overlay_embedding_rows_kernel,
                dim=(self.rows, self.runner.hidden_size),
                inputs=[
                    self.embedding.reshape((self.rows, self.runner.hidden_size)),
                    self.visual_embeddings,
                    self.visual_indices,
                ],
                device=self.device,
            )
        self._execute_op(self.first_norm)
        for index, layer in enumerate(self.layers):
            if layer["type"] == "linear_attention":
                self._execute_linear_attention(layer, index)
            else:
                self._execute_full_attention(layer, index)
            for name in (
                "post_norm",
                "mlp_gate",
                "mlp_up",
                "swiglu",
                "down",
                "next_norm",
            ):
                self._execute_op(layer[name])
        self._execute_op(self.lm_head)
        return self.logits


class Qwen35BatchDecoder(NativeBatchDecoder):
    """Qwen batch decoder using shared persistent-slot execution."""

    def __init__(self, runner, max_batch_size: int):
        super().__init__(runner, max_batch_size, _Qwen35Plan, "Qwen")


class Qwen35Runner(AutoregressiveRunner):
    """Run a Qwen 3.5-family text or multimodal checkpoint entirely with Warp."""

    def __init__(
        self,
        path: str | Path,
        device: str | wp.Device | None = None,
        cache_capacity: int = 4096,
        prefill_chunk_size: int = 16,
        use_cublas: bool = True,
        rope_scaling: Mapping[str, object] | None = None,
        weight_quantization: str | None = None,
        vision_path: str | Path | None = None,
    ):
        path = Path(path)
        self.model_path = path
        self.vision_path = None if vision_path is None else Path(vision_path)
        directory = path if path.is_dir() else path.parent
        if any(directory.glob("*.safetensors")):
            config_data = json.loads(
                (directory / "config.json").read_text(encoding="utf-8")
            )
            self.config = config_data.get("text_config", config_data)
            gguf = None
        else:
            gguf = GGUFArchive(find_gguf_files(path))
            if gguf.metadata.get("general.architecture") != "qwen35":
                raise ValueError("GGUF checkpoint is not a Qwen 3.5 model")
            self.config = _gguf_config(gguf.metadata)
        _validate_config(self.config)
        self.device = parse_device(device)
        self.weight_quantization = normalize_weight_quantization(weight_quantization)
        self.cache_capacity = int(cache_capacity)
        if self.cache_capacity <= 0:
            raise ValueError("cache_capacity must be positive")
        self.rope_parameters = resolve_rope_parameters(
            self.config["rope_parameters"],
            rope_scaling,
            int(self.config["max_position_embeddings"]),
            self.cache_capacity,
        )
        if prefill_chunk_size < 2 or prefill_chunk_size > self.cache_capacity:
            raise ValueError("prefill_chunk_size must be between 2 and cache_capacity")
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.hidden_size = int(self.config["hidden_size"])
        self.query_heads = int(self.config["num_attention_heads"])
        self.kv_heads = int(self.config["num_key_value_heads"])
        self.head_size = int(self.config["head_dim"])
        self.linear_key_heads = int(self.config["linear_num_key_heads"])
        self.linear_value_heads = int(self.config["linear_num_value_heads"])
        self.linear_key_size = int(self.config["linear_key_head_dim"])
        self.linear_value_size = int(self.config["linear_value_head_dim"])
        self.epsilon = float(self.config["rms_norm_eps"])
        rope = self.rope_parameters
        self.rotary_dim = int(
            self.head_size * float(rope.get("partial_rotary_factor", 1.0))
        )
        if self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise ValueError("Qwen 3.5 rotary dimension must be positive and even")

        if gguf is None:
            archive = SafeTensorArchive(directory)
            self.gguf_layout = False
            self.centered_norm_scales = True
            self.ssm_a_is_decay = False
        else:
            archive = MappedWeightArchive(
                gguf, _gguf_weight_map(self.config), gguf.tensor
            )
            self.gguf_layout = True
            self.centered_norm_scales = False
            self.ssm_a_is_decay = True
        names = _weight_names(self.config)
        missing = set(names) - set(archive.names)
        if missing:
            raise ValueError(f"Qwen 3.5 checkpoint is missing {sorted(missing)[:5]}")
        required_bytes, largest_weight_source = estimate_loaded_weight_bytes(
            archive, names, self.weight_quantization
        )
        weight_load_peak = required_bytes + largest_weight_source
        full_layers = self.config["layer_types"].count("full_attention")
        required_bytes += (
            full_layers * 2 * self.kv_heads * self.cache_capacity * self.head_size * 2
        )
        required_bytes += self.cache_capacity * self.rotary_dim * 4
        required_bytes = max(required_bytes, weight_load_peak)
        if self.device.is_cuda and required_bytes > self.device.free_memory * 0.95:
            raise MemoryError(
                f"Qwen 3.5 needs at least {required_bytes / 2**30:.1f} GiB for selected weights and KV cache; "
                f"{self.device.free_memory / 2**30:.1f} GiB is currently free"
            )
        self.weights = load_native_weights(
            archive, self.device, names, self.weight_quantization
        )
        if self.gguf_layout:
            for index, layer_type in enumerate(self.config["layer_types"]):
                if layer_type == "linear_attention":
                    name = (
                        f"model.language_model.layers.{index}.linear_attn.conv1d.weight"
                    )
                    weight = self.weights[name]
                    self.weights[name] = weight.reshape(
                        (weight.shape[0], 1, weight.shape[1])
                    )
        embedding_weight = self.weights["model.language_model.embed_tokens.weight"]
        self.dtype = (
            wp.bfloat16
            if isinstance(embedding_weight, BlockQuantizedTensor)
            else embedding_weight.dtype
        )
        if self.dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Qwen 3.5 activations require FP16 or BF16 weights")
        self.zero_bias = wp.zeros(1, dtype=self.dtype, device=self.device)
        self.cublas = (
            try_create_cublas() if use_cublas and self.device.is_cuda else None
        )
        self.sequence_end = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.conv_states = {}
        self.recurrent_states = {}
        self.kv_caches = {}
        conv_dim = (
            self.linear_key_heads * self.linear_key_size * 2
            + self.linear_value_heads * self.linear_value_size
        )
        conv_state_width = int(self.config["linear_conv_kernel_dim"]) - 1
        for index, layer_type in enumerate(self.config["layer_types"]):
            if layer_type == "linear_attention":
                self.conv_states[index] = wp.zeros(
                    (conv_dim, conv_state_width), dtype=self.dtype, device=self.device
                )
                self.recurrent_states[index] = wp.zeros(
                    (
                        self.linear_value_heads * self.linear_key_size,
                        self.linear_value_size,
                    ),
                    dtype=wp.float32,
                    device=self.device,
                )
            else:
                shape = (self.kv_heads * self.cache_capacity, self.head_size)
                self.kv_caches[index] = (
                    wp.empty(shape, dtype=self.dtype, device=self.device),
                    wp.empty(shape, dtype=self.dtype, device=self.device),
                )
        cos_cache, sin_cache = rotary_cache_values(
            self.cache_capacity, self.rotary_dim, self.rope_parameters
        )
        self.cos_cache = wp.array(cos_cache, dtype=self.dtype, device=self.device)
        self.sin_cache = wp.array(sin_cache, dtype=self.dtype, device=self.device)
        self._decode_plan = _Qwen35Plan(self, 1)
        self._chunk_plan = _Qwen35Plan(self, self.prefill_chunk_size)
        self._chunk_plan._capture_ready = False
        self._record_plan_storage(self._decode_plan)
        self._record_plan_storage(self._chunk_plan)
        self._sample_partial_values = wp.empty(
            128, dtype=wp.float32, device=self.device
        )
        self._sample_partial_tokens = wp.empty(128, dtype=wp.int32, device=self.device)
        self._sampled_token = wp.empty(1, dtype=wp.int32, device=self.device)
        self._sampled_token_host = wp.empty(
            1, dtype=wp.int32, device="cpu", pinned=self.device.is_cuda
        )
        self._sampled_token_host_view = self._sampled_token_host.numpy()
        self._greedy_argmax_kernels = _get_greedy_argmax_kernels(1024, 128, self.dtype)
        self.sequence_length = 0

    def create_batch_decoder(self, max_batch_size: int = 4) -> Qwen35BatchDecoder:
        """Allocate opt-in independent decode state without duplicating weights."""
        return Qwen35BatchDecoder(self, max_batch_size)

    def _multimodal_plan_for_rows(self, rows: int) -> _Qwen35Plan:
        plans = getattr(self, "_multimodal_plans", None)
        if plans is None:
            plans = self._multimodal_plans = {}
        plan = plans.get(rows)
        if plan is None:
            self._require_lazy_plan_headroom(rows)
            plan = plans[rows] = _Qwen35Plan(self, rows, external_embeddings=True)
            plan._capture_ready = False
            self._record_plan_storage(plan)
        return plan

    def _vision_encoder(self):
        encoder = getattr(self, "_vision_encoder_instance", None)
        if encoder is None:
            from .vision import QwenVisionEncoder

            encoder = self._vision_encoder_instance = QwenVisionEncoder(
                self.model_path,
                device=self.device,
                cublas=self.cublas,
                vision_path=self.vision_path,
            )
        return encoder

    def prefill_multimodal(self, prompt) -> wp.array:
        """Reset and prefill a :class:`QwenMultimodalPrompt` entirely on device."""
        from .vision import QwenMultimodalPrompt

        if not isinstance(prompt, QwenMultimodalPrompt):
            raise TypeError("prefill_multimodal expects QwenMultimodalPrompt")
        if not prompt.media:
            return self.prefill(prompt.token_ids)
        if len(prompt.token_ids) >= self.cache_capacity:
            raise ValueError(
                "Qwen multimodal prompt must leave room for one decoded token"
            )
        self.reset()
        encoded = []
        for media in prompt.media:
            output = self._vision_encoder().encode(media)
            if output.shape[1] != self.hidden_size:
                raise ValueError(
                    "vision encoder output does not match text hidden size"
                )
            if output.dtype != self.dtype:
                converted = wp.empty(output.shape, dtype=self.dtype, device=self.device)
                wp.launch(
                    _cast_kernel_for_dtypes(output.dtype, self.dtype),
                    dim=output.size,
                    inputs=[output.flatten(), converted.flatten()],
                    device=self.device,
                )
                output = converted
            encoded.append(output)

        logits = None
        start = 0
        while start < len(prompt.token_ids):
            remaining = len(prompt.token_ids) - start
            rows = min(self.prefill_chunk_size, 1 << (remaining.bit_length() - 1))
            plan = self._multimodal_plan_for_rows(rows)
            end = start + rows
            plan.input_ids.assign(
                np.asarray(prompt.token_ids[start:end], dtype=np.int64)[None, :]
            )
            plan.position_ids.assign(np.arange(start, end, dtype=np.int64)[None, :])
            plan.rope_position_ids.assign(prompt.rope_positions[:, start:end])
            source_indices = np.full(rows, -1, dtype=np.int32)
            for media_index, (feature_start, media) in enumerate(
                zip(prompt.feature_starts, prompt.media, strict=True)
            ):
                feature_end = feature_start + media.feature_count
                overlap_start, overlap_end = (
                    max(start, feature_start),
                    min(end, feature_end),
                )
                if overlap_start >= overlap_end:
                    continue
                count = overlap_end - overlap_start
                destination_row = overlap_start - start
                source_row = overlap_start - feature_start
                wp.copy(
                    plan.visual_embeddings.flatten(),
                    encoded[media_index].flatten(),
                    dest_offset=destination_row * self.hidden_size,
                    src_offset=source_row * self.hidden_size,
                    count=count * self.hidden_size,
                )
                source_indices[destination_row : destination_row + count] = np.arange(
                    destination_row, destination_row + count, dtype=np.int32
                )
            plan.visual_indices.assign(source_indices)
            wp.launch(
                _set_sequence_end,
                dim=1,
                inputs=[self.sequence_end, end - 1],
                device=self.device,
            )
            logits = self._run(plan)
            self.sequence_length = end
            start = end
        self.rope_delta = int(prompt.rope_delta)
        return logits
