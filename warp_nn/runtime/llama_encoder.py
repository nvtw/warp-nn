# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free bidirectional Llama encoder used by LLM2Vec/Kimodo."""

import json
import math
from pathlib import Path

import numpy as np
import warp as wp

from .encoder import _encoder_kernels
from .kernels import (
    _binary_broadcast_kernel,
    _get_masked_mean_pool_kernel,
    _gather_rows_kernel,
    _reorder_heads_kernel,
    _rotary_embedding_kernel_for_dtype,
)
from .operators import (
    Operation,
    execute_operations,
    plan_linear,
    plan_rms_norm,
    plan_swiglu,
)
from .qwen3 import Qwen3Tokenizer
from .rope import rotary_cache_values
from .safetensors import SafeTensorArchive
from .weights import load_cast_weights, merge_lora_weight


class Llama3Tokenizer(Qwen3Tokenizer):
    """Llama 3 ByteLevel-BPE using the shared dependency-free implementation."""

    def __init__(self, path):
        super().__init__(path, pretokenizer="llama3")


def load_llama_config(path):
    data = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    required = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Llama config is missing: {', '.join(missing)}")
    return data


def llama_encoder_weight_names(config):
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for index in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{index}"
        names.extend(
            (
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            )
        )
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.append(f"{prefix}.self_attn.{projection}.weight")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names.append(f"{prefix}.mlp.{projection}.weight")
    return tuple(names)


def _adapter_base_name(name):
    marker = "model.layers."
    position = name.find(marker)
    if position < 0 or not name.endswith(".lora_A.weight"):
        return None
    return name[position:].replace(".lora_A.weight", ".weight")


def merge_lora_adapter(weights, adapter_path, dtype, device):
    """Merge one PEFT LoRA adapter into compute weights without Torch/PEFT."""
    adapter_path = Path(adapter_path)
    config = json.loads(
        (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    rank = int(config["r"])
    scale = float(config.get("lora_alpha", rank)) / rank
    adapter_file = adapter_path / "adapter_model.safetensors"
    archive = SafeTensorArchive(
        adapter_file if adapter_file.is_file() else adapter_path
    )
    merged = 0
    for a_name in archive.names:
        base_name = _adapter_base_name(a_name)
        if base_name is None:
            continue
        b_name = a_name.replace(".lora_A.weight", ".lora_B.weight")
        if b_name not in archive.names or base_name not in weights:
            raise KeyError(f"adapter target '{base_name}' is incomplete or absent")
        pair = load_cast_weights(archive, (a_name, b_name), device, dtype)
        a, b = pair[a_name], pair[b_name]
        if a.shape != (rank, weights[base_name].shape[1]) or b.shape != (
            weights[base_name].shape[0],
            rank,
        ):
            raise ValueError(f"adapter tensors for '{base_name}' have invalid shapes")
        merge_lora_weight(weights[base_name], a, b, scale)
        merged += 1
    if not merged:
        raise ValueError(f"LoRA adapter {adapter_path!s} contains no supported targets")
    return weights


class _LlamaEncoderPlan:
    def __init__(self, runner, sequence):
        self.runner = runner
        self.device = runner.device
        self.sequence = sequence
        self.hidden_size = runner.hidden_size
        self.rows = sequence
        self.input_ids = wp.empty((1, sequence), dtype=wp.int64, device=self.device)
        self.position_ids = wp.array(
            np.arange(sequence, dtype=np.int64)[None], device=self.device
        )
        self.valid = wp.empty((1, sequence), dtype=wp.bool, device=self.device)
        self.embed_mask = wp.empty((sequence,), dtype=wp.bool, device=self.device)
        self.embedding = wp.empty(
            (1, sequence, self.hidden_size), dtype=runner.dtype, device=self.device
        )
        self.tensors = dict(runner.weights)
        self.shapes = {name: value.shape for name, value in self.tensors.items()}
        self.tensors["hidden.0"] = self.embedding.reshape((sequence, self.hidden_size))
        self.shapes["hidden.0"] = (sequence, self.hidden_size)
        self.layers = []
        self._build()
        self.output = wp.empty(
            (1, self.hidden_size), dtype=runner.dtype, device=self.device
        )
        self._graph = None
        self._capture_ready = False

    def _linear(self, name, source, weight):
        op = Operation("Linear", [source, weight], [name])
        plan_linear(op, self.tensors, self.shapes, self.device, self.runner.cublas)
        return op

    def _rms(self, name, source, scale):
        op = Operation(
            "SimplifiedLayerNormalization",
            [source, scale],
            [name],
            {"epsilon": self.runner.epsilon},
        )
        plan_rms_norm(op, self.tensors, self.shapes, self.device)
        return op

    def _build(self):
        current = "hidden.0"
        for index in range(self.runner.layers):
            prefix = f"model.layers.{index}"
            layer = {"input": current}
            layer["norm1"] = self._rms(
                f"layer.{index}.norm1", current, f"{prefix}.input_layernorm.weight"
            )
            normalized = layer["norm1"].outputs[0]
            for projection in ("q", "k", "v"):
                layer[projection] = self._linear(
                    f"layer.{index}.{projection}",
                    normalized,
                    f"{prefix}.self_attn.{projection}_proj.weight",
                )
            q_shape = (1, self.runner.query_heads, self.sequence, self.runner.head_size)
            kv_shape = (1, self.runner.kv_heads, self.sequence, self.runner.head_size)
            layer["query"] = wp.empty(
                q_shape, dtype=self.runner.dtype, device=self.device
            )
            layer["query_rotated"] = wp.empty(
                q_shape, dtype=self.runner.dtype, device=self.device
            )
            layer["key"] = wp.empty(
                kv_shape, dtype=self.runner.dtype, device=self.device
            )
            layer["key_rotated"] = wp.empty(
                kv_shape, dtype=self.runner.dtype, device=self.device
            )
            layer["value"] = wp.empty(
                kv_shape, dtype=self.runner.dtype, device=self.device
            )
            layer["attention"] = wp.empty(
                q_shape, dtype=self.runner.dtype, device=self.device
            )
            attention_flat = wp.empty(
                (self.sequence, self.hidden_size),
                dtype=self.runner.dtype,
                device=self.device,
            )
            layer["attention_flat"] = attention_flat
            self.tensors[f"layer.{index}.attention_flat"] = attention_flat
            self.shapes[f"layer.{index}.attention_flat"] = attention_flat.shape
            layer["o"] = self._linear(
                f"layer.{index}.attention_output",
                f"layer.{index}.attention_flat",
                f"{prefix}.self_attn.o_proj.weight",
            )
            residual1 = wp.empty(
                (self.sequence, self.hidden_size),
                dtype=self.runner.dtype,
                device=self.device,
            )
            layer["residual1"] = residual1
            self.tensors[f"layer.{index}.residual1"] = residual1
            self.shapes[f"layer.{index}.residual1"] = residual1.shape
            layer["norm2"] = self._rms(
                f"layer.{index}.norm2",
                f"layer.{index}.residual1",
                f"{prefix}.post_attention_layernorm.weight",
            )
            mlp_input = layer["norm2"].outputs[0]
            layer["gate"] = self._linear(
                f"layer.{index}.gate", mlp_input, f"{prefix}.mlp.gate_proj.weight"
            )
            layer["up"] = self._linear(
                f"layer.{index}.up", mlp_input, f"{prefix}.mlp.up_proj.weight"
            )
            swiglu = Operation(
                "_SwiGLU",
                [layer["gate"].outputs[0], layer["up"].outputs[0]],
                [f"layer.{index}.swiglu"],
            )
            plan_swiglu(swiglu, self.tensors, self.shapes, self.device)
            layer["swiglu"] = swiglu
            layer["down"] = self._linear(
                f"hidden.{index + 1}",
                swiglu.outputs[0],
                f"{prefix}.mlp.down_proj.weight",
            )
            current = layer["down"].outputs[0]
            self.layers.append(layer)
        self.final_norm = self._rms("final", current, "model.norm.weight")

    def execute(self):
        wp.launch(
            _gather_rows_kernel,
            dim=self.embedding.shape,
            inputs=[
                self.runner.weights["model.embed_tokens.weight"],
                self.input_ids,
                self.embedding,
            ],
            device=self.device,
        )
        *_, merge, attention_kernel = _encoder_kernels(
            self.runner.dtype, self.runner.head_size
        )
        pool = _get_masked_mean_pool_kernel(self.runner.dtype)
        rotary = _rotary_embedding_kernel_for_dtype(self.runner.dtype)
        for layer in self.layers:
            execute_operations([layer["norm1"]], self.tensors, self.shapes, self.device)
            execute_operations(
                [layer[name] for name in ("q", "k", "v")],
                self.tensors,
                self.shapes,
                self.device,
            )
            for name, heads in (
                ("q", self.runner.query_heads),
                ("k", self.runner.kv_heads),
                ("v", self.runner.kv_heads),
            ):
                target = layer[{"q": "query", "k": "key", "v": "value"}[name]]
                wp.launch(
                    _reorder_heads_kernel,
                    dim=(self.sequence, heads, self.runner.head_size),
                    inputs=[
                        self.tensors[layer[name].outputs[0]],
                        target.reshape((heads * self.sequence, self.runner.head_size)),
                        self.runner.head_size,
                    ],
                    device=self.device,
                )
            for name in ("query", "key"):
                source = layer[name]
                output = layer[f"{name}_rotated"]
                wp.launch(
                    rotary,
                    dim=source.shape,
                    inputs=[
                        source,
                        self.position_ids,
                        self.runner.cos_cache,
                        self.runner.sin_cache,
                        output,
                        self.runner.head_size,
                        False,
                        False,
                    ],
                    device=self.device,
                )
            wp.launch_tiled(
                attention_kernel,
                dim=self.runner.query_heads * self.sequence,
                inputs=[
                    layer["query_rotated"],
                    layer["key_rotated"],
                    layer["value"],
                    self.valid,
                    layer["attention"],
                    wp.float32(1.0 / math.sqrt(self.runner.head_size)),
                ],
                block_dim=128,
                device=self.device,
            )
            wp.launch(
                merge,
                dim=layer["attention"].shape,
                inputs=[layer["attention"], layer["attention_flat"]],
                device=self.device,
            )
            execute_operations([layer["o"]], self.tensors, self.shapes, self.device)
            wp.launch(
                _binary_broadcast_kernel,
                dim=layer["residual1"].shape,
                inputs=[
                    self.tensors[layer["input"]],
                    self.tensors[layer["o"].outputs[0]],
                    0,
                ],
                outputs=[layer["residual1"]],
                device=self.device,
            )
            execute_operations(
                [layer["norm2"], layer["gate"], layer["up"]],
                self.tensors,
                self.shapes,
                self.device,
            )
            execute_operations(
                [layer["swiglu"], layer["down"]], self.tensors, self.shapes, self.device
            )
            wp.launch(
                _binary_broadcast_kernel,
                dim=self.tensors[layer["down"].outputs[0]].shape,
                inputs=[
                    layer["residual1"],
                    self.tensors[layer["down"].outputs[0]],
                    0,
                ],
                outputs=[self.tensors[layer["down"].outputs[0]]],
                device=self.device,
            )
        execute_operations([self.final_norm], self.tensors, self.shapes, self.device)
        wp.launch(
            pool,
            dim=self.hidden_size,
            inputs=[
                self.tensors[self.final_norm.outputs[0]],
                self.embed_mask,
                self.output,
            ],
            device=self.device,
        )
        return self.output

    def run(self):
        """Execute eagerly once, then capture and replay a fixed-shape CUDA graph."""
        if not self.device.is_cuda:
            return self.execute()
        if self._graph is not None:
            wp.capture_launch(self._graph)
        elif self._capture_ready:
            wp.capture_begin(device=self.device)
            self.execute()
            self._graph = wp.capture_end(device=self.device)
            wp.capture_launch(self._graph)
        else:
            self.execute()
            self._capture_ready = True
        return self.output


class LLM2VecRunner:
    """Bidirectional Llama encoder with one or more merged LoRA adapters."""

    def __init__(
        self,
        model_path,
        adapter_paths=(),
        *,
        dtype=wp.bfloat16,
        device=None,
        use_cublas=False,
    ):
        self.device = wp.get_device(device)
        self.dtype = dtype
        self.config = load_llama_config(model_path)
        self.hidden_size = int(self.config["hidden_size"])
        self.layers = int(self.config["num_hidden_layers"])
        self.query_heads = int(self.config["num_attention_heads"])
        self.kv_heads = int(self.config["num_key_value_heads"])
        self.head_size = self.hidden_size // self.query_heads
        self.epsilon = float(self.config.get("rms_norm_eps", 1.0e-5))
        model_path = Path(model_path)
        if (model_path / "adapter_config.json").is_file() and not (
            model_path / "model.safetensors"
        ).is_file():
            raise ValueError(
                "LLM2Vec model_path must be the full base Llama checkpoint; "
                "pass MNTP and supervised PEFT directories as adapter_paths"
            )
        archive = SafeTensorArchive(model_path)
        names = llama_encoder_weight_names(self.config)
        self.weights = load_cast_weights(archive, names, self.device, dtype)
        for adapter in adapter_paths:
            merge_lora_adapter(self.weights, adapter, dtype, self.device)
        self.tokenizer = Llama3Tokenizer(model_path)
        self.cublas = None
        if use_cublas:
            from ._cublas import Cublas

            self.cublas = Cublas(self.device)
        parameters = {
            "rope_theta": float(self.config.get("rope_theta", 500000.0)),
            "partial_rotary_factor": 1.0,
            "rope_scaling": None,
        }
        cos, sin = rotary_cache_values(
            int(self.config.get("max_position_embeddings", 8192)),
            self.head_size,
            parameters,
        )
        self.cos_cache = wp.array(cos, dtype=dtype, device=self.device)
        self.sin_cache = wp.array(sin, dtype=dtype, device=self.device)
        self._plans = {}

    def encode(self, text):
        text = text.strip()
        prepared = f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
        ids = [int(self.config.get("bos_token_id", 128000))]
        ids.extend(self.tokenizer.encode(prepared))
        content = self.tokenizer.encode(f"{text}<|eot_id|>")
        if not ids or not content or len(ids) > 512:
            raise ValueError("LLM2Vec text must contain 1-512 tokens")
        sequence = len(ids)
        plan = self._plans.get(sequence)
        if plan is None:
            plan = _LlamaEncoderPlan(self, sequence)
            self._plans[sequence] = plan
        plan.input_ids.assign(np.asarray(ids, dtype=np.int64)[None])
        plan.valid.assign(np.ones((1, sequence), dtype=bool))
        embed_mask = np.zeros(sequence, dtype=bool)
        embed_mask[-len(content) :] = True
        plan.embed_mask.assign(embed_mask)
        return plan.run()
