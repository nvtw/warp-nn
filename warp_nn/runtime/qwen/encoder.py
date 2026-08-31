# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free causal Qwen3 hidden-state and embedding executor."""

from __future__ import annotations

import json
import weakref
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import warp as wp

from .._cublas import try_create_cublas
from ..formats.safetensors import SafeTensorArchive
from ..kernels import (
    _gather_rows_kernel,
    _get_gqa_attention_kernel,
    _reorder_heads_kernel,
    _rotary_embedding_kernel_for_dtype,
)
from ..operators import (
    Operation,
    execute_operations,
    plan_linear,
    plan_residual_rms_norm,
    plan_rms_norm,
    plan_swiglu,
    reuse_linear_outputs,
    rotary_cache_values,
)
from ..tokenizers import Qwen3Tokenizer
from ..weights import MappedWeightArchive, load_cast_weights
from ...utils.device import parse_device


def qwen_encoder_weight_names(config: dict) -> tuple[str, ...]:
    """Return weights used by a supported Qwen language backbone."""
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    qk_norm = bool(config.get("qk_norm", config.get("model_type") == "qwen3"))
    attention_bias = bool(config.get("attention_bias", False))
    for index in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{index}."
        suffixes = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        if qk_norm:
            suffixes += ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
        if attention_bias:
            suffixes += [
                "self_attn.q_proj.bias",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.bias",
            ]
        names.extend(prefix + suffix for suffix in suffixes)
    return tuple(names)


def qwen3_encoder_weight_names(config: dict) -> tuple[str, ...]:
    """Return the exact weights used by a standard Qwen3 base model."""
    return qwen_encoder_weight_names(config)


def load_qwen_encoder_config(path: str | Path) -> dict:
    """Load a Qwen3 or language-only Qwen2.5-VL shape contract."""
    config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    if "text_config" in config:
        config = dict(config["text_config"])
    required = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Qwen encoder config is missing {missing}")
    if config.get("model_type") not in ("qwen3", "qwen2_5_vl"):
        raise ValueError("Qwen encoder requires Qwen3 or Qwen2.5-VL text config")
    layers = int(config["num_hidden_layers"])
    layer_types = config.get("layer_types", ["full_attention"] * layers)
    if len(layer_types) != layers or set(layer_types) != {"full_attention"}:
        raise ValueError("Qwen encoder supports full-attention layers only")
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("Qwen query heads must be a positive multiple of KV heads")
    if "head_dim" in config:
        head_dim = int(config["head_dim"])
    else:
        hidden = int(config["hidden_size"])
        if hidden % query_heads:
            raise ValueError("Qwen hidden size must divide evenly across query heads")
        head_dim = hidden // query_heads
    if head_dim <= 0:
        raise ValueError("Qwen head geometry is inconsistent")
    config["head_dim"] = head_dim
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("Qwen encoder requires SiLU-gated MLPs")
    rope_scaling = config.get("rope_scaling")
    if rope_scaling not in (None, {}):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if config.get("model_type") != "qwen2_5_vl" or rope_type != "mrope":
            raise ValueError("Qwen encoder requires default or text-only M-RoPE")
    return config


def load_qwen3_encoder_config(path: str | Path) -> dict:
    """Load and validate the standard full-attention Qwen3 shape contract."""
    path = Path(path)
    raw_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if "head_dim" not in raw_config:
        raise ValueError("Qwen3 encoder config is missing ['head_dim']")
    config = load_qwen_encoder_config(path)
    required = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Qwen3 encoder config is missing {missing}")
    if config.get("model_type") != "qwen3":
        raise ValueError("Qwen3 encoder requires model_type 'qwen3'")
    layers = int(config["num_hidden_layers"])
    layer_types = config.get("layer_types", ["full_attention"] * layers)
    if len(layer_types) != layers or set(layer_types) != {"full_attention"}:
        raise ValueError("Qwen3 encoder supports full-attention layers only")
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("Qwen3 query heads must be a positive multiple of KV heads")
    if int(config["head_dim"]) <= 0:
        raise ValueError("Qwen3 head_dim must be positive")
    if (
        config.get("attention_bias", False)
        or config.get("hidden_act", "silu") != "silu"
    ):
        raise ValueError("only bias-free SiLU Qwen3 encoders are supported")
    if config.get("rope_scaling") not in (None, {}):
        raise ValueError("Qwen3 encoder currently supports default RoPE only")
    return config


@lru_cache(maxsize=None)
def _projection_bias_kernel(dtype):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def add_bias(values: wp.array2d(dtype=DTYPE), bias: wp.array1d(dtype=DTYPE)):
        row, column = wp.tid()
        values[row, column] = DTYPE(
            wp.float32(values[row, column]) + wp.float32(bias[column])
        )

    return add_bias


class _Qwen3EncoderPlan:
    """One fixed-length causal execution plan sharing immutable weights."""

    def __init__(self, runner: Qwen3Encoder, sequence: int):
        self.runner = weakref.proxy(runner)
        self.sequence = sequence
        self.device = runner.device
        self.dtype = runner.dtype
        self.input_ids = wp.empty((1, sequence), dtype=wp.int64, device=self.device)
        self.position_ids = wp.array(
            np.arange(sequence, dtype=np.int64)[None, :], device=self.device
        )
        self.sequence_end = wp.array(
            np.asarray([sequence - 1], dtype=np.int32), device=self.device
        )
        self.embedding = wp.empty(
            (1, sequence, runner.hidden_size), dtype=self.dtype, device=self.device
        )
        self.tensors = dict(runner.weights)
        self.shapes = {name: tuple(value.shape) for name, value in self.tensors.items()}
        self.tensors["hidden.0"] = self.embedding.reshape(
            (sequence, runner.hidden_size)
        )
        self.shapes["hidden.0"] = (sequence, runner.hidden_size)
        self.layers = []
        self._linear_pool = {}
        self._allocate_attention_buffers()
        self._build()
        self.graph = None
        self._capture_ready = False

    def _allocate_attention_buffers(self) -> None:
        runner = self.runner
        q_shape = (runner.query_heads * self.sequence, runner.head_dim)
        kv_shape = (runner.kv_heads * self.sequence, runner.head_dim)
        self.query = wp.empty(q_shape, dtype=self.dtype, device=self.device)
        self.key = wp.empty(kv_shape, dtype=self.dtype, device=self.device)
        self.value = wp.empty(kv_shape, dtype=self.dtype, device=self.device)
        self.query_rotated = wp.empty_like(self.query)
        self.key_rotated = wp.empty_like(self.key)
        self.attention = wp.empty(
            (self.sequence, runner.query_heads * runner.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.attention_block, self.attention_kernel = _get_gqa_attention_kernel(
            runner.head_dim, self.dtype
        )

    def _linear(
        self, name: str, source: str, weight: str, bias: str | None = None
    ) -> Operation:
        op = Operation("Linear", [source, weight], [name])
        plan_linear(
            op,
            self.tensors,
            self.shapes,
            self.device,
            cublas=self.runner.cublas,
        )
        op.attrs["_sequence"] = (op,)
        op.attrs["_bias"] = bias
        return op

    def _rms(self, name: str, source: str, scale: str) -> Operation:
        op = Operation(
            "SimplifiedLayerNormalization",
            [source, scale],
            [name],
            {"epsilon": self.runner.epsilon},
        )
        plan_rms_norm(op, self.tensors, self.shapes, self.device)
        op.attrs["_sequence"] = (op,)
        return op

    def _residual_rms(
        self,
        name: str,
        source: str,
        residual: str,
        scale: str,
        residual_name: str,
    ) -> Operation:
        op = Operation(
            "SkipSimplifiedLayerNormalization",
            [source, residual, scale],
            [name, "", "", residual_name],
            {"epsilon": self.runner.epsilon},
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
        hidden = "hidden.0"
        normalized = "layer.0.input"
        self.first_norm = self._rms(
            normalized, hidden, "model.layers.0.input_layernorm.weight"
        )
        for index in range(self.runner.layers):
            prefix = f"model.layers.{index}."
            attention = prefix + "self_attn."
            layer = {"hidden": hidden}
            for projection in ("q", "k", "v"):
                layer[projection] = self._linear(
                    f"layer.{index}.{projection}",
                    normalized,
                    attention + f"{projection}_proj.weight",
                    (
                        attention + f"{projection}_proj.bias"
                        if self.runner.attention_bias
                        else None
                    ),
                )
            self.tensors[f"layer.{index}.query"] = self.query
            self.shapes[f"layer.{index}.query"] = tuple(self.query.shape)
            self.tensors[f"layer.{index}.key"] = self.key
            self.shapes[f"layer.{index}.key"] = tuple(self.key.shape)
            if self.runner.qk_norm:
                layer["q_norm"] = self._rms(
                    f"layer.{index}.q_norm",
                    f"layer.{index}.query",
                    attention + "q_norm.weight",
                )
                layer["k_norm"] = self._rms(
                    f"layer.{index}.k_norm",
                    f"layer.{index}.key",
                    attention + "k_norm.weight",
                )
            else:
                layer["q_norm"] = layer["k_norm"] = None
            self.tensors[f"layer.{index}.attention"] = self.attention
            self.shapes[f"layer.{index}.attention"] = tuple(self.attention.shape)
            layer["output"] = self._linear(
                f"layer.{index}.attention_output",
                f"layer.{index}.attention",
                attention + "o_proj.weight",
            )
            residual = f"layer.{index}.attention_residual"
            mlp_input = f"layer.{index}.mlp_input"
            layer["post_norm"] = self._residual_rms(
                mlp_input,
                layer["output"].outputs[0],
                hidden,
                prefix + "post_attention_layernorm.weight",
                residual,
            )
            for projection in ("gate", "up"):
                layer[projection] = self._linear(
                    f"layer.{index}.mlp_{projection}",
                    mlp_input,
                    prefix + f"mlp.{projection}_proj.weight",
                )
            layer["swiglu"] = self._swiglu(
                f"layer.{index}.mlp_hidden",
                layer["gate"].outputs[0],
                layer["up"].outputs[0],
            )
            layer["down"] = self._linear(
                f"layer.{index}.mlp_output",
                layer["swiglu"].outputs[0],
                prefix + "mlp.down_proj.weight",
            )
            hidden = f"hidden.{index + 1}"
            if index + 1 < self.runner.layers:
                normalized = f"layer.{index + 1}.input"
                scale = f"model.layers.{index + 1}.input_layernorm.weight"
            else:
                normalized = "final.normalized"
                scale = "model.norm.weight"
            layer["next_norm"] = self._residual_rms(
                normalized,
                layer["down"].outputs[0],
                residual,
                scale,
                hidden,
            )
            reuse_linear_outputs(layer, self.tensors, self._linear_pool)
            self.layers.append(layer)
        self.output = self.tensors[normalized].reshape(
            (1, self.sequence, self.runner.hidden_size)
        )

    def _execute(self, operation: Operation) -> None:
        execute_operations(
            operation.attrs["_sequence"], self.tensors, self.shapes, self.device
        )
        bias = operation.attrs.get("_bias")
        if bias is not None:
            output = self.tensors[operation.outputs[0]]
            wp.launch(
                _projection_bias_kernel(self.dtype),
                dim=output.shape,
                inputs=[output, self.runner.weights[bias]],
                device=self.device,
            )

    def execute(self) -> wp.array:
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
        self._execute(self.first_norm)
        rotary = _rotary_embedding_kernel_for_dtype(self.dtype)
        for layer in self.layers:
            for name in ("q", "k", "v"):
                self._execute(layer[name])
            for name, target, heads in (
                ("q", self.query, self.runner.query_heads),
                ("k", self.key, self.runner.kv_heads),
                ("v", self.value, self.runner.kv_heads),
            ):
                wp.launch(
                    _reorder_heads_kernel,
                    dim=(self.sequence, heads, self.runner.head_dim),
                    inputs=[
                        self.tensors[layer[name].outputs[0]],
                        target,
                        self.runner.head_dim,
                    ],
                    device=self.device,
                )
            if self.runner.qk_norm:
                self._execute(layer["q_norm"])
                self._execute(layer["k_norm"])
                query = self.tensors[layer["q_norm"].outputs[0]]
                key = self.tensors[layer["k_norm"].outputs[0]]
            else:
                query = self.query
                key = self.key
            for source, output, heads in (
                (query, self.query_rotated, self.runner.query_heads),
                (key, self.key_rotated, self.runner.kv_heads),
            ):
                wp.launch(
                    rotary,
                    dim=(1, heads, self.sequence, self.runner.head_dim),
                    inputs=[
                        source.reshape((1, heads, self.sequence, self.runner.head_dim)),
                        self.position_ids,
                        self.runner.cos_cache,
                        self.runner.sin_cache,
                        output.reshape((1, heads, self.sequence, self.runner.head_dim)),
                        self.runner.head_dim,
                        False,
                        False,
                    ],
                    device=self.device,
                )
            wp.launch_tiled(
                self.attention_kernel,
                dim=self.runner.query_heads * self.sequence,
                inputs=[
                    self.query_rotated,
                    self.key_rotated,
                    self.value,
                    self.sequence_end,
                    self.attention,
                    self.runner.query_heads,
                    self.runner.kv_heads,
                    self.sequence,
                    self.sequence,
                    self.runner.head_dim**-0.5,
                    0,
                ],
                block_dim=self.attention_block,
                device=self.device,
            )
            for name in (
                "output",
                "post_norm",
                "gate",
                "up",
                "swiglu",
                "down",
                "next_norm",
            ):
                self._execute(layer[name])
        return self.output

    def run(self) -> wp.array:
        """Execute eagerly once, then capture and replay this fixed token length."""
        if not self.device.is_cuda:
            return self.execute()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        elif self._capture_ready:
            wp.capture_begin(device=self.device)
            try:
                output = self.execute()
                self.graph = wp.capture_end(device=self.device)
            except Exception:
                wp.capture_end(device=self.device)
                raise
            wp.capture_launch(self.graph)
            return output
        else:
            self.execute()
            self._capture_ready = True
        return self.output


class QwenEncoder:
    """Return final hidden states or embeddings from a Qwen language backbone.

    Caption encoding is causal, matching Hugging Face ``Qwen3Model``.  Lyric
    conditioning can call :meth:`embed_ids` to bypass all transformer layers,
    exactly as ACE-Step 1.5 does before its own lyric encoder.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dtype=wp.bfloat16,
        device=None,
        use_cublas: bool = True,
        tokenizer_path: str | Path | None = None,
    ):
        path = Path(path)
        self.config = self.config_loader(path)
        self.device = parse_device(device)
        self.dtype = dtype
        if dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Qwen3 encoder dtype must be FP16 or BF16")
        self.hidden_size = int(self.config["hidden_size"])
        self.layers = int(self.config["num_hidden_layers"])
        self.query_heads = int(self.config["num_attention_heads"])
        self.kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config["head_dim"])
        self.epsilon = float(self.config.get("rms_norm_eps", 1.0e-6))
        self.qk_norm = bool(
            self.config.get("qk_norm", self.config.get("model_type") == "qwen3")
        )
        self.attention_bias = bool(self.config.get("attention_bias", False))
        archive = SafeTensorArchive(path)
        names = qwen_encoder_weight_names(self.config)
        missing = set(names) - set(archive.names)
        if missing and all(
            name.removeprefix("model.") in archive.names for name in names
        ):
            archive = MappedWeightArchive(
                archive, {name: name.removeprefix("model.") for name in names}
            )
            missing = set()
        if missing:
            raise ValueError(
                f"Qwen encoder checkpoint is missing {sorted(missing)[:5]}"
            )
        self.weights = load_cast_weights(archive, names, self.device, dtype)
        self.tokenizer = Qwen3Tokenizer(
            path if tokenizer_path is None else tokenizer_path
        )
        self.cublas = (
            try_create_cublas() if use_cublas and self.device.is_cuda else None
        )
        maximum = int(self.config["max_position_embeddings"])
        cos, sin = rotary_cache_values(
            maximum,
            self.head_dim,
            {
                "rope_theta": float(self.config.get("rope_theta", 1_000_000.0)),
                "rope_type": "default",
            },
        )
        self.cos_cache = wp.array(cos, dtype=dtype, device=self.device)
        self.sin_cache = wp.array(sin, dtype=dtype, device=self.device)
        self._plans = {}

    config_loader = staticmethod(load_qwen_encoder_config)

    def _plan(self, sequence: int) -> _Qwen3EncoderPlan:
        maximum = int(self.config["max_position_embeddings"])
        if sequence <= 0 or sequence > maximum:
            raise ValueError(f"Qwen sequence length must be between 1 and {maximum}")
        plan = self._plans.get(sequence)
        if plan is None:
            plan = self._plans[sequence] = _Qwen3EncoderPlan(self, sequence)
        return plan

    def encode_ids(self, token_ids: Sequence[int]) -> wp.array:
        """Return ``[1, sequence, hidden]`` final hidden states."""
        values = np.asarray(token_ids, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError("Qwen token IDs must be one-dimensional")
        if values.size == 0:
            raise ValueError("Qwen token IDs must not be empty")
        if values.min() < 0 or values.max() >= int(self.config["vocab_size"]):
            raise ValueError("Qwen token ID is outside the vocabulary")
        plan = self._plan(values.size)
        plan.input_ids.assign(values[None, :])
        return plan.run()

    def encode(self, text: str, *, max_tokens: int = 256) -> wp.array:
        """Tokenize and return the full causal caption hidden-state sequence."""
        token_ids = self.tokenizer.encode(text)[:max_tokens]
        if not token_ids:
            raise ValueError("Qwen3 caption must not be empty")
        return self.encode_ids(token_ids)

    def embed_ids(self, token_ids: Sequence[int] | np.ndarray) -> wp.array:
        """Gather input embeddings for lyrics without running Qwen layers."""
        values = np.asarray(token_ids, dtype=np.int64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.size == 0:
            raise ValueError(
                "Qwen3 lyric token IDs must be a nonempty rank-1 or rank-2 array"
            )
        if values.min() < 0 or values.max() >= int(self.config["vocab_size"]):
            raise ValueError("Qwen3 token ID is outside the vocabulary")
        indices = wp.array(values, dtype=wp.int64, device=self.device)
        output = wp.empty(
            (*values.shape, self.hidden_size), dtype=self.dtype, device=self.device
        )
        wp.launch(
            _gather_rows_kernel,
            dim=output.shape,
            inputs=[self.weights["model.embed_tokens.weight"], indices, output],
            device=self.device,
        )
        return output


class Qwen3Encoder(QwenEncoder):
    """Strict standard Qwen3 encoder API used by ACE-Step."""

    config_loader = staticmethod(load_qwen3_encoder_config)
