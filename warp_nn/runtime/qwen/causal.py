# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native autoregressive execution for standard dense Qwen3 checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import warp as wp

from .._cublas import try_create_cublas
from ..autoregressive import AutoregressiveRunner
from ..formats.safetensors import SafeTensorArchive
from ..kernels import (
    _append_head_cache_kernel,
    _gather_rows_kernel,
    _reorder_heads_kernel,
    _rotary_embedding_kernel_for_dtype,
)
from ..operators import Operation, execute_operations, plan_linear, rotary_cache_values
from ..tokenizers import Qwen3Tokenizer
from ..weights import load_cast_weights
from ...utils.device import parse_device
from .encoder import (
    _Qwen3EncoderPlan,
    load_qwen3_encoder_config,
    qwen3_encoder_weight_names,
)


def qwen3_causal_weight_names(config: dict, available=()) -> tuple[str, ...]:
    """Return the dense backbone and optional untied output projection."""
    names = list(qwen3_encoder_weight_names(config))
    if "lm_head.weight" in available:
        names.append("lm_head.weight")
    elif not bool(config.get("tie_word_embeddings", True)):
        raise ValueError("untied Qwen3 checkpoints require lm_head.weight")
    return tuple(names)


class _Qwen3CausalPlan(_Qwen3EncoderPlan):
    """One fixed-row plan writing K/V into the runner's persistent cache."""

    def __init__(self, runner: Qwen3CausalLM, rows: int):
        super().__init__(runner, rows)
        self.rows = rows
        self.sequence_end = runner.sequence_end
        final = self.layers[-1]["next_norm"].outputs[0]
        self.tensors["final.last"] = self.tensors[final][rows - 1 : rows]
        self.shapes["final.last"] = (1, runner.hidden_size)
        self.lm_head = Operation("Linear", ["final.last", "lm_head.weight"], ["logits"])
        plan_linear(
            self.lm_head,
            self.tensors,
            self.shapes,
            self.device,
            cublas=runner.cublas,
        )
        self.logits = self.tensors["logits"].reshape(
            (1, 1, int(runner.config["vocab_size"]))
        )

    def execute(self) -> wp.array:
        runner = self.runner
        wp.launch(
            _gather_rows_kernel,
            dim=self.embedding.shape,
            inputs=[
                runner.weights["model.embed_tokens.weight"],
                self.input_ids,
                self.embedding,
            ],
            device=self.device,
        )
        self._execute(self.first_norm)
        rotary = _rotary_embedding_kernel_for_dtype(self.dtype)
        for index, layer in enumerate(self.layers):
            for name in ("q", "k", "v"):
                self._execute(layer[name])
            for name, target, heads in (
                ("q", self.query, runner.query_heads),
                ("k", self.key, runner.kv_heads),
                ("v", self.value, runner.kv_heads),
            ):
                wp.launch(
                    _reorder_heads_kernel,
                    dim=(self.rows, heads, runner.head_dim),
                    inputs=[
                        self.tensors[layer[name].outputs[0]],
                        target,
                        runner.head_dim,
                    ],
                    device=self.device,
                )
            if runner.qk_norm:
                self._execute(layer["q_norm"])
                self._execute(layer["k_norm"])
                query = self.tensors[layer["q_norm"].outputs[0]]
                key = self.tensors[layer["k_norm"].outputs[0]]
            else:
                query, key = self.query, self.key
            for source, output, heads in (
                (query, self.query_rotated, runner.query_heads),
                (key, self.key_rotated, runner.kv_heads),
            ):
                wp.launch(
                    rotary,
                    dim=(1, heads, self.rows, runner.head_dim),
                    inputs=[
                        source.reshape((1, heads, self.rows, runner.head_dim)),
                        self.position_ids,
                        runner.cos_cache,
                        runner.sin_cache,
                        output.reshape((1, heads, self.rows, runner.head_dim)),
                        runner.head_dim,
                        False,
                        False,
                    ],
                    device=self.device,
                )
            key_cache, value_cache = runner.kv_caches[index]
            for source, cache in (
                (self.key_rotated, key_cache),
                (self.value, value_cache),
            ):
                wp.launch(
                    _append_head_cache_kernel,
                    dim=(runner.kv_heads, self.rows, runner.head_dim),
                    inputs=[
                        source,
                        self.position_ids,
                        cache,
                        runner.kv_heads,
                        runner.head_dim,
                    ],
                    device=self.device,
                )
            wp.launch_tiled(
                self.attention_kernel,
                dim=runner.query_heads * self.rows,
                inputs=[
                    self.query_rotated,
                    key_cache,
                    value_cache,
                    runner.sequence_end,
                    self.attention,
                    runner.query_heads,
                    runner.kv_heads,
                    self.rows,
                    runner.cache_capacity,
                    runner.head_dim**-0.5,
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
        execute_operations((self.lm_head,), self.tensors, self.shapes, self.device)
        return self.logits


class Qwen3CausalLM(AutoregressiveRunner):
    """Dependency-free dense Qwen3 prefill/decode with a persistent KV cache."""

    def __init__(
        self,
        path: str | Path,
        *,
        dtype=wp.bfloat16,
        device=None,
        cache_capacity: int = 4096,
        prefill_chunk_size: int = 16,
        use_cublas: bool = True,
    ):
        path = Path(path)
        self.config = load_qwen3_encoder_config(path)
        maximum = int(self.config["max_position_embeddings"])
        if not 1 < cache_capacity <= maximum:
            raise ValueError("Qwen3 cache_capacity must be between 2 and model maximum")
        if not 1 <= prefill_chunk_size <= cache_capacity:
            raise ValueError("Qwen3 prefill_chunk_size must fit in cache_capacity")
        if dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Qwen3 causal activations require FP16 or BF16")
        self.model_path = path
        self.device = parse_device(device)
        self.dtype = dtype
        self.cache_capacity = int(cache_capacity)
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.hidden_size = int(self.config["hidden_size"])
        self.layers = int(self.config["num_hidden_layers"])
        self.query_heads = int(self.config["num_attention_heads"])
        self.kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config["head_dim"])
        self.epsilon = float(self.config.get("rms_norm_eps", 1.0e-6))
        self.qk_norm = True
        self.attention_bias = False
        archive = SafeTensorArchive(path)
        names = qwen3_causal_weight_names(self.config, archive.names)
        missing = set(names) - set(archive.names)
        if missing:
            raise ValueError(f"Qwen3 checkpoint is missing {sorted(missing)[:5]}")
        self.weights = load_cast_weights(archive, names, self.device, dtype)
        if "lm_head.weight" not in self.weights:
            self.weights["lm_head.weight"] = self.weights["model.embed_tokens.weight"]
        self.tokenizer = Qwen3Tokenizer(path)
        self.cublas = (
            try_create_cublas() if use_cublas and self.device.is_cuda else None
        )
        self.sequence_end = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.conv_states = {}
        self.recurrent_states = {}
        cache_shape = (self.kv_heads * self.cache_capacity, self.head_dim)
        self.kv_caches = {
            index: (
                wp.empty(cache_shape, dtype=dtype, device=self.device),
                wp.empty(cache_shape, dtype=dtype, device=self.device),
            )
            for index in range(self.layers)
        }
        cos, sin = rotary_cache_values(
            self.cache_capacity,
            self.head_dim,
            {
                "rope_theta": float(self.config.get("rope_theta", 1_000_000.0)),
                "rope_type": "default",
            },
        )
        self.cos_cache = wp.array(cos, dtype=dtype, device=self.device)
        self.sin_cache = wp.array(sin, dtype=dtype, device=self.device)
        self._decode_plan = _Qwen3CausalPlan(self, 1)
        self._chunk_plan = _Qwen3CausalPlan(self, self.prefill_chunk_size)
        self._chunk_plan._capture_ready = False
        self._record_plan_storage(self._decode_plan)
        self._record_plan_storage(self._chunk_plan)
        self._initialize_sampling()
        self.sequence_length = 0
        self.rope_delta = 0

    def _validate_ids(self, token_ids: Sequence[int]) -> None:
        values = np.asarray(token_ids)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                "Qwen3 token IDs must be a nonempty one-dimensional sequence"
            )
        if values.min() < 0 or values.max() >= int(self.config["vocab_size"]):
            raise ValueError("Qwen3 token ID is outside the vocabulary")

    def prefill(self, token_ids: Sequence[int]) -> wp.array:
        self._validate_ids(token_ids)
        return super().prefill(token_ids)

    def append(self, token_ids: Sequence[int]) -> wp.array:
        self._validate_ids(token_ids)
        return super().append(token_ids)

    def decode(self, token_id: int) -> wp.array:
        self._validate_ids((token_id,))
        return super().decode(token_id)

    def sample_greedy_range(self, logits: wp.array, start: int, stop: int) -> int:
        """Select the largest logit in one contiguous token interval."""
        vocabulary = int(self.config["vocab_size"])
        if not 0 <= start < stop <= vocabulary:
            raise ValueError("Qwen3 token interval is outside the vocabulary")
        if (
            logits.device != self.device
            or logits.dtype != self.dtype
            or logits.ndim != 3
        ):
            raise TypeError("sample_greedy_range expects runner logits")
        values = logits.flatten()[start:stop].reshape((1, 1, stop - start))
        wp.launch_tiled(
            self._greedy_argmax_kernels[0],
            dim=128,
            inputs=[values, self._sample_partial_values, self._sample_partial_tokens],
            block_dim=256,
            device=self.device,
        )
        wp.launch_tiled(
            self._greedy_argmax_kernels[1],
            dim=1,
            inputs=[
                self._sample_partial_values,
                self._sample_partial_tokens,
                self._sampled_token,
                stop - start,
            ],
            block_dim=128,
            device=self.device,
        )
        wp.copy(self._sampled_token_host, self._sampled_token, count=1)
        wp.synchronize_stream(self.device)
        return start + int(self._sampled_token_host_view[0])

    def read_top_k_range(
        self, logits: wp.array, start: int, stop: int, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read bounded candidates from one contiguous token interval."""
        return self.read_top_k(logits, top_k, token_start=start, token_stop=stop)
