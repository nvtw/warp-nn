# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal stateful runner for Qwen-style decoder ONNX graphs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import warp as wp

from warp_nn.runtime.chat import sample_token
from warp_nn.runtime.kernels import (
    _get_greedy_argmax_kernels,
    _initialize_attention_mask,
    _initialize_generation_state,
    _set_decode_token,
)
from warp_nn.runtime.onnx_runtime import OnnxRuntime


class Qwen3OnnxRunner:
    """Run prefill and token-by-token decode while keeping weights on device."""

    def __init__(
        self,
        path: str,
        device: str | wp.Device | None = None,
        cache_capacity: int | None = None,
        prefill_chunk_size: int | None = None,
        use_cublas: bool = True,
    ):
        self.runtime = OnnxRuntime(
            path, device=device, use_cublas=use_cublas, _defer_preallocation=True
        )
        self._past_names = [
            name
            for name in self.runtime.input_names
            if name.startswith("past_key_values.")
        ]
        self._present_for_past = {
            name: f"present.{name.split('.')[1]}.{name.split('.')[2]}"
            for name in self._past_names
        }
        if not self._past_names or any(
            name not in self.runtime.output_names
            for name in self._present_for_past.values()
        ):
            raise ValueError(
                "Qwen3OnnxRunner: model does not expose compatible past/present KV-cache tensors"
            )
        self._cache_shapes = {
            name: self.runtime._shapes[name] for name in self._past_names
        }
        self._variable_cache_names = {
            name
            for name in self._past_names
            if name.endswith(".key") or name.endswith(".value")
        }
        rotary_lengths = [
            self.runtime._shapes[name][0]
            for name in ("cos_cache", "sin_cache")
            if name in self.runtime._shapes
        ]
        config_path = Path(path).with_name("config.json")
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        text_config = config.get("text_config", config)
        configured_length = int(text_config.get("max_position_embeddings", 0))
        if not rotary_lengths and not configured_length:
            raise ValueError(
                "Qwen3OnnxRunner: model does not declare a maximum sequence length"
            )
        self.max_sequence_length = (
            min(rotary_lengths) if rotary_lengths else configured_length
        )
        self.cache_capacity = cache_capacity or self.max_sequence_length
        if not 0 < self.cache_capacity <= self.max_sequence_length:
            raise ValueError(
                "Qwen3OnnxRunner: cache_capacity must be within the model's rotary cache"
            )
        if (
            prefill_chunk_size is not None
            and not 1 < prefill_chunk_size <= self.cache_capacity
        ):
            raise ValueError(
                "Qwen3OnnxRunner: prefill_chunk_size must be between 2 and cache_capacity"
            )
        self.prefill_chunk_size = prefill_chunk_size
        self._cache = {
            name: wp.zeros(
                (1, shape[1], self.cache_capacity, shape[3])
                if name in self._variable_cache_names
                else shape,
                dtype=self.runtime._input_dtypes[name],
                device=self.runtime._device,
            )
            for name, shape in self._cache_shapes.items()
        }
        self._decode_input_ids = wp.zeros(
            (1, 1), dtype=wp.int64, device=self.runtime._device
        )
        self._decode_position_ids = wp.zeros(
            (1, 1), dtype=wp.int64, device=self.runtime._device
        )
        self._decode_attention_mask = wp.zeros(
            (1, self.cache_capacity), dtype=wp.int64, device=self.runtime._device
        )
        decode_shapes = {
            "input_ids": (1, 1),
            "attention_mask": (1, self.cache_capacity),
        }
        if "position_ids" in self.runtime.input_names:
            decode_shapes["position_ids"] = (1, 1)
        decode_shapes.update(
            {name: tuple(cache.shape) for name, cache in self._cache.items()}
        )
        self._decode_runtime = self.runtime._fork(decode_shapes, share_kv_cache=True)
        self._decode_inputs = {
            "input_ids": self._decode_input_ids,
            "attention_mask": self._decode_attention_mask,
            **self._cache,
        }
        if "position_ids" in self.runtime.input_names:
            self._decode_inputs["position_ids"] = self._decode_position_ids
        self._chunk_runtimes = {}
        self._chunk_input_ids = {}
        self._chunk_position_ids = {}
        self._chunk_inputs = {}
        if prefill_chunk_size is not None:
            chunk_sizes = (
                (prefill_chunk_size, 4)
                if prefill_chunk_size > 4
                else (prefill_chunk_size,)
            )
            for chunk_size in chunk_sizes:
                self._add_chunk_runtime(chunk_size)
        self._decode_position = wp.zeros(1, dtype=wp.int32, device=self.runtime._device)
        self._generated_count = wp.zeros(1, dtype=wp.int32, device=self.runtime._device)
        self._generated_ids = wp.zeros(
            self.cache_capacity, dtype=wp.int64, device=self.runtime._device
        )
        self._generation_finished = wp.zeros(
            1, dtype=wp.int32, device=self.runtime._device
        )
        self._sample_partial_values = wp.empty(
            128, dtype=wp.float32, device=self.runtime._device
        )
        self._sample_partial_tokens = wp.empty(
            128, dtype=wp.int32, device=self.runtime._device
        )
        self._sampled_token = wp.empty(1, dtype=wp.int32, device=self.runtime._device)
        self._sampled_token_host = wp.empty(
            1, dtype=wp.int32, device="cpu", pinned=self.runtime._device.is_cuda
        )
        self._sampled_token_host_view = self._sampled_token_host.numpy()
        self._greedy_argmax_kernels = _get_greedy_argmax_kernels(1024, 128)
        self._decode_graph = None
        self._decode_graph_outputs = None
        self._generation_graph = None
        self._generation_graph_eos = None
        self._past: dict[str, wp.array] = {}
        self.sequence_length = 0

    def reset(self) -> None:
        """Discard the current conversation's KV cache."""
        self._past.clear()
        self.sequence_length = 0

    def prefill(self, token_ids: Sequence[int]) -> wp.array:
        """Reset state, process a prompt, and return its logits."""
        self.reset()
        current_length = len(token_ids)
        if current_length == 0:
            raise ValueError("Qwen3OnnxRunner: token_ids must not be empty")
        if current_length >= self.cache_capacity:
            raise ValueError(
                "Qwen3OnnxRunner: prompt must leave room for at least one decoded token"
            )
        if self._chunk_runtimes and current_length >= min(self._chunk_runtimes):
            return self._prefill_chunked(token_ids)
        shapes = {
            "input_ids": (1, current_length),
            "attention_mask": (1, current_length),
        }
        if "position_ids" in self.runtime.input_names:
            shapes["position_ids"] = (1, current_length)
        for name, base_shape in self._cache_shapes.items():
            shapes[name] = (
                (1, base_shape[1], 0, base_shape[3])
                if name in self._variable_cache_names
                else base_shape
            )
        self.runtime.resize_inputs(shapes)
        inputs = {
            "input_ids": wp.array(
                np.asarray(token_ids, dtype=np.int64)[None, :],
                dtype=wp.int64,
                device=self.runtime._device,
            ),
            "attention_mask": wp.ones(
                (1, current_length), dtype=wp.int64, device=self.runtime._device
            ),
        }
        if "position_ids" in self.runtime.input_names:
            inputs["position_ids"] = wp.array(
                np.arange(current_length, dtype=np.int64)[None, :],
                dtype=wp.int64,
                device=self.runtime._device,
            )
        for name, shape in shapes.items():
            if name not in inputs:
                inputs[name] = wp.zeros(
                    shape,
                    dtype=self.runtime._input_dtypes[name],
                    device=self.runtime._device,
                )
        outputs = self.runtime(inputs)
        for name, destination in self._cache.items():
            source = outputs[self._present_for_past[name]]
            wp.copy(destination.flatten(), source.flatten(), count=source.size)
        self.sequence_length = current_length
        self._prepare_decode()
        return outputs["logits"]

    def _prefill_chunked(self, token_ids: Sequence[int]) -> wp.array:
        """Prefill through bounded fixed-size chunks and return the final logits."""
        for cache in self._cache.values():
            cache.zero_()
        wp.launch(
            _initialize_attention_mask,
            dim=self.cache_capacity,
            inputs=[self._decode_attention_mask, 0],
            device=self.runtime._device,
        )
        self.sequence_length = 0
        return self._append(token_ids)

    def append(self, token_ids: Sequence[int]) -> wp.array:
        """Process new prompt tokens while retaining the existing KV cache."""
        if self.sequence_length == 0:
            raise RuntimeError(
                "Qwen3OnnxRunner.append requires a preceding prefill call"
            )
        return self._append(token_ids)

    def _append(self, token_ids: Sequence[int]) -> wp.array:
        if not token_ids:
            raise ValueError("Qwen3OnnxRunner.append requires at least one token")
        if self.sequence_length + len(token_ids) > self.cache_capacity:
            raise ValueError(
                "Qwen3OnnxRunner: appended tokens exceed the KV-cache capacity"
            )

        outputs = None
        consumed = 0
        for chunk_size in sorted(self._chunk_runtimes, reverse=True):
            chunk_count = (len(token_ids) - consumed) // chunk_size
            for _ in range(chunk_count):
                position = self.sequence_length
                end = position + chunk_size
                chunk = token_ids[consumed : consumed + chunk_size]
                self._chunk_input_ids[chunk_size].assign(
                    np.asarray(chunk, dtype=np.int64)[None, :]
                )
                if "position_ids" in self.runtime.input_names:
                    self._chunk_position_ids[chunk_size].assign(
                        np.arange(position, end, dtype=np.int64)[None, :]
                    )
                wp.launch(
                    _initialize_attention_mask,
                    dim=self.cache_capacity,
                    inputs=[self._decode_attention_mask, end],
                    device=self.runtime._device,
                )
                outputs = self._chunk_runtimes[chunk_size](
                    self._chunk_inputs[chunk_size]
                )
                self.sequence_length = end
                consumed += chunk_size

        logits = outputs["logits"] if outputs is not None else None
        for token_id in token_ids[consumed:]:
            logits = self.decode(token_id)
        self._past = dict(self._cache)
        return logits

    def _add_chunk_runtime(self, chunk_size: int) -> None:
        """Allocate one fixed-row prefill plan sharing the KV cache."""
        input_ids = wp.zeros(
            (1, chunk_size), dtype=wp.int64, device=self.runtime._device
        )
        position_ids = wp.zeros(
            (1, chunk_size), dtype=wp.int64, device=self.runtime._device
        )
        shapes = {
            "input_ids": (1, chunk_size),
            "attention_mask": (1, self.cache_capacity),
            **{name: tuple(cache.shape) for name, cache in self._cache.items()},
        }
        if "position_ids" in self.runtime.input_names:
            shapes["position_ids"] = (1, chunk_size)
        self._chunk_runtimes[chunk_size] = self.runtime._fork(
            shapes, share_kv_cache=True
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": self._decode_attention_mask,
            **self._cache,
        }
        if "position_ids" in self.runtime.input_names:
            inputs["position_ids"] = position_ids
        self._chunk_input_ids[chunk_size] = input_ids
        self._chunk_position_ids[chunk_size] = position_ids
        self._chunk_inputs[chunk_size] = inputs

    def decode(self, token_id: int) -> wp.array:
        """Append one token and return its logits."""
        if self.sequence_length == 0:
            raise RuntimeError(
                "Qwen3OnnxRunner.decode requires a preceding prefill call"
            )
        if self.sequence_length >= self.cache_capacity:
            raise ValueError("Qwen3OnnxRunner: KV cache is full")
        self._stage_decode_token(token_id)
        if self.runtime._device.is_cuda:
            if self._decode_graph is None:
                wp.capture_begin(device=self.runtime._device)
                try:
                    self._decode_graph_outputs = self._decode_runtime(
                        self._decode_inputs
                    )
                    self._decode_graph = wp.capture_end(device=self.runtime._device)
                except Exception:
                    wp.capture_end(device=self.runtime._device)
                    raise
            wp.capture_launch(self._decode_graph)
            outputs = self._decode_graph_outputs
        else:
            outputs = self._decode_runtime(self._decode_inputs)
        self.sequence_length += 1
        return outputs["logits"]

    def _launch_greedy_partials(self, logits: wp.array) -> None:
        wp.launch_tiled(
            self._greedy_argmax_kernels[0],
            dim=128,
            inputs=[logits, self._sample_partial_values, self._sample_partial_tokens],
            block_dim=256,
            device=self.runtime._device,
        )

    def sample_greedy(self, logits: wp.array) -> int:
        """Select the largest logit while transferring only its token ID to the host."""
        if not self.runtime._device.is_cuda:
            return sample_token(logits, temperature=0.0)
        if (
            logits.device != self.runtime._device
            or logits.dtype != wp.float16
            or logits.ndim != 3
        ):
            raise TypeError(
                "Qwen3OnnxRunner.sample_greedy expects a 3-D FP16 array on the runner device"
            )
        self._launch_greedy_partials(logits)
        wp.launch_tiled(
            self._greedy_argmax_kernels[1],
            dim=1,
            inputs=[
                self._sample_partial_values,
                self._sample_partial_tokens,
                self._sampled_token,
                logits.shape[2],
            ],
            block_dim=128,
            device=self.runtime._device,
        )
        wp.copy(self._sampled_token_host, self._sampled_token, count=1)
        wp.synchronize_stream(self.runtime._device)
        return int(self._sampled_token_host_view[0])

    def generate_greedy(
        self, token_ids: Sequence[int], max_new_tokens: int, eos_token_id: int
    ) -> list[int]:
        """Generate with an allocation-free captured CUDA graph and device-side argmax."""
        if max_new_tokens <= 0:
            return []
        if len(token_ids) + max_new_tokens > self.cache_capacity:
            raise ValueError(
                "Qwen3OnnxRunner: requested generation exceeds KV-cache capacity"
            )
        prompt_logits = self.prefill(token_ids)
        wp.launch(
            _initialize_generation_state,
            dim=1,
            inputs=[
                self._decode_position,
                self._generated_count,
                self._generation_finished,
                self.sequence_length,
            ],
            device=self.runtime._device,
        )
        sample_inputs = [
            self._sample_partial_values,
            self._sample_partial_tokens,
            prompt_logits.shape[2],
            self._decode_input_ids,
            self._decode_attention_mask,
            self._decode_position_ids,
            self._decode_position,
            self._generated_count,
            self._generated_ids,
            self._generation_finished,
            eos_token_id,
        ]
        self._launch_greedy_partials(prompt_logits)
        wp.launch_tiled(
            self._greedy_argmax_kernels[2],
            dim=1,
            inputs=sample_inputs,
            block_dim=128,
            device=self.runtime._device,
        )
        if max_new_tokens > 1 and (
            self._generation_graph is None or self._generation_graph_eos != eos_token_id
        ):
            wp.capture_begin(device=self.runtime._device)
            try:
                outputs = self._decode_runtime(self._decode_inputs)
                self._launch_greedy_partials(outputs["logits"])
                wp.launch_tiled(
                    self._greedy_argmax_kernels[2],
                    dim=1,
                    inputs=sample_inputs,
                    block_dim=128,
                    device=self.runtime._device,
                )
                self._generation_graph = wp.capture_end(device=self.runtime._device)
                self._generation_graph_eos = eos_token_id
            except Exception:
                wp.capture_end(device=self.runtime._device)
                raise
        for _ in range(max_new_tokens - 1):
            wp.capture_launch(self._generation_graph)
        generated = self._generated_ids.numpy()[:max_new_tokens].tolist()
        if eos_token_id in generated:
            generated = generated[: generated.index(eos_token_id) + 1]
        self.sequence_length += len(generated)
        return generated

    def _stage_decode_token(self, token_id: int) -> None:
        wp.launch(
            _set_decode_token,
            dim=1,
            inputs=[
                self._decode_input_ids,
                self._decode_attention_mask,
                self._decode_position_ids,
                token_id,
                self.sequence_length,
            ],
            device=self.runtime._device,
        )

    def _prepare_decode(self) -> None:
        wp.launch(
            _initialize_attention_mask,
            dim=self.cache_capacity,
            inputs=[self._decode_attention_mask, self.sequence_length],
            device=self.runtime._device,
        )
        self._past = dict(self._cache)
