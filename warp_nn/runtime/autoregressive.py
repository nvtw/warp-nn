# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable fixed-buffer lifecycle for autoregressive native runners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import warp as wp

from warp_nn.runtime.formats.gguf import (
    BlockQuantizedTensor,
)
from warp_nn.runtime.kernels import (
    _get_greedy_argmax_kernels,
    _get_top_k_kernels,
    _set_sequence_end,
    _stage_mrope_token_position,
    _stage_token_position,
)
from warp_nn.runtime.operators import (
    Operation,
)


class _PlanMemoryError(MemoryError):
    pass


def _cuda_storage_intervals(
    value, seen: set[int] | None = None
) -> list[tuple[int, int]]:
    """Collect CUDA byte ranges reachable through plan storage containers."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return []
    seen.add(identity)
    if isinstance(value, wp.array):
        if value.device.is_cuda and value.ptr and value.capacity:
            return [(int(value.ptr), int(value.ptr) + int(value.capacity))]
        return []
    if isinstance(value, BlockQuantizedTensor):
        return _cuda_storage_intervals((value.values, value.words, value.scales), seen)
    if isinstance(value, Operation):
        return _cuda_storage_intervals(value.attrs, seen)
    if isinstance(value, Mapping):
        intervals = []
        for item in value.values():
            intervals.extend(_cuda_storage_intervals(item, seen))
        return intervals
    if isinstance(value, (list, tuple, set)):
        intervals = []
        for item in value:
            intervals.extend(_cuda_storage_intervals(item, seen))
        return intervals
    return []


def _union_storage_bytes(intervals: list[tuple[int, int]]) -> int:
    """Return the byte size of overlapping storage ranges counted once."""
    total = 0
    end = 0
    for start, stop in sorted(intervals):
        if stop <= end:
            continue
        total += stop - max(start, end)
        end = stop
    return total


def _storage_bytes(value, excluded=()) -> int:
    intervals = _cuda_storage_intervals(value)
    excluded_intervals = _cuda_storage_intervals(excluded)
    return _union_storage_bytes(intervals + excluded_intervals) - _union_storage_bytes(
        excluded_intervals
    )


class AutoregressiveRunner:
    def _initialize_sampling(self) -> None:
        """Allocate the fixed device and bounded host buffers used by sampling."""
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

    def _record_plan_storage(self, plan) -> None:
        if not self.device.is_cuda:
            plan._owned_storage_bytes = 0
            plan._pool_storage_bytes = 0
            return
        persistent = (
            self.weights,
            getattr(self, "kv_caches", None),
            getattr(self, "conv_states", None),
            getattr(self, "recurrent_states", None),
            getattr(self, "cos_cache", None),
            getattr(self, "sin_cache", None),
            getattr(self, "sequence_end", None),
            getattr(self, "unit_scales", None),
            getattr(self, "zero_bias", None),
        )
        values = {
            name: value
            for name, value in vars(plan).items()
            if name not in {"runner", "graph", "graphs"}
            and not name.endswith("_storage_bytes")
        }
        plan._owned_storage_bytes = _storage_bytes(values, persistent)
        plan._pool_storage_bytes = _storage_bytes(
            getattr(plan, "_layer_buffer_pool", {}), persistent
        )

    def _lazy_plan_allocation_bound(self) -> int:
        if not hasattr(self._chunk_plan, "_owned_storage_bytes"):
            self._record_plan_storage(self._chunk_plan)
        return self._chunk_plan._owned_storage_bytes + (
            self._chunk_plan._pool_storage_bytes
        )

    def _require_lazy_plan_headroom(self, rows: int) -> None:
        if not self.device.is_cuda:
            return
        required = self._lazy_plan_allocation_bound()
        free = self.device.free_memory
        if free < required:
            raise _PlanMemoryError(
                f"Cannot allocate a {rows}-row inference plan: "
                f"{required / 2**20:.1f} MiB of shape-derived headroom is required, "
                f"but only {free / 2**20:.1f} MiB is free"
            )

    def reset(self) -> None:
        """Clear recurrent state while retaining all preallocated buffers."""
        for state in self.conv_states.values():
            state.zero_()
        for state in self.recurrent_states.values():
            state.zero_()
        self.sequence_length = 0
        self.rope_delta = 0

    def _run(self, plan, graph_key=None) -> wp.array:
        if self.device.is_cuda:
            if not hasattr(plan, "graphs"):
                plan.graphs = {}
                if getattr(plan, "graph", None) is not None:
                    plan.graphs[None] = (plan.graph, plan.outputs)
            graph_entry = plan.graphs.get(graph_key)
            if graph_entry is None and graph_key is not None:
                ready_keys = getattr(plan, "_capture_ready_keys", set())
                if graph_key not in ready_keys:
                    ready_keys.add(graph_key)
                    plan._capture_ready_keys = ready_keys
                    return plan.execute()
            if graph_entry is None and not getattr(plan, "_capture_ready", True):
                plan._capture_ready = True
                return plan.execute()
            if graph_entry is None and (
                getattr(plan, "_capture_disabled", False)
                or self.device.free_memory < getattr(plan, "_owned_storage_bytes", 0)
            ):
                plan._capture_disabled = True
                return plan.execute()
            if graph_entry is None:
                wp.capture_begin(device=self.device)
                try:
                    outputs = plan.execute()
                    graph_entry = (
                        wp.capture_end(device=self.device),
                        outputs,
                    )
                    plan.graphs[graph_key] = graph_entry
                except Exception:
                    wp.capture_end(device=self.device)
                    raise
            wp.capture_launch(graph_entry[0])
            return graph_entry[1]
        return plan.execute()

    def _stage_one(self, token_id: int) -> wp.array:
        position = self.sequence_length
        if hasattr(self._decode_plan, "rope_position_ids"):
            wp.launch(
                _stage_mrope_token_position,
                dim=1,
                inputs=[
                    self._decode_plan.input_ids,
                    self._decode_plan.position_ids,
                    self._decode_plan.rope_position_ids,
                    self.sequence_end,
                    token_id,
                    position,
                    position + self.rope_delta,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _stage_token_position,
                dim=1,
                inputs=[
                    self._decode_plan.input_ids,
                    self._decode_plan.position_ids,
                    self.sequence_end,
                    token_id,
                    position,
                ],
                device=self.device,
            )
        partitions = getattr(self._decode_plan, "attention_partitions", 256)
        logits = self._run(self._decode_plan, partitions)
        self.sequence_length += 1
        return logits

    def _plan_for_rows(self, rows: int):
        plans = getattr(self, "_chunk_plans", None)
        if plans is None:
            plans = self._chunk_plans = {self.prefill_chunk_size: self._chunk_plan}
        plan = plans.get(rows)
        if plan is None:
            self._require_lazy_plan_headroom(rows)
            plan = plans[rows] = type(self._chunk_plan)(self, rows)
            plan._capture_ready = False
            self._record_plan_storage(plan)
        return plan

    def _stage_many(self, token_ids: Sequence[int]) -> wp.array:
        rows = len(token_ids)
        plan = self._plan_for_rows(rows)
        end = self.sequence_length + rows
        plan.input_ids.assign(np.asarray(token_ids, dtype=np.int64)[None, :])
        plan.position_ids.assign(
            np.arange(self.sequence_length, end, dtype=np.int64)[None, :]
        )
        if hasattr(plan, "rope_position_ids"):
            plan.rope_position_ids.assign(
                np.broadcast_to(
                    np.arange(
                        self.sequence_length + self.rope_delta,
                        end + self.rope_delta,
                        dtype=np.int64,
                    ),
                    (3, rows),
                )
            )
        wp.launch(
            _set_sequence_end,
            dim=1,
            inputs=[self.sequence_end, end - 1],
            device=self.device,
        )
        logits = self._run(plan)
        self.sequence_length = end
        return logits

    def _append(self, token_ids: Sequence[int]) -> wp.array:
        if not token_ids:
            raise ValueError(f"{type(self).__name__} requires at least one token")
        if self.sequence_length + len(token_ids) > self.cache_capacity:
            raise ValueError(
                f"{type(self).__name__} token sequence exceeds cache_capacity"
            )
        logits = None
        denied_rows = set()
        start = 0
        while start < len(token_ids):
            remaining = len(token_ids) - start
            rows = min(self.prefill_chunk_size, 1 << (remaining.bit_length() - 1))
            if rows in denied_rows:
                rows = max(
                    (
                        existing
                        for existing in getattr(self, "_chunk_plans", {})
                        if existing <= remaining
                    ),
                    default=1,
                )
            if rows == 1:
                logits = self._stage_one(int(token_ids[start]))
            else:
                try:
                    logits = self._stage_many(token_ids[start : start + rows])
                except _PlanMemoryError:
                    denied_rows.add(rows)
                    rows = max(
                        (
                            existing
                            for existing in getattr(self, "_chunk_plans", {})
                            if existing < rows
                        ),
                        default=1,
                    )
                    if rows == 1:
                        logits = self._stage_one(int(token_ids[start]))
                    else:
                        logits = self._stage_many(token_ids[start : start + rows])
            start += rows
        return logits

    def prefill(self, token_ids: Sequence[int]) -> wp.array:
        """Reset state, process a prompt, and return its final logits."""
        self.reset()
        if len(token_ids) >= self.cache_capacity:
            raise ValueError(
                f"{type(self).__name__} prompt must leave room for one decoded token"
            )
        return self._append(token_ids)

    def append(self, token_ids: Sequence[int]) -> wp.array:
        """Append prompt tokens while retaining the current conversation state."""
        if self.sequence_length == 0:
            raise RuntimeError(
                f"{type(self).__name__}.append requires a preceding prefill"
            )
        return self._append(token_ids)

    def decode(self, token_id: int) -> wp.array:
        """Append one generated token and return its logits."""
        if self.sequence_length == 0:
            raise RuntimeError(
                f"{type(self).__name__}.decode requires a preceding prefill"
            )
        if self.sequence_length >= self.cache_capacity:
            raise ValueError(f"{type(self).__name__} KV cache is full")
        return self._stage_one(token_id)

    def sample_greedy(self, logits: wp.array) -> int:
        """Select the largest logit while transferring only its token ID."""
        if (
            logits.device != self.device
            or logits.dtype != self.dtype
            or logits.ndim != 3
        ):
            raise TypeError(
                f"{type(self).__name__}.sample_greedy expects runner logits"
            )
        wp.launch_tiled(
            self._greedy_argmax_kernels[0],
            dim=128,
            inputs=[logits, self._sample_partial_values, self._sample_partial_tokens],
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
                logits.shape[2],
            ],
            block_dim=128,
            device=self.device,
        )
        wp.copy(self._sampled_token_host, self._sampled_token, count=1)
        wp.synchronize_stream(self.device)
        return int(self._sampled_token_host_view[0])

    def read_top_k(
        self,
        logits: wp.array,
        top_k: int,
        *,
        token_start: int = 0,
        token_stop: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact top-k values and token IDs with a bounded host transfer."""
        if (
            logits.device != self.device
            or logits.dtype != self.dtype
            or logits.ndim != 3
            or logits.shape[0] != 1
            or logits.shape[1] == 0
        ):
            raise TypeError(f"{type(self).__name__}.read_top_k expects runner logits")
        full_vocabulary = logits.shape[2]
        if full_vocabulary != int(self.config["vocab_size"]):
            raise ValueError(
                f"{type(self).__name__}.read_top_k received an unexpected vocabulary"
            )
        token_stop = full_vocabulary if token_stop is None else token_stop
        if not 0 <= token_start < token_stop <= full_vocabulary:
            raise ValueError("top-k token interval is outside the vocabulary")
        logits = logits.flatten()[token_start:token_stop].reshape(
            (1, 1, token_stop - token_start)
        )
        vocabulary = token_stop - token_start
        if not 1 <= top_k <= 32:
            raise ValueError("top_k must be between 1 and 32")
        top_k = min(top_k, vocabulary)
        if not self.device.is_cuda:
            values = np.asarray(logits.numpy(), dtype=np.float32).reshape(
                -1, vocabulary
            )[-1]
            tokens = np.lexsort((np.arange(vocabulary), -values))[:top_k]
            return values[tokens], tokens.astype(np.int32) + token_start

        tile_width = 512
        partial_count = (vocabulary + tile_width - 1) // tile_width
        maximum_k = min(32, vocabulary)
        states = getattr(self, "_top_k_states", None)
        if states is None:
            states = self._top_k_states = {}
        state = states.get(vocabulary)
        if state is None:
            candidate_count = partial_count * maximum_k
            merge_count = (partial_count + 15) // 16
            values = wp.empty(candidate_count, dtype=wp.float32, device=self.device)
            tokens = wp.empty(candidate_count, dtype=wp.int32, device=self.device)
            merge_values = wp.empty(
                merge_count * maximum_k, dtype=wp.float32, device=self.device
            )
            merge_tokens = wp.empty(
                merge_count * maximum_k, dtype=wp.int32, device=self.device
            )
            host_values = wp.empty(
                maximum_k, dtype=wp.float32, device="cpu", pinned=True
            )
            host_tokens = wp.empty(maximum_k, dtype=wp.int32, device="cpu", pinned=True)
            state = states[vocabulary] = (
                _get_top_k_kernels(tile_width, maximum_k, self.dtype),
                values,
                tokens,
                merge_values,
                merge_tokens,
                host_values,
                host_tokens,
            )
        # Preserve the established inspection/debug handle while range-specific
        # buffers live in the keyed cache above.
        self._top_k_state = state
        (
            kernels,
            values,
            tokens,
            merge_values,
            merge_tokens,
            host_values,
            host_tokens,
        ) = state
        wp.launch_tiled(
            kernels[0],
            dim=partial_count,
            inputs=[logits, values, tokens],
            block_dim=256,
            device=self.device,
        )
        source_values, source_tokens = values, tokens
        target_values, target_tokens = merge_values, merge_tokens
        input_groups = partial_count
        while input_groups > 1:
            output_groups = (input_groups + 15) // 16
            wp.launch_tiled(
                kernels[1],
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
        wp.copy(host_values, source_values, count=top_k)
        wp.copy(host_tokens, source_tokens, count=top_k)
        wp.synchronize_stream(self.device)
        return (
            host_values.numpy()[:top_k].copy(),
            host_tokens.numpy()[:top_k].copy() + token_start,
        )
