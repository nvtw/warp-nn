# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared persistent-slot execution for native autoregressive batch decoders."""

from __future__ import annotations

import weakref

import numpy as np
import warp as wp

from warp_nn.runtime.kernels import (
    _get_stage_decode_batch_kernel,
    _set_sequence_end,
    _stage_mrope_token_position,
    _stage_token_position,
)


class BatchPlanState:
    """Compact execution controls over persistent physical batch slots."""

    def __init__(self, decoder, rows: int, mapped: bool = False):
        self._decoder = weakref.proxy(decoder)
        self.rows = rows
        self.mapped_state = bool(mapped)
        if mapped:
            self.active = wp.zeros(rows, dtype=wp.bool, device=decoder.device)
            self.positions = wp.zeros(rows, dtype=wp.int32, device=decoder.device)
            self.sequence_end = wp.zeros(rows, dtype=wp.int32, device=decoder.device)
            self.token_ids = wp.zeros(rows, dtype=wp.int64, device=decoder.device)
            self.rope_deltas = wp.zeros(rows, dtype=wp.int32, device=decoder.device)
            self.slot_indices = wp.empty(rows, dtype=wp.int32, device=decoder.device)
            self.conv_states = decoder.conv_states
            self.recurrent_states = decoder.recurrent_states
            self.kv_caches = decoder.kv_caches
        else:
            self.active = decoder.active[:rows]
            self.positions = decoder.positions[:rows]
            self.sequence_end = decoder.sequence_end[:rows]
            self.token_ids = decoder.token_ids[:rows]
            self.rope_deltas = decoder.rope_deltas[:rows]
            self.slot_indices = decoder.slot_indices[:rows]
            self.conv_states = {
                index: value[:rows] for index, value in decoder.conv_states.items()
            }
            self.recurrent_states = {}
            for index, value in decoder.recurrent_states.items():
                per_slot = value.shape[0] // decoder.max_batch_size
                self.recurrent_states[index] = value[: rows * per_slot]
            self.kv_caches = {}
            for index, (key, value) in decoder.kv_caches.items():
                per_slot = key.shape[0] // decoder.max_batch_size
                self.kv_caches[index] = (
                    key[: rows * per_slot],
                    value[: rows * per_slot],
                )

    def __getattr__(self, name):
        return getattr(self._decoder.runner, name)


class SlotPlanState:
    """Mutable zero-copy view selecting one persistent batch slot."""

    def __init__(self, decoder):
        self._decoder = weakref.proxy(decoder)
        self.select(0)

    def select(self, slot: int) -> None:
        decoder = self._decoder
        self.sequence_end = decoder.sequence_end[slot : slot + 1]
        self.conv_states = {
            index: value[slot] for index, value in decoder.conv_states.items()
        }
        self.recurrent_states = {}
        for index, value in decoder.recurrent_states.items():
            rows = value.shape[0] // decoder.max_batch_size
            self.recurrent_states[index] = value[slot * rows : (slot + 1) * rows]
        self.kv_caches = {}
        for index, (key, value) in decoder.kv_caches.items():
            rows = key.shape[0] // decoder.max_batch_size
            self.kv_caches[index] = (
                key[slot * rows : (slot + 1) * rows],
                value[slot * rows : (slot + 1) * rows],
            )

    def __getattr__(self, name):
        return getattr(self._decoder.runner, name)


class NativeBatchDecoder:
    """Independent mutable slots sharing one native runner's immutable weights."""

    def __init__(self, runner, max_batch_size: int, plan_type, model_name: str):
        if max_batch_size not in (2, 4, 8):
            raise ValueError(f"{model_name} batch size must be 2, 4, or 8")
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.plan_type = plan_type
        self.model_name = model_name
        self.device = runner.device
        self.dtype = runner.dtype
        self._uses_mrope = hasattr(runner._decode_plan, "rope_position_ids")
        self._stage_batch_kernel = _get_stage_decode_batch_kernel(self._uses_mrope)
        self._preflight_memory()
        rows = max_batch_size
        self.active = wp.zeros(rows, dtype=wp.bool, device=self.device)
        self.positions = wp.zeros(rows, dtype=wp.int32, device=self.device)
        self.sequence_end = wp.zeros(rows, dtype=wp.int32, device=self.device)
        self.token_ids = wp.zeros(rows, dtype=wp.int64, device=self.device)
        self.rope_deltas = wp.zeros(rows, dtype=wp.int32, device=self.device)
        self.slot_indices = wp.array(
            np.arange(rows, dtype=np.int32), device=self.device
        )
        self.conv_states = {
            index: wp.zeros((rows, *state.shape), dtype=state.dtype, device=self.device)
            for index, state in runner.conv_states.items()
        }
        self.recurrent_states = {
            index: wp.zeros(
                (rows * state.shape[0], state.shape[1]),
                dtype=state.dtype,
                device=self.device,
            )
            for index, state in runner.recurrent_states.items()
        }
        self.kv_caches = {}
        for index, (key, value) in runner.kv_caches.items():
            shape = (rows * key.shape[0], key.shape[1])
            self.kv_caches[index] = (
                wp.empty(shape, dtype=key.dtype, device=self.device),
                wp.empty(shape, dtype=value.dtype, device=self.device),
            )
        self._lengths = [0] * rows
        self._rope_delta_values = [0] * rows
        self._slot_prefill_open = [False] * rows
        self.prefill_outputs = wp.empty(
            (rows, 1, runner.config["vocab_size"]),
            dtype=self.dtype,
            device=self.device,
        )
        self._slot_view = SlotPlanState(self)
        self._incremental_plans = {}
        self._view = BatchPlanState(self, rows)
        self.plan = self._make_plan(self._view, rows, decode_batch=True)
        self._batch_views = {rows: self._view}
        self._batch_plans = {rows: self.plan}
        runner._record_plan_storage(self.plan)

    def _make_plan(self, state, rows: int, *, decode_batch: bool = False):
        return self.plan_type(state, rows, decode_batch=decode_batch)

    def warmup_decode_buckets(self) -> None:
        """Prepare every multi-request graph before the server accepts traffic."""
        if any(self._lengths):
            raise RuntimeError(
                f"{self.model_name} decode buckets must be warmed before admission"
            )
        if not self.device.is_cuda:
            return
        self._lengths[:] = [1] * self.max_batch_size
        for rows in (2, 4, 8):
            if rows > self.max_batch_size:
                break
            if rows == self.max_batch_size:
                self.decode([0] * rows, [True] * rows)
            else:
                self.decode_mapped(range(rows), [0] * rows, [True] * rows, rows)
        wp.synchronize_device(self.device)
        for state in self.conv_states.values():
            state.zero_()
        for state in self.recurrent_states.values():
            state.zero_()
        self._lengths[:] = [0] * self.max_batch_size
        wp.synchronize_device(self.device)

    def _preflight_memory(self) -> None:
        if not self.device.is_cuda:
            return
        state_bytes = sum(state.capacity for state in self.runner.conv_states.values())
        state_bytes += sum(
            state.capacity for state in self.runner.recurrent_states.values()
        )
        state_bytes += sum(
            key.capacity + value.capacity
            for key, value in self.runner.kv_caches.values()
        )
        plan_bytes = getattr(self.runner._decode_plan, "_owned_storage_bytes", 0)
        required = self.max_batch_size * (state_bytes + plan_bytes)
        free = self.device.free_memory
        if required > free * 0.95:
            raise MemoryError(
                f"{self.model_name} batch state and workspace need "
                f"{required / 2**30:.1f} GiB; only {free / 2**30:.1f} GiB is free"
            )

    def prefill(self, slot: int, token_ids, *, rope_delta: int = 0) -> wp.array:
        self.begin_prefill(slot, rope_delta=rope_delta)
        self.append_prefill(slot, token_ids)
        return self.end_prefill(slot)

    def begin_prefill(self, slot: int, *, rope_delta: int = 0) -> None:
        self._validate_slot(slot)
        for state in self.conv_states.values():
            state[slot].zero_()
        for state in self.recurrent_states.values():
            rows = state.shape[0] // self.max_batch_size
            state[slot * rows : (slot + 1) * rows].zero_()
        self._lengths[slot] = 0
        self._rope_delta_values[slot] = int(rope_delta)
        self._slot_prefill_open[slot] = True

    def resume_prefill(self, slot: int) -> None:
        self._validate_slot(slot)
        if self._slot_prefill_open[slot]:
            raise RuntimeError("incremental prefill is already open")
        if self._lengths[slot] == 0:
            raise RuntimeError(f"cannot resume an empty {self.model_name} batch slot")
        self._slot_prefill_open[slot] = True

    def _incremental_plan_for_rows(self, rows: int):
        plan = self._incremental_plans.get(rows)
        if plan is None:
            self.runner._require_lazy_plan_headroom(rows)
            plan = self._make_plan(self._slot_view, rows)
            plan._capture_ready = False
            self.runner._record_plan_storage(plan)
            self._incremental_plans[rows] = plan
        return plan

    def append_prefill(self, slot: int, token_ids) -> wp.array:
        self._validate_slot(slot)
        if not self._slot_prefill_open[slot]:
            raise RuntimeError("begin_prefill must precede append_prefill")
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("append_prefill requires at least one token")
        if self._lengths[slot] + len(tokens) >= self.runner.cache_capacity:
            raise ValueError(
                f"{self.model_name} prompt must leave room for one decoded token"
            )
        logits = None
        start = 0
        while start < len(tokens):
            remaining = len(tokens) - start
            rows = min(
                self.runner.prefill_chunk_size, 1 << (remaining.bit_length() - 1)
            )
            plan = self._incremental_plan_for_rows(rows)
            end = self._lengths[slot] + rows
            positions = np.arange(self._lengths[slot], end, dtype=np.int64)
            plan.input_ids.assign(
                np.asarray(tokens[start : start + rows], dtype=np.int64)[None, :]
            )
            plan.position_ids.assign(positions[None, :])
            if hasattr(plan, "rope_position_ids"):
                plan.rope_position_ids.assign(
                    np.broadcast_to(
                        positions + self._rope_delta_values[slot], (3, rows)
                    )
                )
            self._slot_view.select(slot)
            wp.launch(
                _set_sequence_end,
                dim=1,
                inputs=[self._slot_view.sequence_end, end - 1],
                device=self.device,
            )
            if self.device.is_cuda and slot not in plan.graphs:
                wp.synchronize_stream(self.device)
            logits = self.runner._run(plan, graph_key=slot)
            self._lengths[slot] = end
            start += rows
        wp.copy(
            self.prefill_outputs.flatten(),
            logits.flatten(),
            dest_offset=slot * logits.size,
            count=logits.size,
        )
        return self.prefill_outputs[slot : slot + 1]

    def end_prefill(self, slot: int) -> wp.array:
        self._validate_slot(slot)
        if not self._slot_prefill_open[slot] or self._lengths[slot] == 0:
            raise RuntimeError("incremental prefill has no prompt tokens")
        self._slot_prefill_open[slot] = False
        return self.prefill_outputs[slot : slot + 1]

    def decode_one(self, slot: int, token_id: int) -> wp.array:
        self._validate_slot(slot)
        position = self._lengths[slot]
        self._validate_active(slot, True)
        plan = self._incremental_plan_for_rows(1)
        self._slot_view.select(slot)
        if self._uses_mrope:
            wp.launch(
                _stage_mrope_token_position,
                dim=1,
                inputs=[
                    plan.input_ids,
                    plan.position_ids,
                    plan.rope_position_ids,
                    self._slot_view.sequence_end,
                    int(token_id),
                    position,
                    position + self._rope_delta_values[slot],
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _stage_token_position,
                dim=1,
                inputs=[
                    plan.input_ids,
                    plan.position_ids,
                    self._slot_view.sequence_end,
                    int(token_id),
                    position,
                ],
                device=self.device,
            )
        if self.device.is_cuda and slot not in plan.graphs:
            wp.synchronize_stream(self.device)
        logits = self.runner._run(plan, graph_key=slot)
        self._lengths[slot] += 1
        wp.copy(
            self.prefill_outputs.flatten(),
            logits.flatten(),
            dest_offset=slot * logits.size,
            count=logits.size,
        )
        return self.prefill_outputs[slot : slot + 1]

    def _batch_plan(self, rows: int):
        plan = self._batch_plans.get(rows)
        if plan is None:
            if rows not in (2, 4) or rows >= self.max_batch_size:
                raise ValueError(f"invalid compact {self.model_name} decode bucket")
            self.runner._require_lazy_plan_headroom(rows)
            view = BatchPlanState(self, rows, mapped=True)
            plan = self._make_plan(view, rows, decode_batch=True)
            self.runner._record_plan_storage(plan)
            self._batch_views[rows] = view
            self._batch_plans[rows] = plan
        return self._batch_views[rows], plan

    def _stage_batch(self, view, plan, rows: int) -> None:
        wp.launch(
            self._stage_batch_kernel,
            dim=rows,
            inputs=[
                plan.input_ids,
                view.positions,
                plan.rope_position_ids,
                view.sequence_end,
                view.active,
                view.token_ids,
                view.positions,
                view.rope_deltas,
            ],
            device=self.device,
        )

    def decode_mapped(self, slots, token_ids, active, bucket_size: int) -> wp.array:
        slots = tuple(int(slot) for slot in slots)
        tokens = tuple(int(token) for token in token_ids)
        active_values = tuple(bool(value) for value in active)
        if not (len(slots) == len(tokens) == len(active_values) <= bucket_size):
            raise ValueError("mapped decode inputs do not match the bucket")
        view, plan = self._batch_plan(bucket_size)
        padded_slots = [0] * bucket_size
        padded_tokens = [0] * bucket_size
        padded_active = [False] * bucket_size
        padded_positions = [0] * bucket_size
        padded_deltas = [0] * bucket_size
        for lane, (slot, token, enabled) in enumerate(
            zip(slots, tokens, active_values, strict=True)
        ):
            self._validate_slot(slot)
            self._validate_active(slot, enabled)
            padded_slots[lane] = slot
            padded_tokens[lane] = token
            padded_active[lane] = enabled
            padded_positions[lane] = self._lengths[slot]
            padded_deltas[lane] = self._rope_delta_values[slot]
        view.slot_indices.assign(np.asarray(padded_slots, dtype=np.int32))
        view.token_ids.assign(np.asarray(padded_tokens, dtype=np.int64))
        view.positions.assign(np.asarray(padded_positions, dtype=np.int32))
        view.rope_deltas.assign(np.asarray(padded_deltas, dtype=np.int32))
        view.active.assign(np.asarray(padded_active, dtype=np.bool_))
        self._stage_batch(view, plan, bucket_size)
        logits = self.runner._run(plan, plan.attention_partitions)
        for slot, enabled in zip(slots, active_values, strict=True):
            if enabled:
                self._lengths[slot] += 1
        return logits

    def decode(self, token_ids, active=None) -> wp.array:
        tokens = tuple(int(token) for token in token_ids)
        if len(tokens) != self.max_batch_size:
            raise ValueError("decode token_ids must match max_batch_size")
        active_values = (
            [length > 0 for length in self._lengths]
            if active is None
            else [bool(value) for value in active]
        )
        if len(active_values) != self.max_batch_size:
            raise ValueError("decode active mask must match max_batch_size")
        for slot, enabled in enumerate(active_values):
            self._validate_active(slot, enabled)
        self.token_ids.assign(np.asarray(tokens, dtype=np.int64))
        self.positions.assign(np.asarray(self._lengths, dtype=np.int32))
        self.rope_deltas.assign(np.asarray(self._rope_delta_values, dtype=np.int32))
        self.active.assign(np.asarray(active_values, dtype=np.bool_))
        self._stage_batch(self, self.plan, self.max_batch_size)
        logits = self.runner._run(self.plan, self.plan.attention_partitions)
        for slot, enabled in enumerate(active_values):
            if enabled:
                self._lengths[slot] += 1
        return logits

    def release(self, slot: int) -> None:
        self._validate_slot(slot)
        self._lengths[slot] = 0
        self._rope_delta_values[slot] = 0
        self._slot_prefill_open[slot] = False

    def _validate_active(self, slot: int, active: bool) -> None:
        if active and self._lengths[slot] == 0:
            raise RuntimeError(
                f"{self.model_name} batch slot {slot} has not been prefilled"
            )
        if active and self._lengths[slot] >= self.runner.cache_capacity:
            raise ValueError(f"{self.model_name} batch slot {slot} cache is full")

    def _validate_slot(self, slot: int) -> None:
        if not 0 <= int(slot) < self.max_batch_size:
            raise IndexError(f"{self.model_name} batch slot is out of range")
