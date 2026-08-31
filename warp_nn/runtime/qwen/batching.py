# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuous-batching adapter for Qwen 3.5-family decode runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from warp_nn.runtime.chat import is_eos_token, sample_runner_token
from warp_nn.runtime.services.batching import (
    BatchRequest,
    DecodeResult,
    PrefillItem,
)


@dataclass(frozen=True)
class QwenBatchPayload:
    token_ids: tuple[int, ...]
    temperature: float
    top_k: int
    top_p: float
    presence_penalty: float
    seed: int | None


@dataclass
class _Slot:
    payload: QwenBatchPayload
    logits: Any = None
    generated: list[int] = field(default_factory=list)
    cached_ids: list[int] = field(default_factory=list)
    prefilled: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)


class QwenBatchExecutor:
    """Sample tokens around a shared-weight ``Qwen35BatchDecoder``."""

    def __init__(self, runner, tokenizer, max_batch_size: int):
        self.runner = runner
        self.tokenizer = tokenizer
        self.decoder = runner.create_batch_decoder(max_batch_size)
        self.max_batch_size = max_batch_size
        self._slots: list[_Slot | None] = [None] * max_batch_size

    def prefix_match(self, slot: int, request: BatchRequest) -> int:
        state = self._slots[slot]
        payload = self._payload(request)
        if (
            state is not None
            and state.cached_ids
            and payload.token_ids[: len(state.cached_ids)] == tuple(state.cached_ids)
        ):
            return len(state.cached_ids)
        return 0

    def admit(self, slot: int, request: BatchRequest, reuse_prefix: bool) -> int:
        payload = self._payload(request)
        if reuse_prefix:
            state = self._slots[slot]
            if (
                state is None
                or not state.cached_ids
                or payload.token_ids[: len(state.cached_ids)] != tuple(state.cached_ids)
            ):
                raise RuntimeError("Qwen batch prefix state no longer matches")
            prefix_length = len(state.cached_ids)
            state.payload = payload
            state.generated.clear()
            state.rng = np.random.default_rng(payload.seed)
            state.prefilled = prefix_length
            state.cached_ids = list(payload.token_ids)
            remaining = len(payload.token_ids) - prefix_length
            if remaining:
                self.decoder.resume_prefill(slot)
            return remaining
        self._slots[slot] = _Slot(
            payload=payload,
            cached_ids=list(payload.token_ids),
            rng=np.random.default_rng(payload.seed),
        )
        self.decoder.begin_prefill(slot)
        return len(payload.token_ids)

    def prefill(self, items: list[PrefillItem], max_tokens_per_request: int | None):
        processed = []
        for item in items:
            state = self._state(item.slot)
            count = (
                item.remaining_tokens
                if max_tokens_per_request is None
                else min(item.remaining_tokens, max_tokens_per_request)
            )
            end = state.prefilled + count
            self.decoder.append_prefill(
                item.slot, state.payload.token_ids[state.prefilled : end]
            )
            state.prefilled = end
            if end == len(state.payload.token_ids):
                state.logits = self.decoder.end_prefill(item.slot)
            processed.append(count)
        return processed

    def select_decode_bucket(self, active_count: int) -> int:
        if not 0 < active_count <= self.max_batch_size:
            raise ValueError("invalid Qwen decode batch size")
        return self.max_batch_size

    def decode(self, slots: list[int], bucket_size: int):
        if bucket_size != self.max_batch_size:
            raise ValueError("Qwen decoder bucket does not match its fixed batch size")
        tokens = [0] * self.max_batch_size
        active = [False] * self.max_batch_size
        results = []
        for slot in slots:
            state = self._state(slot)
            token = sample_runner_token(
                self.runner,
                state.logits,
                temperature=state.payload.temperature,
                top_k=state.payload.top_k,
                top_p=state.payload.top_p,
                presence_penalty=state.payload.presence_penalty,
                previous_tokens=state.generated,
                rng=state.rng,
            )
            eos = is_eos_token(self.tokenizer, token)
            tokens[slot] = token
            active[slot] = not eos
            results.append(DecodeResult(token, eos))
            state.generated.append(token)
            if not eos:
                state.cached_ids.append(token)
        if any(active):
            logits = self.decoder.decode(tokens, active)
            for slot in slots:
                if active[slot]:
                    self._state(slot).logits = logits[slot : slot + 1]
        return results

    def release(self, slot: int, retain_prefix: bool) -> None:
        if retain_prefix:
            return
        self.decoder.release(slot)
        self._slots[slot] = None

    @staticmethod
    def _payload(request: BatchRequest) -> QwenBatchPayload:
        if not isinstance(request.payload, QwenBatchPayload):
            raise TypeError("Qwen batch request has an invalid payload")
        return request.payload

    def _state(self, slot: int) -> _Slot:
        state = self._slots[slot]
        if state is None:
            raise RuntimeError("Qwen batch slot is not admitted")
        return state
