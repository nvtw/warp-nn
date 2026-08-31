# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small model-agnostic scheduler for continuous GPU batching."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field
from typing import Any, Protocol


class SchedulerOverloadedError(RuntimeError):
    """Raised when the bounded admission queue is full."""


class StreamBackpressureError(RuntimeError):
    """Raised when a client does not consume streamed tokens."""


@dataclass(frozen=True)
class BatchRequest:
    """Executor payload and scheduler-visible generation limits."""

    payload: Any
    prompt_tokens: int
    max_new_tokens: int
    retain_prefix: bool = False


@dataclass(frozen=True)
class PrefillItem:
    slot: int
    request: BatchRequest
    remaining_tokens: int


@dataclass(frozen=True)
class DecodeResult:
    token: Any
    eos: bool = False


@dataclass(frozen=True)
class Completion:
    tokens: tuple[Any, ...]
    finish_reason: str


@dataclass(frozen=True)
class StreamEvent:
    token: Any | None = None
    finish_reason: str | None = None

    @property
    def done(self) -> bool:
        return self.finish_reason is not None


class BatchExecutor(Protocol):
    """Model-specific operations called exclusively by the scheduler thread."""

    def prefix_match(self, slot: int, request: BatchRequest) -> int:
        """Return the exact reusable prefix length, or zero for no match."""

    def admit(self, slot: int, request: BatchRequest, reuse_prefix: bool) -> int:
        """Prepare a slot and return how many prompt tokens remain to prefill."""

    def prefill(
        self, items: Sequence[PrefillItem], max_tokens_per_request: int | None
    ) -> Sequence[int]:
        """Run prompt chunks and return the processed count for each item."""

    def select_decode_bucket(self, active_count: int) -> int:
        """Select the graph/kernel bucket for ``active_count`` requests."""

    def decode(self, slots: Sequence[int], bucket_size: int) -> Sequence[DecodeResult]:
        """Generate one token for every listed slot."""

    def release(self, slot: int, retain_prefix: bool) -> None:
        """Release mutable request state, optionally retaining its prefix cache."""


class RequestHandle:
    """Completion future plus a bounded stream of token events."""

    def __init__(self, request: BatchRequest, stream_queue_size: int):
        self.request = request
        self.future: Future[Completion] = Future()
        self.events: queue.Queue[StreamEvent] = queue.Queue(stream_queue_size)
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def result(self, timeout: float | None = None) -> Completion:
        return self.future.result(timeout)

    def iter_events(self, timeout: float = 0.1) -> Iterator[StreamEvent]:
        """Yield buffered events until completion, propagating request errors."""

        while True:
            try:
                event = self.events.get(timeout=timeout)
            except queue.Empty:
                if self.future.done():
                    self.future.result()
                    return
                continue
            yield event
            if event.done:
                return


@dataclass
class _Active:
    handle: RequestHandle
    remaining_prompt: int
    generated: list[Any] = field(default_factory=list)


@dataclass
class _Slot:
    active: _Active | None = None
    retained: bool = False
    last_used: int = 0


class ContinuousBatchScheduler:
    """Own an executor and continuously batch requests on one worker thread."""

    def __init__(
        self,
        executor: BatchExecutor,
        *,
        max_active: int = 4,
        queue_size: int = 64,
        idle_wait_ms: float = 2.0,
        active_prefill_chunk: int = 64,
        stream_queue_size: int = 32,
    ):
        if not 0 < max_active <= 8:
            raise ValueError("max_active must be between one and eight")
        if queue_size <= 0 or stream_queue_size <= 0:
            raise ValueError("scheduler capacities must be positive")
        if idle_wait_ms < 0.0 or active_prefill_chunk <= 0:
            raise ValueError("scheduler timing and chunk size must be non-negative")
        self.executor = executor
        self.max_active = max_active
        self.idle_wait = idle_wait_ms / 1000.0
        self.active_prefill_chunk = active_prefill_chunk
        self.stream_queue_size = stream_queue_size
        self._pending: queue.Queue[RequestHandle] = queue.Queue(queue_size)
        self._slots = [_Slot() for _ in range(max_active)]
        self._clock = 0
        self._closed = threading.Event()
        self._wake = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="warp-nn-batch-scheduler", daemon=True
        )
        self._worker.start()

    def submit(self, request: BatchRequest) -> RequestHandle:
        if request.prompt_tokens < 0 or request.max_new_tokens <= 0:
            raise ValueError("invalid prompt or generation token count")
        if self._closed.is_set():
            raise RuntimeError("scheduler is closed")
        handle = RequestHandle(request, self.stream_queue_size)
        try:
            self._pending.put_nowait(handle)
        except queue.Full as error:
            raise SchedulerOverloadedError("batch scheduler queue is full") from error
        self._wake.set()
        return handle

    def close(self, timeout: float | None = 5.0) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._wake.set()
        self._worker.join(timeout)
        if self._worker.is_alive():
            raise TimeoutError("batch scheduler did not stop")

    def __enter__(self) -> ContinuousBatchScheduler:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _run(self) -> None:
        try:
            while not self._closed.is_set():
                if not self._has_active():
                    if not self._wait_for_first():
                        continue
                    self._coalesce_idle()
                self._admit_pending()
                self._cancel_marked()
                decoding = self._decoding()
                if decoding:
                    self._decode(decoding)
                prefilling = self._prefilling()
                if prefilling:
                    limit = self.active_prefill_chunk if decoding else None
                    self._prefill(prefilling, limit)
        finally:
            self._shutdown_requests()

    def _wait_for_first(self) -> bool:
        while not self._closed.is_set():
            if not self._pending.empty():
                return True
            self._wake.clear()
            self._wake.wait(0.05)
        return False

    def _coalesce_idle(self) -> None:
        deadline = time.monotonic() + self.idle_wait
        while self._pending.qsize() < self.max_active and not self._closed.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            self._wake.clear()
            self._wake.wait(remaining)

    def _has_active(self) -> bool:
        return any(slot.active is not None for slot in self._slots)

    def _admit_pending(self) -> None:
        while sum(slot.active is not None for slot in self._slots) < self.max_active:
            try:
                handle = self._pending.get_nowait()
            except queue.Empty:
                return
            if handle.cancelled:
                self._fail_handle(handle, CancelledError())
                continue
            slot = None
            try:
                slot_index, reuse = self._choose_slot(handle.request)
                slot = self._slots[slot_index]
                if not reuse and slot.retained:
                    self.executor.release(slot_index, False)
                    slot.retained = False
                remaining = self.executor.admit(slot_index, handle.request, reuse)
                if remaining < 0 or remaining > handle.request.prompt_tokens:
                    raise ValueError("executor returned an invalid prefill count")
            except BaseException as error:
                if slot is not None:
                    try:
                        self.executor.release(slot_index, False)
                    except BaseException:
                        pass
                    slot.retained = False
                self._fail_handle(handle, error)
                continue
            slot.retained = False
            slot.active = _Active(handle, remaining)

    def _choose_slot(self, request: BatchRequest) -> tuple[int, bool]:
        matches = []
        for index, slot in enumerate(self._slots):
            if slot.active is None and slot.retained:
                score = self.executor.prefix_match(index, request)
                if score < 0 or score > request.prompt_tokens:
                    raise ValueError("executor returned an invalid prefix match")
                if score:
                    matches.append((score, slot.last_used, index))
        if matches:
            _, _, index = max(matches)
            return index, True
        free = [
            (slot.last_used, index)
            for index, slot in enumerate(self._slots)
            if slot.active is None
        ]
        _, index = min(free)
        return index, False

    def _decoding(self) -> list[tuple[int, _Active]]:
        return [
            (index, slot.active)
            for index, slot in enumerate(self._slots)
            if slot.active is not None and slot.active.remaining_prompt == 0
        ]

    def _prefilling(self) -> list[tuple[int, _Active]]:
        return [
            (index, slot.active)
            for index, slot in enumerate(self._slots)
            if slot.active is not None and slot.active.remaining_prompt > 0
        ]

    def _decode(self, active: list[tuple[int, _Active]]) -> None:
        slots = [index for index, _ in active]
        try:
            bucket = self.executor.select_decode_bucket(len(slots))
            if bucket < len(slots):
                raise ValueError("decode bucket is smaller than the active batch")
            results = self.executor.decode(slots, bucket)
            if len(results) != len(active):
                raise ValueError("executor returned the wrong decode result count")
        except BaseException as error:
            for index, _ in active:
                self._retire(index, error=error)
            return
        for (index, state), result in zip(active, results):
            if self._slots[index].active is not state:
                continue
            state.generated.append(result.token)
            try:
                state.handle.events.put_nowait(StreamEvent(token=result.token))
            except queue.Full:
                self._retire(
                    index, error=StreamBackpressureError("stream queue is full")
                )
                continue
            if result.eos:
                self._retire(index, finish_reason="stop")
            elif len(state.generated) >= state.handle.request.max_new_tokens:
                self._retire(index, finish_reason="length")

    def _prefill(self, active: list[tuple[int, _Active]], limit: int | None) -> None:
        items = [
            PrefillItem(index, state.handle.request, state.remaining_prompt)
            for index, state in active
        ]
        try:
            processed = self.executor.prefill(items, limit)
            if len(processed) != len(active):
                raise ValueError("executor returned the wrong prefill result count")
            for count, (_, state) in zip(processed, active):
                maximum = (
                    state.remaining_prompt
                    if limit is None
                    else min(state.remaining_prompt, limit)
                )
                if count <= 0 or count > maximum:
                    raise ValueError(
                        "executor returned an invalid prefill progress count"
                    )
        except BaseException as error:
            for index, _ in active:
                self._retire(index, error=error)
            return
        for count, (index, state) in zip(processed, active):
            if self._slots[index].active is state:
                state.remaining_prompt -= count

    def _cancel_marked(self) -> None:
        for index, slot in enumerate(self._slots):
            if slot.active is not None and slot.active.handle.cancelled:
                self._retire(index, error=CancelledError())

    def _retire(
        self,
        index: int,
        *,
        finish_reason: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        slot = self._slots[index]
        state = slot.active
        if state is None:
            return
        retain = error is None and state.handle.request.retain_prefix
        try:
            self.executor.release(index, retain)
        except BaseException as release_error:
            error = error or release_error
            retain = False
        slot.active = None
        self._clock += 1
        slot.last_used = self._clock
        slot.retained = retain
        if error is not None:
            self._fail_handle(state.handle, error)
            return
        completion = Completion(tuple(state.generated), finish_reason or "stop")
        state.handle.future.set_result(completion)
        try:
            state.handle.events.put_nowait(
                StreamEvent(finish_reason=completion.finish_reason)
            )
        except queue.Full:
            pass

    @staticmethod
    def _fail_handle(handle: RequestHandle, error: BaseException) -> None:
        if not handle.future.done():
            handle.future.set_exception(error)

    def _shutdown_requests(self) -> None:
        for index, slot in enumerate(self._slots):
            if slot.active is not None:
                self._retire(index, error=CancelledError())
            elif slot.retained:
                try:
                    self.executor.release(index, False)
                except BaseException:
                    pass
                slot.retained = False
        while True:
            try:
                handle = self._pending.get_nowait()
            except queue.Empty:
                return
            self._fail_handle(handle, CancelledError())
