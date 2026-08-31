# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from concurrent.futures import CancelledError

import pytest

from warp_nn.runtime.services.batching import (
    BatchRequest,
    ContinuousBatchScheduler,
    DecodeResult,
    SchedulerOverloadedError,
    StreamBackpressureError,
)


class FakeExecutor:
    def __init__(self):
        self.calls = []
        self.tokens = {}
        self.released = []
        self.decode_entered = threading.Event()
        self.decode_gate = threading.Event()
        self.decode_gate.set()
        self.eos_at = {}
        self.slot_payload = {}
        self.retained_payload = {}

    def prefix_match(self, slot, request):
        retained = self.retained_payload[slot]
        common = 0
        for left, right in zip(str(retained), str(request.payload)):
            if left != right:
                break
            common += 1
        return min(common, request.prompt_tokens)

    def admit(self, slot, request, reuse_prefix):
        self.calls.append(("admit", slot, request.payload, reuse_prefix))
        self.slot_payload[slot] = request.payload
        self.tokens[slot] = 0
        return 0 if reuse_prefix else request.prompt_tokens

    def prefill(self, items, max_tokens_per_request):
        counts = [
            item.remaining_tokens
            if max_tokens_per_request is None
            else min(item.remaining_tokens, max_tokens_per_request)
            for item in items
        ]
        self.calls.append(
            (
                "prefill",
                tuple(item.slot for item in items),
                max_tokens_per_request,
                tuple(counts),
            )
        )
        return counts

    def select_decode_bucket(self, active_count):
        return next(size for size in (1, 2, 4, 8) if size >= active_count)

    def decode(self, slots, bucket_size):
        self.calls.append(("decode", tuple(slots), bucket_size))
        self.decode_entered.set()
        self.decode_gate.wait(2.0)
        results = []
        for slot in slots:
            self.tokens[slot] += 1
            token = self.tokens[slot]
            results.append(DecodeResult(token, token >= self.eos_at.get(slot, 10**6)))
        return results

    def release(self, slot, retain_prefix):
        self.calls.append(("release", slot, retain_prefix))
        self.released.append((slot, retain_prefix))
        if retain_prefix:
            self.retained_payload[slot] = self.slot_payload[slot]
        else:
            self.retained_payload.pop(slot, None)


def request(name, *, prompt=0, maximum=1, retain=False):
    return BatchRequest(name, prompt, maximum, retain)


def test_idle_requests_coalesce_and_executor_selects_bucket():
    executor = FakeExecutor()
    with ContinuousBatchScheduler(executor, idle_wait_ms=20) as scheduler:
        handles = [scheduler.submit(request(str(index))) for index in range(4)]
        assert [handle.result(2).tokens for handle in handles] == [(1,)] * 4

    decode = next(call for call in executor.calls if call[0] == "decode")
    assert len(decode[1]) == 4
    assert decode[2] == 4


def test_eight_requests_coalesce_into_the_extended_bucket():
    executor = FakeExecutor()
    with ContinuousBatchScheduler(
        executor, max_active=8, idle_wait_ms=20
    ) as scheduler:
        handles = [scheduler.submit(request(str(index))) for index in range(8)]
        assert [handle.result(2).tokens for handle in handles] == [(1,)] * 8

    decode = next(call for call in executor.calls if call[0] == "decode")
    assert len(decode[1]) == 8
    assert decode[2] == 8


def test_decode_is_prioritized_and_active_prefill_is_chunked():
    executor = FakeExecutor()
    executor.decode_gate.clear()
    with ContinuousBatchScheduler(executor, idle_wait_ms=0) as scheduler:
        decoding = scheduler.submit(request("decode", maximum=3))
        assert executor.decode_entered.wait(1)
        prefilling = scheduler.submit(request("prefill", prompt=130))
        executor.decode_gate.set()
        assert decoding.result(2).tokens == (1, 2, 3)
        assert prefilling.result(2).tokens == (1,)

    calls = [call for call in executor.calls if call[0] in ("decode", "prefill")]
    prefill_calls = [call for call in calls if call[0] == "prefill"]
    assert prefill_calls[0][2:] == (64, (64,))
    assert prefill_calls[1][2:] == (64, (64,))
    assert prefill_calls[2][3] == (2,)
    assert calls.index(prefill_calls[0]) > 0
    assert calls[calls.index(prefill_calls[0]) - 1][0] == "decode"


def test_bounded_admission_queue_reports_overload():
    executor = FakeExecutor()
    executor.decode_gate.clear()
    scheduler = ContinuousBatchScheduler(
        executor, max_active=1, queue_size=2, idle_wait_ms=0
    )
    active = scheduler.submit(request("active", maximum=2))
    assert executor.decode_entered.wait(1)
    queued = [scheduler.submit(request(str(index))) for index in range(2)]
    with pytest.raises(SchedulerOverloadedError):
        scheduler.submit(request("overflow"))
    executor.decode_gate.set()
    active.result(2)
    for handle in queued:
        handle.result(2)
    scheduler.close()


def test_cancellation_eos_and_bounded_stream_backpressure():
    executor = FakeExecutor()
    executor.eos_at[0] = 2
    with ContinuousBatchScheduler(executor, max_active=1, idle_wait_ms=0) as scheduler:
        eos = scheduler.submit(request("eos", maximum=10))
        assert eos.result(2).finish_reason == "stop"
        assert eos.result().tokens == (1, 2)

        executor.decode_gate.clear()
        cancelled = scheduler.submit(request("cancel", maximum=10))
        assert executor.decode_entered.wait(1)
        cancelled.cancel()
        executor.decode_gate.set()
        with pytest.raises(CancelledError):
            cancelled.result(2)

    executor = FakeExecutor()
    with ContinuousBatchScheduler(
        executor, max_active=1, idle_wait_ms=0, stream_queue_size=1
    ) as scheduler:
        slow = scheduler.submit(request("slow", maximum=3))
        with pytest.raises(StreamBackpressureError):
            slow.result(2)


def test_exact_prefix_reuse_and_lru_eviction():
    executor = FakeExecutor()
    with ContinuousBatchScheduler(executor, max_active=2, idle_wait_ms=20) as scheduler:
        first_a = scheduler.submit(request("a1", prompt=5, retain=True))
        first_b = scheduler.submit(request("b1", prompt=7, retain=True))
        first_a.result(2)
        first_b.result(2)

        second_a = scheduler.submit(request("a2", prompt=5, retain=True))
        second_a.result(2)
        c = scheduler.submit(request("c", prompt=3, retain=True))
        c.result(2)

    admits = [call for call in executor.calls if call[0] == "admit"]
    a1 = next(call for call in admits if call[2] == "a1")
    a2 = next(call for call in admits if call[2] == "a2")
    c_admit = next(call for call in admits if call[2] == "c")
    assert a2[1] == a1[1]
    assert a2[3] is True
    assert c_admit[1] != a1[1]
    assert (c_admit[1], False) in executor.released


def test_pending_cancellation_and_event_iteration():
    executor = FakeExecutor()
    executor.decode_gate.clear()
    with ContinuousBatchScheduler(
        executor, max_active=1, idle_wait_ms=0, stream_queue_size=4
    ) as scheduler:
        active = scheduler.submit(request("active"))
        assert executor.decode_entered.wait(1)
        pending = scheduler.submit(request("pending"))
        pending.cancel()
        executor.decode_gate.set()
        active.result(2)
        with pytest.raises(CancelledError):
            pending.result(2)
        events = list(active.iter_events())
        assert [event.token for event in events if not event.done] == [1]
        assert events[-1].finish_reason == "length"


def test_close_cancels_active_and_queued_requests():
    executor = FakeExecutor()
    executor.decode_gate.clear()
    scheduler = ContinuousBatchScheduler(executor, max_active=1, idle_wait_ms=0)
    active = scheduler.submit(request("active", maximum=10))
    assert executor.decode_entered.wait(1)
    queued = scheduler.submit(request("queued"))
    executor.decode_gate.set()
    time.sleep(0.01)
    scheduler.close()
    for handle in (active, queued):
        if not handle.future.done():
            pytest.fail("shutdown left a request unresolved")
