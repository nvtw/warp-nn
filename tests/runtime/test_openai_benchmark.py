# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import threading

import pytest

from examples.benchmark_openai_concurrency import (
    RequestMetrics,
    percentile,
    run_request,
    summarize,
)
from warp_nn.runtime import ChatCompletions, OpenAIHTTPServer


def _request(start, first, end, events, tokens=4):
    return RequestMetrics(
        prompt_tokens=8,
        completion_tokens=tokens,
        started=start,
        first_token=first,
        finished=end,
        event_times=tuple(events),
    )


def test_percentile_interpolates_and_handles_empty_input():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    assert math.isnan(percentile([], 0.5))


def test_summary_uses_concurrent_wall_time_and_exact_usage_tokens():
    summary = summarize(
        2,
        [
            _request(10.0, 10.1, 11.0, [10.1, 10.3, 10.5, 10.7]),
            _request(10.0, 10.2, 12.0, [10.2, 10.6, 11.0, 11.4], tokens=6),
        ],
        memory_peak_mib=1024,
        memory_growth_mib=128,
    )

    assert summary.requests == 2
    assert summary.completion_tokens == 10
    assert summary.wall_seconds == 2.0
    assert summary.aggregate_tokens_per_second == 5.0
    assert summary.per_request_tokens_per_second == pytest.approx(
        (4 / 0.9 + 6 / 1.8) / 2
    )
    assert summary.ttft_p50_ms == pytest.approx(150.0)
    assert summary.ttft_p95_ms == pytest.approx(195.0)
    assert summary.inter_token_p50_ms == pytest.approx(300.0)
    assert summary.inter_token_p95_ms == pytest.approx(400.0)
    assert summary.memory_peak_mib == 1024
    assert summary.memory_growth_mib == 128


def test_single_output_event_has_no_fake_inter_token_latency():
    summary = summarize(1, [_request(1.0, 1.2, 2.0, [1.2], tokens=1)])
    assert math.isnan(summary.inter_token_p50_ms)
    assert math.isnan(summary.inter_token_p95_ms)


class _Tokenizer:
    eos_token_id = 0
    tool_call_start = "<tool_call>"

    def encode_chat(self, messages, **kwargs):
        del messages, kwargs
        return [9]

    def token_bytes(self, token_id, skip_special_tokens=False):
        del skip_special_tokens
        return b"A" if token_id == 1 else b""

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "A" if 1 in token_ids else ""

    def parse_tool_calls(self, text):
        return text, []


class _Runner:
    cache_capacity = 16

    def prefill(self, token_ids):
        del token_ids
        return 1

    def decode(self, token_id):
        del token_id
        return 0

    def sample_greedy(self, logits):
        return logits


def test_streaming_request_uses_reported_token_counts():
    server = OpenAIHTTPServer(
        ("127.0.0.1", 0),
        ChatCompletions("warp-qwen", _Runner(), _Tokenizer(), max_new_tokens=2),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        metrics = run_request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            "warp-qwen",
            "hello",
            2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert metrics.prompt_tokens == 1
    assert metrics.completion_tokens == 2
    assert len(metrics.event_times) == 1
    assert metrics.started <= metrics.first_token <= metrics.finished
