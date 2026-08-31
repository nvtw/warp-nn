# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from warp_nn.runtime.qwen.batching import QwenBatchExecutor, QwenBatchPayload
from warp_nn.runtime.services.batching import (
    BatchRequest,
    ContinuousBatchScheduler,
    SchedulerOverloadedError,
)
from warp_nn.runtime.services.openai_server import ChatCompletions, OpenAIHTTPServer


def _logits(batch, token, vocab=5):
    result = np.full((batch, 1, vocab), -10.0, dtype=np.float32)
    result[:, :, token] = 10.0
    return result


class _Tokenizer:
    eos_token_id = 0
    eos_token_ids = (0,)
    tool_call_start = "<tool_call>"

    def encode_chat(self, messages, **_kwargs):
        return [ord(char) % 4 + 1 for char in messages[-1]["content"]]

    def token_bytes(self, token_id, skip_special_tokens=False):
        del skip_special_tokens
        return b"A" if token_id == 1 else b""

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "A" if 1 in token_ids else ""

    def parse_tool_calls(self, text):
        return text, []


class _ReasoningTokenizer(_Tokenizer):
    def token_bytes(self, token_id, skip_special_tokens=False):
        del skip_special_tokens
        return b"Reasoning</think>A" if token_id == 1 else b""

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "Reasoning</think>A" if 1 in token_ids else ""


class _BatchDecoder:
    def __init__(self, size):
        self.size = size
        self.prefills = []
        self.prefill_buffers = {}
        self.decodes = []
        self.decode_ones = []
        self.releases = []
        self.resumes = []

    def begin_prefill(self, slot):
        self.prefill_buffers[slot] = []

    def resume_prefill(self, slot):
        self.resumes.append(slot)

    def append_prefill(self, slot, token_ids):
        tokens = tuple(token_ids)
        self.prefills.append((slot, tokens))
        self.prefill_buffers[slot].extend(tokens)
        return _logits(1, 1)

    def end_prefill(self, slot):
        return _logits(1, 1)

    def decode(self, token_ids, active=None):
        self.decodes.append((tuple(token_ids), tuple(active)))
        return _logits(self.size, 0)

    def decode_one(self, slot, token_id):
        self.decode_ones.append((slot, token_id))
        return _logits(1, 0)

    def release(self, slot):
        self.releases.append(slot)


class _Runner:
    cache_capacity = 128

    def __init__(self):
        self.batch = None

    def create_batch_decoder(self, size):
        self.batch = _BatchDecoder(size)
        return self.batch

    def sample_greedy(self, logits):
        return int(np.argmax(np.asarray(logits).reshape(-1, logits.shape[-1])[-1]))


def _payload(tokens, seed=None):
    return QwenBatchPayload(tuple(tokens), 0.0, 0, 1.0, 0.0, seed)


def test_qwen_executor_batches_decode_and_reuses_exact_state():
    runner = _Runner()
    executor = QwenBatchExecutor(runner, _Tokenizer(), 2)
    with ContinuousBatchScheduler(executor, max_active=2, idle_wait_ms=20) as scheduler:
        first = scheduler.submit(BatchRequest(_payload([2]), 1, 3, True))
        second = scheduler.submit(BatchRequest(_payload([3]), 1, 3, True))
        assert first.result(2).tokens == (1, 0)
        assert second.result(2).tokens == (1, 0)
        assert len(runner.batch.prefills) == 2
        assert any(active == (True, True) for _, active in runner.batch.decodes)

        continued = scheduler.submit(BatchRequest(_payload([2, 1, 4]), 3, 3, True))
        assert continued.result(2).tokens == (1, 0)
        assert runner.batch.prefills[-1] == (0, (4,))
        assert runner.batch.resumes == [0]


def test_opt_in_chat_completions_preserves_concurrent_response_path():
    runner = _Runner()
    backend = ChatCompletions(
        "warp-qwen",
        runner,
        _Tokenizer(),
        max_new_tokens=3,
        max_batch_size=2,
        batch_wait_ms=20,
    )
    responses = []
    streams = []

    def complete(content):
        chunks = []
        response = backend.complete(
            {
                "model": "warp-qwen",
                "messages": [{"role": "user", "content": content}],
                "stream": True,
            },
            chunks.append,
        )
        responses.append(response)
        streams.append(chunks)

    threads = [threading.Thread(target=complete, args=(value,)) for value in ("x", "y")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    backend.close()

    assert len(responses) == 2
    assert all(
        response["choices"][0]["message"]["content"] == "A" for response in responses
    )
    assert all(
        chunks[0]["choices"][0]["delta"]["role"] == "assistant" for chunks in streams
    )
    assert any(active == (True, True) for _, active in runner.batch.decodes)


def test_opt_in_chat_preserves_reasoning_formatting():
    backend = ChatCompletions(
        "warp-qwen",
        _Runner(),
        _ReasoningTokenizer(),
        max_new_tokens=3,
        enable_thinking=True,
        max_batch_size=2,
        batch_wait_ms=0,
    )
    try:
        response = backend.complete(
            {
                "model": "warp-qwen",
                "messages": [{"role": "user", "content": "x"}],
            }
        )
    finally:
        backend.close()
    message = response["choices"][0]["message"]
    assert message["content"] == "A"
    assert message["reasoning_content"] == "Reasoning"


def test_single_request_uses_optimized_decode_one_bucket():
    runner = _Runner()
    backend = ChatCompletions(
        "warp-qwen",
        runner,
        _Tokenizer(),
        max_new_tokens=3,
        max_batch_size=4,
        batch_wait_ms=0,
    )
    try:
        response = backend.complete(
            {
                "model": "warp-qwen",
                "messages": [{"role": "user", "content": "x"}],
            }
        )
    finally:
        backend.close()
    assert response["choices"][0]["message"]["content"] == "A"
    assert runner.batch.decode_ones == [(0, 1)]
    assert runner.batch.decodes == []


def test_eight_request_executor_uses_one_shared_batch_decoder():
    runner = _Runner()
    executor = QwenBatchExecutor(runner, _Tokenizer(), 8)
    with ContinuousBatchScheduler(
        executor, max_active=8, idle_wait_ms=20
    ) as scheduler:
        handles = [
            scheduler.submit(BatchRequest(_payload([index + 1]), 1, 2))
            for index in range(8)
        ]
        assert [handle.result(2).tokens for handle in handles] == [(1, 0)] * 8

    assert runner.batch.size == 8
    assert any(active == (True,) * 8 for _, active in runner.batch.decodes)


def test_public_endpoint_maps_batch_queue_overload_to_429():
    backend = ChatCompletions("warp-qwen", _Runner(), _Tokenizer(), max_batch_size=2)

    def overloaded(_request):
        raise SchedulerOverloadedError("full")

    backend._batch_scheduler.submit = overloaded
    server = OpenAIHTTPServer(("127.0.0.1", 0), backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = Request(
        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "warp-qwen",
                "messages": [{"role": "user", "content": "x"}],
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 429
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
