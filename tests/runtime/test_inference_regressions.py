# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control-flow regressions using host arrays and mocked device operations."""

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from tests.runtime.test_openai_server import _Runner, _Tokenizer
from warp_nn.runtime.services.openai_server import APIError, ChatCompletions
from warp_nn.runtime.muse.glimmer import MuseGlimmerTokenizer
from warp_nn.runtime.qwen.causal import ExternalEmbeddingQwen3CausalLM
from warp_nn.runtime.qwen.qwen3 import Qwen3OnnxRunner


@pytest.mark.parametrize("thinking", [False, True])
def test_request_sampling_defaults_and_overrides(thinking):
    class Tokenizer(_Tokenizer):
        def sampling_defaults(self, enabled):
            return dict(
                temperature=1.0 if enabled else 0.7,
                top_p=0.95 if enabled else 0.8,
                top_k=20,
                presence_penalty=0.0 if enabled else 1.5,
            )

    backend = ChatCompletions(
        "test", _Runner(), Tokenizer(""), enable_thinking=not thinking
    )
    request = {"chat_template_kwargs": {"enable_thinking": thinking}}
    expected = (1.0, 0.95, 20, 0.0, None) if thinking else (0.7, 0.8, 20, 1.5, None)
    assert backend._sampling_parameters(request) == expected
    backend.temperature = 0.4  # Explicit server override survives mode changes.
    assert backend._sampling_parameters(request)[0] == 0.4
    assert backend._sampling_parameters(dict(request, temperature=0.2))[0] == 0.2


@pytest.mark.parametrize("speculative", [False, True])
def test_http_seed_reaches_ordinary_and_speculative_sampler(speculative):
    class Runner(_Runner):
        cache_capacity = 64
        dflash = SimpleNamespace(block_size=2)

        def prefill(self, ids):
            self.tokens = []
            return np.array([-np.inf, 0.0, 0.0])

        def decode(self, token):
            self.tokens.append(token)
            return np.array([-np.inf, 0.0, 0.0])

        def decode_dflash(self, token, *, sample):
            logits = self.decode(token)
            return [sample(logits)], logits

    runner = Runner()
    backend = ChatCompletions(
        "test",
        runner,
        _Tokenizer("Hi"),
        temperature=1.0,
        max_new_tokens=16,
        use_dflash=speculative,
    )
    request = {"messages": [{"role": "user", "content": "hi"}], "seed": 42}
    backend.complete(request)
    first = runner.tokens.copy()
    backend.complete(request)
    assert runner.tokens == first
    backend.complete(dict(request, seed=43))
    assert runner.tokens != first


@pytest.mark.parametrize("value", [-1, True, 1.5, "42"])
def test_invalid_http_seed_is_rejected_before_prefill(value):
    backend = ChatCompletions("test", _Runner(), _Tokenizer(""))
    with pytest.raises(APIError, match="seed"):
        backend.complete(
            {"messages": [{"role": "user", "content": "hi"}], "seed": value}
        )


@pytest.mark.parametrize(
    "content", ["42", "true", "null", '{"a": 1}', '"quoted"', "  a\n"]
)
def test_muse_preserves_schema_string_arguments(content):
    tokenizer = object.__new__(MuseGlimmerTokenizer)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            },
        }
    ]
    text = (
        '<atem:function_calls><atem:invoke name="write">'
        f'<atem:parameter name="content">{content}</atem:parameter>'
        '<atem:parameter name="count">42</atem:parameter>'
        "</atem:invoke></atem:function_calls>"
    )
    _, calls = tokenizer.parse_tool_calls(text, tools=tools)
    assert calls == [{"name": "write", "arguments": {"content": content, "count": 42}}]


@pytest.mark.parametrize(
    "length,error", [(0, RuntimeError), (4, ValueError), (5, ValueError)]
)
def test_embedding_decode_rejects_invalid_cache_state(length, error):
    runner = object.__new__(ExternalEmbeddingQwen3CausalLM)
    runner.sequence_length = length
    runner.cache_capacity = 4
    with pytest.raises(error):
        runner.decode_embedding(None)  # Must reject before staging any device work.


@pytest.mark.parametrize(
    "samples,limit",
    [([2], 1), ([0], 1), ([2, 3, 4], 3), ([0], 3), ([2, 0], 3), ([2, 3, 0], 3)],
)
def test_onnx_generation_length_matches_written_cache(monkeypatch, samples, limit):
    import warp_nn.runtime.qwen.qwen3 as module

    cache = {0: 8, 1: 9}
    stream = iter(samples)
    staged = {}
    generated = []
    finished = False
    next_position = 2

    def advance(*args, **kwargs):
        nonlocal finished, next_position
        if finished:
            return
        token = next(stream)
        generated.append(token)
        staged.update(token=token, position=next_position)
        next_position += 1
        finished = token == 0

    def replay(graph):
        cache[staged["position"]] = staged["token"]
        advance()

    monkeypatch.setattr(module.wp, "launch", lambda *a, **k: None)
    monkeypatch.setattr(module.wp, "launch_tiled", advance)
    monkeypatch.setattr(module.wp, "capture_launch", replay)
    runner = SimpleNamespace(
        cache_capacity=16,
        sequence_length=2,
        runtime=SimpleNamespace(_device="mock"),
        prefill=lambda ids: SimpleNamespace(shape=(1, 1, 10)),
        _launch_greedy_partials=lambda logits: None,
        _greedy_argmax_kernels=[None, None, None],
        _generation_graph=object(),
        _generation_graph_eos=0,
        _generated_ids=SimpleNamespace(numpy=lambda: np.array(generated)),
    )
    for name in (
        "_decode_position",
        "_generated_count",
        "_generation_finished",
        "_sample_partial_values",
        "_sample_partial_tokens",
        "_decode_input_ids",
        "_decode_attention_mask",
        "_decode_position_ids",
    ):
        setattr(runner, name, None)
    assert Qwen3OnnxRunner.generate_greedy(runner, [8, 9], limit, 0) == samples
    assert runner.sequence_length == max(cache) + 1


def test_safetensors_upload_fence_uses_destination_stream(tmp_path, monkeypatch):
    import warp_nn.runtime.formats.safetensors as module

    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))
    destination = SimpleNamespace(is_cuda=True)
    event = object()
    recorded = []
    monkeypatch.setattr(module.wp, "get_device", lambda device: destination)
    monkeypatch.setattr(module.wp, "array", lambda **kwargs: object())
    monkeypatch.setattr(module.wp, "clone", lambda host, device: object())
    monkeypatch.setattr(
        module.wp,
        "get_stream",
        lambda device: (
            recorded.append(device) or SimpleNamespace(record_event=lambda: event)
        ),
    )
    monkeypatch.setattr(
        module, "_release_mappings", lambda resources, fence: recorded.append(fence)
    )
    assert "w" in module.SafeTensorArchive(path).load(device="cuda:1")
    assert recorded == [destination, event]


def test_batched_http_uses_request_mode_defaults(monkeypatch):
    import warp_nn.runtime.services.openai_server as module

    class Tokenizer(_Tokenizer):
        def sampling_defaults(self, thinking):
            return dict(
                temperature=1.0 if thinking else 0.7,
                top_p=0.95 if thinking else 0.8,
                top_k=20,
                presence_penalty=0.0 if thinking else 1.5,
            )

    payloads = []

    class Proxy(_Runner):
        def __init__(self, scheduler, capacity, factory, max_tokens):
            self.factory = factory

        def prefill(self, ids):
            payloads.append(self.factory(ids))
            return 1

        def cancel(self):
            pass

    monkeypatch.setattr(module, "_ScheduledRunner", Proxy)
    backend = ChatCompletions("test", _Runner(), Tokenizer("Hi"), enable_thinking=True)
    backend._batch_scheduler = object()
    response = backend.complete(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": False,
            "seed": 42,
        }
    )
    assert response["choices"][0]["message"]["content"] == "Hi"
    (payload,) = payloads
    assert (
        payload.temperature,
        payload.top_p,
        payload.top_k,
        payload.presence_penalty,
        payload.seed,
    ) == (0.7, 0.8, 20, 1.5, 42)
