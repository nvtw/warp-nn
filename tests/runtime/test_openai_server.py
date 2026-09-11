# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from urllib.request import Request, urlopen

from warp_nn.runtime import ChatCompletions, OpenAIHTTPServer, parse_qwen_tool_calls


class _Tokenizer:
    eos_token_id = 0
    tool_call_start = "<tool_call>"

    def __init__(self, text):
        self.text = text
        self.request = None

    def encode_chat(self, messages, **kwargs):
        self.request = (messages, kwargs)
        return [9]

    def token_bytes(self, token_id, skip_special_tokens=False):
        return self.text.encode() if token_id == 1 else b""

    def decode(self, token_ids, skip_special_tokens=False):
        return self.text if 1 in token_ids else ""

    def parse_tool_calls(self, text, *, tools=None):
        return parse_qwen_tool_calls(text, tools=tools)


class _Runner:
    cache_capacity = 16

    def prefill(self, token_ids):
        return 1

    def decode(self, token_id):
        return 0

    def sample_greedy(self, logits):
        return logits


class _CachingRunner(_Runner):
    def __init__(self):
        self.calls = []

    def prefill(self, token_ids):
        self.calls.append(("prefill", list(token_ids)))
        return 1

    def append(self, token_ids):
        self.calls.append(("append", list(token_ids)))
        return 1


class _CachingTokenizer(_Tokenizer):
    def encode_chat(self, messages, **kwargs):
        del kwargs
        return {"first": [9], "continued": [9, 1, 0, 7], "different": [5]}[
            messages[-1]["content"]
        ]


class _IncrementalTokenizer(_Tokenizer):
    def __init__(self):
        super().__init__("A")
        self.full_calls = 0
        self.encoded = []
        self.active = 0
        self.max_active = 0
        self.state_lock = threading.Lock()
        self.delay = 0.0

    def format_chat(self, messages, **kwargs):
        add_generation_prompt = kwargs.pop("add_generation_prompt", True)
        del kwargs
        with self.state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        if self.delay:
            time.sleep(self.delay)
        rendered = "".join(
            f"{message['role']}:{message['content']};" for message in messages
        )
        with self.state_lock:
            self.active -= 1
        return rendered + ("assistant:" if add_generation_prompt else "")

    def encode(self, text):
        self.encoded.append(text)
        return [
            1 if char == "A" else 0 if char == ";" else ord(char) + 10 for char in text
        ]

    def encode_chat(self, messages, **kwargs):
        self.full_calls += 1
        return self.encode(self.format_chat(messages, **kwargs))

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(
            "A" if token_id == 1 else ";" if token_id == 0 else ""
            for token_id in token_ids
        )


class _HiddenIncrementalTokenizer(_IncrementalTokenizer):
    def decode(self, token_ids, skip_special_tokens=False):
        if skip_special_tokens:
            return "Reasoning</think>A" if 1 in token_ids else ""
        return "".join(
            "hidden:A" if token_id == 1 else ";" if token_id == 0 else ""
            for token_id in token_ids
        )

    def token_bytes(self, token_id, skip_special_tokens=False):
        del skip_special_tokens
        return b"A" if token_id == 1 else b""


def _chat(content):
    return {"model": "warp-qwen", "messages": content}


def _request(server, body, path="/v1/chat/completions"):
    request = Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        return response.read(), response.headers.get_content_type()


def _serve(text, enable_thinking=False):
    tokenizer = _Tokenizer(text)
    server = OpenAIHTTPServer(
        ("127.0.0.1", 0),
        ChatCompletions(
            "warp-qwen", _Runner(), tokenizer, enable_thinking=enable_thinking
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, tokenizer


def test_chat_completions_reuses_exact_runner_prefix():
    runner = _CachingRunner()
    completions = ChatCompletions(
        "warp-qwen", runner, _CachingTokenizer("Hi"), max_new_tokens=2
    )
    completions.complete(
        {"model": "warp-qwen", "messages": [{"role": "user", "content": "first"}]}
    )
    completions.complete(
        {"model": "warp-qwen", "messages": [{"role": "user", "content": "continued"}]}
    )
    completions.complete(
        {"model": "warp-qwen", "messages": [{"role": "user", "content": "different"}]}
    )
    assert runner.calls == [
        ("prefill", [9]),
        ("append", [0, 7]),
        ("prefill", [5]),
    ]


def test_chat_completions_incrementally_encodes_appended_turns():
    runner = _CachingRunner()
    runner.cache_capacity = 4096
    tokenizer = _IncrementalTokenizer()
    completions = ChatCompletions("warp-qwen", runner, tokenizer, max_new_tokens=2)
    first = [{"role": "user", "content": "x"}]
    continued = [
        *first,
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "y"},
    ]

    completions.complete(_chat(first))
    first_prompt = runner.calls[0][1]
    completions.complete(_chat(continued))

    assert tokenizer.full_calls == 1
    assert tokenizer.encoded[-1] == "user:y;assistant:"
    expected_suffix = tokenizer.encode("user:y;assistant:")
    assert runner.calls[1] == ("append", [0, *expected_suffix])
    assert runner.calls[0] == ("prefill", first_prompt)

    completions.complete(_chat([{"role": "user", "content": "different"}]))
    assert tokenizer.full_calls == 2
    assert runner.calls[-1][0] == "prefill"


def test_chat_completions_preserves_hidden_generated_prefix():
    runner = _CachingRunner()
    runner.cache_capacity = 4096
    tokenizer = _HiddenIncrementalTokenizer()
    completions = ChatCompletions(
        "warp-qwen", runner, tokenizer, max_new_tokens=2, enable_thinking=True
    )
    first = [{"role": "user", "content": "x"}]

    response = completions.complete(_chat(first))
    returned_assistant = response["choices"][0]["message"]
    assert returned_assistant["reasoning_content"] == "Reasoning"
    assistant = {"role": "assistant", "content": "A"}
    completions.complete(_chat([*first, assistant, {"role": "user", "content": "y"}]))

    assert tokenizer.full_calls == 1
    suffix = tokenizer.encode("user:y;assistant:")
    assert runner.calls[1] == ("append", [0, *suffix])

    changed = {
        **assistant,
        "content": "edited",
    }
    completions.complete(_chat([*first, changed, {"role": "user", "content": "z"}]))
    assert tokenizer.full_calls == 2
    assert runner.calls[-1][0] == "prefill"


def test_chat_completions_serializes_incremental_encoder_with_runner():
    runner = _CachingRunner()
    runner.cache_capacity = 4096
    tokenizer = _IncrementalTokenizer()
    tokenizer.delay = 0.02
    completions = ChatCompletions("warp-qwen", runner, tokenizer, max_new_tokens=2)
    errors = []

    def complete(content):
        try:
            completions.complete(_chat([{"role": "user", "content": content}]))
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(target=complete, args=("one",)),
        threading.Thread(target=complete, args=("two",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert tokenizer.max_active == 1


def test_chat_completions_streams_text_and_usage():
    server, thread, tokenizer = _serve("Hi")
    try:
        body, content_type = _request(
            server,
            {
                "model": "warp-qwen",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    events = [
        line[6:] for line in body.decode().splitlines() if line.startswith("data: ")
    ]
    chunks = [json.loads(event) for event in events[:-1]]
    assert content_type == "text/event-stream"
    assert events[-1] == "[DONE]"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"] == "Hi"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert tokenizer.request[0] == [{"role": "user", "content": "Hello"}]


def test_chat_completions_separates_reasoning():
    server, thread, tokenizer = _serve(
        "Reasoning</think>\n\nAnswer", enable_thinking=True
    )
    try:
        body, _ = _request(
            server,
            {
                "model": "warp-qwen",
                "messages": [{"role": "user", "content": "Think"}],
                "reasoning_effort": "low",
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": False,
                },
                "stream": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    events = [
        line[6:] for line in body.decode().splitlines() if line.startswith("data: ")
    ][:-1]
    chunks = [json.loads(event) for event in events]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    assert (
        "".join(delta.get("reasoning_content", "") for delta in deltas) == "Reasoning"
    )
    assert "".join(delta.get("content", "") for delta in deltas) == "Answer"
    assert tokenizer.request[1]["enable_thinking"] is True
    assert tokenizer.request[1]["reasoning_effort"] == "low"
    assert tokenizer.request[1]["preserve_thinking"] is False


def test_chat_completions_returns_reasoning():
    server, thread, _ = _serve("Reasoning</think>\n\nAnswer", enable_thinking=True)
    try:
        body, _ = _request(
            server,
            {"model": "warp-qwen", "messages": [{"role": "user", "content": "Think"}]},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    message = json.loads(body)["choices"][0]["message"]
    assert message["content"] == "Answer"
    assert message["reasoning_content"] == "Reasoning"
    assert message["reasoning"] == "Reasoning"


def test_chat_completions_returns_structured_tool_call():
    server, thread, _ = _serve(
        "<tool_call>\n<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>"
    )
    try:
        body, _ = _request(
            server,
            {
                "model": "warp-qwen",
                "messages": [{"role": "user", "content": "Read it"}],
                "tools": [
                    {"type": "function", "function": {"name": "read", "parameters": {}}}
                ],
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    response = json.loads(body)
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    function = choice["message"]["tool_calls"][0]["function"]
    assert function == {"name": "read", "arguments": '{"path":"README.md"}'}


def test_chat_completions_streams_structured_tool_call():
    server, thread, _ = _serve(
        "Checking. <tool_call>\n<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>"
    )
    try:
        body, _ = _request(
            server,
            {
                "model": "warp-qwen",
                "messages": [{"role": "user", "content": "Read it"}],
                "tools": [
                    {"type": "function", "function": {"name": "read", "parameters": {}}}
                ],
                "stream": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    events = [
        line[6:] for line in body.decode().splitlines() if line.startswith("data: ")
    ][:-1]
    chunks = [json.loads(event) for event in events]
    assert (
        "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
        == "Checking. "
    )
    tool_delta = next(
        chunk for chunk in chunks if chunk["choices"][0]["delta"].get("tool_calls")
    )
    tool_call = tool_delta["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["index"] == 0
    assert tool_call["function"] == {
        "name": "read",
        "arguments": '{"path":"README.md"}',
    }
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_dflash_http_generation_discards_unconsumed_verified_cache():
    from types import SimpleNamespace

    class SpeculativeRunner(_CachingRunner):
        dflash = SimpleNamespace(block_size=2)
        speculative_calls = 0

        def decode_dflash(self, token_id, *, sample=None):
            self.speculative_calls += 1
            return [0, 1], 1  # EOS followed by a committed but unconsumed token

    runner = SpeculativeRunner()
    backend = ChatCompletions(
        "test", runner, _Tokenizer("Hi"), max_new_tokens=8, use_dflash=True
    )
    try:
        response = backend.complete(
            {"messages": [{"role": "user", "content": "hello"}]}
        )
        assert response["choices"][0]["message"]["content"] == "Hi"
        assert runner.speculative_calls == 1
        assert backend._cached_ids == []
    finally:
        backend.close()


def test_browser_page_and_key_enforcement():
    from urllib.error import HTTPError
    import pytest

    backend = ChatCompletions("test", _Runner(), _Tokenizer("Hi"))
    server = OpenAIHTTPServer(
        ("127.0.0.1", 0),
        backend,
        "test-secret",
        chat_html=b"<!doctype html><title>Chat</title>",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base) as response:
            assert response.headers["Content-Type"] == "text/html; charset=utf-8"
            assert b"<title>Chat</title>" in response.read()
        data = json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode()
        with pytest.raises(HTTPError) as error:
            urlopen(
                Request(
                    base + "/v1/chat/completions",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
            )
        assert error.value.code == 401
        with urlopen(
            Request(
                base + "/v1/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test-secret",
                },
            )
        ) as response:
            assert json.load(response)["choices"][0]["message"]["content"] == "Hi"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_connection_banner_has_browser_api_and_agent_details(capsys, monkeypatch):
    from types import SimpleNamespace
    from examples import openai_server

    monkeypatch.setattr(openai_server, "_lan_ipv4", lambda: "192.168.1.42")
    openai_server._print_connection_info(
        SimpleNamespace(server_port=8123), "qwen-test", "0.0.0.0", "test-key"
    )
    text = capsys.readouterr().out
    assert "http://127.0.0.1:8123/" in text
    assert "http://192.168.1.42:8123/v1" in text
    assert "aider --model openai/qwen-test" in text
    assert "--openai-api-key test-key" in text
    assert "not /v1/responses" in text
    openai_server._print_connection_info(
        SimpleNamespace(server_port=8123), "test", "192.168.1.42", None
    )
    assert "127.0.0.1" not in capsys.readouterr().out


def test_http_tool_call_preserves_schema_declared_strings():
    text = "<tool_call><function=write_file><parameter=content>42</parameter></function></tool_call>"
    backend = ChatCompletions("test", _Runner(), _Tokenizer(text))
    try:
        response = backend.complete(
            {
                "messages": [{"role": "user", "content": "write 42"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "parameters": {
                                "type": "object",
                                "properties": {"content": {"type": "string"}},
                            },
                        },
                    }
                ],
            }
        )
        arguments = response["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
        assert json.loads(arguments)["content"] == "42"
    finally:
        backend.close()
