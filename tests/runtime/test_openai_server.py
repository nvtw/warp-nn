# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
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

    def parse_tool_calls(self, text):
        return parse_qwen_tool_calls(text)


class _Runner:
    cache_capacity = 16

    def prefill(self, token_ids):
        return 1

    def decode(self, token_id):
        return 0

    def sample_greedy(self, logits):
        return logits


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
        ("127.0.0.1", 0), ChatCompletions("warp-qwen", _Runner(), tokenizer, enable_thinking=enable_thinking)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, tokenizer


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
    events = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")]
    chunks = [json.loads(event) for event in events[:-1]]
    assert content_type == "text/event-stream"
    assert events[-1] == "[DONE]"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"] == "Hi"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert tokenizer.request[0] == [{"role": "user", "content": "Hello"}]


def test_chat_completions_separates_reasoning():
    server, thread, tokenizer = _serve("Reasoning</think>\n\nAnswer", enable_thinking=True)
    try:
        body, _ = _request(
            server,
            {
                "model": "warp-qwen",
                "messages": [{"role": "user", "content": "Think"}],
                "reasoning_effort": "low",
                "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
                "stream": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    events = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")][:-1]
    chunks = [json.loads(event) for event in events]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    assert "".join(delta.get("reasoning_content", "") for delta in deltas) == "Reasoning"
    assert "".join(delta.get("content", "") for delta in deltas) == "Answer"
    assert tokenizer.request[1]["enable_thinking"] is True
    assert tokenizer.request[1]["reasoning_effort"] == "low"
    assert tokenizer.request[1]["preserve_thinking"] is False


def test_chat_completions_returns_reasoning():
    server, thread, _ = _serve("Reasoning</think>\n\nAnswer", enable_thinking=True)
    try:
        body, _ = _request(server, {"model": "warp-qwen", "messages": [{"role": "user", "content": "Think"}]})
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
                "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
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
                "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
                "stream": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    events = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")][:-1]
    chunks = [json.loads(event) for event in events]
    assert "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks) == "Checking. "
    tool_delta = next(chunk for chunk in chunks if chunk["choices"][0]["delta"].get("tool_calls"))
    tool_call = tool_delta["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["index"] == 0
    assert tool_call["function"] == {"name": "read", "arguments": '{"path":"README.md"}'}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
