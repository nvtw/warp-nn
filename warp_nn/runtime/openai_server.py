# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal dependency-free OpenAI Chat Completions server."""

from __future__ import annotations

import codecs
import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from warp_nn.runtime.chat import Runner, Tokenizer, generate_tokens


class APIError(Exception):
    def __init__(self, message: str, status: int = 400, param: str | None = None):
        super().__init__(message)
        self.status = status
        self.param = param


def _text_content(content: object, param: str) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        text = []
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise APIError("Only text message content is supported", param=param)
            text.append(part["text"])
        return "".join(text)
    raise APIError("Message content must be text", param=param)


def _messages(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise APIError("messages must be a non-empty array", param="messages")
    result = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping) or message.get("role") not in (
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        ):
            raise APIError("Invalid message role", param=f"messages.{index}.role")
        item = dict(message)
        item["content"] = _text_content(item.get("content"), f"messages.{index}.content")
        if item["content"] is None and item["role"] != "assistant":
            raise APIError("Only assistant messages may have null content", param=f"messages.{index}.content")
        result.append(item)
    return result


class ChatCompletions:
    """Translate OpenAI chat requests to a shared text-generation runner."""

    def __init__(
        self,
        model: str,
        runner: Runner,
        tokenizer: Tokenizer,
        max_new_tokens: int = 4096,
        enable_thinking: bool = False,
    ):
        self.model = model
        self.runner = runner
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.lock = threading.Lock()

    def complete(self, request: Mapping[str, object], emit: Callable[[dict[str, object]], None] | None = None):
        if request.get("model") not in (None, self.model):
            raise APIError(f"Model {request['model']!r} is not available", status=404, param="model")
        if request.get("n", 1) != 1:
            raise APIError("Only n=1 is supported", param="n")
        if request.get("stop") not in (None, [], ""):
            raise APIError("Custom stop sequences are not supported", param="stop")
        messages = _messages(request.get("messages"))
        tools = request.get("tools")
        if tools is not None and not isinstance(tools, list):
            raise APIError("tools must be an array", param="tools")
        if request.get("tool_choice") == "none":
            tools = None
        try:
            temperature = float(request.get("temperature", 0.0) or 0.0)
            top_p = float(request.get("top_p", 1.0) or 1.0)
        except (TypeError, ValueError) as error:
            raise APIError("temperature and top_p must be numbers") from error
        if temperature < 0.0 or not 0.0 < top_p <= 1.0:
            raise APIError("temperature must be non-negative and top_p must be within (0, 1]")
        max_tokens = request.get("max_completion_tokens", request.get("max_tokens", self.max_new_tokens))
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise APIError("max tokens must be a positive integer", param="max_completion_tokens")
        max_tokens = min(max_tokens, self.max_new_tokens)

        try:
            prompt_ids = self.tokenizer.encode_chat(
                messages,
                tools=tools,
                enable_thinking=self.enable_thinking,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise APIError(str(error), param="messages") from error
        available = self.runner.cache_capacity - len(prompt_ids)
        if available <= 0:
            raise APIError("Prompt exceeds the model context window", status=400, param="messages")
        max_tokens = min(max_tokens, available)
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        generated = []
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        buffered = bool(tools)

        response_started = False
        with self.lock:
            for token_id in generate_tokens(
                self.runner,
                self.tokenizer,
                prompt_ids,
                max_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                if emit is not None and not response_started:
                    emit(self._chunk(completion_id, created, {"role": "assistant", "content": ""}))
                    response_started = True
                generated.append(token_id)
                if token_id == self.tokenizer.eos_token_id:
                    break
                text = decoder.decode(self.tokenizer.token_bytes(token_id, skip_special_tokens=True))
                if emit is not None and text and not buffered:
                    emit(self._chunk(completion_id, created, {"content": text}))
        tail = decoder.decode(b"", final=True)
        if emit is not None and tail and not buffered:
            emit(self._chunk(completion_id, created, {"content": tail}))

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text, tool_calls = self.tokenizer.parse_tool_calls(text)
        finish_reason = "tool_calls" if tool_calls else (
            "stop" if generated and generated[-1] == self.tokenizer.eos_token_id else "length"
        )
        message: dict[str, object] = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": "call_" + uuid.uuid4().hex,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False, separators=(",", ":")),
                    },
                }
                for call in tool_calls
            ]
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated),
            "total_tokens": len(prompt_ids) + len(generated),
        }
        if emit is not None:
            if buffered:
                delta = {key: value for key, value in message.items() if key != "role"}
                if "tool_calls" in delta:
                    delta["tool_calls"] = [
                        {"index": index, **tool_call} for index, tool_call in enumerate(delta["tool_calls"])
                    ]
                emit(self._chunk(completion_id, created, delta))
            emit(self._chunk(completion_id, created, {}, finish_reason))
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": self.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }

    def _chunk(self, completion_id: str, created: int, delta: dict[str, object], finish_reason=None):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }


class OpenAIHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend: ChatCompletions, api_key: str | None = None):
        self.backend = backend
        self.api_key = api_key
        super().__init__(address, OpenAIRequestHandler)


class OpenAIRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def _server(self) -> OpenAIHTTPServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self):
        if urlsplit(self.path).path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [{"id": self._server.backend.model, "object": "model", "owned_by": "warp-nn"}],
                },
            )
        else:
            self._error(APIError("Not found", status=404))

    def do_POST(self):
        try:
            if urlsplit(self.path).path != "/v1/chat/completions":
                raise APIError("Not found", status=404)
            self._authorize()
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise APIError("Invalid Content-Length header") from error
            if length <= 0 or length > 8 * 1024 * 1024:
                raise APIError("Invalid request body size")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, Mapping):
                raise APIError("Request body must be a JSON object")
            if request.get("stream"):
                self._stream(request)
            else:
                self._json(200, self._server.backend.complete(request))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._error(APIError("Request body must be valid JSON"))
        except APIError as error:
            if not getattr(self, "_stream_started", False):
                self._error(error)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.log_error("inference failed: %s", error)
            if not getattr(self, "_stream_started", False):
                self._error(APIError("Inference failed", status=500))

    def _authorize(self):
        if self._server.api_key is not None and self.headers.get("Authorization") != f"Bearer {self._server.api_key}":
            raise APIError("Invalid API key", status=401)

    def _stream(self, request: Mapping[str, object]):
        stream_options = request.get("stream_options") or {}
        if not isinstance(stream_options, Mapping):
            raise APIError("stream_options must be an object", param="stream_options")
        started = False

        def emit(chunk):
            nonlocal started
            if not started:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                self._stream_started = True
                started = True
            self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        response = self._server.backend.complete(request, emit)
        if stream_options.get("include_usage"):
            usage = {**response, "object": "chat.completion.chunk", "choices": []}
            emit({key: usage[key] for key in ("id", "object", "created", "model", "choices", "usage")})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _json(self, status: int, value: object):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, error: APIError):
        self._json(
            error.status,
            {"error": {"message": str(error), "type": "invalid_request_error", "param": error.param, "code": None}},
        )

    def log_message(self, format, *args):
        print(f"{self.client_address[0]} - {format % args}")
