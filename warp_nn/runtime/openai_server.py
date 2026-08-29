# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal dependency-free OpenAI Chat Completions server."""

from __future__ import annotations

import codecs
import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from warp_nn.runtime.chat import Runner, Tokenizer, generate_tokens, is_eos_token, split_reasoning, split_tool_prefix


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
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        presence_penalty: float = 0.0,
        reasoning_effort: str | None = None,
        preserve_thinking: bool = True,
    ):
        self.model = model
        self.runner = runner
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.reasoning_effort = reasoning_effort
        self.preserve_thinking = preserve_thinking
        self.lock = threading.Lock()
        self._cached_ids: list[int] = []

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
        template_kwargs = request.get("chat_template_kwargs") or {}
        if not isinstance(template_kwargs, Mapping):
            raise APIError("chat_template_kwargs must be an object", param="chat_template_kwargs")
        enable_thinking = template_kwargs.get("enable_thinking", request.get("enable_thinking", self.enable_thinking))
        preserve_thinking = template_kwargs.get(
            "preserve_thinking", request.get("preserve_thinking", self.preserve_thinking)
        )
        reasoning_effort = request.get("reasoning_effort", self.reasoning_effort)
        if not isinstance(enable_thinking, bool) or not isinstance(preserve_thinking, bool):
            raise APIError("thinking controls must be boolean")
        if reasoning_effort is not None and reasoning_effort not in ("low", "medium", "xhigh"):
            raise APIError("reasoning_effort must be low, medium, or xhigh", param="reasoning_effort")
        try:
            temperature = float(request.get("temperature", self.temperature))
            top_p = float(request.get("top_p", self.top_p))
            top_k = int(request.get("top_k", self.top_k))
            presence_penalty = float(request.get("presence_penalty", self.presence_penalty))
        except (TypeError, ValueError) as error:
            raise APIError("invalid sampling parameter") from error
        if temperature < 0.0 or not 0.0 < top_p <= 1.0 or top_k < 0 or not -2.0 <= presence_penalty <= 2.0:
            raise APIError("sampling parameters are outside their supported ranges")
        max_tokens = request.get("max_completion_tokens", request.get("max_tokens", self.max_new_tokens))
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise APIError("max tokens must be a positive integer", param="max_completion_tokens")
        max_tokens = min(max_tokens, self.max_new_tokens)

        try:
            prompt_ids = self.tokenizer.encode_chat(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                preserve_thinking=preserve_thinking,
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
        stream_filter = self.tokenizer.stream_filter() if hasattr(self.tokenizer, "stream_filter") else None
        tool_marker = self.tokenizer.tool_call_start if tools else None
        pending = ""
        tool_started = bool(tools) and not tool_marker
        reasoning_pending = ""
        reasoning_done = stream_filter is not None or not enable_thinking

        response_started = False

        def emit_content(text: str):
            nonlocal pending, tool_started
            if emit is None or not text:
                return
            if tool_started:
                pending += text
            elif tool_marker:
                text, pending, tool_started = split_tool_prefix(pending + text, tool_marker)
                if text:
                    emit(self._chunk(completion_id, created, {"content": text}))
            else:
                emit(self._chunk(completion_id, created, {"content": text}))

        def emit_text(text: str, final: bool = False):
            nonlocal reasoning_pending, reasoning_done
            if reasoning_done:
                emit_content(text)
                return
            text = reasoning_pending + text
            before, marker, after = text.partition("</think>")
            if marker:
                if emit is not None and before:
                    emit(self._chunk(completion_id, created, {"reasoning_content": before}))
                reasoning_pending = ""
                reasoning_done = True
                emit_content(after.lstrip())
                return
            keep = 0 if final else min(len(text), len("</think>") - 1)
            while keep and not "</think>".startswith(text[-keep:]):
                keep -= 1
            streamable = text[:-keep] if keep else text
            reasoning_pending = text[-keep:] if keep else ""
            if emit is not None and streamable:
                emit(self._chunk(completion_id, created, {"reasoning_content": streamable}))

        with self.lock:
            cached_prefix = self._cached_ids
            self._cached_ids = []
            if (
                cached_prefix
                and len(cached_prefix) < len(prompt_ids)
                and prompt_ids[: len(cached_prefix)] == cached_prefix
                and hasattr(self.runner, "append")
            ):
                logits = self.runner.append(prompt_ids[len(cached_prefix) :])
            else:
                logits = self.runner.prefill(prompt_ids)
            for token_id in generate_tokens(
                self.runner,
                self.tokenizer,
                prompt_ids,
                max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                presence_penalty=presence_penalty,
                initial_logits=logits,
            ):
                if emit is not None and not response_started:
                    emit(self._chunk(completion_id, created, {"role": "assistant", "content": ""}))
                    response_started = True
                generated.append(token_id)
                if is_eos_token(self.tokenizer, token_id):
                    break
                text = decoder.decode(
                    self.tokenizer.token_bytes(token_id, skip_special_tokens=stream_filter is None)
                )
                if stream_filter:
                    text = stream_filter.feed(text)
                emit_text(text)
            cached_completion = (
                generated[:-1]
                if generated and is_eos_token(self.tokenizer, generated[-1])
                else generated
            )
            self._cached_ids = [*prompt_ids, *cached_completion]
        tail = decoder.decode(b"", final=True)
        if stream_filter:
            tail = stream_filter.feed(tail, final=True)
        emit_text(tail, final=True)

        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        text, reasoning = (decoded, None) if stream_filter else split_reasoning(decoded, enable_thinking)
        text, tool_calls = self.tokenizer.parse_tool_calls(text)
        finish_reason = "tool_calls" if tool_calls else (
            "stop" if generated and is_eos_token(self.tokenizer, generated[-1]) else "length"
        )
        message: dict[str, object] = {"role": "assistant", "content": text or None}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
            message["reasoning"] = reasoning
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
            if tool_calls:
                stream_calls = [
                    {"index": index, **tool_call} for index, tool_call in enumerate(message["tool_calls"])
                ]
                emit(self._chunk(completion_id, created, {"tool_calls": stream_calls}))
            elif pending:
                emit(self._chunk(completion_id, created, {"content": pending}))
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
