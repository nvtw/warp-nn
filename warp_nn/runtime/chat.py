# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared text-generation helpers for stateful language-model runners."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
from typing import Any, Protocol

import numpy as np

from warp_nn.utils.paths import application_state_dir
from warp_nn.runtime.sampling import (  # re-export the original public API
    sample_candidates as sample_candidates,
    sample_token as sample_token,
    sample_runner_token as sample_runner_token,
    validate_sampling,
)


_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class ChatSessionStore:
    """Persist portable OpenAI-style chat histories as small JSON documents."""

    def __init__(self, model: str | Path, directory: str | Path | None = None):
        self.model = str(Path(model).expanduser().resolve())
        if directory is None:
            directory = application_state_dir() / "chats"
        self.directory = Path(directory).expanduser().resolve()

    @staticmethod
    def new_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{secrets.token_hex(3)}"

    def _path(self, session_id: str) -> Path:
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError("invalid chat session ID")
        return self.directory / f"{session_id}.json"

    @staticmethod
    def _title(messages: Sequence[Mapping[str, object]]) -> str:
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, Mapping) and part.get("type") == "text"
                )
            title = " ".join(str(content).split())
            return title[:77] + ("…" if len(title) > 77 else "") or "Untitled chat"
        return "Untitled chat"

    def save(
        self, session_id: str, messages: Sequence[Mapping[str, object]]
    ) -> Path | None:
        """Atomically save a non-empty conversation and return its path."""
        if not any(message.get("role") != "system" for message in messages):
            return None
        path = self._path(session_id)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        document = {
            "version": 1,
            "id": session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "title": self._title(messages),
            "messages": list(messages),
        }
        temporary = path.with_suffix(f".{secrets.token_hex(3)}.tmp")
        try:
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.chmod(0o600)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load(self, session_id: str) -> dict[str, object]:
        """Load and validate one saved conversation."""
        path = self._path(session_id)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid saved chat '{path}'") from exc
        if (
            not isinstance(document, dict)
            or document.get("version") != 1
            or document.get("id") != session_id
            or not isinstance(document.get("messages"), list)
        ):
            raise ValueError(f"Invalid saved chat '{path}'")
        for message in document["messages"]:
            if not isinstance(message, dict) or not isinstance(
                message.get("role"), str
            ):
                raise ValueError(f"Invalid saved chat '{path}'")
        return document

    def list_sessions(self) -> list[dict[str, object]]:
        """Return valid saved-chat summaries, newest first."""
        if not self.directory.is_dir():
            return []
        sessions = []
        for path in self.directory.glob("*.json"):
            try:
                session = self.load(path.stem)
            except (OSError, ValueError):
                continue
            sessions.append(
                {
                    key: session.get(key)
                    for key in ("id", "updated_at", "model", "title")
                }
            )
        return sorted(
            sessions, key=lambda session: str(session["updated_at"]), reverse=True
        )


class Tokenizer(Protocol):
    eos_token_id: int
    eos_token_ids: tuple[int, ...]
    tool_call_start: str | None

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] | None = None,
        enable_thinking: bool = False,
        reasoning_effort: str | None = None,
        preserve_thinking: bool = True,
    ) -> list[int]: ...

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = False
    ) -> str: ...

    def token_bytes(
        self, token_id: int, skip_special_tokens: bool = False
    ) -> bytes: ...

    def parse_tool_calls(self, text: str) -> tuple[str, list[dict[str, object]]]: ...


class Runner(Protocol):
    cache_capacity: int

    def prefill(self, token_ids: Sequence[int]) -> Any: ...

    def append(self, token_ids: Sequence[int]) -> Any: ...

    def decode(self, token_id: int) -> Any: ...

    def sample_greedy(self, logits: Any) -> int: ...


class ChatEncodingCache:
    """Encode only the suffix added to an unchanged rendered chat history."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self._incremental = callable(
            getattr(tokenizer, "format_chat", None)
        ) and callable(getattr(tokenizer, "encode", None))
        self.reset()

    def reset(self) -> None:
        """Forget the cached rendered and tokenized history."""
        self._rendered = ""
        self._token_ids: list[int] = []

    def extend_raw(self, token_ids: Sequence[int]) -> None:
        """Append exact model-generated IDs to the cached assistant prefix."""
        if not token_ids or not self._incremental:
            return
        if not self._rendered:
            raise ValueError("encode_chat must be called before extend_raw")
        ids = [int(token_id) for token_id in token_ids]
        self._rendered += self.tokenizer.decode(ids, skip_special_tokens=False)
        self._token_ids.extend(ids)

    def encode_chat(
        self, messages: Sequence[Mapping[str, object]], **kwargs: Any
    ) -> list[int]:
        """Return exact chat IDs, reusing an unchanged rendered prefix when possible."""
        if not self._incremental:
            return self.tokenizer.encode_chat(messages, **kwargs)
        rendered = self.tokenizer.format_chat(messages, **kwargs)
        if self._rendered and rendered.startswith(self._rendered):
            token_ids = self._token_ids + self.tokenizer.encode(
                rendered[len(self._rendered) :]
            )
        else:
            token_ids = self.tokenizer.encode_chat(messages, **kwargs)
        self._rendered = rendered
        self._token_ids = list(token_ids)
        return list(token_ids)

    def encode_continuation(
        self,
        prefix: Sequence[Mapping[str, object]],
        suffix: Sequence[Mapping[str, object]],
        **kwargs: Any,
    ) -> list[int]:
        """Append a verified chat suffix after exact generated token IDs."""
        if not self._incremental or not self._rendered:
            return self.encode_chat([*prefix, *suffix], **kwargs)
        prefix_rendered = self.tokenizer.format_chat(
            prefix, add_generation_prompt=False, **kwargs
        )
        rendered = self.tokenizer.format_chat([*prefix, *suffix], **kwargs)
        if not rendered.startswith(prefix_rendered):
            return self.encode_chat([*prefix, *suffix], **kwargs)
        rendered_suffix = rendered[len(prefix_rendered) :]
        token_ids = self._token_ids + self.tokenizer.encode(rendered_suffix)
        self._rendered += rendered_suffix
        self._token_ids = list(token_ids)
        return list(token_ids)


def is_eos_token(tokenizer: Tokenizer, token_id: int) -> bool:
    """Return whether ``token_id`` is any model-declared end token."""
    return token_id in getattr(tokenizer, "eos_token_ids", (tokenizer.eos_token_id,))


def split_tool_prefix(text: str, marker: str) -> tuple[str, str, bool]:
    """Split streamable text from a possible structured tool-call prefix."""
    start = text.find(marker)
    if start >= 0:
        return text[:start], text[start:], True
    keep = min(len(text), len(marker) - 1)
    while keep and not marker.startswith(text[-keep:]):
        keep -= 1
    return (text[:-keep], text[-keep:], False) if keep else (text, "", False)


class ReasoningStream:
    """Strip a Qwen reasoning prefix without buffering the whole response.

    The prompt has already opened ``<think>``. This presentation-only decoder
    never changes generated tokens or the model's KV/recurrent state.
    """

    def __init__(self):
        self.complete = False
        self._tail = ""
        self._trim = True

    def feed(self, text: str, *, final: bool = False) -> str:
        if not self.complete:
            self._tail += text
            _, marker, text = self._tail.partition("</think>")
            if not marker:
                self._tail = "" if final else self._tail[-7:]
                return ""
            self._tail = ""
            self.complete = True
        if self._trim:
            text = text.lstrip()
            self._trim = not bool(text)
        return text


def split_reasoning(text: str, enable_thinking: bool) -> tuple[str, str | None]:
    """Separate a tagged thinking response into answer and reasoning text."""
    if not enable_thinking:
        return text, None
    reasoning, marker, answer = text.partition("</think>")
    return (answer.lstrip(), reasoning.strip()) if marker else ("", reasoning.strip())


class TokenGeneration:
    """One sampling policy for ordinary, MTP, and DFlash token generation.

    Runners own model state; this iterator owns only sampling history and pending
    verified tokens. ``pending`` tells callers whether early stopping discarded
    an already evaluated speculative suffix and requires invalidating their cache.
    """

    def __init__(
        self,
        runner,
        tokenizer,
        logits,
        limit,
        *,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        presence_penalty=0.0,
        rng=None,
        use_dflash=False,
        use_mtp=False,
        cancelled=None,
    ):
        validate_sampling(temperature, top_k, top_p, presence_penalty)
        if use_dflash and use_mtp:
            raise ValueError("choose only one speculative decoder")
        self.runner, self.tokenizer, self.logits = runner, tokenizer, logits
        self.limit = limit
        self.options = dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
        )
        self.rng = rng if rng is not None else np.random.default_rng()
        self.use_dflash, self.use_mtp = use_dflash, use_mtp
        self.cancelled = cancelled
        self.generated = []
        self.pending = deque()
        self.finished = False

    def _sample(self, logits, accepted=()):
        return sample_runner_token(
            self.runner,
            logits,
            **self.options,
            previous_tokens=(*self.generated, *accepted)
            if accepted
            else self.generated,
            rng=self.rng,
        )

    def __iter__(self):
        return self

    def __next__(self):
        if (
            self.finished
            or len(self.generated) >= self.limit
            or (self.cancelled and self.cancelled())
        ):
            raise StopIteration
        token = self.pending.popleft() if self.pending else self._sample(self.logits)
        self.generated.append(token)
        if is_eos_token(self.tokenizer, token):
            self.finished = True
            return token
        if not self.pending:
            remaining = self.limit - len(self.generated)
            sample = (
                self._sample
                if self.options["temperature"] > 0 or self.options["presence_penalty"]
                else None
            )
            if self.use_dflash and remaining >= self.runner.dflash.block_size:
                self.pending.extend(self.runner.decode_dflash(token, sample=sample)[0])
            elif self.use_mtp and remaining >= 3:
                self.pending.extend(
                    self.runner.decode_speculative(token, sample=sample)[0]
                )
            else:
                self.logits = self.runner.decode(token)
        return token


def generate_tokens(
    runner: Runner,
    tokenizer: Tokenizer,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    seed: int | None = None,
    initial_logits: Any | None = None,
    *,
    use_dflash: bool = False,
    use_mtp: bool = False,
) -> Iterator[int]:
    """Generate tokens incrementally through the common stateful runner API."""
    logits = runner.prefill(prompt_ids) if initial_logits is None else initial_logits
    yield from TokenGeneration(
        runner,
        tokenizer,
        logits,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        presence_penalty=presence_penalty,
        rng=np.random.default_rng(seed),
        use_dflash=use_dflash,
        use_mtp=use_mtp,
    )
