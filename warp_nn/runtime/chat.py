# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared text-generation helpers for stateful language-model runners."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
from typing import Any, Protocol

import numpy as np

from warp_nn.utils.paths import application_state_dir


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


def split_reasoning(text: str, enable_thinking: bool) -> tuple[str, str | None]:
    """Separate a tagged thinking response into answer and reasoning text."""
    if not enable_thinking:
        return text, None
    reasoning, marker, answer = text.partition("</think>")
    return (answer.lstrip(), reasoning.strip()) if marker else ("", reasoning.strip())


def _sample_candidates(
    values: np.ndarray,
    candidates: np.ndarray,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    """Apply host probability policy to an already selected candidate set."""
    values = np.asarray(values, dtype=np.float64) / temperature
    candidates = np.asarray(candidates, dtype=np.int64)
    probabilities = np.exp(values - np.max(values))
    probabilities /= probabilities.sum()
    if top_p < 1.0:
        order = np.argsort(probabilities)[::-1]
        keep = np.cumsum(probabilities[order]) - probabilities[order] < top_p
        candidates = candidates[order[keep]]
        probabilities = probabilities[order[keep]]
        probabilities /= probabilities.sum()
    return int(rng.choice(candidates, p=probabilities))


def sample_token(
    logits: Any,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    previous_tokens: Sequence[int] = (),
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one token from the last logits row on the host."""
    values = logits.numpy() if hasattr(logits, "numpy") else np.asarray(logits)
    values = (
        np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])[-1].copy()
    )
    if temperature <= 0.0:
        return int(np.argmax(values))
    if top_k < 0 or not 0.0 < top_p <= 1.0 or not -2.0 <= presence_penalty <= 2.0:
        raise ValueError("invalid top_k, top_p, or presence_penalty")
    if presence_penalty and previous_tokens:
        seen = np.asarray(tuple(previous_tokens), dtype=np.int64)
        seen = seen[(seen >= 0) & (seen < values.size)]
        values[np.unique(seen)] -= presence_penalty
    candidates = np.arange(values.size)
    if 0 < top_k < values.size:
        candidates = np.argpartition(values, -top_k)[-top_k:]
        values = values[candidates]
    return _sample_candidates(
        values, candidates, temperature, top_p, rng or np.random.default_rng()
    )


def sample_runner_token(
    runner: Runner,
    logits: Any,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    previous_tokens: Sequence[int] = (),
    rng: np.random.Generator | None = None,
) -> int:
    """Sample through a runner's bounded device path when one is available."""
    if temperature > 0.0 and (
        top_k < 0 or not 0.0 < top_p <= 1.0 or not -2.0 <= presence_penalty <= 2.0
    ):
        raise ValueError("invalid top_k, top_p, or presence_penalty")
    if temperature <= 0.0 or (top_k == 1 and presence_penalty == 0.0):
        return runner.sample_greedy(logits)
    read_top_k = getattr(runner, "read_top_k", None)
    if callable(read_top_k) and presence_penalty == 0.0 and 1 < top_k <= 32:
        values, candidates = read_top_k(logits, top_k)
        return _sample_candidates(
            values,
            candidates,
            temperature,
            top_p,
            rng or np.random.default_rng(),
        )
    return sample_token(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        presence_penalty=presence_penalty,
        previous_tokens=previous_tokens,
        rng=rng,
    )


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
) -> Iterator[int]:
    """Generate tokens incrementally through the common stateful runner API."""
    logits = runner.prefill(prompt_ids) if initial_logits is None else initial_logits
    generated = []
    rng = np.random.default_rng(seed)
    for _ in range(max_new_tokens):
        token_id = sample_runner_token(
            runner,
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            previous_tokens=generated,
            rng=rng,
        )
        generated.append(token_id)
        eos = is_eos_token(tokenizer, token_id)
        next_logits = None if eos else runner.decode(token_id)
        yield token_id
        if eos:
            break
        logits = next_logits
