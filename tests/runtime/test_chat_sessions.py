# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from examples.qwen_chat import _portable_history
from warp_nn.runtime.chat import ChatSessionStore


def test_chat_sessions_save_load_and_list(tmp_path):
    directory = tmp_path / "chats"
    store = ChatSessionStore(tmp_path / "model", directory)
    session_id = "20260831T180000Z-abcdef"
    messages = [
        {"role": "system", "content": "Be useful."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/tmp/image.png"},
                {"type": "text", "text": "What is in this image?"},
            ],
        },
        {
            "role": "assistant",
            "content": "A mountain.",
            "_raw_token_ids": [1, 2, 3],
        },
    ]

    path = store.save(session_id, messages)
    assert path == directory / f"{session_id}.json"
    assert path.stat().st_mode & 0o777 == 0o600
    document = store.load(session_id)
    assert document["title"] == "What is in this image?"
    assert document["messages"] == messages
    assert store.list_sessions() == [
        {
            "id": session_id,
            "updated_at": document["updated_at"],
            "model": str((tmp_path / "model").resolve()),
            "title": "What is in this image?",
        }
    ]

    portable = _portable_history(document)
    assert "_raw_token_ids" not in portable[-1]


def test_chat_sessions_ignore_empty_and_invalid_documents(tmp_path):
    store = ChatSessionStore(tmp_path / "model", tmp_path / "chats")
    assert store.save("empty", [{"role": "system", "content": "System"}]) is None
    assert not store.directory.exists()
    with pytest.raises(ValueError, match="session ID"):
        store.load("../outside")

    store.directory.mkdir()
    (store.directory / "broken.json").write_text("not JSON", encoding="utf-8")
    (store.directory / "wrong.json").write_text(
        json.dumps({"version": 1, "id": "different", "messages": []}),
        encoding="utf-8",
    )
    assert store.list_sessions() == []


def test_chat_session_ids_are_unique_and_filename_safe():
    identifiers = {ChatSessionStore.new_id() for _ in range(20)}
    assert len(identifiers) == 20
    assert all(Path(identifier).name == identifier for identifier in identifiers)
