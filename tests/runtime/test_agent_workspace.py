# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import shlex
import subprocess

import pytest

from warp_nn.runtime.services.agent_context import compact_messages
from warp_nn.runtime.services.coding_tools import CodingTools
from warp_nn.runtime.services.sandbox import (
    SandboxLimits,
    is_sandbox_available,
    run_sandboxed,
)
from warp_nn.runtime.services.workspace_files import WorkspaceFiles
from warp_nn.runtime.services.workspace_session import WorkspaceSession


def test_atomic_write_does_not_modify_hard_link_alias(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("original")
    os.link(outside, root / "alias")
    files = WorkspaceFiles(root)
    with pytest.raises(ValueError, match="hard links"):
        files.read_file("alias")
    files.write_file("alias", "new")
    assert outside.read_text() == "original"
    assert (root / "alias").read_text() == "new"
    assert (root / "alias").stat().st_nlink == 1


def test_file_tools_reject_parent_links_protected_paths_and_fifo(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "link").symlink_to(tmp_path, target_is_directory=True)
    os.mkfifo(root / "pipe")
    files = WorkspaceFiles(root)
    for path in (
        "link/escape",
        ".git/config",
        ".agents/rules",
        ".env",
        ".env.local",
        "../escape",
    ):
        with pytest.raises((ValueError, OSError)):
            files.write_file(path, "bad")
    with pytest.raises(ValueError, match="regular file"):
        files.read_file("pipe")
    assert not (tmp_path / "escape").exists()


def test_edits_are_atomic_and_reject_stale_hash(tmp_path):
    path = tmp_path / "script"
    path.write_text("alpha\nbeta\n")
    path.chmod(0o700)
    files = WorkspaceFiles(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="0 times"):
        files.edit_file(
            "script",
            edits=[
                {"old_text": "alpha", "new_text": "changed"},
                {"old_text": "missing", "new_text": "oops"},
            ],
        )
    assert path.read_text() == "alpha\nbeta\n"
    files.edit_file(
        "script",
        expected_sha256=digest,
        edits=[
            {"old_text": "alpha", "new_text": "one"},
            {"old_text": "beta", "new_text": "two"},
        ],
    )
    assert path.read_text() == "one\ntwo\n"
    assert path.stat().st_mode & 0o100
    with pytest.raises(ValueError, match="changed since"):
        files.write_file("script", "stale", expected_sha256=digest)
    assert path.read_text() == "one\ntwo\n"


def test_pages_have_stable_offsets_and_file_read_limits(tmp_path):
    files = WorkspaceFiles(tmp_path)
    for name in ("c", "a", "b"):
        files.write_file(name, "needle\nsecond\n")
    assert files.list_files(max_results=2) == "a\nb\n[More results: use offset=2]"
    assert files.list_files(max_results=2, offset=2) == "c"
    assert "line_start=2" in files.read_file("a", line_end=1)
    (tmp_path / "huge").write_bytes(b"x" * (4 * 1024**2 + 1))
    with pytest.raises(ValueError, match="4 MiB"):
        files.read_file("huge")


def test_disposable_copy_breaks_links_and_protects_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("old")
    os.link(source / "code.py", source / "alias.py")
    (source / "outside").symlink_to(tmp_path)
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("private")
    (source / ".env").write_text("secret")
    (source / "large.asset").write_text("excluded")
    session = WorkspaceSession.create(source, exclude=["*.asset"])
    try:
        assert not (session.root / ".git").exists()
        assert not (session.root / ".env").exists()
        assert not (session.root / "outside").exists()
        assert not (session.root / "large.asset").exists()
        (session.root / "code.py").write_text("new")
        assert (source / "code.py").read_text() == "old"
        assert (session.root / "alias.py").read_text() == "old"
        patch = session.review()
        assert "-old" in patch.read_text() and "+new" in patch.read_text()
        result = subprocess.run(
            ["git", "apply", "--check", str(patch)], cwd=source, capture_output=True
        )
        assert result.returncode == 0, result.stderr
    finally:
        import shutil

        shutil.rmtree(session.directory)


def test_compaction_preserves_instructions_and_complete_tool_pairs():
    messages = [
        {"role": "system", "content": "Keep constraints"},
        {"role": "user", "content": "Fix my parser"},
    ]
    for i in range(12):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "thinking " * 1000,
                    "_raw_token_ids": [99],
                    "tool_calls": [
                        {
                            "id": str(i),
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"parser.py"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": str(i), "content": "contents " * 1000},
            ]
        )

    def encode(items):
        return json.dumps(items)

    result = compact_messages(messages, encode, 15000, "Next: test escaped quotes")
    assert result is not None and len(encode(result)) <= 15000
    assert result[0] == messages[0] and result[1] == messages[1]
    assert "test escaped quotes" in encode(result)
    assert "_raw_token_ids" not in encode(result)
    calls = {call["id"] for m in result for call in m.get("tool_calls", [])}
    assert "11" in calls
    assert all(m["tool_call_id"] in calls for m in result if m["role"] == "tool")
    assert compact_messages(messages, encode, 10) is None


@pytest.mark.skipif(not is_sandbox_available(), reason="Landlock/seccomp unavailable")
def test_isolated_file_worker_and_regex_search(tmp_path):
    tools = CodingTools(tmp_path)
    try:
        for index in range(5):
            assert tools.execute(
                "write_file",
                {"path": f"src/{index}.py", "content": f"value={index}\nnext\n"},
            ).startswith("Wrote")
        result = tools.execute(
            "search_files", {"query": r"value=[0-9]", "regex": True, "max_results": 2}
        )
        assert "src/0.py:1:value=0" in result and "offset=2" in result
        result = tools.execute(
            "search_files", {"query": "value=", "offset": 2, "max_results": 2}
        )
        assert "src/2.py:1:value=2" in result
        assert "protected" in tools.execute(
            "write_file", {"path": ".git/config", "content": "bad"}
        )
        (tmp_path / "parent").symlink_to(tmp_path.parent)
        assert tools.execute("read_file", {"path": "parent/secret"}).startswith(
            "Error:"
        )
    finally:
        tools.close()


@pytest.mark.skipif(not is_sandbox_available(), reason="Landlock/seccomp unavailable")
def test_async_jobs_logs_and_cancel(tmp_path):
    tools = CodingTools(tmp_path, limits=SandboxLimits(log_bytes=120000))
    try:
        code = "import time; print('a'*100000, flush=True); time.sleep(.5); print('DONE', flush=True)"
        result = tools.execute(
            "run_command",
            {"command": "python3 -c " + shlex.quote(code), "wait_seconds": 0},
        )
        assert "Running job:" in result
        job_id = next(iter(tools.jobs))
        result = tools.execute("poll_command", {"job_id": job_id, "wait_seconds": 5})
        assert result.startswith("Exit code: 0") and "Display truncated" in result
        assert "DONE" in tools.execute(
            "read_command_log", {"job_id": job_id, "tail": True, "max_bytes": 100}
        )
        assert tools.jobs[job_id]["log"].stat().st_size > 100000
        tools.execute("run_command", {"command": "sleep 60", "wait_seconds": 0})
        job_id = list(tools.jobs)[-1]
        assert "cancelled" in tools.execute("cancel_command", {"job_id": job_id})
        assert tools.execute(
            "run_command", {"command": "echo no", "timeout": float("nan")}
        ).startswith("Error:")
    finally:
        tools.close()


@pytest.mark.skipif(not is_sandbox_available(), reason="Landlock/seccomp unavailable")
def test_job_resource_budgets_and_hard_link_preflight(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("safe")
    os.link(outside, root / "alias")
    with pytest.raises(ValueError, match="hard-linked"):
        run_sandboxed("echo bad > alias", root, 3)
    assert outside.read_text() == "safe"
    (root / "alias").unlink()
    result = run_sandboxed(
        "python3 -c "
        + shlex.quote(
            "import time; open('big','wb').write(b'x'*2000000); time.sleep(3)"
        ),
        root,
        5,
        limits=SandboxLimits(workspace_bytes=1000000),
    )
    assert "workspace size limit" in result.stdout
    (root / "big").unlink()
    code = "import os,time; os.fork(); data=bytearray(40*1024**2); time.sleep(3)"
    result = run_sandboxed(
        "python3 -c " + shlex.quote(code),
        root,
        5,
        limits=SandboxLimits(memory_bytes=64 * 1024**2),
    )
    assert "job memory/process limit" in result.stdout


@pytest.mark.skipif(not is_sandbox_available(), reason="Landlock/seccomp unavailable")
def test_compiled_artifact_and_read_only_dependencies(tmp_path):
    import shutil

    if not shutil.which("cc"):
        pytest.skip("C compiler unavailable")
    root = tmp_path / "workspace"
    root.mkdir()
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    (dependency / "data").write_text("read-only")
    (root / "hello.c").write_text(
        '#include <stdio.h>\nint main(void) { puts("hello"); return 0; }\n'
    )
    tools = CodingTools(root, read_only=[dependency])
    try:
        result = tools.execute(
            "run_command", {"command": "cc hello.c -o hello", "wait_seconds": 10}
        )
        assert result.startswith("Exit code: 0"), result
        assert tools.execute("set_executable", {"path": "hello"}).startswith(
            "Made executable"
        )
        result = tools.execute("run_command", {"command": "./hello", "wait_seconds": 5})
        assert result.startswith("Exit code: 0\nhello"), result
        command = f"cat {shlex.quote(str(dependency / 'data'))}; echo bad > {shlex.quote(str(dependency / 'data'))}"
        result = tools.execute("run_command", {"command": command, "wait_seconds": 5})
        assert "read-only" in result and "Permission denied" in result
        assert (dependency / "data").read_text() == "read-only"
    finally:
        tools.close()


def test_session_lock_and_log_resume(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    session = WorkspaceSession.create(source)
    with pytest.raises(ValueError, match="already in use"):
        WorkspaceSession(session.directory)
    directory = session.directory
    session.close()
    resumed = WorkspaceSession(directory)
    resumed.close()
    state = tmp_path / "state"
    tools = CodingTools(source, shell="unsafe", state_dir=state)
    result = tools.execute("run_command", {"command": "echo retained"})
    assert result.startswith("Exit code: 0")
    job_id = next(iter(tools.jobs))
    tools.close()
    resumed_tools = CodingTools(source, shell="unsafe", state_dir=state)
    try:
        assert "retained" in resumed_tools.execute(
            "read_command_log", {"job_id": job_id}
        )
        assert "Exit code: 0" in resumed_tools.execute(
            "poll_command", {"job_id": job_id}
        )
        assert "unknown job_id" in resumed_tools.execute(
            "read_command_log", {"job_id": "../outside"}
        )
    finally:
        resumed_tools.close()
        import shutil

        shutil.rmtree(directory)


def test_directory_swap_does_not_redirect_write(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sub").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("protected")
    real_open = os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "sub" and not swapped:
            swapped = True
            (root / "sub").rename(root / "original-sub")
            (root / "sub").symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(OSError):
        WorkspaceFiles(root).write_file("sub/file", "bad")
    assert (outside / "file").read_text() == "protected"
