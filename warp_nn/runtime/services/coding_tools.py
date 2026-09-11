# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small coding-tool facade; untrusted filesystem operations run in isolation."""

from __future__ import annotations

import atexit
import json
import math
import shutil
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid

from warp_nn.runtime.services import workspace_files
from warp_nn.runtime.services.sandbox import (
    SandboxLimits,
    is_sandbox_available,
    run_sandboxed,
    _PYTHON_EXECUTABLE,
)

_WORKER_SOURCE = Path(workspace_files.__file__).read_text(encoding="utf-8")
_RG_PATH = shutil.which("rg")
_RG_PATH = str(Path(_RG_PATH).resolve()) if _RG_PATH else None


def _schema(name, description, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


FILE_TOOL_SCHEMAS = (
    _schema(
        "set_executable",
        "Make a compiled workspace artifact or script executable safely. Use this if a compiled program reports Permission denied; chmod is blocked.",
        {"path": {"type": "string"}},
        ("path",),
    ),
    _schema(
        "read_file",
        "Read numbered lines from a UTF-8 text file in the trusted folder.",
        {
            "path": {"type": "string"},
            "line_start": {"type": "integer", "minimum": 1},
            "line_end": {"type": "integer", "minimum": 1},
        },
        ("path",),
    ),
    _schema(
        "list_files",
        "List files deterministically. Use offset to retrieve subsequent pages.",
        {
            "path": {"type": "string", "default": "."},
            "pattern": {"type": "string", "default": "*"},
            "recursive": {"type": "boolean", "default": True},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
            "offset": {"type": "integer", "minimum": 0},
        },
    ),
    _schema(
        "search_files",
        "Search text files, optionally using regex and surrounding lines. Use offset for more results.",
        {
            "query": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "pattern": {"type": "string", "default": "*"},
            "case_sensitive": {"type": "boolean", "default": False},
            "regex": {"type": "boolean", "default": False},
            "context_lines": {"type": "integer", "minimum": 0, "maximum": 5},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
            "offset": {"type": "integer", "minimum": 0},
        },
        ("query",),
    ),
    _schema(
        "write_file",
        "Create or replace a UTF-8 text file in the trusted folder.",
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "expected_sha256": {"type": "string"},
        },
        ("path", "content"),
    ),
    _schema(
        "edit_file",
        "Atomically apply one exact replacement or an edits array. All edits must match once; otherwise nothing changes. Use the SHA256 from read_file to reject stale edits.",
        {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "expected_sha256": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["old_text", "new_text"],
                    "additionalProperties": False,
                },
            },
        },
        ("path",),
    ),
)

COMMAND_TOOL_SCHEMA = _schema(
    "run_command",
    "Run a sandboxed shell command in the trusted folder, without network or host display access.",
    {
        "command": {"type": "string"},
        "timeout": {"type": "number", "minimum": 0.1, "maximum": 3600},
        "wait_seconds": {"type": "number", "minimum": 0, "maximum": 10},
    },
    ("command",),
)


PROGRESS_TOOL_SCHEMA = _schema(
    "save_progress",
    "Save a concise durable checkpoint: task, constraints, discoveries, changed files, tests and next steps. Update before a long sequence of tools.",
    {"summary": {"type": "string"}},
    ("summary",),
)

JOB_TOOL_SCHEMAS = (
    _schema(
        "poll_command",
        "Wait briefly for a command job and retrieve its status and output.",
        {
            "job_id": {"type": "string"},
            "wait_seconds": {"type": "number", "minimum": 0, "maximum": 10},
        },
        ("job_id",),
    ),
    _schema(
        "read_command_log",
        "Read a retained command log by byte offset, or tail its last bytes.",
        {
            "job_id": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "max_bytes": {"type": "integer", "minimum": 1, "maximum": 32768},
            "tail": {"type": "boolean"},
        },
        ("job_id",),
    ),
    _schema(
        "cancel_command",
        "Stop a running command and all of its children.",
        {"job_id": {"type": "string"}},
        ("job_id",),
    ),
)


class CodingTools:
    """One workspace, one concurrent command, bounded retained job logs.

    The dedicated coding example supplies a disposable workspace. Direct callers
    must supply a dedicated directory, never the runtime installation or home.
    shell='none' is an explicit file-only mode, not a sandbox fallback.
    """

    def __init__(
        self, root, shell="sandbox", *, limits=None, state_dir=None, read_only=()
    ):
        if shell not in ("sandbox", "unsafe", "none"):
            raise ValueError("shell must be 'sandbox', 'unsafe', or 'none'")
        self.root = Path(root).resolve()
        self.shell = shell
        self.limits = limits or SandboxLimits()
        self.read_only = tuple(
            str(Path(path).resolve(strict=True)) for path in read_only
        )
        self.shell_available = shell == "unsafe" or (
            shell == "sandbox" and is_sandbox_available()
        )
        self.schemas = (
            FILE_TOOL_SCHEMAS
            + (PROGRESS_TOOL_SCHEMA,)
            + (
                (COMMAND_TOOL_SCHEMA,) + JOB_TOOL_SCHEMAS
                if self.shell_available
                else ()
            )
        )
        self.state_dir = (
            Path(state_dir)
            if state_dir
            else Path(tempfile.mkdtemp(prefix="warp-nn-agent-logs-"))
        )
        self.state_dir = self.state_dir.resolve()
        if self.state_dir.is_relative_to(self.root):
            raise ValueError("tool state must be outside the writable workspace")
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._rg = _RG_PATH
        if self._rg and (
            Path(self._rg).is_relative_to(self.root)
            or not Path(self._rg).is_relative_to(Path("/usr"))
        ):
            self._rg = None
        self.jobs = {}
        atexit.register(self.close)

    @property
    def checkpoint(self):
        path = self.state_dir / "progress.txt"
        return path.read_text() if path.exists() else ""

    def close(self):
        for job in self.jobs.values():
            job["cancel"].set()
        for job in self.jobs.values():
            job["thread"].join(timeout=5)
        atexit.unregister(self.close)

    def execute(self, name, arguments, cancelled=None):
        try:
            if cancelled and cancelled():
                return (
                    "Error: search cancelled"
                    if name == "search_files"
                    else "Error: tool cancelled"
                )
            if name == "save_progress":
                summary = arguments.get("summary")
                if not isinstance(summary, str) or len(summary) > 12000:
                    raise ValueError(
                        "summary must be a string of at most 12000 characters"
                    )
                (self.state_dir / "progress.txt").write_text(summary)
                return "Progress checkpoint saved."
            if name in ("write_file", "edit_file", "set_executable") and any(
                job["thread"].is_alive() for job in self.jobs.values()
            ):
                raise ValueError(
                    "a command is still running; poll or cancel it before editing files"
                )
            if name in {schema["function"]["name"] for schema in FILE_TOOL_SCHEMAS}:
                if self.shell == "sandbox":
                    if not is_sandbox_available():
                        raise RuntimeError(
                            "sandbox unavailable; file tools fail closed"
                        )
                    request = json.dumps(
                        {"name": name, "arguments": dict(arguments), "rg": self._rg}
                    )
                    if len(request) > 8 * 1024**2:
                        raise ValueError("tool request too large")
                    result = run_sandboxed(
                        [_PYTHON_EXECUTABLE, "-I", "-S", "-c", _WORKER_SOURCE],
                        self.root,
                        30,
                        cancelled=cancelled,
                        input_text=request,
                        limits=self.limits,
                        read_only=self.read_only,
                    )
                    if result.returncode:
                        return f"Error: file worker exited {result.returncode}\n{result.stdout}"
                    return result.stdout.rstrip()
                return getattr(workspace_files.WorkspaceFiles(self.root), name)(
                    **arguments
                )
            if not self.shell_available:
                raise ValueError(f"unknown tool {name!r}")
            methods = {
                "run_command": self._command,
                "poll_command": self._poll,
                "read_command_log": self._read_log,
                "cancel_command": self._cancel,
            }
            if name not in methods:
                raise ValueError(f"unknown tool {name!r}")
            return methods[name](**arguments, _cancelled=cancelled)
        except Exception as error:
            return f"Error: {error}"

    @staticmethod
    def _duration(value, maximum):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("duration must be finite and non-negative")
        return min(maximum, value)

    def _command(self, command, timeout=300, wait_seconds=2, _cancelled=None):
        if not isinstance(command, str) or not command:
            raise ValueError("command must be a non-empty string")
        timeout = self._duration(timeout, 3600)
        wait_seconds = self._duration(wait_seconds, 10)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if any(job["thread"].is_alive() for job in self.jobs.values()):
            raise ValueError("a command is still running; poll or cancel it first")
        if len(self.jobs) >= 64:
            raise ValueError("session job limit reached (64); start another session")
        job_id = uuid.uuid4().hex[:12]
        log = self.state_dir / (job_id + ".log")
        job = {"cancel": threading.Event(), "result": None, "log": log}

        metadata = self.state_dir / (job_id + ".json")
        metadata.write_text(
            json.dumps(
                {
                    "command": command,
                    "result": "Interrupted before completion; inspect the retained log and rerun checks.",
                }
            )
        )

        def run():
            try:
                if self.shell == "sandbox":
                    result = run_sandboxed(
                        command,
                        self.root,
                        timeout,
                        cancelled=lambda: job["cancel"].is_set()
                        or bool(_cancelled and _cancelled()),
                        log_path=log,
                        limits=self.limits,
                        read_only=self.read_only,
                    )
                else:
                    # Explicit legacy opt-out; never used by the coding example.
                    result = subprocess.run(
                        command,
                        cwd=self.root,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    log.write_text(
                        (result.stdout + result.stderr)[: self.limits.log_bytes]
                    )
                job["result"] = (
                    f"Exit code: {result.returncode}\n{(result.stdout + result.stderr).strip()}"
                )
            except Exception as error:
                job["result"] = f"Error: {error}"
            finally:
                metadata.write_text(
                    json.dumps({"command": command, "result": job["result"]})
                )

        job["thread"] = threading.Thread(target=run, daemon=True)
        self.jobs[job_id] = job
        job["thread"].start()
        return self._poll(job_id, wait_seconds, _cancelled)

    def _job(self, job_id):
        if job_id in self.jobs:
            return self.jobs[job_id]
        if (
            not isinstance(job_id, str)
            or len(job_id) != 12
            or any(c not in "0123456789abcdef" for c in job_id)
        ):
            raise ValueError("unknown job_id")
        metadata = self.state_dir / (job_id + ".json")
        if not metadata.is_file():
            raise ValueError("unknown job_id")
        return {
            "result": json.loads(metadata.read_text())["result"],
            "log": self.state_dir / (job_id + ".log"),
            "cancel": threading.Event(),
            "thread": threading.Thread(),
        }

    def _poll(self, job_id, wait_seconds=2, _cancelled=None):
        job = self._job(job_id)
        deadline = time.monotonic() + self._duration(wait_seconds, 10)
        while job["thread"].is_alive() and time.monotonic() < deadline:
            if _cancelled and _cancelled():
                job["cancel"].set()
                break
            job["thread"].join(timeout=0.05)
        if job["thread"].is_alive():
            return f"Running job: {job_id}. Use poll_command or read_command_log."
        return f"{job['result']}\nJob: {job_id}; log retained for read_command_log."

    def _read_log(self, job_id, offset=0, max_bytes=16384, tail=False, _cancelled=None):
        job = self._job(job_id)
        count = min(32768, max(1, int(max_bytes)))
        if not job["log"].exists():
            return "(no log output yet)"
        with job["log"].open("rb") as stream:
            size = stream.seek(0, 2)
            offset = max(0, size - count) if tail else max(0, int(offset))
            stream.seek(offset)
            data = stream.read(count)
        return (
            f"Bytes {offset}..{offset + len(data)} of {size}; next offset={offset + len(data)}\n"
            + data.decode(errors="replace")
        )

    def _cancel(self, job_id, _cancelled=None):
        self._job(job_id)["cancel"].set()
        return self._poll(job_id, 2)
