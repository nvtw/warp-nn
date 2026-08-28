# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small workspace tool set for local coding agents."""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

from warp_nn.runtime.sandbox import is_sandbox_available, run_sandboxed


_SKIP_DIRECTORIES = {".git", ".hg", ".svn", ".venv", "__pycache__", "node_modules"}
_MAX_OUTPUT = 64 * 1024
_SEARCH_TIMEOUT = 30.0


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
        "List files in the trusted folder.",
        {
            "path": {"type": "string", "default": "."},
            "pattern": {"type": "string", "default": "*"},
            "recursive": {"type": "boolean", "default": True},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
    ),
    _schema(
        "search_files",
        "Find a literal string in trusted-folder text files.",
        {
            "query": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "pattern": {"type": "string", "default": "*"},
            "case_sensitive": {"type": "boolean", "default": False},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        ("query",),
    ),
    _schema(
        "write_file",
        "Create or replace a UTF-8 text file in the trusted folder.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ("path", "content"),
    ),
    _schema(
        "edit_file",
        "Replace one exact text occurrence in a trusted-folder file.",
        {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        ("path", "old_text", "new_text"),
    ),
)

COMMAND_TOOL_SCHEMA = _schema(
    "run_command",
    "Run a shell command with writes confined to the trusted folder.",
    {
        "command": {"type": "string"},
        "timeout": {"type": "number", "minimum": 0.1, "maximum": 300},
    },
    ("command",),
)


class CodingTools:
    """Execute coding tools within one trusted folder."""

    def __init__(self, root: str | Path, shell: str = "sandbox"):
        if shell not in ("sandbox", "unsafe", "none"):
            raise ValueError("shell must be 'sandbox', 'unsafe', or 'none'")
        self.root = Path(root).resolve()
        self.shell = shell
        self.shell_available = shell == "unsafe" or (shell == "sandbox" and is_sandbox_available())
        self.schemas = FILE_TOOL_SCHEMAS + ((COMMAND_TOOL_SCHEMA,) if self.shell_available else ())
        self._rg = shutil.which("rg")

    def execute(
        self, name: str, arguments: Mapping[str, object], cancelled: Callable[[], bool] | None = None
    ) -> str:
        """Run a named tool and return a bounded text result."""
        methods = {
            "read_file": self._read,
            "list_files": self._list,
            "search_files": self._search,
            "write_file": self._write,
            "edit_file": self._edit,
        }
        if self.shell_available:
            methods["run_command"] = self._command
        try:
            method = methods.get(name)
            if method is None:
                raise ValueError(f"unknown tool {name!r}")
            keywords = dict(arguments)
            if name == "search_files":
                keywords["_cancelled"] = cancelled
            return method(**keywords)[:_MAX_OUTPUT]
        except Exception as error:
            return f"Error: {error}"

    def _path(self, value: object) -> Path:
        if not isinstance(value, str):
            raise ValueError("path must be a string")
        path = (self.root / value).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("path is outside the trusted folder") from error
        return path

    def _walk(self, path: Path, recursive: bool = True) -> Iterator[Path]:
        pending = [path]
        while pending:
            current = pending.pop()
            if current.is_file():
                yield current
                continue
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        if recursive and entry.name not in _SKIP_DIRECTORIES:
                            pending.append(Path(entry.path))
                    else:
                        yield Path(entry.path)

    def _read(self, path, line_start=1, line_end=None):
        path = self._path(path)
        start = max(1, int(line_start))
        end = start + 999 if line_end is None else min(int(line_end), start + 999)
        if end < start:
            raise ValueError("line_end must not precede line_start")
        output = []
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if number > end:
                    break
                if number >= start:
                    output.append(f"{number:>6} | {line}")
        return "".join(output) or "(no matching lines)"

    def _list(self, path=".", pattern="*", recursive=True, max_results=200):
        path = self._path(path)
        limit = min(1000, max(1, int(max_results)))
        items = []
        for item in self._walk(path, bool(recursive)):
            relative = item.relative_to(self.root).as_posix()
            if fnmatch.fnmatch(relative, str(pattern)) or fnmatch.fnmatch(item.name, str(pattern)):
                items.append(relative + ("/" if item.is_dir() else ""))
                if len(items) >= limit:
                    break
        return "\n".join(sorted(items)) or "(no matches)"

    def _search(self, query, path=".", pattern="*", case_sensitive=False, max_results=100, _cancelled=None):
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string")
        path = self._path(path)
        limit = min(1000, max(1, int(max_results)))
        if self._rg:
            return self._search_rg(query, path, str(pattern), bool(case_sensitive), limit, _cancelled)
        needle = query if case_sensitive else query.casefold()
        matches = []
        deadline = time.monotonic() + _SEARCH_TIMEOUT
        for file in self._walk(path):
            if _cancelled and _cancelled():
                raise RuntimeError("search cancelled")
            if time.monotonic() >= deadline:
                raise RuntimeError("search timed out")
            relative = file.relative_to(self.root).as_posix()
            if not (fnmatch.fnmatch(relative, str(pattern)) or fnmatch.fnmatch(file.name, str(pattern))):
                continue
            try:
                with file.open("r", encoding="utf-8") as stream:
                    for number, line in enumerate(stream, 1):
                        candidate = line if case_sensitive else line.casefold()
                        if needle in candidate:
                            matches.append(f"{relative}:{number}:{line.rstrip()}")
                            if len(matches) >= limit:
                                return "\n".join(matches)
            except (OSError, UnicodeError):
                continue
        return "\n".join(matches) or "(no matches)"

    def _search_rg(self, query: str, path: Path, pattern: str, case_sensitive: bool, limit: int, cancelled):
        command = [self._rg, "--line-number", "--no-heading", "--color=never", "--fixed-strings", "--glob", pattern]
        if not case_sensitive:
            command.append("--ignore-case")
        command.extend(["--", query, str(path)])
        process = subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        done = threading.Event()
        stopped = threading.Event()
        timed_out = threading.Event()

        def watch():
            deadline = time.monotonic() + _SEARCH_TIMEOUT
            while not done.wait(0.05):
                if cancelled and cancelled():
                    stopped.set()
                    process.terminate()
                    return
                if time.monotonic() >= deadline:
                    timed_out.set()
                    process.terminate()
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        lines = []
        try:
            for line in process.stdout:
                lines.append(line.rstrip())
                if len(lines) >= limit:
                    process.terminate()
                    break
        finally:
            done.set()
            process.wait()
            watcher.join()
        if stopped.is_set():
            raise RuntimeError("search cancelled")
        if timed_out.is_set():
            raise RuntimeError("search timed out")
        if not lines and process.returncode not in (0, 1):
            raise RuntimeError(process.stderr.read().strip() or "rg failed")
        return "\n".join(lines) or "(no matches)"

    def _write(self, path, content):
        path = self._path(path)
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path.relative_to(self.root).as_posix()}"

    def _edit(self, path, old_text, new_text):
        path = self._path(path)
        if not isinstance(old_text, str) or not old_text or not isinstance(new_text, str):
            raise ValueError("old_text must be non-empty and new_text must be a string")
        content = path.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count != 1:
            raise ValueError(f"old_text occurs {count} times; expected exactly once")
        path.write_text(content.replace(old_text, new_text), encoding="utf-8")
        return f"Edited {path.relative_to(self.root).as_posix()}"

    def _command(self, command, timeout=30):
        if not isinstance(command, str) or not command:
            raise ValueError("command must be a non-empty string")
        timeout = min(300.0, max(0.1, float(timeout)))
        if self.shell == "sandbox":
            result = run_sandboxed(command, self.root, timeout)
        else:
            result = subprocess.run(
                command,
                cwd=self.root,
                shell=True,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        output = (result.stdout + result.stderr).strip()
        return f"Exit code: {result.returncode}\n{output}".rstrip()
