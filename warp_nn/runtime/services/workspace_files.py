# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded file operations, also executed as a standalone isolated worker.

Only standard-library imports: the launcher snapshots this source before tools
can modify the workspace. Never import code from the workspace in this worker.
"""

from __future__ import annotations

from contextlib import contextmanager
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile


PROTECTED = {".git", ".hg", ".svn", ".agents", ".codex", ".env", ".ssh", ".aws"}
SKIP = PROTECTED | {
    ".venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
}
MAX_TEXT = 4 * 1024 * 1024
MAX_OUTPUT = 40 * 1024


class WorkspaceFiles:
    def __init__(self, root, rg=None):
        self.root = Path(root).resolve()
        self.rg = rg

    def parts(self, path):
        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if any(ord(character) < 32 for character in path):
            raise ValueError("control characters are not supported in file paths")
        value = Path(path)
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("path is outside the trusted folder")
        if any(p in PROTECTED or p.startswith(".env.") for p in value.parts):
            raise ValueError("protected workspace path")
        return value.parts

    @contextmanager
    def parent(self, path, create=False):
        parts = self.parts(path)
        if not parts:
            raise ValueError("a file path is required")
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, dir_fd=fd)
                    except FileExistsError:
                        pass
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                os.close(fd)
                fd = child
            yield fd, parts[-1]
        finally:
            os.close(fd)

    def read_bytes(self, path):
        with self.parent(path) as (parent, name):
            fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("expected a regular file without hard links")
                if info.st_size > MAX_TEXT:
                    raise ValueError(
                        "file exceeds the 4 MiB text-tool limit; use a command for large files"
                    )
                data = stream.read(MAX_TEXT + 1)
                if len(data) > MAX_TEXT:
                    raise ValueError("file exceeds the text-tool limit")
                return data

    def replace(self, path, data, expected_sha256=None):
        if len(data) > MAX_TEXT:
            raise ValueError("content exceeds the 4 MiB text-tool limit")
        with self.parent(path, create=True) as (parent, name):
            if expected_sha256 is not None:
                current = hashlib.sha256(self.read_bytes(path)).hexdigest()
                if current != expected_sha256:
                    raise ValueError("file changed since it was read; read it again")
            # Replace the directory entry rather than truncating an inode: an
            # existing hard link cannot modify another alias outside the root.
            temporary = ".agent-write-" + os.urandom(12).hex()
            try:
                info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                mode = info.st_mode & 0o777 if stat.S_ISREG(info.st_mode) else 0o600
            except FileNotFoundError:
                mode = 0o600
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                mode,
                dir_fd=parent,
            )
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass

    def set_executable(self, path):
        """Copy to a fresh executable inode; never chmod an existing alias."""
        with self.parent(path) as (parent, name):
            source = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
            with os.fdopen(source, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_size > 256 * 1024**2
                ):
                    raise ValueError(
                        "expected an unlinked regular artifact of at most 256 MiB"
                    )
                temporary = ".agent-exec-" + os.urandom(12).hex()
                target = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o700,
                    dir_fd=parent,
                )
                try:
                    with os.fdopen(target, "wb") as output:
                        copied = 0
                        while data := stream.read(1024 * 1024):
                            copied += len(data)
                            if copied > 256 * 1024**2:
                                raise ValueError("artifact exceeds 256 MiB")
                            output.write(data)
                    os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
                finally:
                    try:
                        os.unlink(temporary, dir_fd=parent)
                    except FileNotFoundError:
                        pass
        return f"Made executable: {path}"

    def read_file(self, path, line_start=1, line_end=None):
        data = self.read_bytes(path)
        lines = data.decode("utf-8", errors="replace").splitlines()
        start = max(1, int(line_start))
        end = min(
            len(lines),
            start + 999,
            int(line_end) if line_end is not None else start + 999,
        )
        output = [f"SHA256: {hashlib.sha256(data).hexdigest()}\n"]
        size = len(output[0])
        last = start - 1
        for number in range(start, end + 1):
            line = f"{number:>6} | {lines[number - 1]}\n"
            if len(line.encode("utf-8")) > MAX_OUTPUT:
                raise ValueError(
                    "line exceeds display limit; inspect it with a command"
                )
            if size + len(line.encode("utf-8")) > MAX_OUTPUT:
                break
            output.append(line)
            size += len(line.encode("utf-8"))
            last = number
        if last < len(lines):
            output.append(f"[More lines: use line_start={max(start, last + 1)}]\n")
        return "".join(output)

    def walk(self, path=".", recursive=True):
        parts = self.parts(path)
        # Walk with directory FDs and never follow links, even if the directory
        # tree changes between enumeration and opening an entry.
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                os.close(fd)
                fd = child
            yield from self._walk_fd(fd, Path(*parts), recursive)
        finally:
            os.close(fd)

    def _walk_fd(self, fd, prefix, recursive):
        with os.scandir(fd) as entries:
            names = sorted((e.name, e.is_dir(follow_symlinks=False)) for e in entries)
        for name, directory in names:
            if name in SKIP or name.startswith(".env."):
                continue
            relative = prefix / name
            try:
                if directory and recursive:
                    child = os.open(
                        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                    )
                    try:
                        yield from self._walk_fd(child, relative, recursive)
                    finally:
                        os.close(child)
                elif not directory:
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        yield relative.as_posix()
            except (FileNotFoundError, NotADirectoryError):
                continue

    @staticmethod
    def page(items, offset, max_results):
        offset = max(0, int(offset))
        limit = min(1000, max(1, int(max_results)))
        output = []
        size = 0
        for index, item in enumerate(items):
            if index < offset:
                continue
            item = item[:8192]
            if (
                len(output) >= limit
                or size + len(item.encode("utf-8")) + 1 > MAX_OUTPUT
            ):
                output.append(f"[More results: use offset={offset + len(output)}]")
                break
            output.append(item)
            size += len(item.encode("utf-8")) + 1
        return "\n".join(output) or "(no matches)"

    def list_files(
        self, path=".", pattern="*", recursive=True, max_results=200, offset=0
    ):
        return self.page(
            (
                p
                for p in self.walk(path, recursive)
                if fnmatch.fnmatch(p, pattern) or fnmatch.fnmatch(Path(p).name, pattern)
            ),
            offset,
            max_results,
        )

    def search_files(
        self,
        query,
        path=".",
        pattern="*",
        case_sensitive=False,
        max_results=100,
        offset=0,
        regex=False,
        context_lines=0,
    ):
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string")
        self.parts(path)
        context = min(5, max(0, int(context_lines)))
        # rg's linear-time regex engine avoids Python regex denial of service.
        # It runs inside the same sandbox, with configuration disabled.
        rg = self.rg or next(
            (p for p in ("/usr/bin/rg", "/usr/local/bin/rg") if Path(p).is_file()), None
        )
        if rg:
            command = [
                rg,
                "--no-config",
                "--sort",
                "path",
                "--line-number",
                "--with-filename",
                "--no-heading",
                "--color=never",
                "--max-columns",
                "2048",
                "--max-columns-preview",
                "--glob",
                pattern,
                "--context",
                str(context),
            ]
            for excluded in sorted(SKIP):
                command.extend(
                    ["--glob", f"!**/{excluded}/**", "--glob", f"!**/{excluded}"]
                )
            command.extend(["--glob", "!**/.env.*"])
            if not case_sensitive:
                command.append("--ignore-case")
            if not regex:
                command.append("--fixed-strings")
            # Explicit file arguments can follow a symlink; reject those before
            # handing the path to rg. Directory links are never followed by rg.
            target = self.root / path
            if not target.is_dir():
                self.read_bytes(path)
            command.extend(["--", query, path])
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    stdout=subprocess.PIPE,
                    stderr=errors,
                    text=True,
                )
                try:
                    result = self.page(
                        (line.rstrip() for line in process.stdout), offset, max_results
                    )
                finally:
                    process.stdout.close()
                    # Closing stdout lets rg exit on EPIPE without requiring
                    # signal privileges in the sandbox.
                    process.wait()
                if (
                    process.returncode not in (0, 1, -13)
                    and not result.startswith("[More")
                    and result == "(no matches)"
                ):
                    errors.seek(0)
                    raise ValueError(errors.read(4096).decode(errors="replace"))
                return result
        if regex:
            raise ValueError("regex search requires the system ripgrep executable")
        needle = query if case_sensitive else query.casefold()

        def matches():
            for file in self.walk(path):
                if not (
                    fnmatch.fnmatch(file, pattern)
                    or fnmatch.fnmatch(Path(file).name, pattern)
                ):
                    continue
                try:
                    data = self.read_bytes(file)
                    if b"\0" in data:
                        continue
                    lines = data.decode("utf-8").splitlines()
                    shown = set()
                    for index, line in enumerate(lines):
                        if needle in (line if case_sensitive else line.casefold()):
                            for near in range(
                                max(0, index - context),
                                min(len(lines), index + context + 1),
                            ):
                                if near not in shown:
                                    shown.add(near)
                                    yield f"{file}:{near + 1}:{lines[near][:2048]}"
                except (OSError, ValueError):
                    continue

        return self.page(matches(), offset, max_results)

    def write_file(self, path, content, expected_sha256=None):
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        self.replace(path, content.encode(), expected_sha256)
        return f"Wrote {len(content)} characters to {path}"

    def edit_file(
        self, path, old_text=None, new_text=None, edits=None, expected_sha256=None
    ):
        data = self.read_bytes(path)
        text = data.decode("utf-8")
        if edits is not None and (old_text is not None or new_text is not None):
            raise ValueError("provide edits or old_text/new_text, not both")
        replacements = (
            edits
            if edits is not None
            else [{"old_text": old_text, "new_text": new_text}]
        )
        if (
            not isinstance(replacements, list)
            or not replacements
            or len(replacements) > 100
        ):
            raise ValueError("provide between 1 and 100 edits")
        for edit in replacements:
            old, new = edit["old_text"], edit["new_text"]
            if not isinstance(old, str) or not old or not isinstance(new, str):
                raise ValueError(
                    "old_text must be non-empty and new_text must be a string"
                )
            if text.count(old) != 1:
                raise ValueError(
                    f"old_text occurs {text.count(old)} times; expected exactly once"
                )
            text = text.replace(old, new, 1)
        self.replace(
            path, text.encode(), expected_sha256 or hashlib.sha256(data).hexdigest()
        )
        return f"Edited {path} ({len(replacements)} replacements)"


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        worker = WorkspaceFiles(Path.cwd(), rg=request.get("rg"))
        if request["name"] not in {
            "read_file",
            "list_files",
            "search_files",
            "write_file",
            "edit_file",
            "set_executable",
        }:
            raise ValueError("unknown file tool")
        print(getattr(worker, request["name"])(**request["arguments"]))
    except Exception as error:
        print(f"Error: {error}")
