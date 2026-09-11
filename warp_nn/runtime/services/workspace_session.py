# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disposable repository copies with retained originals and reviewable patches."""

from __future__ import annotations

import argparse
import atexit
import fcntl
import difflib
import fnmatch
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

from warp_nn.runtime.services.workspace_files import MAX_TEXT, SKIP, WorkspaceFiles


class WorkspaceSession:
    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        self.root = self.directory / "workspace"
        self.baseline = self.directory / "baseline"
        if (
            self.root.is_symlink()
            or self.baseline.is_symlink()
            or not self.root.is_dir()
            or not self.baseline.is_dir()
        ):
            raise ValueError("not a coding-agent session directory")
        self._lock = (self.directory / "session.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            raise ValueError("workspace session is already in use") from None
        atexit.register(self.close)

    def close(self):
        self._lock.close()
        atexit.unregister(self.close)

    @classmethod
    def create(cls, source, *, exclude=(), max_bytes=2 * 1024**3):
        source = Path(source).resolve(strict=True)
        if not source.is_dir():
            raise ValueError("source must be a directory")
        directory = Path(tempfile.mkdtemp(prefix="warp-nn-agent-"))
        if directory.is_relative_to(source):
            directory.rmdir()
            raise ValueError("source must not contain the temporary session directory")
        baseline = directory / "baseline"
        baseline.mkdir()
        skipped = []
        total = 0
        try:
            for current, dirs, files, directory_fd in os.fwalk(
                source, follow_symlinks=False
            ):
                relative = Path(current).relative_to(source)

                def omit(name):
                    path = (relative / name).as_posix()
                    return (
                        name in SKIP
                        or name.startswith(".env.")
                        or any(fnmatch.fnmatch(path, p) for p in exclude)
                    )

                dirs[:] = sorted(name for name in dirs if not omit(name))
                for name in sorted(files):
                    path = relative / name
                    if omit(name):
                        continue
                    # Copy bytes, never hard links, links to external files,
                    # devices or pipes. Opening relative to fwalk's FD closes
                    # the enumeration/open symlink race.
                    try:
                        fd = os.open(
                            name,
                            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                            dir_fd=directory_fd,
                        )
                    except OSError:
                        skipped.append(path.as_posix())
                        continue
                    with os.fdopen(fd, "rb") as src:
                        info = os.fstat(src.fileno())
                        if not stat.S_ISREG(info.st_mode):
                            skipped.append(path.as_posix())
                            continue
                        if info.st_size > 256 * 1024**2:
                            raise ValueError(
                                f"{path} exceeds 256 MiB; exclude large assets explicitly"
                            )
                        target = baseline / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open("wb") as dst:
                            while data := src.read(1024 * 1024):
                                total += len(data)
                                if total > max_bytes:
                                    raise ValueError(
                                        "repository copy exceeds workspace budget; use --workspace-exclude"
                                    )
                                dst.write(data)
                        target.chmod(0o700 if info.st_mode & 0o111 else 0o600)
            shutil.copytree(
                baseline, directory / "workspace", copy_function=shutil.copyfile
            )
            # Preserve executable bits without importing metadata or links.
            for current, _, files in os.walk(baseline):
                for name in files:
                    original = Path(current) / name
                    (directory / "workspace" / original.relative_to(baseline)).chmod(
                        original.stat().st_mode & 0o777
                    )
            (directory / "session.json").write_text(
                json.dumps(
                    {
                        "source": str(source),
                        "excluded": list(exclude),
                        "skipped": skipped,
                    },
                    indent=2,
                )
            )
        except Exception:
            shutil.rmtree(directory)
            raise
        return cls(directory)

    def review(self):
        """Write a bounded patch and change list. Never apply changes to source."""
        before, after = WorkspaceFiles(self.baseline), WorkspaceFiles(self.root)
        originals = set(before.walk())
        current = set(after.walk())
        patch = []
        changes = []
        size = 0
        for path in sorted(originals | current):
            if any(ord(character) < 32 for character in path):
                changes.append(f"? {path!r}: unsupported filename; review manually")
                continue
            if (
                path in originals
                and path not in current
                and os.path.lexists(self.root / path)
            ):
                changes.append(
                    f"? {path}: replaced by a link or special file; omitted from patch"
                )
                continue
            try:
                old = before.read_bytes(path) if path in originals else b""
                new = after.read_bytes(path) if path in current else b""
                if old == new and (path in originals) == (path in current):
                    continue
                status = (
                    "M"
                    if path in originals and path in current
                    else "A"
                    if path in current
                    else "D"
                )
                changes.append(f"{status} {path}")
                if b"\0" in old or b"\0" in new:
                    changes.append(f"  Binary file; review directly: workspace/{path}")
                    continue
                lines = difflib.unified_diff(
                    old.decode("utf-8").splitlines(keepends=True),
                    new.decode("utf-8").splitlines(keepends=True),
                    fromfile=f"a/{path}" if path in originals else "/dev/null",
                    tofile=f"b/{path}" if path in current else "/dev/null",
                )
                text = "".join(
                    line
                    if line.endswith("\n")
                    else line + "\n\\ No newline at end of file\n"
                    for line in lines
                )
                if size + len(text) <= 4 * MAX_TEXT:
                    patch.append(text)
                    size += len(text)
                else:
                    changes.append("  Omitted from patch: review size limit reached")
            except (OSError, ValueError, UnicodeError) as error:
                changes.append(f"? {path}: review directly ({error})")
        (self.directory / "changes.patch").write_text("".join(patch))
        (self.directory / "changes.txt").write_text(
            "\n".join(changes) or "No file changes.\n"
        )
        return self.directory / "changes.patch"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    options = parser.parse_args()
    print(WorkspaceSession(options.session).review())
