# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free filesystem sandbox for coding-tool subprocesses."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    """Raised when the host cannot enforce the requested sandbox."""


def _landlock_abi() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(444, None, 0, 1)  # landlock_create_ruleset(..., VERSION)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def is_sandbox_available() -> bool:
    """Return whether this host supports write-confined subprocesses."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        return _landlock_abi() >= 1
    except OSError:
        return False


def run_sandboxed(command: str, root: str | Path, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a shell command with writes confined below ``root``."""
    root = Path(root).resolve(strict=True)
    if not is_sandbox_available():
        raise SandboxUnavailable(f"sandboxed commands are unsupported on {sys.platform}")
    return subprocess.run(
        [sys.executable, "-m", __name__, "--landlock", str(root), command],
        cwd=root,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def _restrict_linux(root: Path) -> None:
    class RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathBeneathAttr(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    abi = _landlock_abi()
    writes = (1 << 1) | sum(1 << bit for bit in range(4, 13))
    if abi >= 2:
        writes |= 1 << 13  # REFER
    if abi >= 3:
        writes |= 1 << 14  # TRUNCATE
    libc = ctypes.CDLL(None, use_errno=True)
    ruleset_attr = RulesetAttr(writes)
    ruleset = libc.syscall(444, ctypes.byref(ruleset_attr), ctypes.sizeof(ruleset_attr), 0)
    if ruleset < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    root_fd = os.open(root, os.O_PATH | os.O_CLOEXEC)
    try:
        path_attr = PathBeneathAttr(writes, root_fd)
        if libc.syscall(445, ruleset, 1, ctypes.byref(path_attr), 0) < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if libc.prctl(38, 1, 0, 0, 0) < 0 or libc.syscall(446, ruleset, 0) < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    finally:
        os.close(root_fd)
        os.close(ruleset)


def _run_landlock(root: str, command: str) -> None:
    root_path = Path(root).resolve(strict=True)
    os.chdir(root_path)
    _restrict_linux(root_path)
    os.execv("/bin/sh", ["sh", "-c", command])


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--landlock":
    _run_landlock(sys.argv[2], sys.argv[3])
