# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import time

import pytest

from warp_nn.runtime.services.sandbox import is_sandbox_available, run_sandboxed

pytestmark = pytest.mark.skipif(
    not is_sandbox_available(), reason="Landlock/seccomp unavailable"
)


def run_python(root, code, timeout=5, **kwargs):
    return run_sandboxed("python3 -c " + shlex.quote(code), root, timeout, **kwargs)


def test_sandbox_runs_python_and_writes_only_workspace(tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "escape").symlink_to(outside)
    code = f"""
from pathlib import Path
import os
Path('ok.py').write_text('print(42)')
for name in [{str(outside)!r}, 'escape', '/proc/self/environ']:
    try:
        Path(name).read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError(('read escaped', name))
for operation in [lambda: Path({str(outside)!r}).write_text('bad'),
                  lambda: os.truncate({str(outside)!r}, 0),
                  lambda: os.chmod({str(outside)!r}, 0),
                  lambda: os.utime({str(outside)!r}),
                  lambda: Path('escape').write_text('bad')]:
    try:
        operation()
    except PermissionError:
        pass
    else:
        raise AssertionError('write escaped')
print('confined')
"""
    result = run_python(root, code)
    assert result.returncode == 0, result.stdout
    assert "confined" in result.stdout
    assert outside.read_text() == "secret"
    assert (root / "ok.py").read_text() == "print(42)"


def test_sandbox_blocks_sockets_and_process_escape(tmp_path):
    result = run_python(
        tmp_path,
        f"""
import socket, os, resource, fcntl
for family, kind in [(socket.AF_INET, socket.SOCK_STREAM), (socket.AF_INET, socket.SOCK_DGRAM),
                     (socket.AF_UNIX, socket.SOCK_STREAM)]:
    try:
        socket.socket(family, kind)
    except PermissionError:
        pass
    else:
        raise AssertionError('socket permitted')
for operation in [os.setsid, lambda: os.setpgid(0, 0), lambda: os.kill({os.getpid()}, 0),
                  lambda: resource.prlimit({os.getpid()}, resource.RLIMIT_NOFILE),
                  lambda: fcntl.fcntl(1, fcntl.F_SETOWN, {os.getpid()})]:
    try:
        operation()
    except PermissionError:
        pass
    else:
        raise AssertionError('process escape permitted')
""",
    )
    assert result.returncode == 0, result.stdout


def test_sandbox_does_not_inherit_secrets_or_fds(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_TEST_SECRET", "secret")
    result = run_python(
        tmp_path,
        "import os; assert 'SANDBOX_TEST_SECRET' not in os.environ; print(os.read(0, 1))",
    )
    assert result.returncode == 0, result.stdout
    assert "b''" in result.stdout


def test_sandbox_output_timeout_and_cancel_are_bounded(tmp_path):
    result = run_python(
        tmp_path, "print('x'*100000); open('finished', 'w').write('yes')"
    )
    assert result.returncode == 0
    assert (tmp_path / "finished").read_text() == "yes"
    assert "Display truncated" in result.stdout
    assert len(result.stdout) < 66000
    start = time.monotonic()
    result = run_python(tmp_path, "import time; time.sleep(60)", timeout=0.3)
    assert "timed out" in result.stdout
    assert time.monotonic() - start < 3
    result = run_python(tmp_path, "import time; time.sleep(60)", cancelled=lambda: True)
    assert "cancelled" in result.stdout


def test_sandbox_kills_background_children_on_success(tmp_path):
    code = "import time; time.sleep(.5); open('escaped','w').write('bad')"
    command = "python3 -c " + shlex.quote(code) + " & echo done"
    result = run_sandboxed(command, tmp_path, 5)
    assert result.returncode == 0, result.stdout
    time.sleep(0.7)
    assert not (tmp_path / "escaped").exists()


def test_sandbox_executes_python_file_and_returns_runtime_errors(tmp_path):
    script = tmp_path / "program.py"
    script.write_text("print(missing)\n")
    result = run_sandboxed("python3 program.py", tmp_path, 5)
    assert result.returncode == 1, result.stdout
    assert "NameError" in result.stdout
    script.write_text("print(42)\n")
    result = run_sandboxed("python3 program.py", tmp_path, 5)
    assert result.returncode == 0, result.stdout
    assert result.stdout.strip() == "42"


def test_sandbox_preserves_failure_in_output_pipeline(tmp_path):
    result = run_sandboxed("false | cat", tmp_path, 5)
    assert result.returncode == 1, result.stdout
