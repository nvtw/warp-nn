# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linux coding sandbox: Landlock filesystem policy plus a seccomp syscall allowlist.

Requires Landlock ABI 6 and libseccomp. No network, host display, private host
files, or detached children. Resource limits are per process, not cgroup quotas.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


class SandboxUnavailable(RuntimeError):
    """Raised when the host cannot enforce the requested sandbox."""


_MAX_OUTPUT = 64 * 1024
if __name__ != "__main__":
    # Snapshot trusted code and resolve the interpreter before exposing a folder
    # to tools. Editing this repository or a venv symlink cannot replace the next
    # helper, and -S prevents workspace .pth files from running before isolation.
    _HELPER_SOURCE = Path(__file__).read_text(encoding="utf-8")
    _PYTHON_EXECUTABLE = str(Path(sys.executable).resolve())


def _landlock_abi() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(444, None, 0, 1)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def is_sandbox_available() -> bool:
    """Check prerequisites; each execution also fails closed on setup errors."""
    if not sys.platform.startswith("linux"):
        return False
    if os.uname().machine not in ("x86_64", "aarch64"):
        return False
    try:
        return _landlock_abi() >= 6 and ctypes.util.find_library("seccomp") is not None
    except OSError:
        return False


def run_sandboxed(
    command: str, root: str | Path, timeout: float, cancelled=None
) -> subprocess.CompletedProcess[str]:
    """Execute with bounded output, killing the process group on every exit."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("trusted folder must be a directory")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if not is_sandbox_available():
        raise SandboxUnavailable(
            "sandbox requires Linux Landlock ABI >= 6 and libseccomp"
        )
    # Do not inherit credentials, site customization, or display FDs.
    process = subprocess.Popen(
        [
            _PYTHON_EXECUTABLE,
            "-I",
            "-S",
            "-c",
            _HELPER_SOURCE,
            "--isolate",
            str(root),
            str(timeout),
            command,
        ],
        cwd=root,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(root),
            "TMPDIR": str(root),
            "LANG": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
        start_new_session=True,
    )
    output = bytearray()
    reason = None
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                if cancelled and cancelled():
                    reason = "command cancelled"
                    break
                if time.monotonic() >= deadline:
                    reason = "command timed out"
                    break
                if not selector.select(min(0.05, max(0, deadline - time.monotonic()))):
                    if os.waitid(
                        os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                    ):
                        break
                    continue
                data = os.read(process.stdout.fileno(), 8192)
                if not data:
                    break
                remaining = _MAX_OUTPUT - len(output)
                output.extend(data[:remaining])
                if len(data) > remaining:
                    reason = "command output limit exceeded"
                    break
    finally:
        # Do not poll/reap the group leader before killpg: its PID must not be
        # reused, and even successful shells may leave background children.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()
    text = output.decode("utf-8", errors="replace")
    if reason:
        text += f"\n[Sandbox: {reason}]\n"
    return subprocess.CompletedProcess(
        command, process.returncode if not reason else -signal.SIGKILL, text, ""
    )


def _restrict_linux(root: Path) -> None:
    class RulesetAttr(ctypes.Structure):
        _fields_ = [
            ("handled_access_fs", ctypes.c_uint64),
            ("handled_access_net", ctypes.c_uint64),
            ("scoped", ctypes.c_uint64),
        ]

    class PathBeneathAttr(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    if _landlock_abi() < 6:
        raise SandboxUnavailable("Landlock ABI >= 6 is required")
    libc = ctypes.CDLL(None, use_errno=True)
    all_fs = (1 << 16) - 1  # through IOCTL_DEV; includes REFER and TRUNCATE
    read_execute = (1 << 0) | (1 << 2) | (1 << 3)
    attr = RulesetAttr(all_fs, 0, 3)  # scope abstract UNIX sockets and signals
    ruleset = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")

    def allow(path, rights):
        path = Path(path).resolve()
        if not path.exists():
            return
        if not path.is_dir():
            rights &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14) | (1 << 15)
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            entry = PathBeneathAttr(rights, fd)
            if libc.syscall(445, ruleset, 1, ctypes.byref(entry), 0) < 0:
                raise OSError(ctypes.get_errno(), f"landlock_add_rule: {path}")
        finally:
            os.close(fd)

    try:
        allow(root, all_fs)
        for path in ("/usr", "/bin", "/lib", "/lib64", sys.base_prefix):
            allow(path, read_execute)
        for path in ("/etc/ld.so.cache", "/etc/localtime", "/dev/urandom", "/dev/zero"):
            allow(path, 1 << 2)
        allow("/dev/null", (1 << 1) | (1 << 2))
        if libc.prctl(38, 1, 0, 0, 0) < 0 or libc.syscall(446, ruleset, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(ruleset)


def _restrict_syscalls(library: str | None) -> None:
    """Default deny also covers alternate syscall ABIs and future syscalls."""
    if library is None:
        raise SandboxUnavailable("libseccomp is required")
    seccomp = ctypes.CDLL(library, use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    deny = 0x00050000 | errno.EPERM
    allow = 0x7FFF0000
    context = seccomp.seccomp_init(deny)
    if not context:
        raise SandboxUnavailable("seccomp_init failed")
    # Filesystem content access is further restricted by Landlock. Deliberately
    # absent: sockets, IPC, io_uring, metadata mutations (not covered by ABI 6),
    # ptrace, process_vm, signals to other processes, and session/group changes.
    names = """read write readv writev pread64 pwrite64 preadv pwritev
        open openat close close_range dup dup2 dup3 flock
        stat lstat fstat newfstatat statx statfs fstatfs access faccessat faccessat2
        lseek getdents getdents64 readlink readlinkat getcwd chdir fchdir
        mkdir mkdirat rmdir unlink unlinkat rename renameat renameat2
        link linkat symlink symlinkat truncate ftruncate fallocate fsync fdatasync
        mmap mprotect munmap mremap madvise brk msync
        rt_sigaction rt_sigprocmask rt_sigreturn rt_sigpending rt_sigtimedwait
        sigaltstack restart_syscall futex futex_waitv set_robust_list rseq
        arch_prctl set_tid_address getpid getppid gettid getpgrp getpgid getsid
        getuid geteuid getgid getegid getresuid getresgid getgroups
        uname sysinfo getrandom clock_gettime clock_getres gettimeofday time
        nanosleep clock_nanosleep times getrusage getrlimit
        sched_getaffinity sched_yield sched_getparam sched_getscheduler
        sched_get_priority_max sched_get_priority_min getcpu
        poll ppoll select pselect6 epoll_create epoll_create1 epoll_ctl epoll_wait epoll_pwait
        pipe pipe2 eventfd eventfd2
        execve execveat fork vfork wait4 waitid exit exit_group
        sendfile copy_file_range""".split()
    try:
        for name in names:
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and seccomp.seccomp_rule_add(context, allow, number, 0) < 0:
                raise SandboxUnavailable(f"seccomp rule failed: {name}")
        # Metadata reads are not Landlock-confined. Report these optional
        # facilities as unsupported rather than exposing private xattrs.
        for name in (
            "getxattr",
            "lgetxattr",
            "fgetxattr",
            "listxattr",
            "llistxattr",
            "flistxattr",
        ):
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if (
                number >= 0
                and seccomp.seccomp_rule_add(
                    context, 0x00050000 | errno.ENOTSUP, number, 0
                )
                < 0
            ):
                raise SandboxUnavailable(f"seccomp rule failed: {name}")
        # clone3 has an opaque struct, so force libc's filterable clone fallback.
        number = seccomp.seccomp_syscall_resolve_name(b"clone3")
        if (
            number >= 0
            and seccomp.seccomp_rule_add(context, 0x00050000 | errno.ENOSYS, number, 0)
            < 0
        ):
            raise SandboxUnavailable("seccomp clone3 rule failed")

        class ArgCompare(ctypes.Structure):
            _fields_ = [
                ("arg", ctypes.c_uint),
                ("op", ctypes.c_uint),
                ("datum_a", ctypes.c_uint64),
                ("datum_b", ctypes.c_uint64),
            ]

        for name, argument, permitted in (
            ("prlimit64", 0, [0]),  # never alter another process's limits
            (
                "ioctl",
                1,
                [0x5450, 0x5451, 0x5401],
            ),  # FIONCLEX/FIOCLEX and read-only TCGETS
            ("fcntl", 1, [0, 1, 2, 3, 4, 5, 6, 7, 1030]),  # no signal owner / leases
        ):
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number >= 0:
                for value in permitted:
                    condition = ArgCompare(argument, 4, value, 0)  # SCMP_CMP_EQ
                    if (
                        seccomp.seccomp_rule_add(context, allow, number, 1, condition)
                        < 0
                    ):
                        raise SandboxUnavailable(f"seccomp rule failed: {name}")
        # No new namespaces, alternate parent, or legacy CLONE_DETACHED.
        forbidden = 0x7E020000 | 0x00008000 | 0x00400000
        condition = ArgCompare(0, 7, forbidden, 0)  # SCMP_CMP_MASKED_EQ
        number = seccomp.seccomp_syscall_resolve_name(b"clone")
        if (
            number >= 0
            and seccomp.seccomp_rule_add(context, allow, number, 1, condition) < 0
        ):
            raise SandboxUnavailable("seccomp clone rule failed")
        if seccomp.seccomp_load(context) < 0:
            raise SandboxUnavailable("seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)


def _run_isolated(root: str, timeout: float, command: str) -> None:
    import resource

    root_path = Path(root).resolve(strict=True)
    os.chdir(root_path)
    library = ctypes.util.find_library("seccomp")
    # NPROC counts this UID's existing threads, including the inference runner.
    # Permit a small number of additional tasks rather than a host-dependent
    # fixed limit that could prevent even the first shell child from starting.
    tasks = 0
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            try:
                if entry.stat().st_uid == os.getuid():
                    tasks += len(list((entry / "task").iterdir()))
            except OSError:
                pass
    for limit, value in (
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_FSIZE, 16 * 1024 * 1024),
        (resource.RLIMIT_NPROC, tasks + 32),
        (resource.RLIMIT_AS, 2 * 1024**3),
        (resource.RLIMIT_NOFILE, 128),
        (resource.RLIMIT_CPU, max(1, math.ceil(timeout))),
    ):
        resource.setrlimit(limit, (value, value))
    # Install both policies before any untrusted code executes.
    _restrict_linux(root_path)
    _restrict_syscalls(library)
    os.execv("/bin/sh", ["sh", "-c", command])


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--isolate":
    _run_isolated(sys.argv[2], float(sys.argv[3]), sys.argv[4])
