# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Small shared helpers for resident interactive generation examples."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_toggle(value: str, current: bool) -> bool:
    """Parse on/off or toggle when no value is supplied."""
    value = value.strip().lower()
    if not value:
        return not current
    if value in ("on", "yes", "true", "1"):
        return True
    if value in ("off", "no", "false", "0"):
        return False
    raise ValueError("expected on or off")


def output_path(directory: str | Path, prompt: str, suffix: str) -> Path:
    """Return a readable, collision-free timestamped output path."""
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:48] or "generation"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = directory / f"{stamp}-{slug}{suffix}"
    index = 2
    while candidate.exists():
        candidate = directory / f"{stamp}-{slug}-{index}{suffix}"
        index += 1
    return candidate


def open_output(path: str | Path) -> bool:
    """Open an output with the platform's default application."""
    path = str(Path(path).resolve())
    if sys.platform == "win32":
        getattr(os, "startfile")(path)
        return True
    command = "open" if sys.platform == "darwin" else "xdg-open"
    executable = shutil.which(command)
    if executable is None:
        return False
    subprocess.Popen(
        [executable, path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return True


class TerminalProgress:
    """One-line progress display driven entirely by an existing host loop."""

    def __init__(self, label: str, enabled: bool = True, width: int = 28):
        self.label = label
        self.enabled = enabled
        self.width = width

    def __call__(self, completed: int, total: int) -> None:
        if not self.enabled:
            return
        filled = min(self.width, self.width * completed // max(1, total))
        bar = "█" * filled + "·" * (self.width - filled)
        end = "\n" if completed >= total else ""
        print(f"\r{self.label} [{bar}] {completed}/{total}", end=end, flush=True)
