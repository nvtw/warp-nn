# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native filesystem locations for warp-nn application state."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def application_state_dir() -> Path:
    """Return the platform-native directory for persistent warp-nn state.

    ``WARP_NN_STATE_HOME`` overrides the complete application directory.
    The path is returned without creating it.
    """
    override = os.environ.get("WARP_NN_STATE_HOME")
    if override:
        return Path(override).expanduser().resolve()
    home = Path.home()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        return (Path(base) if base else home / "AppData" / "Local") / "warp-nn"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "warp-nn"
    state = os.environ.get("XDG_STATE_HOME")
    base = (
        Path(state)
        if state and Path(state).is_absolute()
        else home / ".local" / "state"
    )
    return base / "warp-nn"
