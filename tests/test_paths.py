# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from warp_nn.utils.paths import application_state_dir


def _clear_platform_variables(monkeypatch):
    for name in (
        "WARP_NN_STATE_HOME",
        "XDG_STATE_HOME",
        "LOCALAPPDATA",
        "APPDATA",
    ):
        monkeypatch.delenv(name, raising=False)


def test_application_state_dir_linux(monkeypatch, tmp_path):
    _clear_platform_variables(monkeypatch)
    monkeypatch.setattr("warp_nn.utils.paths.sys.platform", "linux")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert application_state_dir() == tmp_path / "state" / "warp-nn"


def test_application_state_dir_windows(monkeypatch, tmp_path):
    _clear_platform_variables(monkeypatch)
    monkeypatch.setattr("warp_nn.utils.paths.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    assert application_state_dir() == tmp_path / "LocalAppData" / "warp-nn"


def test_application_state_dir_macos(monkeypatch):
    _clear_platform_variables(monkeypatch)
    monkeypatch.setattr("warp_nn.utils.paths.sys.platform", "darwin")
    assert application_state_dir() == (
        Path.home() / "Library" / "Application Support" / "warp-nn"
    )


def test_application_state_dir_override(monkeypatch, tmp_path):
    _clear_platform_variables(monkeypatch)
    monkeypatch.setenv("WARP_NN_STATE_HOME", str(tmp_path / "custom"))
    assert application_state_dir() == (tmp_path / "custom").resolve()
