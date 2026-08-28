# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess

from warp_nn.runtime.coding_tools import CodingTools


def test_coding_tools_read_search_write_and_edit(tmp_path):
    tools = CodingTools(tmp_path)
    assert tools.execute("write_file", {"path": "src/code.py", "content": "alpha\nbeta\n"}).startswith("Wrote")
    assert "2 | beta" in tools.execute("read_file", {"path": "src/code.py", "line_start": 2})
    assert "code.py:1:alpha" in tools.execute("search_files", {"query": "ALPHA", "path": "src"})
    assert tools.execute(
        "edit_file", {"path": "src/code.py", "old_text": "beta", "new_text": "gamma"}
    ).startswith("Edited")
    assert (tmp_path / "src" / "code.py").read_text() == "alpha\ngamma\n"
    assert "src/code.py" in tools.execute("list_files", {"path": ".", "pattern": "*.py"})


def test_coding_tools_reject_paths_outside_trusted_folder(tmp_path):
    tools = CodingTools(tmp_path / "workspace")
    tools.root.mkdir()
    assert tools.execute("read_file", {"path": "../outside.txt"}).startswith("Error: path is outside")


def test_coding_tools_run_command(tmp_path):
    result = CodingTools(tmp_path, shell="unsafe").execute("run_command", {"command": "echo hello"})
    assert result.startswith("Exit code: 0")
    assert "hello" in result


def test_coding_tools_hide_unavailable_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr("warp_nn.runtime.coding_tools.is_sandbox_available", lambda: False)
    tools = CodingTools(tmp_path)
    assert "run_command" not in {schema["function"]["name"] for schema in tools.schemas}
    assert "unknown tool" in tools.execute("run_command", {"command": "echo hello"})


def test_coding_tools_use_available_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr("warp_nn.runtime.coding_tools.is_sandbox_available", lambda: True)
    monkeypatch.setattr(
        "warp_nn.runtime.coding_tools.run_sandboxed",
        lambda command, root, timeout: subprocess.CompletedProcess(command, 0, "hello", ""),
    )
    result = CodingTools(tmp_path).execute("run_command", {"command": "echo hello"})
    assert result == "Exit code: 0\nhello"
