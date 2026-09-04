# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

from examples.ace_step_chat import _parser as ace_parser
from examples.interactive import TerminalProgress, output_path, parse_toggle
from examples.qwen_image_chat import _parser as image_parser


def test_generation_examples_share_interactive_defaults(tmp_path, capsys):
    assert ace_parser().parse_args(["model"]).auto_open
    assert ace_parser().parse_args(["model"]).progress
    assert image_parser().parse_args(["model"]).auto_open
    assert image_parser().parse_args(["model"]).progress
    assert parse_toggle("", True) is False
    assert parse_toggle("on", False) is True
    assert parse_toggle("off", True) is False

    first = output_path(tmp_path, "A bright summer anthem!", ".wav")
    first.touch()
    second = output_path(tmp_path, "A bright summer anthem!", ".wav")
    assert first.name.endswith("-a-bright-summer-anthem.wav")
    assert second.name.endswith("-a-bright-summer-anthem-2.wav")

    progress = TerminalProgress("Diffusion", width=4)
    progress(0, 2)
    progress(2, 2)
    assert capsys.readouterr().out.endswith("2/2\n")
