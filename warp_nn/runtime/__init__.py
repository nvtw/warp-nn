# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

from warp_nn.runtime.gguf import GGUFArchive
from warp_nn.runtime.onnx_runtime import OnnxRuntime
from warp_nn.runtime.openai_server import ChatCompletions, OpenAIHTTPServer
from warp_nn.runtime.muse_glimmer import MuseGlimmerRunner, MuseGlimmerTokenizer, parse_atem_tool_calls
from warp_nn.runtime.nemotron_h import NemotronHRunner
from warp_nn.runtime.qwen3 import Qwen3OnnxRunner, Qwen3Tokenizer, parse_qwen_tool_calls, sample_token
from warp_nn.runtime.qwen35 import Qwen35Runner


def create_text_runner(path, **kwargs):
    """Create the matching text runner for a supported local model directory."""
    path = Path(path)
    if (path / "model.onnx").is_file():
        return Qwen3OnnxRunner(path / "model.onnx", **kwargs)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    runner_types = {"muse_glimmer": MuseGlimmerRunner, "nemotron_h": NemotronHRunner}
    runner_type = runner_types.get(config.get("model_type"), Qwen35Runner)
    return runner_type(path, **kwargs)


def create_tokenizer(path):
    """Create the matching dependency-free tokenizer for a model directory."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    tokenizer_type = MuseGlimmerTokenizer if config.get("model_type") == "muse_glimmer" else Qwen3Tokenizer
    return tokenizer_type(path)


__all__ = [
    "GGUFArchive",
    "OnnxRuntime",
    "ChatCompletions",
    "OpenAIHTTPServer",
    "NemotronHRunner",
    "MuseGlimmerRunner",
    "MuseGlimmerTokenizer",
    "Qwen3OnnxRunner",
    "Qwen35Runner",
    "Qwen3Tokenizer",
    "create_text_runner",
    "create_tokenizer",
    "parse_atem_tool_calls",
    "parse_qwen_tool_calls",
    "sample_token",
]
