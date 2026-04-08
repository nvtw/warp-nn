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

from __future__ import annotations

import contextlib
import io

import warp as wp


__all__ = ["__version__", "print_diagnostics"]


# read library version from metadata
try:
    import importlib.metadata

    __version__ = importlib.metadata.version("warp_nn")
except ImportError:
    __version__ = "unknown"


def print_diagnostics() -> dict:
    """Print the diagnostic information of the library.

    :return: Dictionary containing the diagnostic information.
    """

    def _field(label, value, indent=2):
        lines.append(f"{' ' * indent}{label:<{w}}{value}")

    def _section(title):
        if lines:
            lines.append("")
        lines.append(title)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        info = wp.print_diagnostics()
        info["warp_nn"] = __version__
    diagnostics = buffer.getvalue()

    w = 18
    lines = []
    _section("Library")
    _field("Warp-NN:", info["warp_nn"])

    print(f"\n{chr(10).join(lines)}\n\n{diagnostics}")
    return info
