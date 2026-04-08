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

import pytest

import warp as wp

from warp_nn.utils import parse_device


@pytest.mark.parametrize("device", [None, "cpu", "cuda", "cuda:0", "cuda:10", "edge-case"])
def test_parse_device(capsys, device):
    target_device = None
    if device in [None, "edge-case"]:
        target_device = wp.get_device()
    elif device.startswith("cuda"):
        try:
            index = int(f"{device}:0".split(":")[1])
            target_device = wp.get_device(f"cuda:{index}")
        except Exception as e:
            target_device = wp.get_device()
    if not target_device:
        target_device = wp.get_device(device)

    runtime_device = parse_device(device)
    assert runtime_device == target_device
