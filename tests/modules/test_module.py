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

import hypothesis
import hypothesis.strategies as st
import pytest

import warp as wp

import warp_nn.nn as nn

from .. import utilities


class ModuleA(nn.Module):  # parameters: 0, modules: 0
    def __init__(self):
        super().__init__()
        super().__post_init__()


class ModuleB(nn.Module):  # parameters: 1, modules: 0
    def __init__(self):
        super().__init__()
        self.param_b0 = nn.Parameter(wp.full(shape=(1, 1), value=0.0, dtype=wp.float32))
        super().__post_init__()


class ModuleC(nn.Module):  # parameters: 3 (2 + (1)), modules: 1
    def __init__(self):
        super().__init__()
        self.module_b = ModuleB()
        self.param_c0 = nn.Parameter(wp.full(shape=(2, 2), value=1.0, dtype=wp.float32))
        self.param_c1 = nn.Parameter(wp.full(shape=(3, 3), value=2.0, dtype=wp.float32))
        super().__post_init__()


class ModuleD(nn.Module):  # parameters: 5 (1 + (3 + 1)), modules: 3 (1 + 1 + 1)
    def __init__(self):
        super().__init__()
        self.module_a = ModuleA()
        self.module_b = ModuleB()
        self.module_c = ModuleC()
        self.param_d0 = nn.Parameter(wp.full(shape=(4, 4), value=4.0, dtype=wp.float32))
        super().__post_init__()


@pytest.fixture
def modules():
    return [ModuleA(), ModuleB(), ModuleC(), ModuleD()]


@pytest.mark.parametrize("as_array", [True, False])
@pytest.mark.parametrize("include_submodules", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_parameters(capsys, device, include_submodules, as_array, modules: list[nn.Module]):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    def _check(parameters):
        assert all(
            isinstance(parameter, wp.array) if as_array else isinstance(parameter, nn.Parameter)
            for parameter in parameters
        )
        assert all(parameter.device.is_cuda == (device == "cuda") for parameter in parameters)

    module_a, module_b, module_c, module_d = [module.to(device) for module in modules]
    # number of parameters
    assert len(module_a.parameters(include_submodules=include_submodules, as_array=as_array)) == 0
    assert len(module_b.parameters(include_submodules=include_submodules, as_array=as_array)) == 1
    assert len(module_c.parameters(include_submodules=include_submodules, as_array=as_array)) == (
        3 if include_submodules else 2
    )
    assert len(module_d.parameters(include_submodules=include_submodules, as_array=as_array)) == (
        5 if include_submodules else 1
    )
    # class/device
    _check(module_a.parameters(include_submodules=include_submodules, as_array=as_array))
    _check(module_b.parameters(include_submodules=include_submodules, as_array=as_array))
    _check(module_c.parameters(include_submodules=include_submodules, as_array=as_array))
    _check(module_d.parameters(include_submodules=include_submodules, as_array=as_array))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_state_dict(capsys, device, modules: list[nn.Module]):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")
    module_a, module_b, module_c, module_d = [module.to(device) for module in modules]
    # state dict
    assert set(module_a.state_dict().keys()) == set()
    assert set(module_b.state_dict().keys()) == set(["param_b0"])
    assert set(module_c.state_dict().keys()) == set(["module_b.param_b0", "param_c0", "param_c1"])
    assert set(module_d.state_dict().keys()) == set(
        ["module_b.param_b0", "module_c.module_b.param_b0", "module_c.param_c0", "module_c.param_c1", "param_d0"]
    )
