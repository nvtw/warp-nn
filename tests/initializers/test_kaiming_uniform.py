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

import numpy as np
import warp as wp

from warp_nn import initializers

from .. import utilities


# module-specific parameters
@pytest.mark.parametrize("mode", ["fan_in", "fan_out", "scale"])
# test-specific parameters
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initialization(capsys, device, dtype, requires_grad, mode):
    if not utilities.is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    def _check(array):
        assert array.dtype == dtype
        assert array.device == device
        assert array.requires_grad == requires_grad

    array = wp.zeros(shape=(10, 1), dtype=dtype, device=device, requires_grad=requires_grad)
    # in-place: False
    output = initializers.kaiming_uniform(array, inplace=False, mode=mode)
    assert output is not array
    assert np.allclose(array.numpy(), 0.0)
    _check(output)
    # in-place: True
    array = wp.zeros(shape=(10, 1), dtype=dtype, device=device, requires_grad=requires_grad)
    output = initializers.kaiming_uniform(array, inplace=True, mode=mode)
    assert output is array
    _check(output)
