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

from typing import Callable

import warp as wp


def overload_kernels(*, kernels: list[Callable], dtypes: list[type] = [wp.float16, wp.float32, wp.float64]):
    _kernels = {}
    for i, kernel in enumerate(kernels):
        for dtype in dtypes:
            ndim = i + 1
            _kernels[(ndim, dtype)] = wp.overload(
                kernel, [wp.array(ndim=ndim, dtype=dtype), wp.array(ndim=ndim, dtype=dtype)]
            )
    return _kernels
