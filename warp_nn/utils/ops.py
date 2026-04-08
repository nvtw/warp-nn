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

import math

from warp_nn.utils.config import KernelConfig


def resolve_dim(
    *, config: KernelConfig, shape: tuple[int, ...], tiled: bool, dimensions: int | None = None
) -> tuple[int, ...]:
    if tiled:
        ndim = len(shape)
        if ndim == 1:
            return (math.ceil(shape[0] / config.tile_1d[0]),)
        elif ndim == 2:
            grid = (
                math.ceil(shape[0] / config.tile_2d[0]),
                math.ceil(shape[1] / config.tile_2d[1]),
            )
            return grid if dimensions is None else grid[:dimensions]
        elif ndim == 3:
            grid = (
                math.ceil(shape[0] / config.tile_3d[0]),
                math.ceil(shape[1] / config.tile_3d[1]),
                math.ceil(shape[2] / config.tile_3d[2]),
            )
            return grid if dimensions is None else grid[:dimensions]
        elif ndim == 4:
            grid = (
                math.ceil(shape[0] / config.tile_4d[0]),
                math.ceil(shape[1] / config.tile_4d[1]),
                math.ceil(shape[2] / config.tile_4d[2]),
                math.ceil(shape[3] / config.tile_4d[3]),
            )
            return grid[:3] if dimensions is None else grid[:dimensions]  # tiled launch grid must be less than 4D
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")
    return shape
