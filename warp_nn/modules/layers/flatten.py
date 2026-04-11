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

import warp as wp

from warp_nn.modules.module import Module


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        r"""Flatten a contiguous range of dimensions into a single dimension.

        Given an input of shape :math:`(*, D_{\text{start}}, \ldots, D_{\text{end}}, *)`,
        where :math:`*` means any number or absence of dimensions, the output has shape
        :math:`\left(*, \prod_{i=\text{start}}^{\text{end}} D_i, *\right)`.

        :param start_dim: Dimension to start flattening from.
        :param end_dim: Dimension to end flattening at.
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, input: wp.array) -> wp.array:
        """Forward pass of the module.

        :param input: The input array, with shape ``(*, start_dim, ..., end_dim, *)``.

        :return: The output array, with shape ``(*, prod(start_dim, ..., end_dim), *)``.

        :raises IndexError: If the start dimension comes after the end dimension.
        """
        shape = input.shape
        start = self.start_dim if self.start_dim >= 0 else len(shape) + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        if start > end:
            raise IndexError(f"The start dimension ({start}) cannot come after the end dimension ({end})")
        output_shape = shape[:start] + (math.prod(shape[start : end + 1]),) + shape[end + 1 :]
        return input.reshape(output_shape)
