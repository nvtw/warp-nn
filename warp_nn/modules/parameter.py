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

import warp as wp

from warp_nn.utils import parse_device


class Parameter:
    def __init__(self, data: wp.array, *, requires_grad: bool = True) -> None:
        """Class representing a learnable parameter.

        .. note::

            All learnable parameters in a :py:class:`~warp_nn.modules.module.Module` are of the ``Parameter`` type.

        :param data: The underlying data container (array) of the parameter.
        :param requires_grad: Whether the parameter requires gradients.
        """
        self._data = data
        self._data.requires_grad = requires_grad

    @property
    def data(self) -> wp.array:
        """The underlying data container (array) of the parameter."""
        return self._data

    @property
    def device(self) -> wp.Device:
        """Device on which the parameter is allocated."""
        return self._data.device

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the parameter."""
        return self._data.shape

    @property
    def dtype(self) -> type:
        """Data type of the parameter."""
        return self._data.dtype

    @property
    def requires_grad(self) -> bool:
        """Whether the parameter requires gradients."""
        return self._data.requires_grad

    def to(self, device: wp.Device) -> Parameter:
        """Move the parameter to the specified device.

        :param device: The device to move the parameter to.
        :return: The parameter itself.
        """
        device = parse_device(device)
        self._data = self._data.to(device=device, requires_grad=self._data.requires_grad)
        return self
