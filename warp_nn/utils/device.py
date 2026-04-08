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

import warp as wp

from warp_nn.utils.logging import logger


def parse_device(device: str | wp.Device | None) -> wp.Device:
    """Parse the input device and return a :py:class:`~warp.Device` instance.

    If the specified device is ``None`` or it cannot be resolved, the default available device
    will be returned instead. For invalid device specifications, a warning message will be logged.

    :param device: Device specification.

    :return: Device.
    """
    if isinstance(device, wp.Device):
        return device
    elif isinstance(device, str):
        try:
            return wp.get_device(device)
        except Exception as e:
            default_device = wp.get_device()
            logger.warning(f"Invalid device specification ({device}): {e}. Using default device: {default_device}")
            return default_device
    return wp.get_device()
