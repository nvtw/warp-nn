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


def zeros(array: wp.array, *, inplace: bool = True) -> wp.array:
    """Initialize the array with zeros.

    :param array: The array to initialize.
    :param inplace: Whether to fill the array in place.

    :return: The initialized array.
    """
    if inplace:
        array.fill_(0.0)
        return array
    return wp.full_like(array, 0.0)
