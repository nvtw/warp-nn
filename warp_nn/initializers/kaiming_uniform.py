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

from typing import Literal

import numpy as np
import warp as wp

from ._common import compute_fans


def kaiming_uniform(
    array: wp.array,
    *,
    scale: float = 2.0,
    mode: Literal["fan_in", "fan_out", "scale"] = "fan_in",
    in_axis: int | list[int] = -1,
    out_axis: int | list[int] = -2,
    batch_axis: int | list[int] = [],
    inplace: bool = True,
) -> wp.array:
    r"""Initialize the array using the Kaiming (aka He) uniform initialization method.

    .. math::

        x \leftarrow \mathcal{U}(-b, b)
        \quad \text{where} \quad
        b = \sqrt{\frac{\text{scale}}{\text{denominator}}}

    given that

    .. math::

        \text{denominator} = \begin{cases}
            \text{fan\_in}, & \text{ if } \text{mode} = \, "fan\_in"\\
            \text{fan\_out}, & \text{ if } \text{mode} = \, "fan\_out"\\
            1.0, & \text{ if } \text{mode} = \, "scale".
        \end{cases}

    :param array: The array to initialize.
    :param scale: Scale used as numerator in the square root formula.
    :param mode: Initialization mode.
    :param in_axis: Axis used to compute the fan-in.
    :param out_axis: Axis used to compute the fan-out.
    :param batch_axis: Axis used to compute the batch size.
    :param inplace: Whether to fill the array in place.

    :return: The initialized array.
    """
    fan_in, fan_out = compute_fans(shape=array.shape, in_axis=in_axis, out_axis=out_axis, batch_axis=batch_axis)
    denominator = {"fan_in": fan_in, "fan_out": fan_out, "scale": 1.0}[mode]
    bound = np.sqrt(scale / denominator)
    value = np.random.uniform(-bound, bound, size=array.shape)
    if inplace:
        wp.copy(array, wp.array(value, dtype=array.dtype))
        return array
    return wp.array(value, dtype=array.dtype, device=array.device, requires_grad=array.requires_grad)
