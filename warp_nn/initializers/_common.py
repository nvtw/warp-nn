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


def compute_fans(
    *,
    shape: list[int],
    in_axis: int | list[int] = -1,
    out_axis: int | list[int] = -2,
    batch_axis: int | list[int] = [],
) -> tuple[float, float]:
    """Compute the fan-in and fan-out of an array.

    :param shape: The shape of the array.
    :param in_axis: Axis used to compute the fan-in.
    :param out_axis: Axis used to compute the fan-out.
    :param batch_axis: Axis used to compute the batch size.

    :return: Fan-in and fan-out.
    """
    in_size = math.prod([shape[i] for i in ([in_axis] if isinstance(in_axis, int) else in_axis)])
    out_size = math.prod([shape[i] for i in ([out_axis] if isinstance(out_axis, int) else out_axis)])
    batch_size = math.prod([shape[i] for i in ([batch_axis] if isinstance(batch_axis, int) else batch_axis)])
    receptive_field_size = math.prod(shape) / in_size / out_size / batch_size
    return in_size * receptive_field_size, out_size * receptive_field_size
