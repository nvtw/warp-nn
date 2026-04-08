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

from typing import Any

from warp_nn.modules.module import Module


class Sequential(Module):
    def __init__(self, *args):
        """Apply callable modules (e.g. layers, activation functions, etc.) connected in a cascading sequence.

        :param args: Callable modules to apply in sequence (in the order they are passed to the class constructor).
        """
        super().__init__()
        # register modules
        modules = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
        for i, module in enumerate(modules):
            if not isinstance(module, Module):
                raise ValueError(f"Item at index {i} (of type {type(module).__name__}) is not a Module subclass")
            self.register_module(str(i), module)

    def __len__(self) -> int:
        """Get the number of modules in the sequential container.

        :return: The number of registered modules.
        """
        return len(self.modules())

    def __call__(self, input: Any) -> Any:
        """Forward pass of the sequential container.

        :param input: The input to the first module in the container.
        :return: The output of the last module in the container.
        """
        for module in self.modules():
            input = module(input)
        return input
