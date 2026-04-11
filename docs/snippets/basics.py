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

# fmt: off
# [basics-imports-start]
import warp as wp
from warp_nn import nn
from warp_nn import optimizers
# [basics-imports-end]
# fmt: on


epochs = 10
dataset = (
    (
        wp.zeros((10, 128), dtype=wp.float32, requires_grad=True, device="cuda"),
        wp.zeros((10, 10), dtype=wp.float32, device="cuda"),
    ),
)


# [basics-model-definition-start]
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        super().__post_init__()  # must be called last

    def __call__(self, x):
        return self.fc2(self.act(self.fc1(x)))
        # [basics-model-definition-end]


# [basics-model-initialization-start]
model = MLP()
model.to("cuda")
# [basics-model-initialization-end]


# [basics-loss-function-start]
@wp.kernel
def mse_loss(
    prediction: wp.array2d[float],
    target: wp.array2d[float],
    loss: wp.array1d[float],
):
    i, j = wp.tid()
    diff = prediction[i, j] - target[i, j]
    wp.atomic_add(loss, 0, diff * diff)
    # [basics-loss-function-end]


# [basics-loss-array-start]
loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True, device="cuda")
# [basics-loss-array-end]
# [basics-optimizer-start]
optimizer = optimizers.Adam(model.parameters(), lr=1e-3, device="cuda")
# [basics-optimizer-end]

# [basics-learning-loop-start]
for epoch in range(epochs):
    for input, target in dataset:
        # [basics-learning-loop-end]
        # [basics-record-start]
        loss.zero_()  # reset loss array before each loss computation
        with wp.Tape() as tape:
            prediction = model(input)  # model's forward pass
            wp.launch(  # loss computation
                mse_loss,
                dim=prediction.shape,
                inputs=[prediction, target],
                outputs=[loss],
                device="cuda",
            )
        # [basics-record-end]
        # [basics-optimization-start]
        tape.backward(loss)  # compute gradients
        optimizer.step()
        tape.zero()  # reset tape and zero gradients
# [basics-optimization-end]
