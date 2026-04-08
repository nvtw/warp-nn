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

from typing import Iterator, Literal

import datasets

import numpy as np
import warp as wp

from warp_nn import utils


class MnistDataset:
    def __init__(self, mode: Literal["train", "test"], batch_size: int, device: str | None = None):
        """MNIST dataset.

        :param mode: The mode of the dataset to load.
        :param batch_size: The batch size.
        :param device: The device to use for dataset batches' data arrays.
        """
        self._num_samples = 0
        self._batch_size = batch_size
        self._device = utils.parse_device(device)
        # fetch the dataset
        print(f"[MNIST dataset][{mode}] Fetching data...")
        self._dataset = datasets.load_dataset("ylecun/mnist").with_format("numpy")[mode]
        # iterate over the dataset and convert batches to warp arrays to cache them
        print(f"[MNIST dataset][{mode}] Batching data (as Warp arrays)...")
        self._batches = []
        for batch in self._dataset.iter(batch_size=self._batch_size, drop_last_batch=True):
            images = (batch["image"] / 255.0).astype(np.float32)
            labels = batch["label"].astype(np.int32)
            self._num_samples += images.shape[0]
            self._batches.append((wp.array(images, device=self._device), wp.array(labels, device=self._device)))
        print(f"[MNIST dataset][{mode}] Data ready!")

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[tuple[wp.array, wp.array]]:
        return iter(self._batches)

    def __next__(self) -> tuple[wp.array, wp.array]:
        return next(self._batches)

    @property
    def num_samples(self) -> int:
        return self._num_samples


if __name__ == "__main__":
    dataset = MnistDataset(mode="train", batch_size=128, device="cuda")
    print(f"Dataset size: {len(dataset)}")
    for x, y in dataset:
        print(f"[images] range: [{x.numpy().min()}, {x.numpy().max()}], dtype: {x.dtype}, shape: {x.shape}")
        print(f"[labels] range: [{y.numpy().min()}, {y.numpy().max()}], dtype: {y.dtype}, shape: {y.shape}")
        break
