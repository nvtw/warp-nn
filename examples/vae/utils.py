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

import cv2

import numpy as np


def _normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val) * 255
    image = image.astype(np.uint8)
    return image


def save_images(images, path, *, rows=None):
    batch_size = images.shape[0]
    width = images.shape[1]
    height = images.shape[2]
    # convert batched images (batch_size, width, height) to a tiled image
    rows = rows if rows is not None else int(np.sqrt(batch_size))
    cols = int(np.ceil(batch_size / rows))
    image = np.zeros((rows * height, cols * width), dtype=images.dtype)
    for i in range(rows):
        for j in range(cols):
            image[i * height : (i + 1) * height, j * width : (j + 1) * width] = images[i * cols + j]
    # save image
    cv2.imwrite(path, _normalize_image(image))


def save_image(image, path):
    cv2.imwrite(path, _normalize_image(image))
