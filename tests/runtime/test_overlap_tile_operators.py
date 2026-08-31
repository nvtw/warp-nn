# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.operators import OverlapTileBlendPlan


def test_overlap_tile_blend_linear_seam():
    canvas = wp.zeros((1, 1, 2, 6), dtype=wp.float32, device="cpu")
    left = wp.array(np.ones((1, 1, 2, 4), dtype=np.float32), device="cpu")
    right = wp.array(np.full((1, 1, 2, 4), 3.0, dtype=np.float32), device="cpu")

    OverlapTileBlendPlan(left, canvas, 0, 0, 0, 2, 2, 6).execute()
    OverlapTileBlendPlan(right, canvas, 0, 2, 0, 2, 2, 6).execute()

    expected = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(canvas.numpy()[0, 0], np.tile(expected, (2, 1)))


def test_overlap_tiles_cover_and_crop_target():
    canvas = wp.array(np.full((1, 1, 5, 5), np.nan, dtype=np.float32), device="cpu")
    for origin_y in (0, 3):
        height = min(4, 5 - origin_y)
        for origin_x in (0, 3):
            width = min(4, 5 - origin_x)
            tile = wp.array(
                np.ones((1, 1, height, width), dtype=np.float32), device="cpu"
            )
            OverlapTileBlendPlan(tile, canvas, origin_y, origin_x, 1, 1, 5, 5).execute()

    np.testing.assert_array_equal(canvas.numpy(), np.ones((1, 1, 5, 5)))


def test_overlap_tile_rejects_out_of_bounds_target():
    tile = wp.zeros((1, 1, 2, 2), dtype=wp.float32, device="cpu")
    canvas = wp.zeros((1, 1, 3, 3), dtype=wp.float32, device="cpu")
    with pytest.raises(ValueError, match="outside the canvas"):
        OverlapTileBlendPlan(tile, canvas, 3, 0, 1, 1, 3, 3)
