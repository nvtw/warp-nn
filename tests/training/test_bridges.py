# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.bridges import (
    accumulate_fp32_gradient,
    add_fp32_gradients,
    cast_from_float32,
    cast_to_float32,
    merge_heads,
    split_heads,
)


@pytest.mark.parametrize("dtype,atol", [(wp.float16, 5.0e-4), (wp.bfloat16, 4.0e-3)])
def test_cast_round_trip_uses_caller_buffers(dtype, atol):
    values = np.array([[[[0.1, -1.25, 3.5], [2.0, -0.75, 0.03125]]]], dtype=np.float32)
    storage = wp.array(values, dtype=dtype, device="cpu")
    fp32 = wp.empty(values.shape, dtype=wp.float32, device="cpu")
    round_trip = wp.empty(values.shape, dtype=dtype, device="cpu")
    pointers = (fp32.ptr, round_trip.ptr)

    cast_to_float32(storage, fp32)
    cast_from_float32(fp32, round_trip)

    quantized = storage.numpy().astype(np.float32)
    np.testing.assert_array_equal(fp32.numpy(), quantized)
    np.testing.assert_allclose(round_trip.numpy(), quantized, atol=atol, rtol=atol)
    assert pointers == (fp32.ptr, round_trip.ptr)


@pytest.mark.parametrize("dtype", [wp.float16, wp.bfloat16])
def test_split_merge_heads_are_exact_inverses(dtype):
    batch, sequence, head_count, head_size = 2, 3, 2, 4
    packed_values = np.arange(
        batch * sequence * head_count * head_size, dtype=np.float32
    ).reshape(batch * sequence, head_count * head_size)
    packed = wp.array(packed_values, dtype=dtype, device="cpu")
    heads = wp.empty(
        (batch, head_count, sequence, head_size), dtype=dtype, device="cpu"
    )
    restored = wp.empty_like(packed)

    split_heads(packed, heads)
    merge_heads(heads, restored)

    expected = packed.numpy().reshape(batch, sequence, head_count, head_size)
    expected = expected.transpose(0, 2, 1, 3)
    np.testing.assert_array_equal(heads.numpy(), expected)
    np.testing.assert_array_equal(restored.numpy(), packed.numpy())


def test_fp32_gradient_add_and_accumulate_are_allocation_free():
    left_values = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    right_values = np.full((2, 2, 3), 0.25, dtype=np.float32)
    left = wp.array(left_values, device="cpu")
    right = wp.array(right_values, device="cpu")
    output = wp.empty(left.shape, dtype=wp.float32, device="cpu")
    output_ptr = output.ptr

    add_fp32_gradients(left, right, output)
    np.testing.assert_array_equal(output.numpy(), left_values + right_values)
    accumulate_fp32_gradient(right, output)
    np.testing.assert_array_equal(output.numpy(), left_values + 2.0 * right_values)
    assert output.ptr == output_ptr


def test_bridges_reject_incompatible_shapes_and_dtypes():
    fp32 = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    fp16 = wp.zeros((2, 3), dtype=wp.float16, device="cpu")
    wrong_shape = wp.zeros((3, 2), dtype=wp.float32, device="cpu")
    with pytest.raises(TypeError, match="FP16/BF16"):
        cast_to_float32(fp32, fp32)
    with pytest.raises(ValueError, match="shapes must match"):
        cast_to_float32(fp16, wrong_shape)

    heads = wp.empty((1, 2, 2, 3), dtype=wp.float16, device="cpu")
    with pytest.raises(ValueError, match="packed must have shape"):
        split_heads(fp16, heads)
    with pytest.raises(TypeError, match="FP32"):
        accumulate_fp32_gradient(fp16, fp16)
