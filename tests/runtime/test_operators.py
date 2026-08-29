# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime._cublas import try_create_cublas
from warp_nn.runtime.gguf import BlockQuantizedTensor
from warp_nn.runtime.kernels import (
    _append_head_cache_kernel,
    _append_circular_head_cache_kernel,
    _causal_conv_rows_kernel,
    _decode_attention_head_group,
    _decode_attention_partitions,
    _get_gated_rms_norm_kernel,
    _get_gqa_attention_kernel,
    _get_greedy_argmax_kernels,
    _get_grouped_decode_linear_kernel,
    _get_top_k_kernels,
    _get_matmul_int8_q8_kernel,
    _get_linear_attention_kernel,
    _allocate_partitioned_gqa,
    _launch_partitioned_gqa,
    _prepare_gated_delta_kernel,
    _relu2_kernel,
    _reorder_heads_kernel,
    _sigmoid_gate_kernel,
    _logit_softcap_kernel,
    _scale_kernel,
    _unpack_gated_heads_kernel,
    _update_conv_rows_state_kernel,
)
from warp_nn.runtime.operators import (
    Operation,
    execute_operations,
    plan_linear,
    plan_rms_norm,
)


@pytest.mark.parametrize(
    ("device", "rows"), [("cpu", 3), ("cuda:0", 3), ("cuda:0", 32)]
)
def test_linear_operation(device, rows):
    if not is_device_available(device):
        pytest.skip(f"Device {device} is not available")
    dtype = wp.bfloat16 if device.startswith("cuda") else wp.float32
    rng = np.random.default_rng(13)
    x_np = rng.normal(size=(rows, 37)).astype(np.float32)
    weight_np = rng.normal(size=(41, 37)).astype(np.float32)
    tensors = {
        "x": wp.array(x_np, dtype=dtype, device=device),
        "weight": wp.array(weight_np, dtype=dtype, device=device),
    }
    shapes = {name: tuple(value.shape) for name, value in tensors.items()}
    operation = Operation("Linear", ["x", "weight"], ["output"])
    plan_linear(operation, tensors, shapes, wp.get_device(device))

    execute_operations([operation], tensors, shapes, wp.get_device(device))

    np.testing.assert_allclose(
        tensors["output"].numpy(), x_np @ weight_np.T, atol=0.2, rtol=0.02
    )


@pytest.mark.parametrize("dtype", [wp.float16, wp.bfloat16])
def test_linear_operation_uses_m64_when_grid_stays_large(dtype):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    device = wp.get_device("cuda:0")
    rows, inner, columns = 64, 32, 64 * device.sm_count
    rng = np.random.default_rng(31)
    x_np = rng.normal(0.0, 0.2, size=(rows, inner)).astype(np.float32)
    weight_np = rng.normal(0.0, 0.2, size=(columns, inner)).astype(np.float32)
    tensors = {
        "x": wp.array(x_np, dtype=dtype, device=device),
        "weight": wp.array(weight_np, dtype=dtype, device=device),
    }
    shapes = {name: tuple(value.shape) for name, value in tensors.items()}
    operation = Operation("Linear", ["x", "weight"], ["output"])
    plan_linear(operation, tensors, shapes, device)

    assert operation.attrs["_tile_shape"] == (64, 32)
    execute_operations([operation], tensors, shapes, device)

    np.testing.assert_allclose(
        tensors["output"].numpy(),
        x_np @ weight_np.T,
        atol=0.2,
        rtol=0.02,
    )


@pytest.mark.parametrize("dtype", [wp.float16, wp.bfloat16])
def test_grouped_decode_linear_kernel(dtype):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(29)
    columns, inner = 40, 64
    x_np = rng.normal(size=(1, inner)).astype(np.float32)
    weight_np = rng.normal(size=(columns, inner)).astype(np.float32)
    x = wp.array(x_np, dtype=dtype, device="cuda:0")
    weight = wp.array(weight_np, dtype=dtype, device="cuda:0")
    output = wp.empty((1, columns), dtype=dtype, device="cuda:0")
    wp.launch(
        _get_grouped_decode_linear_kernel(dtype),
        dim=(columns // 8) * 32,
        inputs=[x, weight, output, inner],
        block_dim=128,
        device="cuda:0",
    )
    np.testing.assert_allclose(output.numpy(), x_np @ weight_np.T, atol=0.2, rtol=0.02)


def test_linear_operation_cublas():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    cublas = try_create_cublas()
    if cublas is None:
        pytest.skip("cuBLAS is not available")
    rng = np.random.default_rng(17)
    x_np = rng.normal(size=(5, 32)).astype(np.float32)
    weight_np = rng.normal(size=(48, 32)).astype(np.float32)
    tensors = {
        "x": wp.array(x_np, dtype=wp.bfloat16, device="cuda:0"),
        "weight": wp.array(weight_np, dtype=wp.bfloat16, device="cuda:0"),
    }
    shapes = {name: tuple(value.shape) for name, value in tensors.items()}
    operation = Operation("Linear", ["x", "weight"], ["output"])
    plan_linear(operation, tensors, shapes, wp.get_device("cuda:0"), cublas=cublas)

    execute_operations([operation], tensors, shapes, wp.get_device("cuda:0"))

    np.testing.assert_allclose(
        tensors["output"].numpy(), x_np @ weight_np.T, atol=0.2, rtol=0.02
    )
    cublas.close()
    cublas.close()


@pytest.mark.parametrize(
    "query_heads,kv_heads,head_size,group",
    [(32, 2, 128, 16), (16, 2, 128, 8), (24, 4, 128, 4), (24, 4, 256, 4)],
)
def test_decode_attention_head_group_follows_gqa_geometry(
    query_heads, kv_heads, head_size, group
):
    assert _decode_attention_head_group(query_heads, kv_heads, head_size) == group


@pytest.mark.parametrize("query_heads,kv_heads", [(4, 0), (2, 4), (5, 2)])
def test_decode_attention_head_group_rejects_invalid_geometry(query_heads, kv_heads):
    with pytest.raises(ValueError, match="positive multiple"):
        _decode_attention_head_group(query_heads, kv_heads, 128)


@pytest.mark.parametrize(
    "head_size,partitions", [(4, 64), (128, 128), (256, 256), (512, 256)]
)
def test_decode_attention_partitions_follow_head_geometry(head_size, partitions):
    assert _decode_attention_partitions(head_size) == partitions


@pytest.mark.parametrize("rows", [1, 3, 32])
def test_q8_0_linear_operation(rows):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(23)
    columns, inner = 40, 64
    blocks = inner // 32
    x_np = rng.normal(0.0, 0.5, (rows, inner)).astype(np.float32)
    weight_values = rng.integers(-127, 128, (columns, blocks, 32), dtype=np.int8)
    weight_scales = rng.uniform(0.001, 0.02, (columns, blocks)).astype(np.float16)
    values = wp.array(weight_values, dtype=wp.int8, device="cuda:0")
    words = wp.array(
        ptr=values.ptr,
        capacity=values.capacity,
        shape=(columns, blocks, 8),
        dtype=wp.uint32,
        device="cuda:0",
        copy=False,
    )
    scales = wp.array(weight_scales, dtype=wp.float16, device="cuda:0")
    weight = BlockQuantizedTensor(values, words, scales, (columns, inner), "Q8_0")
    tensors = {
        "x": wp.array(x_np, dtype=wp.bfloat16, device="cuda:0"),
        "weight": weight,
    }
    shapes = {name: tuple(value.shape) for name, value in tensors.items()}
    operation = Operation("Linear", ["x", "weight"], ["output"])
    device = wp.get_device("cuda:0")
    plan_linear(operation, tensors, shapes, device)

    execute_operations([operation], tensors, shapes, device)

    x_bf16 = tensors["x"].numpy().astype(np.float32)
    reshaped = x_bf16.reshape(rows, blocks, 32)
    activation_scales = np.max(np.abs(reshaped), axis=2, keepdims=True) / 127.0
    activation_scales[activation_scales == 0.0] = 1.0
    x_bf16 = (
        np.clip(np.rint(reshaped / activation_scales), -127, 127) * activation_scales
    ).reshape(rows, inner)
    dequantized = (
        weight_values.astype(np.float32) * weight_scales.astype(np.float32)[:, :, None]
    ).reshape(columns, inner)
    np.testing.assert_allclose(
        tensors["output"].numpy(), x_bf16 @ dequantized.T, atol=0.3, rtol=0.03
    )


def test_q8_grouped_outputs_match_single_output_kernel():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(29)
    rows, columns, blocks = 3, 41, 2
    activation_values = wp.array(
        rng.integers(-127, 128, (rows, blocks, 32), dtype=np.int8),
        device="cuda:0",
    )
    activation_words = wp.array(
        ptr=activation_values.ptr,
        capacity=activation_values.capacity,
        shape=(rows, blocks, 8),
        dtype=wp.uint32,
        device="cuda:0",
        copy=False,
    )
    activation_scales = wp.array(
        rng.uniform(0.001, 0.02, (rows, blocks)).astype(np.float32),
        device="cuda:0",
    )
    weight_values = wp.array(
        rng.integers(-127, 128, (columns, blocks, 32), dtype=np.int8),
        device="cuda:0",
    )
    weight_words = wp.array(
        ptr=weight_values.ptr,
        capacity=weight_values.capacity,
        shape=(columns, blocks, 8),
        dtype=wp.uint32,
        device="cuda:0",
        copy=False,
    )
    weight_scales = wp.array(
        rng.uniform(0.001, 0.02, (columns, blocks)).astype(np.float16),
        device="cuda:0",
    )
    single = wp.empty((rows, columns), dtype=wp.bfloat16, device="cuda:0")
    grouped = wp.empty_like(single)
    wp.launch(
        _get_matmul_int8_q8_kernel(8, wp.bfloat16, True, 1),
        dim=rows * columns * 8,
        inputs=[
            activation_words,
            activation_scales,
            weight_words,
            weight_scales,
            single,
        ],
        block_dim=128,
        device="cuda:0",
    )
    wp.launch(
        _get_matmul_int8_q8_kernel(8, wp.bfloat16, True, 2),
        dim=rows * ((columns + 1) // 2) * 8,
        inputs=[
            activation_words,
            activation_scales,
            weight_words,
            weight_scales,
            grouped,
        ],
        block_dim=128,
        device="cuda:0",
    )
    np.testing.assert_array_equal(grouped.numpy(), single.numpy())


def test_gated_rms_norm_bfloat16():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(19)
    x = wp.array(
        rng.normal(size=(3, 8)).astype(np.float32), dtype=wp.bfloat16, device="cuda:0"
    )
    gate = wp.array(
        rng.normal(size=(3, 8)).astype(np.float32), dtype=wp.bfloat16, device="cuda:0"
    )
    scale = wp.array(
        rng.normal(size=(1, 8)).astype(np.float32), dtype=wp.bfloat16, device="cuda:0"
    )
    output = wp.empty_like(x)
    tile_width, kernel = _get_gated_rms_norm_kernel(8, wp.bfloat16)

    wp.launch_tiled(
        kernel,
        dim=3,
        inputs=[x, gate, scale, output, 1.0e-6],
        block_dim=tile_width,
        device="cuda:0",
    )

    x_np = x.numpy().astype(np.float32)
    gate_np = gate.numpy().astype(np.float32)
    expected = x_np / np.sqrt(np.mean(x_np * x_np, axis=1, keepdims=True) + 1.0e-6)
    expected *= scale.numpy().astype(np.float32)
    expected *= gate_np / (1.0 + np.exp(-gate_np))
    np.testing.assert_allclose(output.numpy(), expected, atol=0.04, rtol=0.02)


def test_group_query_attention_bfloat16():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(21)
    sequence_length, query_heads, kv_heads, head_size = 3, 2, 1, 8
    query_np = rng.normal(size=(query_heads, sequence_length, head_size)).astype(
        np.float32
    )
    key_np = rng.normal(size=(kv_heads, sequence_length, head_size)).astype(np.float32)
    value_np = rng.normal(size=(kv_heads, sequence_length, head_size)).astype(
        np.float32
    )
    query = wp.array(
        query_np.reshape(-1, head_size), dtype=wp.bfloat16, device="cuda:0"
    )
    key = wp.array(key_np.reshape(-1, head_size), dtype=wp.bfloat16, device="cuda:0")
    value = wp.array(
        value_np.reshape(-1, head_size), dtype=wp.bfloat16, device="cuda:0"
    )
    lengths = wp.array(np.array([sequence_length - 1], dtype=np.int32), device="cuda:0")
    output = wp.empty(
        (sequence_length, query_heads * head_size), dtype=wp.bfloat16, device="cuda:0"
    )
    block_dim, kernel = _get_gqa_attention_kernel(head_size, wp.bfloat16)

    wp.launch_tiled(
        kernel,
        dim=query_heads * sequence_length,
        inputs=[
            query,
            key,
            value,
            lengths,
            output,
            query_heads,
            kv_heads,
            sequence_length,
            sequence_length,
            head_size**-0.5,
            0,
        ],
        block_dim=block_dim,
        device="cuda:0",
    )

    expected = np.empty((sequence_length, query_heads * head_size), dtype=np.float32)
    for token in range(sequence_length):
        for head in range(query_heads):
            scores = (
                query_np[head, token] @ key_np[0, : token + 1].T / np.sqrt(head_size)
            )
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            expected[token, head * head_size : (head + 1) * head_size] = (
                weights @ value_np[0, : token + 1]
            )
    np.testing.assert_allclose(output.numpy(), expected, atol=0.04, rtol=0.02)


def test_circular_window_attention_and_logit_softcap():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(31)
    capacity, window, head_size = 4, 3, 8
    keys = rng.normal(size=(6, head_size)).astype(np.float32)
    values = rng.normal(size=(6, head_size)).astype(np.float32)
    key_ring = np.empty((capacity, head_size), dtype=np.float32)
    value_ring = np.empty_like(key_ring)
    for position in range(2, 6):
        key_ring[position % capacity] = keys[position]
        value_ring[position % capacity] = values[position]
    query_np = rng.normal(size=(1, head_size)).astype(np.float32)
    query = wp.array(query_np, dtype=wp.bfloat16, device="cuda:0")
    key = wp.array(key_ring, dtype=wp.bfloat16, device="cuda:0")
    value = wp.array(value_ring, dtype=wp.bfloat16, device="cuda:0")
    lengths = wp.array(np.array([5], dtype=np.int32), device="cuda:0")
    output = wp.empty((1, head_size), dtype=wp.bfloat16, device="cuda:0")
    block_dim, kernel = _get_gqa_attention_kernel(head_size, wp.bfloat16)

    wp.launch_tiled(
        kernel,
        dim=1,
        inputs=[
            query,
            key,
            value,
            lengths,
            output,
            1,
            1,
            1,
            capacity,
            head_size**-0.5,
            window,
        ],
        block_dim=block_dim,
        device="cuda:0",
    )
    scores = query_np[0] @ keys[3:6].T / np.sqrt(head_size)
    weights = np.exp(scores - scores.max())
    expected = weights @ values[3:6] / weights.sum()
    np.testing.assert_allclose(output.numpy()[0], expected, atol=0.04, rtol=0.02)

    source = wp.array(
        np.arange(16, dtype=np.float32).reshape(2, 8),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    positions = wp.array(np.array([[4, 5]], dtype=np.int64), device="cuda:0")
    cache = wp.zeros((capacity, head_size), dtype=wp.bfloat16, device="cuda:0")
    wp.launch(
        _append_circular_head_cache_kernel,
        dim=(1, 2, head_size),
        inputs=[source, positions, cache, 1, head_size],
        device="cuda:0",
    )
    np.testing.assert_array_equal(cache.numpy()[:2], source.numpy())

    logits = wp.array(
        np.array([[[-40.0, 0.0, 40.0]]], dtype=np.float32),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    capped = wp.empty_like(logits)
    wp.launch(
        _logit_softcap_kernel,
        dim=logits.shape,
        inputs=[logits, capped, 0.2, 20.0],
        device="cuda:0",
    )
    expected_logits = 20.0 * np.tanh(np.array([-40.0, 0.0, 40.0]) * 0.2 / 20.0)
    np.testing.assert_allclose(capped.numpy()[0, 0], expected_logits, atol=0.03)

    wp.launch(
        _scale_kernel,
        dim=(1, 3),
        inputs=[capped.reshape((1, 3)), capped.reshape((1, 3)), 2.0],
    )
    np.testing.assert_allclose(capped.numpy()[0, 0], expected_logits * 2.0, atol=0.06)


@pytest.mark.parametrize(
    (
        "query_heads",
        "kv_heads",
        "head_size",
        "length",
        "capacity",
        "window",
        "rows",
        "rows_per_group",
    ),
    [
        (4, 2, 32, 1, 1, 0, 1, 1),
        (4, 2, 32, 13, 16, 0, 1, 1),
        (6, 1, 32, 19, 20, 0, 1, 1),
        (4, 2, 32, 19, 8, 5, 1, 1),
        (8, 1, 128, 13, 16, 0, 1, 1),
        (12, 1, 128, 13, 16, 0, 1, 1),
        (16, 1, 128, 13, 16, 0, 1, 1),
        (6, 1, 32, 19, 20, 0, 4, 1),
        (4, 2, 32, 19, 8, 5, 4, 1),
        (6, 1, 32, 19, 20, 0, 3, 2),
        (4, 2, 32, 19, 8, 5, 3, 2),
        (6, 1, 32, 19, 20, 0, 5, 4),
        (4, 2, 32, 19, 8, 5, 5, 4),
    ],
)
def test_partitioned_decode_attention_matches_serial(
    query_heads, kv_heads, head_size, length, capacity, window, rows, rows_per_group
):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(41)
    query = wp.array(
        rng.normal(size=(query_heads * rows, head_size)).astype(np.float32),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    key_np = rng.normal(size=(kv_heads, length, head_size)).astype(np.float32)
    value_np = rng.normal(size=(kv_heads, length, head_size)).astype(np.float32)
    key_cache = np.zeros((kv_heads, capacity, head_size), dtype=np.float32)
    value_cache = np.zeros_like(key_cache)
    for token in range(length):
        key_cache[:, token % capacity] = key_np[:, token]
        value_cache[:, token % capacity] = value_np[:, token]
    if window == 0 and length < capacity:
        key_cache[:, length:] = np.nan
        value_cache[:, length:] = np.nan
    key = wp.array(key_cache.reshape(-1, head_size), dtype=wp.bfloat16, device="cuda:0")
    value = wp.array(
        value_cache.reshape(-1, head_size), dtype=wp.bfloat16, device="cuda:0"
    )
    lengths = wp.array(np.array([length - 1], dtype=np.int32), device="cuda:0")
    expected = wp.empty(
        (rows, query_heads * head_size), dtype=wp.bfloat16, device="cuda:0"
    )
    actual = wp.empty_like(expected)

    block_dim, serial = _get_gqa_attention_kernel(head_size, wp.bfloat16)
    wp.launch_tiled(
        serial,
        dim=query_heads * rows,
        inputs=[
            query,
            key,
            value,
            lengths,
            expected,
            query_heads,
            kv_heads,
            rows,
            capacity,
            head_size**-0.5,
            window,
        ],
        block_dim=block_dim,
        device="cuda:0",
    )

    workspace = _allocate_partitioned_gqa(
        query_heads,
        head_size,
        wp.bfloat16,
        "cuda:0",
        rows=rows,
        rows_per_group=rows_per_group,
        kv_heads=kv_heads,
    )
    _launch_partitioned_gqa(
        workspace,
        query,
        key,
        value,
        lengths,
        actual,
        query_heads,
        kv_heads,
        capacity,
        head_size**-0.5,
        window,
        "cuda:0",
    )
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=0.01, rtol=0.01)


def test_partitioned_attention_130k_indexing():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    length, query_heads, kv_heads, head_size = 130_001, 4, 1, 32
    query = wp.zeros((query_heads, head_size), dtype=wp.bfloat16, device="cuda:0")
    key = wp.zeros((length, head_size), dtype=wp.bfloat16, device="cuda:0")
    feature = np.linspace(-1.0, 1.0, head_size, dtype=np.float32)
    alternating = (np.arange(length, dtype=np.int32) & 1).astype(np.float32)
    values_np = alternating[:, None] * feature[None, :]
    value = wp.array(values_np, dtype=wp.bfloat16, device="cuda:0")
    sequence_end = wp.array([length - 1], dtype=wp.int32, device="cuda:0")
    output = wp.empty((1, query_heads * head_size), dtype=wp.bfloat16, device="cuda:0")
    workspace = _allocate_partitioned_gqa(query_heads, head_size, wp.bfloat16, "cuda:0")
    _launch_partitioned_gqa(
        workspace,
        query,
        key,
        value,
        sequence_end,
        output,
        query_heads,
        kv_heads,
        length,
        head_size**-0.5,
        0,
        "cuda:0",
    )

    expected_head = values_np.mean(axis=0)
    expected = np.tile(expected_head, query_heads)
    np.testing.assert_allclose(output.numpy()[0], expected, atol=0.005, rtol=0.005)


def test_head_layout_cache_and_bfloat16_argmax():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rows, heads, head_size, capacity = 2, 2, 2, 5
    packed_np = np.arange(rows * heads * head_size * 2, dtype=np.float32).reshape(
        rows, -1
    )
    packed = wp.array(packed_np, dtype=wp.bfloat16, device="cuda:0")
    values = wp.empty((heads * rows, head_size), dtype=wp.bfloat16, device="cuda:0")
    gate = wp.empty((rows, heads * head_size), dtype=wp.bfloat16, device="cuda:0")
    reordered = wp.empty_like(values)
    gated = wp.empty_like(gate)
    positions = wp.array(np.array([[1, 3]], dtype=np.int64), device="cuda:0")
    cache = wp.zeros((heads * capacity, head_size), dtype=wp.bfloat16, device="cuda:0")

    wp.launch(
        _unpack_gated_heads_kernel,
        dim=(rows, heads, head_size),
        inputs=[packed, values, gate, head_size, False],
        device="cuda:0",
    )
    row_major = wp.array(
        packed_np[:, : heads * head_size], dtype=wp.bfloat16, device="cuda:0"
    )
    wp.launch(
        _reorder_heads_kernel,
        dim=(rows, heads, head_size),
        inputs=[row_major, reordered, head_size],
        device="cuda:0",
    )
    wp.launch(
        _append_head_cache_kernel,
        dim=(heads, rows, head_size),
        inputs=[reordered, positions, cache, heads, head_size],
        device="cuda:0",
    )
    wp.launch(
        _sigmoid_gate_kernel,
        dim=gate.shape,
        inputs=[gate, gate, gated],
        device="cuda:0",
    )

    expected_values = np.empty((heads, rows, head_size), dtype=np.float32)
    expected_gate = np.empty((rows, heads, head_size), dtype=np.float32)
    for row in range(rows):
        per_head = packed_np[row].reshape(heads, 2, head_size)
        expected_values[:, row] = per_head[:, 0]
        expected_gate[row] = per_head[:, 1]
    np.testing.assert_array_equal(
        values.numpy(), expected_values.reshape(-1, head_size)
    )
    np.testing.assert_array_equal(gate.numpy(), expected_gate.reshape(rows, -1))
    expected_reordered = (
        packed_np[:, : heads * head_size]
        .reshape(rows, heads, head_size)
        .transpose(1, 0, 2)
    )
    np.testing.assert_array_equal(
        reordered.numpy(), expected_reordered.reshape(-1, head_size)
    )
    cache_np = cache.numpy().reshape(heads, capacity, head_size)
    np.testing.assert_array_equal(cache_np[:, [1, 3]], expected_reordered)
    expected_gated = expected_gate.reshape(rows, -1)
    expected_gated = expected_gated / (1.0 + np.exp(-expected_gated))
    np.testing.assert_allclose(gated.numpy(), expected_gated, atol=0.04)

    logits_np = np.arange(34, dtype=np.float32).reshape(1, 2, 17)
    logits_np[0, -1, 7] = 100.0
    logits = wp.array(logits_np, dtype=wp.bfloat16, device="cuda:0")
    partial_values = wp.empty(4, dtype=wp.float32, device="cuda:0")
    partial_tokens = wp.empty(4, dtype=wp.int32, device="cuda:0")
    token = wp.empty(1, dtype=wp.int32, device="cuda:0")
    partial, final, _ = _get_greedy_argmax_kernels(32, 4, wp.bfloat16)
    wp.launch_tiled(
        partial,
        dim=4,
        inputs=[logits, partial_values, partial_tokens],
        block_dim=32,
        device="cuda:0",
    )
    wp.launch_tiled(
        final,
        dim=1,
        inputs=[partial_values, partial_tokens, token, 17],
        block_dim=32,
        device="cuda:0",
    )
    assert token.numpy()[0] == 7


def test_bfloat16_top_k_matches_host():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(51)
    vocabulary, tile_width, top_k = 2053, 512, 20
    logits = wp.array(
        rng.normal(size=(1, 2, vocabulary)).astype(np.float32),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    partial_count = (vocabulary + tile_width - 1) // tile_width
    partial_values = wp.empty(partial_count * top_k, dtype=wp.float32, device="cuda:0")
    partial_tokens = wp.empty(partial_count * top_k, dtype=wp.int32, device="cuda:0")
    values = wp.empty(top_k, dtype=wp.float32, device="cuda:0")
    tokens = wp.empty(top_k, dtype=wp.int32, device="cuda:0")
    partial, merge = _get_top_k_kernels(tile_width, top_k, wp.bfloat16)
    wp.launch_tiled(
        partial,
        dim=partial_count,
        inputs=[logits, partial_values, partial_tokens],
        block_dim=256,
        device="cuda:0",
    )
    wp.launch_tiled(
        merge,
        dim=1,
        inputs=[partial_values, partial_tokens, values, tokens, partial_count],
        block_dim=256,
        device="cuda:0",
    )

    host_logits = logits.numpy()[0, -1].astype(np.float32)
    expected = np.lexsort((np.arange(vocabulary), -host_logits))[:top_k]
    np.testing.assert_array_equal(tokens.numpy(), expected)
    np.testing.assert_array_equal(values.numpy(), host_logits[expected])

    negative_infinity = wp.array(
        np.full((1, 1, vocabulary), -np.inf, dtype=np.float32),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    wp.launch_tiled(
        partial,
        dim=partial_count,
        inputs=[negative_infinity, partial_values, partial_tokens],
        block_dim=256,
        device="cuda:0",
    )
    wp.launch_tiled(
        merge,
        dim=1,
        inputs=[partial_values, partial_tokens, values, tokens, partial_count],
        block_dim=256,
        device="cuda:0",
    )
    np.testing.assert_array_equal(tokens.numpy(), np.arange(top_k))
    assert np.isneginf(values.numpy()).all()


def test_bfloat16_top_k_real_vocabulary_resource_shape():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    vocabulary, tile_width, top_k = 248_320, 512, 32
    logits_np = (
        np.random.default_rng(52).normal(size=(1, 1, vocabulary)).astype(np.float32)
    )
    logits_np[0, 0, 7] = 10.0
    logits_np[0, 0, 5000] = 10.0
    logits = wp.array(logits_np, dtype=wp.bfloat16, device="cuda:0")
    partial_count = (vocabulary + tile_width - 1) // tile_width
    partial_values = wp.empty(partial_count * top_k, dtype=wp.float32, device="cuda:0")
    partial_tokens = wp.empty(partial_count * top_k, dtype=wp.int32, device="cuda:0")
    merge_count = (partial_count + 15) // 16
    merge_values = wp.empty(merge_count * top_k, dtype=wp.float32, device="cuda:0")
    merge_tokens = wp.empty(merge_count * top_k, dtype=wp.int32, device="cuda:0")
    partial, merge = _get_top_k_kernels(tile_width, top_k, wp.bfloat16)
    wp.launch_tiled(
        partial,
        dim=partial_count,
        inputs=[logits, partial_values, partial_tokens],
        block_dim=256,
        device="cuda:0",
    )
    source_values, source_tokens = partial_values, partial_tokens
    target_values, target_tokens = merge_values, merge_tokens
    input_groups = partial_count
    while input_groups > 1:
        output_groups = (input_groups + 15) // 16
        wp.launch_tiled(
            merge,
            dim=output_groups,
            inputs=[
                source_values,
                source_tokens,
                target_values,
                target_tokens,
                input_groups,
            ],
            block_dim=256,
            device="cuda:0",
        )
        source_values, target_values = target_values, source_values
        source_tokens, target_tokens = target_tokens, source_tokens
        input_groups = output_groups
    host_logits = logits.numpy()[0, 0].astype(np.float32)

    expected = np.lexsort((np.arange(vocabulary), -host_logits))[:top_k]
    np.testing.assert_array_equal(source_tokens.numpy()[:top_k], expected)


@pytest.mark.parametrize("tiled_value_heads", [False, True])
@pytest.mark.parametrize("scalar_gated_delta", [False, True])
def test_mixed_state_linear_attention_bfloat16(tiled_value_heads, scalar_gated_delta):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(23)
    rows, query_heads, key_heads, value_heads, width = 2, 2, 2, 4, 8
    q = wp.array(
        rng.normal(size=(rows, query_heads * width)).astype(np.float32),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    k = wp.array(
        rng.normal(size=(rows, key_heads * width)).astype(np.float32),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    v = wp.array(
        rng.normal(size=(rows, value_heads * width)).astype(np.float32),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    past = wp.array(
        (0.05 * rng.normal(size=(value_heads * width, width))).astype(np.float32),
        device="cuda:0",
    )
    decay = wp.array(
        rng.uniform(-0.2, -0.01, size=(rows, value_heads)).astype(np.float32),
        device="cuda:0",
    )
    beta = wp.array(
        rng.uniform(0.1, 0.9, size=(rows, value_heads)).astype(np.float32),
        device="cuda:0",
    )
    output = wp.empty((rows, value_heads * width), dtype=wp.bfloat16, device="cuda:0")
    present = wp.empty_like(past)
    kernel = _get_linear_attention_kernel(
        width,
        width,
        wp.bfloat16,
        wp.float32,
        scalar_gated_delta=scalar_gated_delta,
    )

    wp.launch_tiled(
        kernel,
        dim=value_heads,
        inputs=[
            q,
            k,
            v,
            past,
            decay,
            beta,
            output,
            present,
            rows,
            query_heads,
            key_heads,
            value_heads,
            tiled_value_heads,
            True,
            False,
            True,
            True,
            width**-0.5,
        ],
        block_dim=32,
        device="cuda:0",
    )

    q_np, k_np, v_np = (array.numpy().astype(np.float32) for array in (q, k, v))
    state = past.numpy().reshape(value_heads, width, width).astype(np.float32)
    expected = np.empty((rows, value_heads * width), dtype=np.float32)
    decay_np, beta_np = decay.numpy(), beta.numpy()
    for row in range(rows):
        for value_head in range(value_heads):
            key_head = (
                value_head % key_heads
                if tiled_value_heads
                else value_head * key_heads // value_heads
            )
            key_vector = k_np[row, key_head * width : (key_head + 1) * width]
            value_vector = v_np[row, value_head * width : (value_head + 1) * width]
            state[value_head] *= np.exp(decay_np[row, value_head])
            delta = beta_np[row, value_head] * (
                value_vector - key_vector @ state[value_head]
            )
            state[value_head] += np.outer(key_vector, delta)
            query_head = (
                (key_head * query_heads // key_heads)
                if tiled_value_heads
                else (value_head * query_heads // value_heads)
            )
            query_vector = q_np[row, query_head * width : (query_head + 1) * width]
            expected[row, value_head * width : (value_head + 1) * width] = (
                width**-0.5 * query_vector @ state[value_head]
            )
    np.testing.assert_allclose(output.numpy(), expected, atol=0.08, rtol=0.03)
    np.testing.assert_allclose(
        present.numpy(),
        state.reshape(value_heads * width, width),
        atol=2.0e-4,
        rtol=2.0e-4,
    )


def test_gated_delta_preparation_and_row_causal_conv():
    rows, channels, heads, kernel_size = 2, 3, 2, 3
    x_np = np.arange(rows * channels, dtype=np.float32).reshape(rows, channels) / 5.0
    weight_np = (
        np.arange(channels * kernel_size, dtype=np.float32).reshape(
            channels, 1, kernel_size
        )
        / 10.0
    )
    state_np = np.full((channels, kernel_size - 1), 0.25, dtype=np.float32)
    x = wp.array(x_np, device="cpu")
    weight = wp.array(weight_np, device="cpu")
    state = wp.array(state_np, device="cpu")
    bias_np = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    bias = wp.array(bias_np, device="cpu")
    output = wp.empty_like(x)
    a = wp.array(np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32), device="cpu")
    b = wp.array(np.array([[-0.5, 0.2], [0.7, -0.1]], dtype=np.float32), device="cpu")
    a_log = wp.array(np.array([0.0, 0.5], dtype=np.float32), device="cpu")
    dt_bias = wp.array(np.array([0.2, -0.3], dtype=np.float32), device="cpu")
    decay = wp.empty((rows, heads), dtype=wp.float32, device="cpu")
    beta = wp.empty_like(decay)

    wp.launch(
        _causal_conv_rows_kernel,
        dim=(rows, channels),
        inputs=[x, weight, bias, state, output, True],
        device="cpu",
    )
    wp.launch(
        _update_conv_rows_state_kernel, dim=channels, inputs=[x, state], device="cpu"
    )
    wp.launch(
        _prepare_gated_delta_kernel,
        dim=(rows, heads),
        inputs=[a, b, a_log, dt_bias, False, decay, beta],
        device="cpu",
    )

    padded = np.concatenate((state_np.T, x_np), axis=0)
    expected_conv = np.empty_like(x_np)
    for row in range(rows):
        for channel in range(channels):
            total = (
                padded[row : row + kernel_size, channel] @ weight_np[channel, 0]
                + bias_np[channel]
            )
            expected_conv[row, channel] = total / (1.0 + np.exp(-total))
    np.testing.assert_allclose(output.numpy(), expected_conv, atol=1.0e-6)
    np.testing.assert_array_equal(state.numpy(), padded[-(kernel_size - 1) :].T)
    a_np, b_np = a.numpy(), b.numpy()
    expected_beta = 1.0 / (1.0 + np.exp(-b_np))
    expected_decay = -np.exp(a_log.numpy()) * np.logaddexp(0.0, a_np + dt_bias.numpy())
    np.testing.assert_allclose(beta.numpy(), expected_beta, atol=1.0e-6)
    np.testing.assert_allclose(decay.numpy(), expected_decay, atol=1.0e-6)


def test_grouped_gated_rms_norm_and_relu2():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(25)
    groups, width = 2, 4
    x_np = rng.normal(size=(3 * groups, width)).astype(np.float32)
    gate_np = rng.normal(size=x_np.shape).astype(np.float32)
    scale_np = rng.normal(size=(groups, width)).astype(np.float32)
    x = wp.array(x_np, dtype=wp.bfloat16, device="cuda:0")
    gate = wp.array(gate_np, dtype=wp.bfloat16, device="cuda:0")
    scale = wp.array(scale_np, dtype=wp.bfloat16, device="cuda:0")
    normalized = wp.empty_like(x)
    relu2 = wp.empty_like(x)
    tile_width, kernel = _get_gated_rms_norm_kernel(width, wp.bfloat16, False)

    wp.launch_tiled(
        kernel,
        dim=x.shape[0],
        inputs=[x, gate, scale, normalized, 1.0e-5],
        block_dim=tile_width,
    )
    wp.launch(_relu2_kernel, dim=x.shape, inputs=[x, relu2])

    x_rounded = x.numpy().astype(np.float32)
    gate_rounded = gate.numpy().astype(np.float32)
    gated = x_rounded * gate_rounded / (1.0 + np.exp(-gate_rounded))
    expected = gated / np.sqrt(np.mean(gated**2, axis=1, keepdims=True) + 1.0e-5)
    expected *= scale.numpy()[np.arange(x.shape[0]) % groups]
    np.testing.assert_allclose(normalized.numpy(), expected, atol=0.04, rtol=0.02)
    np.testing.assert_allclose(
        relu2.numpy(), np.maximum(x_rounded, 0.0) ** 2, atol=0.03, rtol=0.02
    )


def test_rms_norm_accepts_float32_scale():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(29)
    x_np = rng.normal(size=(3, 8)).astype(np.float32)
    scale_np = rng.normal(size=8).astype(np.float32)
    tensors = {
        "x": wp.array(x_np, dtype=wp.bfloat16, device="cuda:0"),
        "scale": wp.array(scale_np, dtype=wp.float32, device="cuda:0"),
    }
    shapes = {name: tuple(value.shape) for name, value in tensors.items()}
    operation = Operation(
        "SimplifiedLayerNormalization", ["x", "scale"], ["output"], {"epsilon": 1.0e-5}
    )
    plan_rms_norm(operation, tensors, shapes, wp.get_device("cuda:0"))

    execute_operations([operation], tensors, shapes, wp.get_device("cuda:0"))

    x_rounded = tensors["x"].numpy().astype(np.float32)
    expected = (
        x_rounded
        * scale_np
        / np.sqrt(np.mean(x_rounded**2, axis=1, keepdims=True) + 1.0e-5)
    )
    np.testing.assert_allclose(
        tensors["output"].numpy(), expected, atol=0.04, rtol=0.02
    )
