# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused layout, correctness, and graph tests for native SM120 NVFP4."""

import gc
import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.formats.gguf import BlockQuantizedTensor
from warp_nn.runtime.operators import (
    Operation,
    execute_operations,
    plan_linear,
    reuse_operation_outputs,
)
from warp_nn.runtime.quantization import (
    enable_nvfp4_native,
    launch_nvfp4_linear,
    launch_quantize_nvfp4,
    load_native_weights,
    repack_gguf_nvfp4_weight,
)

_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)


def _sm120():
    if not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    device = wp.get_device("cuda:0")
    if device.arch != 120:
        pytest.skip("native NVFP4 requires SM120")
    return enable_nvfp4_native(device)


def _decode_e4m3(code):
    exponent = (int(code) >> 3) & 15
    mantissa = int(code) & 7
    if exponent == 0:
        return np.ldexp(float(mantissa), -9)
    return np.ldexp(1.0 + mantissa / 8.0, exponent - 7)


def _dequantize(packed, scales, inner, global_scales=None):
    output = np.empty((packed.shape[0], inner), np.float32)
    for row in range(packed.shape[0]):
        for column in range(inner):
            code = (packed[row, column // 2] >> (4 * (column & 1))) & 15
            sign = -1.0 if code & 8 else 1.0
            output[row, column] = (
                sign * _E2M1[code & 7] * _decode_e4m3(scales[row, column // 16])
            )
    if global_scales is not None:
        output *= np.asarray(global_scales, dtype=np.float32)[:, None]
    return output


def test_nvfp4_gguf_repack_basis_order():
    device = _sm120()
    # GGUF stores K0..7 in low nibbles and K8..15 in high nibbles.
    block = np.zeros((1, 1, 32), np.uint8)
    for subblock in range(4):
        for index in range(8):
            low = (subblock * 16 + index) & 15
            high = (subblock * 16 + index + 8) & 15
            block[0, 0, subblock * 8 + index] = low | (high << 4)
    values = wp.array(block, device=device)
    scales = wp.ones((1, 1, 4), dtype=wp.uint8, device=device)
    words = wp.array(
        ptr=values.ptr,
        dtype=wp.uint32,
        shape=(1, 1, 8),
        capacity=values.capacity,
        device=device,
        copy=False,
    )
    weight = BlockQuantizedTensor(values, words, scales, (1, 64), "NVFP4")
    repacked = repack_gguf_nvfp4_weight(weight).values.numpy()[0, 0]
    expected = np.array([index | ((index + 1) << 4) for index in range(0, 16, 2)] * 4)
    np.testing.assert_array_equal(repacked, expected.astype(np.uint8))


def test_nvfp4_quantization_adjacent_nibbles():
    device = _sm120()
    host = np.zeros((1, 16), np.float32)
    host[0, :2] = (0.5, 6.0)
    values = wp.array(host, dtype=wp.bfloat16, device=device)
    packed = wp.empty((1, 8), dtype=wp.uint8, device=device)
    scales = wp.empty((1, 1), dtype=wp.uint8, device=device)
    global_scales = wp.empty(1, dtype=wp.float32, device=device)
    launch_quantize_nvfp4(values, packed, scales, global_scales)
    wp.synchronize_device(device)
    assert scales.numpy()[0, 0] == 0x7E  # E4M3 448.0
    assert packed.numpy()[0, 0] == 0x71  # E2M1 [0.5, 6.0]
    np.testing.assert_allclose(global_scales.numpy(), [1.0 / 448.0])


def test_nvfp4_two_level_scaling_prevents_small_activation_underflow():
    device = _sm120()
    rng = np.random.default_rng(91)
    host = rng.normal(0.0, 0.002, (1, 256)).astype(np.float32)
    host[0, -1] = 0.108
    values = wp.array(host, dtype=wp.bfloat16, device=device)
    dynamic_values = wp.empty((1, 128), dtype=wp.uint8, device=device)
    dynamic_scales = wp.empty((1, 16), dtype=wp.uint8, device=device)
    dynamic_global = wp.empty(1, dtype=wp.float32, device=device)
    direct_values = wp.empty((1, 128), dtype=wp.uint8, device=device)
    direct_scales = wp.empty((1, 16), dtype=wp.uint8, device=device)
    direct_global = wp.ones(1, dtype=wp.float32, device=device)
    launch_quantize_nvfp4(values, dynamic_values, dynamic_scales, dynamic_global)
    launch_quantize_nvfp4(
        values,
        direct_values,
        direct_scales,
        direct_global,
        compute_global_scale=False,
    )
    wp.synchronize_device(device)
    source = values.numpy().astype(np.float32)
    dynamic = _dequantize(
        dynamic_values.numpy(),
        dynamic_scales.numpy(),
        256,
        dynamic_global.numpy(),
    )
    direct = _dequantize(
        direct_values.numpy(), direct_scales.numpy(), 256, direct_global.numpy()
    )
    dynamic_error = np.linalg.norm(dynamic - source) / np.linalg.norm(source)
    direct_error = np.linalg.norm(direct - source) / np.linalg.norm(source)
    assert np.mean(direct_scales.numpy() == 0) > 0.8
    assert np.count_nonzero(dynamic_scales.numpy()) == dynamic_scales.size
    assert dynamic_error < direct_error * 0.3


@pytest.mark.parametrize("inner", [128, 5120])
@pytest.mark.parametrize(
    ("rows", "split_k", "reuse_weights"),
    [
        (16, 1, False),
        (16, 8, False),
        (64, 1, True),
    ],
)
def test_nvfp4_mma_random_reference_and_graph(inner, rows, split_k, reuse_weights):
    device = _sm120()
    rng = np.random.default_rng(123)
    activations_host = rng.normal(0.0, 0.7, (rows, inner)).astype(np.float32)
    weights_host = rng.normal(0.0, 0.7, (16, inner)).astype(np.float32)
    activations = wp.array(activations_host, dtype=wp.bfloat16, device=device)
    weights = wp.array(weights_host, dtype=wp.bfloat16, device=device)
    activation_values = wp.empty((rows, inner // 2), dtype=wp.uint8, device=device)
    activation_scales = wp.empty((rows, inner // 16), dtype=wp.uint8, device=device)
    activation_global_scales = wp.empty(rows, dtype=wp.float32, device=device)
    weight_values = wp.empty((16, inner // 2), dtype=wp.uint8, device=device)
    weight_scales = wp.empty((16, inner // 16), dtype=wp.uint8, device=device)
    weight_global_scales = wp.ones(16, dtype=wp.float32, device=device)
    output = wp.empty((rows, 16), dtype=wp.bfloat16, device=device)

    def launch():
        launch_quantize_nvfp4(
            activations,
            activation_values,
            activation_scales,
            activation_global_scales,
        )
        launch_quantize_nvfp4(
            weights,
            weight_values,
            weight_scales,
            weight_global_scales,
            compute_global_scale=False,
        )
        launch_nvfp4_linear(
            activation_values,
            activation_scales,
            activation_global_scales,
            weight_values,
            weight_scales,
            output,
            reuse_weights=reuse_weights,
            split_k=split_k,
        )

    launch()
    wp.synchronize_device(device)
    with wp.ScopedCapture(device) as capture:
        launch()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    a = _dequantize(
        activation_values.numpy(),
        activation_scales.numpy(),
        inner,
        activation_global_scales.numpy(),
    )
    w = _dequantize(weight_values.numpy(), weight_scales.numpy(), inner)
    reference = a @ w.T
    np.testing.assert_allclose(
        output.numpy().astype(np.float32), reference, rtol=1e-2, atol=2.5e-1
    )


def test_nvfp4_quantization_accepts_zeroed_decode_padding():
    device = _sm120()
    values = wp.ones((1, 64), dtype=wp.bfloat16, device=device)
    packed = wp.zeros((16, 32), dtype=wp.uint8, device=device)
    scales = wp.zeros((16, 4), dtype=wp.uint8, device=device)
    global_scales = wp.zeros(16, dtype=wp.float32, device=device)
    launch_quantize_nvfp4(values, packed, scales, global_scales)
    wp.synchronize_device(device)
    assert np.all(packed.numpy()[0] != 0)
    assert np.all(packed.numpy()[1:] == 0)
    assert np.all(scales.numpy()[1:] == 0)
    assert np.all(global_scales.numpy()[1:] == 0)


def test_nvfp4_linear_operation_pads_single_row():
    device = _sm120()
    rows, columns, inner = 1, 8, 64
    # E2M1 code 2 is 1.0; E4M3 code 0x38 is also 1.0. Uniform
    # nibbles deliberately remain valid in GGUF's split-half ordering.
    values = wp.full((columns, 1, 32), 0x22, dtype=wp.uint8, device=device)
    scales = wp.full((columns, 1, 4), 0x38, dtype=wp.uint8, device=device)
    words = wp.array(
        ptr=values.ptr,
        dtype=wp.uint32,
        shape=(columns, 1, 8),
        capacity=values.capacity,
        device=device,
        copy=False,
    )
    tensors = {
        "x": wp.full((rows, inner), 6.0, dtype=wp.bfloat16, device=device),
        "weight": BlockQuantizedTensor(
            values, words, scales, (columns, inner), "NVFP4"
        ),
    }
    shapes = {"x": (rows, inner), "weight": (columns, inner)}
    operations = [
        Operation(
            "Linear",
            ["x", "weight"],
            [f"output.{index}"],
            {"_output_scale": 0.5 / (index + 1)},
        )
        for index in range(2)
    ]
    cache = {}
    for operation in operations:
        plan_linear(
            operation,
            tensors,
            shapes,
            device,
            quantized_activation_cache=cache,
        )
    assert operations[0].attrs["_nvfp4_padded_rows"] == 16
    assert tensors["output.0"].shape == (1, 8)
    assert len(cache) == 1
    assert "_nvfp4_quantize_kernel" in operations[0].attrs
    assert "_nvfp4_quantize_kernel" not in operations[1].attrs
    assert (
        operations[0].attrs["_nvfp4_activations"].ptr
        == operations[1].attrs["_nvfp4_activations"].ptr
    )
    assert (
        operations[0].attrs["_nvfp4_global_scales"].ptr
        == operations[1].attrs["_nvfp4_global_scales"].ptr
    )
    first_owner = operations[0].attrs["_nvfp4_output"]
    pool = {}
    reuse_operation_outputs({"projection": operations[0]}, tensors, pool)
    reuse_operation_outputs({"projection": operations[1]}, tensors, pool)
    assert operations[0].attrs["_nvfp4_output"] is first_owner
    assert tensors["output.0"].ptr == tensors["output.1"].ptr
    assert operations[1].attrs["_nvfp4_output"].ptr == tensors["output.1"].ptr
    # Exercise ownership: later aliases must not replace/free the pool founder.
    del first_owner
    gc.collect()
    wp.empty((1024,), dtype=wp.float32, device=device)
    execute_operations(operations, tensors, shapes, device)
    np.testing.assert_array_equal(
        tensors["output.1"].numpy(), np.full((1, 8), 96.0, np.float32)
    )


def test_nvfp4_archive_load_prepares_weight_once_for_all_plans():
    device = _sm120()

    class Metadata:
        shape = (8, 64)
        dtype = wp.uint8
        format = "NVFP4"
        nbytes = 8 * 36

    class Archive:
        def __init__(self):
            self.loads = []

        def metadata(self, name):
            assert name == "weight"
            return Metadata()

        def load(self, target, names):
            self.loads.append(tuple(names))
            if not names:
                return {}
            values = wp.full((8, 1, 32), 0x22, dtype=wp.uint8, device=target)
            scales = wp.full((8, 1, 4), 0x38, dtype=wp.uint8, device=target)
            words = wp.array(
                ptr=values.ptr,
                dtype=wp.uint32,
                shape=(8, 1, 8),
                capacity=values.capacity,
                device=target,
                copy=False,
            )
            return {
                "weight": BlockQuantizedTensor(values, words, scales, (8, 64), "NVFP4")
            }

    archive = Archive()
    loaded = load_native_weights(archive, device, ["weight"], None)
    weight = loaded["weight"]
    assert weight.format == "NVFP4_MMA"
    prepared_ptr = weight.values.ptr
    tensors = {
        "x": wp.full((1, 64), 6.0, dtype=wp.bfloat16, device=device),
        "weight": weight,
    }
    shapes = {"x": (1, 64), "weight": (8, 64)}
    cache = {}
    operations = [
        Operation("Linear", ["x", "weight"], [f"output.{index}"]) for index in range(2)
    ]
    for operation in operations:
        plan_linear(
            operation,
            tensors,
            shapes,
            device,
            quantized_activation_cache=cache,
        )
    assert tensors["weight"].format == "NVFP4_MMA"
    assert tensors["weight"].values.ptr == prepared_ptr
    assert archive.loads == [(), ("weight",)]
