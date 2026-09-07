# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

pytest.importorskip("onnx")
import onnx
from onnx import TensorProto, helper, numpy_helper
import warp as wp

from warp_nn.runtime.formats.onnx import OnnxInitializerArchive, onnx_array_to_warp


def test_onnx_initializer_archive_loads_internal_and_external_data(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    graph = helper.make_graph(
        [],
        "weights",
        [],
        [helper.make_tensor_value_info("weight", onnx.TensorProto.FLOAT, [3, 4])],
        initializer=[numpy_helper.from_array(values, name="weight")],
    )
    model = helper.make_model(graph)
    internal = tmp_path / "internal.onnx"
    onnx.save(model, internal)
    archive = OnnxInitializerArchive(internal)
    assert archive.metadata("weight").shape == (3, 4)
    np.testing.assert_array_equal(archive.load("cpu")["weight"].numpy(), values)

    external = tmp_path / "external.onnx"
    onnx.save_model(
        model,
        external,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="external.onnx.data",
        size_threshold=0,
    )
    mapped = OnnxInitializerArchive(external)
    assert mapped.metadata("weight").path == tmp_path / "external.onnx.data"
    np.testing.assert_array_equal(mapped.load("cpu")["weight"].numpy(), values)


def test_onnx_initializer_archive_preserves_bfloat16_bits(tmp_path):
    values = np.asarray((1.0, -0.5, 3.25), dtype=np.float32)
    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
    tensor = helper.make_tensor(
        "weight", TensorProto.BFLOAT16, [3], bits.tobytes(), raw=True
    )
    graph = helper.make_graph(
        [],
        "bf16_weights",
        [],
        [helper.make_tensor_value_info("weight", TensorProto.BFLOAT16, [3])],
        initializer=[tensor],
    )
    path = tmp_path / "bf16.onnx"
    onnx.save(helper.make_model(graph), path)
    weight = OnnxInitializerArchive(path).load("cpu")["weight"]
    assert weight.dtype == wp.bfloat16
    np.testing.assert_array_equal(weight.numpy(), values)

    scalar = onnx_array_to_warp(
        np.asarray(0.5, dtype=np.dtype("bfloat16")), TensorProto.BFLOAT16, "cpu"
    )
    assert scalar.shape == (1,)
    np.testing.assert_array_equal(scalar.numpy(), (0.5,))
