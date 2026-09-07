# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused NumPy oracles for generic ONNX operators used by SkinTokens."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import warp as wp


pytest.importorskip("onnx")
import onnx
from onnx import TensorProto, helper, numpy_helper

from warp_nn.runtime import OnnxRuntime


def _initializer(name, value):
    return numpy_helper.from_array(np.asarray(value), name=name)


def _run(model, feeds):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as stream:
        path = Path(stream.name)
    try:
        onnx.save(model, path)
        runtime = OnnxRuntime(path, device="cpu", use_cublas=False)
        inputs = {
            name: wp.array(value, dtype=wp.dtype_from_numpy(value.dtype), device="cpu")
            for name, value in feeds.items()
        }
        return {name: value.numpy() for name, value in runtime(inputs).items()}
    finally:
        path.unlink(missing_ok=True)


def test_skintokens_indexing_math_and_reduction_ops():
    x = np.linspace(-1.2, 1.4, 24, dtype=np.float32).reshape(2, 3, 4)
    scores = np.asarray([2.0, 7.0, 7.0, -1.0, 4.0], dtype=np.float32)
    initializers = [
        _initializer("starts", np.asarray([1], dtype=np.int64)),
        _initializer("ends", np.asarray([4], dtype=np.int64)),
        _initializer("axes", np.asarray([2], dtype=np.int64)),
        _initializer("steps", np.asarray([1], dtype=np.int64)),
        _initializer("indices", np.asarray([2, 0], dtype=np.int64)),
        _initializer("two", np.asarray(2.0, dtype=np.float32)),
        _initializer("sum_axis", np.asarray([-1], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node("Identity", ["x"], ["identity"]),
        helper.make_node(
            "Slice", ["identity", "starts", "ends", "axes", "steps"], ["slice"]
        ),
        helper.make_node("Gather", ["slice", "indices"], ["gather"], axis=1),
        helper.make_node("Pow", ["gather", "two"], ["square"]),
        helper.make_node("ReduceSum", ["square", "sum_axis"], ["sum"], keepdims=0),
        helper.make_node("Min", ["sum", "sum"], ["minimum"]),
        helper.make_node("Sin", ["gather"], ["sin"]),
        helper.make_node("Cos", ["gather"], ["cos"]),
        helper.make_node("Erf", ["gather"], ["erf"]),
        helper.make_node("Neg", ["gather"], ["neg"]),
        helper.make_node("ArgMax", ["scores"], ["argmax"], axis=0, keepdims=0),
    ]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in (
            ("minimum", [2, 2]),
            ("sin", [2, 2, 3]),
            ("cos", [2, 2, 3]),
            ("erf", [2, 2, 3]),
            ("neg", [2, 2, 3]),
        )
    ]
    outputs.append(helper.make_tensor_value_info("argmax", TensorProto.INT64, []))
    graph = helper.make_graph(
        nodes,
        "skintokens_indexing_math",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [5]),
        ],
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8

    actual = _run(model, {"x": x, "scores": scores})
    gathered = x[:, :, 1:4][:, [2, 0]]
    np.testing.assert_allclose(
        actual["minimum"], np.sum(gathered**2, axis=-1), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(actual["sin"], np.sin(gathered), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual["cos"], np.cos(gathered), rtol=1e-6, atol=1e-6)
    expected_erf = np.vectorize(math.erf, otypes=[np.float32])(gathered)
    np.testing.assert_allclose(actual["erf"], expected_erf, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual["neg"], -gathered, rtol=0.0, atol=0.0)
    assert int(actual["argmax"].reshape(-1)[0]) == 1


def test_skintokens_attention_primitives():
    rng = np.random.default_rng(20260907)
    x = rng.standard_normal((2, 2, 3), dtype=np.float32)
    weight = rng.standard_normal((3, 5), dtype=np.float32)
    scale = rng.uniform(0.5, 1.5, 5).astype(np.float32)
    bias = rng.standard_normal(5, dtype=np.float32)
    nodes = [
        helper.make_node("MatMul", ["x", "weight"], ["projected"]),
        helper.make_node(
            "LayerNormalization",
            ["projected", "scale", "bias"],
            ["normalized"],
            axis=-1,
        ),
        helper.make_node("Softmax", ["normalized"], ["probabilities"], axis=-1),
        helper.make_node(
            "Concat", ["probabilities", "probabilities"], ["concatenated"], axis=1
        ),
        helper.make_node("Transpose", ["concatenated"], ["transposed"], perm=[0, 2, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "skintokens_attention_primitives",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2, 3])],
        [helper.make_tensor_value_info("transposed", TensorProto.FLOAT, [2, 5, 4])],
        initializer=[
            _initializer("weight", weight),
            _initializer("scale", scale),
            _initializer("bias", bias),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8

    actual = _run(model, {"x": x})["transposed"]
    projected = x @ weight
    normalized = (projected - projected.mean(axis=-1, keepdims=True)) / np.sqrt(
        projected.var(axis=-1, keepdims=True) + 1.0e-5
    )
    normalized = normalized * scale + bias
    exponentials = np.exp(normalized - normalized.max(axis=-1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
    expected = np.concatenate([probabilities, probabilities], axis=1).transpose(0, 2, 1)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_skintokens_shape_comparison_and_expand_ops():
    shape = np.asarray([2, 3], dtype=np.int64)
    nodes = [
        helper.make_node("LessOrEqual", ["x", "zero"], ["condition"]),
        helper.make_node("Equal", ["x", "x"], ["equal"]),
        helper.make_node("Where", ["condition", "negative", "positive"], ["selected"]),
        helper.make_node("Expand", ["selected", "shape"], ["expanded"]),
        helper.make_node(
            "ConstantOfShape",
            ["shape"],
            ["filled"],
            value=_initializer("fill", np.asarray([4], dtype=np.int64)),
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "skintokens_shape_ops",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
        [
            helper.make_tensor_value_info("expanded", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("equal", TensorProto.BOOL, [1, 1]),
            helper.make_tensor_value_info("filled", TensorProto.INT64, [2, 3]),
        ],
        initializer=[
            _initializer("zero", np.asarray(0.0, dtype=np.float32)),
            _initializer("negative", np.asarray(-2.0, dtype=np.float32)),
            _initializer("positive", np.asarray(3.0, dtype=np.float32)),
            _initializer("shape", shape),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8

    actual = _run(model, {"x": np.asarray([[-1.0]], dtype=np.float32)})
    np.testing.assert_array_equal(
        actual["expanded"], np.full((2, 3), -2.0, dtype=np.float32)
    )
    np.testing.assert_array_equal(actual["equal"], np.ones((1, 1), dtype=np.bool_))
    np.testing.assert_array_equal(actual["filled"], np.full((2, 3), 4, dtype=np.int64))


def test_skintokens_rank_five_shape_views_use_flat_storage():
    axes = np.asarray([0], dtype=np.int64)
    nodes = [
        helper.make_node("Unsqueeze", ["x", "axes"], ["rank_five"]),
        helper.make_node("Squeeze", ["rank_five", "axes"], ["restored"]),
    ]
    graph = helper.make_graph(
        nodes,
        "skintokens_rank_five_views",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 1, 3])],
        [helper.make_tensor_value_info("restored", TensorProto.FLOAT, [1, 2, 1, 3])],
        initializer=[_initializer("axes", axes)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8

    x = np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3)
    np.testing.assert_array_equal(_run(model, {"x": x})["restored"], x)
