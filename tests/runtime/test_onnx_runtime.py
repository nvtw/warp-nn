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

import pytest

import tempfile
from copy import deepcopy
from pathlib import Path


pytest.importorskip("onnx")

import onnx
from onnx import TensorProto, helper, numpy_helper

import numpy as np
import warp as wp

from tests.utilities import check_arrays, is_device_available
from warp_nn.runtime import OnnxRuntime, Qwen3OnnxRunner


def _convert_float_model_dtype(model: onnx.ModelProto, np_dtype, tensor_type: int) -> onnx.ModelProto:
    model = deepcopy(model)
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.data_type == TensorProto.FLOAT:
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np_dtype), name=initializer.name)
            )
    for value_info in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = tensor_type
    onnx.checker.check_model(model)
    return model


def _node_attrs(node) -> dict[str, float | int]:
    attrs: dict[str, float | int] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = attr.i
    return attrs


def _run_numpy_mlp(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {name: arr.astype(np.float32) for name, arr in feeds.items()}
    values.update({init.name: numpy_helper.to_array(init).astype(np.float32) for init in model.graph.initializer})

    for node in model.graph.node:
        attrs = _node_attrs(node)
        if node.op_type == "Gemm":
            A = values[node.input[0]]
            B = values[node.input[1]]
            bias = values[node.input[2]]
            if int(attrs.get("transA", 0)) != 0 or int(attrs.get("transB", 0)) != 1:
                raise NotImplementedError("policy reference only supports transA=0, transB=1")
            alpha = float(attrs.get("alpha", 1.0))
            beta = float(attrs.get("beta", 1.0))
            values[node.output[0]] = (alpha * (A @ B.T) + beta * bias).astype(np.float32)
        elif node.op_type == "Elu":
            x = values[node.input[0]]
            alpha = float(attrs.get("alpha", 1.0))
            values[node.output[0]] = np.where(x >= 0.0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32)
        else:
            raise NotImplementedError(f"unsupported op in policy reference: {node.op_type}")

    return {out.name: values[out.name] for out in model.graph.output}


def _finite_difference_gradient(fwd, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float32)
    x_flat = x.reshape(-1)
    grad_flat = grad.reshape(-1)
    for i in range(x_flat.size):
        xp = x_flat.copy()
        xm = x_flat.copy()
        xp[i] += eps
        xm[i] -= eps
        grad_flat[i] = (fwd(xp.reshape(x.shape)) - fwd(xm.reshape(x.shape))) / (2.0 * eps)
    return grad


def _runtime_input_gradient(
    rt: OnnxRuntime,
    feeds: dict[str, np.ndarray],
    *,
    input_name: str,
    output_name: str,
    seed: np.ndarray,
    device: str,
) -> np.ndarray:
    inputs = {
        name: wp.array(arr, dtype=wp.float32, device=device, requires_grad=name == input_name)
        for name, arr in feeds.items()
    }
    seed_wp = wp.array(seed, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        out = rt(inputs)[output_name]
    tape.backward(grads={out: seed_wp})

    return inputs[input_name].grad.numpy().copy()


def _build_mlp_policy_model(
    layer_sizes: tuple[int, ...],
    *,
    batch: int = 1,
    seed: int = 0,
) -> onnx.ModelProto:
    """Build a multi-layer Gemm/Elu policy network."""
    rng = np.random.default_rng(seed)
    nodes = []
    initializers = []
    in_dim = layer_sizes[0]
    x_vi = helper.make_tensor_value_info("observation", TensorProto.FLOAT, [batch, in_dim])
    prev = "observation"

    for i, out_dim in enumerate(layer_sizes[1:]):
        W = (rng.standard_normal((out_dim, in_dim)) * 0.3).astype(np.float32)
        b = (rng.standard_normal((out_dim,)) * 0.05).astype(np.float32)
        w_name, b_name = f"W{i}", f"b{i}"
        initializers.extend(
            [
                numpy_helper.from_array(W, name=w_name),
                numpy_helper.from_array(b, name=b_name),
            ]
        )
        is_last = i == len(layer_sizes) - 2
        hidden = "action" if is_last else f"hidden{i}"
        nodes.append(helper.make_node("Gemm", [prev, w_name, b_name], [hidden], alpha=1.0, beta=1.0, transB=1))
        if not is_last:
            activated = f"activated{i}"
            nodes.append(helper.make_node("Elu", [hidden], [activated], alpha=1.0))
            prev = activated
        else:
            prev = hidden
        in_dim = out_dim

    y_vi = helper.make_tensor_value_info("action", TensorProto.FLOAT, [batch, layer_sizes[-1]])
    graph = helper.make_graph(nodes, "mlp_policy", [x_vi], [y_vi], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _build_general_ops_model(batch: int, input_size: int, output_size: int, seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    arrays = {
        "weight": (rng.standard_normal((output_size, input_size)) * 0.3).astype(np.float32),
        "bias": (rng.standard_normal(output_size) * 0.05).astype(np.float32),
        "scale": rng.uniform(0.5, 1.5, output_size).astype(np.float32),
        "bn_bias": (rng.standard_normal(output_size) * 0.05).astype(np.float32),
        "mean": (rng.standard_normal(output_size) * 0.1).astype(np.float32),
        "variance": rng.uniform(0.5, 1.5, output_size).astype(np.float32),
        "epsilon": np.asarray([1.0e-6], dtype=np.float32),
        "rms_scale": rng.uniform(0.8, 1.2, output_size).astype(np.float32),
        "zero": np.asarray([0.0], dtype=np.float32),
    }
    nodes = [
        helper.make_node("Gemm", ["input", "weight", "bias"], ["linear"], transB=1),
        helper.make_node(
            "BatchNormalization",
            ["linear", "scale", "bn_bias", "mean", "variance"],
            ["normalized"],
            epsilon=1.0e-5,
        ),
        helper.make_node("Relu", ["normalized"], ["activated"]),
        helper.make_node("Mul", ["activated", "activated"], ["squared"]),
        helper.make_node("ReduceMean", ["squared"], ["mean_square"], axes=[1], keepdims=1),
        helper.make_node("Add", ["mean_square", "epsilon"], ["mean_square_epsilon"]),
        helper.make_node("Sqrt", ["mean_square_epsilon"], ["root"]),
        helper.make_node("Div", ["activated", "root"], ["unit"]),
        helper.make_node("Mul", ["unit", "rms_scale"], ["scaled_unit"]),
        helper.make_node("Sub", ["scaled_unit", "zero"], ["shifted"]),
        helper.make_node("Tanh", ["shifted"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "general_ops",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, input_size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, output_size])],
        [numpy_helper.from_array(value, name=name) for name, value in arrays.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _lstm_step_reference(
    x: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    W: np.ndarray,
    R: np.ndarray,
    Bx: np.ndarray,
    Bh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    H = h_prev.shape[1]
    gates = x @ W.T + h_prev @ R.T + Bx + Bh
    g_i = 1.0 / (1.0 + np.exp(-gates[:, 0 * H : 1 * H]))
    g_o = 1.0 / (1.0 + np.exp(-gates[:, 1 * H : 2 * H]))
    g_f = 1.0 / (1.0 + np.exp(-gates[:, 2 * H : 3 * H]))
    g_c = np.tanh(gates[:, 3 * H : 4 * H])
    c_new = g_f * c_prev + g_i * g_c
    h_new = g_o * np.tanh(c_new)
    return h_new.astype(np.float32), c_new.astype(np.float32)


def _build_lstm_step_model(batch: int, input_size: int, hidden_size: int, seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    H = hidden_size
    W = (rng.standard_normal((1, 4 * H, input_size)) * 0.3).astype(np.float32)
    R = (rng.standard_normal((1, 4 * H, H)) * 0.3).astype(np.float32)
    B = (rng.standard_normal((1, 8 * H)) * 0.05).astype(np.float32)
    Wd = (rng.standard_normal((1, H)) * 0.3).astype(np.float32)
    bd = np.zeros((1,), dtype=np.float32)

    x_in = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, batch, input_size])
    h_in = helper.make_tensor_value_info("h_in", TensorProto.FLOAT, [1, batch, H])
    c_in = helper.make_tensor_value_info("c_in", TensorProto.FLOAT, [1, batch, H])
    y_out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, 1])
    h_out_v = helper.make_tensor_value_info("h_out", TensorProto.FLOAT, [1, batch, H])
    c_out_v = helper.make_tensor_value_info("c_out", TensorProto.FLOAT, [1, batch, H])

    initializers = [
        numpy_helper.from_array(W, name="W"),
        numpy_helper.from_array(R, name="R"),
        numpy_helper.from_array(B, name="B"),
        numpy_helper.from_array(Wd, name="Wd"),
        numpy_helper.from_array(bd, name="bd"),
        numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="squeeze_axes"),
    ]
    nodes = [
        helper.make_node(
            "LSTM",
            ["input", "W", "R", "B", "", "h_in", "c_in"],
            ["Y", "h_out", "c_out"],
            hidden_size=H,
        ),
        helper.make_node("Squeeze", ["Y", "squeeze_axes"], ["Y_2d"]),
        helper.make_node("Gemm", ["Y_2d", "Wd", "bd"], ["output"], alpha=1.0, beta=1.0, transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "lstm_step",
        [x_in, h_in, c_in],
        [y_out, h_out_v, c_out_v],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


@pytest.mark.parametrize(
    "node",
    [
        helper.make_node("Gemm", [], ["out"], transB=1),
        helper.make_node("Gemm", ["missing", "B", "bias"], ["out"], transB=1),
    ],
    ids=["missing-inputs", "undefined-input"],
)
def test_rejects_invalid_model(node):
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])
    model = helper.make_model(helper.make_graph([node], "invalid", [], [output]))

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        with pytest.raises(ValueError, match="OnnxRuntime: invalid ONNX model"):
            OnnxRuntime(str(path), device="cpu")
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_mlp_policy(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    model = _build_mlp_policy_model((48, 128, 64, 12), batch=1, seed=20260430)
    rng = np.random.default_rng(20260430)
    obs = rng.standard_normal((1, 48)).astype(np.float32)
    expected = _run_numpy_mlp(model, {"observation": obs})

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=1)
        out = rt({"observation": wp.array(obs, dtype=wp.float32, device=device)})
        for name, expected_arr in expected.items():
            check_arrays(
                out[name],
                wp.array(expected_arr, dtype=wp.float32, device=device),
                rtol=1e-3,
                atol=1e-4,
            )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_streams_external_initializers(device, monkeypatch):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    model = _convert_float_model_dtype(_build_mlp_policy_model((8, 6, 4), seed=23), np.float16, TensorProto.FLOAT16)
    observation = np.random.default_rng(23).standard_normal((1, 8)).astype(np.float16)
    expected = _run_numpy_mlp(model, {"observation": observation})["action"]

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.onnx"
        data_path = Path(directory) / "weights.bin"
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=0,
        )
        unloaded = onnx.load(path, load_external_data=False)
        assert data_path.is_file()
        assert all(onnx.external_data_helper.uses_external_data(item) for item in unloaded.graph.initializer)

        loaded_tensors = []
        load_external_data_for_tensor = onnx.external_data_helper.load_external_data_for_tensor

        def track_external_load(tensor, base_dir):
            assert all(not previous.HasField("raw_data") for previous in loaded_tensors)
            load_external_data_for_tensor(tensor, base_dir)
            loaded_tensors.append(tensor)

        monkeypatch.setattr(onnx.external_data_helper, "load_external_data_for_tensor", track_external_load)
        runtime = OnnxRuntime(str(path), device=device)
        assert len(loaded_tensors) == len(model.graph.initializer)
        assert all(not tensor.HasField("raw_data") for tensor in loaded_tensors)
        assert {tensor.dtype for tensor in runtime._tensors.values()} == {wp.float16}

        output = runtime({"observation": wp.array(observation, dtype=wp.float16, device=device)})["action"]
        np.testing.assert_allclose(output.numpy(), expected, rtol=2.0e-2, atol=2.0e-2)


@pytest.mark.parametrize(
    "np_dtype,tensor_type,warp_dtype,rtol,atol",
    [
        (np.float16, TensorProto.FLOAT16, wp.float16, 2.0e-2, 2.0e-2),
        (np.float64, TensorProto.DOUBLE, wp.float64, 1.0e-6, 1.0e-6),
    ],
)
@pytest.mark.parametrize("device", ["cuda"])
def test_preserves_floating_point_dtypes(device, np_dtype, tensor_type, warp_dtype, rtol, atol):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    model = _convert_float_model_dtype(_build_mlp_policy_model((8, 6, 4), seed=17), np_dtype, tensor_type)
    rng = np.random.default_rng(17)
    observation = rng.standard_normal((1, 8)).astype(np_dtype)
    expected = _run_numpy_mlp(model, {"observation": observation})["action"]

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device)
        assert {tensor.dtype for tensor in rt._tensors.values()} == {warp_dtype}

        with pytest.raises(TypeError, match="expected"):
            rt({"observation": wp.array(observation.astype(np.float32), dtype=wp.float32, device=device)})

        output = rt({"observation": wp.array(observation, dtype=warp_dtype, device=device)})["action"]
        assert output.dtype == warp_dtype
        np.testing.assert_allclose(output.numpy(), expected, rtol=rtol, atol=atol)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_mlp_policy_input_gradients(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    model = _build_mlp_policy_model((3, 4, 1), batch=1, seed=20260716)
    rng = np.random.default_rng(20260716)
    obs = rng.standard_normal((1, 3)).astype(np.float32)
    seed = np.ones((1, 1), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=1, requires_grad=True)
        actual = _runtime_input_gradient(
            rt,
            {"observation": obs},
            input_name="observation",
            output_name="action",
            seed=seed,
            device=device,
        )

        def fwd(x):
            out = _run_numpy_mlp(model, {"observation": x})["action"]
            return float(np.sum(out * seed))

        expected = _finite_difference_gradient(fwd, obs)
        check_arrays(
            wp.array(actual, dtype=wp.float32, device=device),
            wp.array(expected, dtype=wp.float32, device=device),
            rtol=1e-2,
            atol=1e-3,
        )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_graph_capture_replay(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    model = _build_mlp_policy_model((16, 8, 4), batch=1, seed=123)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=1)
        rng = np.random.default_rng(123)
        obs_np = rng.standard_normal((1, 16)).astype(np.float32)
        obs = wp.array(obs_np, dtype=wp.float32, device=device)

        rt({"observation": obs})

        wp.capture_begin(device=device)
        try:
            out = rt({"observation": obs})
            graph = wp.capture_end(device=device)
        except Exception:
            wp.capture_end(device=device)
            raise

        replay_obs_np = rng.standard_normal((1, 16)).astype(np.float32)
        obs.assign(replay_obs_np)
        wp.capture_launch(graph)

        expected = _run_numpy_mlp(model, {"observation": replay_obs_np})
        for name, expected_arr in expected.items():
            check_arrays(
                out[name],
                wp.array(expected_arr, dtype=wp.float32, device=device),
                rtol=1e-3,
                atol=1e-4,
            )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_general_ops_graph_capture_is_deterministic(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    batch, input_size, output_size = 65, 16, 9
    model = _build_general_ops_model(batch, input_size, output_size, seed=31)
    values = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    rng = np.random.default_rng(31)
    input_np = rng.standard_normal((batch, input_size)).astype(np.float32)

    def reference(x, output="output"):
        linear = x @ values["weight"].T + values["bias"]
        normalized = (linear - values["mean"]) / np.sqrt(values["variance"] + 1.0e-5)
        activated = np.maximum(normalized * values["scale"] + values["bn_bias"], 0.0)
        root = np.sqrt(np.mean(activated * activated, axis=1, keepdims=True) + values["epsilon"])
        unit = activated / root
        if output == "unit":
            return unit.astype(np.float32)
        return np.tanh(unit * values["rms_scale"] - values["zero"]).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        with pytest.raises(NotImplementedError, match="deterministic gradients"):
            OnnxRuntime(str(path), device=device, batch_size=batch, requires_grad=True)
        rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        op_types = {op.op_type for op in rt._ops}
        assert {"_BatchNormalizationRelu", "_RmsNormalization"} <= op_types
        assert "Mul" not in op_types
        owned_ptrs = {name: int(value.ptr) for name, value in rt._tensors.items()}
        input_wp = wp.array(input_np, dtype=wp.float32, device=device)
        outputs = rt({"input": input_wp})
        check_arrays(
            outputs["output"],
            wp.array(reference(input_np), dtype=wp.float32, device=device),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

        wp.capture_begin(device=device)
        try:
            outputs = rt({"input": input_wp})
            graph = wp.capture_end(device=device)
        except Exception:
            wp.capture_end(device=device)
            raise

        replay_input = rng.standard_normal((batch, input_size)).astype(np.float32)
        input_wp.assign(replay_input)
        wp.capture_launch(graph)
        first = outputs["output"].numpy().copy()
        wp.capture_launch(graph)
        second = outputs["output"].numpy().copy()
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, reference(replay_input), rtol=1.0e-5, atol=1.0e-6)
        assert owned_ptrs == {name: int(rt._tensors[name].ptr) for name in owned_ptrs}

        model.graph.output.extend(
            [
                helper.make_tensor_value_info("normalized", TensorProto.FLOAT, [batch, output_size]),
                helper.make_tensor_value_info("squared", TensorProto.FLOAT, [batch, output_size]),
            ]
        )
        onnx.save(model, str(path))
        unfused_rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        assert all(not op.op_type.startswith("_") for op in unfused_rt._ops)
        unfused_output = unfused_rt({"input": input_wp})["output"]
        np.testing.assert_allclose(unfused_output.numpy(), reference(replay_input), rtol=1.0e-5, atol=1.0e-6)

        observable_unit_model = _build_general_ops_model(batch, input_size, output_size, seed=31)
        observable_unit_model.graph.output.append(
            helper.make_tensor_value_info("unit", TensorProto.FLOAT, [batch, output_size])
        )
        onnx.checker.check_model(observable_unit_model)
        onnx.save(observable_unit_model, str(path))
        observable_unit_rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        observable_op_types = {op.op_type for op in observable_unit_rt._ops}
        assert "_RmsNormalization" in observable_op_types
        assert "Mul" in observable_op_types
        observable_outputs = observable_unit_rt({"input": input_wp})
        np.testing.assert_allclose(
            observable_outputs["unit"].numpy(), reference(replay_input, "unit"), rtol=1.0e-5, atol=1.0e-6
        )
        np.testing.assert_allclose(
            observable_outputs["output"].numpy(), reference(replay_input), rtol=1.0e-5, atol=1.0e-6
        )

        dynamic_scale_model = _build_general_ops_model(batch, input_size, output_size, seed=31)
        scale_index = next(
            i for i, item in enumerate(dynamic_scale_model.graph.initializer) if item.name == "rms_scale"
        )
        del dynamic_scale_model.graph.initializer[scale_index]
        dynamic_scale_model.graph.input.append(
            helper.make_tensor_value_info("rms_scale", TensorProto.FLOAT, [output_size])
        )
        onnx.checker.check_model(dynamic_scale_model)
        onnx.save(dynamic_scale_model, str(path))
        dynamic_scale_rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        dynamic_op_types = {op.op_type for op in dynamic_scale_rt._ops}
        assert "_RmsNormalization" in dynamic_op_types
        assert "Mul" in dynamic_op_types
        dynamic_scale_output = dynamic_scale_rt(
            {
                "input": input_wp,
                "rms_scale": wp.array(values["rms_scale"], dtype=wp.float32, device=device),
            }
        )["output"]
        np.testing.assert_allclose(dynamic_scale_output.numpy(), reference(replay_input), rtol=1.0e-5, atol=1.0e-6)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_general_ops_float16(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    batch, input_size, output_size = 3, 8, 5
    model = _convert_float_model_dtype(
        _build_general_ops_model(batch, input_size, output_size, seed=11), np.float16, TensorProto.FLOAT16
    )
    values = {item.name: numpy_helper.to_array(item).astype(np.float32) for item in model.graph.initializer}
    input_np = np.random.default_rng(11).standard_normal((batch, input_size)).astype(np.float16)
    linear = input_np.astype(np.float32) @ values["weight"].T + values["bias"]
    normalized = (linear - values["mean"]) / np.sqrt(values["variance"] + 1.0e-5)
    activated = np.maximum(normalized * values["scale"] + values["bn_bias"], 0.0)
    unit = activated / np.sqrt(np.mean(activated * activated, axis=1, keepdims=True) + values["epsilon"])
    expected = np.tanh(unit * values["rms_scale"] - values["zero"])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        output = OnnxRuntime(str(path), device=device, batch_size=batch)(
            {"input": wp.array(input_np, dtype=wp.float16, device=device)}
        )["output"]
        assert output.dtype == wp.float16
        np.testing.assert_allclose(output.numpy(), expected, rtol=2.0e-2, atol=2.0e-2)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_constant_reshape_is_a_view(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    shape = numpy_helper.from_array(np.asarray([0, -1, 2], dtype=np.int64), name="shape_value")
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Constant", [], ["shape"], value=shape),
                helper.make_node("Reshape", ["input", "shape"], ["output"]),
            ],
            "reshape_view",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2, 3, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [2, 6, 2])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    input_np = np.arange(24, dtype=np.float16).reshape(2, 3, 4)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        runtime = OnnxRuntime(str(path), device=device)
        input_wp = wp.array(input_np, dtype=wp.float16, device=device)
        output = runtime({"input": input_wp})["output"]
        assert output.shape == (2, 6, 2)
        assert output.dtype == wp.float16
        assert output.ptr == input_wp.ptr
        np.testing.assert_array_equal(output.numpy(), input_np.reshape(2, 6, 2))
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_gather_block_quantized_int8(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    rng = np.random.default_rng(29)
    data = rng.integers(0, 256, size=(4, 256), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.02, size=(4, 2)).astype(np.float16)
    zero_points = rng.integers(96, 160, size=(4, 2), dtype=np.uint8)
    indices = np.asarray([[0, 3], [2, 1]], dtype=np.int64)
    expected = np.empty((2, 2, 256), dtype=np.float16)
    for batch in range(2):
        for sequence in range(2):
            row = indices[batch, sequence]
            for column in range(256):
                block = column // 128
                expected[batch, sequence, column] = (
                    np.float32(data[row, column]) - np.float32(zero_points[row, block])
                ) * np.float32(scales[row, block])

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "GatherBlockQuantized",
                    ["data", "input_ids", "scales", "zero_points"],
                    ["output"],
                    domain="com.microsoft",
                    bits=8,
                    block_size=128,
                )
            ],
            "quantized_embedding",
            [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [2, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [2, 2, 256])],
            [
                numpy_helper.from_array(data, name="data"),
                numpy_helper.from_array(scales, name="scales"),
                numpy_helper.from_array(zero_points, name="zero_points"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        output = OnnxRuntime(str(path), device=device)({"input_ids": wp.array(indices, dtype=wp.int64, device=device)})[
            "output"
        ]
        np.testing.assert_array_equal(output.numpy(), expected)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_gather_bfloat16_embedding(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    table = np.arange(35, dtype=np.float32).reshape(7, 5).astype(np.dtype("bfloat16"))
    indices = np.array([[6, 1, 3], [0, 4, 2]], dtype=np.int64)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Gather", ["table", "indices"], ["output"])],
            "embedding",
            [helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape))],
            [helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [2, 3, 5])],
            [numpy_helper.from_array(table, name="table")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        output = OnnxRuntime(str(path), device=device)({"indices": wp.array(indices, dtype=wp.int64, device=device)})[
            "output"
        ]
        np.testing.assert_array_equal(output.numpy(), table[indices])
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_transformer_cast_and_int32_metadata(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
    values = np.arange(12, dtype=np.float16).reshape(2, 2, 3)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Cast", ["mask"], ["mask_i32"], to=TensorProto.INT32),
                helper.make_node("ReduceSum", ["mask_i32", "axis"], ["lengths"], keepdims=0),
                helper.make_node("ReduceMax", ["lengths"], ["max_length"], keepdims=0),
                helper.make_node("Sub", ["lengths", "one"], ["last_indices"]),
                helper.make_node("Cast", ["values"], ["values_f32"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["values_f32"], ["restored"], to=TensorProto.FLOAT16),
            ],
            "transformer_metadata",
            [
                helper.make_tensor_value_info("mask", TensorProto.INT64, list(mask.shape)),
                helper.make_tensor_value_info("values", TensorProto.FLOAT16, list(values.shape)),
            ],
            [
                helper.make_tensor_value_info("last_indices", TensorProto.INT32, [2]),
                helper.make_tensor_value_info("max_length", TensorProto.INT32, []),
                helper.make_tensor_value_info("restored", TensorProto.FLOAT16, list(values.shape)),
            ],
            [
                numpy_helper.from_array(np.array([1], dtype=np.int64), name="axis"),
                numpy_helper.from_array(np.array([1], dtype=np.int32), name="one"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        outputs = OnnxRuntime(str(path), device=device)(
            {
                "mask": wp.array(mask, dtype=wp.int64, device=device),
                "values": wp.array(values, dtype=wp.float16, device=device),
            }
        )
        np.testing.assert_array_equal(outputs["last_indices"].numpy(), mask.sum(axis=1).astype(np.int32) - 1)
        assert outputs["max_length"].numpy()[0] == 3
        np.testing.assert_array_equal(outputs["restored"].numpy(), values)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_transformer_layout_ops(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    x = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["x", "axis"], ["expanded"]),
                helper.make_node("Squeeze", ["expanded", "axis"], ["restored"]),
                helper.make_node("Transpose", ["x"], ["transposed3"], perm=[0, 2, 1]),
                helper.make_node("Transpose", ["expanded"], ["transposed4"], perm=[0, 2, 1, 3]),
                helper.make_node("Split", ["transposed3", "split"], ["left", "right"], axis=-1),
                helper.make_node("Tile", ["x", "repeats"], ["tiled"]),
            ],
            "transformer_layout",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, list(x.shape))],
            [
                helper.make_tensor_value_info("restored", TensorProto.FLOAT16, list(x.shape)),
                helper.make_tensor_value_info("transposed4", TensorProto.FLOAT16, [1, 3, 2, 4]),
                helper.make_tensor_value_info("left", TensorProto.FLOAT16, [2, 4, 1]),
                helper.make_tensor_value_info("right", TensorProto.FLOAT16, [2, 4, 2]),
                helper.make_tensor_value_info("tiled", TensorProto.FLOAT16, [4, 3, 4]),
            ],
            [
                numpy_helper.from_array(np.array([0], dtype=np.int64), name="axis"),
                numpy_helper.from_array(np.array([1, 2], dtype=np.int64), name="split"),
                numpy_helper.from_array(np.array([2, 1, 1], dtype=np.int64), name="repeats"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        outputs = OnnxRuntime(str(path), device=device)({"x": wp.array(x, dtype=wp.float16, device=device)})
        transposed = x.transpose(0, 2, 1)
        np.testing.assert_array_equal(outputs["restored"].numpy(), x)
        np.testing.assert_array_equal(outputs["transposed4"].numpy(), x[None].transpose(0, 2, 1, 3))
        np.testing.assert_array_equal(outputs["left"].numpy(), transposed[..., :1])
        np.testing.assert_array_equal(outputs["right"].numpy(), transposed[..., 1:])
        np.testing.assert_array_equal(outputs["tiled"].numpy(), np.tile(x, (2, 1, 1)))
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_transformer_activations_and_normalization(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    rng = np.random.default_rng(83)
    x = rng.standard_normal((2, 3, 4)).astype(np.float16)
    q = rng.standard_normal((1, 2, 3, 4)).astype(np.float16)
    bias = rng.standard_normal(4).astype(np.float16)
    scale = np.array([0.25], dtype=np.float16)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Sigmoid", ["x"], ["sigmoid"]),
                helper.make_node("Softplus", ["x"], ["softplus"]),
                helper.make_node("Add", ["x", "bias"], ["biased"]),
                helper.make_node("Mul", ["x", "scale"], ["scaled"]),
                helper.make_node("LpNormalization", ["q"], ["normalized"], axis=-1, p=2),
            ],
            "transformer_activations",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT16, list(x.shape)),
                helper.make_tensor_value_info("q", TensorProto.FLOAT16, list(q.shape)),
            ],
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT16, list(x.shape))
                for name in ("sigmoid", "softplus", "biased", "scaled")
            ]
            + [helper.make_tensor_value_info("normalized", TensorProto.FLOAT16, list(q.shape))],
            [numpy_helper.from_array(bias, name="bias"), numpy_helper.from_array(scale, name="scale")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        outputs = OnnxRuntime(str(path), device=device)(
            {
                "x": wp.array(x, dtype=wp.float16, device=device),
                "q": wp.array(q, dtype=wp.float16, device=device),
            }
        )
        x_fp32 = x.astype(np.float32)
        np.testing.assert_allclose(outputs["sigmoid"].numpy(), 1.0 / (1.0 + np.exp(-x_fp32)), atol=1e-3)
        np.testing.assert_allclose(outputs["softplus"].numpy(), np.logaddexp(0.0, x_fp32), atol=1e-3)
        np.testing.assert_array_equal(outputs["biased"].numpy(), (x + bias).astype(np.float16))
        np.testing.assert_array_equal(outputs["scaled"].numpy(), (x * scale[0]).astype(np.float16))
        expected_norm = q.astype(np.float32) / np.linalg.norm(q.astype(np.float32), axis=-1, keepdims=True)
        np.testing.assert_allclose(outputs["normalized"].numpy(), expected_norm, rtol=2e-3, atol=2e-3)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_transformer_selection_ops(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    condition = np.array([True, False, True, False])
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    y = -x
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Range", ["start", "limit", "delta"], ["range"]),
                helper.make_node("Slice", ["source", "starts", "ends", "axes"], ["slice"]),
                helper.make_node("Where", ["condition", "x", "y"], ["selected"]),
            ],
            "transformer_selection",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
                helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y.shape)),
            ],
            [
                helper.make_tensor_value_info("range", TensorProto.INT64, [4]),
                helper.make_tensor_value_info("slice", TensorProto.INT64, [2]),
                helper.make_tensor_value_info("selected", TensorProto.FLOAT, list(x.shape)),
            ],
            [
                numpy_helper.from_array(np.array(0, dtype=np.int64), name="start"),
                numpy_helper.from_array(np.array(4, dtype=np.int64), name="limit"),
                numpy_helper.from_array(np.array(1, dtype=np.int64), name="delta"),
                numpy_helper.from_array(np.array([4, 5, 6], dtype=np.int64), name="source"),
                numpy_helper.from_array(np.array([1], dtype=np.int64), name="starts"),
                numpy_helper.from_array(np.array([3], dtype=np.int64), name="ends"),
                numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes"),
                numpy_helper.from_array(condition, name="condition"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        outputs = OnnxRuntime(str(path), device=device)(
            {
                "x": wp.array(x, dtype=wp.float32, device=device),
                "y": wp.array(y, dtype=wp.float32, device=device),
            }
        )
        np.testing.assert_array_equal(outputs["range"].numpy(), np.arange(4, dtype=np.int64))
        np.testing.assert_array_equal(outputs["slice"].numpy(), np.array([5, 6], dtype=np.int64))
        np.testing.assert_array_equal(outputs["selected"].numpy(), np.where(condition, x, y))
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("batch,sequence", [(2, 3), (1, 1)])
@pytest.mark.parametrize("use_cublas", [False, True])
@pytest.mark.parametrize(
    "data_type,bits,block_size,has_zero_points",
    [
        (TensorProto.FLOAT16, 4, 128, True),
        (TensorProto.FLOAT16, 8, 128, True),
        (TensorProto.BFLOAT16, 4, 32, False),
    ],
)
@pytest.mark.parametrize("device", ["cuda"])
def test_matmul_nbits(device, bits, batch, sequence, use_cublas, data_type, block_size, has_zero_points):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    rng = np.random.default_rng(37 + bits)
    K, N = 256, 5
    np_dtype = np.float16 if data_type == TensorProto.FLOAT16 else np.dtype("bfloat16")
    wp_dtype = wp.float16 if data_type == TensorProto.FLOAT16 else wp.bfloat16
    blocks = K // block_size
    activations = rng.standard_normal((batch, sequence, K)).astype(np_dtype)
    quantized = rng.integers(0, 1 << bits, size=(N, blocks, block_size), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.02, size=(N, blocks)).astype(np_dtype)
    zero_values = (
        rng.integers(0, 1 << bits, size=(N, blocks), dtype=np.uint8)
        if has_zero_points
        else np.full((N, blocks), 1 << (bits - 1), dtype=np.uint8)
    )
    if bits == 4:
        weights = quantized[:, :, 0::2] | (quantized[:, :, 1::2] << 4)
        zero_points = zero_values[:, 0::2] | (zero_values[:, 1::2] << 4)
    else:
        weights = quantized
        zero_points = zero_values

    dequantized = (
        (quantized.astype(np.float32) - zero_values[:, :, None].astype(np.float32))
        * scales[:, :, None].astype(np.float32)
    ).reshape(N, K)
    expected = (activations.astype(np.float32) @ dequantized.T).astype(np_dtype)
    node_inputs = ["activations", "weights", "scales"]
    initializers = [
        numpy_helper.from_array(weights, name="weights"),
        numpy_helper.from_array(scales, name="scales"),
    ]
    if has_zero_points:
        node_inputs.append("zero_points")
        initializers.append(numpy_helper.from_array(zero_points, name="zero_points"))
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "MatMulNBits",
                    node_inputs,
                    ["output"],
                    domain="com.microsoft",
                    K=K,
                    N=N,
                    bits=bits,
                    block_size=block_size,
                    accuracy_level=0 if has_zero_points else 4,
                )
            ],
            "quantized_matmul",
            [helper.make_tensor_value_info("activations", data_type, [batch, sequence, K])],
            [helper.make_tensor_value_info("output", data_type, [batch, sequence, N])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        runtime = OnnxRuntime(str(path), device=device, use_cublas=use_cublas)
        if not use_cublas:
            assert runtime._cublas is None
        inputs = {"activations": wp.array(activations, dtype=wp_dtype, device=device)}
        output = runtime(inputs)["output"]
        np.testing.assert_allclose(output.numpy(), expected, rtol=2.0e-2, atol=2.0e-2)

        if use_cublas and batch * sequence > 1 and runtime._cublas is not None:
            wp.capture_begin(device=device)
            try:
                runtime(inputs)
                graph = wp.capture_end(device=device)
            except Exception:
                wp.capture_end(device=device)
                raise
            wp.capture_launch(graph)
            np.testing.assert_allclose(output.numpy(), expected, rtol=2.0e-2, atol=2.0e-2)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "data_type,np_dtype,wp_dtype,sequence,activation",
    [
        (TensorProto.FLOAT16, np.float16, wp.float16, 5, "silu"),
        (TensorProto.BFLOAT16, np.dtype("bfloat16"), wp.bfloat16, 1, "none"),
    ],
)
@pytest.mark.parametrize("device", ["cuda"])
def test_causal_conv_with_state(device, data_type, np_dtype, wp_dtype, sequence, activation):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    rng = np.random.default_rng(71)
    batch, channels, kernel_size = 2, 3, 4
    x = rng.standard_normal((batch, channels, sequence)).astype(np_dtype)
    weight = rng.standard_normal((channels, 1, kernel_size)).astype(np_dtype)
    bias = rng.standard_normal(channels).astype(np_dtype)
    past = rng.standard_normal((batch, channels, kernel_size - 1)).astype(np_dtype)

    history = np.concatenate((past.astype(np.float32), x.astype(np.float32)), axis=2)
    expected = np.empty_like(x)
    for position in range(sequence):
        window = history[:, :, position : position + kernel_size]
        value = np.sum(window * weight[:, 0, :].astype(np.float32)[None, :, :], axis=2)
        value += bias.astype(np.float32)[None, :]
        if activation == "silu":
            value /= 1.0 + np.exp(-value)
        expected[:, :, position] = value.astype(np_dtype)
    expected_state = history[:, :, -(kernel_size - 1) :].astype(np_dtype)

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "CausalConvWithState",
                    ["x", "weight", "bias", "past"],
                    ["output", "present"],
                    domain="com.microsoft",
                    ndim=1,
                    activation=activation,
                )
            ],
            "causal_conv",
            [
                helper.make_tensor_value_info("x", data_type, list(x.shape)),
                helper.make_tensor_value_info("past", data_type, list(past.shape)),
            ],
            [
                helper.make_tensor_value_info("output", data_type, list(x.shape)),
                helper.make_tensor_value_info("present", data_type, list(past.shape)),
            ],
            [numpy_helper.from_array(weight, name="weight"), numpy_helper.from_array(bias, name="bias")],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        runtime = OnnxRuntime(str(path), device=device)
        inputs = {
            "x": wp.array(x, dtype=wp_dtype, device=device),
            "past": wp.array(past, dtype=wp_dtype, device=device),
        }
        outputs = runtime(inputs)
        np.testing.assert_allclose(outputs["output"].numpy(), expected, rtol=1.0e-2, atol=1.0e-2)
        np.testing.assert_array_equal(outputs["present"].numpy(), expected_state)
        wp.capture_begin(device=device)
        try:
            runtime(inputs)
            graph = wp.capture_end(device=device)
        except Exception:
            wp.capture_end(device=device)
            raise
        wp.capture_launch(graph)
        np.testing.assert_allclose(outputs["output"].numpy(), expected, rtol=1.0e-2, atol=1.0e-2)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "query_heads,value_heads,key_heads,key_size,value_size,update_rule",
    [(16, 32, 16, 128, 128, "gated_delta"), (4, 2, 2, 8, 6, "linear")],
)
@pytest.mark.parametrize("device", ["cuda"])
def test_linear_attention(device, query_heads, value_heads, key_heads, key_size, value_size, update_rule):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    rng = np.random.default_rng(79)
    batch, sequence = 1, 3
    q = rng.standard_normal((batch, sequence, query_heads * key_size)).astype(np.float16)
    k = rng.standard_normal((batch, sequence, key_heads * key_size)).astype(np.float16)
    q = (
        (
            q.reshape(batch, sequence, query_heads, key_size)
            / np.linalg.norm(
                q.astype(np.float32).reshape(batch, sequence, query_heads, key_size), axis=-1, keepdims=True
            )
        )
        .reshape(batch, sequence, -1)
        .astype(np.float16)
    )
    k = (
        (
            k.reshape(batch, sequence, key_heads, key_size)
            / np.linalg.norm(k.astype(np.float32).reshape(batch, sequence, key_heads, key_size), axis=-1, keepdims=True)
        )
        .reshape(batch, sequence, -1)
        .astype(np.float16)
    )
    v = rng.standard_normal((batch, sequence, value_heads * value_size)).astype(np.float16)
    past = (0.1 * rng.standard_normal((batch, value_heads, key_size, value_size))).astype(np.float16)
    decay = rng.uniform(-0.2, -0.01, (batch, sequence, value_heads)).astype(np.float16)
    beta = rng.uniform(0.1, 0.9, (batch, sequence, value_heads)).astype(np.float16)
    scale = 0.7

    state = past.astype(np.float32)
    expected = np.empty((batch, sequence, max(query_heads, value_heads) * value_size), dtype=np.float16)
    for token in range(sequence):
        for value_head in range(value_heads):
            key_head = value_head * key_heads // value_heads
            query_head = value_head * query_heads // value_heads if query_heads < value_heads else None
            key_vector = k[0, token, key_head * key_size : (key_head + 1) * key_size].astype(np.float32)
            value_vector = v[0, token, value_head * value_size : (value_head + 1) * value_size].astype(np.float32)
            if update_rule == "gated_delta":
                state[0, value_head] *= np.exp(np.float32(decay[0, token, value_head]))
                retrieved = key_vector @ state[0, value_head]
                delta = np.float32(beta[0, token, value_head]) * (value_vector - retrieved)
            else:
                delta = value_vector
            state[0, value_head] += np.outer(key_vector, delta)
            if query_heads >= value_heads:
                for group in range(query_heads // value_heads):
                    query_head = value_head * (query_heads // value_heads) + group
                    query_vector = q[0, token, query_head * key_size : (query_head + 1) * key_size].astype(np.float32)
                    expected[0, token, query_head * value_size : (query_head + 1) * value_size] = (
                        scale * query_vector @ state[0, value_head]
                    ).astype(np.float16)
            else:
                query_vector = q[0, token, query_head * key_size : (query_head + 1) * key_size].astype(np.float32)
                expected[0, token, value_head * value_size : (value_head + 1) * value_size] = (
                    scale * query_vector @ state[0, value_head]
                ).astype(np.float16)
    expected_state = state.astype(np.float16)

    inputs = ["q", "k", "v", "past", "decay", "beta"]
    graph_inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, list(value.shape))
        for name, value in {"q": q, "k": k, "v": v, "past": past}.items()
    ]
    feeds = {"q": q, "k": k, "v": v, "past": past}
    if update_rule == "gated_delta":
        graph_inputs.extend(
            [
                helper.make_tensor_value_info("decay", TensorProto.FLOAT16, list(decay.shape)),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT16, list(beta.shape)),
            ]
        )
        feeds.update(decay=decay, beta=beta)
    else:
        inputs[4:] = ["", ""]

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "LinearAttention",
                    inputs,
                    ["output", "present"],
                    domain="com.microsoft",
                    q_num_heads=query_heads,
                    kv_num_heads=value_heads,
                    update_rule=update_rule,
                    scale=scale,
                )
            ],
            "linear_attention",
            graph_inputs,
            [
                helper.make_tensor_value_info("output", TensorProto.FLOAT16, list(expected.shape)),
                helper.make_tensor_value_info("present", TensorProto.FLOAT16, list(expected_state.shape)),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        runtime = OnnxRuntime(str(path), device=device)
        warp_feeds = {name: wp.array(value, dtype=wp.float16, device=device) for name, value in feeds.items()}
        outputs = runtime(warp_feeds)
        np.testing.assert_allclose(outputs["output"].numpy(), expected, rtol=5.0e-2, atol=5.0e-2)
        np.testing.assert_allclose(outputs["present"].numpy(), expected_state, rtol=5.0e-2, atol=5.0e-2)
        wp.capture_begin(device=device)
        try:
            runtime(warp_feeds)
            graph = wp.capture_end(device=device)
        except Exception:
            wp.capture_end(device=device)
            raise
        wp.capture_launch(graph)
        np.testing.assert_allclose(outputs["output"].numpy(), expected, rtol=5.0e-2, atol=5.0e-2)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "data_type,np_dtype,wp_dtype",
    [
        (TensorProto.FLOAT16, np.float16, wp.float16),
        (TensorProto.BFLOAT16, np.dtype("bfloat16"), wp.bfloat16),
    ],
)
@pytest.mark.parametrize("device", ["cuda"])
def test_qwen_normalization_and_swiglu(device, data_type, np_dtype, wp_dtype):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    rng = np.random.default_rng(53)
    shape = (2, 1, 3, 128)
    x = rng.standard_normal(shape).astype(np_dtype)
    skip = rng.standard_normal(shape).astype(np_dtype)
    gate = rng.standard_normal(shape).astype(np_dtype)
    up = rng.standard_normal(shape).astype(np_dtype)
    scale = rng.uniform(0.8, 1.2, shape[-1]).astype(np_dtype)
    epsilon = 1.0e-6

    def rms_norm(value):
        inverse_rms = 1.0 / np.sqrt(np.mean(value.astype(np.float32) ** 2, axis=-1, keepdims=True) + epsilon)
        return (value.astype(np.float32) * inverse_rms * scale.astype(np.float32)).astype(np_dtype)

    residual_fp32 = x.astype(np.float32) + skip.astype(np.float32)
    residual = residual_fp32.astype(np_dtype)
    expected = {
        "normalized": rms_norm(x),
        "skip_normalized": rms_norm(residual_fp32),
        "residual": residual,
        "swiglu": (gate.astype(np.float32) / (1.0 + np.exp(-gate.astype(np.float32))) * up.astype(np.float32)).astype(
            np_dtype
        ),
    }
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "SimplifiedLayerNormalization",
                    ["x", "scale"],
                    ["normalized"],
                    epsilon=epsilon,
                    axis=-1,
                    stash_type=1,
                ),
                helper.make_node(
                    "SkipSimplifiedLayerNormalization",
                    ["x", "skip", "scale"],
                    ["skip_normalized", "", "", "residual"],
                    domain="com.microsoft",
                    epsilon=epsilon,
                ),
                helper.make_node("Sigmoid", ["gate"], ["sigmoid"]),
                helper.make_node("Mul", ["gate", "sigmoid"], ["silu"]),
                helper.make_node("Mul", ["silu", "up"], ["swiglu"]),
            ],
            "qwen_feed_forward",
            [helper.make_tensor_value_info(name, data_type, list(shape)) for name in ("x", "skip", "gate", "up")],
            [helper.make_tensor_value_info(name, data_type, list(shape)) for name in expected],
            [numpy_helper.from_array(scale, name="scale")],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        runtime = OnnxRuntime(str(path), device=device)
        assert "_SwiGLU" in {op.op_type for op in runtime._ops}
        outputs = runtime(
            {
                name: wp.array(value, dtype=wp_dtype, device=device)
                for name, value in {"x": x, "skip": skip, "gate": gate, "up": up}.items()
            }
        )
        for name, reference in expected.items():
            np.testing.assert_allclose(outputs[name].numpy(), reference, rtol=1.0e-2, atol=1.0e-2)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("sequence_length,past_length", [(2, 2), (7, 5)])
@pytest.mark.parametrize("device", ["cuda"])
def test_qwen_group_query_attention(device, sequence_length, past_length):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    rng = np.random.default_rng(59)
    batch = 1
    query_heads, kv_heads, head_size = 4, 2, 16
    total_length = sequence_length + past_length
    query = rng.normal(0.0, 0.2, (batch, sequence_length, query_heads * head_size)).astype(np.float16)
    key = rng.normal(0.0, 0.2, (batch, sequence_length, kv_heads * head_size)).astype(np.float16)
    value = rng.normal(0.0, 0.2, key.shape).astype(np.float16)
    past_key = rng.normal(0.0, 0.2, (batch, kv_heads, past_length, head_size)).astype(np.float16)
    past_value = rng.normal(0.0, 0.2, past_key.shape).astype(np.float16)
    attention_mask = np.ones((batch, total_length), dtype=np.int64)
    positions = np.arange(16, dtype=np.float32)[:, None]
    frequencies = 1.0 / (10000.0 ** (np.arange(head_size // 2, dtype=np.float32) * 2.0 / head_size))
    cos_cache = np.cos(positions * frequencies).astype(np.float16)
    sin_cache = np.sin(positions * frequencies).astype(np.float16)

    def rotate(x, heads):
        shaped = x.reshape(batch, sequence_length, heads, head_size).transpose(0, 2, 1, 3).astype(np.float32)
        first, second = np.split(shaped, 2, axis=-1)
        cosine = cos_cache[past_length : past_length + sequence_length].astype(np.float32)[None, None]
        sine = sin_cache[past_length : past_length + sequence_length].astype(np.float32)[None, None]
        return np.concatenate((first * cosine - second * sine, second * cosine + first * sine), axis=-1).astype(
            np.float16
        )

    rotated_query = rotate(query, query_heads)
    rotated_key = rotate(key, kv_heads)
    present_key = np.concatenate((past_key, rotated_key), axis=2)
    present_value = np.concatenate(
        (past_value, value.reshape(batch, sequence_length, kv_heads, head_size).transpose(0, 2, 1, 3)), axis=2
    )
    expanded_key = np.repeat(present_key, query_heads // kv_heads, axis=1)
    expanded_value = np.repeat(present_value, query_heads // kv_heads, axis=1)
    scores = np.einsum("bhsd,bhtd->bhst", rotated_query.astype(np.float32), expanded_key.astype(np.float32)) * (
        head_size**-0.5
    )
    for token in range(sequence_length):
        scores[:, :, token, past_length + token + 1 :] = -np.inf
    probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    output = np.einsum("bhst,bhtd->bshd", probabilities, expanded_value.astype(np.float32)).reshape(
        batch, sequence_length, query_heads * head_size
    )
    expected = {"output": output.astype(np.float16), "present_key": present_key, "present_value": present_value}

    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    one = numpy_helper.from_array(np.array([1], dtype=np.int64), name="one")
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("ReduceSum", ["attention_mask", "axes"], ["mask_sum"], keepdims=0),
                helper.make_node("Sub", ["mask_sum", "one"], ["sequence_lengths_i64"]),
                helper.make_node("Cast", ["sequence_lengths_i64"], ["sequence_lengths"], to=TensorProto.INT32),
                helper.make_node("Shape", ["attention_mask"], ["mask_shape"]),
                helper.make_node("Gather", ["mask_shape", "one"], ["total_length_i64"], axis=0),
                helper.make_node("Cast", ["total_length_i64"], ["total_length"], to=TensorProto.INT32),
                helper.make_node(
                    "GroupQueryAttention",
                    [
                        "query",
                        "key",
                        "value",
                        "past_key",
                        "past_value",
                        "sequence_lengths",
                        "total_length",
                        "cos_cache",
                        "sin_cache",
                        "",
                        "",
                    ],
                    ["output", "present_key", "present_value"],
                    domain="com.microsoft",
                    num_heads=query_heads,
                    kv_num_heads=kv_heads,
                    scale=head_size**-0.5,
                    softcap=0.0,
                    do_rotary=1,
                    rotary_interleaved=0,
                ),
            ],
            "qwen_attention",
            [
                helper.make_tensor_value_info(
                    "query", TensorProto.FLOAT16, ["batch", "sequence", query_heads * head_size]
                ),
                helper.make_tensor_value_info("key", TensorProto.FLOAT16, ["batch", "sequence", kv_heads * head_size]),
                helper.make_tensor_value_info(
                    "value", TensorProto.FLOAT16, ["batch", "sequence", kv_heads * head_size]
                ),
                helper.make_tensor_value_info("past_key", TensorProto.FLOAT16, ["batch", kv_heads, "past", head_size]),
                helper.make_tensor_value_info(
                    "past_value", TensorProto.FLOAT16, ["batch", kv_heads, "past", head_size]
                ),
                helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "total"]),
            ],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT16, ["batch", "sequence", query_heads * head_size]
                ),
                helper.make_tensor_value_info(
                    "present_key", TensorProto.FLOAT16, ["batch", kv_heads, "total", head_size]
                ),
                helper.make_tensor_value_info(
                    "present_value", TensorProto.FLOAT16, ["batch", kv_heads, "total", head_size]
                ),
            ],
            [
                axes,
                one,
                numpy_helper.from_array(cos_cache, name="cos_cache"),
                numpy_helper.from_array(sin_cache, name="sin_cache"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    inputs = {
        "query": query,
        "key": key,
        "value": value,
        "past_key": past_key,
        "past_value": past_value,
        "attention_mask": attention_mask,
    }
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        runtime = OnnxRuntime(
            str(path), device=device, input_shapes={name: value.shape for name, value in inputs.items()}
        )
        assert all("_scores" not in op.attrs for op in runtime._ops)
        outputs = runtime(
            {
                name: wp.array(value, dtype=wp.dtype_from_numpy(value.dtype), device=device)
                for name, value in inputs.items()
            }
        )
        for name, reference in expected.items():
            np.testing.assert_allclose(outputs[name].numpy(), reference, rtol=4.0e-3, atol=4.0e-3)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_qwen_stateful_prefill_and_decode(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    rng = np.random.default_rng(67)
    vocabulary, head_size, cache_length = 4, 128, 8
    embedding = rng.integers(96, 160, size=(vocabulary, head_size), dtype=np.uint8)
    scales = rng.uniform(0.005, 0.02, size=(vocabulary, 1)).astype(np.float16)
    zero_points = np.full((vocabulary, 1), 128, dtype=np.uint8)
    lm_weight = rng.integers(96, 160, size=(vocabulary, 1, head_size), dtype=np.uint8)
    lm_scales = rng.uniform(0.005, 0.02, size=(vocabulary, 1)).astype(np.float16)
    lm_zero_points = np.full((vocabulary, 1), 128, dtype=np.uint8)
    positions = np.arange(cache_length, dtype=np.float32)[:, None]
    frequencies = 1.0 / (10000.0 ** (np.arange(head_size // 2, dtype=np.float32) * 2.0 / head_size))
    cos_cache = np.cos(positions * frequencies).astype(np.float16)
    sin_cache = np.sin(positions * frequencies).astype(np.float16)
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    one = numpy_helper.from_array(np.array([1], dtype=np.int64), name="one")
    nodes = [
        helper.make_node("ReduceSum", ["attention_mask", "axes"], ["mask_sum"], keepdims=0),
        helper.make_node("Sub", ["mask_sum", "one"], ["sequence_lengths_i64"]),
        helper.make_node("Cast", ["sequence_lengths_i64"], ["sequence_lengths"], to=TensorProto.INT32),
        helper.make_node("Shape", ["attention_mask"], ["mask_shape"]),
        helper.make_node("Gather", ["mask_shape", "one"], ["total_length_i64"], axis=0),
        helper.make_node("Cast", ["total_length_i64"], ["total_length"], to=TensorProto.INT32),
        helper.make_node(
            "GatherBlockQuantized",
            ["embedding", "input_ids", "scales", "zero_points"],
            ["hidden"],
            domain="com.microsoft",
            bits=8,
            block_size=128,
        ),
        helper.make_node(
            "GroupQueryAttention",
            [
                "hidden",
                "hidden",
                "hidden",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "sequence_lengths",
                "total_length",
                "cos_cache",
                "sin_cache",
                "",
                "",
            ],
            ["attention", "present.0.key", "present.0.value"],
            domain="com.microsoft",
            num_heads=1,
            kv_num_heads=1,
            scale=head_size**-0.5,
            softcap=0.0,
            do_rotary=1,
            rotary_interleaved=0,
        ),
        helper.make_node(
            "MatMulNBits",
            ["attention", "lm_weight", "lm_scales", "lm_zero_points"],
            ["logits"],
            domain="com.microsoft",
            bits=8,
            block_size=128,
            accuracy_level=0,
            K=head_size,
            N=vocabulary,
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "stateful_qwen",
            [
                helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
                helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "total"]),
                helper.make_tensor_value_info(
                    "past_key_values.0.key", TensorProto.FLOAT16, ["batch", 1, "past", head_size]
                ),
                helper.make_tensor_value_info(
                    "past_key_values.0.value", TensorProto.FLOAT16, ["batch", 1, "past", head_size]
                ),
            ],
            [
                helper.make_tensor_value_info("logits", TensorProto.FLOAT16, ["batch", "sequence", vocabulary]),
                helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT16, ["batch", 1, "total", head_size]),
                helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT16, ["batch", 1, "total", head_size]),
            ],
            [
                axes,
                one,
                numpy_helper.from_array(embedding, name="embedding"),
                numpy_helper.from_array(scales, name="scales"),
                numpy_helper.from_array(zero_points, name="zero_points"),
                numpy_helper.from_array(lm_weight, name="lm_weight"),
                numpy_helper.from_array(lm_scales, name="lm_scales"),
                numpy_helper.from_array(lm_zero_points, name="lm_zero_points"),
                numpy_helper.from_array(cos_cache, name="cos_cache"),
                numpy_helper.from_array(sin_cache, name="sin_cache"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        runner = Qwen3OnnxRunner(str(path), device=device)
        prompt_logits = runner.prefill([0, 1])
        assert prompt_logits.shape == (1, 2, vocabulary)
        decoded = runner.decode(2).numpy()
        assert runner.sequence_length == 3
        full = runner.prefill([0, 1, 2]).numpy()
        np.testing.assert_allclose(decoded, full[:, -1:, :], rtol=1.0e-3, atol=1.0e-3)

        logits = runner.prefill([0, 1])
        expected = []
        for _ in range(3):
            token_id = int(np.argmax(logits.numpy()[0, -1]))
            expected.append(token_id)
            logits = runner.decode(token_id)
        assert runner.generate_greedy([0, 1], max_new_tokens=3, eos_token_id=-1) == expected
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_rejects_unsupported_ops(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    model = helper.make_model(
        helper.make_graph(
            nodes=[helper.make_node("Sigmoid", ["A"], ["Y"])],
            name="reject",
            inputs=[helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        with pytest.raises(NotImplementedError, match="unsupported op 'Sigmoid'"):
            OnnxRuntime(str(path), device=device)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_rejects_unsupported_op_variants(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        model = _build_general_ops_model(2, 4, 4)
        del model.graph.node[0].attribute[:]
        onnx.save(model, str(path))
        with pytest.raises(NotImplementedError, match="only transB=1"):
            OnnxRuntime(str(path), device=device, batch_size=2)

        model = _build_general_ops_model(2, 4, 4)
        epsilon = next(item for item in model.graph.initializer if item.name == "epsilon")
        epsilon.CopyFrom(numpy_helper.from_array(np.asarray([[1.0e-6]], dtype=np.float32), name="epsilon"))
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        with pytest.raises(ValueError, match=r"epsilon must have shape \(1,\)"):
            OnnxRuntime(str(path), device=device, batch_size=2)

        model = helper.make_model(
            helper.make_graph(
                nodes=[helper.make_node("Add", ["A", "B"], ["Y"])],
                name="reject_1d_binary",
                inputs=[
                    helper.make_tensor_value_info("A", TensorProto.FLOAT, [4]),
                    helper.make_tensor_value_info("B", TensorProto.FLOAT, [4]),
                ],
                outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
        )
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        with pytest.raises(NotImplementedError, match="at least one input must be 2-D"):
            OnnxRuntime(str(path), device=device)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_lstm_single_step(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    batch, input_size, hidden_size = 4, 2, 8
    model = _build_lstm_step_model(batch, input_size, hidden_size, seed=20260430)

    W = numpy_helper.to_array(model.graph.initializer[0])[0]
    R = numpy_helper.to_array(model.graph.initializer[1])[0]
    B_full = numpy_helper.to_array(model.graph.initializer[2])[0]
    Wd = numpy_helper.to_array(model.graph.initializer[3])[0]
    bd = numpy_helper.to_array(model.graph.initializer[4])
    Bx, Bh = B_full[: 4 * hidden_size], B_full[4 * hidden_size :]

    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((1, batch, input_size)).astype(np.float32)
    h_np = rng.standard_normal((1, batch, hidden_size)).astype(np.float32) * 0.1
    c_np = rng.standard_normal((1, batch, hidden_size)).astype(np.float32) * 0.1

    h_ref, c_ref = _lstm_step_reference(x_np[0], h_np[0], c_np[0], W, R, Bx, Bh)
    expected_out = (h_ref @ Wd.T + bd).reshape(batch, 1).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        out = rt(
            {
                "input": wp.array(x_np, dtype=wp.float32, device=device),
                "h_in": wp.array(h_np, dtype=wp.float32, device=device),
                "c_in": wp.array(c_np, dtype=wp.float32, device=device),
            }
        )
        check_arrays(
            out["output"],
            wp.array(expected_out, dtype=wp.float32, device=device),
            rtol=1e-3,
            atol=1e-4,
        )
        check_arrays(
            out["h_out"],
            wp.array(h_ref.reshape(1, batch, hidden_size), dtype=wp.float32, device=device),
            rtol=1e-3,
            atol=1e-4,
        )
        check_arrays(
            out["c_out"],
            wp.array(c_ref.reshape(1, batch, hidden_size), dtype=wp.float32, device=device),
            rtol=1e-3,
            atol=1e-4,
        )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_lstm_float16_preserves_integer_axes(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    batch, input_size, hidden_size = 2, 3, 4
    model = _convert_float_model_dtype(
        _build_lstm_step_model(batch, input_size, hidden_size, seed=13), np.float16, TensorProto.FLOAT16
    )
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        assert rt._tensors["squeeze_axes"].dtype == wp.int64
        zeros = {
            "input": wp.zeros((1, batch, input_size), dtype=wp.float16, device=device),
            "h_in": wp.zeros((1, batch, hidden_size), dtype=wp.float16, device=device),
            "c_in": wp.zeros((1, batch, hidden_size), dtype=wp.float16, device=device),
        }
        assert {output.dtype for output in rt(zeros).values()} == {wp.float16}
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_lstm_input_gradients(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    batch, input_size, hidden_size = 1, 3, 4
    model = _build_lstm_step_model(batch, input_size, hidden_size, seed=20260716)

    W = numpy_helper.to_array(model.graph.initializer[0])[0]
    R = numpy_helper.to_array(model.graph.initializer[1])[0]
    B_full = numpy_helper.to_array(model.graph.initializer[2])[0]
    Wd = numpy_helper.to_array(model.graph.initializer[3])[0]
    bd = numpy_helper.to_array(model.graph.initializer[4])
    Bx, Bh = B_full[: 4 * hidden_size], B_full[4 * hidden_size :]

    rng = np.random.default_rng(16)
    x_np = rng.standard_normal((1, batch, input_size)).astype(np.float32)
    h_np = rng.standard_normal((1, batch, hidden_size)).astype(np.float32) * 0.1
    c_np = rng.standard_normal((1, batch, hidden_size)).astype(np.float32) * 0.1
    seed = np.ones((batch, 1), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=batch, requires_grad=True)
        actual = _runtime_input_gradient(
            rt,
            {"input": x_np, "h_in": h_np, "c_in": c_np},
            input_name="input",
            output_name="output",
            seed=seed,
            device=device,
        )

        def fwd(x):
            h_ref, _ = _lstm_step_reference(x[0], h_np[0], c_np[0], W, R, Bx, Bh)
            out = (h_ref @ Wd.T + bd).reshape(batch, 1).astype(np.float32)
            return float(np.sum(out * seed))

        expected = _finite_difference_gradient(fwd, x_np)
        check_arrays(
            wp.array(actual, dtype=wp.float32, device=device),
            wp.array(expected, dtype=wp.float32, device=device),
            rtol=1e-2,
            atol=1e-3,
        )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", ["cuda"])
def test_lstm_graph_capture_replay(device):
    if not is_device_available(device):
        pytest.skip(f"Device '{device}' is not available")

    batch, input_size, hidden_size = 2, 2, 4
    model = _build_lstm_step_model(batch, input_size, hidden_size, seed=4242)

    W = numpy_helper.to_array(model.graph.initializer[0])[0]
    R = numpy_helper.to_array(model.graph.initializer[1])[0]
    B_full = numpy_helper.to_array(model.graph.initializer[2])[0]
    Wd = numpy_helper.to_array(model.graph.initializer[3])[0]
    bd = numpy_helper.to_array(model.graph.initializer[4])
    Bx, Bh = B_full[: 4 * hidden_size], B_full[4 * hidden_size :]

    rng = np.random.default_rng(9)
    x_np = rng.standard_normal((1, batch, input_size)).astype(np.float32)
    h_np = np.zeros((1, batch, hidden_size), dtype=np.float32)
    c_np = np.zeros((1, batch, hidden_size), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        onnx.save(model, str(path))
        rt = OnnxRuntime(str(path), device=device, batch_size=batch)
        x_wp = wp.array(x_np, dtype=wp.float32, device=device)
        h_wp = wp.array(h_np, dtype=wp.float32, device=device)
        c_wp = wp.array(c_np, dtype=wp.float32, device=device)

        rt({"input": x_wp, "h_in": h_wp, "c_in": c_wp})

        wp.capture_begin(device=device)
        try:
            out = rt({"input": x_wp, "h_in": h_wp, "c_in": c_wp})
            graph = wp.capture_end(device=device)
        except Exception:
            wp.capture_end(device=device)
            raise

        replay_x = rng.standard_normal((1, batch, input_size)).astype(np.float32)
        x_wp.assign(replay_x)
        wp.capture_launch(graph)

        h_ref, _ = _lstm_step_reference(replay_x[0], h_np[0], c_np[0], W, R, Bx, Bh)
        expected = (h_ref @ Wd.T + bd).reshape(batch, 1).astype(np.float32)
        check_arrays(
            out["output"],
            wp.array(expected, dtype=wp.float32, device=device),
            rtol=1e-3,
            atol=1e-4,
        )
    finally:
        path.unlink(missing_ok=True)
