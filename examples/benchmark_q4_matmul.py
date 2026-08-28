# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare packed-INT4 Warp and cuBLAS MatMulNBits execution."""

import argparse
import tempfile
from pathlib import Path
import onnx
from onnx import TensorProto, helper, numpy_helper

import numpy as np
import warp as wp

from warp_nn.runtime import OnnxRuntime


def make_model(path: Path, rows: int, columns: int, inner: int) -> None:
    rng = np.random.default_rng(17)
    blocks = inner // 32
    weights = rng.integers(0, 256, size=(columns, blocks, 16), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.02, size=(columns, blocks)).astype(np.float16)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "MatMulNBits",
                    ["activations", "weights", "scales"],
                    ["output"],
                    domain="com.microsoft",
                    K=inner,
                    N=columns,
                    bits=4,
                    block_size=32,
                    accuracy_level=4,
                )
            ],
            "q4_matmul_benchmark",
            [helper.make_tensor_value_info("activations", TensorProto.FLOAT16, [rows, inner])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [rows, columns])],
            [
                numpy_helper.from_array(weights, name="weights"),
                numpy_helper.from_array(scales, name="scales"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    onnx.save(model, path)


def measure(runtime: OnnxRuntime, activations: wp.array, iterations: int) -> tuple[float, np.ndarray]:
    inputs = {"activations": activations}
    for _ in range(5):
        output = runtime(inputs)["output"]
    wp.synchronize_device(activations.device)

    start = wp.Event(activations.device, enable_timing=True)
    end = wp.Event(activations.device, enable_timing=True)
    samples = []
    for _ in range(3):
        wp.record_event(start)
        for _ in range(iterations):
            output = runtime(inputs)["output"]
        wp.record_event(end)
        samples.append(wp.get_event_elapsed_time(start, end) / iterations)
    return float(np.median(samples)), output.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--inner", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.rows < 2 or args.inner % 32 or min(args.columns, args.iterations) < 1:
        parser.error("rows must be at least 2, inner must be divisible by 32, and other sizes must be positive")

    wp.init()
    device = wp.get_device(args.device)
    activations = wp.array(
        np.random.default_rng(23).standard_normal((args.rows, args.inner)).astype(np.float16),
        device=device,
    )
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "matmul.onnx"
        make_model(model_path, args.rows, args.columns, args.inner)
        warp_runtime = OnnxRuntime(model_path, device=device, use_cublas=False)
        cublas_runtime = OnnxRuntime(model_path, device=device, use_cublas=True)
        warp_ms, warp_output = measure(warp_runtime, activations, args.iterations)
        if cublas_runtime._cublas is None:
            print(f"Warp packed INT4: {warp_ms:.3f} ms (cuBLAS unavailable)")
            return
        cublas_ms, cublas_output = measure(cublas_runtime, activations, args.iterations)

    np.testing.assert_allclose(warp_output, cublas_output, rtol=2.0e-2, atol=2.0e-2)
    print(f"Shape: ({args.rows}, {args.inner}) @ ({args.columns}, {args.inner}).T")
    print(f"Warp packed INT4: {warp_ms:.3f} ms")
    print(f"cuBLAS + dequant: {cublas_ms:.3f} ms")
    print(f"cuBLAS speedup:   {warp_ms / cublas_ms:.2f}x")


if __name__ == "__main__":
    main()
