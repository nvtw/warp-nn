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
from warp_nn.runtime.kernels import _get_dequantize_nbits_kernel


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
    if args.rows < 1 or args.inner % 32 or min(args.columns, args.iterations) < 1:
        parser.error("rows must be positive, inner must be divisible by 32, and other sizes must be positive")

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
        if args.rows == 1 and cublas_runtime._cublas is not None:
            for op in cublas_runtime._ops:
                if op.op_type == "MatMulNBits":
                    for name in tuple(op.attrs):
                        if name.startswith("_q8_"):
                            del op.attrs[name]
                    op.attrs["_cublas"] = cublas_runtime._cublas
                    op.attrs["_dequantized_weights"] = wp.empty(
                        (args.columns, args.inner), dtype=wp.float16, device=device
                    )
                    op.attrs["_dequantize_kernel"] = _get_dequantize_nbits_kernel(4, 32, wp.float16)
        warp_ms, warp_output = measure(warp_runtime, activations, args.iterations)
        if cublas_runtime._cublas is None:
            print(f"Warp packed INT4: {warp_ms:.3f} ms (cuBLAS unavailable)")
            return
        cublas_ms, cublas_output = measure(cublas_runtime, activations, args.iterations)

    difference = np.abs(warp_output.astype(np.float32) - cublas_output.astype(np.float32))
    np.testing.assert_allclose(
        warp_output,
        cublas_output,
        rtol=5.0e-2 if args.rows == 1 else 2.0e-2,
        atol=7.5e-2 if args.rows == 1 else 2.0e-2,
    )
    print(f"Shape: ({args.rows}, {args.inner}) @ ({args.columns}, {args.inner}).T")
    print(f"Warp {'INT4/Q8 DP4A' if args.rows == 1 else 'packed INT4'}: {warp_ms:.3f} ms")
    print(f"cuBLAS + dequant: {cublas_ms:.3f} ms")
    faster = "Warp" if warp_ms < cublas_ms else "cuBLAS"
    print(f"Faster path:      {faster} {max(warp_ms, cublas_ms) / min(warp_ms, cublas_ms):.2f}x")
    print(f"Difference:       mean {difference.mean():.5f}, max {difference.max():.5f}")


if __name__ == "__main__":
    main()
