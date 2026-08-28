# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare graph-replayed pure-Warp and cuBLAS dense projections."""

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import warp as wp

from warp_nn.runtime._cublas import try_create_cublas
from warp_nn.runtime.operators import Operation, execute_operations, plan_linear


def _projection(rows, columns, inner, dtype, device, stream, cublas=None, force_cublas=False):
    tensors = {
        "x": wp.ones((rows, inner), dtype=dtype, device=device),
        "weight": wp.ones((columns, inner), dtype=dtype, device=device),
    }
    shapes = {name: tuple(value.shape) for name, value in tensors.items()}
    operation = Operation("Linear", ["x", "weight"], ["output"])
    plan_linear(operation, tensors, shapes, device, cublas=cublas)
    if force_cublas:
        for name in tuple(operation.attrs):
            if name.startswith(("_packed", "_mma")):
                del operation.attrs[name]
        operation.attrs["_cublas"] = cublas
    wp.synchronize_device(device)
    with wp.ScopedStream(stream):
        with wp.ScopedCapture(device, stream=stream) as capture:
            execute_operations((operation,), tensors, shapes, device)
    return capture.graph, tensors


def _measure_pair(graphs, device, streams, iterations):
    for _ in range(10):
        for graph, stream in zip(graphs, streams):
            wp.capture_launch(graph, stream=stream)
    wp.synchronize_device(device)

    samples = ([], [])
    for sample in range(6):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for index in order:
            graph, stream = graphs[index], streams[index]
            with wp.ScopedStream(stream):
                start = wp.Event(device, enable_timing=True)
                end = wp.Event(device, enable_timing=True)
                wp.record_event(start)
                for _ in range(iterations):
                    wp.capture_launch(graph, stream=stream)
                wp.record_event(end)
                wp.synchronize_event(end)
                samples[index].append(wp.get_event_elapsed_time(start, end) / iterations)
    return tuple(statistics.median(values) for values in samples)


def _randomize(tensor_sets, rows, columns, inner, dtype, device):
    rng = np.random.default_rng(17)
    x = wp.array(rng.normal(0.0, 0.1, (rows, inner)).astype(np.float32), dtype=dtype, device=device)
    weight = wp.array(rng.normal(0.0, 0.1, (columns, inner)).astype(np.float32), dtype=dtype, device=device)
    for tensors in tensor_sets:
        wp.copy(tensors["x"], x)
        wp.copy(tensors["weight"], weight)


def _benchmark(rows, columns, inner, dtype_name, iterations, device, streams, cublas, random):
    dtype = wp.float16 if dtype_name == "float16" else wp.bfloat16
    warp_stream, cublas_stream = streams
    cublas_graph, cublas_tensors = _projection(
        rows, columns, inner, dtype, device, cublas_stream, cublas=cublas, force_cublas=True
    )
    warp_graph, warp_tensors = _projection(rows, columns, inner, dtype, device, warp_stream)
    if random:
        _randomize((warp_tensors, cublas_tensors), rows, columns, inner, dtype, device)
        wp.synchronize_device(device)
    wp.capture_launch(warp_graph, stream=warp_stream)
    wp.capture_launch(cublas_graph, stream=cublas_stream)
    wp.synchronize_device(device)
    warp_ms, cublas_ms = _measure_pair((warp_graph, cublas_graph), device, streams, iterations)
    expected = warp_tensors["x"].numpy().astype(np.float32) @ warp_tensors["weight"].numpy().astype(np.float32).T
    np.testing.assert_allclose(warp_tensors["output"].numpy(), expected, rtol=2.0e-2, atol=2.0e-2)
    np.testing.assert_allclose(cublas_tensors["output"].numpy(), expected, rtol=2.0e-2, atol=2.0e-2)
    np.testing.assert_allclose(
        warp_tensors["output"].numpy(), cublas_tensors["output"].numpy(), rtol=2.0e-2, atol=2.0e-2
    )
    faster = "Warp" if warp_ms < cublas_ms else "cuBLAS"
    ratio = max(warp_ms, cublas_ms) / min(warp_ms, cublas_ms)
    print(f"Shape:  ({rows}, {inner}) @ ({columns}, {inner}).T ({dtype_name})")
    print(f"Warp:   {warp_ms:.4f} ms")
    print(f"cuBLAS: {cublas_ms:.4f} ms")
    print(f"Faster: {faster} {ratio:.2f}x")
    print(f"Speedup: {cublas_ms / warp_ms:.4f}x vs cuBLAS")


def _run_suite(args):
    shapes = ((4096, 4096), (12288, 4096), (4096, 12288))
    for rows, label in ((1, "decode"), (16, "prefill")):
        gains = []
        print(f"\n{label}")
        for dtype_name in ("float16", "bfloat16"):
            for columns, inner in shapes:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--rows",
                    str(rows),
                    "--columns",
                    str(columns),
                    "--inner",
                    str(inner),
                    "--dtype",
                    dtype_name,
                    "--iterations",
                    str(args.iterations),
                    "--device",
                    args.device,
                    "--random",
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                lines = result.stdout.splitlines()
                print("\n".join(line for line in lines if line.startswith(("Shape:", "Warp:", "cuBLAS:", "Faster:"))))
                speedup = next(line for line in lines if line.startswith("Speedup:"))
                gains.append(float(speedup.split()[1][:-1]))
        print(f"{label} geometric mean: {statistics.geometric_mean(gains):.2f}x vs cuBLAS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--inner", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--random", action="store_true", help="Validate with deterministic random inputs")
    parser.add_argument("--suite", action="store_true", help="Run representative decode and prefill shapes")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.rows, args.columns, args.inner, args.iterations) < 1:
        parser.error("matrix sizes and iterations must be positive")

    if args.suite:
        _run_suite(args)
        return

    device = wp.get_device(args.device)
    cublas = try_create_cublas()
    if cublas is None:
        parser.error("cuBLAS is unavailable")
    streams = (wp.Stream(device), wp.Stream(device))
    _benchmark(
        args.rows,
        args.columns,
        args.inner,
        args.dtype,
        args.iterations,
        device,
        streams,
        cublas,
        args.random,
    )


if __name__ == "__main__":
    main()
