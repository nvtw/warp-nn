# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare graph-replayed pure-Warp and cuBLAS dense projections."""

import argparse
import statistics

import numpy as np
import warp as wp

from warp_nn.runtime._cublas import try_create_cublas
from warp_nn.runtime.operators import Operation, execute_operations, plan_linear


def _projection(rows, columns, inner, dtype, device, cublas=None, force_cublas=False):
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
    with wp.ScopedCapture(device) as capture:
        execute_operations((operation,), tensors, shapes, device)
    return capture.graph, tensors


def _measure(graph, device, iterations):
    for _ in range(5):
        wp.capture_launch(graph)
    wp.synchronize_device(device)
    start = wp.Event(device, enable_timing=True)
    end = wp.Event(device, enable_timing=True)
    samples = []
    for _ in range(5):
        wp.record_event(start)
        for _ in range(iterations):
            wp.capture_launch(graph)
        wp.record_event(end)
        wp.synchronize_event(end)
        samples.append(wp.get_event_elapsed_time(start, end) / iterations)
    return statistics.median(samples)


def _randomize(tensor_sets, rows, columns, inner, dtype, device):
    rng = np.random.default_rng(17)
    x = wp.array(rng.normal(0.0, 0.1, (rows, inner)).astype(np.float32), dtype=dtype, device=device)
    weight = wp.array(
        rng.normal(0.0, 0.1, (columns, inner)).astype(np.float32), dtype=dtype, device=device
    )
    for tensors in tensor_sets:
        wp.copy(tensors["x"], x)
        wp.copy(tensors["weight"], weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--inner", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--random", action="store_true", help="Validate with deterministic random inputs")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.rows, args.columns, args.inner, args.iterations) < 1:
        parser.error("matrix sizes and iterations must be positive")

    device = wp.get_device(args.device)
    dtype = wp.float16 if args.dtype == "float16" else wp.bfloat16
    cublas = try_create_cublas()
    if cublas is None:
        parser.error("cuBLAS is unavailable")
    warp_graph, warp_tensors = _projection(args.rows, args.columns, args.inner, dtype, device)
    cublas_graph, cublas_tensors = _projection(
        args.rows, args.columns, args.inner, dtype, device, cublas=cublas, force_cublas=True
    )
    if args.random:
        _randomize((warp_tensors, cublas_tensors), args.rows, args.columns, args.inner, dtype, device)
    warp_ms = _measure(warp_graph, device, args.iterations)
    cublas_ms = _measure(cublas_graph, device, args.iterations)
    np.testing.assert_allclose(
        warp_tensors["output"].numpy(), cublas_tensors["output"].numpy(), rtol=2.0e-2, atol=2.0e-2
    )

    faster = "Warp" if warp_ms < cublas_ms else "cuBLAS"
    ratio = max(warp_ms, cublas_ms) / min(warp_ms, cublas_ms)
    print(f"Shape:  ({args.rows}, {args.inner}) @ ({args.columns}, {args.inner}).T ({args.dtype})")
    print(f"Warp:   {warp_ms:.4f} ms")
    print(f"cuBLAS: {cublas_ms:.4f} ms")
    print(f"Faster: {faster} {ratio:.2f}x")


if __name__ == "__main__":
    main()
