# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal optional cuBLAS binding used by ONNX transformer prefill."""

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path


def _load_library():
    names = []
    found = ctypes.util.find_library("cublas")
    if found:
        names.append(found)
    if sys.platform == "win32":
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            names.extend(
                str(Path(cuda_path) / "bin" / name)
                for name in ("cublas64_13.dll", "cublas64_12.dll")
            )
        names.extend(("cublas64_13.dll", "cublas64_12.dll"))
        loader = ctypes.WinDLL
    else:
        names.extend(("libcublas.so.13", "libcublas.so.12"))
        loader = ctypes.CDLL
    for name in names:
        try:
            return loader(name)
        except OSError:
            pass
    return None


class Cublas:
    """The small subset of cuBLAS needed for row-major 16-bit GEMM."""

    def __init__(self):
        self._lib = _load_library()
        if self._lib is None:
            raise OSError("cuBLAS is not available")

        self._lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
        self._lib.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._lib.cublasGemmEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._handle = ctypes.c_void_p()
        self._check(
            self._lib.cublasCreate_v2(ctypes.byref(self._handle)), "cublasCreate"
        )
        self._alpha = ctypes.c_float(1.0)
        self._beta = ctypes.c_float(0.0)
        self._stream = None

    @staticmethod
    def _check(status, operation):
        if status:
            raise RuntimeError(f"{operation} failed with cuBLAS status {status}")

    def close(self):
        """Release the native handle; repeated calls are harmless."""
        handle = getattr(self, "_handle", None)
        if not handle:
            return
        self._handle = None
        self._stream = None
        self._check(self._lib.cublasDestroy_v2(handle), "cublasDestroy")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def gemm(
        self, activations, weights, output, rows, columns, inner, stream, data_type
    ):
        """Compute row-major ``activations @ weights.T``."""
        if stream != self._stream:
            self._check(
                self._lib.cublasSetStream_v2(self._handle, ctypes.c_void_p(stream)),
                "cublasSetStream",
            )
            self._stream = stream
        self._check(
            self._lib.cublasGemmEx(
                self._handle,
                1,
                0,
                columns,
                rows,
                inner,
                ctypes.byref(self._alpha),
                ctypes.c_void_p(weights),
                data_type,
                inner,
                ctypes.c_void_p(activations),
                data_type,
                inner,
                ctypes.byref(self._beta),
                ctypes.c_void_p(output),
                data_type,
                columns,
                68,
                99,
            ),
            "cublasGemmEx",
        )


def try_create_cublas():
    try:
        return Cublas()
    except (OSError, RuntimeError):
        return None
