# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Read-only ONNX initializer archive with mmap external-data uploads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import warp as wp


@dataclass(frozen=True)
class OnnxTensorMetadata:
    """Type, shape, and byte location of one ONNX initializer."""

    dtype: type
    shape: tuple[int, ...]
    nbytes: int
    path: Path | None
    offset: int


def _require_onnx():
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - only on missing extra
        raise ImportError(
            "ONNX weight loading requires `pip install onnx>=1.16.0`"
        ) from exc
    return onnx, numpy_helper


def onnx_array_to_warp(
    array, element_type: int, device=None, *, requires_grad: bool = False
) -> wp.array:
    """Upload one ONNX array while preserving BF16 storage bits exactly."""
    onnx, _ = _require_onnx()
    device = wp.get_device(device)
    host = np.asarray(array)
    if element_type != onnx.TensorProto.BFLOAT16:
        host = np.ascontiguousarray(host)
        dtype = wp.dtype_from_numpy(host.dtype)
        return wp.array(
            host,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad
            and dtype in (wp.float16, wp.float32, wp.float64),
        )
    storage_shape = host.shape or (1,)
    bits = np.ascontiguousarray(host).view(np.uint16).reshape(storage_shape)
    output = wp.empty(
        storage_shape,
        dtype=wp.bfloat16,
        device=device,
        requires_grad=requires_grad,
    )
    output_bits = wp.array(
        ptr=output.ptr,
        capacity=output.capacity,
        shape=output.shape,
        dtype=wp.uint16,
        device=device,
        copy=False,
    )
    output_bits.assign(bits)
    return output


class OnnxInitializerArchive:
    """Expose ONNX initializers through Warp-NN's common archive protocol."""

    def __init__(self, path: str | Path):
        onnx, _ = _require_onnx()
        self.path = Path(path).expanduser().resolve()
        self._model = onnx.load(self.path, load_external_data=False)
        self._initializers = {
            value.name: value for value in self._model.graph.initializer
        }
        self._metadata = {}
        for name, value in self._initializers.items():
            try:
                np_dtype = np.dtype(
                    onnx.helper.tensor_dtype_to_np_dtype(value.data_type)
                )
                dtype = (
                    wp.bfloat16
                    if value.data_type == onnx.TensorProto.BFLOAT16
                    else wp.dtype_from_numpy(np_dtype)
                )
            except (KeyError, TypeError) as exc:
                raise TypeError(
                    f"unsupported ONNX initializer dtype for '{name}'"
                ) from exc
            shape = tuple(int(dimension) for dimension in value.dims)
            nbytes = int(np.prod(shape, dtype=np.int64)) * np_dtype.itemsize
            external = {entry.key: entry.value for entry in value.external_data}
            location = external.get("location")
            source = None
            offset = 0
            if location is not None:
                source = (self.path.parent / location).resolve()
                if source.parent != self.path.parent or not source.is_file():
                    raise ValueError(
                        f"ONNX initializer '{name}' references invalid external data"
                    )
                offset = int(external.get("offset", 0))
                length = int(external.get("length", nbytes))
                if (
                    offset < 0
                    or length < nbytes
                    or offset + nbytes > source.stat().st_size
                ):
                    raise ValueError(
                        f"ONNX initializer '{name}' has an invalid byte range"
                    )
            self._metadata[name] = OnnxTensorMetadata(
                dtype, shape, nbytes, source, offset
            )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._initializers)

    def metadata(self, name: str) -> OnnxTensorMetadata:
        return self._metadata[name]

    def load(
        self, device=None, names: Iterable[str] | None = None
    ) -> dict[str, wp.array]:
        """Upload selected initializers, mapping external storage without a copy."""
        onnx, numpy_helper = _require_onnx()
        device = wp.get_device(device)
        selected = self.names if names is None else tuple(names)
        unknown = set(selected) - self._initializers.keys()
        if unknown:
            raise KeyError(f"Unknown ONNX initializers: {sorted(unknown)}")
        mappings = {}
        arrays = {}
        for name in selected:
            value = self._initializers[name]
            metadata = self._metadata[name]
            if metadata.path is None:
                host = np.ascontiguousarray(numpy_helper.to_array(value))
            else:
                mapping = mappings.get(metadata.path)
                if mapping is None:
                    mapping = np.memmap(metadata.path, mode="r", dtype=np.uint8)
                    mappings[metadata.path] = mapping
                np_dtype = np.dtype(
                    onnx.helper.tensor_dtype_to_np_dtype(value.data_type)
                )
                host = np.ndarray(
                    metadata.shape,
                    dtype=np_dtype,
                    buffer=mapping,
                    offset=metadata.offset,
                )
            arrays[name] = onnx_array_to_warp(host, value.data_type, device)
        if device.is_cuda and mappings:
            wp.synchronize_stream(device)
        return arrays


__all__ = [
    "OnnxInitializerArchive",
    "OnnxTensorMetadata",
    "onnx_array_to_warp",
]
