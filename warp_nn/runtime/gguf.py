# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free loading of unquantized single-file GGUF weights."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
import mmap
from pathlib import Path
import struct
from threading import Thread

import numpy as np
import warp as wp


_VALUE_FORMATS = {
    0: "<B",  # UINT8
    1: "<b",  # INT8
    2: "<H",  # UINT16
    3: "<h",  # INT16
    4: "<I",  # UINT32
    5: "<i",  # INT32
    6: "<f",  # FLOAT32
    7: "<B",  # BOOL
    10: "<Q",  # UINT64
    11: "<q",  # INT64
    12: "<d",  # FLOAT64
}
_STRING = 8
_ARRAY = 9
_TENSOR_TYPES = {
    0: (wp.float32, 4, "F32"),
    1: (wp.float16, 2, "F16"),
    30: (wp.bfloat16, 2, "BF16"),
}


@dataclass(frozen=True)
class GGUFTensorMetadata:
    """Location, type, and row-major shape of one GGUF tensor."""

    dtype: type
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    format: str


class _Reader:
    def __init__(self, stream, path: Path, size: int):
        self.stream = stream
        self.path = path
        self.size = size

    @property
    def position(self) -> int:
        return self.stream.tell()

    @property
    def remaining(self) -> int:
        return self.size - self.position

    def read(self, size: int, what: str) -> bytes:
        if size < 0 or size > self.remaining:
            raise ValueError(f"GGUF file '{self.path}' is truncated while reading {what}")
        value = self.stream.read(size)
        if len(value) != size:
            raise ValueError(f"GGUF file '{self.path}' is truncated while reading {what}")
        return value

    def unpack(self, format: str, what: str):
        return struct.unpack(format, self.read(struct.calcsize(format), what))[0]

    def string(self, what: str) -> str:
        length = self.unpack("<Q", f"{what} length")
        try:
            return self.read(length, what).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"GGUF file '{self.path}' has invalid UTF-8 in {what}") from exc

    def value(self, value_type: int, what: str, *, in_array: bool = False):
        if value_type in _VALUE_FORMATS:
            value = self.unpack(_VALUE_FORMATS[value_type], what)
            if value_type == 7:
                if value not in (0, 1):
                    raise ValueError(f"GGUF file '{self.path}' has an invalid boolean in {what}")
                return bool(value)
            return value
        if value_type == _STRING:
            return self.string(what)
        if value_type == _ARRAY and not in_array:
            element_type = self.unpack("<I", f"{what} element type")
            if element_type == _ARRAY or element_type not in (*_VALUE_FORMATS, _STRING):
                raise ValueError(f"GGUF file '{self.path}' has an unsupported array type {element_type} in {what}")
            count = self.unpack("<Q", f"{what} length")
            if count > self.remaining:
                raise ValueError(f"GGUF file '{self.path}' has an invalid array length in {what}")
            return tuple(self.value(element_type, f"{what}[{index}]", in_array=True) for index in range(count))
        raise ValueError(f"GGUF file '{self.path}' has an unsupported metadata type {value_type} in {what}")


def _release_mapping(resources, event: wp.Event | None) -> None:
    if event is not None:
        wp.synchronize_event(event)
    host_views, byte_views, mapping, stream = resources
    host_views.clear()
    byte_views.clear()
    mapping.close()
    stream.close()


class GGUFArchive:
    """Index and upload an unquantized, single-file GGUF archive."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"GGUF file not found: '{self.path}'")

        size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            reader = _Reader(stream, self.path, size)
            if reader.read(4, "magic") != b"GGUF":
                raise ValueError(f"File '{self.path}' does not have GGUF magic")
            self.version = reader.unpack("<I", "version")
            if self.version not in (2, 3):
                raise ValueError(f"GGUF file '{self.path}' has unsupported version {self.version}")
            tensor_count = reader.unpack("<Q", "tensor count")
            metadata_count = reader.unpack("<Q", "metadata count")
            if tensor_count > reader.remaining or metadata_count > reader.remaining:
                raise ValueError(f"GGUF file '{self.path}' has invalid header counts")

            self.metadata: dict[str, object] = {}
            for _ in range(metadata_count):
                key = reader.string("metadata key")
                if key in self.metadata:
                    raise ValueError(f"GGUF file '{self.path}' has duplicate metadata key '{key}'")
                value_type = reader.unpack("<I", f"metadata type for '{key}'")
                self.metadata[key] = reader.value(value_type, f"metadata '{key}'")

            descriptors = []
            seen = set()
            for _ in range(tensor_count):
                name = reader.string("tensor name")
                if name in seen:
                    raise ValueError(f"GGUF file '{self.path}' has duplicate tensor '{name}'")
                seen.add(name)
                dimensions = reader.unpack("<I", f"dimension count for tensor '{name}'")
                if dimensions > 4:
                    raise ValueError(f"GGUF tensor '{name}' has invalid dimension count {dimensions}")
                gguf_shape = tuple(reader.unpack("<Q", f"dimension for tensor '{name}'") for _ in range(dimensions))
                tensor_type = reader.unpack("<I", f"type for tensor '{name}'")
                relative_offset = reader.unpack("<Q", f"offset for tensor '{name}'")
                descriptors.append((name, gguf_shape, tensor_type, relative_offset))

            alignment = self.metadata.get("general.alignment", 32)
            if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
                raise ValueError(f"GGUF file '{self.path}' has invalid general.alignment")
            self.alignment = alignment
            self.data_offset = (reader.position + alignment - 1) // alignment * alignment
            if self.data_offset > size:
                raise ValueError(f"GGUF file '{self.path}' has no complete tensor data section")

        self._tensors: dict[str, GGUFTensorMetadata] = {}
        for name, gguf_shape, tensor_type, relative_offset in descriptors:
            try:
                dtype, itemsize, format = _TENSOR_TYPES[tensor_type]
            except KeyError as exc:
                raise ValueError(f"GGUF tensor '{name}' has unsupported type {tensor_type}") from exc
            if relative_offset % self.alignment:
                raise ValueError(f"GGUF tensor '{name}' has an unaligned data offset")
            shape = tuple(reversed(gguf_shape))
            elements = math.prod(shape)
            offset = self.data_offset + relative_offset
            nbytes = elements * itemsize
            if offset > size or nbytes > size - offset:
                raise ValueError(f"GGUF tensor '{name}' has an invalid data range")
            self._tensors[name] = GGUFTensorMetadata(dtype, shape, offset, nbytes, format)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tensors)

    def tensor(self, name: str) -> GGUFTensorMetadata:
        return self._tensors[name]

    def load(self, device=None, names: Iterable[str] | None = None) -> dict[str, wp.array]:
        """Upload selected tensors and release the file mapping after copies finish."""
        device = wp.get_device(device)
        selected = self.names if names is None else tuple(names)
        unknown = set(selected) - self._tensors.keys()
        if unknown:
            raise KeyError(f"Unknown GGUF tensors: {sorted(unknown)}")
        if not selected:
            return {}

        stream = self.path.open("rb")
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        output = {}
        byte_views = []
        host_views = []
        for name in selected:
            info = self._tensors[name]
            raw = np.ndarray((info.nbytes,), dtype=np.uint8, buffer=mapping, offset=info.offset)
            host = wp.array(
                ptr=raw.ctypes.data,
                dtype=info.dtype,
                shape=info.shape,
                capacity=info.nbytes,
                device="cpu",
            )
            byte_views.append(raw)
            host_views.append(host)
            output[name] = wp.clone(host, device=device)
        del host, raw

        event = wp.record_event() if device.is_cuda else None
        resources = (host_views, byte_views, mapping, stream)
        if event is None:
            _release_mapping(resources, None)
        else:
            Thread(target=_release_mapping, args=(resources, event), daemon=True).start()
        return output
