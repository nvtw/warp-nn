# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free loading of single-file and sharded safetensors weights."""

from __future__ import annotations

from typing import Iterable

import json
import mmap
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

import numpy as np
import warp as wp


_DTYPES = {
    "BOOL": (wp.bool, 1),
    "I8": (wp.int8, 1),
    "U8": (wp.uint8, 1),
    "I16": (wp.int16, 2),
    "U16": (wp.uint16, 2),
    "I32": (wp.int32, 4),
    "U32": (wp.uint32, 4),
    "I64": (wp.int64, 8),
    "U64": (wp.uint64, 8),
    "F16": (wp.float16, 2),
    "BF16": (wp.bfloat16, 2),
    "F32": (wp.float32, 4),
    "F64": (wp.float64, 8),
}


@dataclass(frozen=True)
class SafeTensorMetadata:
    """Location, type, and shape of one tensor in a safetensors shard."""

    shard: Path
    dtype: type
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def _read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Safetensors file '{path}' has no complete header length")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length > path.stat().st_size - 8:
            raise ValueError(f"Safetensors file '{path}' has an invalid header length")
        try:
            header = json.loads(stream.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Safetensors file '{path}' has an invalid JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors file '{path}' has a non-object header")
    return 8 + header_length, header


def _release_mappings(resources, event: wp.Event | None) -> None:
    if event is not None:
        wp.synchronize_event(event)
    host_views, byte_views, mappings, streams = resources
    host_views.clear()
    byte_views.clear()
    for mapping in mappings:
        mapping.close()
    for stream in streams:
        stream.close()


class SafeTensorArchive:
    """Index and upload a Hugging Face safetensors checkpoint without safetensors."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if path.is_dir():
            index_path = path / "model.safetensors.index.json"
            single_path = path / "model.safetensors"
            if index_path.is_file():
                try:
                    index = json.loads(index_path.read_text(encoding="utf-8"))
                    weight_map = index["weight_map"]
                except (OSError, KeyError, json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"Invalid safetensors index '{index_path}'") from exc
            elif single_path.is_file():
                _, header = _read_header(single_path)
                weight_map = {name: single_path.name for name in header if name != "__metadata__"}
            else:
                raise FileNotFoundError(f"No safetensors checkpoint found in '{path}'")
            base_dir = path
        else:
            base_dir = path.parent
            _, header = _read_header(path)
            weight_map = {name: path.name for name in header if name != "__metadata__"}

        self._metadata: dict[str, SafeTensorMetadata] = {}
        base_dir = base_dir.resolve()
        by_shard: dict[Path, list[str]] = defaultdict(list)
        for name, filename in weight_map.items():
            shard = (base_dir / filename).resolve()
            if shard.parent != base_dir:
                raise ValueError(f"Safetensors shard '{filename}' is outside '{base_dir}'")
            by_shard[shard].append(name)

        for shard, names in by_shard.items():
            data_start, header = _read_header(shard)
            data_size = shard.stat().st_size - data_start
            for name in names:
                try:
                    entry = header[name]
                    dtype, itemsize = _DTYPES[entry["dtype"]]
                    shape = tuple(int(dim) for dim in entry["shape"])
                    begin, end = (int(offset) for offset in entry["data_offsets"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid metadata for safetensor '{name}' in '{shard}'") from exc
                elements = int(np.prod(shape, dtype=np.int64))
                if begin < 0 or end < begin or end > data_size or end - begin != elements * itemsize:
                    raise ValueError(f"Invalid data range for safetensor '{name}' in '{shard}'")
                self._metadata[name] = SafeTensorMetadata(
                    shard=shard,
                    dtype=dtype,
                    shape=shape,
                    offset=data_start + begin,
                    nbytes=end - begin,
                )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._metadata)

    def metadata(self, name: str) -> SafeTensorMetadata:
        return self._metadata[name]

    def load(self, device=None, names: Iterable[str] | None = None) -> dict[str, wp.array]:
        """Upload selected weights and release file mappings after the copies finish."""
        device = wp.get_device(device)
        selected = self.names if names is None else tuple(names)
        unknown = set(selected) - self._metadata.keys()
        if unknown:
            raise KeyError(f"Unknown safetensors: {sorted(unknown)}")

        by_shard: dict[Path, list[str]] = defaultdict(list)
        for name in selected:
            by_shard[self._metadata[name].shard].append(name)

        output = {}
        streams = []
        mappings = []
        byte_views = []
        host_views = []
        for shard, shard_names in by_shard.items():
            stream = shard.open("rb")
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            streams.append(stream)
            mappings.append(mapping)
            for name in shard_names:
                info = self._metadata[name]
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

        event = wp.record_event() if device.is_cuda and output else None
        resources = (host_views, byte_views, mappings, streams)
        if event is None:
            _release_mappings(resources, None)
        else:
            Thread(target=_release_mappings, args=(resources, event), daemon=True).start()
        return output
