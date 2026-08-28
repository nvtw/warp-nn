# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free loading of unquantized single-file GGUF weights."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import mmap
from pathlib import Path
import re
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


def find_gguf_files(path: str | Path) -> tuple[Path, ...]:
    """Resolve a model file or directory to one complete GGUF shard set."""
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    if path.is_dir():
        first_files = sorted(directory.glob("*-00001-of-*.gguf"))
        if not first_files:
            first_files = [
                item
                for item in sorted(directory.glob("*.gguf"))
                if "mmproj" not in item.name.lower()
            ]
        if len(first_files) != 1:
            raise FileNotFoundError(
                f"Expected one GGUF model in '{directory}', found {len(first_files)}"
            )
        path = first_files[0]
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
    if match is None:
        return (path,)
    count = int(match.group(3))
    return tuple(
        directory / f"{match.group(1)}-{index:05d}-of-{count:05d}.gguf"
        for index in range(1, count + 1)
    )


def gguf_tokenizer_data(metadata: Mapping[str, object]) -> dict:
    """Translate an embedded GGUF GPT-2 BPE tokenizer to Qwen3Tokenizer data."""
    if metadata.get("tokenizer.ggml.model") != "gpt2":
        raise ValueError("GGUF checkpoint requires an embedded GPT-2 BPE tokenizer")
    tokens = metadata["tokenizer.ggml.tokens"]
    token_types = metadata["tokenizer.ggml.token_type"]
    merges = [merge.split(" ", 1) for merge in metadata["tokenizer.ggml.merges"]]
    added = [
        {"id": token_id, "content": token, "special": token_type == 3}
        for token_id, (token, token_type) in enumerate(zip(tokens, token_types))
        if token_type != 1
    ]
    end_ids = [
        int(metadata[key])
        for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id")
        if key in metadata
    ]
    return {
        "normalizer": None,
        "added_tokens": added,
        "model": {
            "type": "BPE",
            "vocab": dict(zip(tokens, range(len(tokens)))),
            "merges": merges,
        },
        "generation_config": {
            "eos_token_id": list(dict.fromkeys(end_ids)),
            "pad_token_id": int(metadata["tokenizer.ggml.padding_token_id"]),
        },
        "chat_template": metadata.get("tokenizer.chat_template", ""),
        "_pretokenizer": "o200k",
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
            raise ValueError(
                f"GGUF file '{self.path}' is truncated while reading {what}"
            )
        value = self.stream.read(size)
        if len(value) != size:
            raise ValueError(
                f"GGUF file '{self.path}' is truncated while reading {what}"
            )
        return value

    def unpack(self, format: str, what: str):
        return struct.unpack(format, self.read(struct.calcsize(format), what))[0]

    def string(self, what: str) -> str:
        length = self.unpack("<Q", f"{what} length")
        try:
            return self.read(length, what).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"GGUF file '{self.path}' has invalid UTF-8 in {what}"
            ) from exc

    def value(self, value_type: int, what: str, *, in_array: bool = False):
        if value_type in _VALUE_FORMATS:
            value = self.unpack(_VALUE_FORMATS[value_type], what)
            if value_type == 7:
                if value not in (0, 1):
                    raise ValueError(
                        f"GGUF file '{self.path}' has an invalid boolean in {what}"
                    )
                return bool(value)
            return value
        if value_type == _STRING:
            return self.string(what)
        if value_type == _ARRAY and not in_array:
            element_type = self.unpack("<I", f"{what} element type")
            if element_type == _ARRAY or element_type not in (*_VALUE_FORMATS, _STRING):
                raise ValueError(
                    f"GGUF file '{self.path}' has an unsupported array type {element_type} in {what}"
                )
            count = self.unpack("<Q", f"{what} length")
            if count > self.remaining:
                raise ValueError(
                    f"GGUF file '{self.path}' has an invalid array length in {what}"
                )
            return tuple(
                self.value(element_type, f"{what}[{index}]", in_array=True)
                for index in range(count)
            )
        raise ValueError(
            f"GGUF file '{self.path}' has an unsupported metadata type {value_type} in {what}"
        )


def _release_mapping(resources, event: wp.Event | None) -> None:
    if event is not None:
        wp.synchronize_event(event)
    host_views, byte_views, mapping, stream = resources
    host_views.clear()
    byte_views.clear()
    mapping.close()
    stream.close()


class GGUFArchive:
    """Index and upload an unquantized GGUF archive, including split files."""

    def __init__(self, path: str | Path | Sequence[str | Path]):
        if not isinstance(path, (str, Path)):
            paths = tuple(path)
            if not paths:
                raise ValueError("A split GGUF archive needs at least one file")
            archives = tuple(GGUFArchive(item) for item in paths)
            expected = len(archives)
            if expected > 1:
                for index, archive in enumerate(archives):
                    if (
                        archive.metadata.get("split.count") != expected
                        or archive.metadata.get("split.no") != index
                    ):
                        raise ValueError(
                            "GGUF shards are missing, duplicated, or out of order"
                        )
            self.path = archives[0].path
            self.paths = tuple(archive.path for archive in archives)
            self.version = archives[0].version
            self.alignment = archives[0].alignment
            self.data_offset = archives[0].data_offset
            self.metadata = dict(archives[0].metadata)
            self._archives = archives
            self._tensors = {}
            self._tensor_archives = {}
            for archive in archives:
                for name in archive.names:
                    if name in self._tensors:
                        raise ValueError(
                            f"Split GGUF archive has duplicate tensor '{name}'"
                        )
                    self._tensors[name] = archive.tensor(name)
                    self._tensor_archives[name] = archive
            total = self.metadata.get("split.tensors.count")
            if total is not None and total != len(self._tensors):
                raise ValueError(
                    f"Split GGUF archive has {len(self._tensors)} of {total} tensors"
                )
            return

        self.path = Path(path).resolve()
        self.paths = (self.path,)
        self._archives = None
        if not self.path.is_file():
            raise FileNotFoundError(f"GGUF file not found: '{self.path}'")

        size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            reader = _Reader(stream, self.path, size)
            if reader.read(4, "magic") != b"GGUF":
                raise ValueError(f"File '{self.path}' does not have GGUF magic")
            self.version = reader.unpack("<I", "version")
            if self.version not in (2, 3):
                raise ValueError(
                    f"GGUF file '{self.path}' has unsupported version {self.version}"
                )
            tensor_count = reader.unpack("<Q", "tensor count")
            metadata_count = reader.unpack("<Q", "metadata count")
            if tensor_count > reader.remaining or metadata_count > reader.remaining:
                raise ValueError(f"GGUF file '{self.path}' has invalid header counts")

            self.metadata: dict[str, object] = {}
            for _ in range(metadata_count):
                key = reader.string("metadata key")
                if key in self.metadata:
                    raise ValueError(
                        f"GGUF file '{self.path}' has duplicate metadata key '{key}'"
                    )
                value_type = reader.unpack("<I", f"metadata type for '{key}'")
                self.metadata[key] = reader.value(value_type, f"metadata '{key}'")

            descriptors = []
            seen = set()
            for _ in range(tensor_count):
                name = reader.string("tensor name")
                if name in seen:
                    raise ValueError(
                        f"GGUF file '{self.path}' has duplicate tensor '{name}'"
                    )
                seen.add(name)
                dimensions = reader.unpack("<I", f"dimension count for tensor '{name}'")
                if dimensions > 4:
                    raise ValueError(
                        f"GGUF tensor '{name}' has invalid dimension count {dimensions}"
                    )
                gguf_shape = tuple(
                    reader.unpack("<Q", f"dimension for tensor '{name}'")
                    for _ in range(dimensions)
                )
                tensor_type = reader.unpack("<I", f"type for tensor '{name}'")
                relative_offset = reader.unpack("<Q", f"offset for tensor '{name}'")
                descriptors.append((name, gguf_shape, tensor_type, relative_offset))

            alignment = self.metadata.get("general.alignment", 32)
            if (
                type(alignment) is not int
                or alignment <= 0
                or alignment & (alignment - 1)
            ):
                raise ValueError(
                    f"GGUF file '{self.path}' has invalid general.alignment"
                )
            self.alignment = alignment
            self.data_offset = (
                (reader.position + alignment - 1) // alignment * alignment
            )
            if self.data_offset > size:
                raise ValueError(
                    f"GGUF file '{self.path}' has no complete tensor data section"
                )

        self._tensors: dict[str, GGUFTensorMetadata] = {}
        for name, gguf_shape, tensor_type, relative_offset in descriptors:
            try:
                dtype, itemsize, format = _TENSOR_TYPES[tensor_type]
            except KeyError as exc:
                raise ValueError(
                    f"GGUF tensor '{name}' has unsupported type {tensor_type}"
                ) from exc
            if relative_offset % self.alignment:
                raise ValueError(f"GGUF tensor '{name}' has an unaligned data offset")
            shape = tuple(reversed(gguf_shape))
            elements = math.prod(shape)
            offset = self.data_offset + relative_offset
            nbytes = elements * itemsize
            if offset > size or nbytes > size - offset:
                raise ValueError(f"GGUF tensor '{name}' has an invalid data range")
            self._tensors[name] = GGUFTensorMetadata(
                dtype, shape, offset, nbytes, format
            )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tensors)

    def tensor(self, name: str) -> GGUFTensorMetadata:
        return self._tensors[name]

    def load(
        self, device=None, names: Iterable[str] | None = None
    ) -> dict[str, wp.array]:
        """Upload selected tensors and release the file mapping after copies finish."""
        device = wp.get_device(device)
        selected = self.names if names is None else tuple(names)
        unknown = set(selected) - self._tensors.keys()
        if unknown:
            raise KeyError(f"Unknown GGUF tensors: {sorted(unknown)}")
        if not selected:
            return {}

        if self._archives is not None:
            by_archive = {}
            for name in selected:
                by_archive.setdefault(self._tensor_archives[name], []).append(name)
            output = {}
            for archive, shard_names in by_archive.items():
                output.update(archive.load(device, shard_names))
            return output

        stream = self.path.open("rb")
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        output = {}
        byte_views = []
        host_views = []
        for name in selected:
            info = self._tensors[name]
            raw = np.ndarray(
                (info.nbytes,), dtype=np.uint8, buffer=mapping, offset=info.offset
            )
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
            Thread(
                target=_release_mapping, args=(resources, event), daemon=True
            ).start()
        return output


class MappedGGUFArchive:
    """Expose GGUF tensors under runtime-native checkpoint names."""

    def __init__(self, archive: GGUFArchive, names: Mapping[str, str]):
        self.archive = archive
        self._names = dict(names)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._names)

    def metadata(self, name: str):
        return self.archive.tensor(self._names[name])

    def load(self, device=None, names=None) -> dict[str, wp.array]:
        selected = self.names if names is None else tuple(names)
        loaded = self.archive.load(device, [self._names[name] for name in selected])
        return {name: loaded[self._names[name]] for name in selected}
