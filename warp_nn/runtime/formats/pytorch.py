# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Restricted, dependency-free reader for tensors in PyTorch ZIP archives."""

from __future__ import annotations

import io
import pickle
import sys
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_STORAGE_DTYPES = {
    "BoolStorage": np.dtype("?"),
    "ByteStorage": np.dtype("u1"),
    "CharStorage": np.dtype("i1"),
    "ShortStorage": np.dtype("i2"),
    "IntStorage": np.dtype("i4"),
    "LongStorage": np.dtype("i8"),
    "HalfStorage": np.dtype("f2"),
    "FloatStorage": np.dtype("f4"),
    "DoubleStorage": np.dtype("f8"),
    "BFloat16Storage": np.dtype("u2"),
}


@dataclass(frozen=True)
class _StorageType:
    dtype: np.dtype
    bfloat16: bool = False


@dataclass(frozen=True)
class _Storage:
    key: str
    dtype: np.dtype
    elements: int
    bfloat16: bool


@dataclass(frozen=True)
class _Tensor:
    storage: _Storage
    offset: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]


def _rebuild_tensor(storage, offset, shape, strides, *_):
    return _Tensor(
        storage, int(offset), tuple(map(int, shape)), tuple(map(int, strides))
    )


class _RestrictedUnpickler(pickle.Unpickler):
    """Accept only globals needed by plain tensor/state-dict metadata."""

    def find_class(self, module, name):
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(
                _STORAGE_DTYPES[name], bfloat16=name == "BFloat16Storage"
            )
        if module == "torch._utils" and name in (
            "_rebuild_tensor",
            "_rebuild_tensor_v2",
        ):
            return _rebuild_tensor
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        raise pickle.UnpicklingError(
            f"unsupported PyTorch pickle global {module}.{name}"
        )

    def persistent_load(self, identity):
        if (
            not isinstance(identity, tuple)
            or len(identity) < 5
            or identity[0] != "storage"
            or not isinstance(identity[1], _StorageType)
        ):
            raise pickle.UnpicklingError("unsupported PyTorch persistent object")
        storage_type = identity[1]
        return _Storage(
            str(identity[2]),
            storage_type.dtype,
            int(identity[4]),
            storage_type.bfloat16,
        )


def _decode_tensor(archive, prefix, byteorder, tensor):
    storage = tensor.storage
    raw = archive.read(f"{prefix}data/{storage.key}")
    expected = storage.elements * storage.dtype.itemsize
    if len(raw) != expected:
        raise ValueError(
            f"PyTorch storage {storage.key} has {len(raw)} bytes, expected {expected}"
        )
    dtype = storage.dtype.newbyteorder("<" if byteorder == "little" else ">")
    if len(tensor.shape) != len(tensor.strides) or min(tensor.shape, default=0) < 0:
        raise ValueError("invalid PyTorch tensor shape or strides")
    itemsize = dtype.itemsize
    try:
        view = np.ndarray(
            tensor.shape,
            dtype=dtype,
            buffer=raw,
            offset=tensor.offset * itemsize,
            strides=tuple(stride * itemsize for stride in tensor.strides),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("PyTorch tensor view exceeds its storage") from error
    if storage.bfloat16:
        words = np.asarray(view, dtype=np.uint16).astype(np.uint32) << 16
        return words.view(np.float32)
    return np.asarray(view, dtype=storage.dtype).copy()


def _resolve(value, decoder):
    if isinstance(value, _Tensor):
        return decoder(value)
    if isinstance(value, dict):
        return type(value)(
            (key, _resolve(item, decoder)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [_resolve(item, decoder) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve(item, decoder) for item in value)
    return value


def load_pytorch_zip(path: str | Path):
    """Load tensors from a modern ``torch.save`` ZIP without importing Torch.

    Only tensor metadata, primitive containers, and ordered dictionaries are
    accepted. Arbitrary pickle globals are rejected instead of executed.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith("/data.pkl")
        ]
        if len(metadata_names) != 1:
            raise ValueError(f"'{path}' is not a supported single-root PyTorch ZIP")
        metadata_name = metadata_names[0]
        prefix = metadata_name[: -len("data.pkl")]
        try:
            byteorder = archive.read(f"{prefix}byteorder").decode("ascii").strip()
        except KeyError:
            byteorder = sys.byteorder
        if byteorder not in ("little", "big"):
            raise ValueError(f"invalid PyTorch ZIP byte order {byteorder!r}")
        value = _RestrictedUnpickler(io.BytesIO(archive.read(metadata_name))).load()
        return _resolve(
            value, lambda tensor: _decode_tensor(archive, prefix, byteorder, tensor)
        )
