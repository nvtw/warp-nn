# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free LoRA checkpoints in the standard safetensors format."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import math
import os
import struct
import tempfile

import numpy as np
import warp as wp

from .adapters import LoRAAdapterConfig


_FORMAT = "warp-nn-lora"
_VERSION = 1
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_MAX_METADATA_BYTES = 64 * 1024
_MAX_TEXT_LENGTH = 4096


@dataclass(frozen=True)
class LoRACheckpoint:
    """Loaded FP32 adapter tensors and validated checkpoint metadata."""

    tensors: dict[str, np.ndarray | wp.array]
    configs: dict[str, LoRAAdapterConfig]
    base_identifier: str | None
    caller_metadata: dict[str, str | int | float | bool | None]


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json(payload: bytes | str, description: str):
    try:
        return json.loads(
            payload,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid {description}") from exc


def _validate_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT_LENGTH:
        raise ValueError(f"{name} must be a non-empty bounded string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains a control character")
    return value


def _validate_tensor_name(name: object) -> str:
    name = _validate_text(name, "tensor name")
    if name == "__metadata__" or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"Unsafe tensor name {name!r}")
    return name


def _validate_metadata(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("LoRA metadata must be an object")
    if "configs" not in metadata:
        raise ValueError("LoRA metadata is missing configs")
    unknown = set(metadata) - {"configs", "base_identifier", "caller_metadata"}
    if unknown:
        raise ValueError(f"Unknown LoRA metadata keys: {sorted(unknown)}")

    raw_configs = metadata["configs"]
    if not isinstance(raw_configs, dict) or not raw_configs:
        raise ValueError("configs must be a non-empty object")
    if any(not isinstance(target, str) for target in raw_configs):
        raise ValueError("config target names must be strings")
    configs = {}
    for target in sorted(raw_configs):
        _validate_tensor_name(target)
        raw_config = raw_configs[target]
        if not isinstance(raw_config, dict) or set(raw_config) != {"rank", "alpha"}:
            raise ValueError(f"Invalid LoRA config for target {target!r}")
        rank = raw_config["rank"]
        alpha = raw_config["alpha"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"Invalid LoRA rank for target {target!r}")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError(f"Invalid LoRA alpha for target {target!r}")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"Invalid LoRA alpha for target {target!r}")
        configs[target] = LoRAAdapterConfig(rank=rank, alpha=alpha)

    base_identifier = metadata.get("base_identifier")
    if base_identifier is not None:
        base_identifier = _validate_text(base_identifier, "base_identifier")
    caller_metadata = metadata.get("caller_metadata", {})
    if not isinstance(caller_metadata, dict):
        raise ValueError("caller_metadata must be an object")
    checked_caller = {}
    for key, value in caller_metadata.items():
        _validate_text(key, "caller metadata key")
        if key.startswith("__"):
            raise ValueError(f"Unsafe caller metadata key {key!r}")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("caller metadata values must be JSON scalars")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("caller metadata floats must be finite")
        if isinstance(value, str):
            _validate_text(value, f"caller metadata value {key!r}")
        checked_caller[key] = value
    return {
        "configs": configs,
        "base_identifier": base_identifier,
        "caller_metadata": checked_caller,
    }


def _serialized_metadata(
    configs: Mapping[str, LoRAAdapterConfig],
    base_identifier: str | None,
    caller_metadata: Mapping[str, str | int | float | bool | None] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(configs, Mapping) or not configs:
        raise ValueError("configs must be a non-empty mapping")
    if any(not isinstance(target, str) for target in configs):
        raise ValueError("config target names must be strings")
    raw_configs = {}
    for target in sorted(configs):
        _validate_tensor_name(target)
        config = configs[target]
        if not isinstance(config, LoRAAdapterConfig):
            raise TypeError("configs must contain LoRAAdapterConfig values")
        alpha = config.rank if config.alpha is None else config.alpha
        raw_configs[target] = {"rank": config.rank, "alpha": float(alpha)}
    raw = {
        "configs": raw_configs,
        "base_identifier": base_identifier,
        "caller_metadata": dict(caller_metadata or {}),
    }
    return raw, _validate_metadata(raw)


def _expected_tensors(configs: Mapping[str, LoRAAdapterConfig]):
    return {
        f"{target}.lora_{side}.weight": (target, side)
        for target in configs
        for side in ("A", "B")
    }


def _fp32_numpy(value: np.ndarray | wp.array, name: str) -> np.ndarray:
    if isinstance(value, wp.array):
        if value.dtype != wp.float32:
            raise TypeError(f"adapter {name!r} must be FP32")
        value = value.numpy()
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise TypeError(f"adapter {name!r} must be an FP32 NumPy or Warp array")
    if (
        value.size == 0
        or value.ndim == 0
        or any(dimension <= 0 for dimension in value.shape)
    ):
        raise ValueError(f"adapter {name!r} must be a non-empty tensor")
    return np.ascontiguousarray(value, dtype="<f4")


def save_lora_safetensors(
    path: str | Path,
    adapters: Mapping[str, np.ndarray | wp.array],
    *,
    configs: Mapping[str, LoRAAdapterConfig],
    base_identifier: str | None = None,
    caller_metadata: Mapping[str, str | int | float | bool | None] | None = None,
) -> None:
    """Atomically save authoritative FP32 adapter state as safetensors."""
    if not isinstance(adapters, Mapping) or not adapters:
        raise ValueError("adapters must be a non-empty mapping")
    raw_metadata, checked_metadata = _serialized_metadata(
        configs, base_identifier, caller_metadata
    )
    expected = _expected_tensors(checked_metadata["configs"])
    if set(adapters) != set(expected):
        missing = sorted(set(expected) - set(adapters))
        extra = sorted(set(adapters) - set(expected))
        raise ValueError(
            f"Adapter tensors do not match configs; missing={missing}, extra={extra}"
        )
    metadata_json = json.dumps(
        raw_metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if len(metadata_json.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError("LoRA metadata is too large")

    tensors = {
        _validate_tensor_name(name): _fp32_numpy(value, name)
        for name, value in adapters.items()
    }
    for name, (target, side) in expected.items():
        tensor = tensors[name]
        rank = checked_metadata["configs"][target].rank
        if tensor.ndim != 2:
            raise ValueError(f"Adapter {name!r} must be 2-D")
        rank_dimension = tensor.shape[0] if side == "A" else tensor.shape[1]
        if rank_dimension != rank:
            raise ValueError(f"Adapter {name!r} does not match configured rank {rank}")

    header: dict[str, Any] = {
        "__metadata__": {
            "format": _FORMAT,
            "version": str(_VERSION),
            "warp_nn_lora": metadata_json,
        }
    }
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        end = offset + tensor.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    if len(encoded) > _MAX_HEADER_BYTES:
        raise ValueError("Safetensors header is too large")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(struct.pack("<Q", len(encoded)))
            stream.write(encoded)
            for name in sorted(tensors):
                stream.write(tensors[name].tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_lora_safetensors(path: str | Path, *, as_warp: bool = False) -> LoRACheckpoint:
    """Load validated adapter state into NumPy or CPU Warp arrays."""
    source = Path(path)
    file_size = source.stat().st_size
    with source.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Safetensors file has no complete header length")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length == 0 or header_length > _MAX_HEADER_BYTES:
            raise ValueError("Safetensors header length is invalid")
        if header_length > file_size - 8:
            raise ValueError("Safetensors file has a truncated header")
        header = _parse_json(stream.read(header_length), "safetensors JSON header")
        if not isinstance(header, dict):
            raise ValueError("Safetensors header must be an object")
        raw_metadata = header.get("__metadata__")
        if not isinstance(raw_metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in raw_metadata.items()
        ):
            raise ValueError("Safetensors metadata must contain string pairs")
        if raw_metadata.get("format") != _FORMAT or raw_metadata.get("version") != str(
            _VERSION
        ):
            raise ValueError("Unsupported LoRA checkpoint format or version")
        metadata_payload = raw_metadata.get("warp_nn_lora")
        if (
            metadata_payload is None
            or len(metadata_payload.encode("utf-8")) > _MAX_METADATA_BYTES
        ):
            raise ValueError("LoRA metadata is missing or too large")
        metadata = _validate_metadata(
            _parse_json(metadata_payload, "LoRA checkpoint metadata")
        )
        expected = _expected_tensors(metadata["configs"])
        tensor_names = set(header) - {"__metadata__"}
        if tensor_names != set(expected):
            missing = sorted(set(expected) - tensor_names)
            extra = sorted(tensor_names - set(expected))
            raise ValueError(
                f"Adapter tensors do not match configs; missing={missing}, extra={extra}"
            )

        data_start = 8 + header_length
        data_size = file_size - data_start
        entries = []
        for raw_name, entry in header.items():
            if raw_name == "__metadata__":
                continue
            name = _validate_tensor_name(raw_name)
            if not isinstance(entry, dict) or set(entry) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ValueError(f"Invalid metadata for adapter {name!r}")
            shape = entry["shape"]
            offsets = entry["data_offsets"]
            if entry["dtype"] != "F32":
                raise ValueError(f"Adapter {name!r} must use F32 storage")
            if (
                not isinstance(shape, list)
                or not shape
                or any(
                    isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension <= 0
                    for dimension in shape
                )
            ):
                raise ValueError(f"Invalid shape for adapter {name!r}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in offsets
                )
            ):
                raise ValueError(f"Invalid offsets for adapter {name!r}")
            begin, end = offsets
            elements = math.prod(shape)
            if (
                begin < 0
                or end < begin
                or end - begin != elements * 4
                or end > data_size
            ):
                raise ValueError(f"Invalid data range for adapter {name!r}")
            target, side = expected[name]
            rank = metadata["configs"][target].rank
            rank_dimension = (
                shape[0]
                if side == "A" and len(shape) == 2
                else (shape[1] if side == "B" and len(shape) == 2 else None)
            )
            if rank_dimension != rank:
                raise ValueError(
                    f"Adapter {name!r} must be 2-D and match configured rank {rank}"
                )
            entries.append((begin, end, name, tuple(shape)))
        if not entries:
            raise ValueError("LoRA checkpoint contains no adapter tensors")
        entries.sort()
        cursor = 0
        for begin, end, name, _ in entries:
            if begin != cursor:
                raise ValueError(
                    f"Overlapping or non-contiguous data for adapter {name!r}"
                )
            cursor = end
        if cursor != data_size:
            raise ValueError("Safetensors file has trailing or missing tensor data")

        output: dict[str, np.ndarray | wp.array] = {}
        for begin, end, name, shape in entries:
            stream.seek(data_start + begin)
            payload = stream.read(end - begin)
            if len(payload) != end - begin:
                raise ValueError(f"Truncated data for adapter {name!r}")
            array = np.frombuffer(payload, dtype="<f4").reshape(shape).copy()
            output[name] = (
                wp.array(array, dtype=wp.float32, device="cpu") if as_warp else array
            )
    return LoRACheckpoint(
        output,
        metadata["configs"],
        metadata["base_identifier"],
        metadata["caller_metadata"],
    )


def save_lora_collection(
    path: str | Path,
    collection,
    *,
    base_identifier: str | None = None,
    caller_metadata: Mapping[str, str | int | float | bool | None] | None = None,
) -> None:
    """Save one live collection's authoritative FP32 masters atomically."""
    save_lora_safetensors(
        path,
        collection.named_masters,
        configs=collection.configs,
        base_identifier=base_identifier,
        caller_metadata=caller_metadata,
    )


def restore_lora_collection(path: str | Path, collection) -> LoRACheckpoint:
    """Restore adapter weights into fixed buffers and reset AdamW state."""
    checkpoint = load_lora_safetensors(path)
    collection.load_fp32_state(checkpoint.tensors, checkpoint.configs)
    return checkpoint
