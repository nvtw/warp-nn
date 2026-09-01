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
_TRAINING_FORMAT = "warp-nn-lora-training-state"
_TRAINING_VERSION = 1
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


@dataclass(frozen=True)
class LoRATrainingCheckpoint:
    """Loaded exact-resume optimizer state and validated metadata."""

    tensors: dict[str, np.ndarray]
    configs: dict[str, LoRAAdapterConfig]
    optimizer: dict[str, Any]
    parameter_dtypes: dict[str, str]
    backend: str
    base_identifier: str | None


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


def _write_safetensors(
    path: str | Path,
    metadata: Mapping[str, str],
    tensors: Mapping[str, np.ndarray],
) -> None:
    """Write sorted F32/I32 tensors atomically with standard safetensors layout."""
    header: dict[str, Any] = {"__metadata__": dict(metadata)}
    normalized = {}
    offset = 0
    formats = {
        np.dtype("float32"): ("F32", "<f4"),
        np.dtype("int32"): ("I32", "<i4"),
    }
    for name in sorted(tensors):
        _validate_tensor_name(name)
        tensor = np.ascontiguousarray(tensors[name])
        if tensor.dtype not in formats:
            raise TypeError(f"Tensor {name!r} has unsupported checkpoint dtype")
        dtype, numpy_dtype = formats[tensor.dtype]
        tensor = tensor.astype(numpy_dtype, copy=False)
        if tensor.size == 0 or not tensor.shape:
            raise ValueError(f"Tensor {name!r} must be non-empty")
        normalized[name] = tensor
        end = offset + tensor.nbytes
        header[name] = {
            "dtype": dtype,
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
            for name in sorted(normalized):
                stream.write(normalized[name].tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _read_safetensors(
    path: str | Path,
    *,
    format: str,
    version: int,
    metadata_key: str,
) -> tuple[str, dict[str, tuple[str, np.ndarray]]]:
    """Read one strictly contiguous F32/I32 safetensors file."""
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
        if raw_metadata.get("format") != format or raw_metadata.get("version") != str(
            version
        ):
            raise ValueError("Unsupported checkpoint format or version")
        metadata_payload = raw_metadata.get(metadata_key)
        if (
            metadata_payload is None
            or len(metadata_payload.encode("utf-8")) > _MAX_METADATA_BYTES
        ):
            raise ValueError("Checkpoint metadata is missing or too large")

        data_start = 8 + header_length
        data_size = file_size - data_start
        entries = []
        formats = {"F32": ("<f4", 4), "I32": ("<i4", 4)}
        for raw_name, entry in header.items():
            if raw_name == "__metadata__":
                continue
            name = _validate_tensor_name(raw_name)
            if not isinstance(entry, dict) or set(entry) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ValueError(f"Invalid metadata for tensor {name!r}")
            dtype, shape, offsets = (
                entry["dtype"],
                entry["shape"],
                entry["data_offsets"],
            )
            if dtype not in formats:
                raise ValueError(f"Unsupported dtype for tensor {name!r}")
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
                raise ValueError(f"Invalid shape for tensor {name!r}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in offsets
                )
            ):
                raise ValueError(f"Invalid offsets for tensor {name!r}")
            begin, end = offsets
            itemsize = formats[dtype][1]
            if (
                begin < 0
                or end - begin != math.prod(shape) * itemsize
                or end > data_size
            ):
                raise ValueError(f"Invalid data range for tensor {name!r}")
            entries.append((begin, end, name, tuple(shape), dtype))
        if not entries:
            raise ValueError("Checkpoint contains no tensors")
        entries.sort()
        cursor = 0
        for begin, end, name, _, _ in entries:
            if begin != cursor:
                raise ValueError(
                    f"Overlapping or non-contiguous data for tensor {name!r}"
                )
            cursor = end
        if cursor != data_size:
            raise ValueError("Safetensors file has trailing or missing tensor data")
        tensors = {}
        for begin, end, name, shape, dtype in entries:
            stream.seek(data_start + begin)
            payload = stream.read(end - begin)
            if len(payload) != end - begin:
                raise ValueError(f"Truncated data for tensor {name!r}")
            tensors[name] = (
                dtype,
                np.frombuffer(payload, dtype=formats[dtype][0]).reshape(shape).copy(),
            )
    return metadata_payload, tensors


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

    _write_safetensors(
        path,
        {"format": _FORMAT, "version": str(_VERSION), "warp_nn_lora": metadata_json},
        tensors,
    )


def load_lora_safetensors(path: str | Path, *, as_warp: bool = False) -> LoRACheckpoint:
    """Load validated adapter state into NumPy or CPU Warp arrays."""
    metadata_payload, stored = _read_safetensors(
        path, format=_FORMAT, version=_VERSION, metadata_key="warp_nn_lora"
    )
    metadata = _validate_metadata(
        _parse_json(metadata_payload, "LoRA checkpoint metadata")
    )
    expected = _expected_tensors(metadata["configs"])
    if set(stored) != set(expected):
        missing = sorted(set(expected) - set(stored))
        extra = sorted(set(stored) - set(expected))
        raise ValueError(
            f"Adapter tensors do not match configs; missing={missing}, extra={extra}"
        )
    output: dict[str, np.ndarray | wp.array] = {}
    for name in sorted(stored):
        dtype, array = stored[name]
        target, side = expected[name]
        rank = metadata["configs"][target].rank
        rank_dimension = (
            array.shape[0]
            if side == "A" and array.ndim == 2
            else (array.shape[1] if side == "B" and array.ndim == 2 else None)
        )
        if dtype != "F32" or rank_dimension != rank:
            raise ValueError(
                f"Adapter {name!r} must be 2-D and match configured rank {rank} with F32 storage"
            )
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


_OPTIMIZER_FIELDS = (
    "beta1",
    "beta2",
    "epsilon",
    "gradient_multiplier",
    "learning_rate",
    "loss_scale",
    "max_grad_norm",
    "min_learning_rate_ratio",
    "normalize_by_valid_tokens",
    "total_steps",
    "warmup_steps",
    "weight_decay",
)


def _parameter_dtype_name(value: wp.array) -> str:
    names = {wp.float16: "F16", wp.bfloat16: "BF16", wp.float32: "F32"}
    if value.dtype not in names:
        raise TypeError("LoRA parameters must use FP16, BF16, or FP32")
    return names[value.dtype]


def _optimizer_fingerprint(optimizer) -> dict[str, Any]:
    return {name: getattr(optimizer, name) for name in _OPTIMIZER_FIELDS}


def _validate_optimizer_fingerprint(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(_OPTIMIZER_FIELDS):
        raise ValueError("Invalid optimizer fingerprint")
    checked = {}
    float_fields = set(_OPTIMIZER_FIELDS) - {
        "normalize_by_valid_tokens",
        "total_steps",
        "warmup_steps",
        "max_grad_norm",
    }
    for name in sorted(float_fields):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"Invalid optimizer fingerprint value {name!r}")
        item = float(item)
        if not math.isfinite(item):
            raise ValueError(f"Invalid optimizer fingerprint value {name!r}")
        checked[name] = item
    for name in ("total_steps", "warmup_steps"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"Invalid optimizer fingerprint value {name!r}")
        checked[name] = item
    normalized = value["normalize_by_valid_tokens"]
    if not isinstance(normalized, bool):
        raise ValueError("Invalid optimizer gradient normalization")
    checked["normalize_by_valid_tokens"] = normalized
    maximum = value["max_grad_norm"]
    if maximum is not None:
        if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
            raise ValueError("Invalid optimizer max_grad_norm")
        maximum = float(maximum)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("Invalid optimizer max_grad_norm")
    checked["max_grad_norm"] = maximum
    return checked


def _training_tensor_names(parameter_names) -> set[str]:
    names = {"optimizer.step_count"}
    for name in parameter_names:
        names.update(
            (f"{name}.master", f"{name}.first_moment", f"{name}.second_moment")
        )
    return names


def save_lora_training_state(
    path: str | Path,
    collection,
    *,
    base_identifier: str | None = None,
) -> None:
    """Atomically save full LoRA/AdamW state for bitwise continuation."""
    if base_identifier is not None:
        _validate_text(base_identifier, "base_identifier")
    optimizer = collection.optimizer
    raw_lora, _ = _serialized_metadata(collection.configs, base_identifier, None)
    metadata = {
        "backend": "cublas" if collection.cublas is not None else "warp",
        "base_identifier": base_identifier,
        "configs": raw_lora["configs"],
        "optimizer": _optimizer_fingerprint(optimizer),
        "parameter_dtypes": {
            name: _parameter_dtype_name(parameter)
            for name, parameter in sorted(collection.named_parameters.items())
        },
    }
    metadata_json = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if len(metadata_json.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError("LoRA training-state metadata is too large")
    tensors = {"optimizer.step_count": optimizer.step_count.numpy()}
    # Optimizer slots and named masters deliberately share insertion order.
    for index, (name, master) in enumerate(collection.named_masters.items()):
        tensors[f"{name}.master"] = master.numpy()
        tensors[f"{name}.first_moment"] = (
            optimizer.first_moments[index].numpy().reshape(master.shape)
        )
        tensors[f"{name}.second_moment"] = (
            optimizer.second_moments[index].numpy().reshape(master.shape)
        )
    _write_safetensors(
        path,
        {
            "format": _TRAINING_FORMAT,
            "version": str(_TRAINING_VERSION),
            "warp_nn_lora_training": metadata_json,
        },
        tensors,
    )


def load_lora_training_state(path: str | Path) -> LoRATrainingCheckpoint:
    """Load and structurally validate an exact-resume LoRA training state."""
    payload, stored = _read_safetensors(
        path,
        format=_TRAINING_FORMAT,
        version=_TRAINING_VERSION,
        metadata_key="warp_nn_lora_training",
    )
    metadata = _parse_json(payload, "LoRA training-state metadata")
    if not isinstance(metadata, dict) or set(metadata) != {
        "backend",
        "base_identifier",
        "configs",
        "optimizer",
        "parameter_dtypes",
    }:
        raise ValueError("Invalid LoRA training-state metadata")
    lora_metadata = _validate_metadata(
        {
            "configs": metadata["configs"],
            "base_identifier": metadata["base_identifier"],
            "caller_metadata": {},
        }
    )
    configs = lora_metadata["configs"]
    optimizer = _validate_optimizer_fingerprint(metadata["optimizer"])
    base_identifier = lora_metadata["base_identifier"]
    backend = metadata["backend"]
    if backend not in {"warp", "cublas"}:
        raise ValueError("Invalid training-state backend")
    parameter_dtypes = metadata["parameter_dtypes"]
    if not isinstance(parameter_dtypes, dict) or not parameter_dtypes:
        raise ValueError("parameter_dtypes must be a non-empty object")
    for name, dtype in parameter_dtypes.items():
        _validate_tensor_name(name)
        if dtype not in {"F16", "BF16", "F32"}:
            raise ValueError(f"Invalid parameter dtype for {name!r}")
    expected = _training_tensor_names(parameter_dtypes)
    if set(stored) != expected:
        missing = sorted(expected - set(stored))
        extra = sorted(set(stored) - expected)
        raise ValueError(
            "Training-state tensors do not match parameters; "
            f"missing={missing}, extra={extra}"
        )
    tensors = {}
    for name, (dtype, tensor) in stored.items():
        required_dtype = "I32" if name == "optimizer.step_count" else "F32"
        if dtype != required_dtype:
            raise ValueError(f"Invalid dtype for training-state tensor {name!r}")
        tensors[name] = tensor
    step = tensors["optimizer.step_count"]
    if step.shape != (1,) or step[0] < 0:
        raise ValueError("optimizer step_count must be one non-negative INT32 value")
    return LoRATrainingCheckpoint(
        tensors,
        configs,
        optimizer,
        dict(sorted(parameter_dtypes.items())),
        backend,
        base_identifier,
    )


def restore_lora_training_state(
    path: str | Path,
    collection,
    *,
    base_identifier: str | None = None,
) -> LoRATrainingCheckpoint:
    """Restore full optimizer state into existing fixed collection buffers."""
    checkpoint = load_lora_training_state(path)
    if base_identifier is not None and checkpoint.base_identifier != base_identifier:
        raise ValueError("training-state base identifier does not match")
    collection.load_training_state(
        checkpoint.tensors,
        checkpoint.configs,
        checkpoint.optimizer,
        checkpoint.parameter_dtypes,
        checkpoint.backend,
    )
    return checkpoint
