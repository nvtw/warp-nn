# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct

import numpy as np
import pytest

import warp as wp

from warp_nn.runtime.formats.safetensors import SafeTensorArchive
from warp_nn.training.adapters import LoRAAdapterConfig
from warp_nn.training.checkpoint import load_lora_safetensors, save_lora_safetensors


def _metadata(configs=None, caller_metadata=None):
    configs = configs or {"layers.0.q_proj": {"rank": 1, "alpha": 1.0}}
    content = {"configs": configs}
    if caller_metadata is not None:
        content["caller_metadata"] = caller_metadata
    return {
        "format": "warp-nn-lora",
        "version": "1",
        "warp_nn_lora": json.dumps(content, separators=(",", ":")),
    }


def _write_raw(path, header_text, payload=b""):
    encoded = header_text.encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _configs():
    return {
        "layers.0.q_proj": LoRAAdapterConfig(rank=2, alpha=4.0),
        "layers.0.v_proj": LoRAAdapterConfig(rank=1, alpha=0.5),
    }


def _adapters():
    return {
        "layers.0.q_proj.lora_A.weight": wp.array(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            dtype=wp.float32,
            device="cpu",
        ),
        "layers.0.q_proj.lora_B.weight": np.arange(8, dtype=np.float32).reshape(4, 2),
        "layers.0.v_proj.lora_A.weight": np.array(
            [[1.5, -2.0, 0.25]], dtype=np.float32
        ),
        "layers.0.v_proj.lora_B.weight": np.arange(4, dtype=np.float32).reshape(4, 1),
    }


def test_lora_safetensors_heterogeneous_deterministic_round_trip(tmp_path):
    adapters = _adapters()
    first = tmp_path / "adapter.safetensors"
    second = tmp_path / "adapter-copy.safetensors"
    options = {
        "configs": _configs(),
        "base_identifier": "org/base-model",
        "caller_metadata": {"step": 17, "method": "adamw"},
    }

    save_lora_safetensors(first, adapters, **options)
    save_lora_safetensors(second, adapters, **options)
    assert first.read_bytes() == second.read_bytes()
    assert not list(tmp_path.glob("*.tmp"))

    checkpoint = load_lora_safetensors(first)
    assert tuple(checkpoint.tensors) == tuple(sorted(adapters))
    for name, expected in adapters.items():
        expected = expected.numpy() if isinstance(expected, wp.array) else expected
        np.testing.assert_array_equal(checkpoint.tensors[name], expected)
    assert checkpoint.configs == {
        "layers.0.q_proj": LoRAAdapterConfig(rank=2, alpha=4.0),
        "layers.0.v_proj": LoRAAdapterConfig(rank=1, alpha=0.5),
    }
    assert checkpoint.base_identifier == "org/base-model"
    assert checkpoint.caller_metadata == {"method": "adamw", "step": 17}

    cpu_checkpoint = load_lora_safetensors(first, as_warp=True)
    assert all(
        isinstance(tensor, wp.array)
        and tensor.device.is_cpu
        and tensor.dtype == wp.float32
        for tensor in cpu_checkpoint.tensors.values()
    )
    archive = SafeTensorArchive(first)
    assert archive.names == tuple(checkpoint.tensors)
    archive_values = archive.load("cpu")
    for name, expected in checkpoint.tensors.items():
        np.testing.assert_array_equal(archive_values[name].numpy(), expected)


def test_lora_safetensors_rejects_missing_extra_and_rank_mismatch_on_save(tmp_path):
    adapters = _adapters()
    missing = dict(adapters)
    missing.pop("layers.0.v_proj.lora_B.weight")
    with pytest.raises(ValueError, match="missing="):
        save_lora_safetensors(
            tmp_path / "missing.safetensors", missing, configs=_configs()
        )

    extra = dict(adapters)
    extra["unconfigured.lora_A.weight"] = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="extra="):
        save_lora_safetensors(tmp_path / "extra.safetensors", extra, configs=_configs())

    mismatched = dict(adapters)
    mismatched["layers.0.q_proj.lora_A.weight"] = np.ones((1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="configured rank 2"):
        save_lora_safetensors(
            tmp_path / "mismatch.safetensors", mismatched, configs=_configs()
        )


def test_lora_safetensors_rejects_malformed_truncated_or_duplicate_files(tmp_path):
    malformed = tmp_path / "malformed.safetensors"
    _write_raw(malformed, "{")
    with pytest.raises(ValueError, match="Invalid safetensors JSON header"):
        load_lora_safetensors(malformed)

    truncated = tmp_path / "truncated.safetensors"
    save_lora_safetensors(truncated, _adapters(), configs=_configs())
    truncated.write_bytes(truncated.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Invalid data range|trailing or missing"):
        load_lora_safetensors(truncated)

    duplicate = tmp_path / "duplicate.safetensors"
    tensor_a = json.dumps(
        {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]},
        separators=(",", ":"),
    )
    tensor_b = json.dumps(
        {"dtype": "F32", "shape": [1, 1], "data_offsets": [4, 8]},
        separators=(",", ":"),
    )
    metadata = json.dumps(_metadata(), separators=(",", ":"))
    prefix = "layers.0.q_proj"
    _write_raw(
        duplicate,
        f'{{"__metadata__":{metadata},"{prefix}.lora_A.weight":{tensor_a},'
        f'"{prefix}.lora_A.weight":{tensor_a},"{prefix}.lora_B.weight":{tensor_b}}}',
        np.array([1.0, 2.0], dtype=np.float32).tobytes(),
    )
    with pytest.raises(ValueError, match="Invalid safetensors JSON header"):
        load_lora_safetensors(duplicate)

    missing = tmp_path / "load-missing.safetensors"
    missing_header = {
        "__metadata__": _metadata(),
        f"{prefix}.lora_A.weight": {
            "dtype": "F32",
            "shape": [1, 1],
            "data_offsets": [0, 4],
        },
    }
    _write_raw(
        missing,
        json.dumps(missing_header, separators=(",", ":")),
        np.array([1.0], dtype=np.float32).tobytes(),
    )
    with pytest.raises(ValueError, match="missing="):
        load_lora_safetensors(missing)

    extra = tmp_path / "load-extra.safetensors"
    extra_header = dict(missing_header)
    extra_header[f"{prefix}.lora_B.weight"] = {
        "dtype": "F32",
        "shape": [1, 1],
        "data_offsets": [4, 8],
    }
    extra_header["extra"] = {
        "dtype": "F32",
        "shape": [1, 1],
        "data_offsets": [8, 12],
    }
    _write_raw(
        extra,
        json.dumps(extra_header, separators=(",", ":")),
        np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes(),
    )
    with pytest.raises(ValueError, match="extra="):
        load_lora_safetensors(extra)


def test_lora_safetensors_rejects_unsafe_metadata_names_and_load_shape(tmp_path):
    with pytest.raises(ValueError, match="JSON scalars"):
        save_lora_safetensors(
            tmp_path / "unsafe-save.safetensors",
            _adapters(),
            configs=_configs(),
            caller_metadata={"nested": {"not": "allowed"}},
        )
    with pytest.raises(ValueError, match="Unsafe tensor name"):
        save_lora_safetensors(
            tmp_path / "unsafe-name.safetensors",
            {
                "../adapter.lora_A.weight": np.ones((1, 1), dtype=np.float32),
                "../adapter.lora_B.weight": np.ones((1, 1), dtype=np.float32),
            },
            configs={"../adapter": LoRAAdapterConfig(1)},
        )

    unsafe = tmp_path / "unsafe-load.safetensors"
    target = "layers.0.q_proj"
    header = {
        "__metadata__": _metadata(caller_metadata={"__proto__": "unsafe"}),
        f"{target}.lora_A.weight": {
            "dtype": "F32",
            "shape": [1, 1],
            "data_offsets": [0, 4],
        },
        f"{target}.lora_B.weight": {
            "dtype": "F32",
            "shape": [1, 1],
            "data_offsets": [4, 8],
        },
    }
    _write_raw(
        unsafe,
        json.dumps(header, separators=(",", ":")),
        np.array([1.0, 2.0], dtype=np.float32).tobytes(),
    )
    with pytest.raises(ValueError, match="Unsafe caller metadata key"):
        load_lora_safetensors(unsafe)

    mismatch = tmp_path / "load-mismatch.safetensors"
    header["__metadata__"] = _metadata({target: {"rank": 2, "alpha": 2.0}})
    _write_raw(
        mismatch,
        json.dumps(header, separators=(",", ":")),
        np.array([1.0, 2.0], dtype=np.float32).tobytes(),
    )
    with pytest.raises(ValueError, match="2-D and match configured rank 2"):
        load_lora_safetensors(mismatch)
