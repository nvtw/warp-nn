# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import json

import numpy as np
import warp as wp

from tests.utilities import write_safetensors
from warp_nn.runtime.kernels import _dequantize_e4m3_kernel
from warp_nn.runtime.formats.safetensors import SafeTensorArchive


def test_safetensor_archive_loads_single_file(tmp_path):
    values = np.array([[1.5, -2.0], [3.25, 4.0]], dtype=np.float32)
    write_safetensors(
        tmp_path / "model.safetensors",
        {"weight": ("F32", values.shape, values.tobytes())},
    )

    archive = SafeTensorArchive(tmp_path)
    loaded = archive.load("cpu")

    assert archive.names == ("weight",)
    assert archive.metadata("weight").dtype == wp.float32
    np.testing.assert_array_equal(loaded["weight"].numpy(), values)


def test_safetensor_archive_loads_shards_and_bfloat16_bits(tmp_path):
    bits = np.array([0x3F80, 0xC000], dtype=np.uint16)
    write_safetensors(
        tmp_path / "part.safetensors", {"bf16": ("BF16", bits.shape, bits.tobytes())}
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"bf16": "part.safetensors"}}),
        encoding="utf-8",
    )

    loaded = SafeTensorArchive(tmp_path).load("cpu")["bf16"]

    np.testing.assert_array_equal(loaded.numpy().view(np.uint16), bits)


def test_safetensor_archive_rejects_invalid_range(tmp_path):
    write_safetensors(tmp_path / "model.safetensors", {"bad": ("F32", (2,), b"\0" * 4)})

    with pytest.raises(ValueError, match="Invalid data range"):
        SafeTensorArchive(tmp_path)


def test_safetensor_archive_loads_and_dequantizes_e4m3(tmp_path):
    bits = np.array([0x00, 0x01, 0x08, 0x38, 0x3C, 0x7E, 0xB8], dtype=np.uint8)
    write_safetensors(
        tmp_path / "model.safetensors",
        {"weight": ("F8_E4M3", bits.shape, bits.tobytes())},
    )
    archive = SafeTensorArchive(tmp_path)

    packed = archive.load("cuda:0")["weight"]
    scale = wp.array([2.0], dtype=wp.float32, device="cuda:0")
    output = wp.empty(bits.shape, dtype=wp.bfloat16, device="cuda:0")
    wp.launch(
        _dequantize_e4m3_kernel,
        dim=bits.size,
        inputs=[packed, scale, output],
        device="cuda:0",
    )

    assert archive.metadata("weight").format == "F8_E4M3"
    expected = np.array([0.0, 2**-8, 2**-5, 2.0, 3.0, 896.0, -2.0], dtype=np.float32)
    np.testing.assert_array_equal(output.numpy(), expected)


def test_safetensor_archive_flattens_rank_five_tensor(tmp_path):
    values = np.arange(2 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2, 2)
    write_safetensors(
        tmp_path / "model.safetensors",
        {"weight": ("F32", values.shape, values.tobytes())},
    )
    archive = SafeTensorArchive(tmp_path)

    with pytest.raises(ValueError, match="flatten=True"):
        archive.load("cpu")
    loaded = archive.load("cpu", flatten=True)["weight"]

    assert loaded.shape == (values.size,)
    np.testing.assert_array_equal(loaded.numpy(), values.reshape(-1))


def test_safetensor_archive_accepts_explicit_custom_index_name(tmp_path):
    values = np.array([1.0, 2.0], dtype=np.float32)
    write_safetensors(
        tmp_path / "diffusion_pytorch_model-00001-of-00001.safetensors",
        {"weight": ("F32", values.shape, values.tobytes())},
    )
    index = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight": "diffusion_pytorch_model-00001-of-00001.safetensors"
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = SafeTensorArchive(index).load("cpu")["weight"]
    np.testing.assert_array_equal(loaded.numpy(), values)
