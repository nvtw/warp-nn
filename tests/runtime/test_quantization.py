# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.formats.gguf import BlockQuantizedTensor
from warp_nn.runtime.quantization import (
    estimate_loaded_weight_bytes,
    is_q8_linear_weight,
    load_native_weights,
    normalize_weight_quantization,
    q8_storage_nbytes,
    quantize_q8_0_weight,
)


@dataclass(frozen=True)
class _Metadata:
    shape: tuple[int, ...]
    dtype: type
    format: str
    nbytes: int


class _Archive:
    def __init__(self, arrays):
        self.arrays = arrays
        self.loads = []

    def metadata(self, name):
        value = self.arrays[name]
        return _Metadata(tuple(value.shape), value.dtype, "BF16", value.size * 2)

    def load(self, device, names):
        self.loads.append(tuple(names))
        return {
            name: wp.array(self.arrays[name], dtype=wp.bfloat16, device=device)
            for name in names
        }


@pytest.mark.parametrize(
    ("name", "selected"),
    [
        ("model.language_model.layers.0.self_attn.q_proj.weight", True),
        ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", True),
        ("model.language_model.layers.0.mlp.down_proj.weight", True),
        ("model.language_model.embed_tokens.weight", False),
        ("lm_head.weight", False),
        ("model.language_model.layers.0.input_layernorm.weight", False),
        ("model.language_model.layers.0.linear_attn.conv1d.weight", False),
    ],
)
def test_q8_linear_weight_selection(name, selected):
    metadata = _Metadata((64, 32), wp.bfloat16, "BF16", 4096)
    assert is_q8_linear_weight(name, metadata) is selected


def test_q8_weight_policy_validation_and_storage():
    assert normalize_weight_quantization(None) is None
    assert normalize_weight_quantization("Q8_0") == "q8_0"
    with pytest.raises(ValueError, match="weight_quantization"):
        normalize_weight_quantization("q4")
    assert q8_storage_nbytes((3, 64)) == 3 * 2 * 34
    with pytest.raises(ValueError, match="divisible"):
        q8_storage_nbytes((3, 63))


@pytest.mark.parametrize("dtype", [wp.float16, wp.bfloat16])
def test_quantize_q8_0_weight_matches_block_reference(dtype):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(83)
    source_np = rng.normal(0.0, 0.3, size=(3, 64)).astype(np.float32)
    source_np[0, :32] = 0.0
    source = wp.array(source_np, dtype=dtype, device="cuda:0")
    quantized = quantize_q8_0_weight(source)
    wp.synchronize_device("cuda:0")

    cast_source = source.numpy().astype(np.float32).reshape(3, 2, 32)
    maxima = np.max(np.abs(cast_source), axis=2)
    scales = maxima / 127.0
    divisors = np.where(maxima > 0.0, scales, 1.0)
    rounded = np.sign(cast_source) * np.floor(
        np.abs(cast_source / divisors[:, :, None]) + 0.5
    )
    expected_values = np.clip(rounded, -127, 127).astype(np.int8)

    assert isinstance(quantized, BlockQuantizedTensor)
    assert quantized.shape == (3, 64)
    assert quantized.values.shape == (3, 2, 32)
    assert quantized.words.shape == (3, 2, 8)
    np.testing.assert_array_equal(quantized.values.numpy(), expected_values)
    np.testing.assert_array_equal(quantized.scales.numpy(), scales.astype(np.float16))
    np.testing.assert_array_equal(
        quantized.words.numpy().view(np.int8).reshape(3, 2, 32), expected_values
    )


def test_load_native_weights_quantizes_projections_largest_first():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    arrays = {
        "model.language_model.embed_tokens.weight": np.ones((8, 32), np.float32),
        "model.language_model.layers.0.self_attn.q_proj.weight": np.ones(
            (4, 32), np.float32
        ),
        "model.language_model.layers.0.mlp.up_proj.weight": np.ones(
            (16, 32), np.float32
        ),
    }
    archive = _Archive(arrays)
    weights = load_native_weights(archive, "cuda:0", arrays, "q8_0")
    assert isinstance(weights["model.language_model.embed_tokens.weight"], wp.array)
    assert isinstance(
        weights["model.language_model.layers.0.self_attn.q_proj.weight"],
        BlockQuantizedTensor,
    )
    assert archive.loads[1:] == [
        ("model.language_model.layers.0.mlp.up_proj.weight",),
        ("model.language_model.layers.0.self_attn.q_proj.weight",),
    ]
    final, transient = estimate_loaded_weight_bytes(archive, arrays, "q8_0")
    assert final == 8 * 32 * 2 + 16 * 34 + 4 * 34
    assert transient == 16 * 32 * 2
