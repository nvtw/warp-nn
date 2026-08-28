# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.gguf import GGUFArchive


def _string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _write_gguf(path, tensors, metadata=(), *, version=3, alignment=32):
    fields = [("general.alignment", 4, alignment), *metadata]
    header = bytearray(b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(fields)))
    for key, value_type, value in fields:
        header += _string(key) + struct.pack("<I", value_type)
        if value_type == 8:
            header += _string(value)
        elif value_type == 9:
            element_type, values = value
            header += struct.pack("<IQ", element_type, len(values))
            if element_type == 8:
                header += b"".join(_string(item) for item in values)
            else:
                formats = {4: "I", 6: "f"}
                header += struct.pack("<" + formats[element_type] * len(values), *values)
        else:
            formats = {4: "I", 7: "B", 11: "q", 12: "d"}
            header += struct.pack("<" + formats[value_type], value)

    data = bytearray()
    for name, tensor_type, values in tensors:
        offset = (len(data) + alignment - 1) // alignment * alignment
        data += b"\0" * (offset - len(data))
        header += _string(name)
        header += struct.pack("<I", values.ndim)
        header += struct.pack("<" + "Q" * values.ndim, *reversed(values.shape))
        header += struct.pack("<IQ", tensor_type, offset)
        data += values.tobytes()
    header += b"\0" * ((-len(header)) % alignment)
    path.write_bytes(header + data)


def test_gguf_loads_metadata_and_unquantized_tensors(tmp_path):
    f32 = np.array([[1.5, -2.0], [3.25, 4.0]], dtype=np.float32)
    f16 = np.array([0.5, -8.0], dtype=np.float16)
    bf16 = np.array([0x3F80, 0xC000], dtype=np.uint16)
    path = tmp_path / "tiny.gguf"
    _write_gguf(
        path,
        [("f32", 0, f32), ("f16", 1, f16), ("bf16", 30, bf16)],
        [
            ("model.name", 8, "tiny"),
            ("layers", 4, 2),
            ("scores", 9, (6, (1.5, 2.5))),
            ("labels", 9, (8, ("one", "two"))),
        ],
        alignment=64,
    )

    archive = GGUFArchive(path)
    loaded = archive.load("cpu")

    assert archive.version == 3
    assert archive.alignment == 64
    assert archive.metadata["model.name"] == "tiny"
    assert archive.metadata["layers"] == 2
    assert archive.metadata["scores"] == pytest.approx((1.5, 2.5))
    assert archive.metadata["labels"] == ("one", "two")
    assert archive.names == ("f32", "f16", "bf16")
    assert archive.tensor("f32").shape == (2, 2)
    assert archive.tensor("f32").offset % 64 == 0
    assert archive.tensor("bf16").dtype == wp.bfloat16
    np.testing.assert_array_equal(loaded["f32"].numpy(), f32)
    np.testing.assert_array_equal(loaded["f16"].numpy(), f16)
    np.testing.assert_array_equal(loaded["bf16"].numpy().view(np.uint16), bf16)


def test_gguf_uploads_selected_tensor_to_cuda(tmp_path):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    values = np.arange(12, dtype=np.float16).reshape(3, 4)
    path = tmp_path / "cuda.gguf"
    _write_gguf(path, [("weight", 1, values), ("unused", 1, values)])

    loaded = GGUFArchive(path).load("cuda:0", ["weight"])

    assert tuple(loaded) == ("weight",)
    np.testing.assert_array_equal(loaded["weight"].numpy(), values)


def test_gguf_indexes_q8_0_blocks_as_raw_storage(tmp_path):
    raw = np.arange(68, dtype=np.uint8)
    path = tmp_path / "q8.gguf"
    _write_gguf(path, [("weight", 8, raw[:64].reshape(2, 32))])
    path.write_bytes(path.read_bytes() + raw[64:].tobytes())

    archive = GGUFArchive(path)
    info = archive.tensor("weight")

    assert info.shape == (2, 32)
    assert info.format == "Q8_0"
    assert info.nbytes == 68
    np.testing.assert_array_equal(archive.load("cpu")["weight"].numpy(), raw)


def test_gguf_supports_v2_and_selected_loading(tmp_path):
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "v2.gguf"
    _write_gguf(path, [("weight", 0, values)], [("flag", 7, True), ("count", 11, -3)], version=2)

    archive = GGUFArchive(path)

    assert archive.metadata["flag"] is True
    assert archive.metadata["count"] == -3
    np.testing.assert_array_equal(archive.load("cpu", ["weight"])["weight"].numpy(), values)
    with pytest.raises(KeyError, match="Unknown GGUF tensors"):
        archive.load("cpu", ["missing"])


def test_gguf_loads_split_archive(tmp_path):
    first = tmp_path / "tiny-00001-of-00002.gguf"
    second = tmp_path / "tiny-00002-of-00002.gguf"
    split_total = ("split.tensors.count", 4, 2)
    _write_gguf(
        first,
        [("first", 0, np.array([1.0], dtype=np.float32))],
        [("split.count", 4, 2), ("split.no", 4, 0), split_total],
    )
    _write_gguf(
        second,
        [("second", 0, np.array([2.0], dtype=np.float32))],
        [("split.count", 4, 2), ("split.no", 4, 1), split_total],
    )

    archive = GGUFArchive((first, second))
    loaded = archive.load("cpu")

    assert archive.names == ("first", "second")
    assert loaded["first"].numpy()[0] == 1.0
    assert loaded["second"].numpy()[0] == 2.0
    with pytest.raises(ValueError, match="out of order"):
        GGUFArchive((second, first))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: b"BAD!" + data[4:], "magic"),
        (lambda data: data[:4] + struct.pack("<I", 1) + data[8:], "unsupported version"),
        (lambda data: data[:-1], "invalid data range"),
    ],
)
def test_gguf_rejects_malformed_files(tmp_path, mutate, message):
    path = tmp_path / "bad.gguf"
    _write_gguf(path, [("weight", 0, np.array([1.0], dtype=np.float32))])
    path.write_bytes(mutate(path.read_bytes()))

    with pytest.raises(ValueError, match=message):
        GGUFArchive(path)


def test_gguf_rejects_partial_q8_0_rows(tmp_path):
    path = tmp_path / "partial-q8.gguf"
    _write_gguf(path, [("weight", 8, np.zeros((2, 16), dtype=np.uint8))])

    with pytest.raises(ValueError, match="partial Q8_0 block"):
        GGUFArchive(path)


def test_gguf_rejects_quantized_and_unaligned_tensors(tmp_path):
    path = tmp_path / "unsupported.gguf"
    _write_gguf(path, [("weight", 2, np.array([0], dtype=np.uint8))])
    with pytest.raises(ValueError, match="unsupported type"):
        GGUFArchive(path)

    data = bytearray(path.read_bytes())
    type_position = data.find(struct.pack("<IQ", 2, 0))
    data[type_position : type_position + 12] = struct.pack("<IQ", 0, 1)
    path.write_bytes(data + b"\0" * 4)
    with pytest.raises(ValueError, match="unaligned"):
        GGUFArchive(path)


def test_gguf_rejects_invalid_boolean_metadata(tmp_path):
    path = tmp_path / "boolean.gguf"
    _write_gguf(path, [], [("flag", 7, True)])
    data = bytearray(path.read_bytes())
    data[data.find(_string("flag")) + len(_string("flag")) + 4] = 2
    path.write_bytes(data)

    with pytest.raises(ValueError, match="invalid boolean"):
        GGUFArchive(path)
