# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import struct
import zipfile

import numpy as np
import pytest

from warp_nn.runtime.formats.pytorch import load_pytorch_zip


def _tensor_pickle():
    def text(value):
        return b"X" + struct.pack("<I", len(value)) + value

    return b"".join(
        (
            b"\x80\x02",
            b"ctorch._utils\n_rebuild_tensor_v2\n",
            b"(",
            b"(",
            text(b"storage"),
            b"ctorch\nFloatStorage\n",
            text(b"0"),
            text(b"cpu"),
            b"J" + struct.pack("<i", 6),
            b"tQ",
            b"K\x00",
            b"K\x01K\x02K\x03\x87",
            b"K\x06K\x03K\x01\x87",
            b"\x89",
            b"ccollections\nOrderedDict\n)R",
            b"tR.",
        )
    )


def _write_archive(path, metadata=None):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("tensor/data.pkl", metadata or _tensor_pickle())
        archive.writestr("tensor/byteorder", "little")
        archive.writestr("tensor/data/0", np.arange(6, dtype=np.float32).tobytes())


def test_load_dependency_free_pytorch_tensor(tmp_path):
    path = tmp_path / "tensor.pt"
    _write_archive(path)
    value = load_pytorch_zip(path)
    np.testing.assert_array_equal(
        value, np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.float32)
    )


def test_pytorch_reader_rejects_arbitrary_pickle_globals(tmp_path):
    path = tmp_path / "unsafe.pt"
    _write_archive(path, b"\x80\x02cposix\nsystem\n.")
    with pytest.raises(Exception, match="unsupported PyTorch pickle global"):
        load_pytorch_zip(path)
