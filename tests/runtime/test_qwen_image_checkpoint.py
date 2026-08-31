# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import pytest

from warp_nn.runtime.qwen_image import (
    QwenImageTransformerConfig,
    QwenImageTransformerManifest,
)


def _config():
    return QwenImageTransformerConfig(
        patch_size=2,
        input_channels=64,
        output_channels=16,
        layers=60,
        heads=24,
        head_dim=128,
        text_width=3584,
        rope_axes=(16, 56, 56),
        guidance_embeds=False,
    )


@dataclass(frozen=True)
class _Metadata:
    shape: tuple[int, ...]
    format: str = "BF16"

    @property
    def nbytes(self):
        return 2 * math.prod(self.shape)


class _Archive:
    def __init__(self, shapes):
        self._metadata = {name: _Metadata(shape) for name, shape in shapes.items()}

    @property
    def names(self):
        return tuple(self._metadata)

    def metadata(self, name):
        return self._metadata[name]


def test_official_transformer_manifest_matches_checkpoint_contract():
    manifest = QwenImageTransformerManifest.from_config(_config())
    shapes = manifest.shapes()

    assert len(manifest.tensors) == 1933
    assert manifest.parameter_count == 20_430_401_088
    assert manifest.bfloat16_bytes == 40_860_802_176
    assert shapes["img_in.weight"] == (3072, 64)
    assert shapes["transformer_blocks.0.img_mod.1.weight"] == (18432, 3072)
    assert shapes["transformer_blocks.59.txt_mlp.net.2.weight"] == (3072, 12288)
    assert shapes["transformer_blocks.59.attn.norm_added_k.weight"] == (128,)
    assert shapes["norm_out.linear.weight"] == (6144, 3072)
    assert shapes["proj_out.weight"] == (64, 3072)


def test_transformer_manifest_validates_metadata_without_loading():
    manifest = QwenImageTransformerManifest.from_config(_config())
    manifest.validate_archive(_Archive(manifest.shapes()))


def test_transformer_manifest_rejects_missing_and_misshaped_tensors():
    manifest = QwenImageTransformerManifest.from_config(_config())
    shapes = manifest.shapes()
    shapes.pop("transformer_blocks.12.attn.to_q.weight")
    with pytest.raises(ValueError, match="missing tensor"):
        manifest.validate_archive(_Archive(shapes))

    shapes = manifest.shapes()
    shapes["proj_out.weight"] = (32, 3072)
    with pytest.raises(ValueError, match="proj_out.weight.*expected"):
        manifest.validate_archive(_Archive(shapes))


def test_transformer_manifest_rejects_non_bfloat16_metadata():
    manifest = QwenImageTransformerManifest.from_config(_config())
    archive = _Archive(manifest.shapes())
    archive._metadata["img_in.weight"] = _Metadata((3072, 64), "F16")
    with pytest.raises(TypeError, match="BF16"):
        manifest.validate_archive(archive)
