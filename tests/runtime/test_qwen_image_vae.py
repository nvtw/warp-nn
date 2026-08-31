# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from math import prod

import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.qwen_image.runner import QwenImageVAEConfig
from warp_nn.runtime.qwen_image.vae import (
    QwenImageVAEWeightSpec,
    prepare_qwen_image_vae_decoder_weights,
    qwen_image_2512_vae_decoder_weight_specs,
)


def _config():
    return QwenImageVAEConfig(
        base_dim=96,
        dimension_multipliers=(1, 2, 4, 4),
        residual_blocks=2,
        latent_channels=16,
        temporal_downsample=(False, True, True),
        latent_mean=(0.0,) * 16,
        latent_std=(1.0,) * 16,
    )


def test_qwen_image_2512_decoder_checkpoint_contract():
    specs = qwen_image_2512_vae_decoder_weight_specs(_config())
    by_name = {spec.name: spec for spec in specs}

    assert len(specs) == 104
    assert len(by_name) == len(specs)
    assert sum(prod(spec.source_shape) for spec in specs) == 71_524_595
    assert sum(prod(spec.prepared_shape) for spec in specs) == 25_291_955
    assert by_name["post_quant_conv.weight"].source_shape == (16, 16, 1, 1, 1)
    assert by_name["decoder.conv_in.weight"].source_shape == (384, 16, 3, 3, 3)
    assert by_name["decoder.conv_in.weight"].prepared_shape == (384, 16, 3, 3)
    assert by_name["decoder.mid_block.attentions.0.to_qkv.weight"].source_shape == (
        1152,
        384,
        1,
        1,
    )
    assert by_name[
        "decoder.up_blocks.1.resnets.0.conv_shortcut.weight"
    ].source_shape == (384, 192, 1, 1, 1)
    assert by_name[
        "decoder.up_blocks.2.upsamplers.0.resample.1.weight"
    ].source_shape == (96, 192, 3, 3)
    assert by_name["decoder.norm_out.gamma"].prepared_shape == (96,)
    assert by_name["decoder.conv_out.weight"].source_shape == (3, 96, 3, 3, 3)
    assert not any(".time_conv." in spec.name for spec in specs)


def test_qwen_image_2512_decoder_contract_is_strict():
    config = _config()
    incompatible = QwenImageVAEConfig(
        base_dim=64,
        dimension_multipliers=config.dimension_multipliers,
        residual_blocks=config.residual_blocks,
        latent_channels=config.latent_channels,
        temporal_downsample=config.temporal_downsample,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
    )
    with pytest.raises(ValueError, match="unsupported Qwen-Image-2512"):
        qwen_image_2512_vae_decoder_weight_specs(incompatible)


@dataclass(frozen=True)
class _Metadata:
    shape: tuple[int, ...]


class _SyntheticArchive:
    def __init__(self, tensors):
        self.tensors = tensors
        self.calls = []

    @property
    def names(self):
        return tuple(self.tensors)

    def metadata(self, name):
        return _Metadata(tuple(self.tensors[name].shape))

    def load(self, device=None, names=None, *, flatten=False):
        self.calls.append((tuple(names), flatten))
        assert flatten
        return {
            name: wp.array(
                self.tensors[name].reshape(-1),
                dtype=wp.float32,
                device=device or "cpu",
            )
            for name in names
        }


def test_qwen_image_decoder_prepares_temporal_plane_and_conv2d():
    causal = np.arange(2 * 3 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 3, 2, 2)
    spatial = np.arange(4 * 2 * 3 * 3, dtype=np.float32).reshape(4, 2, 3, 3)
    gamma = np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1)
    archive = _SyntheticArchive(
        {"causal.weight": causal, "spatial.weight": spatial, "norm.gamma": gamma}
    )
    specs = (
        QwenImageVAEWeightSpec(
            "causal.weight", causal.shape, (2, 3, 2, 2), temporal_index=2
        ),
        QwenImageVAEWeightSpec("spatial.weight", spatial.shape, spatial.shape),
        QwenImageVAEWeightSpec("norm.gamma", gamma.shape, (4,)),
    )

    weights = prepare_qwen_image_vae_decoder_weights(archive, specs, "cpu")

    np.testing.assert_array_equal(weights["causal.weight"].numpy(), causal[:, :, 2])
    np.testing.assert_array_equal(weights["spatial.weight"].numpy(), spatial)
    np.testing.assert_array_equal(weights["norm.gamma"].numpy(), gamma[:, 0, 0, 0])
    assert archive.calls == [
        (("causal.weight",), True),
        (("spatial.weight",), True),
        (("norm.gamma",), True),
    ]


def test_qwen_image_decoder_validates_all_metadata_before_loading():
    archive = _SyntheticArchive({"weight": np.zeros((2, 3), dtype=np.float32)})
    specs = (QwenImageVAEWeightSpec("weight", (2, 4), (2, 4)),)
    with pytest.raises(ValueError, match=r"expected \(2, 4\)"):
        prepare_qwen_image_vae_decoder_weights(archive, specs, "cpu")
    assert archive.calls == []
