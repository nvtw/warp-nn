# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.qwen_image.runner import QwenImageVAEConfig
from warp_nn.runtime.qwen_image.vae import _qwen_image_vae_decoder_weight_specs
from warp_nn.runtime.qwen_image.vae_decoder import _QwenImageVAEDecoderPlan


def _tiny_config():
    return QwenImageVAEConfig(
        base_dim=2,
        dimension_multipliers=(1,),
        residual_blocks=1,
        latent_channels=1,
        temporal_downsample=(),
        latent_mean=(0.5,),
        latent_std=(2.0,),
    )


def _tiny_weights(config):
    values = {}
    for spec in _qwen_image_vae_decoder_weight_specs(config):
        value = np.zeros(spec.prepared_shape, dtype=np.float32)
        if spec.name.endswith(".gamma"):
            value.fill(1.0)
        values[spec.name] = value

    values["post_quant_conv.weight"][0, 0, 0, 0] = 1.0
    values["decoder.conv_in.weight"][0, 0, 1, 1] = 1.0
    values["decoder.conv_in.weight"][1, 0, 1, 1] = -1.0
    values["decoder.conv_out.weight"][0, 0, 1, 1] = 2.0
    values["decoder.conv_out.weight"][1, 1, 1, 1] = 1.0
    values["decoder.conv_out.weight"][2, :, 1, 1] = 1.0
    return {name: wp.array(value, device="cpu") for name, value in values.items()}


def test_tiny_qwen_image_vae_decoder_matches_reference():
    config = _tiny_config()
    decoder = _QwenImageVAEDecoderPlan(config, _tiny_weights(config), 2, 2)
    latent = np.array([[[[-0.75, -0.25], [0.25, 0.75]]]], dtype=np.float32)
    wp.copy(decoder.input, wp.array(latent, device="cpu"))

    actual = decoder.execute().numpy()

    raw = latent[:, 0] * 2.0 + 0.5
    features = np.stack((raw, -raw), axis=-1)
    normalized = features / np.sqrt(
        np.mean(features * features, axis=-1, keepdims=True) + 1.0e-12
    )
    activated = normalized / (1.0 + np.exp(-normalized))
    expected = np.stack(
        (2.0 * activated[..., 0], activated[..., 1], activated.sum(axis=-1)), axis=1
    )
    np.testing.assert_allclose(actual, np.clip(expected, -1.0, 1.0), atol=2.0e-6)


def test_qwen_image_vae_decoder_capture_requires_cuda():
    config = _tiny_config()
    decoder = _QwenImageVAEDecoderPlan(config, _tiny_weights(config), 1, 1)
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        decoder.capture()
