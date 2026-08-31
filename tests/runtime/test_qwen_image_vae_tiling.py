# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.runtime.test_qwen_image_vae_decoder import _tiny_config, _tiny_weights
from warp_nn.runtime.qwen_image.vae_decoder import (
    QwenImage2512VAEDecoder,
    _QwenImageVAEDecoderPlan,
)
from warp_nn.runtime.qwen_image.vae_tiling import (
    _QwenImageVAETiledDecoderPlan,
    QwenImageVAETilingConfig,
    _tile_origins,
)


def test_official_qwen_image_vae_tiling_geometry():
    config = QwenImageVAETilingConfig.create()
    assert config.tile == (32, 32)
    assert config.stride == (24, 24)
    assert config.overlap == (8, 8)
    assert _tile_origins(166, 24) == (0, 24, 48, 72, 96, 120, 144)
    assert config.sample_geometry(8) == ((256, 256), (192, 192), (64, 64))
    assert QwenImage2512VAEDecoder.mode == "exact_untiled"
    assert not QwenImage2512VAEDecoder.approximate


def test_tiled_qwen_image_vae_matches_seam_free_tiny_reference():
    config = _tiny_config()
    weights = _tiny_weights(config)
    exact = _QwenImageVAEDecoderPlan(config, weights, 3, 3)
    tiled = _QwenImageVAETiledDecoderPlan(config, weights, 3, 3, tile=2, stride=1)
    latent = np.array(
        [[[[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3], [0.4, -0.4, 0.5]]]],
        dtype=np.float32,
    )
    source = wp.array(latent, device="cpu")
    wp.copy(exact.input, source)
    wp.copy(tiled.input, source)

    expected = exact.execute().numpy()
    actual = tiled.execute().numpy()

    assert tiled.mode == "official_overlap_tiled"
    assert tiled.approximate
    assert tiled.tile_count == 9
    assert tiled.decoder_shape_count == 4
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2.0e-6)


def test_qwen_image_vae_tiling_is_explicit_and_validated():
    with pytest.raises(ValueError, match="stride cannot exceed"):
        QwenImageVAETilingConfig.create(tile=8, stride=9)
    config = _tiny_config()
    tiled = _QwenImageVAETiledDecoderPlan(
        config, _tiny_weights(config), 2, 2, tile=2, stride=1
    )
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        tiled.capture()
