# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.ace_step.vae import OobleckVAEConfig, OobleckVAEDecoder


def _tiny_decoder_weights(config, device):
    weights = {}

    def zeros(name, shape):
        weights[name] = wp.zeros(shape, dtype=wp.float32, device=device)

    multiples = (1, *config.channel_multiples)
    channels = config.decoder_channels
    zeros(
        "decoder.conv1.weight",
        (channels * multiples[-1], config.decoder_input_channels, 7),
    )
    zeros("decoder.conv1.bias", (channels * multiples[-1],))
    for block, stride in enumerate(reversed(config.downsampling_ratios)):
        input_channels = channels * multiples[len(config.downsampling_ratios) - block]
        output_channels = (
            channels * multiples[len(config.downsampling_ratios) - block - 1]
        )
        prefix = f"decoder.block.{block}"
        zeros(f"{prefix}.snake1.alpha", (input_channels,))
        zeros(f"{prefix}.snake1.beta", (input_channels,))
        zeros(f"{prefix}.conv_t1.weight", (input_channels, output_channels, 2 * stride))
        zeros(f"{prefix}.conv_t1.bias", (output_channels,))
        for unit in range(1, 4):
            residual = f"{prefix}.res_unit{unit}"
            for layer, kernel in ((1, 7), (2, 1)):
                zeros(f"{residual}.snake{layer}.alpha", (output_channels,))
                zeros(f"{residual}.snake{layer}.beta", (output_channels,))
                zeros(
                    f"{residual}.conv{layer}.weight",
                    (output_channels, output_channels, kernel),
                )
                zeros(f"{residual}.conv{layer}.bias", (output_channels,))
    zeros("decoder.snake1.alpha", (channels,))
    zeros("decoder.snake1.beta", (channels,))
    zeros("decoder.conv2.weight", (config.audio_channels, channels, 7))
    return weights


def test_ace_step_15_config_geometry():
    config = OobleckVAEConfig.from_dict(
        {
            "downsampling_ratios": [2, 4, 4, 6, 10],
            "channel_multiples": [1, 2, 4, 8, 16],
            "sampling_rate": 48_000,
        }
    )
    assert config.hop_length == 1920
    assert config.decoder_input_channels == 64
    assert config.audio_channels == 2


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_oobleck_decoder_shape_and_zero_weights(device):
    if not is_device_available(device):
        pytest.skip(f"{device} is unavailable")
    config = OobleckVAEConfig(
        encoder_hidden_size=2,
        downsampling_ratios=(2, 2),
        channel_multiples=(1, 2),
        decoder_channels=2,
        decoder_input_channels=3,
        audio_channels=2,
        sampling_rate=48_000,
    )
    decoder = OobleckVAEDecoder(
        config,
        _tiny_decoder_weights(config, device),
        latent_frames=3,
        device=device,
        dtype=wp.float32,
    )
    decoder.input.assign(np.ones(decoder.input.shape, dtype=np.float32))
    output = decoder.execute().numpy()
    assert output.shape == (1, 12, 2)
    np.testing.assert_array_equal(output, np.zeros_like(output))


def test_oobleck_decoder_cuda_graph():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    config = OobleckVAEConfig(
        encoder_hidden_size=2,
        downsampling_ratios=(2,),
        channel_multiples=(1,),
        decoder_channels=2,
        decoder_input_channels=3,
        audio_channels=2,
    )
    decoder = OobleckVAEDecoder(
        config,
        _tiny_decoder_weights(config, "cuda:0"),
        latent_frames=2,
        device="cuda:0",
        dtype=wp.float32,
    )
    decoder.capture()
    output = decoder.execute().numpy()
    assert output.shape == (1, 4, 2)
    np.testing.assert_array_equal(output, np.zeros_like(output))
