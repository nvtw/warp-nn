# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import warp as wp

from warp_nn.runtime.qwen.tts_codec import (
    Qwen3TTSCodecDecoder,
    _causal_conv,
    _causal_transpose,
    _decoder_weight_names,
)


def _config():
    return {
        "latent_dim": 1024,
        "codebook_dim": 512,
        "codebook_size": 2048,
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_quantizers": 16,
        "max_position_embeddings": 8000,
        "upsampling_ratios": [2, 2],
        "upsample_rates": [8, 5, 4, 3],
    }


def test_qwen3_tts_codec_weight_contract_is_complete_and_unique():
    names = _decoder_weight_names(_config())
    assert len(names) == len(set(names)) == 269
    assert "decoder.quantizer.rvq_first.input_proj.weight" not in names
    assert "decoder.quantizer.rvq_rest.input_proj.weight" not in names
    assert "decoder.pre_transformer.layers.7.mlp_layer_scale.scale" in names
    assert "decoder.decoder.6.conv.weight" in names


def test_qwen3_tts_codec_exact_temporal_geometry_cpu():
    frames = 7
    x = wp.empty((1, frames, 4), dtype=wp.float32, device="cpu")
    conv, causal = _causal_conv(
        x,
        wp.empty((6, 4, 7), dtype=wp.float32, device="cpu"),
        wp.empty(6, dtype=wp.float32, device="cpu"),
        dilation=3,
    )
    assert conv.plan.output.shape[1] > frames
    assert causal.shape == (1, frames, 6)

    transpose, cropped = _causal_transpose(
        causal,
        wp.empty((6, 3, 10), dtype=wp.float32, device="cpu"),
        wp.empty(3, dtype=wp.float32, device="cpu"),
        5,
    )
    assert transpose.plan.output.shape[1] == frames * 5 + 5
    assert cropped.shape == (1, frames * 5, 3)


def test_qwen3_tts_codec_frame_rate_contract():
    config = _config()
    assert math.prod(config["upsampling_ratios"] + config["upsample_rates"]) == 1920
    assert Qwen3TTSCodecDecoder.sample_rate == 24_000
    assert Qwen3TTSCodecDecoder.samples_per_frame == 1920
