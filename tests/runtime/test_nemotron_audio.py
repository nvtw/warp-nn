# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from tests.utilities import local_model_root
from warp_nn.runtime.nemotron.audio import (
    REQUIRED_SHARED_OPERATOR_APIS,
    NemotronAudioConfig,
    _AudioPlan,
    _audio_kernels,
    parakeet_mel_filter_bank,
    parakeet_subsampled_length,
    parakeet_weight_names,
    preprocess_parakeet_audio,
    preprocess_parakeet_wav,
)
from warp_nn.runtime.formats.wav import write_wav_pcm16
from warp_nn.runtime.nemotron.omni import NemotronMultimodalProcessor
from warp_nn.runtime.tokenizers import Qwen3Tokenizer


def _document():
    return {
        "sound_config": {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_hidden_layers": 24,
            "intermediate_size": 4096,
            "conv_kernel_size": 9,
            "subsampling_conv_channels": 256,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "subsampling_factor": 8,
            "num_mel_bins": 128,
            "projection_hidden_size": 4096,
            "projection_bias": False,
            "sampling_rate": 16000,
        },
        "llm_config": {"hidden_size": 2688},
    }


def test_omni_audio_config_and_weight_manifest(tmp_path):
    document = _document()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    config = NemotronAudioConfig.from_path(path)
    assert config.hidden_size == 1024
    assert config.llm_hidden_size == 2688
    assert config.subsampling_layers == 3
    assert parakeet_subsampled_length(101, config) == 13

    names = parakeet_weight_names(config)
    assert len(names) == 5 * 2 + 2 + 24 * 28 + 3
    assert "sound_encoder.encoder.subsampling.layers.6.weight" in names
    assert "sound_encoder.encoder.layers.23.self_attn.relative_k_proj.weight" in names
    assert names[-3:] == (
        "sound_projection.norm.weight",
        "sound_projection.linear1.weight",
        "sound_projection.linear2.weight",
    )
    assert REQUIRED_SHARED_OPERATOR_APIS == (
        "GroupedConv2dPlan(x, weight, bias, stride, padding, groups)",
        "GroupedConv1dPlan(x, weight, bias, stride, padding, groups)",
        "RelativeBidirectionalAttentionPlan(query, key, value, relative_key, bias_u, bias_v, valid)",
    )


def test_parakeet_feature_extraction_is_normalized_and_masked():
    time = np.arange(4000, dtype=np.float32) / 16000.0
    first = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)
    second = np.sin(2.0 * np.pi * 880.0 * time[:3200]).astype(np.float32)
    features = preprocess_parakeet_audio([first, second])
    assert features.input_features.shape == (2, 26, 128)
    assert features.attention_mask.shape == (2, 26)
    assert features.attention_mask.sum(axis=1).tolist() == [25, 20]
    assert np.isfinite(features.input_features).all()
    np.testing.assert_allclose(features.input_features[1, 20:], 0.0, rtol=0.0, atol=0.0)
    for index, length in enumerate((25, 20)):
        valid = features.input_features[index, :length]
        np.testing.assert_allclose(valid.mean(axis=0), 0.0, atol=2.0e-4)
        np.testing.assert_allclose(valid.std(axis=0, ddof=1), 1.0, atol=2.0e-3)


def test_mel_bank_and_wav_boundary(tmp_path):
    filters = parakeet_mel_filter_bank()
    assert filters.shape == (128, 257)
    assert filters.dtype == np.float32
    assert np.all(filters >= 0.0)
    assert np.count_nonzero(filters) > 128

    time = np.arange(1600, dtype=np.float32) / 16000.0
    mono = np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
    stereo = np.column_stack((mono, mono))
    good = tmp_path / "good.wav"
    write_wav_pcm16(good, stereo, 16000)
    assert preprocess_parakeet_wav(good).input_features.shape == (1, 11, 128)

    mono_path = tmp_path / "mono.wav"
    pcm = np.rint(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(mono_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(pcm.tobytes())
    assert preprocess_parakeet_wav(mono_path).input_features.shape == (1, 11, 128)

    wrong_rate = tmp_path / "wrong-rate.wav"
    write_wav_pcm16(wrong_rate, stereo, 48000)
    with pytest.raises(ValueError, match="16000 Hz"):
        preprocess_parakeet_wav(wrong_rate)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("attention_bias", True, "bias-free"),
        ("convolution_bias", True, "bias-free"),
        ("projection_bias", True, "bias-free"),
        ("subsampling_factor", 4, "convolution stack"),
        ("num_mel_bins", 127, "divisible"),
    ],
)
def test_audio_config_rejects_unsupported_geometry(field, value, match):
    document = _document()
    document["sound_config"][field] = value
    with pytest.raises(ValueError, match=match):
        NemotronAudioConfig.from_document(document)


def test_tiny_audio_plan_runs_end_to_end_on_cpu():
    config = NemotronAudioConfig(
        hidden_size=4,
        num_attention_heads=1,
        num_hidden_layers=1,
        intermediate_size=8,
        conv_kernel_size=3,
        subsampling_conv_channels=2,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        subsampling_factor=8,
        num_mel_bins=8,
        projection_hidden_size=8,
        llm_hidden_size=4,
    )
    rng = np.random.default_rng(79)

    def tensor(shape, scale=0.1, positive=False):
        values = rng.normal(scale=scale, size=shape).astype(np.float32)
        if positive:
            values = np.abs(values) + 0.5
        return wp.array(values, dtype=wp.bfloat16, device="cpu")

    weights = {}
    prefix = "sound_encoder.encoder.subsampling."
    for index, shape in (
        (0, (2, 1, 3, 3)),
        (2, (2, 1, 3, 3)),
        (3, (2, 2, 1, 1)),
        (5, (2, 1, 3, 3)),
        (6, (2, 2, 1, 1)),
    ):
        weights[prefix + f"layers.{index}.weight"] = tensor(shape)
        weights[prefix + f"layers.{index}.bias"] = tensor((2,))
    weights[prefix + "linear.weight"] = tensor((4, 2))
    weights[prefix + "linear.bias"] = tensor((4,))

    block = "sound_encoder.encoder.layers.0."
    for feed_forward in ("feed_forward1", "feed_forward2"):
        weights[block + feed_forward + ".linear1.weight"] = tensor((8, 4))
        weights[block + feed_forward + ".linear2.weight"] = tensor((4, 8))
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"):
        weights[block + f"self_attn.{projection}.weight"] = tensor((4, 4))
    weights[block + "self_attn.bias_u"] = tensor((1, 4))
    weights[block + "self_attn.bias_v"] = tensor((1, 4))
    weights[block + "conv.pointwise_conv1.weight"] = tensor((8, 4, 1))
    weights[block + "conv.depthwise_conv.weight"] = tensor((4, 1, 3))
    weights[block + "conv.pointwise_conv2.weight"] = tensor((4, 4, 1))
    for name in ("weight", "bias", "running_mean"):
        weights[block + f"conv.norm.{name}"] = tensor((4,))
    weights[block + "conv.norm.running_var"] = tensor((4,), positive=True)
    for norm in (
        "norm_feed_forward1",
        "norm_self_att",
        "norm_conv",
        "norm_feed_forward2",
        "norm_out",
    ):
        weights[block + norm + ".weight"] = tensor((4,), positive=True)
        weights[block + norm + ".bias"] = tensor((4,))
    weights["sound_projection.norm.weight"] = tensor((4,), positive=True)
    weights["sound_projection.linear1.weight"] = tensor((8, 4))
    weights["sound_projection.linear2.weight"] = tensor((4, 8))

    encoder = SimpleNamespace(
        config=config,
        device=wp.get_device("cpu"),
        dtype=wp.bfloat16,
        cublas=None,
        weights=weights,
        kernels=_audio_kernels(wp.bfloat16),
    )
    plan = _AudioPlan(encoder, (1, 9, 8))
    features = rng.normal(size=(1, 9, 8)).astype(np.float32)
    valid = np.ones((1, 9), dtype=np.bool_)
    valid[0, -1] = False
    masks = [valid]
    for width in (5, 3, 2):
        masks.append(np.ones((1, width), dtype=np.bool_))
    plan.features.assign(features)
    for source, target in zip(masks, plan.masks):
        target.assign(source)
    first = plan.run().numpy().astype(np.float32)
    second = plan.run().numpy().astype(np.float32)
    assert first.shape == (1, 2, 4)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_nemotron_multimodal_processor_matches_audio_placeholders():
    path = local_model_root() / "nvidia" / "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    if not (path / "tokenizer.json").is_file():
        pytest.skip("local Nemotron Omni tokenizer is unavailable")
    tokenizer = Qwen3Tokenizer(path)
    config = NemotronAudioConfig.from_path(path)
    processor = NemotronMultimodalProcessor(tokenizer, config)
    waveform = np.sin(
        2.0 * np.pi * 440.0 * np.arange(3200, dtype=np.float32) / 16000.0
    ).astype(np.float32)
    prompt = processor.encode_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": waveform},
                    {"type": "text", "text": "What can you hear?"},
                ],
            }
        ],
        enable_thinking=False,
    )
    assert len(prompt.audios) == 1
    start = prompt.audio_starts[0]
    tokens = parakeet_subsampled_length(
        int(prompt.audios[0].attention_mask.sum()), config
    )
    assert (
        prompt.token_ids[start : start + tokens] == (processor.audio_token_id,) * tokens
    )
