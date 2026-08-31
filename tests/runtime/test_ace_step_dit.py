# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest

from warp_nn.runtime.ace_step.dit import (
    AceStepDiTConfig,
    AceStepDiTLayout,
    TURBO_TIMESTEPS,
    bidirectional_attention_mask,
    dit_weight_names,
    flow_euler_step,
    split_adaln_modulation,
    timestep_embedding,
    turbo_schedule,
)


def _config(**overrides):
    config = {
        "hidden_size": 16,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "hidden_act": "silu",
        "in_channels": 12,
        "audio_acoustic_hidden_dim": 4,
        "patch_size": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "use_sliding_window": True,
        "sliding_window": 8,
        "rope_theta": 1_000_000,
        "rms_norm_eps": 1.0e-6,
        "attention_bias": False,
        "model_version": "turbo",
        "is_turbo": True,
    }
    config.update(overrides)
    return AceStepDiTConfig.from_dict(config)


def test_config_supports_distinct_xl_encoder_dimensions():
    config = _config(
        hidden_size=20,
        num_attention_heads=5,
        num_key_value_heads=1,
        head_dim=4,
        encoder_hidden_size=16,
    )
    assert config.hidden_size == 20
    assert config.encoder_hidden_size == 16
    assert config.layer_types == ("sliding_attention", "full_attention")


def test_config_rejects_incompatible_attention_and_context_channels():
    with pytest.raises(ValueError, match="head dimensions"):
        _config(head_dim=8)
    with pytest.raises(ValueError, match="three times"):
        _config(in_channels=16)
    with pytest.raises(ValueError, match="sliding layers"):
        _config(use_sliding_window=False)


def test_layout_fixes_padding_cache_shapes_and_layer_windows():
    config = _config()
    layout = AceStepDiTLayout.create(config, 2, 11, 7)
    assert layout.padded_frames == 12
    assert layout.patch_rows == 6
    assert layout.hidden_shape == (2, 6, 16)
    assert layout.self_kv_shape == (2, 2, 6, 4)
    assert layout.cross_kv_shape == (2, 2, 7, 4)
    assert layout.attention_window(config, 0) == 8
    assert layout.attention_window(config, 1) is None


def test_manifest_matches_official_dit_names_and_bias_policy():
    names = dit_weight_names(_config())
    assert len(names) == len(set(names)) == 58
    assert "decoder.proj_in.1.weight" in names
    assert "decoder.time_embed_r.time_proj.bias" in names
    assert "decoder.layers.1.scale_shift_table" in names
    assert "decoder.layers.0.cross_attn.k_norm.weight" in names
    assert "decoder.layers.0.self_attn.q_proj.bias" not in names
    biased = dit_weight_names(_config(attention_bias=True))
    assert "decoder.layers.0.self_attn.q_proj.bias" in biased


def test_timestep_embedding_matches_official_formula():
    values = np.array([0.0, 0.25], dtype=np.float32)
    actual = timestep_embedding(values, 5)
    frequencies = np.exp(-math.log(10000.0) * np.arange(2, dtype=np.float32) / 2)
    arguments = values[:, None] * 1000.0 * frequencies[None]
    expected = np.concatenate(
        (np.cos(arguments), np.sin(arguments), np.zeros((2, 1))), axis=1
    ).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_turbo_schedule_matches_distilled_and_variable_step_policy():
    assert turbo_schedule(shift=3.0) == TURBO_TIMESTEPS[3.0]
    assert turbo_schedule(shift=2.7) == TURBO_TIMESTEPS[3.0]
    np.testing.assert_allclose(
        turbo_schedule(shift=3.0, steps=2), (1.0, 0.75), rtol=0, atol=1.0e-12
    )
    assert turbo_schedule(timesteps=[0.95, 0.0]) == (0.9545454545454546,)
    with pytest.raises(ValueError):
        turbo_schedule(timesteps=[0.0])


def test_bidirectional_masks_adaln_split_and_euler_update():
    full = bidirectional_attention_mask(4)
    assert full.shape == (1, 1, 4, 4)
    assert full.all()
    local = bidirectional_attention_mask(
        4, valid=np.array([[1, 1, 1, 0]], dtype=np.bool_), window=1
    )
    assert local.shape == (1, 1, 4, 4)
    assert local[0, 0, 1].tolist() == [True, True, True, False]
    assert not local[..., 3].any()

    modulation = np.arange(2 * 6 * 3).reshape(2, 6, 3)
    parts = split_adaln_modulation(modulation)
    assert len(parts) == 6
    assert all(part.shape == (2, 1, 3) for part in parts)
    np.testing.assert_array_equal(parts[4], modulation[:, 4:5])

    latent = np.ones((1, 2, 2), dtype=np.float32)
    velocity = np.full_like(latent, 0.25)
    np.testing.assert_allclose(flow_euler_step(latent, velocity, 0.8, 0.3), 0.875)
    np.testing.assert_allclose(flow_euler_step(latent, velocity, 0.8), 0.8)
