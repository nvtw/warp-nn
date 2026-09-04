# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.ace_step.dit import (
    AceStepDiTConfig,
    AceStepDiTLayout,
    AceStepAttentionPlan,
    AceStepDiTLayerPlan,
    AceStepDiTPlan,
    AceStepGuidedDiTPlan,
    TURBO_TIMESTEPS,
    _apg_flow_kernels,
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
        encoder_num_attention_heads=4,
        encoder_num_key_value_heads=1,
    )
    assert config.hidden_size == 20
    assert config.encoder_hidden_size == 16
    assert config.layer_types == ("sliding_attention", "full_attention")


def test_config_rejects_incompatible_attention_and_context_channels():
    projected = _config(hidden_size=20, head_dim=8, encoder_hidden_size=32)
    assert projected.num_attention_heads * projected.head_dim == 32
    assert projected.hidden_size == 20
    with pytest.raises(ValueError, match="head groups"):
        _config(num_attention_heads=3)
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


def test_apg_flow_kernels_match_official_equations():
    condition = np.array([[[1.0, -2.0], [0.5, 3.0]]], dtype=np.float32)
    unconditional = np.array([[[0.25, -1.0], [-0.5, 1.0]]], dtype=np.float32)
    prediction = wp.array(
        np.concatenate((condition, unconditional)),
        dtype=wp.float32,
        device="cpu",
    )
    initial_momentum = np.full_like(condition, 0.2)
    momentum = wp.array(initial_momentum, dtype=wp.float32, device="cpu")
    latent = wp.ones((2, 2, 2), dtype=wp.float32, device="cpu")
    stats = wp.empty((1, 2, 3), dtype=wp.float64, device="cpu")
    active = wp.ones(1, dtype=wp.int32, device="cpu")
    guidance = wp.array([3.0], dtype=wp.float32, device="cpu")
    timestep = wp.ones(2, dtype=wp.float32, device="cpu")
    next_timestep = wp.array([0.5, 0.5], dtype=wp.float32, device="cpu")
    kernels = _apg_flow_kernels(wp.float32)
    wp.launch(
        kernels[0],
        dim=momentum.shape,
        inputs=[prediction, momentum, active],
        device="cpu",
    )
    wp.launch(
        kernels[1],
        dim=(1, 2),
        inputs=[prediction, momentum, stats],
        device="cpu",
    )
    wp.launch(
        kernels[2],
        dim=momentum.shape,
        inputs=[
            latent,
            prediction,
            momentum,
            stats,
            timestep,
            next_timestep,
            guidance,
        ],
        device="cpu",
    )

    difference = condition - unconditional - 0.75 * initial_momentum
    clip = np.minimum(1.0, 2.5 / np.linalg.norm(difference, axis=1, keepdims=True))
    projection = (
        clip
        * np.sum(difference * condition, axis=1, keepdims=True)
        / np.sum(condition**2, axis=1, keepdims=True)
    )
    update = clip * difference - projection * condition
    velocity = condition + 2.0 * update
    expected = 1.0 - 0.5 * velocity
    np.testing.assert_allclose(momentum.numpy(), difference, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(latent.numpy()[0], expected[0], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(latent.numpy()[1], expected[0], rtol=1.0e-6, atol=1.0e-6)


def _rms_norm(x, scale, epsilon):
    return x * scale / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon)


def test_executable_cross_attention_matches_reference_and_reuses_fixed_kv():
    from tests.utilities import is_device_available

    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    config = _config()
    rng = np.random.default_rng(109)
    hidden = rng.normal(0.0, 0.2, size=(2, 3, 16)).astype(np.float32)
    context = rng.normal(0.0, 0.2, size=(2, 5, 16)).astype(np.float32)
    prefix = "decoder.layers.0.cross_attn"
    arrays = {
        prefix + ".q_proj.weight": rng.normal(0.0, 0.2, size=(16, 16)).astype(
            np.float32
        ),
        prefix + ".k_proj.weight": rng.normal(0.0, 0.2, size=(8, 16)).astype(
            np.float32
        ),
        prefix + ".v_proj.weight": rng.normal(0.0, 0.2, size=(8, 16)).astype(
            np.float32
        ),
        prefix + ".o_proj.weight": rng.normal(0.0, 0.2, size=(16, 16)).astype(
            np.float32
        ),
        prefix + ".q_norm.weight": rng.normal(1.0, 0.1, size=4).astype(np.float32),
        prefix + ".k_norm.weight": rng.normal(1.0, 0.1, size=4).astype(np.float32),
    }
    weights = {
        name: wp.array(value, dtype=wp.bfloat16, device="cuda:0")
        for name, value in arrays.items()
    }
    hidden_wp = wp.array(hidden, dtype=wp.bfloat16, device="cuda:0")
    context_wp = wp.array(context, dtype=wp.bfloat16, device="cuda:0")
    plan = AceStepAttentionPlan(hidden_wp, weights, prefix, config, context=context_wp)
    actual = plan.execute().numpy()

    query = (hidden @ arrays[prefix + ".q_proj.weight"].T).reshape(2, 3, 4, 4)
    query = np.transpose(query, (0, 2, 1, 3))
    key = (context @ arrays[prefix + ".k_proj.weight"].T).reshape(2, 5, 2, 4)
    key = np.transpose(key, (0, 2, 1, 3))
    value = (context @ arrays[prefix + ".v_proj.weight"].T).reshape(2, 5, 2, 4)
    value = np.transpose(value, (0, 2, 1, 3))
    query = _rms_norm(query, arrays[prefix + ".q_norm.weight"], config.rms_norm_eps)
    key = _rms_norm(key, arrays[prefix + ".k_norm.weight"], config.rms_norm_eps)
    attended = np.zeros_like(query)
    for batch in range(2):
        for head in range(4):
            kv_head = head // 2
            scores = query[batch, head] @ key[batch, kv_head].T / 2.0
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            attended[batch, head] = probabilities @ value[batch, kv_head]
    merged = np.transpose(attended, (0, 2, 1, 3)).reshape(2, 3, 16)
    expected = merged @ arrays[prefix + ".o_proj.weight"].T
    np.testing.assert_allclose(actual, expected, rtol=0.04, atol=0.02)

    wp.capture_begin(device="cuda:0")
    plan.execute()
    graph = wp.capture_end(device="cuda:0")
    context_wp.assign(np.zeros_like(context))
    wp.capture_launch(graph)
    np.testing.assert_allclose(plan.output.numpy(), expected, rtol=0.04, atol=0.02)


def test_executable_self_attention_matches_sliding_reference():
    from tests.utilities import is_device_available

    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    config = _config(sliding_window=1)
    rng = np.random.default_rng(113)
    hidden = rng.normal(0.0, 0.2, size=(1, 5, 16)).astype(np.float32)
    prefix = "decoder.layers.0.self_attn"
    arrays = {
        prefix + ".q_proj.weight": rng.normal(0.0, 0.2, size=(16, 16)).astype(
            np.float32
        ),
        prefix + ".k_proj.weight": rng.normal(0.0, 0.2, size=(8, 16)).astype(
            np.float32
        ),
        prefix + ".v_proj.weight": rng.normal(0.0, 0.2, size=(8, 16)).astype(
            np.float32
        ),
        prefix + ".o_proj.weight": rng.normal(0.0, 0.2, size=(16, 16)).astype(
            np.float32
        ),
        prefix + ".q_norm.weight": rng.normal(1.0, 0.1, size=4).astype(np.float32),
        prefix + ".k_norm.weight": rng.normal(1.0, 0.1, size=4).astype(np.float32),
    }
    weights = {
        name: wp.array(value, dtype=wp.bfloat16, device="cuda:0")
        for name, value in arrays.items()
    }
    plan = AceStepAttentionPlan(
        wp.array(hidden, dtype=wp.bfloat16, device="cuda:0"),
        weights,
        prefix,
        config,
        position_ids=wp.array(np.arange(5, dtype=np.int64)[None], device="cuda:0"),
        cos_cache=wp.ones((5, 2), dtype=wp.bfloat16, device="cuda:0"),
        sin_cache=wp.zeros((5, 2), dtype=wp.bfloat16, device="cuda:0"),
        layer_index=0,
    )
    actual = plan.execute().numpy()
    query = (hidden @ arrays[prefix + ".q_proj.weight"].T).reshape(1, 5, 4, 4)
    query = _rms_norm(
        np.transpose(query, (0, 2, 1, 3)),
        arrays[prefix + ".q_norm.weight"],
        config.rms_norm_eps,
    )
    key = (hidden @ arrays[prefix + ".k_proj.weight"].T).reshape(1, 5, 2, 4)
    key = _rms_norm(
        np.transpose(key, (0, 2, 1, 3)),
        arrays[prefix + ".k_norm.weight"],
        config.rms_norm_eps,
    )
    value = (hidden @ arrays[prefix + ".v_proj.weight"].T).reshape(1, 5, 2, 4)
    value = np.transpose(value, (0, 2, 1, 3))
    attended = np.zeros_like(query)
    for head in range(4):
        kv_head = head // 2
        for token in range(5):
            first, end = max(0, token - 1), min(5, token + 2)
            scores = query[0, head, token] @ key[0, kv_head, first:end].T / 2.0
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            attended[0, head, token] = probabilities @ value[0, kv_head, first:end]
    merged = np.transpose(attended, (0, 2, 1, 3)).reshape(1, 5, 16)
    expected = merged @ arrays[prefix + ".o_proj.weight"].T
    np.testing.assert_allclose(actual, expected, rtol=0.04, atol=0.02)


def _attention_reference(x, source, arrays, prefix, heads, kv_heads, window=None):
    batch, sequence, hidden = x.shape
    width = hidden // heads
    key_length = source.shape[1]
    query = (x @ arrays[prefix + ".q_proj.weight"].T).reshape(
        batch, sequence, heads, width
    )
    query = _rms_norm(
        np.transpose(query, (0, 2, 1, 3)),
        arrays[prefix + ".q_norm.weight"],
        1.0e-6,
    )
    key = (source @ arrays[prefix + ".k_proj.weight"].T).reshape(
        batch, key_length, kv_heads, width
    )
    key = _rms_norm(
        np.transpose(key, (0, 2, 1, 3)),
        arrays[prefix + ".k_norm.weight"],
        1.0e-6,
    )
    value = (source @ arrays[prefix + ".v_proj.weight"].T).reshape(
        batch, key_length, kv_heads, width
    )
    value = np.transpose(value, (0, 2, 1, 3))
    attended = np.zeros_like(query)
    for b in range(batch):
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            for token in range(sequence):
                first = 0 if window is None else max(0, token - window)
                end = (
                    key_length
                    if window is None
                    else min(key_length, token + window + 1)
                )
                scores = query[b, head, token] @ key[b, kv_head, first:end].T
                scores /= np.sqrt(width)
                probability = np.exp(scores - scores.max())
                probability /= probability.sum()
                attended[b, head, token] = probability @ value[b, kv_head, first:end]
    merged = np.transpose(attended, (0, 2, 1, 3)).reshape(batch, sequence, hidden)
    return merged @ arrays[prefix + ".o_proj.weight"].T


def test_full_dit_layer_matches_numpy_reference_and_graph():
    from tests.utilities import is_device_available

    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    config = _config(sliding_window=1)
    rng = np.random.default_rng(131)
    hidden = rng.normal(0.0, 0.15, size=(1, 4, 16)).astype(np.float32)
    context = rng.normal(0.0, 0.15, size=(1, 6, 16)).astype(np.float32)
    timestep = rng.normal(0.0, 0.05, size=(1, 6, 16)).astype(np.float32)
    layer = "decoder.layers.0"
    arrays = {
        layer + ".scale_shift_table": rng.normal(0.0, 0.05, size=(1, 6, 16)).astype(
            np.float32
        ),
        layer + ".self_attn_norm.weight": rng.normal(1.0, 0.05, size=16).astype(
            np.float32
        ),
        layer + ".cross_attn_norm.weight": rng.normal(1.0, 0.05, size=16).astype(
            np.float32
        ),
        layer + ".mlp_norm.weight": rng.normal(1.0, 0.05, size=16).astype(np.float32),
        layer + ".mlp.gate_proj.weight": rng.normal(0.0, 0.15, size=(48, 16)).astype(
            np.float32
        ),
        layer + ".mlp.up_proj.weight": rng.normal(0.0, 0.15, size=(48, 16)).astype(
            np.float32
        ),
        layer + ".mlp.down_proj.weight": rng.normal(0.0, 0.15, size=(16, 48)).astype(
            np.float32
        ),
    }
    for module in ("self_attn", "cross_attn"):
        prefix = layer + "." + module
        arrays.update(
            {
                prefix + ".q_proj.weight": rng.normal(0.0, 0.15, size=(16, 16)).astype(
                    np.float32
                ),
                prefix + ".k_proj.weight": rng.normal(0.0, 0.15, size=(8, 16)).astype(
                    np.float32
                ),
                prefix + ".v_proj.weight": rng.normal(0.0, 0.15, size=(8, 16)).astype(
                    np.float32
                ),
                prefix + ".o_proj.weight": rng.normal(0.0, 0.15, size=(16, 16)).astype(
                    np.float32
                ),
                prefix + ".q_norm.weight": rng.normal(1.0, 0.05, size=4).astype(
                    np.float32
                ),
                prefix + ".k_norm.weight": rng.normal(1.0, 0.05, size=4).astype(
                    np.float32
                ),
            }
        )
    weights = {
        name: wp.array(value, dtype=wp.bfloat16, device="cuda:0")
        for name, value in arrays.items()
    }
    plan = AceStepDiTLayerPlan(
        wp.array(hidden, dtype=wp.bfloat16, device="cuda:0"),
        wp.array(timestep, dtype=wp.bfloat16, device="cuda:0"),
        wp.array(context, dtype=wp.bfloat16, device="cuda:0"),
        weights,
        config,
        0,
        position_ids=wp.array(np.arange(4, dtype=np.int64)[None], device="cuda:0"),
        cos_cache=wp.ones((4, 2), dtype=wp.bfloat16, device="cuda:0"),
        sin_cache=wp.zeros((4, 2), dtype=wp.bfloat16, device="cuda:0"),
    )
    plan.prepare_fixed_condition()
    wp.capture_begin(device="cuda:0")
    plan.execute()
    graph = wp.capture_end(device="cuda:0")
    wp.capture_launch(graph)

    table = arrays[layer + ".scale_shift_table"] + timestep
    self_input = _rms_norm(hidden, arrays[layer + ".self_attn_norm.weight"], 1.0e-6)
    self_input = self_input * (1.0 + table[:, 1, None]) + table[:, 0, None]
    self_output = _attention_reference(
        self_input, self_input, arrays, layer + ".self_attn", 4, 2, 1
    )
    after_self = hidden + self_output * table[:, 2, None]
    cross_input = _rms_norm(
        after_self, arrays[layer + ".cross_attn_norm.weight"], 1.0e-6
    )
    cross_output = _attention_reference(
        cross_input, context, arrays, layer + ".cross_attn", 4, 2
    )
    after_cross = after_self + cross_output
    mlp_input = _rms_norm(after_cross, arrays[layer + ".mlp_norm.weight"], 1.0e-6)
    mlp_input = mlp_input * (1.0 + table[:, 4, None]) + table[:, 3, None]
    gate = mlp_input @ arrays[layer + ".mlp.gate_proj.weight"].T
    up = mlp_input @ arrays[layer + ".mlp.up_proj.weight"].T
    activated = gate / (1.0 + np.exp(-gate)) * up
    down = activated @ arrays[layer + ".mlp.down_proj.weight"].T
    expected = after_cross + down * table[:, 5, None]
    np.testing.assert_allclose(plan.output.numpy(), expected, rtol=0.08, atol=0.04)


def test_top_level_dit_graph_and_sampler_smoke():
    from tests.utilities import is_device_available

    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")
    config = _config(
        num_hidden_layers=1,
        layer_types=["full_attention"],
        use_sliding_window=False,
    )
    rng = np.random.default_rng(137)
    arrays = {
        "decoder.proj_in.1.weight": rng.normal(0, 0.1, (16, 12, 2)).astype(np.float32),
        "decoder.proj_in.1.bias": np.zeros(16, dtype=np.float32),
        "decoder.condition_embedder.weight": rng.normal(0, 0.1, (16, 16)).astype(
            np.float32
        ),
        "decoder.condition_embedder.bias": np.zeros(16, dtype=np.float32),
        "decoder.norm_out.weight": np.ones(16, dtype=np.float32),
        "decoder.scale_shift_table": rng.normal(0, 0.02, (1, 2, 16)).astype(np.float32),
        "decoder.proj_out.1.weight": rng.normal(0, 0.1, (16, 4, 2)).astype(np.float32),
        "decoder.proj_out.1.bias": np.zeros(4, dtype=np.float32),
    }
    for time in ("time_embed", "time_embed_r"):
        arrays.update(
            {
                f"decoder.{time}.linear_1.weight": rng.normal(
                    0, 0.05, (16, 256)
                ).astype(np.float32),
                f"decoder.{time}.linear_1.bias": np.zeros(16, dtype=np.float32),
                f"decoder.{time}.linear_2.weight": rng.normal(0, 0.05, (16, 16)).astype(
                    np.float32
                ),
                f"decoder.{time}.linear_2.bias": np.zeros(16, dtype=np.float32),
                f"decoder.{time}.time_proj.weight": rng.normal(
                    0, 0.05, (96, 16)
                ).astype(np.float32),
                f"decoder.{time}.time_proj.bias": np.zeros(96, dtype=np.float32),
            }
        )
    layer = "decoder.layers.0"
    arrays.update(
        {
            layer + ".scale_shift_table": rng.normal(0, 0.02, (1, 6, 16)).astype(
                np.float32
            ),
            layer + ".self_attn_norm.weight": np.ones(16, dtype=np.float32),
            layer + ".cross_attn_norm.weight": np.ones(16, dtype=np.float32),
            layer + ".mlp_norm.weight": np.ones(16, dtype=np.float32),
            layer + ".mlp.gate_proj.weight": rng.normal(0, 0.08, (48, 16)).astype(
                np.float32
            ),
            layer + ".mlp.up_proj.weight": rng.normal(0, 0.08, (48, 16)).astype(
                np.float32
            ),
            layer + ".mlp.down_proj.weight": rng.normal(0, 0.08, (16, 48)).astype(
                np.float32
            ),
        }
    )
    for module in ("self_attn", "cross_attn"):
        prefix = layer + "." + module
        arrays.update(
            {
                prefix + ".q_proj.weight": rng.normal(0, 0.08, (16, 16)).astype(
                    np.float32
                ),
                prefix + ".k_proj.weight": rng.normal(0, 0.08, (8, 16)).astype(
                    np.float32
                ),
                prefix + ".v_proj.weight": rng.normal(0, 0.08, (8, 16)).astype(
                    np.float32
                ),
                prefix + ".o_proj.weight": rng.normal(0, 0.08, (16, 16)).astype(
                    np.float32
                ),
                prefix + ".q_norm.weight": np.ones(4, dtype=np.float32),
                prefix + ".k_norm.weight": np.ones(4, dtype=np.float32),
            }
        )
    weights = {
        name: wp.array(value, dtype=wp.bfloat16, device="cuda:0")
        for name, value in arrays.items()
    }
    hidden = wp.array(
        rng.normal(0, 0.2, (1, 5, 4)).astype(np.float32),
        dtype=wp.bfloat16,
        device="cuda:0",
    )
    plan = AceStepDiTPlan(
        hidden,
        wp.array(
            rng.normal(0, 0.2, (1, 5, 8)).astype(np.float32),
            dtype=wp.bfloat16,
            device="cuda:0",
        ),
        wp.array(
            rng.normal(0, 0.2, (1, 3, 16)).astype(np.float32),
            dtype=wp.bfloat16,
            device="cuda:0",
        ),
        weights,
        config,
    )
    plan.timestep.assign(np.array([0.8], dtype=np.float32))
    plan.timestep_r.assign(np.array([0.8], dtype=np.float32))
    plan.time_t.execute()
    frequency = plan.time_t.frequency.numpy().astype(np.float32)
    linear1 = (
        frequency @ arrays["decoder.time_embed.linear_1.weight"].T
        + arrays["decoder.time_embed.linear_1.bias"]
    )
    expected_activation = linear1 / (1.0 + np.exp(-linear1))
    np.testing.assert_allclose(
        plan.time_t.first_activation.numpy(),
        expected_activation,
        rtol=0.03,
        atol=0.01,
    )
    before = hidden.numpy().copy()
    graph = plan.capture()
    assert graph is not None
    output = plan.run_schedule((0.8, 0.5)).numpy()
    assert output.shape == (1, 5, 4)
    assert np.isfinite(output).all()
    assert not np.array_equal(output, before)
    with pytest.raises(ValueError, match="strictly descending"):
        plan.run_schedule((0.5, 0.8))


def test_guided_schedule_switches_to_non_cover_at_official_step(monkeypatch):
    from types import SimpleNamespace

    plan = object.__new__(AceStepGuidedDiTPlan)
    plan.audio_cover_strength = 0.6
    plan.guidance_start = 0.0
    plan.guidance_end = 1.0
    plan.guidance_scale = 7.0
    plan.graph = object()
    plan.hidden = wp.zeros((1, 1, 1), dtype=wp.float32, device="cpu")
    plan.active = wp.zeros(1, dtype=wp.int32, device="cpu")
    plan.guidance = wp.ones(1, dtype=wp.float32, device="cpu")
    cover_context = wp.zeros((2, 1, 2), dtype=wp.float32, device="cpu")
    non_cover_context = wp.ones((2, 1, 2), dtype=wp.float32, device="cpu")
    cover_condition = wp.zeros((2, 2, 3), dtype=wp.float32, device="cpu")
    non_cover_condition = wp.ones((2, 2, 3), dtype=wp.float32, device="cpu")
    cover_valid = wp.zeros((2, 2), dtype=wp.bool, device="cpu")
    non_cover_valid = wp.ones((2, 2), dtype=wp.bool, device="cpu")
    key_valid = wp.empty_like(cover_valid)
    key_valid.assign(cover_valid)
    switches = []
    plan.plan = SimpleNamespace(
        context_latents=cover_context,
        _condition_tensors={"condition": cover_condition.reshape((4, 3))},
        layers=[
            SimpleNamespace(
                cross_attention=SimpleNamespace(
                    attention=SimpleNamespace(key_valid=key_valid)
                )
            )
        ],
        prepare_fixed_condition=lambda: switches.append(True),
        timestep=wp.ones(2, dtype=wp.float32, device="cpu"),
        timestep_r=wp.ones(2, dtype=wp.float32, device="cpu"),
        next_timestep=wp.zeros(2, dtype=wp.float32, device="cpu"),
    )
    plan.non_cover_context = non_cover_context
    plan.non_cover_condition = non_cover_condition
    plan.non_cover_valid = non_cover_valid
    contexts = []
    progress = []
    monkeypatch.setattr(
        wp,
        "capture_launch",
        lambda _graph: contexts.append(float(cover_context.numpy()[0, 0, 0])),
    )
    plan.run_schedule(
        (1.0, 0.8, 0.6, 0.4, 0.2),
        progress=lambda completed, total: progress.append((completed, total)),
    )
    assert contexts == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert progress == [(index, 5) for index in range(6)]
    assert len(switches) == 1
    np.testing.assert_array_equal(key_valid.numpy(), True)
