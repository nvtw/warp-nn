# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.qwen_image import (
    QwenImageMMDiTPlan,
    QwenImageTransformerConfig,
    QwenImageTransformerManifest,
)
from warp_nn.runtime.operators import multi_axis_rotary_cache_values
from warp_nn.runtime.qwen_image.mmdit import (
    qwen_image_mmdit_workspace_bytes,
    qwen_image_rotary_coordinates,
)


def _config(layers=1):
    return QwenImageTransformerConfig(
        patch_size=2,
        input_channels=4,
        output_channels=1,
        layers=layers,
        heads=2,
        head_dim=6,
        text_width=8,
        rope_axes=(2, 2, 2),
        guidance_embeds=False,
    )


def _weights(config, rng):
    arrays = {}
    for name, shape in (
        QwenImageTransformerManifest.from_config(config).shapes().items()
    ):
        if name == "txt_norm.weight" or ".attn.norm_" in name:
            value = rng.normal(1.0, 0.04, shape)
        else:
            value = rng.normal(0.0, 0.04, shape)
        arrays[name] = value.astype(np.float32)
    width = config.hidden_size
    for stream in ("img", "txt"):
        bias = arrays[f"transformer_blocks.0.{stream}_mod.1.bias"]
        bias[2 * width : 3 * width] += 0.7
        bias[5 * width : 6 * width] += 0.7
    return {
        name: wp.array(value, dtype=wp.bfloat16, device="cuda:0")
        for name, value in arrays.items()
    }


def _linear(x, arrays, prefix):
    return x @ arrays[prefix + ".weight"].T + arrays[prefix + ".bias"]


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))


def _rms(x, weight):
    return x * weight / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1.0e-6)


def _layer_norm(x):
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
        x.var(axis=-1, keepdims=True) + 1.0e-6
    )


def _adaptive(x, modulation, shift, scale):
    return (
        _layer_norm(x) * (1.0 + modulation[:, None, scale]) + modulation[:, None, shift]
    )


def _rope(x, cosine, sine):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    output = np.empty_like(pairs)
    output[..., 0] = (
        pairs[..., 0] * cosine[None, None] - pairs[..., 1] * sine[None, None]
    )
    output[..., 1] = (
        pairs[..., 1] * cosine[None, None] + pairs[..., 0] * sine[None, None]
    )
    return output.reshape(x.shape)


def _qkv(x, arrays, prefix, heads, cosine, sine, added):
    stem = "add_{}_proj" if added else "to_{}"
    values = []
    for kind in ("q", "k", "v"):
        projected = _linear(x, arrays, prefix + "." + stem.format(kind))
        values.append(
            projected.reshape(x.shape[0], x.shape[1], heads, -1).transpose(0, 2, 1, 3)
        )
    q_norm = "norm_added_q" if added else "norm_q"
    k_norm = "norm_added_k" if added else "norm_k"
    query = _rope(
        _rms(values[0], arrays[prefix + "." + q_norm + ".weight"]), cosine, sine
    )
    key = _rope(
        _rms(values[1], arrays[prefix + "." + k_norm + ".weight"]), cosine, sine
    )
    return query, key, values[2]


def _reference(image, text, valid, timestep, arrays, config, height, width):
    hidden = _linear(image, arrays, "img_in")
    encoded = _linear(_rms(text, arrays["txt_norm.weight"]), arrays, "txt_in")
    half = 128
    frequencies = np.exp(-np.log(10000.0) * np.arange(half) / half)
    frequency = np.concatenate(
        (
            np.cos(timestep[:, None] * 1000.0 * frequencies),
            np.sin(timestep[:, None] * 1000.0 * frequencies),
        ),
        axis=1,
    )
    temb = _linear(
        _silu(_linear(frequency, arrays, "time_text_embed.timestep_embedder.linear_1")),
        arrays,
        "time_text_embed.timestep_embedder.linear_2",
    )

    text_coordinates, image_coordinates = qwen_image_rotary_coordinates(
        text.shape[1], height, width
    )
    text_cos, text_sin = multi_axis_rotary_cache_values(
        text_coordinates, config.rope_axes
    )
    image_cos, image_sin = multi_axis_rotary_cache_values(
        image_coordinates, config.rope_axes
    )
    prefix = "transformer_blocks.0"
    image_mod = _linear(_silu(temb), arrays, prefix + ".img_mod.1").reshape(
        image.shape[0], 6, -1
    )
    text_mod = _linear(_silu(temb), arrays, prefix + ".txt_mod.1").reshape(
        image.shape[0], 6, -1
    )
    image_norm = _adaptive(hidden, image_mod, 0, 1)
    text_norm = _adaptive(encoded, text_mod, 0, 1)
    attention = prefix + ".attn"
    image_qkv = _qkv(
        image_norm, arrays, attention, config.heads, image_cos, image_sin, False
    )
    text_qkv = _qkv(
        text_norm, arrays, attention, config.heads, text_cos, text_sin, True
    )
    query = np.concatenate((text_qkv[0], image_qkv[0]), axis=2)
    key = np.concatenate((text_qkv[1], image_qkv[1]), axis=2)
    value = np.concatenate((text_qkv[2], image_qkv[2]), axis=2)
    scores = np.einsum("bhqd,bhkd->bhqk", query, key) / np.sqrt(config.head_dim)
    key_valid = np.concatenate((valid, np.ones(image.shape[:2], dtype=bool)), axis=1)
    scores = np.where(key_valid[:, None, None], scores, -1.0e30)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    attended = np.einsum("bhqk,bhkd->bhqd", probabilities, value)
    text_attended = (
        attended[:, :, : text.shape[1]].transpose(0, 2, 1, 3).reshape(encoded.shape)
    )
    image_attended = (
        attended[:, :, text.shape[1] :].transpose(0, 2, 1, 3).reshape(hidden.shape)
    )
    hidden = hidden + image_mod[:, None, 2] * _linear(
        image_attended, arrays, attention + ".to_out.0"
    )
    encoded = encoded + text_mod[:, None, 2] * _linear(
        text_attended, arrays, attention + ".to_add_out"
    )
    image_up = _gelu(
        _linear(
            _adaptive(hidden, image_mod, 3, 4), arrays, prefix + ".img_mlp.net.0.proj"
        )
    )
    text_up = _gelu(
        _linear(
            _adaptive(encoded, text_mod, 3, 4), arrays, prefix + ".txt_mlp.net.0.proj"
        )
    )
    hidden = hidden + image_mod[:, None, 5] * _linear(
        image_up, arrays, prefix + ".img_mlp.net.2"
    )
    encoded = encoded + text_mod[:, None, 5] * _linear(
        text_up, arrays, prefix + ".txt_mlp.net.2"
    )
    final_mod = _linear(_silu(temb), arrays, "norm_out.linear").reshape(
        image.shape[0], 2, -1
    )
    hidden = _adaptive(hidden, final_mod, 1, 0)
    return _linear(hidden, arrays, "proj_out")


@pytest.mark.skipif(not is_device_available("cuda:0"), reason="CUDA is unavailable")
def test_tiny_mmdit_matches_numpy_and_replays_captured_graph():
    config = _config()
    rng = np.random.default_rng(401)
    weights = _weights(config, rng)
    arrays = {name: value.numpy() for name, value in weights.items()}
    image_values = rng.normal(0.0, 0.2, (1, 4, 4)).astype(np.float32)
    text_values = rng.normal(0.0, 0.2, (1, 3, 8)).astype(np.float32)
    valid_values = np.array([[True, True, False]])
    timestep_values = np.array([0.6], dtype=np.float32)
    image = wp.array(image_values, dtype=wp.bfloat16, device="cuda:0")
    text = wp.array(text_values, dtype=wp.bfloat16, device="cuda:0")
    valid = wp.array(valid_values, dtype=wp.bool, device="cuda:0")
    timestep = wp.array(timestep_values, dtype=wp.float32, device="cuda:0")
    plan = QwenImageMMDiTPlan(image, text, valid, timestep, weights, config, 2, 2)
    actual = plan.execute().numpy()
    expected = _reference(
        image.numpy(), text.numpy(), valid_values, timestep_values, arrays, config, 2, 2
    )
    np.testing.assert_allclose(actual, expected, rtol=0.08, atol=0.04)

    plan.capture()
    replacement = rng.normal(0.0, 0.15, image_values.shape).astype(np.float32)
    next_timestep = np.array([0.25], dtype=np.float32)
    plan.replay(image_tokens=replacement, timestep=next_timestep)
    expected = _reference(
        image.numpy(), text.numpy(), valid_values, next_timestep, arrays, config, 2, 2
    )
    np.testing.assert_allclose(plan.output.numpy(), expected, rtol=0.08, atol=0.04)


def test_official_workspace_is_bounded_independent_of_layer_count():
    official = QwenImageTransformerConfig(
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
    expected = 1_450_221_047
    assert qwen_image_mmdit_workspace_bytes(official, 83 * 83, 990) == expected
    assert (
        qwen_image_mmdit_workspace_bytes(
            QwenImageTransformerConfig(**{**official.__dict__, "layers": 1}),
            83 * 83,
            990,
        )
        == expected
    )


@pytest.mark.skipif(not is_device_available("cuda:0"), reason="CUDA is unavailable")
def test_two_layers_alias_scratch_and_replay():
    config = _config(layers=2)
    rng = np.random.default_rng(402)
    weights = _weights(config, rng)
    image = wp.array(
        rng.normal(0.0, 0.2, (1, 4, 4)), dtype=wp.bfloat16, device="cuda:0"
    )
    text = wp.array(rng.normal(0.0, 0.2, (1, 3, 8)), dtype=wp.bfloat16, device="cuda:0")
    valid = wp.array([[True, True, False]], dtype=wp.bool, device="cuda:0")
    timestep = wp.array([0.4], dtype=wp.float32, device="cuda:0")
    plan = QwenImageMMDiTPlan(image, text, valid, timestep, weights, config, 2, 2)

    first, second = plan.layers
    assert first.image_qkv.q.output.ptr == second.image_qkv.q.output.ptr
    assert first.image_mlp_up.output.ptr == second.image_mlp_up.output.ptr
    assert first.image_output.ptr == plan.image_input.output.ptr
    assert second.image_output.ptr == plan.image_input.output.ptr
    assert plan.activation_workspace_nbytes == qwen_image_mmdit_workspace_bytes(
        config, 4, 3
    )

    expected = plan.execute().numpy()
    plan.capture()
    np.testing.assert_array_equal(plan.output.numpy(), expected)
    plan.replay()
    np.testing.assert_array_equal(plan.output.numpy(), expected)
    assert plan.layer_scratch_nbytes == 4_842
