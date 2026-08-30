# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import warp as wp

from warp_nn.runtime.encoder import EncoderLayerPlan


def _layer_norm(x, weight, bias):
    mean = x.mean(-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + 1.0e-5) * weight + bias


def _reference(x, valid, weights, heads):
    batch, sequence, hidden = x.shape
    qkv = x.reshape(-1, hidden) @ weights["layer.self_attn.in_proj_weight"].T
    qkv += weights["layer.self_attn.in_proj_bias"]
    qkv = qkv.reshape(batch, sequence, 3, heads, hidden // heads)
    query, key, value = (
        np.transpose(qkv[:, :, index], (0, 2, 1, 3)) for index in range(3)
    )
    scores = query @ np.swapaxes(key, -1, -2) / math.sqrt(hidden // heads)
    scores = np.where(valid[:, None, None, :], scores, -np.inf)
    probability = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probability /= np.sum(probability, axis=-1, keepdims=True)
    attention = probability @ value
    attention = np.transpose(attention, (0, 2, 1, 3)).reshape(-1, hidden)
    branch = attention @ weights["layer.self_attn.out_proj.weight"].T
    branch += weights["layer.self_attn.out_proj.bias"]
    norm1 = _layer_norm(
        branch + x.reshape(-1, hidden),
        weights["layer.norm1.weight"],
        weights["layer.norm1.bias"],
    )
    ff = norm1 @ weights["layer.linear1.weight"].T + weights["layer.linear1.bias"]
    ff *= 0.5 * (1.0 + np.vectorize(math.erf)(ff / math.sqrt(2.0)))
    ff = ff @ weights["layer.linear2.weight"].T + weights["layer.linear2.bias"]
    return _layer_norm(
        ff + norm1, weights["layer.norm2.weight"], weights["layer.norm2.bias"]
    ).reshape(batch, sequence, hidden)


def test_encoder_layer_matches_numpy_cpu():
    rng = np.random.default_rng(19)
    batch, sequence, hidden, heads, feedforward = 2, 3, 4, 2, 6
    arrays = {
        "layer.self_attn.in_proj_weight": rng.normal(0, 0.2, (3 * hidden, hidden)),
        "layer.self_attn.in_proj_bias": rng.normal(0, 0.1, 3 * hidden),
        "layer.self_attn.out_proj.weight": rng.normal(0, 0.2, (hidden, hidden)),
        "layer.self_attn.out_proj.bias": rng.normal(0, 0.1, hidden),
        "layer.linear1.weight": rng.normal(0, 0.2, (feedforward, hidden)),
        "layer.linear1.bias": rng.normal(0, 0.1, feedforward),
        "layer.linear2.weight": rng.normal(0, 0.2, (hidden, feedforward)),
        "layer.linear2.bias": rng.normal(0, 0.1, hidden),
        "layer.norm1.weight": rng.normal(1, 0.1, hidden),
        "layer.norm1.bias": rng.normal(0, 0.1, hidden),
        "layer.norm2.weight": rng.normal(1, 0.1, hidden),
        "layer.norm2.bias": rng.normal(0, 0.1, hidden),
    }
    x = rng.normal(0, 0.5, (batch, sequence, hidden)).astype(np.float16)
    valid = np.array([[True, True, True], [True, True, False]])
    expected = _reference(x.astype(np.float32), valid, arrays, heads)
    weights = {
        name: wp.array(value.astype(np.float16), dtype=wp.float16, device="cpu")
        for name, value in arrays.items()
    }
    plan = EncoderLayerPlan(
        wp.array(x, dtype=wp.float16, device="cpu"),
        wp.array(valid, dtype=wp.bool, device="cpu"),
        weights,
        "layer",
        heads,
    )
    plan.execute()
    np.testing.assert_allclose(plan.output.numpy(), expected, rtol=4e-3, atol=4e-3)
