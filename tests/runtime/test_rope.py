# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest

from warp_nn.runtime.rope import resolve_rope_parameters, rotary_cache_values


def test_default_and_factor_one_yarn_match():
    default = {"rope_type": "default", "rope_theta": 10000.0}
    yarn = resolve_rope_parameters(
        default,
        {"rope_type": "yarn", "factor": 1.0},
        native_context=32,
        target_context=32,
    )
    expected = rotary_cache_values(32, 16, default)
    actual = rotary_cache_values(32, 16, yarn)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_yarn_cache_matches_reference_formula():
    dim, original, factor, theta = 16, 32, 2.0, 10000.0
    parameters = resolve_rope_parameters(
        {"rope_type": "default", "rope_theta": theta},
        {"rope_type": "yarn", "factor": factor},
        native_context=original,
        target_context=64,
    )
    cosine, sine = rotary_cache_values(64, dim, parameters)
    indices = np.arange(dim // 2, dtype=np.float32)
    frequencies = theta ** (2.0 * indices / dim)

    def correction(rotations):
        return (
            dim
            * math.log(original / (rotations * 2.0 * math.pi))
            / (2.0 * math.log(theta))
        )

    low = max(math.floor(correction(32.0)), 0)
    high = min(math.ceil(correction(1.0)), dim - 1)
    ramp = np.clip((indices - low) / (high - low), 0.0, 1.0)
    inverse = (1.0 / frequencies) * (1.0 - ramp) + (1.0 / (factor * frequencies)) * ramp
    magnitude = 1.0 + 0.1 * math.log(factor)
    angles = np.arange(64, dtype=np.float32)[:, None] * inverse[None, :]
    np.testing.assert_allclose(cosine, magnitude * np.cos(angles), rtol=1.0e-6)
    np.testing.assert_allclose(sine, magnitude * np.sin(angles), rtol=1.0e-6)
    np.testing.assert_allclose(cosine[0], magnitude)
    overridden = {**parameters, "attention_factor": 1.25}
    np.testing.assert_allclose(rotary_cache_values(1, dim, overridden)[0], 1.25)


@pytest.mark.parametrize(
    "scaling,match",
    [
        ({"rope_type": "linear"}, "only YaRN"),
        ({"rope_type": "yarn", "factor": 0.5}, "factor"),
        ({"rope_type": "yarn", "factor": float("nan")}, "factor"),
        ({"rope_type": "yarn", "original_max_position_embeddings": 0}, "original"),
        ({"rope_type": "yarn", "beta_fast": 1, "beta_slow": 2}, "beta_fast"),
    ],
)
def test_invalid_yarn_parameters(scaling, match):
    with pytest.raises(ValueError, match=match):
        resolve_rope_parameters(
            {"rope_type": "default", "rope_theta": 10000.0},
            scaling,
            native_context=32,
            target_context=64,
        )


def test_context_extension_requires_explicit_sufficient_yarn():
    base = {"rope_type": "default", "rope_theta": 10000.0}
    with pytest.raises(ValueError, match="enable YaRN"):
        resolve_rope_parameters(base, None, 32, 33)
    with pytest.raises(ValueError, match="covered"):
        resolve_rope_parameters(base, {"rope_type": "yarn", "factor": 2.0}, 32, 65)
    resolved = resolve_rope_parameters(base, {"rope_type": "yarn"}, 32, 63)
    assert resolved["factor"] == pytest.approx(63 / 32)
