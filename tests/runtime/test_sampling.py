# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.autoregressive import AutoregressiveRunner
from warp_nn.runtime.sampling import sample_runner_token, sample_token


class CaptureDistribution:
    def choice(self, candidates, p):
        self.probabilities = dict(zip(candidates.tolist(), p.tolist()))
        return candidates[np.argmax(p)]


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_temperature_top_k_then_nucleus(temperature):
    logits = np.log([0.05, 0.15, 0.3, 0.5])
    rng = CaptureDistribution()
    sample_token(logits, temperature=temperature, top_k=3, top_p=0.8, rng=rng)
    expected = np.array([0.5, 0.3, 0.15]) ** (1 / temperature)
    expected /= expected.sum()
    keep = np.cumsum(expected) - expected < 0.8
    expected = expected[keep] / expected[keep].sum()
    assert rng.probabilities == pytest.approx(
        dict(zip(np.array([3, 2, 1])[keep], expected))
    )


def test_presence_penalty_is_once_per_token_and_applies_before_greedy():
    logits = np.array([0.0, 3.0, 2.5])
    assert (
        sample_token(
            logits, temperature=0, presence_penalty=1, previous_tokens=[1, 1, 1]
        )
        == 2
    )
    rng = CaptureDistribution()
    sample_token(
        logits, presence_penalty=1, previous_tokens=np.array([1, 1, 1]), rng=rng
    )
    expected = np.exp([0, 2, 2.5])
    expected /= expected.sum()
    assert rng.probabilities == pytest.approx(dict(enumerate(expected)))
    np.testing.assert_array_equal(logits, [0, 3, 2.5])


@pytest.mark.parametrize(
    "options",
    [
        dict(temperature=float("nan")),
        dict(temperature=-1),
        dict(top_p=float("nan")),
        dict(top_k=1.5),
        dict(presence_penalty=float("nan")),
    ],
)
def test_invalid_sampling_rejected_even_for_greedy(options):
    with pytest.raises(ValueError):
        sample_runner_token(object(), np.array([1, 2]), **options)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_top_k_uses_last_row_and_vocabulary_range(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    runner = AutoregressiveRunner()
    runner.device = wp.get_device(device)
    runner.dtype = wp.float16
    runner.config = {"vocab_size": 128}
    values = np.zeros((1, 3, 128), dtype=np.float16)
    values[0, 0, 11] = 100
    values[0, -1] = np.arange(128)
    logits = wp.array(values, dtype=wp.float16, device=device)
    actual, tokens = runner.read_top_k(logits, 20, token_start=10, token_stop=100)
    np.testing.assert_array_equal(tokens, np.arange(99, 79, -1))
    np.testing.assert_array_equal(actual, tokens)
