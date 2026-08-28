# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from tests.utilities import is_device_available
from warp_nn.runtime.kernels import _get_mamba2_decode_kernel, _get_mamba2_prefill_kernel


def _decode_reference(x, b, c, dt, a_log, dt_bias, d, state, heads_per_group, step_min, step_max):
    step = np.logaddexp(dt.astype(np.float32) + dt_bias, 0.0)
    step = np.clip(step, step_min, step_max)
    output = np.empty_like(x)
    for head in range(x.shape[0]):
        group = head // heads_per_group
        decay = np.exp(-np.exp(a_log[head]) * step[head])
        for channel in range(x.shape[1]):
            row = head * x.shape[1] + channel
            state[row] = state[row] * decay + b[group].astype(np.float32) * step[head] * x[head, channel]
            output[head, channel] = state[row] @ c[group].astype(np.float32) + d[head] * x[head, channel]
    return output.astype(x.dtype), state


def test_mamba2_decode_matches_reference_and_captures():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(109)
    heads, head_dim, groups, state_size = 4, 5, 2, 7
    heads_per_group = heads // groups
    x = rng.normal(0.0, 0.2, (heads, head_dim)).astype(np.float16)
    b = rng.normal(0.0, 0.2, (groups, state_size)).astype(np.float16)
    c = rng.normal(0.0, 0.2, (groups, state_size)).astype(np.float16)
    dt = rng.normal(-1.0, 0.3, heads).astype(np.float16)
    a_log = rng.normal(-0.5, 0.2, heads).astype(np.float32)
    dt_bias = rng.normal(-1.0, 0.2, heads).astype(np.float32)
    d = rng.normal(0.0, 0.2, heads).astype(np.float32)
    initial = rng.normal(0.0, 0.1, (heads * head_dim, state_size)).astype(np.float32)
    expected, expected_state = _decode_reference(
        x, b, c, dt, a_log, dt_bias, d, initial.copy(), heads_per_group, 1.0e-4, 0.1
    )

    device = "cuda:0"
    arrays = [wp.array(value, device=device) for value in (x, b, c, dt, a_log, dt_bias, d)]
    state = wp.array(initial, device=device)
    output = wp.empty(x.shape, dtype=wp.float16, device=device)
    block_dim, kernel = _get_mamba2_decode_kernel(head_dim, state_size, heads_per_group, wp.float16)
    inputs = [*arrays, state, output, 1.0e-4, 0.1]
    wp.launch_tiled(kernel, dim=heads * head_dim, inputs=inputs, block_dim=block_dim, device=device)
    np.testing.assert_allclose(output.numpy(), expected, rtol=3.0e-3, atol=3.0e-3)
    np.testing.assert_allclose(state.numpy(), expected_state, rtol=2.0e-5, atol=2.0e-5)

    state.assign(initial)
    wp.capture_begin(device=device)
    wp.launch_tiled(kernel, dim=heads * head_dim, inputs=inputs, block_dim=block_dim, device=device)
    graph = wp.capture_end(device=device)
    wp.capture_launch(graph)
    np.testing.assert_allclose(output.numpy(), expected, rtol=3.0e-3, atol=3.0e-3)
    np.testing.assert_allclose(state.numpy(), expected_state, rtol=2.0e-5, atol=2.0e-5)


def test_mamba2_prefill_matches_recurrent_reference_and_captures():
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(812)
    sequence_length, heads, head_dim, groups, state_size = 5, 4, 7, 2, 9
    heads_per_group = heads // groups
    x = rng.normal(0.0, 0.2, (sequence_length, heads, head_dim)).astype(np.float16)
    b = rng.normal(0.0, 0.2, (sequence_length, groups, state_size)).astype(np.float16)
    c = rng.normal(0.0, 0.2, (sequence_length, groups, state_size)).astype(np.float16)
    dt = rng.normal(-1.0, 0.3, (sequence_length, heads)).astype(np.float16)
    a_log = rng.normal(-0.5, 0.2, heads).astype(np.float32)
    dt_bias = rng.normal(-1.0, 0.2, heads).astype(np.float32)
    d = rng.normal(0.0, 0.2, heads).astype(np.float32)
    initial = rng.normal(0.0, 0.1, (heads * head_dim, state_size)).astype(np.float32)

    expected_state = initial.copy()
    expected = []
    for token in range(sequence_length):
        token_output, expected_state = _decode_reference(
            x[token],
            b[token],
            c[token],
            dt[token],
            a_log,
            dt_bias,
            d,
            expected_state,
            heads_per_group,
            1.0e-4,
            0.1,
        )
        expected.append(token_output)
    expected = np.stack(expected).reshape(sequence_length, heads * head_dim)

    device = "cuda:0"
    arrays = [
        wp.array(value, device=device)
        for value in (
            x.reshape(sequence_length, -1),
            b.reshape(sequence_length, -1),
            c.reshape(sequence_length, -1),
            dt,
        )
    ]
    parameters = [wp.array(value, device=device) for value in (a_log, dt_bias, d)]
    state = wp.array(initial, device=device)
    output = wp.empty(expected.shape, dtype=wp.float16, device=device)
    channel_blocks, block_dim, kernel = _get_mamba2_prefill_kernel(
        head_dim, state_size, heads_per_group, wp.float16
    )
    inputs = [*arrays, *parameters, state, output, sequence_length, 1.0e-4, 0.1]
    launch_dim = heads * channel_blocks

    wp.launch_tiled(kernel, dim=launch_dim, inputs=inputs, block_dim=block_dim, device=device)
    np.testing.assert_allclose(output.numpy(), expected, rtol=3.0e-3, atol=3.0e-3)
    np.testing.assert_allclose(state.numpy(), expected_state, rtol=3.0e-5, atol=3.0e-5)

    state.assign(initial)
    wp.capture_begin(device=device)
    wp.launch_tiled(kernel, dim=launch_dim, inputs=inputs, block_dim=block_dim, device=device)
    graph = wp.capture_end(device=device)
    wp.capture_launch(graph)
    np.testing.assert_allclose(output.numpy(), expected, rtol=3.0e-3, atol=3.0e-3)
    np.testing.assert_allclose(state.numpy(), expected_state, rtol=3.0e-5, atol=3.0e-5)
