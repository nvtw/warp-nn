# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.stack import LoRATransformerStackPlan


@wp.kernel(enable_backward=False)
def _scaled_forward(
    x: wp.array2d(dtype=wp.bfloat16),
    scale: wp.float32,
    output: wp.array2d(dtype=wp.bfloat16),
):
    row, column = wp.tid()
    output[row, column] = wp.bfloat16(wp.float32(x[row, column]) * scale)


@wp.kernel(enable_backward=False)
def _scaled_backward(
    gradient: wp.array2d(dtype=wp.bfloat16),
    scale: wp.float32,
    output: wp.array2d(dtype=wp.float32),
):
    row, column = wp.tid()
    output[row, column] = wp.float32(gradient[row, column]) * scale


class _Adapters:
    def __init__(self):
        self.zeros = 0
        self.steps = 0

    def zero_grad(self):
        self.zeros += 1

    def step(self):
        self.steps += 1


class _Block:
    def __init__(self, adapters, scale, device):
        self.adapters = adapters
        self.device = wp.get_device(device)
        self.dtype = wp.bfloat16
        self.rows = 3
        self.hidden = 4
        self.scale = float(scale)
        self.output = wp.empty(
            (self.rows, self.hidden), dtype=self.dtype, device=self.device
        )
        self.input_grad = wp.empty(
            (self.rows, self.hidden), dtype=wp.float32, device=self.device
        )

    def forward(self, x, lengths, positions=None, cosine=None, sine=None):
        del lengths, positions, cosine, sine
        wp.launch(
            _scaled_forward,
            dim=self.output.shape,
            inputs=[x, self.scale],
            outputs=[self.output],
            device=self.device,
        )
        return self.output

    def backward(
        self,
        x,
        lengths,
        grad_output,
        positions=None,
        cosine=None,
        sine=None,
        *,
        accumulate=False,
    ):
        del x, lengths, positions, cosine, sine, accumulate
        wp.launch(
            _scaled_backward,
            dim=self.input_grad.shape,
            inputs=[grad_output, self.scale],
            outputs=[self.input_grad],
            device=self.device,
        )
        return self.input_grad


def _fixture(device):
    adapters = _Adapters()
    blocks = (_Block(adapters, 2.0, device), _Block(adapters, 3.0, device))
    stack = LoRATransformerStackPlan(blocks)
    values = np.arange(12, dtype=np.float32).reshape(3, 4) / 8.0
    x = wp.array(values, dtype=wp.bfloat16, device=device)
    gradient = wp.ones((3, 4), dtype=wp.bfloat16, device=device)
    lengths = wp.array([3], dtype=wp.int32, device=device)
    return adapters, stack, x, gradient, lengths


def test_transformer_stack_chains_forward_backward_and_optimizer_cpu():
    adapters, stack, x, gradient, lengths = _fixture("cpu")
    boundary_pointer = stack.boundary_grad.ptr

    output = stack.forward(x, lengths)
    input_gradient = stack.backward(x, lengths, gradient)

    np.testing.assert_allclose(output.numpy(), 6.0 * x.numpy())
    np.testing.assert_allclose(input_gradient.numpy(), np.full((3, 4), 6.0))
    assert stack.boundary_grad.ptr == boundary_pointer
    assert stack.output is stack.blocks[-1].output
    stack.zero_grad()
    stack.step()
    assert (adapters.zeros, adapters.steps) == (1, 1)


def test_transformer_stack_rejects_incompatible_or_reused_blocks():
    adapters = _Adapters()
    block = _Block(adapters, 2.0, "cpu")
    with pytest.raises(ValueError, match="cannot be reused"):
        LoRATransformerStackPlan([block, block])
    with pytest.raises(ValueError, match="adapter collection"):
        LoRATransformerStackPlan([block, _Block(_Adapters(), 3.0, "cpu")])


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_transformer_stack_cuda_graph_replay():
    _, stack, x, gradient, lengths = _fixture(CUDA_DEVICES[0])
    stack.forward(x, lengths)
    stack.backward(x, lengths, gradient)
    expected = stack.output.numpy().copy(), stack.blocks[0].input_grad.numpy().copy()

    wp.capture_begin(device=stack.device)
    try:
        stack.forward(x, lengths)
        stack.backward(x, lengths, gradient)
        graph = wp.capture_end(device=stack.device)
    except Exception:
        wp.capture_end(device=stack.device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)

    np.testing.assert_array_equal(stack.output.numpy(), expected[0])
    np.testing.assert_array_equal(stack.blocks[0].input_grad.numpy(), expected[1])
