# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.adapters import LoRAAdapterCollection, LoRAAdapterConfig
from warp_nn.training.gqa import GQALoRAAttentionPlan
from warp_nn.training.mlp import LoRASwiGLUPlan
from warp_nn.training.muse import MuseLoRATransformerBlockPlan
from warp_nn.training.qk import QKTransformPlan


def _fixture(device):
    rng = np.random.default_rng(83)
    dtype = wp.bfloat16
    rows, hidden, head_size, intermediate = 3, 16, 8, 24
    shapes = {
        "q": (16, 16),
        "k": (8, 16),
        "v": (8, 16),
        "attention_gate": (16, 16),
        "o": (16, 16),
        "mlp_gate": (intermediate, hidden),
        "mlp_up": (intermediate, hidden),
        "mlp_down": (hidden, intermediate),
    }
    weights = {
        name: wp.array(
            rng.normal(0.0, 0.15, shape).astype(np.float32),
            dtype=dtype,
            device=device,
        )
        for name, shape in shapes.items()
    }
    adapters = LoRAAdapterCollection(
        weights,
        rows=rows,
        configs=LoRAAdapterConfig(rank=2, alpha=4.0),
        seed=5,
        use_cublas=False,
    )
    query_transform = QKTransformPlan(
        1, 2, rows, head_size, dtype, scale=3.87, device=device
    )
    key_transform = QKTransformPlan(1, 1, rows, head_size, dtype, device=device)
    unit_head = wp.ones(head_size, dtype=dtype, device=device)
    attention = GQALoRAAttentionPlan(
        adapters,
        query="q",
        key="k",
        value="v",
        output="o",
        gate="attention_gate",
        batch=1,
        sequence=rows,
        query_heads=2,
        kv_heads=1,
        head_size=head_size,
        window=2,
        query_transform=query_transform,
        key_transform=key_transform,
        query_norm_weight=unit_head,
        key_norm_weight=unit_head,
    )
    mlp = LoRASwiGLUPlan(adapters, gate="mlp_gate", up="mlp_up", down="mlp_down")
    norm_weights = tuple(
        wp.array(
            rng.normal(0.0, 0.03, hidden).astype(np.float32),
            dtype=dtype,
            device=device,
        )
        for _ in range(4)
    )
    block = MuseLoRATransformerBlockPlan(
        attention,
        mlp,
        input_norm_weight=norm_weights[0],
        post_attention_norm_weight=norm_weights[1],
        feedforward_norm_weight=norm_weights[2],
        post_feedforward_norm_weight=norm_weights[3],
        rms_epsilon=1.0e-6,
        post_epsilon=1.0e-5,
        centered_norm_scales=True,
    )
    x = wp.array(
        rng.normal(0.0, 0.2, (rows, hidden)).astype(np.float32),
        dtype=dtype,
        device=device,
    )
    grad_output = wp.array(
        rng.normal(0.0, 0.2, (rows, hidden)).astype(np.float32),
        dtype=dtype,
        device=device,
    )
    lengths = wp.array(np.array([rows], dtype=np.int32), device=device)
    positions = wp.array(np.arange(rows, dtype=np.int64)[None], device=device)
    angles = rng.normal(0.0, 0.2, (rows, head_size // 2)).astype(np.float32)
    cosine = wp.array(np.cos(angles), dtype=dtype, device=device)
    sine = wp.array(np.sin(angles), dtype=dtype, device=device)
    return adapters, block, x, grad_output, lengths, positions, cosine, sine


def test_muse_lora_block_fixed_buffers_and_accumulation_cpu():
    adapters, block, x, grad_output, lengths, positions, cosine, sine = _fixture("cpu")
    pointers = tuple(
        array.ptr
        for array in (
            block.output,
            block.attention_residual,
            block.residual_grad,
            block.input_grad,
            *adapters.named_gradients.values(),
        )
    )
    block.forward(x, lengths, positions, cosine, sine)
    block.backward(x, lengths, grad_output, positions, cosine, sine)
    first = {
        name: gradient.numpy().copy()
        for name, gradient in adapters.named_gradients.items()
    }
    block.backward(x, lengths, grad_output, positions, cosine, sine, accumulate=True)

    assert np.isfinite(block.output.numpy()).all()
    assert np.isfinite(block.input_grad.numpy()).all()
    assert (
        tuple(
            array.ptr
            for array in (
                block.output,
                block.attention_residual,
                block.residual_grad,
                block.input_grad,
                *adapters.named_gradients.values(),
            )
        )
        == pointers
    )
    for name, gradient in adapters.named_gradients.items():
        np.testing.assert_allclose(gradient.numpy(), 2.0 * first[name])


def test_muse_lora_block_zero_projection_is_residual_identity_cpu():
    adapters, block, x, grad_output, lengths, positions, cosine, sine = _fixture("cpu")
    for target in adapters.targets.values():
        target.weight.zero_()
        target.lora_a.zero_()
        target.lora_b.zero_()

    output = block.forward(x, lengths, positions, cosine, sine)
    input_grad = block.backward(x, lengths, grad_output, positions, cosine, sine)

    np.testing.assert_array_equal(output.numpy(), x.numpy())
    np.testing.assert_array_equal(input_grad.numpy(), grad_output.numpy())


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_muse_lora_block_cuda_graph_replay():
    _, block, x, grad_output, lengths, positions, cosine, sine = _fixture(
        CUDA_DEVICES[0]
    )
    block.forward(x, lengths, positions, cosine, sine)
    block.backward(x, lengths, grad_output, positions, cosine, sine)
    references = block.output.numpy().copy(), block.input_grad.numpy().copy()

    wp.capture_begin(device=block.device)
    try:
        block.forward(x, lengths, positions, cosine, sine)
        block.backward(x, lengths, grad_output, positions, cosine, sine)
        graph = wp.capture_end(device=block.device)
    except Exception:
        wp.capture_end(device=block.device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)
    np.testing.assert_array_equal(block.output.numpy(), references[0])
    np.testing.assert_array_equal(block.input_grad.numpy(), references[1])
