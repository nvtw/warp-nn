# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.adapters import LoRAAdapterCollection, LoRAAdapterConfig
from warp_nn.training.gqa import GQALoRAAttentionPlan
from warp_nn.training.mlp import LoRASwiGLUPlan
from warp_nn.training.qk import QKTransformPlan
from warp_nn.training.qwen import QwenLoRATransformerBlockPlan


def _fixture(device):
    rng = np.random.default_rng(109)
    dtype = wp.bfloat16
    rows, hidden, head_size, intermediate = 3, 16, 8, 24
    shapes = {
        "q": (32, hidden),
        "k": (head_size, hidden),
        "v": (head_size, hidden),
        "o": (hidden, hidden),
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
        seed=13,
        use_cublas=False,
    )
    query_transform = QKTransformPlan(1, 2, rows, head_size, dtype, device=device)
    key_transform = QKTransformPlan(1, 1, rows, head_size, dtype, device=device)
    query_norm = wp.array(rng.normal(1.0, 0.03, head_size), dtype=dtype, device=device)
    key_norm = wp.array(rng.normal(1.0, 0.03, head_size), dtype=dtype, device=device)
    attention = GQALoRAAttentionPlan(
        adapters,
        query="q",
        key="k",
        value="v",
        output="o",
        packed_query_gate=True,
        batch=1,
        sequence=rows,
        query_heads=2,
        kv_heads=1,
        head_size=head_size,
        query_transform=query_transform,
        key_transform=key_transform,
        query_norm_weight=query_norm,
        key_norm_weight=key_norm,
    )
    mlp = LoRASwiGLUPlan(adapters, gate="mlp_gate", up="mlp_up", down="mlp_down")
    input_norm = wp.array(rng.normal(1.0, 0.03, hidden), dtype=dtype, device=device)
    post_norm = wp.array(rng.normal(1.0, 0.03, hidden), dtype=dtype, device=device)
    block = QwenLoRATransformerBlockPlan(
        attention,
        mlp,
        input_norm_weight=input_norm,
        post_attention_norm_weight=post_norm,
        epsilon=1.0e-6,
        centered_norm_scales=False,
    )
    x = wp.array(rng.normal(0.0, 0.2, (rows, hidden)), dtype=dtype, device=device)
    gradient = wp.array(
        rng.normal(0.0, 0.2, (rows, hidden)), dtype=dtype, device=device
    )
    lengths = wp.array([rows], dtype=wp.int32, device=device)
    positions = wp.array(np.arange(rows, dtype=np.int64)[None], device=device)
    angles = rng.normal(0.0, 0.2, (rows, head_size // 2)).astype(np.float32)
    cosine = wp.array(np.cos(angles), dtype=dtype, device=device)
    sine = wp.array(np.sin(angles), dtype=dtype, device=device)
    return adapters, block, x, gradient, lengths, positions, cosine, sine


def test_qwen_lora_block_accumulation_and_residual_identity_cpu():
    adapters, block, x, gradient, lengths, positions, cosine, sine = _fixture("cpu")
    pointers = tuple(
        array.ptr
        for array in (
            block.output,
            block.input_grad,
            *adapters.named_gradients.values(),
        )
    )
    block.forward(x, lengths, positions, cosine, sine)
    block.backward(x, lengths, gradient, positions, cosine, sine)
    first = {
        name: value.numpy().copy() for name, value in adapters.named_gradients.items()
    }
    block.backward(x, lengths, gradient, positions, cosine, sine, accumulate=True)
    assert (
        tuple(
            array.ptr
            for array in (
                block.output,
                block.input_grad,
                *adapters.named_gradients.values(),
            )
        )
        == pointers
    )
    for name, value in adapters.named_gradients.items():
        np.testing.assert_allclose(value.numpy(), 2.0 * first[name])

    for target in adapters.targets.values():
        target.weight.zero_()
        target.lora_a.zero_()
        target.lora_b.zero_()
    np.testing.assert_array_equal(
        block.forward(x, lengths, positions, cosine, sine).numpy(), x.numpy()
    )
    np.testing.assert_array_equal(
        block.backward(x, lengths, gradient, positions, cosine, sine).numpy(),
        gradient.numpy(),
    )


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_qwen_lora_block_cuda_graph_replay():
    _, block, x, gradient, lengths, positions, cosine, sine = _fixture(CUDA_DEVICES[0])
    block.forward(x, lengths, positions, cosine, sine)
    block.backward(x, lengths, gradient, positions, cosine, sine)
    references = block.output.numpy().copy(), block.input_grad.numpy().copy()
    wp.capture_begin(device=block.device)
    try:
        block.forward(x, lengths, positions, cosine, sine)
        block.backward(x, lengths, gradient, positions, cosine, sine)
        graph = wp.capture_end(device=block.device)
    except Exception:
        wp.capture_end(device=block.device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)
    np.testing.assert_array_equal(block.output.numpy(), references[0])
    np.testing.assert_array_equal(block.input_grad.numpy(), references[1])
