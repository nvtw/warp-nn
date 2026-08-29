# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.adapters import LoRAAdapterCollection, LoRAAdapterConfig
from warp_nn.training.gated_delta import GatedDeltaInputPlan
from warp_nn.training.gated_delta_rule import GatedDeltaRulePlan
from warp_nn.training.gated_norm import GatedRMSNormPlan
from warp_nn.training.linear_attention import QwenGatedDeltaLoRAAttentionPlan
from warp_nn.training.mlp import LoRASwiGLUPlan
from warp_nn.training.qwen import QwenLoRATransformerBlockPlan


def _fixture(device, *, sequence=3, hidden=4, key_size=2, value_size=2):
    rng = np.random.default_rng(173)
    dtype = wp.bfloat16
    batch, key_heads, value_heads, kernel_size = 1, 1, 2, 3
    rows = batch * sequence
    conv_width = 2 * key_heads * key_size + value_heads * value_size
    value_width = value_heads * value_size
    intermediate = hidden + hidden // 2
    shapes = {
        "qkv": (conv_width, hidden),
        "z": (value_width, hidden),
        "a": (value_heads, hidden),
        "b": (value_heads, hidden),
        "o": (hidden, value_width),
        "mlp_gate": (intermediate, hidden),
        "mlp_up": (intermediate, hidden),
        "mlp_down": (hidden, intermediate),
    }
    weights = {
        name: wp.array(
            rng.normal(0.0, 0.12, shape).astype(np.float32),
            dtype=dtype,
            device=device,
        )
        for name, shape in shapes.items()
    }
    adapters = LoRAAdapterCollection(
        weights,
        rows=rows,
        configs=LoRAAdapterConfig(rank=2, alpha=4.0),
        seed=17,
        use_cublas=False,
    )
    inputs = GatedDeltaInputPlan(
        batch,
        sequence,
        key_heads,
        value_heads,
        key_size,
        value_size,
        kernel_size,
        dtype,
        device=device,
    )
    rule = GatedDeltaRulePlan(
        batch,
        sequence,
        key_heads,
        value_heads,
        key_size,
        value_size,
        dtype,
        device=device,
    )
    gated_norm = GatedRMSNormPlan(
        batch * value_heads * sequence,
        value_size,
        dtype,
        epsilon=1.0e-6,
        device=device,
    )
    conv_weight = wp.array(
        rng.normal(0.0, 0.1, (conv_width, kernel_size)),
        dtype=dtype,
        device=device,
    )
    conv_state = wp.array(
        rng.normal(0.0, 0.05, (batch, conv_width, kernel_size - 1)),
        dtype=dtype,
        device=device,
    )
    a_log = wp.array(rng.normal(-2.0, 0.1, value_heads), dtype=dtype, device=device)
    dt_bias = wp.array(rng.normal(0.0, 0.1, value_heads), dtype=dtype, device=device)
    recurrent_state = wp.array(
        rng.normal(0.0, 0.03, (batch, value_heads, key_size, value_size)),
        dtype=wp.float32,
        device=device,
    )
    norm_weight = wp.array(
        rng.normal(1.0, 0.03, value_size), dtype=dtype, device=device
    )
    attention = QwenGatedDeltaLoRAAttentionPlan(
        adapters,
        qkv="qkv",
        gate="z",
        decay="a",
        beta="b",
        output="o",
        inputs=inputs,
        rule=rule,
        gated_norm=gated_norm,
        conv_weight=conv_weight,
        conv_state=conv_state,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        norm_weight=norm_weight,
    )
    mlp = LoRASwiGLUPlan(adapters, gate="mlp_gate", up="mlp_up", down="mlp_down")
    block = QwenLoRATransformerBlockPlan(
        attention,
        mlp,
        input_norm_weight=wp.array(
            rng.normal(1.0, 0.03, hidden), dtype=dtype, device=device
        ),
        post_attention_norm_weight=wp.array(
            rng.normal(1.0, 0.03, hidden), dtype=dtype, device=device
        ),
        epsilon=1.0e-6,
        centered_norm_scales=False,
    )
    x = wp.array(rng.normal(0.0, 0.2, (rows, hidden)), dtype=dtype, device=device)
    gradient = wp.array(
        rng.normal(0.0, 0.2, (rows, hidden)), dtype=dtype, device=device
    )
    lengths = wp.array([sequence], dtype=wp.int32, device=device)
    return adapters, attention, block, x, gradient, lengths


def test_qwen_gated_delta_lora_block_backward_and_accumulation_cpu():
    adapters, attention, block, x, gradient, lengths = _fixture("cpu")
    pointers = tuple(
        array.ptr
        for array in (
            block.output,
            block.input_grad,
            attention.conv_state_grad,
            attention.recurrent_state_grad,
            *adapters.named_gradients.values(),
        )
    )
    output = block.forward(x, lengths)
    input_grad = block.backward(x, lengths, gradient)
    assert np.isfinite(output.numpy()).all()
    assert np.isfinite(input_grad.numpy()).all()
    assert np.linalg.norm(attention.conv_state_grad.numpy()) > 0.0
    assert np.linalg.norm(attention.recurrent_state_grad.numpy()) > 0.0
    first = {
        name: value.numpy().copy() for name, value in adapters.named_gradients.items()
    }
    block.backward(x, lengths, gradient, accumulate=True)
    for name, value in adapters.named_gradients.items():
        np.testing.assert_allclose(value.numpy(), 2.0 * first[name])
    assert pointers == tuple(
        array.ptr
        for array in (
            block.output,
            block.input_grad,
            attention.conv_state_grad,
            attention.recurrent_state_grad,
            *adapters.named_gradients.values(),
        )
    )

    for target in adapters.targets.values():
        target.weight.zero_()
        target.lora_a.zero_()
        target.lora_b.zero_()
    np.testing.assert_array_equal(block.forward(x, lengths).numpy(), x.numpy())
    np.testing.assert_array_equal(
        block.backward(x, lengths, gradient).numpy(), gradient.numpy()
    )


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_qwen_gated_delta_lora_block_cuda_graph_replay():
    device = CUDA_DEVICES[0]
    if device.arch < 80:
        pytest.skip("BF16 tensor cores require SM80")
    _, _, block, x, gradient, lengths = _fixture(
        device, sequence=16, hidden=128, key_size=128, value_size=128
    )
    block.forward(x, lengths)
    block.backward(x, lengths, gradient)
    references = block.output.numpy().copy(), block.input_grad.numpy().copy()
    wp.capture_begin(device=device)
    try:
        block.forward(x, lengths)
        block.backward(x, lengths, gradient)
        graph = wp.capture_end(device=device)
    except Exception:
        wp.capture_end(device=device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)
    np.testing.assert_array_equal(block.output.numpy(), references[0])
    np.testing.assert_allclose(
        block.input_grad.numpy(), references[1], rtol=1.0e-5, atol=5.0e-6
    )
