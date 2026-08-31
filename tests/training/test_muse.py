# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from tests.training._model_validation import assert_adapter_directional_gradients
from warp_nn.training.adapters import LoRAAdapterCollection, LoRAAdapterConfig
from warp_nn.training.gqa import GQALoRAAttentionPlan
from warp_nn.training.mlp import LoRASwiGLUPlan
from warp_nn.training.muse import (
    MuseLoRATransformerBlockPlan,
    build_muse_lora_training_plan,
)
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


def _model_fixture(device):
    rng = np.random.default_rng(191)
    dtype = wp.bfloat16
    batch, sequence, hidden, heads, kv_heads, head_size = 1, 2, 8, 2, 1, 4
    intermediate, vocabulary = 12, 19
    config = {
        "hidden_size": hidden,
        "num_hidden_layers": 1,
        "layer_types": ["sliding_attention"],
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_size,
        "sliding_window": 8,
        "qk_scale_factor": 1.25,
        "rms_norm_eps": 1.0e-6,
        "post_norm_eps": 1.0e-5,
        "output_multiplier": 0.75,
        "final_logit_softcapping": 4.0,
    }
    shapes = {
        "model.language_model.embed_tokens.weight": (vocabulary, hidden),
        "model.language_model.norm.weight": (hidden,),
        "lm_head.weight": (vocabulary, hidden),
        "model.language_model.layers.0.input_layernorm.weight": (hidden,),
        "model.language_model.layers.0.post_attention_layernorm.weight": (hidden,),
        "model.language_model.layers.0.pre_feedforward_layernorm.weight": (hidden,),
        "model.language_model.layers.0.post_feedforward_layernorm.weight": (hidden,),
        "model.language_model.layers.0.self_attn.q_proj.weight": (
            heads * head_size,
            hidden,
        ),
        "model.language_model.layers.0.self_attn.k_proj.weight": (
            kv_heads * head_size,
            hidden,
        ),
        "model.language_model.layers.0.self_attn.v_proj.weight": (
            kv_heads * head_size,
            hidden,
        ),
        "model.language_model.layers.0.self_attn.gate_proj.weight": (
            heads * head_size,
            hidden,
        ),
        "model.language_model.layers.0.self_attn.o_proj.weight": (
            hidden,
            heads * head_size,
        ),
        "model.language_model.layers.0.mlp.gate_proj.weight": (
            intermediate,
            hidden,
        ),
        "model.language_model.layers.0.mlp.up_proj.weight": (
            intermediate,
            hidden,
        ),
        "model.language_model.layers.0.mlp.down_proj.weight": (
            hidden,
            intermediate,
        ),
    }
    weights = {
        name: wp.array(
            (
                rng.normal(1.0, 0.02, shape)
                if len(shape) == 1
                else rng.normal(0.0, 0.12, shape)
            ).astype(np.float32),
            dtype=dtype,
            device=device,
        )
        for name, shape in shapes.items()
    }
    model = build_muse_lora_training_plan(
        config,
        weights,
        batch=batch,
        sequence=sequence,
        adapter_config=LoRAAdapterConfig(rank=2, alpha=4.0),
        centered_norm_scales=False,
        seed=29,
        optimizer_options={"learning_rate": 0.03},
        use_cublas=False,
    )
    input_ids = wp.array([1, 3], dtype=wp.int32, device=device)
    targets = wp.array([3, 5], dtype=wp.int32, device=device)
    lengths = wp.array([sequence], dtype=wp.int32, device=device)
    positions = wp.array([[0, 1]], dtype=wp.int64, device=device)
    cosine = wp.ones((sequence, head_size // 2), dtype=dtype, device=device)
    sine = wp.zeros_like(cosine)
    return model, input_ids, targets, lengths, positions, cosine, sine


def test_muse_model_builder_runs_full_training_step_cpu():
    model, input_ids, targets, lengths, positions, cosine, sine = _model_fixture("cpu")
    inputs = (input_ids, targets, lengths, positions, cosine, sine)
    initial = float(model.forward(*inputs).numpy()[0])
    assert_adapter_directional_gradients(
        model,
        inputs,
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
        ),
    )
    for _ in range(20):
        model.train_step(*inputs)
    final = float(model.forward(*inputs).numpy()[0])
    assert final < initial
    assert int(model.adapters.optimizer.step_count.numpy()[0]) == 20
    assert all(
        np.isfinite(gradient.numpy()).all()
        for gradient in model.adapters.named_gradients.values()
    )


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_muse_model_builder_full_step_cuda_graph_replay():
    model, input_ids, targets, lengths, positions, cosine, sine = _model_fixture(
        CUDA_DEVICES[0]
    )
    inputs = (input_ids, targets, lengths, positions, cosine, sine)
    model.train_step(*inputs)
    initial = float(model.forward(*inputs).numpy()[0])
    wp.capture_begin(device=model.device)
    try:
        model.train_step(*inputs)
        graph = wp.capture_end(device=model.device)
    except Exception:
        wp.capture_end(device=model.device)
        raise
    before = int(model.adapters.optimizer.step_count.numpy()[0])
    for _ in range(20):
        wp.capture_launch(graph)
    final = float(model.forward(*inputs).numpy()[0])
    assert int(model.adapters.optimizer.step_count.numpy()[0]) == before + 20
    assert final < initial
    assert np.isfinite(model.output.loss.numpy()).all()
