# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from tests.training._model_validation import assert_adapter_directional_gradients
from warp_nn.training.adapters import LoRAAdapterCollection, LoRAAdapterConfig
from warp_nn.training.gqa import GQALoRAAttentionPlan
from warp_nn.training.mlp import LoRASwiGLUPlan
from warp_nn.training.qk import QKTransformPlan
from warp_nn.training.qwen import (
    QwenLoRATransformerBlockPlan,
    build_qwen_lora_training_plan,
)


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


def _model_fixture(device):
    rng = np.random.default_rng(211)
    dtype = wp.bfloat16
    batch, sequence, hidden, heads, kv_heads, head_size = 1, 2, 8, 2, 1, 4
    key_heads, value_heads, state_size, kernel_size = 1, 2, 2, 3
    intermediate, vocabulary = 12, 19
    config = {
        "hidden_size": hidden,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_size,
        "linear_num_key_heads": key_heads,
        "linear_num_value_heads": value_heads,
        "linear_key_head_dim": state_size,
        "linear_value_head_dim": state_size,
        "linear_conv_kernel_dim": kernel_size,
        "rms_norm_eps": 1.0e-6,
        "rope_parameters": {"partial_rotary_factor": 1.0},
    }
    shapes = {
        "model.language_model.embed_tokens.weight": (vocabulary, hidden),
        "model.language_model.norm.weight": (hidden,),
        "lm_head.weight": (vocabulary, hidden),
    }
    for index in range(2):
        prefix = f"model.language_model.layers.{index}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (hidden,),
                prefix + "post_attention_layernorm.weight": (hidden,),
                prefix + "mlp.gate_proj.weight": (intermediate, hidden),
                prefix + "mlp.up_proj.weight": (intermediate, hidden),
                prefix + "mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    linear = "model.language_model.layers.0.linear_attn."
    conv_width = 2 * key_heads * state_size + value_heads * state_size
    shapes.update(
        {
            linear + "in_proj_qkv.weight": (conv_width, hidden),
            linear + "in_proj_z.weight": (value_heads * state_size, hidden),
            linear + "in_proj_a.weight": (value_heads, hidden),
            linear + "in_proj_b.weight": (value_heads, hidden),
            linear + "conv1d.weight": (conv_width, 1, kernel_size),
            linear + "A_log": (value_heads,),
            linear + "dt_bias": (value_heads,),
            linear + "norm.weight": (state_size,),
            linear + "out_proj.weight": (hidden, value_heads * state_size),
        }
    )
    attention = "model.language_model.layers.1.self_attn."
    shapes.update(
        {
            attention + "q_proj.weight": (2 * heads * head_size, hidden),
            attention + "k_proj.weight": (kv_heads * head_size, hidden),
            attention + "v_proj.weight": (kv_heads * head_size, hidden),
            attention + "q_norm.weight": (head_size,),
            attention + "k_norm.weight": (head_size,),
            attention + "o_proj.weight": (hidden, heads * head_size),
        }
    )
    weights = {
        name: wp.array(
            (
                rng.normal(1.0, 0.02, shape)
                if len(shape) == 1 and "A_log" not in name and "dt_bias" not in name
                else rng.normal(0.0, 0.1, shape)
            ).astype(np.float32),
            dtype=dtype,
            device=device,
        )
        for name, shape in shapes.items()
    }
    for name in tuple(weights):
        if (
            "layernorm.weight" in name
            or name.endswith(("q_norm.weight", "k_norm.weight", "conv1d.weight"))
            or name.endswith(("A_log", "dt_bias", "linear_attn.norm.weight"))
            or name == "model.language_model.norm.weight"
        ):
            weights[name] = wp.array(
                weights[name].numpy(), dtype=wp.float32, device=device
            )
    model = build_qwen_lora_training_plan(
        config,
        weights,
        batch=batch,
        sequence=sequence,
        adapter_config=LoRAAdapterConfig(rank=2, alpha=4.0),
        centered_norm_scales=False,
        gguf_layout=True,
        ssm_a_is_decay=True,
        seed=31,
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


def test_qwen_model_builder_runs_full_training_step_cpu():
    model, input_ids, targets, lengths, positions, cosine, sine = _model_fixture("cpu")
    inputs = (input_ids, targets, lengths, positions, cosine, sine)
    initial = float(model.forward(*inputs).numpy()[0])
    assert_adapter_directional_gradients(
        model,
        inputs,
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.1.self_attn.q_proj.weight",
            "model.language_model.layers.1.mlp.down_proj.weight",
        ),
    )
    for _ in range(20):
        model.train_step(*inputs)
    final = float(model.forward(*inputs).numpy()[0])
    assert final < initial
    assert int(model.adapters.optimizer.step_count.numpy()[0]) == 20
    assert len(model.stack.blocks) == 2
    assert all(
        np.isfinite(gradient.numpy()).all()
        for gradient in model.adapters.named_gradients.values()
    )


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_qwen_model_builder_full_step_cuda_graph_replay():
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
