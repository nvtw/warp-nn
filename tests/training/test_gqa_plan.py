# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.adapters import LoRAAdapterCollection, LoRAAdapterConfig
from warp_nn.training.gqa import GQALoRAAttentionPlan
from warp_nn.training.qk import QKTransformPlan


def _fixture(device):
    dtype = wp.bfloat16
    rng = np.random.default_rng(41)
    shapes = {"q": (8, 8), "k": (4, 8), "v": (4, 8), "o": (8, 8)}
    weights = {
        name: wp.array(
            rng.normal(0.0, 0.2, shape).astype(np.float32),
            dtype=dtype,
            device=device,
        )
        for name, shape in shapes.items()
    }
    adapters = LoRAAdapterCollection(
        weights,
        rows=3,
        configs=LoRAAdapterConfig(rank=2, alpha=4.0),
        seed=7,
        use_cublas=False,
    )
    plan = GQALoRAAttentionPlan(
        adapters,
        query="q",
        key="k",
        value="v",
        output="o",
        batch=1,
        sequence=3,
        query_heads=2,
        kv_heads=1,
        head_size=4,
        window=2,
    )
    x = wp.array(
        rng.normal(0.0, 0.2, (3, 8)).astype(np.float32),
        dtype=dtype,
        device=device,
    )
    lengths = wp.array(np.array([3], dtype=np.int32), dtype=wp.int32, device=device)
    grad_output = wp.array(
        rng.normal(0.0, 0.2, (3, 8)).astype(np.float32),
        dtype=dtype,
        device=device,
    )
    return adapters, plan, x, lengths, grad_output


def test_gqa_lora_attention_fixed_buffers_and_accumulation_cpu():
    adapters, plan, x, lengths, grad_output = _fixture("cpu")
    arrays = (
        plan.output,
        plan.query,
        plan.key,
        plan.value,
        plan.core,
        plan.merged,
        plan.lse,
        plan.workspace,
        plan.query_grad,
        plan.key_grad,
        plan.value_grad,
        plan.input_grad,
        *adapters.named_gradients.values(),
    )
    pointers = tuple(array.ptr for array in arrays)

    output = plan.forward(x, lengths)
    input_grad = plan.backward(x, lengths, grad_output)
    first_gradients = {
        name: gradient.numpy().copy()
        for name, gradient in adapters.named_gradients.items()
    }
    plan.backward(x, lengths, grad_output, accumulate=True)

    assert output is plan.output
    assert input_grad is plan.input_grad
    assert tuple(array.ptr for array in arrays) == pointers
    assert np.isfinite(output.numpy()).all()
    assert np.isfinite(input_grad.numpy()).all()
    assert any(np.any(values) for values in first_gradients.values())
    for name, gradient in adapters.named_gradients.items():
        np.testing.assert_allclose(gradient.numpy(), 2.0 * first_gradients[name])


def test_gqa_lora_attention_composes_qk_norm_and_rope_cpu():
    adapters, _, x, lengths, grad_output = _fixture("cpu")
    query_transform = QKTransformPlan(1, 2, 3, 4, wp.bfloat16, device="cpu")
    key_transform = QKTransformPlan(1, 1, 3, 4, wp.bfloat16, device="cpu")
    unit = wp.ones(4, dtype=wp.bfloat16, device="cpu")
    plan = GQALoRAAttentionPlan(
        adapters,
        query="q",
        key="k",
        value="v",
        output="o",
        batch=1,
        sequence=3,
        query_heads=2,
        kv_heads=1,
        head_size=4,
        window=2,
        query_transform=query_transform,
        key_transform=key_transform,
        query_norm_weight=unit,
        key_norm_weight=unit,
    )
    positions = wp.array([[0, 1, 2]], dtype=wp.int64, device="cpu")
    cosine = wp.ones((3, 2), dtype=wp.bfloat16, device="cpu")
    sine = wp.zeros((3, 2), dtype=wp.bfloat16, device="cpu")

    plan.forward(x, lengths, positions, cosine, sine)
    plan.backward(x, lengths, grad_output, positions, cosine, sine)

    assert np.isfinite(plan.output.numpy()).all()
    assert np.isfinite(plan.input_grad.numpy()).all()
    assert np.isfinite(query_transform.input_grad.numpy()).all()
    assert np.isfinite(key_transform.input_grad.numpy()).all()


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_gqa_lora_attention_cuda_graph_replay():
    _, plan, x, lengths, grad_output = _fixture(CUDA_DEVICES[0])
    plan.forward(x, lengths)
    plan.backward(x, lengths, grad_output)
    wp.synchronize_device(plan.device)
    references = tuple(
        array.numpy().copy()
        for array in (plan.output, plan.input_grad, plan.query_grad, plan.key_grad)
    )

    wp.capture_begin(device=plan.device)
    try:
        plan.forward(x, lengths)
        plan.backward(x, lengths, grad_output)
        graph = wp.capture_end(device=plan.device)
    except Exception:
        wp.capture_end(device=plan.device)
        raise
    wp.capture_launch(graph)
    wp.capture_launch(graph)

    for array, reference in zip(
        (plan.output, plan.input_grad, plan.query_grad, plan.key_grad), references
    ):
        np.testing.assert_array_equal(array.numpy(), reference)
