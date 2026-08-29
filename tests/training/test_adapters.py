# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from warp_nn.training.adapters import LoRAAdapterCollection, LoRAAdapterConfig


def _array(rng, shape, dtype=wp.float16):
    return wp.array(
        rng.normal(size=shape).astype(np.float32), dtype=dtype, device="cpu"
    )


def _collection(weights, seed=31):
    configs = {
        "attention.q": LoRAAdapterConfig(2, alpha=4.0, init_std=0.05),
        "attention.v": LoRAAdapterConfig(1, init_std=0.03),
        "mlp.up": LoRAAdapterConfig(3, alpha=1.5, init_std=0.04),
    }
    rows = {"attention.q": 3, "attention.v": 2, "mlp.up": 4}
    return LoRAAdapterCollection(
        weights,
        rows,
        configs,
        seed=seed,
        optimizer_options={
            "learning_rate": 0.05,
            "beta1": 0.0,
            "beta2": 0.0,
            "epsilon": 1.0e-6,
        },
    )


def test_named_lora_adapters_deterministic_shapes_and_masters_cpu():
    rng = np.random.default_rng(7)
    weight_values = {
        "attention.q": rng.normal(size=(5, 4)).astype(np.float32),
        "attention.v": rng.normal(size=(3, 4)).astype(np.float32),
        "mlp.up": rng.normal(size=(7, 6)).astype(np.float32),
    }
    weights = {
        name: wp.array(values, dtype=wp.bfloat16, device="cpu")
        for name, values in weight_values.items()
    }
    duplicate_weights = {
        name: wp.array(values, dtype=wp.bfloat16, device="cpu")
        for name, values in reversed(tuple(weight_values.items()))
    }
    collection = _collection(weights)
    duplicate = _collection(duplicate_weights)

    assert tuple(collection.targets) == ("attention.q", "attention.v", "mlp.up")
    assert tuple(collection.configs) == tuple(collection.targets)
    assert all(
        collection.configs[name] is target.config
        for name, target in collection.targets.items()
    )
    assert collection.targets["attention.q"].lora_a.shape == (2, 4)
    assert collection.targets["attention.v"].lora_b.shape == (3, 1)
    assert collection.targets["mlp.up"].lora_a.shape == (3, 6)
    for name, parameter in collection.named_parameters.items():
        np.testing.assert_array_equal(
            parameter.numpy(), duplicate.named_parameters[name].numpy()
        )
        master = collection.named_masters[name]
        assert master.dtype == wp.float32
        assert master.shape == parameter.shape
        np.testing.assert_array_equal(
            master.numpy(), parameter.numpy().astype(np.float32)
        )
        if name.endswith("lora_B.weight"):
            np.testing.assert_array_equal(parameter.numpy(), 0.0)


def test_named_lora_adapter_lifecycle_keeps_pointers_and_base_frozen_cpu():
    rng = np.random.default_rng(13)
    weights = {
        "attention.q": _array(rng, (5, 4)),
        "attention.v": _array(rng, (3, 4)),
        "mlp.up": _array(rng, (7, 6)),
    }
    collection = _collection(weights)
    frozen = {name: weight.numpy().copy() for name, weight in weights.items()}
    pointer_groups = {
        category: {name: array.ptr for name, array in arrays.items()}
        for category, arrays in (
            ("parameters", collection.named_parameters),
            ("gradients", collection.named_gradients),
            ("masters", collection.named_masters),
        )
    }
    for name, target in collection.targets.items():
        x = _array(rng, (target.plan.rows, target.plan.in_features))
        grad_output = _array(rng, (target.plan.rows, target.plan.out_features))
        output = collection.forward(name, x)
        assert output.ptr == target.plan.output.ptr
        collection.backward(name, x, grad_output)
        assert np.isfinite(target.plan.grad_a.numpy()).all()
        assert np.isfinite(target.plan.grad_b.numpy()).all()

    before = {
        name: master.numpy().copy() for name, master in collection.named_masters.items()
    }
    collection.step()
    assert any(
        np.any(collection.named_masters[name].numpy() != value)
        for name, value in before.items()
    )
    collection.zero_grad()
    for gradient in collection.named_gradients.values():
        np.testing.assert_array_equal(gradient.numpy(), np.zeros(gradient.shape))
    for name, weight in weights.items():
        np.testing.assert_array_equal(weight.numpy(), frozen[name])
    for category, arrays in (
        ("parameters", collection.named_parameters),
        ("gradients", collection.named_gradients),
        ("masters", collection.named_masters),
    ):
        assert {name: array.ptr for name, array in arrays.items()} == pointer_groups[
            category
        ]


def test_named_lora_adapter_validation_cpu():
    weight = wp.zeros((3, 4), dtype=wp.float16, device="cpu")
    with pytest.raises(ValueError, match="exact target mapping"):
        LoRAAdapterCollection({"x": weight}, {"x": 2}, {"wrong": LoRAAdapterConfig(1)})
    with pytest.raises(TypeError, match="FP16 or BF16"):
        LoRAAdapterCollection(
            {"x": wp.zeros((3, 4), dtype=wp.float32, device="cpu")},
            2,
            LoRAAdapterConfig(1),
        )
    with pytest.raises(ValueError, match="rank"):
        LoRAAdapterConfig(0)
