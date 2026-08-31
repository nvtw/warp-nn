# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent full-chain checks shared by Qwen and Muse training tests."""

import numpy as np


def assert_adapter_directional_gradients(
    model,
    inputs,
    target_names,
    *,
    epsilon=0.1,
    rtol=0.35,
    atol=3.0e-3,
):
    """Compare composed analytical LoRA-B gradients with central differences."""
    model.adapters.zero_grad()
    model.forward(*inputs)
    model.backward(*inputs)
    rng = np.random.default_rng(307)
    for name in target_names:
        target = model.adapters.targets[name]
        original = np.asarray(target.lora_b.numpy(), dtype=np.float32)
        direction = rng.normal(size=original.shape).astype(np.float32)
        direction /= np.linalg.norm(direction)
        analytic = float(np.sum(target.plan.grad_b.numpy() * direction))

        target.lora_b.assign(original + epsilon * direction)
        positive = float(model.forward(*inputs).numpy()[0])
        target.lora_b.assign(original - epsilon * direction)
        negative = float(model.forward(*inputs).numpy()[0])
        target.lora_b.assign(original)
        numerical = (positive - negative) / (2.0 * epsilon)

        np.testing.assert_allclose(
            analytic,
            numerical,
            rtol=rtol,
            atol=atol,
            err_msg=f"full-chain gradient mismatch for {name}",
        )
