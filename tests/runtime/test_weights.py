# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import warp as wp

from tests.utilities import write_safetensors
from warp_nn.runtime.llama_encoder import merge_lora_adapter
from warp_nn.runtime.weights import merge_lora_weight


def test_merge_lora_weight_and_peft_adapter_cpu(tmp_path):
    weight = wp.zeros((2, 2), dtype=wp.float32, device="cpu")
    a_values = np.array([[1.0, 2.0]], dtype=np.float32)
    b_values = np.array([[3.0], [4.0]], dtype=np.float32)
    a = wp.array(a_values, device="cpu")
    b = wp.array(b_values, device="cpu")
    merge_lora_weight(weight, a, b, 0.5)
    np.testing.assert_allclose(weight.numpy(), 0.5 * b_values @ a_values)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"r": 1, "lora_alpha": 2}), encoding="utf-8"
    )
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    write_safetensors(
        adapter / "adapter_model.safetensors",
        {
            f"{prefix}.lora_A.weight": (
                "F32",
                a_values.shape,
                a_values.tobytes(),
            ),
            f"{prefix}.lora_B.weight": (
                "F32",
                b_values.shape,
                b_values.tobytes(),
            ),
        },
    )
    name = "model.layers.0.self_attn.q_proj.weight"
    weights = {name: wp.zeros((2, 2), dtype=wp.float32, device="cpu")}
    merge_lora_adapter(weights, adapter, wp.float32, wp.get_device("cpu"))
    np.testing.assert_allclose(weights[name].numpy(), 2.0 * b_values @ a_values)
