# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from tests.training.test_muse import _model_fixture
from warp_nn.training import LoRATrainer, SFTExample, prepare_sft_batch


def _trainer(device):
    model, _, _, _, _, cosine, sine = _model_fixture(device)
    trainer = LoRATrainer(model, cosine, sine)
    trainer.load_batch(
        prepare_sft_batch(
            [SFTExample(prompt=[1], response=[3, 5])],
            2,
            pad_token_id=0,
        )
    )
    return trainer


def test_trainer_direct_steps_and_adapter_restore_cpu(tmp_path):
    trainer = _trainer("cpu")
    initial = trainer.evaluate_loss()
    for _ in range(12):
        trainer.step()
    assert trainer.evaluate_loss() < initial
    collection = trainer.model.adapters
    pointers = {
        "masters": tuple(value.ptr for value in collection.named_masters.values()),
        "parameters": tuple(
            value.ptr for value in collection.named_parameters.values()
        ),
    }
    expected = {
        name: value.numpy().copy() for name, value in collection.named_masters.items()
    }
    checkpoint_path = tmp_path / "adapter.safetensors"
    trainer.save_adapters(
        checkpoint_path,
        base_identifier="test/muse",
        metadata={"purpose": "roundtrip"},
    )
    for _ in range(3):
        trainer.step()
    assert any(
        np.any(value.numpy() != expected[name])
        for name, value in collection.named_masters.items()
    )

    checkpoint = trainer.load_adapters(checkpoint_path)

    assert checkpoint.base_identifier == "test/muse"
    assert checkpoint.caller_metadata == {"purpose": "roundtrip"}
    assert int(collection.optimizer.step_count.numpy()[0]) == 0
    for name, master in collection.named_masters.items():
        np.testing.assert_array_equal(master.numpy(), expected[name])
        np.testing.assert_array_equal(
            collection.named_parameters[name].numpy().astype(np.float32),
            wp.array(expected[name], dtype=collection.named_parameters[name].dtype)
            .numpy()
            .astype(np.float32),
        )
    for moment in (
        *collection.optimizer.first_moments,
        *collection.optimizer.second_moments,
    ):
        np.testing.assert_array_equal(moment.numpy(), 0.0)
    assert pointers == {
        "masters": tuple(value.ptr for value in collection.named_masters.values()),
        "parameters": tuple(
            value.ptr for value in collection.named_parameters.values()
        ),
    }


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_trainer_captures_without_warmup_update_and_converges():
    trainer = _trainer(CUDA_DEVICES[0])
    initial = trainer.evaluate_loss()
    trainer.capture()
    optimizer = trainer.model.adapters.optimizer
    assert int(optimizer.step_count.numpy()[0]) == 0

    for _ in range(20):
        trainer.step()
    final = trainer.evaluate_loss()

    assert int(optimizer.step_count.numpy()[0]) == 20
    assert final < initial
