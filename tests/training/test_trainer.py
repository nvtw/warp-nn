# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import warp as wp

from tests.training.test_muse import _model_fixture as _muse_model_fixture
from tests.training.test_qwen import _model_fixture as _qwen_model_fixture
from warp_nn.training import LoRATrainer, SFTExample, prepare_sft_batch


def _trainer(device, model_fixture=_muse_model_fixture):
    model, _, _, _, _, cosine, sine = model_fixture(device)
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


def test_trainer_accumulation_matches_manual_valid_token_update_cpu():
    trainer = _trainer("cpu")
    reference = _trainer("cpu")
    trainer.model.adapters.optimizer.normalize_by_valid_tokens = True
    reference.model.adapters.optimizer.normalize_by_valid_tokens = True
    batches = (
        prepare_sft_batch([SFTExample(prompt=[1], response=[3, 5])], 2, pad_token_id=0),
        prepare_sft_batch([SFTExample(prompt=[2], response=[4, 6])], 2, pad_token_id=0),
    )

    trainer.begin_accumulation()
    for batch in batches:
        trainer.load_batch(batch)
        trainer.accumulate()
    trainer.finish_accumulation()

    model = reference.model
    model.adapters.zero_grad()
    for batch in batches:
        inputs = (*batch.upload("cpu"), reference.cosine, reference.sine)
        model.forward(*inputs, reduction="sum")
        model.backward(*inputs, reduction="sum", accumulate=True)
        model.adapters.optimizer.accumulate_valid_tokens(model.output.valid_count)
    model.adapters.step()

    assert int(trainer.model.adapters.optimizer.step_count.numpy()[0]) == 1
    for name, master in trainer.model.adapters.named_masters.items():
        np.testing.assert_array_equal(
            master.numpy(), reference.model.adapters.named_masters[name].numpy()
        )


def test_training_state_resumes_bitwise_and_preserves_buffers_cpu(tmp_path):
    uninterrupted = _trainer("cpu")
    for _ in range(4):
        uninterrupted.step()
    path = tmp_path / "training-state.safetensors"
    copy_path = tmp_path / "training-state-copy.safetensors"
    uninterrupted.save_training_state(path, base_identifier="test/muse")
    uninterrupted.save_training_state(copy_path, base_identifier="test/muse")
    assert path.read_bytes() == copy_path.read_bytes()

    resumed = _trainer("cpu")
    collection = resumed.model.adapters
    pointers = {
        "parameters": tuple(
            value.ptr for value in collection.named_parameters.values()
        ),
        "masters": tuple(value.ptr for value in collection.named_masters.values()),
        "first": tuple(value.ptr for value in collection.optimizer.first_moments),
        "second": tuple(value.ptr for value in collection.optimizer.second_moments),
    }
    checkpoint = resumed.load_training_state(path, base_identifier="test/muse")
    assert checkpoint.base_identifier == "test/muse"
    assert checkpoint.backend == "warp"
    assert int(collection.optimizer.step_count.numpy()[0]) == 4
    assert pointers == {
        "parameters": tuple(
            value.ptr for value in collection.named_parameters.values()
        ),
        "masters": tuple(value.ptr for value in collection.named_masters.values()),
        "first": tuple(value.ptr for value in collection.optimizer.first_moments),
        "second": tuple(value.ptr for value in collection.optimizer.second_moments),
    }

    for _ in range(3):
        uninterrupted.step()
        resumed.step()
    _assert_training_state_equal(uninterrupted, resumed)


def test_training_state_rejects_mismatch_and_active_accumulation_cpu(tmp_path):
    trainer = _trainer("cpu")
    path = tmp_path / "training-state.safetensors"
    trainer.step()
    trainer.save_training_state(path, base_identifier="test/muse")

    mismatch = _trainer("cpu")
    mismatch.model.adapters.optimizer.learning_rate *= 2.0
    with pytest.raises(ValueError, match="optimizer configuration"):
        mismatch.load_training_state(path, base_identifier="test/muse")
    with pytest.raises(ValueError, match="base identifier"):
        trainer.load_training_state(path, base_identifier="other/model")

    trainer.model.adapters.optimizer.normalize_by_valid_tokens = True
    trainer.begin_accumulation()
    with pytest.raises(RuntimeError, match="finish gradient accumulation"):
        trainer.save_training_state(path)
    with pytest.raises(RuntimeError, match="finish gradient accumulation"):
        trainer.load_training_state(path)


CUDA_DEVICES = [device for device in wp.get_devices() if device.is_cuda]


def _assert_training_state_equal(left, right):
    np.testing.assert_array_equal(
        left.model.output.loss.numpy(), right.model.output.loss.numpy()
    )
    left_adapters, right_adapters = left.model.adapters, right.model.adapters
    assert tuple(left_adapters.targets) == tuple(right_adapters.targets)
    for left_values, right_values in (
        (left_adapters.named_parameters, right_adapters.named_parameters),
        (left_adapters.named_masters, right_adapters.named_masters),
        (left_adapters.named_gradients, right_adapters.named_gradients),
    ):
        assert tuple(left_values) == tuple(right_values)
        for name, value in left_values.items():
            np.testing.assert_array_equal(value.numpy(), right_values[name].numpy())
    left_optimizer = left_adapters.optimizer
    right_optimizer = right_adapters.optimizer
    for left_values, right_values in (
        (left_optimizer.first_moments, right_optimizer.first_moments),
        (left_optimizer.second_moments, right_optimizer.second_moments),
    ):
        for left_value, right_value in zip(left_values, right_values):
            np.testing.assert_array_equal(left_value.numpy(), right_value.numpy())
    for name in (
        "step_count",
        "valid_token_count",
        "all_finite",
        "step_enabled",
        "normalization_multiplier",
        "effective_learning_rate",
        "global_grad_norm",
        "clip_scale",
    ):
        left_value = getattr(left_optimizer, name)
        right_value = getattr(right_optimizer, name)
        assert (left_value is None) == (right_value is None)
        if left_value is not None:
            np.testing.assert_array_equal(left_value.numpy(), right_value.numpy())


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
@pytest.mark.parametrize(
    "model_fixture", [_muse_model_fixture, _qwen_model_fixture], ids=["muse", "qwen"]
)
@pytest.mark.parametrize("accumulated", [False, True], ids=["ordinary", "accumulated"])
def test_trainer_direct_and_graph_trajectories_are_bitwise_equal(
    model_fixture, accumulated
):
    device = CUDA_DEVICES[0]
    direct = _trainer(device, model_fixture)
    captured = _trainer(device, model_fixture)
    batches = (
        prepare_sft_batch([SFTExample(prompt=[1], response=[3, 5])], 2, pad_token_id=0),
        prepare_sft_batch([SFTExample(prompt=[2], response=[4, 6])], 2, pad_token_id=0),
    )
    if accumulated:
        direct.model.adapters.optimizer.normalize_by_valid_tokens = True
        captured.model.adapters.optimizer.normalize_by_valid_tokens = True
        captured.capture_accumulation()
        for _ in range(2):
            direct.begin_accumulation()
            captured.begin_accumulation()
            for batch in batches:
                direct.load_batch(batch)
                captured.load_batch(batch)
                direct.accumulate()
                captured.accumulate()
            direct.finish_accumulation()
            captured.finish_accumulation()
            wp.synchronize_device(device)
            _assert_training_state_equal(direct, captured)
    else:
        captured.capture()
        for batch in (*batches, batches[0]):
            direct.load_batch(batch)
            captured.load_batch(batch)
            direct.step()
            captured.step()
            wp.synchronize_device(device)
            _assert_training_state_equal(direct, captured)


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_training_state_restore_keeps_captured_graph_valid(tmp_path):
    uninterrupted = _trainer(CUDA_DEVICES[0])
    uninterrupted.capture()
    for _ in range(2):
        uninterrupted.step()
    path = tmp_path / "captured-training-state.safetensors"
    uninterrupted.save_training_state(path)

    resumed = _trainer(CUDA_DEVICES[0])
    resumed.capture()
    resumed.load_training_state(path)
    for _ in range(2):
        uninterrupted.step()
        resumed.step()
    wp.synchronize_device(CUDA_DEVICES[0])
    _assert_training_state_equal(uninterrupted, resumed)


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


@pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA device required")
def test_trainer_captures_valid_token_accumulation():
    trainer = _trainer(CUDA_DEVICES[0])
    optimizer = trainer.model.adapters.optimizer
    optimizer.normalize_by_valid_tokens = True
    trainer.capture_accumulation()
    assert int(optimizer.step_count.numpy()[0]) == 0

    trainer.begin_accumulation()
    trainer.accumulate()
    trainer.accumulate()
    trainer.finish_accumulation()
    wp.synchronize_device(trainer.device)

    assert int(optimizer.step_count.numpy()[0]) == 1
    assert int(optimizer.valid_token_count.numpy()[0]) == 4
