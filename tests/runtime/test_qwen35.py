# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import json

import numpy as np
import warp as wp

from tests.utilities import is_device_available, write_safetensors
from warp_nn.runtime.kernels import _get_small_batch_grouped_linear_kernel
from warp_nn.runtime.autoregressive import _PlanMemoryError, _union_storage_bytes
from warp_nn.runtime.qwen.qwen35 import (
    Qwen35Runner,
    _validate_config,
    _weight_names,
)


def _bfloat16_bytes(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16).tobytes()


def _write_tiny_qwen35(path):
    config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 8,
        "intermediate_size": 12,
        "vocab_size": 16,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1.0e-6,
        "attention_bias": False,
        "hidden_act": "silu",
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    }
    rng = np.random.default_rng(97)
    shapes = {
        "model.language_model.embed_tokens.weight": (16, 8),
        "model.language_model.norm.weight": (8,),
        "lm_head.weight": (16, 8),
    }
    for index in range(2):
        prefix = f"model.language_model.layers.{index}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (8,),
                prefix + "post_attention_layernorm.weight": (8,),
                prefix + "mlp.gate_proj.weight": (12, 8),
                prefix + "mlp.up_proj.weight": (12, 8),
                prefix + "mlp.down_proj.weight": (8, 12),
            }
        )
    linear = "model.language_model.layers.0.linear_attn."
    shapes.update(
        {
            linear + "in_proj_qkv.weight": (16, 8),
            linear + "in_proj_z.weight": (8, 8),
            linear + "in_proj_a.weight": (2, 8),
            linear + "in_proj_b.weight": (2, 8),
            linear + "conv1d.weight": (16, 1, 3),
            linear + "A_log": (2,),
            linear + "dt_bias": (2,),
            linear + "norm.weight": (4,),
            linear + "out_proj.weight": (8, 8),
        }
    )
    attention = "model.language_model.layers.1.self_attn."
    shapes.update(
        {
            attention + "q_proj.weight": (24, 8),
            attention + "k_proj.weight": (4, 8),
            attention + "v_proj.weight": (4, 8),
            attention + "q_norm.weight": (4,),
            attention + "k_norm.weight": (4,),
            attention + "o_proj.weight": (8, 12),
        }
    )
    tensors = {}
    for name in _weight_names(config):
        shape = shapes[name]
        if (
            name.endswith("layernorm.weight")
            or name.endswith("q_norm.weight")
            or name.endswith("k_norm.weight")
        ):
            values = np.zeros(shape, dtype=np.float32)
        elif name.endswith("linear_attn.norm.weight"):
            values = np.ones(shape, dtype=np.float32)
        elif name.endswith("A_log"):
            values = np.zeros(shape, dtype=np.float32)
        else:
            values = rng.normal(0.0, 0.08, shape).astype(np.float32)
        tensors[name] = (
            ("F32", shape, values.tobytes())
            if name.endswith("linear_attn.in_proj_a.weight")
            else ("BF16", shape, _bfloat16_bytes(values))
        )
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"text_config": config}), encoding="utf-8"
    )
    write_safetensors(path / "model.safetensors", tensors)


def test_qwen38_text_metadata_compatibility():
    config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "num_hidden_layers": 64,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        * 16,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1.0e-6,
        "attention_bias": False,
        "hidden_act": "silu",
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
        },
    }

    _validate_config(config)
    names = _weight_names(config)
    assert len(names) == 851
    assert "model.language_model.layers.62.linear_attn.in_proj_qkv.weight" in names
    assert "model.language_model.layers.63.self_attn.q_proj.weight" in names

    config["rope_parameters"] = {
        **config["rope_parameters"],
        "rope_type": "yarn",
        "factor": 4.0,
    }
    with pytest.raises(ValueError, match="default Qwen rotary"):
        _validate_config(config)


def test_union_storage_bytes_deduplicates_overlapping_views():
    assert _union_storage_bytes([(100, 140), (120, 180), (220, 228)]) == 88


@pytest.mark.parametrize("use_cublas", [False, True])
def test_qwen35_native_prefill_decode_and_graph_replay(tmp_path, use_cublas):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    model_path = tmp_path / "tiny-qwen35"
    _write_tiny_qwen35(model_path)
    runner = Qwen35Runner(
        model_path,
        device="cuda:0",
        cache_capacity=8,
        prefill_chunk_size=4,
        use_cublas=use_cublas,
    )
    assert (
        runner.weights[
            "model.language_model.layers.0.linear_attn.in_proj_a.weight"
        ].dtype
        == wp.bfloat16
    )
    plan = runner._chunk_plan
    assert plan._owned_storage_bytes > 0
    assert 0 < plan._pool_storage_bytes <= plan._owned_storage_bytes
    assert (
        plan.tensors[plan.layers[0]["mlp_gate"].outputs[0]].ptr
        == plan.tensors[plan.layers[1]["mlp_gate"].outputs[0]].ptr
    )
    assert (
        plan.tensors[plan.layers[0]["swiglu"].outputs[0]].ptr
        != plan.tensors[plan.layers[1]["swiglu"].outputs[0]].ptr
    )

    first = runner.prefill([1, 2, 3]).numpy()
    assert set(runner._chunk_plans) == {2, 4}
    assert first.shape == (1, 1, 16)
    assert np.isfinite(first).all()
    decoded = runner.decode(4).numpy()
    assert decoded.shape == (1, 1, 16)
    assert np.isfinite(decoded).all()
    replayed = runner.prefill([1, 2, 3])
    np.testing.assert_allclose(replayed.numpy(), first, atol=2.0e-2, rtol=2.0e-2)
    assert 0 <= runner.sample_greedy(replayed) < 16
    top_values, top_tokens = runner.read_top_k(replayed, 8)
    host_logits = replayed.numpy()[0, -1].astype(np.float32)
    expected = np.lexsort((np.arange(host_logits.size), -host_logits))[:8]
    np.testing.assert_array_equal(top_tokens, expected)
    np.testing.assert_array_equal(top_values, host_logits[expected])
    state = runner._top_k_state
    all_values, all_tokens = runner.read_top_k(replayed, 20)
    assert runner._top_k_state is state
    assert len(all_tokens) == 16
    np.testing.assert_array_equal(all_tokens[:8], top_tokens)
    np.testing.assert_array_equal(all_values[:8], top_values)
    full_chunk = runner.prefill([1, 2, 3, 4]).numpy()
    runner.prefill([1, 2, 3])
    sequential = runner.decode(4).numpy()
    assert full_chunk.shape == (1, 1, 16)
    np.testing.assert_allclose(full_chunk, sequential, atol=2.0e-2, rtol=2.0e-2)


def test_qwen35_low_memory_tail_falls_back_to_decode(tmp_path, monkeypatch):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    model_path = tmp_path / "tiny-qwen35"
    _write_tiny_qwen35(model_path)
    runner = Qwen35Runner(
        model_path,
        device="cuda:0",
        cache_capacity=8,
        prefill_chunk_size=4,
        use_cublas=False,
    )
    expected = runner.prefill([1, 2, 3]).numpy()
    runner._chunk_plans = {4: runner._chunk_plan}
    original = runner._plan_for_rows

    def deny_two_rows(rows):
        if rows == 2:
            raise _PlanMemoryError("test headroom denial")
        return original(rows)

    monkeypatch.setattr(runner, "_plan_for_rows", deny_two_rows)
    actual = runner.prefill([1, 2, 3]).numpy()
    np.testing.assert_allclose(actual, expected, atol=2.0e-2, rtol=2.0e-2)
    assert runner.sequence_length == 3
    assert set(runner._chunk_plans) == {4}
    runner._chunk_plan._owned_storage_bytes = runner.device.total_memory
    warm = runner.prefill([1, 2, 3, 4]).numpy()
    uncaptured = runner.prefill([1, 2, 3, 4]).numpy()
    assert runner._chunk_plan._capture_disabled
    assert not runner._chunk_plan.graphs
    np.testing.assert_allclose(uncaptured, warm, atol=2.0e-2, rtol=2.0e-2)


@pytest.mark.parametrize(
    "rows,outputs_per_group",
    [(2, 4), (2, 8), (4, 4), (4, 8), (8, 4)],
)
def test_small_batch_grouped_projection_matches_numpy(rows, outputs_per_group):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(113 + rows + outputs_per_group)
    x = rng.normal(0.0, 0.2, (rows, 32)).astype(np.float32)
    weight = rng.normal(0.0, 0.2, (16, 32)).astype(np.float32)
    x_device = wp.array(x, dtype=wp.bfloat16, device="cuda:0")
    weight_device = wp.array(weight, dtype=wp.bfloat16, device="cuda:0")
    output = wp.empty((rows, 16), dtype=wp.bfloat16, device="cuda:0")
    wp.launch(
        _get_small_batch_grouped_linear_kernel(wp.bfloat16, rows, outputs_per_group),
        dim=(16 // outputs_per_group) * 32,
        inputs=[x_device, weight_device, output, 32],
        block_dim=128,
        device="cuda:0",
    )
    expected = x @ weight.T
    np.testing.assert_allclose(
        output.numpy().astype(np.float32), expected, atol=6.0e-3, rtol=2.0e-2
    )


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_qwen35_independent_batch_decode_matches_sequential_and_captures(
    tmp_path, batch_size
):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    model_path = tmp_path / "tiny-qwen35-batch"
    _write_tiny_qwen35(model_path)
    runner = Qwen35Runner(
        model_path,
        device="cuda:0",
        cache_capacity=8,
        prefill_chunk_size=4,
        use_cublas=False,
    )
    batch = runner.create_batch_decoder(batch_size)
    assert (
        batch.plan.tensors["lm_head.weight"].ptr == runner.weights["lm_head.weight"].ptr
    )
    references = []
    prompts = ([1, 2, 3], [5, 6], [7, 8, 9], [10, 11])[:batch_size]
    for slot, prompt in enumerate(prompts):
        batch.prefill(slot, prompt)
        runner.prefill(prompt)
        references.append(runner.decode(4).numpy())

    actual = batch.decode([4] * batch_size).numpy()
    for slot, expected in enumerate(references):
        np.testing.assert_array_equal(actual[slot], expected[0])
    assert len(batch.plan.graphs) == 0

    recurrent_before = batch.recurrent_states[0].numpy().copy()
    conv_before = batch.conv_states[0].numpy().copy()
    key_before = batch.kv_caches[1][0].numpy().copy()
    batch.decode([3] * batch_size, active=[True] + [False] * (batch_size - 1))
    recurrent_after = batch.recurrent_states[0].numpy()
    conv_after = batch.conv_states[0].numpy()
    key_after = batch.kv_caches[1][0].numpy()
    recurrent_rows = recurrent_before.shape[0] // batch_size
    cache_rows = key_before.shape[0] // batch_size
    np.testing.assert_array_equal(
        recurrent_after[recurrent_rows:], recurrent_before[recurrent_rows:]
    )
    np.testing.assert_array_equal(conv_after[1:], conv_before[1:])
    np.testing.assert_array_equal(key_after[cache_rows:], key_before[cache_rows:])
    assert len(batch.plan.graphs) == 1


def test_qwen35_compact_decode_maps_noncontiguous_physical_slots(tmp_path):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    model_path = tmp_path / "tiny-qwen35-mapped-batch"
    _write_tiny_qwen35(model_path)
    runner = Qwen35Runner(
        model_path,
        device="cuda:0",
        cache_capacity=8,
        prefill_chunk_size=4,
        use_cublas=False,
    )
    batch = runner.create_batch_decoder(8)
    slots = (0, 2, 5, 7)
    prompts = ([1, 2], [3, 4, 5], [6, 7], [8, 9, 10])
    references = []
    for slot, prompt in zip(slots, prompts, strict=True):
        batch.prefill(slot, prompt)
        runner.prefill(prompt)
        references.append(runner.decode(11).numpy())

    recurrent_before = batch.recurrent_states[0].numpy().copy()
    conv_before = batch.conv_states[0].numpy().copy()
    untouched = (1, 3, 4, 6)
    cache_rows = batch.kv_caches[1][0].shape[0] // 8
    for slot in untouched:
        batch.kv_caches[1][0][slot * cache_rows : (slot + 1) * cache_rows].zero_()
    key_before = batch.kv_caches[1][0].numpy().copy()
    actual = batch.decode_mapped(slots, [11] * 4, [True] * 4, 4).numpy()
    for lane, expected in enumerate(references):
        np.testing.assert_array_equal(actual[lane], expected[0])

    recurrent_after = batch.recurrent_states[0].numpy()
    conv_after = batch.conv_states[0].numpy()
    key_after = batch.kv_caches[1][0].numpy()
    recurrent_rows = recurrent_before.shape[0] // 8
    for slot in untouched:
        np.testing.assert_array_equal(conv_after[slot], conv_before[slot])
        np.testing.assert_array_equal(
            recurrent_after[slot * recurrent_rows : (slot + 1) * recurrent_rows],
            recurrent_before[slot * recurrent_rows : (slot + 1) * recurrent_rows],
        )
        np.testing.assert_array_equal(
            key_after[slot * cache_rows : (slot + 1) * cache_rows],
            key_before[slot * cache_rows : (slot + 1) * cache_rows],
        )
    assert batch._batch_views[4].mapped_state
    assert len(batch._batch_plans[4].graphs) == 0
    batch.decode_mapped(slots, [12] * 4, [True] * 4, 4)
    assert len(batch._batch_plans[4].graphs) == 1


def test_qwen35_incremental_prefill_interleaves_slots_without_state_copies(tmp_path):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    model_path = tmp_path / "tiny-qwen35-incremental"
    _write_tiny_qwen35(model_path)
    runner = Qwen35Runner(
        model_path,
        device="cuda:0",
        cache_capacity=8,
        prefill_chunk_size=2,
        use_cublas=False,
    )
    batch = runner.create_batch_decoder(2)
    prompts = ([1, 2, 3, 4], [5, 6, 7, 8])
    expected = [runner.prefill(prompt).numpy().copy() for prompt in prompts]

    for _ in range(2):
        for slot in range(2):
            batch.begin_prefill(slot)
        batch.append_prefill(0, prompts[0][:2])
        batch.append_prefill(1, prompts[1][:2])
        actual0 = batch.append_prefill(0, prompts[0][2:])
        actual1 = batch.append_prefill(1, prompts[1][2:])
        np.testing.assert_array_equal(actual0.numpy(), expected[0])
        np.testing.assert_array_equal(actual1.numpy(), expected[1])
        batch.end_prefill(0)
        batch.end_prefill(1)

    assert set(batch._incremental_plans) == {2}
    plan = batch._incremental_plans[2]
    assert set(plan.graphs) == {0, 1}
    assert plan.tensors["lm_head.weight"].ptr == runner.weights["lm_head.weight"].ptr

    batch.resume_prefill(0)
    continued = batch.append_prefill(0, [9])
    batch.end_prefill(0)
    expected_continued = runner.prefill([*prompts[0], 9]).numpy()
    np.testing.assert_array_equal(continued.numpy(), expected_continued)

    actual = batch.decode([10, 10]).numpy()
    decode_prompts = ([*prompts[0], 9], prompts[1])
    for slot, prompt in enumerate(decode_prompts):
        runner.prefill(prompt)
        expected_decode = runner.decode(10).numpy()
        np.testing.assert_array_equal(actual[slot], expected_decode[0])

    batch.release(0)
    with pytest.raises(RuntimeError, match="begin_prefill"):
        batch.append_prefill(0, [1])
    with pytest.raises(RuntimeError, match="empty"):
        batch.resume_prefill(0)


def test_qwen35_single_slot_decode_uses_batch_one_plan_and_isolates_state(tmp_path):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    model_path = tmp_path / "tiny-qwen35-single-slot"
    _write_tiny_qwen35(model_path)
    runner = Qwen35Runner(
        model_path,
        device="cuda:0",
        cache_capacity=8,
        prefill_chunk_size=4,
        use_cublas=False,
    )
    batch = runner.create_batch_decoder(4)
    prompts = ([1, 2], [3, 4], [5, 6], [7, 8])
    for slot, prompt in enumerate(prompts):
        batch.prefill(slot, prompt)

    retained = list(prompts[1])
    for token in (12, 13, 14):
        batch.resume_prefill(1)
        tail_logits = batch.append_prefill(1, [token])
        batch.end_prefill(1)
        retained.append(token)
        wp.synchronize_stream(runner.device)
        expected_tail = runner.prefill(retained).numpy()
        np.testing.assert_array_equal(tail_logits.numpy(), expected_tail)

    recurrent_before = batch.recurrent_states[0].numpy().copy()
    conv_before = batch.conv_states[0].numpy().copy()
    key_before = batch.kv_caches[1][0].numpy().copy()
    wp.synchronize_stream(runner.device)
    runner.prefill(prompts[2])
    for token in (9, 10, 11):
        expected = runner.decode(token).numpy()
        actual = batch.decode_one(2, token).numpy()
        np.testing.assert_array_equal(actual, expected)

    plan = batch._incremental_plans[1]
    assert set(plan.graphs) == {1, 2}
    assert plan.tensors["lm_head.weight"].ptr == runner.weights["lm_head.weight"].ptr
    recurrent_after = batch.recurrent_states[0].numpy()
    conv_after = batch.conv_states[0].numpy()
    key_after = batch.kv_caches[1][0].numpy()
    recurrent_rows = recurrent_before.shape[0] // 4
    cache_rows = key_before.shape[0] // 4
    for slot in (0, 1, 3):
        np.testing.assert_array_equal(
            recurrent_after[slot * recurrent_rows : (slot + 1) * recurrent_rows],
            recurrent_before[slot * recurrent_rows : (slot + 1) * recurrent_rows],
        )
        np.testing.assert_array_equal(conv_after[slot], conv_before[slot])
        # Unwritten cache rows may hold NaNs; isolation requires exact values
        # and matching NaN positions, not initialized padding.
        assert np.array_equal(
            key_after[slot * cache_rows : (slot + 1) * cache_rows],
            key_before[slot * cache_rows : (slot + 1) * cache_rows],
            equal_nan=True,
        )
