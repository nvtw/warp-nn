# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from warp_nn.training import SFTExample, prepare_sft_batch


def test_sft_batch_masks_prompt_padding_and_isolates_rows():
    batch = prepare_sft_batch(
        [
            SFTExample(prompt=[1, 2, 3], response=[4, 5]),
            SFTExample(prompt=[7], response=[8]),
        ],
        6,
        pad_token_id=0,
        eos_token_id=9,
    )

    np.testing.assert_array_equal(batch.input_ids[0], [1, 2, 3, 4, 5, 0])
    np.testing.assert_array_equal(batch.targets[0], [-100, -100, 4, 5, 9, -100])
    np.testing.assert_array_equal(batch.input_ids[1], [7, 8, 0, 0, 0, 0])
    np.testing.assert_array_equal(batch.targets[1], [8, 9, -100, -100, -100, -100])
    np.testing.assert_array_equal(batch.lengths, [5, 2])
    np.testing.assert_array_equal(batch.positions[0], np.arange(6))

    input_ids, targets, lengths, positions = batch.upload("cpu")
    assert input_ids.shape == (12,)
    assert targets.shape == (12,)
    np.testing.assert_array_equal(lengths.numpy(), batch.lengths)
    np.testing.assert_array_equal(positions.numpy(), batch.positions)


def test_sft_batch_truncation_is_explicit_and_must_preserve_response():
    example = SFTExample(prompt=[1, 2], response=[3, 4, 5])
    with pytest.raises(ValueError, match="needs 4 positions"):
        prepare_sft_batch([example], 2, pad_token_id=0)
    truncated = prepare_sft_batch([example], 2, pad_token_id=0, truncation="right")
    np.testing.assert_array_equal(truncated.input_ids, [[1, 2]])
    np.testing.assert_array_equal(truncated.targets, [[-100, 3]])

    with pytest.raises(ValueError, match="no response target"):
        prepare_sft_batch(
            [SFTExample(prompt=[1, 2, 3], response=[4])],
            1,
            pad_token_id=0,
            truncation="right",
        )


def test_sft_batch_can_train_on_prompt_without_duplicate_eos():
    batch = prepare_sft_batch(
        [SFTExample(prompt=[1, 2], response=[3, 9])],
        4,
        pad_token_id=0,
        eos_token_id=9,
        train_on_prompt=True,
    )
    np.testing.assert_array_equal(batch.input_ids, [[1, 2, 3, 0]])
    np.testing.assert_array_equal(batch.targets, [[2, 3, 9, -100]])
