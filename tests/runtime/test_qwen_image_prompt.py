# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.qwen_image.prompt import (
    QWEN_IMAGE_SYSTEM_PROMPT,
    QwenImagePromptEncoder,
    format_qwen_image_prompt,
    tokenize_qwen_image_prompt,
)


class _Tokenizer:
    def encode(self, text):
        prefix = (
            f"<|im_start|>system\n{QWEN_IMAGE_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
        )
        if text == prefix:
            return list(range(34))
        return list(range(200))


def test_qwen_image_prompt_template_and_truncation():
    prompt = format_qwen_image_prompt("A red fox")
    assert prompt == (
        f"<|im_start|>system\n{QWEN_IMAGE_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\nA red fox<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert tokenize_qwen_image_prompt(_Tokenizer(), "A red fox", 16) == list(range(50))


def test_qwen_image_prompt_rejects_invalid_contract():
    with pytest.raises(TypeError, match="string"):
        format_qwen_image_prompt(None)
    assert "user\n<|im_end|>" in format_qwen_image_prompt("")
    with pytest.raises(ValueError, match="between 1 and 1024"):
        tokenize_qwen_image_prompt(_Tokenizer(), "x", 1025)


def test_qwen_image_prompt_encoder_strips_prefix_contiguously():
    from warp_nn.runtime.qwen.encoder import QwenEncoder

    encoder = object.__new__(QwenEncoder)
    encoder.config = {"model_type": "qwen2_5_vl"}
    encoder.tokenizer = _Tokenizer()
    encoder._outputs = {}

    def encode_ids(token_ids):
        length = len(token_ids)
        values = np.arange(length * 4, dtype=np.float32).reshape(1, length, 4)
        output = encoder._outputs.get(length)
        if output is None:
            output = encoder._outputs[length] = wp.array(values, device="cpu")
        else:
            output.assign(values)
        return output

    encoder.encode_ids = encode_ids
    prompt_encoder = QwenImagePromptEncoder(encoder)
    output = prompt_encoder.encode("A red fox", max_sequence_length=16)
    expected = np.arange(50 * 4, dtype=np.float32).reshape(1, 50, 4)[:, 34:]
    assert output.is_contiguous
    np.testing.assert_array_equal(output.numpy(), expected)
