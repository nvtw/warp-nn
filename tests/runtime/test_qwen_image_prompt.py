# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import pytest

from warp_nn.runtime.qwen_image.prompt import (
    QWEN_IMAGE_SYSTEM_PROMPT,
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
    with pytest.raises(ValueError, match="nonempty"):
        format_qwen_image_prompt("")
    with pytest.raises(ValueError, match="between 1 and 1024"):
        tokenize_qwen_image_prompt(_Tokenizer(), "x", 1025)
