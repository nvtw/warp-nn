# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Official Qwen-Image prompt formatting and token contracts."""

from __future__ import annotations


QWEN_IMAGE_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:"
)
QWEN_IMAGE_PREFIX_TOKENS = 34


def format_qwen_image_prompt(prompt: str) -> str:
    """Wrap one caption exactly as the official Diffusers pipeline does."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Qwen-Image prompt must be a nonempty string")
    return (
        f"<|im_start|>system\n{QWEN_IMAGE_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def tokenize_qwen_image_prompt(tokenizer, prompt: str, max_sequence_length=512):
    """Return official wrapped IDs, including the 34-token removable prefix."""
    max_sequence_length = int(max_sequence_length)
    if not 1 <= max_sequence_length <= 1024:
        raise ValueError("Qwen-Image prompt length must be between 1 and 1024")
    prefix = (
        f"<|im_start|>system\n{QWEN_IMAGE_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
    )
    prefix_ids = tokenizer.encode(prefix)
    if len(prefix_ids) != QWEN_IMAGE_PREFIX_TOKENS:
        raise ValueError(
            "Qwen-Image tokenizer does not produce the official 34-token prefix"
        )
    token_ids = tokenizer.encode(format_qwen_image_prompt(prompt))
    return token_ids[: max_sequence_length + QWEN_IMAGE_PREFIX_TOKENS]
