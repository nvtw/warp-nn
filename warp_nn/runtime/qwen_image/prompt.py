# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Official Qwen-Image prompt formatting and token contracts."""

from __future__ import annotations

from pathlib import Path

import warp as wp

from ..operators import SequenceSlicePlan
from ..qwen.encoder import QwenEncoder


QWEN_IMAGE_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:"
)
QWEN_IMAGE_PREFIX_TOKENS = 34


def format_qwen_image_prompt(prompt: str) -> str:
    """Wrap one caption exactly as the official Diffusers pipeline does."""
    if not isinstance(prompt, str):
        raise TypeError("Qwen-Image prompt must be a string")
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


class QwenImagePromptEncoder:
    """Official prompt adapter around a dependency-free Qwen language encoder."""

    def __init__(self, encoder):
        if not isinstance(encoder, QwenEncoder):
            raise TypeError("Qwen-Image prompt encoder requires QwenEncoder")
        if encoder.config.get("model_type") != "qwen2_5_vl":
            raise ValueError("Qwen-Image prompt encoder requires Qwen2.5-VL")
        self.encoder = encoder
        self._slices = {}

    @classmethod
    def from_pretrained(
        cls,
        path,
        *,
        dtype=wp.bfloat16,
        device=None,
        use_cublas=True,
    ):
        """Load the local text_encoder and sibling tokenizer directories."""
        root = Path(path)
        return cls(
            QwenEncoder(
                root / "text_encoder",
                dtype=dtype,
                device=device,
                use_cublas=use_cublas,
                tokenizer_path=root / "tokenizer",
            )
        )

    def encode(self, prompt: str, *, max_sequence_length=512):
        """Return contiguous ``[1, sequence, 3584]`` prompt hidden states."""
        token_ids = tokenize_qwen_image_prompt(
            self.encoder.tokenizer, prompt, max_sequence_length
        )
        length = len(token_ids) - QWEN_IMAGE_PREFIX_TOKENS
        if length <= 0:
            raise ValueError("Qwen-Image prompt produced no retained tokens")
        hidden = self.encoder.encode_ids(token_ids)
        plan = self._slices.get(len(token_ids))
        if plan is None:
            plan = self._slices[len(token_ids)] = SequenceSlicePlan(
                hidden, QWEN_IMAGE_PREFIX_TOKENS, length
            )
        elif plan.input.ptr != hidden.ptr:
            raise RuntimeError("Qwen prompt encoder changed a cached plan buffer")
        return plan.execute()
