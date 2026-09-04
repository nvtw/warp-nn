# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in end-to-end OCR quality check for the local Nemotron Omni model."""

import os
import numpy as np
import pytest

from tests.utilities import is_device_available, local_model_root
from warp_nn.runtime import (
    create_multimodal_processor,
    create_text_runner,
)


_FONT = {
    " ": ("00000",) * 7,
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
}


def _text_image(lines, scale=10, padding=40):
    """Render uppercase test text without a font or imaging dependency."""
    widths = [sum(len(_FONT[char][0]) + 1 for char in line) - 1 for line in lines]
    canvas = np.full(
        (len(lines) * 7 + (len(lines) - 1) * 3, max(widths)), 255, np.uint8
    )
    y = 0
    for line in lines:
        x = 0
        for char in line:
            glyph = _FONT[char]
            for row, bits in enumerate(glyph):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        canvas[y + row, x + column] = 0
            x += len(glyph[0]) + 1
        y += 10
    canvas = np.repeat(np.repeat(canvas, scale, axis=0), scale, axis=1)
    canvas = np.pad(canvas, padding, constant_values=255)
    return np.repeat(canvas[..., None], 3, axis=2)


def _normalized_ocr(text):
    return " ".join(text.upper().split())


def test_nemotron_omni_reconstructs_generated_text_image():
    """Read known text through vision, language prefill, and greedy decode."""
    if os.environ.get("WARP_NN_RUN_LARGE_MODEL_TESTS") != "1":
        pytest.skip("set WARP_NN_RUN_LARGE_MODEL_TESTS=1 to load the 30B checkpoint")
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is unavailable")

    model = (
        local_model_root()
        / "nvidia"
        / "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    )
    if not (model / "model.safetensors.index.json").is_file():
        pytest.skip("local Nemotron Omni BF16 checkpoint is unavailable")

    expected_lines = (
        "THIS OCR TEST PRESENTS",
        "A CLEAR PARAGRAPH.",
        "NEMOTRON SHOULD READ",
        "EVERY WORD IN ORDER.",
        "THE VERIFICATION CODE",
        "IS 4827",
    )
    image = _text_image(expected_lines)
    processor = create_multimodal_processor(model)
    prompt = processor.encode_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Transcribe all text in this image exactly. Return only "
                            "the complete paragraph, preserving the reading order."
                        ),
                    },
                ],
            }
        ],
        enable_thinking=False,
    )
    runner = create_text_runner(
        model,
        device="cuda:0",
        cache_capacity=2048,
        prefill_chunk_size=256,
    )
    logits = runner.prefill_multimodal(prompt)
    generated = []
    for _ in range(96):
        token = runner.sample_greedy(logits)
        generated.append(token)
        if token in processor.tokenizer.eos_token_ids:
            break
        logits = runner.decode(token)

    actual = processor.tokenizer.decode(generated, skip_special_tokens=True)
    assert _normalized_ocr(actual) == _normalized_ocr(" ".join(expected_lines)), actual
