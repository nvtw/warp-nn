# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein's Qwen3-4B intermediate-state text conditioning."""

from functools import lru_cache
from pathlib import Path

import warp as wp

from ..qwen.encoder import QwenEncoder


@lru_cache(maxsize=None)
def _concat_kernel(dtype):
    D = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def concatenate(
        first: wp.array3d(dtype=D),
        second: wp.array3d(dtype=D),
        third: wp.array3d(dtype=D),
        output: wp.array3d(dtype=D),
    ):
        batch, token, channel = wp.tid()
        width = first.shape[2]
        if channel < width:
            output[batch, token, channel] = first[batch, token, channel]
        elif channel < 2 * width:
            output[batch, token, channel] = second[batch, token, channel - width]
        else:
            output[batch, token, channel] = third[batch, token, channel - 2 * width]

    return concatenate


class Flux2KleinPromptEncoder:
    """Extract hidden states 9, 18, and 27 from the bundled Qwen3 backbone."""

    def __init__(self, root, *, dtype=wp.bfloat16, device=None, use_cublas=True):
        root = Path(root)
        self.encoder = QwenEncoder(
            root / "text_encoder",
            tokenizer_path=root / "tokenizer",
            dtype=dtype,
            device=device,
            use_cublas=use_cublas,
            last_layer=27,
        )
        if (
            self.encoder.config["model_type"] != "qwen3"
            or self.encoder.hidden_size != 2560
        ):
            raise ValueError("FLUX.2 Klein requires the bundled Qwen3-4B encoder")

    def tokenize(self, prompt, *, max_sequence_length=512):
        if not isinstance(prompt, str):
            raise TypeError("FLUX.2 prompt must be a string")
        if int(max_sequence_length) != 512:
            raise ValueError(
                "FLUX.2 Klein currently requires the official 512 text tokens"
            )
        tokenizer = self.encoder.tokenizer
        ids = tokenizer.encode_chat(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )[:512]
        return ids + [tokenizer.pad_token_id] * (512 - len(ids)), len(ids)

    def encode(self, prompt):
        ids, valid_tokens = self.tokenize(prompt)
        states = self.encoder.encode_intermediate_ids(
            ids, (9, 18, 27), valid_tokens=valid_tokens
        )
        first = states[0]
        output = wp.empty((1, 512, 7680), dtype=first.dtype, device=first.device)
        wp.launch(
            _concat_kernel(first.dtype),
            dim=output.shape,
            inputs=[*states, output],
            device=first.device,
        )
        return output
