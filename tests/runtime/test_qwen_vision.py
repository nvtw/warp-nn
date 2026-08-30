# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import warp as wp


from tests.runtime.test_qwen35 import _write_tiny_qwen35
from tests.utilities import is_device_available
from warp_nn.runtime.gguf import GGUFArchive
from warp_nn.runtime.qwen35 import Qwen35Runner
from warp_nn.runtime.qwen_vision import (
    QwenMultimodalProcessor,
    QwenMultimodalPrompt,
    _gguf_map,
    _vision_weight_names,
)
from warp_nn.runtime.vision import (
    preprocess_qwen_media,
    qwen_vision_positions,
    resize_bicubic,
    smart_resize,
)


def test_smart_resize_and_bicubic_are_dependency_free():
    assert smart_resize(
        100, 200, factor=32, minimum_pixels=1, maximum_pixels=1_000_000
    ) == (96, 192)
    image = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    resized = resize_bicubic(image, 16, 20)
    assert resized.shape == (16, 20, 3)
    assert resized.dtype == np.float32
    assert np.isfinite(resized).all()


def test_qwen_patch_order_and_vision_positions():
    frames = np.zeros((2, 32, 32, 3), dtype=np.float32)
    for temporal in range(2):
        for channel in range(3):
            frames[temporal, :, :, channel] = temporal * 60 + channel * 20
    media = preprocess_qwen_media(
        frames,
        minimum_pixels=32 * 32,
        maximum_pixels=32 * 32,
    )
    assert media.grid_thw == (1, 2, 2)
    assert media.patches.shape == (4, 1536)
    assert media.feature_count == 1
    first = media.patches[0].reshape(3, 2, 16, 16)
    expected = frames[0, 0, 0, 0] / 127.5 - 1.0
    assert np.isclose(first[0, 0, 0, 0], expected)
    expected = frames[1, 0, 0, 2] / 127.5 - 1.0
    assert np.isclose(first[2, 1, 0, 0], expected)
    positions = qwen_vision_positions((1, 4, 4))
    assert positions.tolist() == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 0],
        [0, 2, 1],
        [0, 3, 0],
        [0, 3, 1],
        [0, 2, 2],
        [0, 2, 3],
        [0, 3, 2],
        [0, 3, 3],
    ]


class _Tokenizer:
    specials = {
        "<|vision_start|>": 10,
        "<|vision_end|>": 11,
        "<|image_pad|>": 12,
        "<|video_pad|>": 13,
    }

    def format_chat(self, messages, **_kwargs):
        return "".join(str(message["content"]) for message in messages)

    def encode(self, text):
        output = []
        while text:
            for token, token_id in self.specials.items():
                if text.startswith(token):
                    output.append(token_id)
                    text = text[len(token) :]
                    break
            else:
                output.append(1)
                text = text[1:]
        return output


def test_multimodal_processor_builds_exact_feature_span_and_mrope():
    processor = QwenMultimodalProcessor(_Tokenizer())
    prompt = processor.encode_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A"},
                    {"type": "image", "image": np.zeros((32, 32, 3), dtype=np.uint8)},
                    {"type": "text", "text": "B"},
                ],
            }
        ]
    )
    media = prompt.media[0]
    start = prompt.feature_starts[0]
    assert (
        prompt.token_ids[start : start + media.feature_count]
        == (12,) * media.feature_count
    )
    assert prompt.rope_positions.shape == (3, len(prompt.token_ids))
    assert prompt.rope_delta < 0
    assert np.all(prompt.rope_positions[:, -1] == prompt.rope_positions[0, -1])


def test_local_mmproj_header_matches_complete_mapping():
    path = Path(
        "/home/twidmer/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf"
    )
    if not path.is_file():
        return
    archive = GGUFArchive(path)
    depth = int(archive.metadata["clip.vision.block_count"])
    mapping = _gguf_map(depth)
    names = [
        name for name in _vision_weight_names(depth) if name != "patch_embed.weight"
    ]
    names += ["patch_embed.weight.0", "patch_embed.weight.1"]
    assert archive.metadata["general.type"] == "mmproj"
    assert archive.metadata["clip.projector_type"] == "qwen3vl_merger"
    assert set(mapping[name] for name in names) <= set(archive.names)
    assert len(names) == 334


def test_tiny_qwen_multimodal_prefill_and_graph_replay(tmp_path):
    if not is_device_available("cuda:0"):
        pytest.skip("CUDA is not available")
    model_path = tmp_path / "tiny-qwen35-vision"
    _write_tiny_qwen35(model_path)
    runner = Qwen35Runner(
        model_path,
        device="cuda:0",
        cache_capacity=8,
        prefill_chunk_size=2,
        use_cublas=False,
    )
    media = preprocess_qwen_media(
        np.zeros((32, 32, 3), dtype=np.uint8),
        minimum_pixels=32 * 32,
        maximum_pixels=32 * 32,
    )

    class _Encoder:
        def encode(self, _media):
            return wp.ones(
                (1, runner.hidden_size), dtype=runner.dtype, device=runner.device
            )

    runner._vision_encoder_instance = _Encoder()
    positions = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.int64)
    prompt = QwenMultimodalPrompt(
        token_ids=(1, 2, 3),
        media=(media,),
        feature_starts=(1,),
        rope_positions=positions,
        rope_delta=-1,
    )
    first = runner.prefill_multimodal(prompt).numpy()
    replay = runner.prefill_multimodal(prompt).numpy()
    assert first.shape == (1, 1, 16)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(replay, first, atol=2.0e-2, rtol=2.0e-2)
    assert runner.sequence_length == 3
    assert runner.rope_delta == -1
