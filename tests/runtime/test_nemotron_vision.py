# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp
from types import SimpleNamespace

from tests.utilities import local_model_root
from warp_nn.runtime.formats.safetensors import SafeTensorArchive
from warp_nn.runtime.nemotron.omni import NemotronMultimodalProcessor
from warp_nn.runtime.tokenizers import Qwen3Tokenizer
from warp_nn.runtime.nemotron.vision import (
    _kernels,
    _VisionPlan,
    pixel_unshuffle_v2,
    preprocess_nemotron_image,
    target_patch_grid,
    vision_weight_names,
)
from warp_nn.runtime.nemotron.video import (
    prune_video_embeddings,
    preprocess_nemotron_video,
    target_video_patch_grid,
    video_prompt_chunks,
)
from warp_nn.runtime.formats.media import _sample_indices


def test_nemotron_image_preprocessing_and_pixel_unshuffle():
    image = np.arange(18 * 30 * 3, dtype=np.uint16).reshape((18, 30, 3)) % 256
    media = preprocess_nemotron_image(
        image.astype(np.uint8),
        min_patches=4,
        max_patches=64,
        max_model_length=20,
    )
    assert media.patch_grid == target_patch_grid(
        18, 30, min_patches=4, max_patches=64, max_model_length=20
    )
    assert all(size % 2 == 0 for size in media.patch_grid)
    assert media.pixels.shape == (
        3,
        media.patch_grid[0] * 16,
        media.patch_grid[1] * 16,
    )
    assert np.isfinite(media.pixels).all()

    values = np.arange(4 * 6 * 3).reshape((4, 6, 3))
    actual = pixel_unshuffle_v2(values)
    expected = np.stack(
        (
            values[0::2, 0::2],
            values[0::2, 1::2],
            values[1::2, 0::2],
            values[1::2, 1::2],
        ),
        axis=2,
    ).reshape((2, 3, 12))
    np.testing.assert_array_equal(actual, expected)


def test_nemotron_unshuffle_skips_prefix_tokens():
    source = np.arange(6 * 2, dtype=np.float32).reshape((6, 2))
    output = wp.empty((1, 8), dtype=wp.float32, device="cpu")
    unshuffle = _kernels(wp.float32, 1, 2)[6]
    wp.launch(
        unshuffle,
        dim=output.shape,
        inputs=[wp.array(source, device="cpu"), output, 2, 2],
        device="cpu",
    )
    np.testing.assert_array_equal(output.numpy(), source[2:].reshape((1, 8)))


def test_nemotron_vision_manifest_matches_local_checkpoint():
    path = local_model_root() / "nvidia" / "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    if not (path / "model.safetensors.index.json").is_file():
        pytest.skip("local Nemotron Omni BF16 checkpoint is unavailable")
    archive = SafeTensorArchive(path)
    assert set(vision_weight_names(32)) <= set(archive.names)
    assert set(vision_weight_names(32, include_video=True)) <= set(archive.names)


def test_nemotron_multimodal_processor_matches_image_placeholders():
    path = local_model_root() / "nvidia" / "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    if not (path / "tokenizer.json").is_file():
        pytest.skip("local Nemotron Omni tokenizer is unavailable")
    tokenizer = Qwen3Tokenizer(path)
    processor = NemotronMultimodalProcessor(tokenizer)
    prompt = processor.encode_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": np.zeros((32, 48, 3), dtype=np.uint8)},
                    {"type": "text", "text": "What is shown?"},
                ],
            }
        ],
        enable_thinking=False,
    )
    assert len(prompt.images) == 1
    start = prompt.image_starts[0]
    count = prompt.images[0].tokens
    assert (
        prompt.token_ids[start : start + count] == (processor.image_token_id,) * count
    )


def test_video_sampling_temporal_groups_and_prompt_labels():
    assert _sample_indices(10, 1) == (0,)
    assert _sample_indices(10, 3) == (0, 4, 9)
    assert target_video_patch_grid(720, 1280) == (24, 42)
    frames = [np.full((24, 32, 3), index * 20, dtype=np.uint8) for index in range(3)]
    video = preprocess_nemotron_video(
        frames, fps=2.0, temporal_patch_size=2, target_patches=4
    )
    assert video.groups == 2
    assert video.tokens_per_group == 1
    chunks = video_prompt_chunks(video)
    assert (
        "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.50 seconds"
        in chunks[0]
    )
    assert "Frame 3 sampled at 1.00 seconds" in chunks[1]


def test_video_evs_keeps_first_frame_and_largest_change():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    compact, retained = prune_video_embeddings(
        wp.array(embeddings, device="cpu"), 3, 2, 0.5
    )
    assert retained == (0, 1, 4)
    np.testing.assert_array_equal(compact.numpy(), embeddings[list(retained)])


def test_tiny_temporal_vision_plan_reuses_image_pipeline_on_cpu():
    rng = np.random.default_rng(83)

    def tensor(shape):
        return wp.array(
            rng.normal(scale=0.1, size=shape).astype(np.float32),
            dtype=wp.bfloat16,
            device="cpu",
        )

    prefix = "vision_model.radio_model.model."
    weights = {
        prefix + "patch_generator.video_embedder.weight": tensor((4, 24)),
        prefix + "patch_generator.pos_embed": tensor((1, 16384, 4)),
        prefix + "patch_generator.cls_token.token": tensor((0, 4)),
        "mlp1.0.weight": tensor((16,)),
        "mlp1.1.weight": tensor((8, 16)),
        "mlp1.3.weight": tensor((4, 8)),
    }
    encoder = SimpleNamespace(
        patch_size=2,
        prefix_tokens=0,
        hidden_size=4,
        depth=0,
        heads=1,
        epsilon=1.0e-6,
        output_size=4,
        dtype=wp.bfloat16,
        device=wp.get_device("cpu"),
        cublas=None,
        weights=weights,
        kernels=_kernels(wp.bfloat16, 1, 4),
    )
    plan = _VisionPlan(encoder, (2, 2), temporal_patch_size=2)
    plan.pixels.assign(rng.normal(size=plan.pixels.shape).astype(np.float32))
    first = plan.run().numpy()
    second = plan.run().numpy()
    assert first.shape == (1, 4)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_nemotron_processor_preserves_interleaved_image_video_order():
    path = local_model_root() / "nvidia" / "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    if not (path / "tokenizer.json").is_file():
        pytest.skip("local Nemotron Omni tokenizer is unavailable")
    processor = NemotronMultimodalProcessor(
        Qwen3Tokenizer(path), video_target_patches=4
    )
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    video = [image, image, image]
    prompt = processor.encode_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video, "fps": 2.0},
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe both."},
                ],
            }
        ],
        enable_thinking=False,
    )
    assert len(prompt.videos) == len(prompt.images) == 1
    assert len(prompt.video_starts[0]) == 2
    assert prompt.video_starts[0][-1] < prompt.image_starts[0]
    for start in prompt.video_starts[0]:
        assert prompt.token_ids[start] == processor.image_token_id
