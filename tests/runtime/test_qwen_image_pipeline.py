# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import warp_nn.runtime.qwen_image.pipeline as pipeline_module
from warp_nn.runtime.qwen_image.pipeline import (
    QwenImage2512Pipeline,
    _padded_conditioning,
    qwen_image_to_rgb8,
)


def test_padded_conditioning_stays_on_device_and_marks_valid_tokens():
    positive = wp.array(np.ones((1, 2, 3), dtype=np.float32), dtype=wp.bfloat16)
    negative = wp.array(np.full((1, 1, 3), 2.0, dtype=np.float32), dtype=wp.bfloat16)
    pos, pos_valid, neg, neg_valid = _padded_conditioning(positive, negative)
    assert pos.shape == neg.shape == (1, 2, 3)
    np.testing.assert_array_equal(pos_valid.numpy(), [[True, True]])
    np.testing.assert_array_equal(neg_valid.numpy(), [[True, False]])
    np.testing.assert_allclose(neg.numpy()[:, 1], 0.0)


def test_qwen_image_output_conversion_has_exact_endpoints_and_layout():
    sample = np.array([[[[-1.0, 1.0]], [[0.0, 0.0]], [[1.0, -1.0]]]])
    image = qwen_image_to_rgb8(sample)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image, [[[0, 128, 255], [255, 128, 0]]])


def test_denoise_does_not_overwrite_positive_conditioning(monkeypatch, tmp_path):
    calls = []

    class Config:
        input_channels = 4
        patch_size = 2

    class Scheduler:
        @staticmethod
        def schedule(steps, sequence):
            return np.array([1.0, 0.5, 0.0], dtype=np.float32)

    class Bundle:
        root = tmp_path
        transformer = Config()
        scheduler = Scheduler()

        @staticmethod
        def latent_geometry(width, height):
            return 4, 4, 4

    class Plan:
        def __init__(self, sample, text, valid, timestep, *args, **kwargs):
            self.text = text
            self.valid = valid
            self.output = wp.empty_like(sample)

        def replay(self, *, text, text_valid):
            self.text.assign(text)
            self.valid.assign(text_valid)
            value = float(self.text.numpy()[0, 0, 0])
            calls.append(value)
            self.output.assign(np.full(self.output.shape, value, dtype=np.float32))
            return self.output

    monkeypatch.setattr(pipeline_module, "SafeTensorArchive", lambda path: object())
    monkeypatch.setattr(
        pipeline_module, "load_qwen_image_transformer_weights", lambda *args: {}
    )
    monkeypatch.setattr(pipeline_module, "QwenImageMMDiTPlan", Plan)
    pipeline = object.__new__(QwenImage2512Pipeline)
    pipeline.bundle = Bundle()
    pipeline.dtype = wp.bfloat16
    pipeline.device = wp.get_device("cpu")
    pipeline.use_cublas = False
    positive = wp.array(np.ones((1, 2, 3)), dtype=wp.bfloat16)
    negative = wp.array(np.full((1, 2, 3), 2.0), dtype=wp.bfloat16)
    valid = wp.array(np.ones((1, 2), dtype=bool), dtype=wp.bool)
    latent = pipeline.denoise(
        positive,
        valid,
        negative,
        valid,
        width=8,
        height=8,
        steps=2,
    )
    assert latent.shape == (1, 1, 4, 4)
    assert calls == [1.0, 2.0, 1.0, 2.0]
    np.testing.assert_allclose(positive.numpy(), 1.0)
