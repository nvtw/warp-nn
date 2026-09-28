# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import pytest

from warp_nn.runtime.qwen.encoder import load_qwen_encoder_config
from warp_nn.runtime.qwen_image import (
    QwenImage21Bundle,
    QwenImageVAEConfig,
    qwen_image_21_vae_decoder_weight_specs,
    qwen_image_to_rgba8,
)


def _bundle(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    write(
        "model_index.json",
        {
            "_class_name": "QwenImage21Pipeline",
            "processor": ["transformers", "Qwen3VLProcessor"],
            "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
            "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLQwenImage21"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        },
    )
    write(
        "transformer/config.json",
        {
            "in_channels": 64,
            "out_channels": 64,
            "patch_size": 1,
            "context_in_dim": 4096,
        },
    )
    write("vae/config.json", {"z_dim": 64, "scale_factor_spatial": 16})
    write("text_encoder/config.json", {"text_config": {"hidden_size": 4096}})
    write(
        "scheduler/scheduler_config.json",
        {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "num_train_timesteps": 1000,
            "base_image_seq_len": 256,
            "max_image_seq_len": 8192,
            "base_shift": 0.5,
            "max_shift": 0.9,
            "shift_terminal": 0.02,
            "use_dynamic_shifting": True,
            "time_shift_type": "exponential",
        },
    )
    for component, index, shard in (
        (
            "transformer",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model-00001-of-00001.safetensors",
        ),
        (
            "text_encoder",
            "model.safetensors.index.json",
            "model-00001-of-00001.safetensors",
        ),
    ):
        write(
            f"{component}/{index}",
            {
                "metadata": {"total_size": 16},
                "weight_map": {"weight": shard},
            },
        )
    return tmp_path


def test_bundle_geometry_and_missing_weights(tmp_path):
    root = _bundle(tmp_path)
    bundle = QwenImage21Bundle.inspect(root)
    assert bundle.latent_geometry(2048, 2048) == (128, 128, 16384)
    assert len(bundle.missing_weight_files()) == 3
    with pytest.raises(FileNotFoundError, match="missing 3 weight file"):
        QwenImage21Bundle.inspect(root, require_weights=True)
    with pytest.raises(ValueError, match="divisible by 32"):
        bundle.latent_geometry(1025, 1024)


def test_bundle_rejects_old_model_and_wrong_geometry(tmp_path):
    root = _bundle(tmp_path)
    index = json.loads((root / "model_index.json").read_text())
    index["_class_name"] = "QwenImagePipeline"
    (root / "model_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="official Qwen-Image-2.1"):
        QwenImage21Bundle.inspect(root)
    index["_class_name"] = "QwenImage21Pipeline"
    (root / "model_index.json").write_text(json.dumps(index))
    config = json.loads((root / "vae/config.json").read_text())
    config["scale_factor_spatial"] = 8
    (root / "vae/config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="geometry"):
        QwenImage21Bundle.inspect(root)


def test_qwen3_vl_text_config_and_rgba_output(tmp_path):
    config = {
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
            },
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert load_qwen_encoder_config(path)["model_type"] == "qwen3_vl"
    rgba = qwen_image_to_rgba8(
        np.array([[[[-1.0]], [[0.0]], [[1.0]], [[0.5]]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(rgba[0, 0], [0, 128, 255, 191])


def test_qwen_image21_vae_decoder_selects_single_frame_rgba_weights():
    config = QwenImageVAEConfig(
        base_dim=96,
        dimension_multipliers=(1, 2, 4, 8, 8),
        residual_blocks=2,
        latent_channels=64,
        temporal_downsample=(False, True, True, True),
        latent_mean=(0.0,) * 64,
        latent_std=(1.0,) * 64,
    )
    specs = qwen_image_21_vae_decoder_weight_specs(config)
    names = {spec.name for spec in specs}
    assert len(names) == 128
    assert "decoder.up_blocks.0.upsampler.time_conv.weight" not in names
    assert "decoder.up_blocks.0.upsampler.resample.1.weight" in names
    assert next(
        spec for spec in specs if spec.name == "decoder.conv_out.weight"
    ).source_shape == (4, 144, 3, 3)
