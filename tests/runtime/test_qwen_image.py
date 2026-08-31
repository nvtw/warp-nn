# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from warp_nn.runtime.formats.safetensors import read_safetensors_index
from warp_nn.runtime.qwen_image import QwenImage2512Bundle


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _bundle(path):
    _write_json(
        path / "model_index.json",
        {
            "_class_name": "QwenImagePipeline",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": [
                "transformers",
                "Qwen2_5_VLForConditionalGeneration",
            ],
            "tokenizer": ["transformers", "Qwen2Tokenizer"],
            "transformer": ["diffusers", "QwenImageTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLQwenImage"],
        },
    )
    _write_json(
        path / "transformer/config.json",
        {
            "_class_name": "QwenImageTransformer2DModel",
            "attention_head_dim": 128,
            "axes_dims_rope": [16, 56, 56],
            "guidance_embeds": False,
            "in_channels": 64,
            "joint_attention_dim": 3584,
            "num_attention_heads": 24,
            "num_layers": 60,
            "out_channels": 16,
            "patch_size": 2,
        },
    )
    _write_json(
        path / "vae/config.json",
        {
            "_class_name": "AutoencoderKLQwenImage",
            "base_dim": 96,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "temperal_downsample": [False, True, True],
            "z_dim": 16,
            "latents_mean": [0.0] * 16,
            "latents_std": [1.0] * 16,
        },
    )
    _write_json(
        path / "scheduler/scheduler_config.json",
        {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "max_image_seq_len": 8192,
            "max_shift": 0.9,
            "num_train_timesteps": 1000,
            "shift_terminal": 0.02,
            "time_shift_type": "exponential",
            "use_dynamic_shifting": True,
        },
    )
    _write_json(
        path / "text_encoder/config.json",
        {
            "model_type": "qwen2_5_vl",
            "hidden_size": 3584,
            "num_hidden_layers": 28,
        },
    )
    _write_json(
        path / "transformer/diffusion_pytorch_model.safetensors.index.json",
        {
            "metadata": {"total_size": 40_860_802_176},
            "weight_map": {
                "img_in.weight": "diffusion_pytorch_model-00001-of-00009.safetensors",
                "proj_out.weight": "diffusion_pytorch_model-00009-of-00009.safetensors",
            },
        },
    )
    _write_json(
        path / "text_encoder/model.safetensors.index.json",
        {
            "metadata": {"total_size": 16_584_333_312},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
                "model.norm.weight": "model-00004-of-00004.safetensors",
            },
        },
    )


def test_qwen_image_metadata_only_bundle(tmp_path):
    _bundle(tmp_path)
    bundle = QwenImage2512Bundle.inspect(tmp_path)

    assert bundle.transformer.hidden_size == 3072
    assert bundle.vae.spatial_scale_factor == 8
    assert bundle.image_multiple == 16
    assert bundle.latent_geometry(1328, 1328) == (166, 166, 6889)
    assert bundle.transformer_index.total_size == 40_860_802_176
    assert len(bundle.missing_weight_files()) == 5


def test_qwen_image_bundle_can_require_weights(tmp_path):
    _bundle(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing 5 weight file"):
        QwenImage2512Bundle.inspect(tmp_path, require_weights=True)


def test_qwen_image_rejects_incompatible_geometry(tmp_path):
    _bundle(tmp_path)
    config = json.loads((tmp_path / "transformer/config.json").read_text())
    config["in_channels"] = 32
    _write_json(tmp_path / "transformer/config.json", config)
    with pytest.raises(ValueError, match="packed latent input"):
        QwenImage2512Bundle.inspect(tmp_path)


def test_qwen_image_rejects_non_aligned_resolution(tmp_path):
    _bundle(tmp_path)
    bundle = QwenImage2512Bundle.inspect(tmp_path)
    with pytest.raises(ValueError, match="divisible by 16"):
        bundle.latent_geometry(1025, 1024)


def test_safetensors_index_rejects_escaping_shard(tmp_path):
    path = tmp_path / "model.safetensors.index.json"
    _write_json(path, {"weight_map": {"weight": "../outside.safetensors"}})
    with pytest.raises(ValueError, match="local filename"):
        read_safetensors_index(path)
