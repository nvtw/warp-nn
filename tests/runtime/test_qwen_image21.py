# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace

import pytest

from warp_nn.runtime.qwen_image import QwenImage21Bundle, QwenImage21Pipeline


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


def test_pipeline_uses_cached_local_reference_and_preserves_rgba(tmp_path, monkeypatch):
    root = _bundle(tmp_path)
    bundle = QwenImage21Bundle.inspect(root)
    for missing in bundle.missing_weight_files():
        missing.touch()

    calls = {}
    image = object()

    class Reference:
        def __init__(self):
            self.transformer = SimpleNamespace(
                compile=lambda: calls.setdefault("compiled", True)
            )

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["load"] = (path, kwargs)
            return cls()

        def to(self, device):
            calls["device"] = str(device)
            return self

        def set_progress_bar_config(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            calls["generate"] = kwargs
            return SimpleNamespace(images=[image])

    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(QwenImage21Pipeline=Reference)
    )
    pipeline = QwenImage21Pipeline(bundle, device="cpu")
    assert (
        pipeline.generate("a glass bird", width=512, height=512, steps=3, seed=42)
        is image
    )
    assert calls["load"][0] == str(root)
    assert calls["load"][1]["local_files_only"] is True
    assert calls["generate"]["use_kv_cache"] is True
    assert calls["generate"]["true_cfg_scale"] == 1.0
    assert calls["generate"]["generator"].initial_seed() == 42
    pipeline.generate("edit", image=[object()])
    assert calls["generate"]["width"] is None
    assert calls["generate"]["height"] is None
    QwenImage21Pipeline(bundle, device="cpu", compile=True)
    assert calls["compiled"] is True


def test_example_reuses_pipeline_for_multiple_outputs(tmp_path, monkeypatch):
    import examples.qwen_image21 as example

    bundle = QwenImage21Bundle.inspect(_bundle(tmp_path / "model"))
    monkeypatch.setattr(
        example.QwenImage21Bundle, "inspect", lambda *args, **kwargs: bundle
    )
    loaded = []
    seeds = []

    class Image:
        mode = "RGBA"

        def save(self, path):
            path.write_bytes(b"png")

    class Pipeline:
        def __init__(self, *args, **kwargs):
            loaded.append(1)

        def generate(self, prompt, **kwargs):
            seeds.append(kwargs["seed"])
            return Image()

    monkeypatch.setattr(example, "QwenImage21Pipeline", Pipeline)
    output = tmp_path / "out.png"
    assert (
        example.main(
            [
                str(bundle.root),
                "--prompt",
                "a bird",
                "--width",
                "512",
                "--height",
                "512",
                "--repeat",
                "2",
                "--seed",
                "5",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert loaded == [1]
    assert seeds == [5, 6]
    assert (tmp_path / "out-000.png").read_bytes() == b"png"
    assert (tmp_path / "out-001.png").read_bytes() == b"png"
