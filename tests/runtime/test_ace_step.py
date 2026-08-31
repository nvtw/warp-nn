# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from warp_nn.runtime.ace_step.runner import (
    AceStep15Bundle,
    AceStep15Pipeline,
    AceStepDiTConfig,
    OobleckVAEConfig,
    Qwen3EmbeddingConfig,
    format_lyrics,
    format_text_prompt,
    load_silence_latent,
    pack_conditioning_sequences,
    prepare_conditioning_tokens,
    seeded_normal,
    text_to_music_inputs,
    tile_silence_latent,
)
from warp_nn.runtime.tokenizers import Qwen3Tokenizer, _BYTE_ENCODER


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _text_config() -> dict:
    return {
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151669,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1.0e-6,
        "rope_theta": 1_000_000,
        "layer_types": ["full_attention"] * 28,
    }


def _dit_config() -> dict:
    return {
        "model_type": "acestep",
        "model_version": "turbo",
        "is_turbo": True,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "in_channels": 192,
        "text_hidden_dim": 1024,
        "num_lyric_encoder_hidden_layers": 8,
        "num_timbre_encoder_hidden_layers": 4,
        "num_audio_decoder_hidden_layers": 24,
        "patch_size": 2,
        "use_sliding_window": True,
        "sliding_window": 128,
        "layer_types": ["sliding_attention", "full_attention"] * 12,
    }


def _vae_config() -> dict:
    return {
        "_class_name": "AutoencoderOobleck",
        "sampling_rate": 48000,
        "audio_channels": 2,
        "encoder_hidden_size": 128,
        "decoder_input_channels": 64,
        "decoder_channels": 128,
        "channel_multiples": [1, 2, 4, 8, 16],
        "downsampling_ratios": [2, 4, 4, 6, 10],
    }


def _write_tokenizer(path: Path) -> Qwen3Tokenizer:
    vocabulary = {
        character: index for index, character in enumerate(_BYTE_ENCODER.values())
    }
    added = len(vocabulary)
    _write_json(
        path / "tokenizer.json",
        {
            "normalizer": {"type": "NFC"},
            "model": {"type": "BPE", "vocab": vocabulary, "merges": []},
            "added_tokens": [
                {"id": added, "content": "<|endoftext|>", "special": True},
                {"id": added + 1, "content": "<|im_end|>", "special": True},
            ],
        },
    )
    _write_json(
        path / "tokenizer_config.json",
        {"pad_token": "<|endoftext|>", "eos_token": "<|endoftext|>"},
    )
    return Qwen3Tokenizer(path)


def _write_bundle(path: Path) -> None:
    text = path / "Qwen3-Embedding-0.6B"
    dit = path / "acestep-v15-turbo"
    vae = path / "vae"
    _write_json(text / "config.json", _text_config())
    _write_json(dit / "config.json", _dit_config())
    _write_json(vae / "config.json", _vae_config())
    _write_tokenizer(text)
    for item in (
        text / "model.safetensors",
        dit / "model.safetensors",
        dit / "silence_latent.pt",
        vae / "diffusion_pytorch_model.safetensors",
    ):
        item.touch()


def test_official_component_configs(tmp_path):
    value = _text_config()
    value["num_key_value_heads"] = 6
    _write_json(tmp_path / "invalid.json", value)
    with pytest.raises(ValueError, match="head geometry"):
        Qwen3EmbeddingConfig.load(tmp_path / "invalid.json")


def test_bundle_discovery_and_compatibility(tmp_path):
    _write_bundle(tmp_path)
    bundle = AceStep15Bundle.discover(tmp_path)
    assert bundle.text.hidden_size == 1024
    assert bundle.dit.is_turbo
    assert bundle.dit.sliding_window == 128
    assert bundle.vae.samples_per_latent == 1920
    assert bundle.planner_path is None

    with pytest.raises(FileNotFoundError, match="5 Hz planner"):
        AceStep15Bundle.discover(tmp_path, require_planner=True)

    dit_config = _dit_config()
    dit_config["text_hidden_dim"] = 768
    _write_json(tmp_path / "acestep-v15-turbo" / "config.json", dit_config)
    with pytest.raises(ValueError, match="hidden sizes"):
        AceStep15Bundle.discover(tmp_path)


def test_component_config_values(tmp_path):
    _write_json(tmp_path / "text.json", _text_config())
    _write_json(tmp_path / "dit.json", _dit_config())
    _write_json(tmp_path / "vae.json", _vae_config())
    text = Qwen3EmbeddingConfig.load(tmp_path / "text.json")
    dit = AceStepDiTConfig.load(tmp_path / "dit.json")
    vae = OobleckVAEConfig.load(tmp_path / "vae.json")
    assert (text.layers, text.query_heads, text.kv_heads) == (28, 16, 8)
    assert (dit.layers, dit.lyric_layers, dit.audio_decoder_layers) == (24, 8, 24)
    assert (vae.sampling_rate, vae.audio_channels) == (48000, 2)


def test_prompt_format_and_token_batch(tmp_path):
    tokenizer = _write_tokenizer(tmp_path)
    prompt = format_text_prompt(
        "dreamy synthwave", instruction="Compose", metadata="bpm: 120"
    )
    assert prompt == (
        "# Instruction\nCompose:\n\n# Caption\ndreamy synthwave\n\n"
        "# Metas\nbpm: 120<|endoftext|>\n"
    )
    assert format_lyrics("hello", "en") == (
        "# Languages\nen\n\n# Lyric\nhello<|endoftext|>"
    )
    batch = prepare_conditioning_tokens(
        tokenizer,
        ["short", "a much longer caption"],
        ["la", "la la"],
        languages=["en", "de"],
    )
    assert batch.text_ids.shape == batch.text_mask.shape
    assert batch.lyric_ids.shape == batch.lyric_mask.shape
    assert batch.text_ids.dtype == np.int64
    assert batch.text_mask.dtype == bool
    assert np.all(batch.text_ids[~batch.text_mask] == tokenizer.pad_token_id)
    assert "# Languages\nde" in batch.lyric_prompts[1]


def test_pack_conditioning_sequences_is_stable():
    first = np.array([[[1.0], [99.0], [2.0]]])
    second = np.array([[[3.0], [98.0]]])
    first_mask = np.array([[True, False, True]])
    second_mask = np.array([[True, False]])
    packed, mask = pack_conditioning_sequences(first, second, first_mask, second_mask)
    np.testing.assert_array_equal(packed[0, :, 0], [1.0, 2.0, 3.0, 99.0, 98.0])
    np.testing.assert_array_equal(mask, [[True, True, True, False, False]])


def test_silence_latent_slicing_and_tiling():
    silence = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    np.testing.assert_array_equal(
        tile_silence_latent(silence, 7),
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2]],
    )
    with pytest.raises(ValueError, match="positive"):
        tile_silence_latent(silence, 0)


def test_text_to_music_inputs_use_exact_25hz_context():
    silence = np.arange(3 * 64, dtype=np.float32).reshape(1, 3, 64)
    first = text_to_music_inputs(silence, 0.101, batch_size=2)
    assert first.source_latents.shape == (2, 3, 64)
    assert first.context_latents.shape == (2, 3, 128)
    assert first.timbre_latents.shape == (2, 750, 64)
    np.testing.assert_array_equal(first.chunk_mask, 1.0)
    np.testing.assert_array_equal(first.context_latents[..., :64], first.source_latents)
    np.testing.assert_array_equal(first.context_latents[..., 64:], 1.0)


def test_seeded_normal_is_deterministic_on_cpu():
    first = seeded_normal((2, 7), seed=41, dtype=wp.float32, device="cpu")
    second = seeded_normal((2, 7), seed=41, dtype=wp.float32, device="cpu")
    third = seeded_normal((2, 7), seed=42, dtype=wp.float32, device="cpu")
    np.testing.assert_array_equal(first.numpy(), second.numpy())
    assert not np.array_equal(first.numpy(), third.numpy())


def test_pipeline_never_claims_incomplete_execution_is_ready(tmp_path):
    _write_bundle(tmp_path)
    pipeline = AceStep15Pipeline(AceStep15Bundle.discover(tmp_path))
    assert not pipeline.ready
    assert pipeline.missing_components == (
        "Qwen3 embedding encoder",
        "ACE-Step condition encoder",
        "ACE-Step DiT",
        "Oobleck VAE decoder",
    )
    with pytest.raises(RuntimeError, match="not ready"):
        pipeline.generate(conditioning=None)


def test_pipeline_orchestrates_minimal_turbo_path_on_device(monkeypatch):
    from types import SimpleNamespace

    dtype = wp.float32
    conditioning = SimpleNamespace(
        text_hidden_states=wp.zeros((1, 2, 1024), dtype=dtype, device="cpu"),
        text_attention_mask=wp.ones((1, 2), dtype=wp.bool, device="cpu"),
        lyric_hidden_states=wp.zeros((1, 3, 1024), dtype=dtype, device="cpu"),
        lyric_attention_mask=wp.ones((1, 3), dtype=wp.bool, device="cpu"),
    )
    condition = wp.zeros((1, 6, 16), dtype=dtype, device="cpu")
    valid = wp.ones((1, 6), dtype=wp.bool, device="cpu")

    class Condition:
        def plan(self, text, text_mask, lyric, lyric_mask, reference):
            assert reference.shape == (1, 750, 64)
            return SimpleNamespace(execute=lambda: (condition, valid))

    class DiT:
        def __init__(self, hidden):
            self.hidden = hidden

        def run_schedule(self, schedule):
            assert len(schedule) == 4
            return self.hidden

    class Decoder:
        def __init__(self, frames):
            self.device = wp.get_device("cpu")
            self.input = wp.empty((1, frames, 64), dtype=dtype, device="cpu")
            self.output = wp.zeros((1, frames * 1920, 2), dtype=dtype, device="cpu")

        def execute(self):
            return self.output

    bundle = SimpleNamespace(
        text_encoder_path=Path("."),
        dit_path=Path("."),
    )
    monkeypatch.setattr(
        "warp_nn.runtime.ace_step.runner.Qwen3Tokenizer",
        lambda _path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "warp_nn.runtime.ace_step.runner.load_silence_latent",
        lambda *_args, **_kwargs: np.zeros((1, 4, 64), dtype=np.float32),
    )

    def dit_factory(hidden, context, packed, packed_valid):
        assert hidden.shape == (1, 6, 64)
        assert context.shape == (1, 6, 128)
        assert packed is condition and packed_valid is valid
        return DiT(hidden)

    pipeline = AceStep15Pipeline(
        bundle,
        text_executor=object(),
        condition_executor=Condition(),
        dit_executor=dit_factory,
        vae_decoder=lambda frames, batch: Decoder(frames),
    )
    audio = pipeline.generate(
        conditioning=conditioning,
        duration_seconds=0.21,
        seed=7,
        steps=4,
    )
    assert audio.shape == (1, 6 * 1920, 2)


def test_load_silence_latent_transposes_official_layout(monkeypatch):
    channel_first = np.arange(24, dtype=np.float32).reshape(1, 3, 8)
    monkeypatch.setattr(
        "warp_nn.runtime.ace_step.runner.load_pytorch_zip",
        lambda _path: channel_first,
    )
    loaded = load_silence_latent("silence_latent.pt", channels=3)
    assert loaded.shape == (1, 8, 3)
    assert loaded.flags.c_contiguous
    np.testing.assert_array_equal(loaded, channel_first.transpose(0, 2, 1))

    with pytest.raises(ValueError, match="expected 4"):
        load_silence_latent("silence_latent.pt", channels=4)


def test_pipeline_prepares_exact_gpu_conditioning_boundary(tmp_path):
    from types import SimpleNamespace

    import warp as wp

    from tests.runtime.test_qwen3_encoder import _tiny_checkpoint
    from warp_nn.runtime.qwen.encoder import Qwen3Encoder

    path = tmp_path / "qwen"
    _, weights = _tiny_checkpoint(path)
    bundle = SimpleNamespace(text_encoder_path=path)
    pipeline = AceStep15Pipeline(bundle)
    with pytest.raises(RuntimeError, match="not loaded"):
        tokens = prepare_conditioning_tokens(pipeline.tokenizer, ["first"], ["la"])
        pipeline.prepare_gpu_conditioning(tokens)

    encoder = Qwen3Encoder(path, dtype=wp.float16, device="cpu", use_cublas=False)
    pipeline.text_executor = encoder
    tokens = prepare_conditioning_tokens(
        pipeline.tokenizer,
        ["short", "a longer caption"],
        ["la", "la la"],
    )
    condition = pipeline.prepare_gpu_conditioning(tokens)
    assert condition.text_hidden_states.shape == (
        2,
        tokens.text_ids.shape[1],
        4,
    )
    assert condition.lyric_hidden_states.shape == (
        2,
        tokens.lyric_ids.shape[1],
        4,
    )
    np.testing.assert_array_equal(
        condition.text_attention_mask.numpy(), tokens.text_mask
    )
    np.testing.assert_array_equal(
        condition.lyric_attention_mask.numpy(), tokens.lyric_mask
    )
    np.testing.assert_allclose(
        condition.lyric_hidden_states.numpy(),
        weights["model.embed_tokens.weight"][tokens.lyric_ids],
        rtol=5.0e-4,
        atol=1.0e-4,
    )


def test_ace_cli_check_validates_without_loading_weights(tmp_path, capsys):
    from examples.ace_step import main

    _write_bundle(tmp_path)
    assert main([str(tmp_path), "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["variant"] == "acestep-v15-turbo"
    assert report["sample_rate"] == 48000
    assert report["samples_per_latent"] == 1920


def test_ace_cli_writes_stereo_pcm16_only_after_ready_generation(tmp_path, monkeypatch):
    import examples.ace_step as cli
    from warp_nn.runtime.formats.wav import read_wav_pcm16

    _write_bundle(tmp_path)
    generated = np.array([[[-1.0, 1.0], [0.25, -0.5]]], dtype=np.float32)

    class Pipeline:
        ready = True
        missing_components = ()

        def __init__(self, bundle):
            self.bundle = bundle

        def load_generation_stack(self, **kwargs):
            return self

        def prepare_conditioning(self, *args, **kwargs):
            return "conditioning"

        def generate(self, *, conditioning, duration_seconds, seed, steps):
            assert conditioning == "conditioning"
            assert (duration_seconds, seed, steps) == (30.0, 0, 8)
            return generated

    monkeypatch.setattr(cli, "AceStep15Pipeline", Pipeline)
    output = tmp_path / "result.wav"
    assert cli.main([str(tmp_path), "--prompt", "music", "--output", str(output)]) == 0
    audio = read_wav_pcm16(output)
    assert audio.sample_rate == 48_000
    np.testing.assert_allclose(audio.samples, generated[0], atol=1.0 / 32767.0)
