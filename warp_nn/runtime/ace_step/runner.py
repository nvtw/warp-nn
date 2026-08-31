# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step 1.5 bundle discovery and dependency-free conditioning inputs.

This module intentionally separates checkpoint/layout validation and host-side
conditioning from the GPU execution plans.  ``AceStep15Pipeline`` only reports
itself ready once its text encoder, DiT, and VAE executors have been attached;
constructing it never implies that an incomplete model can generate audio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..tokenizers import Qwen3Tokenizer


DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
SFT_GENERATION_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"ACE-Step file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in ACE-Step file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"ACE-Step JSON object expected in {path}")
    return value


def _require_fields(data: dict, names: Sequence[str], label: str) -> None:
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"{label} config is missing {missing}")


@dataclass(frozen=True)
class Qwen3EmbeddingConfig:
    """Shape contract of ACE-Step's Qwen3-Embedding-0.6B component."""

    hidden_size: int
    intermediate_size: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    vocabulary_size: int
    max_sequence_length: int
    rms_norm_epsilon: float
    rope_theta: float

    @classmethod
    def load(cls, path: str | Path) -> Qwen3EmbeddingConfig:
        data = _read_json(Path(path))
        _require_fields(
            data,
            (
                "model_type",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "max_position_embeddings",
            ),
            "Qwen3 embedding",
        )
        if data["model_type"] != "qwen3":
            raise ValueError("ACE-Step text encoder must use the Qwen3 architecture")
        if data.get("hidden_act", "silu") != "silu" or data.get(
            "attention_bias", False
        ):
            raise ValueError("only bias-free SiLU Qwen3 embedding models are supported")
        query_heads = int(data["num_attention_heads"])
        kv_heads = int(data["num_key_value_heads"])
        hidden_size = int(data["hidden_size"])
        head_dim = int(data["head_dim"])
        # Qwen3 permits a head dimension independent of hidden_size; the official
        # 0.6B encoder uses 16 * 128 Q channels with hidden size 1024.
        if query_heads % kv_heads or head_dim <= 0:
            raise ValueError("invalid Qwen3 embedding head geometry")
        layers = int(data["num_hidden_layers"])
        layer_types = data.get("layer_types")
        if layer_types is not None and (
            len(layer_types) != layers or set(layer_types) != {"full_attention"}
        ):
            raise ValueError(
                "ACE-Step text encoder requires full-attention Qwen3 layers"
            )
        return cls(
            hidden_size=hidden_size,
            intermediate_size=int(data["intermediate_size"]),
            layers=layers,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            vocabulary_size=int(data["vocab_size"]),
            max_sequence_length=int(data["max_position_embeddings"]),
            rms_norm_epsilon=float(data.get("rms_norm_eps", 1.0e-6)),
            rope_theta=float(data.get("rope_theta", 1_000_000.0)),
        )


@dataclass(frozen=True)
class AceStepDiTConfig:
    """Validated shape contract for an ACE-Step 1.5 DiT checkpoint."""

    model_version: str
    hidden_size: int
    intermediate_size: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    input_channels: int
    text_hidden_size: int
    lyric_layers: int
    timbre_layers: int
    audio_decoder_layers: int
    patch_size: int
    sliding_window: int | None
    is_turbo: bool

    @classmethod
    def load(cls, path: str | Path) -> AceStepDiTConfig:
        data = _read_json(Path(path))
        _require_fields(
            data,
            (
                "model_type",
                "model_version",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "in_channels",
                "text_hidden_dim",
                "num_lyric_encoder_hidden_layers",
                "num_timbre_encoder_hidden_layers",
                "num_audio_decoder_hidden_layers",
                "patch_size",
            ),
            "ACE-Step DiT",
        )
        if data["model_type"] != "acestep":
            raise ValueError("ACE-Step DiT config has an incompatible model_type")
        query_heads = int(data["num_attention_heads"])
        kv_heads = int(data["num_key_value_heads"])
        hidden_size = int(data["hidden_size"])
        head_dim = int(data["head_dim"])
        if query_heads % kv_heads or query_heads * head_dim != hidden_size:
            raise ValueError("invalid ACE-Step DiT head geometry")
        layers = int(data["num_hidden_layers"])
        layer_types = data.get("layer_types")
        if layer_types is not None and len(layer_types) != layers:
            raise ValueError(
                "ACE-Step DiT layer_types does not match num_hidden_layers"
            )
        is_turbo = bool(data.get("is_turbo", data["model_version"] == "turbo"))
        return cls(
            model_version=str(data["model_version"]),
            hidden_size=hidden_size,
            intermediate_size=int(data["intermediate_size"]),
            layers=layers,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            input_channels=int(data["in_channels"]),
            text_hidden_size=int(data["text_hidden_dim"]),
            lyric_layers=int(data["num_lyric_encoder_hidden_layers"]),
            timbre_layers=int(data["num_timbre_encoder_hidden_layers"]),
            audio_decoder_layers=int(data["num_audio_decoder_hidden_layers"]),
            patch_size=int(data["patch_size"]),
            sliding_window=(
                int(data["sliding_window"])
                if data.get("use_sliding_window", False)
                else None
            ),
            is_turbo=is_turbo,
        )


@dataclass(frozen=True)
class OobleckVAEConfig:
    """Validated audio geometry for the official stereo Oobleck VAE."""

    sampling_rate: int
    audio_channels: int
    encoder_hidden_size: int
    decoder_input_channels: int
    decoder_channels: int
    channel_multiples: tuple[int, ...]
    sampling_ratios: tuple[int, ...]

    @property
    def samples_per_latent(self) -> int:
        return int(np.prod(self.sampling_ratios))

    @classmethod
    def load(cls, path: str | Path) -> OobleckVAEConfig:
        data = _read_json(Path(path))
        _require_fields(
            data,
            (
                "_class_name",
                "sampling_rate",
                "audio_channels",
                "encoder_hidden_size",
                "decoder_input_channels",
                "decoder_channels",
                "channel_multiples",
                "downsampling_ratios",
            ),
            "Oobleck VAE",
        )
        if data["_class_name"] != "AutoencoderOobleck":
            raise ValueError("ACE-Step VAE must use AutoencoderOobleck")
        multiples = tuple(map(int, data["channel_multiples"]))
        ratios = tuple(map(int, data["downsampling_ratios"]))
        if not multiples or len(multiples) != len(ratios) or min(ratios) <= 0:
            raise ValueError("invalid Oobleck VAE channel or sampling stages")
        return cls(
            sampling_rate=int(data["sampling_rate"]),
            audio_channels=int(data["audio_channels"]),
            encoder_hidden_size=int(data["encoder_hidden_size"]),
            decoder_input_channels=int(data["decoder_input_channels"]),
            decoder_channels=int(data["decoder_channels"]),
            channel_multiples=multiples,
            sampling_ratios=ratios,
        )


@dataclass(frozen=True)
class AceStep15Bundle:
    """Paths and configs for one official ACE-Step 1.5 checkpoint bundle."""

    root: Path
    variant: str
    text_encoder_path: Path
    dit_path: Path
    vae_path: Path
    planner_path: Path | None
    text: Qwen3EmbeddingConfig
    dit: AceStepDiTConfig
    vae: OobleckVAEConfig

    @classmethod
    def discover(
        cls,
        root: str | Path,
        *,
        variant: str = "acestep-v15-turbo",
        require_planner: bool = False,
        validate_weights: bool = True,
    ) -> AceStep15Bundle:
        """Discover the official multi-component layout without loading tensors."""
        root = Path(root).expanduser().resolve()
        text_path = root / "Qwen3-Embedding-0.6B"
        dit_path = root / variant
        vae_path = root / "vae"
        planner = root / "acestep-5Hz-lm-1.7B"
        if require_planner and not planner.is_dir():
            raise FileNotFoundError(f"ACE-Step 5 Hz planner not found: {planner}")
        if validate_weights:
            required = (
                text_path / "model.safetensors",
                text_path / "tokenizer.json",
                dit_path / "model.safetensors",
                dit_path / "silence_latent.pt",
                vae_path / "diffusion_pytorch_model.safetensors",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"ACE-Step bundle is incomplete: {missing}")
        text = Qwen3EmbeddingConfig.load(text_path / "config.json")
        dit = AceStepDiTConfig.load(dit_path / "config.json")
        vae = OobleckVAEConfig.load(vae_path / "config.json")
        if text.hidden_size != dit.text_hidden_size:
            raise ValueError("ACE-Step text encoder and DiT hidden sizes do not match")
        if vae.decoder_input_channels != dit.input_channels // 3:
            raise ValueError(
                "ACE-Step VAE latent channels do not match DiT input packing"
            )
        return cls(
            root=root,
            variant=variant,
            text_encoder_path=text_path,
            dit_path=dit_path,
            vae_path=vae_path,
            planner_path=planner if planner.is_dir() else None,
            text=text,
            dit=dit,
            vae=vae,
        )


@dataclass(frozen=True)
class AceStepTokenBatch:
    """Padded token IDs and masks consumed by ACE-Step's conditioning encoders."""

    text_ids: np.ndarray
    text_mask: np.ndarray
    lyric_ids: np.ndarray
    lyric_mask: np.ndarray
    prompts: tuple[str, ...]
    lyric_prompts: tuple[str, ...]


def format_text_prompt(
    caption: str,
    *,
    instruction: str = DEFAULT_DIT_INSTRUCTION,
    metadata: str = "",
) -> str:
    """Format a caption exactly as the official ACE-Step 1.5 DiT expects."""
    if not instruction.endswith(":"):
        instruction += ":"
    return SFT_GENERATION_PROMPT.format(instruction, caption, metadata)


def format_lyrics(lyrics: str, language: str = "en") -> str:
    """Format lyrics exactly as the official ACE-Step 1.5 pipeline expects."""
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"


def _padded_tokens(
    tokenizer: Qwen3Tokenizer, texts: Sequence[str], maximum: int
) -> tuple[np.ndarray, np.ndarray]:
    encoded = [tokenizer.encode(text)[:maximum] for text in texts]
    if not encoded or any(not ids for ids in encoded):
        raise ValueError(
            "ACE-Step conditioning text must not be empty after tokenization"
        )
    width = max(map(len, encoded))
    ids = np.full((len(encoded), width), tokenizer.pad_token_id, dtype=np.int64)
    mask = np.zeros((len(encoded), width), dtype=bool)
    for row, values in enumerate(encoded):
        ids[row, : len(values)] = values
        mask[row, : len(values)] = True
    return ids, mask


def prepare_conditioning_tokens(
    tokenizer: Qwen3Tokenizer,
    captions: Sequence[str],
    lyrics: Sequence[str],
    *,
    instructions: Sequence[str] | None = None,
    metadata: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    max_text_tokens: int = 256,
    max_lyric_tokens: int = 2048,
) -> AceStepTokenBatch:
    """Build the caption and lyric branches without Torch or Transformers."""
    count = len(captions)
    if not count or len(lyrics) != count:
        raise ValueError("captions and lyrics must have the same nonzero batch size")

    def values(items, default, label):
        if items is None:
            return (default,) * count
        if len(items) != count:
            raise ValueError(f"{label} must match the conditioning batch size")
        return tuple(items)

    instructions = values(instructions, DEFAULT_DIT_INSTRUCTION, "instructions")
    metadata = values(metadata, "", "metadata")
    languages = values(languages, "en", "languages")
    prompts = tuple(
        format_text_prompt(caption, instruction=instruction, metadata=meta)
        for caption, instruction, meta in zip(captions, instructions, metadata)
    )
    lyric_prompts = tuple(
        format_lyrics(text, language) for text, language in zip(lyrics, languages)
    )
    text_ids, text_mask = _padded_tokens(tokenizer, prompts, max_text_tokens)
    lyric_ids, lyric_mask = _padded_tokens(tokenizer, lyric_prompts, max_lyric_tokens)
    return AceStepTokenBatch(
        text_ids=text_ids,
        text_mask=text_mask,
        lyric_ids=lyric_ids,
        lyric_mask=lyric_mask,
        prompts=prompts,
        lyric_prompts=lyric_prompts,
    )


def pack_conditioning_sequences(
    first: np.ndarray,
    second: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stable-pack two padded sequences with every valid item before padding."""
    first = np.asarray(first)
    second = np.asarray(second)
    first_mask = np.asarray(first_mask, dtype=bool)
    second_mask = np.asarray(second_mask, dtype=bool)
    if first.ndim != 3 or second.ndim != 3 or first.shape[0] != second.shape[0]:
        raise ValueError("conditioning hidden states must be [batch, sequence, hidden]")
    if first.shape[2] != second.shape[2]:
        raise ValueError("conditioning hidden sizes must match")
    if first_mask.shape != first.shape[:2] or second_mask.shape != second.shape[:2]:
        raise ValueError("conditioning masks must match their hidden-state sequences")
    hidden = np.concatenate((first, second), axis=1)
    mask = np.concatenate((first_mask, second_mask), axis=1)
    order = np.argsort(~mask, axis=1, kind="stable")
    packed = np.take_along_axis(hidden, order[..., None], axis=1)
    lengths = mask.sum(axis=1)
    packed_mask = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    return packed, packed_mask


def tile_silence_latent(silence: np.ndarray, frames: int) -> np.ndarray:
    """Return exactly ``frames`` channels-last silence latents, tiling if needed."""
    silence = np.asarray(silence)
    if silence.ndim == 3:
        if silence.shape[0] != 1:
            raise ValueError("silence latent batch dimension must be one")
        silence = silence[0]
    if silence.ndim != 2 or silence.shape[0] == 0:
        raise ValueError("silence latent must have shape [1, frames, channels]")
    if frames <= 0:
        raise ValueError("silence latent frame count must be positive")
    repeats = (frames + silence.shape[0] - 1) // silence.shape[0]
    return np.tile(silence, (repeats, 1))[:frames]


class AceStep15Pipeline:
    """Composition root for ACE-Step 1.5 executors.

    The caption executor must return the final hidden state of the complete
    causal Qwen3-Embedding model.  The lyric executor must gather only that
    checkpoint's input embeddings; ACE's own lyric encoder processes them next.
    """

    def __init__(
        self,
        bundle: AceStep15Bundle,
        *,
        text_executor: Callable | None = None,
        dit_executor: Callable | None = None,
        vae_decoder: Callable | None = None,
    ):
        self.bundle = bundle
        self.tokenizer = Qwen3Tokenizer(bundle.text_encoder_path)
        self.text_executor = text_executor
        self.dit_executor = dit_executor
        self.vae_decoder = vae_decoder

    @property
    def missing_components(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, value in (
                ("Qwen3 embedding encoder", self.text_executor),
                ("ACE-Step DiT", self.dit_executor),
                ("Oobleck VAE decoder", self.vae_decoder),
            )
            if value is None
        )

    @property
    def ready(self) -> bool:
        return not self.missing_components

    def generate(self, *args, **kwargs):
        if not self.ready:
            missing = ", ".join(self.missing_components)
            raise RuntimeError(f"ACE-Step 1.5 pipeline is not ready; missing {missing}")
        raise NotImplementedError(
            "ACE-Step sampling orchestration will be enabled with the DiT executor"
        )
