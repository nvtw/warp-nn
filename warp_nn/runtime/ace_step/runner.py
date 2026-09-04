# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step 1.5 bundle discovery and dependency-free conditioning inputs.

This module intentionally separates checkpoint/layout validation and host-side
conditioning from the GPU execution plans.  ``AceStep15Pipeline`` only reports
itself ready once its text encoder, DiT, and VAE executors have been attached;
constructing it never implies that an incomplete model can generate audio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import warp as wp

from ..formats.pytorch import load_pytorch_zip
from ..kernels import _cast_kernel_for_dtypes
from ..operators import seeded_normal
from ..qwen.encoder import Qwen3Encoder, load_qwen3_encoder_config
from ..tokenizers import Qwen3Tokenizer
from .._cublas import try_create_cublas
from .dit import AceStepDiTConfig
from .vae import OobleckVAEConfig


DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
SFT_GENERATION_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""


@lru_cache(maxsize=None)
def _semantic_context_kernel(dtype):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def pack(
        hints: wp.array3d(dtype=DTYPE),
        fallback: wp.array3d(dtype=DTYPE),
        context: wp.array3d(dtype=DTYPE),
    ):
        batch, frame, channel = wp.tid()
        if channel < 64:
            if frame < hints.shape[1]:
                context[batch, frame, channel] = hints[0, frame, channel]
            else:
                context[batch, frame, channel] = fallback[batch, frame, channel]
        else:
            context[batch, frame, channel] = DTYPE(1.0)

    return pack


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
        try:
            data = load_qwen3_encoder_config(path)
        except ValueError as error:
            if "heads" in str(error):
                raise ValueError("invalid Qwen3 embedding head geometry") from error
            raise
        return cls(
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            layers=int(data["num_hidden_layers"]),
            query_heads=int(data["num_attention_heads"]),
            kv_heads=int(data["num_key_value_heads"]),
            head_dim=int(data["head_dim"]),
            vocabulary_size=int(data["vocab_size"]),
            max_sequence_length=int(data["max_position_embeddings"]),
            rms_norm_epsilon=float(data.get("rms_norm_eps", 1.0e-6)),
            rope_theta=float(data.get("rope_theta", 1_000_000.0)),
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
        base = root / "Ace-Step1.5" if (root / "Ace-Step1.5").is_dir() else root
        locations = tuple(dict.fromkeys((root, base, root.parent, base.parent)))

        def component(name, *, required=True):
            path = next(
                (
                    directory / name
                    for directory in locations
                    if (directory / name).is_dir()
                ),
                None,
            )
            if path is None and required:
                raise FileNotFoundError(f"ACE-Step component not found: {name}")
            return path

        text_path = component("Qwen3-Embedding-0.6B")
        dit_path = component(variant)
        vae_path = component("vae")
        planner = component("acestep-5Hz-lm-4B", required=False)
        if planner is None:
            planner = component("acestep-5Hz-lm-1.7B", required=False)
        if require_planner and planner is None:
            raise FileNotFoundError("ACE-Step 5 Hz planner not found")
        if validate_weights:
            required = [
                text_path / "model.safetensors",
                text_path / "tokenizer.json",
                dit_path / "silence_latent.pt",
                vae_path / "diffusion_pytorch_model.safetensors",
            ]
            if not (
                (dit_path / "model.safetensors").is_file()
                or (dit_path / "model.safetensors.index.json").is_file()
            ):
                required.append(dit_path / "model.safetensors[.index.json]")
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
            root=base,
            variant=variant,
            text_encoder_path=text_path,
            dit_path=dit_path,
            vae_path=vae_path,
            planner_path=planner,
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


def format_dit_metadata(metadata: dict[str, str], duration_seconds: float) -> str:
    """Format the four checkpoint-native ACE DiT metadata fields."""
    duration = metadata.get("duration", f"{duration_seconds:g}")
    try:
        duration = f"{int(float(duration))} seconds"
    except (TypeError, ValueError):
        pass
    return (
        f"- bpm: {metadata.get('bpm', 'N/A')}\n"
        f"- timesignature: {metadata.get('timesignature', 'N/A')}\n"
        f"- keyscale: {metadata.get('keyscale', 'N/A')}\n"
        f"- duration: {duration}\n"
    )


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


def load_silence_latent(path: str | Path, *, channels: int | None = None) -> np.ndarray:
    """Load ACE's channel-first ``.pt`` tensor as ``[1, frames, channels]``."""
    silence = load_pytorch_zip(path)
    if not isinstance(silence, np.ndarray) or silence.ndim != 3:
        raise ValueError("ACE-Step silence latent must be one tensor with three axes")
    if silence.shape[0] != 1:
        raise ValueError("ACE-Step silence latent batch dimension must be one")
    if channels is not None and silence.shape[1] != channels:
        raise ValueError(
            f"ACE-Step silence latent has {silence.shape[1]} channels, expected {channels}"
        )
    return np.ascontiguousarray(silence.transpose(0, 2, 1))


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


@dataclass(frozen=True)
class AceStepConditioning:
    """GPU-resident raw inputs for ACE's model-owned condition encoder.

    Caption states have passed through the complete causal Qwen3 model. Lyric
    states are input embeddings only; the DiT checkpoint's lyric encoder must
    process them before condition packing and cross-attention.
    """

    text_hidden_states: wp.array
    text_attention_mask: wp.array
    lyric_hidden_states: wp.array
    lyric_attention_mask: wp.array
    prompts: tuple[str, ...]
    lyric_prompts: tuple[str, ...]

    def row(self, index: int) -> "AceStepConditioning":
        """Return one batch row while preserving its fixed padded shape."""
        if not 0 <= index < self.text_hidden_states.shape[0]:
            raise IndexError("ACE-Step conditioning row is out of range")
        return AceStepConditioning(
            text_hidden_states=self.text_hidden_states[index : index + 1],
            text_attention_mask=self.text_attention_mask[index : index + 1],
            lyric_hidden_states=self.lyric_hidden_states[index : index + 1],
            lyric_attention_mask=self.lyric_attention_mask[index : index + 1],
            prompts=(self.prompts[index],),
            lyric_prompts=(self.lyric_prompts[index],),
        )


@dataclass(frozen=True)
class AceStepTextToMusicInputs:
    """Host-side fixed-shape inputs for ordinary turbo text-to-music."""

    source_latents: np.ndarray
    chunk_mask: np.ndarray
    context_latents: np.ndarray
    timbre_latents: np.ndarray


def text_to_music_inputs(
    silence: np.ndarray,
    duration_seconds: float,
    *,
    batch_size: int = 1,
    latent_rate: int = 25,
    timbre_frames: int = 750,
) -> AceStepTextToMusicInputs:
    """Construct the official non-cover source, context, and silence timbre."""
    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("ACE-Step duration must be a positive finite value")
    if batch_size <= 0 or latent_rate <= 0 or timbre_frames <= 0:
        raise ValueError("ACE-Step batch and frame rates must be positive")
    values = np.asarray(silence)
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError("silence latent batch dimension must be one")
        values = values[0]
    if values.ndim != 2 or values.shape[1] != 64:
        raise ValueError("ACE-Step silence latent must have 64 channels")
    frames = max(1, int(math.ceil(duration_seconds * latent_rate)))
    source_row = tile_silence_latent(values, frames).astype(np.float32, copy=False)
    timbre_row = tile_silence_latent(values, timbre_frames).astype(
        np.float32, copy=False
    )
    source = np.broadcast_to(source_row, (batch_size, *source_row.shape)).copy()
    timbre = np.broadcast_to(timbre_row, (batch_size, *timbre_row.shape)).copy()
    chunk = np.ones_like(source)
    context = np.concatenate((source, chunk), axis=-1)
    return AceStepTextToMusicInputs(
        source_latents=source,
        chunk_mask=chunk,
        context_latents=context,
        timbre_latents=timbre,
    )


class AceStep15Pipeline:
    """Composition root for ACE-Step 1.5 executors.

    Construction validates paths but does not upload weights. Attach the exact
    Qwen3 encoder with :meth:`load_text_encoder`; DiT and VAE executors remain
    explicit so ``ready`` never overstates end-to-end support.
    """

    def __init__(
        self,
        bundle: AceStep15Bundle,
        *,
        text_executor: Qwen3Encoder | None = None,
        condition_executor: Callable | None = None,
        dit_executor: Callable | None = None,
        vae_decoder: Callable | None = None,
        planner_executor=None,
        audio_code_decoder=None,
    ):
        self.bundle = bundle
        self.tokenizer = Qwen3Tokenizer(bundle.text_encoder_path)
        self.text_executor = text_executor
        self.condition_executor = condition_executor
        self.dit_executor = dit_executor
        self.vae_decoder = vae_decoder
        self.planner_executor = planner_executor
        self.audio_code_decoder = audio_code_decoder

    def load_text_encoder(
        self,
        *,
        dtype=wp.bfloat16,
        device=None,
        use_cublas: bool = True,
    ) -> Qwen3Encoder:
        """Load and attach ACE's exact caption/lyric Qwen3 executor."""
        encoder = Qwen3Encoder(
            self.bundle.text_encoder_path,
            dtype=dtype,
            device=device,
            use_cublas=use_cublas,
        )
        self.text_executor = encoder
        return encoder

    def load_generation_stack(
        self,
        *,
        dtype=wp.bfloat16,
        device=None,
        use_cublas: bool = True,
    ):
        """Load all fixed weights needed by ACE-Step generation."""
        from .conditioning import AceStepConditionEncoder
        from .dit import (
            AceStepDiTPlan,
            AceStepGuidedDiTPlan,
            load_ace_dit_weights,
        )
        from .vae import OobleckVAEDecoder

        encoder = self.load_text_encoder(
            dtype=dtype, device=device, use_cublas=use_cublas
        )
        config = self.bundle.dit
        condition = AceStepConditionEncoder(
            self.bundle.dit_path,
            config,
            dtype=dtype,
            device=encoder.device,
            use_cublas=use_cublas,
        )
        weights = load_ace_dit_weights(
            self.bundle.dit_path, config, encoder.device, dtype
        )
        cublas = try_create_cublas() if use_cublas and encoder.device.is_cuda else None

        def dit_factory(
            hidden,
            context,
            packed_condition,
            valid,
            null_condition=None,
            guidance_scale=7.0,
            *,
            non_cover_context=None,
            non_cover_condition=None,
            non_cover_valid=None,
            audio_cover_strength=1.0,
        ):
            if config.is_turbo:
                return AceStepDiTPlan(
                    hidden,
                    context,
                    packed_condition,
                    weights,
                    config,
                    condition_valid=valid,
                    cublas=cublas,
                )
            return AceStepGuidedDiTPlan(
                hidden,
                context,
                packed_condition,
                null_condition,
                weights,
                config,
                condition_valid=valid,
                cublas=cublas,
                guidance_scale=guidance_scale,
                non_cover_context=non_cover_context,
                non_cover_condition=non_cover_condition,
                non_cover_valid=non_cover_valid,
                audio_cover_strength=audio_cover_strength,
            )

        vae_cache = {}

        def vae_factory(frames, batch):
            key = (frames, batch)
            if key not in vae_cache:
                vae_cache.clear()
                vae_cache[key] = OobleckVAEDecoder.from_pretrained(
                    self.bundle.vae_path,
                    frames,
                    batch_size=batch,
                    device=encoder.device,
                    dtype=dtype,
                )
            return vae_cache[key]

        self.condition_executor = condition
        self.dit_executor = dit_factory
        self.vae_decoder = vae_factory
        return self

    def load_planner(
        self,
        *,
        dtype=wp.bfloat16,
        device=None,
        cache_capacity: int = 4096,
        prefill_chunk_size: int = 16,
        use_cublas: bool = True,
    ):
        """Load the optional 5 Hz composition planner and semantic decoder."""
        if self.bundle.planner_path is None:
            raise FileNotFoundError("ACE-Step 5 Hz planner not found")
        from ..qwen.causal import Qwen3CausalLM
        from .planner import AceAudioCodeDecoder, AceStepPlanner

        runner = Qwen3CausalLM(
            self.bundle.planner_path,
            dtype=dtype,
            device=device,
            cache_capacity=cache_capacity,
            prefill_chunk_size=prefill_chunk_size,
            use_cublas=use_cublas,
        )
        self.planner_executor = AceStepPlanner(runner)
        self.audio_code_decoder = AceAudioCodeDecoder(
            self.bundle.dit_path,
            dtype=dtype,
            device=runner.device,
            use_cublas=use_cublas,
        )
        return self.planner_executor

    def plan_music(
        self,
        caption: str,
        lyrics: str,
        *,
        duration_seconds: float,
        **sampling_options,
    ):
        """Generate the metadata and 5 Hz semantic plan used by LM-DiT."""
        if self.planner_executor is None:
            raise RuntimeError("ACE-Step 5 Hz planner is not loaded")
        return self.planner_executor.generate(
            caption, lyrics, duration_seconds=duration_seconds, **sampling_options
        )

    def prepare_gpu_conditioning(
        self, tokens: AceStepTokenBatch
    ) -> AceStepConditioning:
        """Encode one padded conditioning batch entirely on the target GPU.

        Qwen's reusable executor is currently optimized for one causal sequence,
        so batch rows execute through one graph-cached plan and are copied into
        a contiguous output. No hidden states round-trip through host memory.
        """
        encoder = self.text_executor
        if encoder is None:
            raise RuntimeError("ACE-Step Qwen3 embedding encoder is not loaded")
        if not isinstance(encoder, Qwen3Encoder):
            raise TypeError("ACE-Step GPU conditioning requires Qwen3Encoder")
        text_ids = np.asarray(tokens.text_ids, dtype=np.int64)
        lyric_ids = np.asarray(tokens.lyric_ids, dtype=np.int64)
        text_mask = np.asarray(tokens.text_mask, dtype=bool)
        lyric_mask = np.asarray(tokens.lyric_mask, dtype=bool)
        if text_ids.ndim != 2 or lyric_ids.ndim != 2:
            raise ValueError("ACE-Step token batches must be rank two")
        if text_mask.shape != text_ids.shape or lyric_mask.shape != lyric_ids.shape:
            raise ValueError("ACE-Step token masks must match token IDs")
        if text_ids.shape[0] != lyric_ids.shape[0]:
            raise ValueError("ACE-Step caption and lyric batch sizes must match")
        batch, text_length = text_ids.shape
        text_hidden = wp.empty(
            (batch, text_length, encoder.hidden_size),
            dtype=encoder.dtype,
            device=encoder.device,
        )
        row_values = text_length * encoder.hidden_size
        for row in range(batch):
            encoded = encoder.encode_ids(text_ids[row])
            wp.copy(
                text_hidden.flatten(),
                encoded.flatten(),
                dest_offset=row * row_values,
                count=row_values,
            )
        lyric_hidden = encoder.embed_ids(lyric_ids)
        return AceStepConditioning(
            text_hidden_states=text_hidden,
            text_attention_mask=wp.array(
                text_mask, dtype=wp.bool, device=encoder.device
            ),
            lyric_hidden_states=lyric_hidden,
            lyric_attention_mask=wp.array(
                lyric_mask, dtype=wp.bool, device=encoder.device
            ),
            prompts=tokens.prompts,
            lyric_prompts=tokens.lyric_prompts,
        )

    def prepare_conditioning(
        self,
        captions: Sequence[str],
        lyrics: Sequence[str],
        **token_options,
    ) -> AceStepConditioning:
        """Tokenize and encode caption/lyric inputs into GPU-resident tensors."""
        tokens = prepare_conditioning_tokens(
            self.tokenizer, captions, lyrics, **token_options
        )
        return self.prepare_gpu_conditioning(tokens)

    @property
    def missing_components(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, value in (
                ("Qwen3 embedding encoder", self.text_executor),
                ("ACE-Step condition encoder", self.condition_executor),
                ("ACE-Step DiT", self.dit_executor),
                ("Oobleck VAE decoder", self.vae_decoder),
            )
            if value is None
        )

    @property
    def ready(self) -> bool:
        return not self.missing_components

    def generate_music(
        self,
        caption: str,
        lyrics: str = "",
        *,
        language: str = "en",
        metadata: str = "",
        instruction: str | None = None,
        duration_seconds: float = 30.0,
        seed: int = 0,
        steps: int | None = None,
        lm_codes_strength: float = 0.6,
        progress: Callable[[int, int], None] | None = None,
    ):
        """Plan, condition, and generate one song from user-facing text."""
        plan = None
        if self.planner_executor is not None:
            plan = self.plan_music(
                caption, lyrics, duration_seconds=duration_seconds, seed=seed
            )
            caption = plan.metadata.get("caption", caption)
            if not metadata:
                metadata = format_dit_metadata(plan.metadata, duration_seconds)

        non_cover_conditioning = None
        token_options = {"languages": [language], "metadata": [metadata]}
        if instruction is not None:
            token_options["instructions"] = [instruction]
        if plan is None or instruction is None:
            conditioning = self.prepare_conditioning(
                [caption], [lyrics], **token_options
            )
        else:
            paired = self.prepare_conditioning(
                [caption, caption],
                [lyrics, lyrics],
                languages=[language, language],
                metadata=[metadata, metadata],
                instructions=[instruction, DEFAULT_DIT_INSTRUCTION],
            )
            conditioning = paired.row(0)
            non_cover_conditioning = paired.row(1)

        options = {
            "conditioning": conditioning,
            "duration_seconds": duration_seconds,
            "seed": seed,
            "steps": steps,
            "progress": progress,
        }
        if plan is not None:
            options.update(
                audio_codes=plan.audio_codes,
                audio_cover_strength=lm_codes_strength,
                non_cover_conditioning=non_cover_conditioning,
            )
        return self.generate(**options), plan

    def generate(
        self,
        *,
        conditioning: AceStepConditioning,
        duration_seconds: float = 30.0,
        seed: int = 0,
        steps: int | None = None,
        guidance_scale: float = 7.0,
        audio_codes: Sequence[int] | None = None,
        audio_cover_strength: float = 1.0,
        non_cover_conditioning: AceStepConditioning | None = None,
        progress: Callable[[int, int], None] | None = None,
    ):
        """Generate stereo audio through condition encoder, DiT, and VAE."""
        if not self.ready:
            missing = ", ".join(self.missing_components)
            raise RuntimeError(f"ACE-Step 1.5 pipeline is not ready; missing {missing}")
        if steps is None:
            steps = 8 if self.bundle.dit.is_turbo else 50
        if steps <= 0 or (self.bundle.dit.is_turbo and steps > 8):
            limit = " between 1 and 8" if self.bundle.dit.is_turbo else " positive"
            raise ValueError(f"ACE-Step steps must be{limit}")
        if not 0.0 <= audio_cover_strength <= 1.0:
            raise ValueError("ACE-Step audio-cover strength must be within [0, 1]")
        from .dit import turbo_schedule

        batch = conditioning.text_hidden_states.shape[0]
        silence = load_silence_latent(
            self.bundle.dit_path / "silence_latent.pt", channels=64
        )
        inputs = text_to_music_inputs(silence, duration_seconds, batch_size=batch)
        device = conditioning.text_hidden_states.device
        dtype = conditioning.text_hidden_states.dtype
        reference = wp.array(inputs.timbre_latents, dtype=dtype, device=device)
        condition_plan = self.condition_executor.plan(
            conditioning.text_hidden_states,
            conditioning.text_attention_mask,
            conditioning.lyric_hidden_states,
            conditioning.lyric_attention_mask,
            reference,
        )
        packed_condition, condition_valid = condition_plan.execute()
        non_cover_packed = non_cover_valid = None
        if audio_codes is not None and audio_cover_strength < 1.0:
            if non_cover_conditioning is None:
                non_cover_packed, non_cover_valid = packed_condition, condition_valid
            else:
                if non_cover_conditioning.text_hidden_states.shape[0] != batch:
                    raise ValueError("ACE-Step non-cover condition batch must match")
                non_cover_plan = self.condition_executor.plan(
                    non_cover_conditioning.text_hidden_states,
                    non_cover_conditioning.text_attention_mask,
                    non_cover_conditioning.lyric_hidden_states,
                    non_cover_conditioning.lyric_attention_mask,
                    reference,
                )
                non_cover_packed, non_cover_valid = non_cover_plan.execute()
                if non_cover_packed.shape != packed_condition.shape:
                    raise ValueError(
                        "ACE-Step cover and non-cover conditions must share a padded shape"
                    )
        hidden = seeded_normal(
            inputs.source_latents.shape, seed=seed, dtype=dtype, device=device
        )
        if audio_codes is None:
            context = wp.array(inputs.context_latents, dtype=dtype, device=device)
        else:
            if self.audio_code_decoder is None:
                raise RuntimeError("ACE-Step audio-code decoder is not loaded")
            hints = self.audio_code_decoder.decode(audio_codes)
            if hints.shape[2] != inputs.source_latents.shape[2]:
                raise ValueError(
                    "ACE-Step audio-code hints must have 64 latent channels"
                )
            source = wp.array(inputs.source_latents, dtype=dtype, device=device)
            context = wp.empty(
                (batch, source.shape[1], source.shape[2] * 2),
                dtype=dtype,
                device=device,
            )
            wp.launch(
                _semantic_context_kernel(dtype),
                dim=context.shape,
                inputs=[hints, source, context],
                device=device,
            )
        if self.bundle.dit.is_turbo:
            dit = self.dit_executor(hidden, context, packed_condition, condition_valid)
        else:
            dit = self.dit_executor(
                hidden,
                context,
                packed_condition,
                condition_valid,
                condition_plan.null_condition(),
                guidance_scale,
                non_cover_context=wp.array(
                    inputs.context_latents, dtype=dtype, device=device
                )
                if non_cover_packed is not None
                else None,
                non_cover_condition=non_cover_packed,
                non_cover_valid=non_cover_valid,
                audio_cover_strength=audio_cover_strength,
            )
        schedule = turbo_schedule(
            shift=3.0 if self.bundle.dit.is_turbo else 1.0,
            steps=steps,
        )
        latent = dit.run_schedule(schedule, progress=progress)
        decoder = self.vae_decoder(latent.shape[1], batch)
        decoder.input.assign(latent)
        decoder.execute()
        if decoder.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(decoder.device))
            decoder.capture()
        audio = decoder.execute()
        if audio.dtype == wp.float32:
            return audio
        output = wp.empty(audio.shape, dtype=wp.float32, device=audio.device)
        wp.launch(
            _cast_kernel_for_dtypes(audio.dtype, wp.float32),
            dim=audio.size,
            inputs=[audio.flatten(), output.flatten()],
            device=audio.device,
        )
        return output
