# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Native Qwen3-TTS hierarchical audio-token inference.

The talker and its small per-frame code predictor share the dense Qwen causal
executor. This module owns only TTS prompting, checkpoint names, and schedule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import warp as wp

from ..chat import sample_token
from ..formats.safetensors import SafeTensorArchive
from ..kernels import (
    _add_arrays_kernel,
    _gather_rows_kernel,
)
from ..operators import BiasedLinearPlan, Operation, plan_linear
from ..tokenizers import Qwen3Tokenizer
from ..weights import load_cast_weights
from .causal import (
    ExternalEmbeddingQwen3CausalLM,
    _ExternalEmbeddingQwen3CausalPlan,
)
from .encoder import qwen3_encoder_weight_names
from .tts_codec import Qwen3TTSCodecDecoder
from .tts_speaker import (
    Qwen3TTSSpeakerConfig,
    Qwen3TTSSpeakerEncoderPlan,
    qwen3_tts_log_mel_from_wav,
    qwen3_tts_speaker_weight_names,
)


SUPPORTED_TTS_LANGUAGES = (
    "chinese",
    "english",
    "french",
    "german",
    "italian",
    "japanese",
    "korean",
    "portuguese",
    "russian",
    "spanish",
)


def load_qwen3_tts_config(path: str | Path) -> dict:
    """Load and validate the compact Qwen3-TTS Base shape contract."""
    path = Path(path)
    document = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if document.get("model_type") != "qwen3_tts":
        raise ValueError("Qwen3-TTS requires model_type 'qwen3_tts'")
    if document.get("tts_model_type") != "base":
        raise ValueError("Qwen3-TTS runner currently requires the Base checkpoint")
    talker = document.get("talker_config", {})
    predictor = talker.get("code_predictor_config", {})
    required = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
    )
    for label, config in (("talker", talker), ("code predictor", predictor)):
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(f"Qwen3-TTS {label} config is missing {missing}")
        query_heads = int(config["num_attention_heads"])
        kv_heads = int(config["num_key_value_heads"])
        if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
            raise ValueError(f"Qwen3-TTS {label} has invalid grouped-query heads")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError(f"Qwen3-TTS {label} requires SiLU")
    groups = int(talker.get("num_code_groups", 0))
    if groups < 2 or groups != int(predictor.get("num_code_groups", -1)):
        raise ValueError("Qwen3-TTS code-group configuration is inconsistent")
    if int(talker.get("text_hidden_size", 0)) != int(talker["hidden_size"]):
        raise ValueError("Qwen3-TTS text and talker widths must match")
    return document


def _qwen_config(config: dict) -> dict:
    """Present one TTS decoder through the standard dense Qwen contract."""
    result = dict(config)
    result.update(
        model_type="qwen3",
        qk_norm=True,
        attention_bias=False,
        tie_word_embeddings=False,
        rope_scaling=None,
        layer_types=["full_attention"] * int(config["num_hidden_layers"]),
    )
    return result


def _backbone_mapping(config: dict, prefix: str, head: str) -> dict[str, str]:
    mapping = {
        name: prefix + name.removeprefix("model")
        for name in qwen3_encoder_weight_names(_qwen_config(config))
        if name != "model.embed_tokens.weight"
    }
    mapping["lm_head.weight"] = head
    return mapping


class _TTSQwenRunner(ExternalEmbeddingQwen3CausalLM):
    """Qwen causal state initialized from names nested in a TTS checkpoint."""

    plan_type = _ExternalEmbeddingQwen3CausalPlan

    def __init__(
        self,
        path: Path,
        config: dict,
        mapping: dict[str, str],
        *,
        cache_capacity: int,
        prefill_chunk_size: int,
        device,
        dtype,
        use_cublas: bool,
    ):
        super().__init__(
            path,
            _qwen_config(config),
            SafeTensorArchive(path),
            mapping,
            cache_capacity=cache_capacity,
            prefill_chunk_size=prefill_chunk_size,
            device=device,
            dtype=dtype,
            use_cublas=use_cublas,
        )


class _Projection:
    """Cached biased projection, optionally with fused SiLU."""

    def __init__(self, weight, bias, *, activation=None, cublas=None):
        self.weight = weight
        self.bias = bias
        self.activation = activation
        self.cublas = cublas
        self._plans = {}

    def __call__(self, value: wp.array) -> wp.array:
        rows = value.shape[0]
        state = self._plans.get(rows)
        if state is None:
            source = wp.empty_like(value)
            plan = BiasedLinearPlan(
                source,
                self.weight,
                self.bias,
                activation=self.activation,
                cublas=self.cublas,
            )
            state = self._plans[rows] = (source, plan)
        source, plan = state
        wp.copy(source, value)
        return plan.execute()


class _TextProjection:
    """Cached embedding gather plus the checkpoint's two-layer resize MLP."""

    def __init__(self, weights, *, device, dtype, cublas):
        self.weights = weights
        self.device = device
        self.dtype = dtype
        self.first = _Projection(
            weights["fc1"],
            weights["fc1_bias"],
            activation="silu",
            cublas=cublas,
        )
        self.second = _Projection(weights["fc2"], weights["fc2_bias"], cublas=cublas)
        self._inputs = {}

    def __call__(self, ids: Sequence[int]) -> wp.array:
        values = np.asarray(ids, dtype=np.int64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Qwen3-TTS text IDs must be a nonempty vector")
        state = self._inputs.get(values.size)
        if state is None:
            indices = wp.empty((1, values.size), dtype=wp.int64, device=self.device)
            gathered = wp.empty(
                (1, values.size, self.weights["embedding"].shape[1]),
                dtype=self.dtype,
                device=self.device,
            )
            state = self._inputs[values.size] = (indices, gathered)
        indices, gathered = state
        indices.assign(values[None, :])
        wp.launch(
            _gather_rows_kernel,
            dim=gathered.shape,
            inputs=[self.weights["embedding"], indices, gathered],
            device=self.device,
        )
        return self.second(self.first(gathered.reshape((values.size, -1))))


class _CodePredictorPlan(_ExternalEmbeddingQwen3CausalPlan):
    """One predictor plan with lightweight selectable per-codebook heads."""

    def __init__(self, runner, rows: int):
        super().__init__(runner, rows)
        self.head_operations = [self.lm_head]
        self.head_logits = [self.logits]
        for index in range(1, runner.code_groups - 1):
            weight_name = f"lm_head.{index}.weight"
            output_name = f"logits.{index}"
            self.tensors[weight_name] = runner.weights[weight_name]
            self.shapes[weight_name] = runner.weights[weight_name].shape
            operation = Operation("Linear", ["final.last", weight_name], [output_name])
            plan_linear(
                operation,
                self.tensors,
                self.shapes,
                self.device,
                cublas=runner.cublas,
            )
            self.head_operations.append(operation)
            self.head_logits.append(
                self.tensors[output_name].reshape(
                    (1, 1, int(runner.config["vocab_size"]))
                )
            )
        self.select_head(0)

    def select_head(self, index: int) -> None:
        self.lm_head = self.head_operations[index]
        self.logits = self.head_logits[index]


class _EmbeddingLookup:
    """Shape-cached device embedding gather."""

    def __init__(self, table: wp.array):
        self.table = table
        self._states = {}

    def __call__(self, token_ids: Sequence[int]) -> wp.array:
        values = np.asarray(token_ids, dtype=np.int64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("embedding token IDs must be a nonempty vector")
        state = self._states.get(values.size)
        if state is None:
            indices = wp.empty(
                (1, values.size), dtype=wp.int64, device=self.table.device
            )
            output = wp.empty(
                (1, values.size, self.table.shape[1]),
                dtype=self.table.dtype,
                device=self.table.device,
            )
            state = self._states[values.size] = (indices, output)
        indices, output = state
        indices.assign(values[None, :])
        wp.launch(
            _gather_rows_kernel,
            dim=output.shape,
            inputs=[self.table, indices, output],
            device=self.table.device,
        )
        return output.reshape((values.size, self.table.shape[1]))


def _concat_rows(values: Sequence[wp.array]) -> wp.array:
    values = tuple(values)
    if not values:
        raise ValueError("cannot concatenate an empty embedding sequence")
    first = values[0]
    width = first.shape[1]
    if any(
        value.ndim != 2
        or value.shape[1] != width
        or value.dtype != first.dtype
        or value.device != first.device
        for value in values
    ):
        raise TypeError("concatenated embeddings must have matching device rows")
    output = wp.empty(
        (sum(value.shape[0] for value in values), width),
        dtype=first.dtype,
        device=first.device,
    )
    offset = 0
    flat = output.flatten()
    for value in values:
        count = value.size
        wp.copy(flat, value.flatten(), dest_offset=offset, count=count)
        offset += count
    return output


def _repeat_row(value: wp.array, rows: int) -> wp.array:
    if value.ndim != 2 or value.shape[0] != 1 or rows <= 0:
        raise ValueError("repeat_row expects one embedding row and a positive count")
    output = wp.empty((rows, value.shape[1]), dtype=value.dtype, device=value.device)
    for row in range(rows):
        wp.copy(
            output.flatten(),
            value.flatten(),
            dest_offset=row * value.shape[1],
            count=value.shape[1],
        )
    return output


def _add_embeddings(left: wp.array, right: wp.array) -> wp.array:
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        raise TypeError("added embeddings must match")
    output = wp.empty_like(left)
    wp.launch(
        _add_arrays_kernel,
        dim=left.size,
        inputs=[left.flatten(), right.flatten(), output.flatten()],
        device=left.device,
    )
    return output


def _sample_tts_logits(
    logits,
    *,
    rng: np.random.Generator,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.0,
    previous_tokens: Sequence[int] = (),
    token_stop: int | None = None,
    eos_token: int | None = None,
    allow_eos: bool = True,
) -> int:
    """Apply the official Qwen sampling policy to one compact vocabulary."""
    values = logits.numpy() if hasattr(logits, "numpy") else np.asarray(logits)
    values = (
        np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])[-1].copy()
    )
    if token_stop is not None:
        if not 0 < token_stop <= values.size:
            raise ValueError("TTS sampling token interval is invalid")
        if eos_token is None or not token_stop <= eos_token < values.size:
            values[token_stop:] = -np.inf
        else:
            eos_value = values[eos_token]
            values[token_stop:] = -np.inf
            values[eos_token] = eos_value
    if eos_token is not None and not allow_eos:
        values[eos_token] = -np.inf
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be positive")
    if repetition_penalty != 1.0 and previous_tokens:
        seen = np.unique(np.asarray(previous_tokens, dtype=np.int64))
        seen = seen[(seen >= 0) & (seen < values.size)]
        selected = values[seen]
        values[seen] = np.where(
            selected < 0.0,
            selected * repetition_penalty,
            selected / repetition_penalty,
        )
    return sample_token(
        values,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        rng=rng,
    )


class _CodePredictor(_TTSQwenRunner):
    """Generate codebooks 1..15 conditioned on the talker's first code."""

    plan_type = _CodePredictorPlan

    def __init__(
        self,
        path: Path,
        talker_config: dict,
        *,
        device,
        dtype,
        use_cublas: bool,
    ):
        config = talker_config["code_predictor_config"]
        groups = int(config["num_code_groups"])
        mapping = _backbone_mapping(
            config,
            "talker.code_predictor.model",
            "talker.code_predictor.lm_head.0.weight",
        )
        for index in range(1, groups - 1):
            mapping[f"lm_head.{index}.weight"] = (
                f"talker.code_predictor.lm_head.{index}.weight"
            )
        for index in range(groups - 1):
            mapping[f"codec_embedding.{index}.weight"] = (
                f"talker.code_predictor.model.codec_embedding.{index}.weight"
            )
        mapping.update(
            {
                "small_projection.weight": (
                    "talker.code_predictor.small_to_mtp_projection.weight"
                ),
                "small_projection.bias": (
                    "talker.code_predictor.small_to_mtp_projection.bias"
                ),
            }
        )
        self.code_groups = groups
        self._active_head = 0
        super().__init__(
            path,
            config,
            mapping,
            cache_capacity=groups + 1,
            prefill_chunk_size=2,
            device=device,
            dtype=dtype,
            use_cublas=use_cublas,
        )
        self.input_projection = _Projection(
            self.weights["small_projection.weight"],
            self.weights["small_projection.bias"],
            cublas=self.cublas,
        )
        self.embedding_lookups = tuple(
            _EmbeddingLookup(self.weights[f"codec_embedding.{index}.weight"])
            for index in range(groups - 1)
        )

    def _run(self, plan, graph_key=None):
        plan.select_head(self._active_head)
        return super()._run(plan, (graph_key, self._active_head))

    def generate_frame(
        self,
        talker_hidden: wp.array,
        first_embedding: wp.array,
        *,
        rng: np.random.Generator,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> list[int]:
        if talker_hidden.shape[0] != 1 or first_embedding.shape[0] != 1:
            raise ValueError("Qwen3-TTS predictor expects one frame at a time")
        self._active_head = 0
        inputs = self.input_projection(_concat_rows((talker_hidden, first_embedding)))
        logits = self.prefill_embeddings(inputs)
        codes = []
        for head in range(self.code_groups - 1):
            token = _sample_tts_logits(
                logits,
                rng=rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                token_stop=int(self.config["vocab_size"]),
            )
            codes.append(token)
            if head + 1 < self.code_groups - 1:
                self._active_head = head + 1
                embedding = self.input_projection(
                    self.embedding_lookups[head]((token,))
                )
                logits = self.decode_embedding(embedding)
        return codes


class Qwen3TTSTokenGenerator:
    """Dependency-free Qwen3-TTS text-to-16-codebook generation."""

    frames_per_second = 12.5

    def __init__(
        self,
        path: str | Path,
        *,
        dtype=wp.bfloat16,
        device=None,
        cache_capacity: int = 4096,
        prefill_chunk_size: int = 16,
        use_cublas: bool = True,
    ):
        path = Path(path)
        self.path = path
        self.config = load_qwen3_tts_config(path)
        talker_config = self.config["talker_config"]
        groups = int(talker_config["num_code_groups"])
        mapping = _backbone_mapping(
            talker_config, "talker.model", "talker.codec_head.weight"
        )
        mapping.update(
            {
                "codec_embedding.weight": "talker.model.codec_embedding.weight",
                "text_embedding.weight": "talker.model.text_embedding.weight",
                "text_projection.fc1.weight": (
                    "talker.text_projection.linear_fc1.weight"
                ),
                "text_projection.fc1.bias": ("talker.text_projection.linear_fc1.bias"),
                "text_projection.fc2.weight": (
                    "talker.text_projection.linear_fc2.weight"
                ),
                "text_projection.fc2.bias": ("talker.text_projection.linear_fc2.bias"),
            }
        )
        self.talker = _TTSQwenRunner(
            path,
            talker_config,
            mapping,
            cache_capacity=cache_capacity,
            prefill_chunk_size=prefill_chunk_size,
            device=device,
            dtype=dtype,
            use_cublas=use_cublas,
        )
        self.predictor = _CodePredictor(
            path,
            talker_config,
            device=self.talker.device,
            dtype=dtype,
            use_cublas=use_cublas,
        )
        self.tokenizer = Qwen3Tokenizer(path)
        self.text_projection = _TextProjection(
            {
                "embedding": self.talker.weights["text_embedding.weight"],
                "fc1": self.talker.weights["text_projection.fc1.weight"],
                "fc1_bias": self.talker.weights["text_projection.fc1.bias"],
                "fc2": self.talker.weights["text_projection.fc2.weight"],
                "fc2_bias": self.talker.weights["text_projection.fc2.bias"],
            },
            device=self.talker.device,
            dtype=dtype,
            cublas=self.talker.cublas,
        )
        self.codec_lookup = _EmbeddingLookup(
            self.talker.weights["codec_embedding.weight"]
        )
        self.code_groups = groups

    @property
    def device(self):
        return self.talker.device

    @property
    def dtype(self):
        return self.talker.dtype

    def _prompt_embeddings(
        self,
        text: str,
        language: str,
        speaker_embedding: wp.array | None,
    ) -> tuple[wp.array, wp.array, wp.array]:
        language = language.lower()
        if language not in SUPPORTED_TTS_LANGUAGES:
            raise ValueError(
                f"unsupported Qwen3-TTS language '{language}'; "
                f"choose one of {', '.join(SUPPORTED_TTS_LANGUAGES)}"
            )
        if not text.strip():
            raise ValueError("Qwen3-TTS text must not be empty")
        formatted = (
            f"<|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n"
        )
        text_ids = self.tokenizer.encode(formatted)
        if len(text_ids) < 9:
            raise ValueError("Qwen3-TTS text prompt tokenization is unexpectedly short")
        cfg = self.config
        talker = cfg["talker_config"]
        special = self.text_projection(
            (
                int(cfg["tts_bos_token_id"]),
                int(cfg["tts_pad_token_id"]),
            )
        )
        tts_bos, tts_pad = special[0:1], special[1:2]
        codec_ids = [
            int(talker["codec_think_id"]),
            int(talker["codec_think_bos_id"]),
            int(talker["codec_language_id"][language]),
            int(talker["codec_think_eos_id"]),
        ]
        codec_prefix_parts = [self.codec_lookup(codec_ids)]
        if speaker_embedding is not None:
            if (
                speaker_embedding.shape != (1, self.talker.hidden_size)
                or speaker_embedding.dtype != self.dtype
                or speaker_embedding.device != self.device
            ):
                raise TypeError(
                    "speaker embedding must match the TTS talker width/device"
                )
            codec_prefix_parts.append(speaker_embedding)
        codec_prefix_parts.append(
            self.codec_lookup(
                (int(talker["codec_pad_id"]), int(talker["codec_bos_id"]))
            )
        )
        codec_prefix = _concat_rows(codec_prefix_parts)
        text_side = _concat_rows(
            (
                _repeat_row(tts_pad, codec_prefix.shape[0] - 2),
                tts_bos,
            )
        )
        mixed_prefix = _add_embeddings(
            text_side, codec_prefix[:-1].reshape(text_side.shape)
        )
        role = self.text_projection(text_ids[:3])
        first_text = _add_embeddings(
            self.text_projection((text_ids[3],)),
            codec_prefix[-1:].reshape((1, self.talker.hidden_size)),
        )
        prompt = _concat_rows((role, mixed_prefix, first_text))
        trailing_ids = tuple(text_ids[4:-5]) + (int(cfg["tts_eos_token_id"]),)
        trailing = self.text_projection(trailing_ids)
        return prompt, trailing, tts_pad

    def generate_codes(
        self,
        text: str,
        *,
        language: str = "english",
        speaker_embedding: wp.array | None = None,
        max_seconds: float = 30.0,
        seed: int | None = None,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        subtalker_temperature: float = 0.9,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        progress=None,
    ) -> np.ndarray:
        """Generate acoustic codes; waveform decoding is handled separately."""
        if not np.isfinite(max_seconds) or max_seconds <= 0.0:
            raise ValueError("max_seconds must be finite and positive")
        max_frames = min(
            int(np.ceil(max_seconds * self.frames_per_second)),
            self.talker.cache_capacity - 1,
        )
        if max_frames < 1:
            raise ValueError("Qwen3-TTS cache has no room for an audio frame")
        prompt, trailing, tts_pad = self._prompt_embeddings(
            text, language, speaker_embedding
        )
        if prompt.shape[0] + max_frames >= self.talker.cache_capacity:
            max_frames = self.talker.cache_capacity - prompt.shape[0]
        if max_frames < 1:
            raise ValueError("Qwen3-TTS prompt leaves no KV-cache room")
        rng = np.random.default_rng(seed)
        logits = self.talker.prefill_embeddings(prompt)
        eos = int(self.config["talker_config"]["codec_eos_token_id"])
        first_history = []
        frames = []
        for frame_index in range(max_frames):
            first = _sample_tts_logits(
                logits,
                rng=rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                previous_tokens=first_history,
                token_stop=2048,
                eos_token=eos,
                allow_eos=frame_index >= 2,
            )
            if first == eos:
                break
            first_history.append(first)
            first_embedding = self.codec_lookup((first,))
            rest = self.predictor.generate_frame(
                self.talker.last_hidden,
                first_embedding,
                rng=rng,
                temperature=subtalker_temperature,
                top_k=subtalker_top_k,
                top_p=subtalker_top_p,
            )
            frame = [first, *rest]
            frames.append(frame)
            if progress is not None:
                progress(frame_index + 1, max_frames)
            if frame_index + 1 >= max_frames:
                break
            summed = first_embedding
            for group, token in enumerate(rest):
                value = self.predictor.embedding_lookups[group]((token,))
                summed = _add_embeddings(summed, value)
            text_embedding = (
                trailing[frame_index : frame_index + 1]
                if frame_index < trailing.shape[0]
                else tts_pad
            )
            logits = self.talker.decode_embedding(
                _add_embeddings(summed, text_embedding)
            )
        if not frames:
            return np.empty((0, self.code_groups), dtype=np.int32)
        if progress is not None:
            progress(len(frames), len(frames))
        return np.asarray(frames, dtype=np.int32)


class Qwen3TTSPipeline:
    """Resident native Qwen3-TTS Base pipeline for text-to-waveform synthesis.

    Reference audio is optional at the low-level API, but strongly recommended
    for this Base checkpoint. When supplied, its x-vector conditions the voice
    without requiring a transcript or the codec encoder.
    """

    sample_rate = Qwen3TTSCodecDecoder.sample_rate

    def __init__(
        self,
        path: str | Path,
        *,
        dtype=wp.bfloat16,
        device=None,
        cache_capacity: int = 4096,
        prefill_chunk_size: int = 16,
        use_cublas: bool = True,
    ):
        self.path = Path(path)
        self.generator = Qwen3TTSTokenGenerator(
            self.path,
            dtype=dtype,
            device=device,
            cache_capacity=cache_capacity,
            prefill_chunk_size=prefill_chunk_size,
            use_cublas=use_cublas,
        )
        self.decoder = Qwen3TTSCodecDecoder(
            self.path,
            dtype=dtype,
            device=self.generator.device,
            use_cublas=use_cublas,
        )
        self.dtype = dtype
        self.device = self.generator.device
        self._speaker_config = Qwen3TTSSpeakerConfig.from_path(self.path)
        self._speaker_weights = None
        self._speaker_plans = {}

    def speaker_embedding(self, reference_audio: str | Path) -> wp.array:
        """Extract a speaker x-vector from an exact 24-kHz PCM16 WAV."""
        if self.dtype != wp.bfloat16:
            raise TypeError("native Qwen3-TTS speaker encoding currently requires BF16")
        features = qwen3_tts_log_mel_from_wav(reference_audio)
        if self._speaker_weights is None:
            names = qwen3_tts_speaker_weight_names(self._speaker_config)
            self._speaker_weights = load_cast_weights(
                SafeTensorArchive(self.path),
                names,
                self.device,
                self.dtype,
            )
        plan = self._speaker_plans.get(features.shape[0])
        if plan is None:
            source = wp.empty(
                (1, features.shape[0], features.shape[1]),
                dtype=self.dtype,
                device=self.device,
            )
            plan = Qwen3TTSSpeakerEncoderPlan(
                source, self._speaker_weights, self._speaker_config
            )
            self._speaker_plans[features.shape[0]] = plan
        plan.input.assign(features[None, :, :])
        return plan.execute()

    def generate(
        self,
        text: str,
        *,
        language: str = "english",
        reference_audio: str | Path | None = None,
        max_seconds: float = 30.0,
        seed: int | None = None,
        progress=None,
        **sampling,
    ) -> tuple[wp.array, np.ndarray]:
        """Generate mono 24-kHz audio and its exact 16-codebook sequence."""
        speaker = (
            None if reference_audio is None else self.speaker_embedding(reference_audio)
        )
        codes = self.generator.generate_codes(
            text,
            language=language,
            speaker_embedding=speaker,
            max_seconds=max_seconds,
            seed=seed,
            progress=progress,
            **sampling,
        )
        if codes.shape[0] == 0:
            raise RuntimeError("Qwen3-TTS stopped before producing any audio")
        return self.decoder.decode(codes), codes
