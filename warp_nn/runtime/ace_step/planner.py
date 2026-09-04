# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step 1.5 planner grammar and 5 Hz semantic-code decoding."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import warp as wp

from .._cublas import try_create_cublas
from ..chat import sample_runner_token, sample_token
from ..formats.safetensors import SafeTensorArchive
from ..operators import Operation, execute_operations, plan_linear
from ..qwen.causal import Qwen3CausalLM
from ..qwen.encoder import _projection_bias_kernel
from ..weights import load_cast_weights
from .conditioning import _EncoderStackPlan


DEFAULT_PLANNER_INSTRUCTION = (
    "Generate audio semantic tokens based on the given conditions:"
)
AUDIO_CODE_TOKEN_BASE = 151_669
AUDIO_CODE_COUNT = 64_000
AUDIO_CODE_TOKEN_STOP = AUDIO_CODE_TOKEN_BASE + AUDIO_CODE_COUNT
FSQ_LEVELS = (8, 8, 8, 5, 5, 5)


def audio_code_token_id(code: int) -> int:
    """Map one ACE FSQ code to its contiguous Qwen token ID."""
    if not 0 <= code < AUDIO_CODE_COUNT:
        raise ValueError("ACE audio code must be between 0 and 63999")
    return AUDIO_CODE_TOKEN_BASE + int(code)


def audio_code_from_token_id(token_id: int) -> int:
    """Map one constrained Qwen token ID back to its ACE FSQ code."""
    if not AUDIO_CODE_TOKEN_BASE <= token_id < AUDIO_CODE_TOKEN_STOP:
        raise ValueError("token is not an ACE audio-code token")
    return int(token_id) - AUDIO_CODE_TOKEN_BASE


def fsq_indices_to_codes(
    indices: Sequence[int] | np.ndarray,
    levels: Sequence[int] = FSQ_LEVELS,
) -> np.ndarray:
    """Invert vector-quantize-pytorch FSQ mixed-radix indices."""
    values = np.asarray(indices, dtype=np.int64)
    level_values = np.asarray(levels, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("FSQ indices must be a nonempty one-dimensional sequence")
    if level_values.ndim != 1 or np.any(level_values < 2):
        raise ValueError("FSQ levels must be one-dimensional and at least two")
    codebook_size = int(np.prod(level_values, dtype=np.int64))
    if np.any(values < 0) or np.any(values >= codebook_size):
        raise ValueError(f"FSQ index must be between 0 and {codebook_size - 1}")
    basis = np.cumprod(np.concatenate(([1], level_values[:-1])))
    digits = (values[:, None] // basis[None, :]) % level_values[None, :]
    half_width = level_values // 2
    return ((digits - half_width) / half_width).astype(np.float32)


def format_planner_prompt(caption: str, lyrics: str, *, cot: str | None = None) -> str:
    """Format the checkpoint's exact phase-one or phase-two chat boundary."""
    prefix = (
        "<|im_start|>system\n# Instruction\n"
        f"{DEFAULT_PLANNER_INSTRUCTION}\n\n<|im_end|>\n"
        "<|im_start|>user\n# Caption\n"
        f"{caption.strip()}\n\n# Lyric\n{lyrics.strip()}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if cot is None:
        return prefix
    body = cot.strip()
    if body.startswith("<think>"):
        body = body.removeprefix("<think>")
    if body.endswith("</think>"):
        body = body.removesuffix("</think>")
    return prefix + f"<think>\n{body.strip()}\n</think>\n\n"


def format_planner_unconditional_prompt() -> str:
    """Format the checkpoint's training-aligned empty phase-two CFG branch."""
    return (
        "<|im_start|>system\n# Instruction\n"
        f"{DEFAULT_PLANNER_INSTRUCTION}\n\n<|im_end|>\n"
        "<|im_start|>user\nNO USER INPUT<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def parse_planner_metadata(cot: str) -> dict[str, str]:
    """Extract the official small metadata vocabulary from planner reasoning."""
    result = {}
    aliases = {"key": "keyscale", "time_signature": "timesignature"}
    pattern = re.compile(
        r"^\s*(bpm|duration|keyscale|key|timesignature|time_signature|language)\s*:\s*(.*?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    for name, value in pattern.findall(cot):
        if value:
            result[aliases.get(name.lower(), name.lower())] = value
    return result


@dataclass(frozen=True)
class AcePlannerResult:
    cot: str
    metadata: dict[str, str]
    audio_codes: tuple[int, ...]


def _set_planner_metadata(cot: str, name: str, value: str) -> str:
    """Replace or append one constrained phase-one metadata value."""
    pattern = re.compile(rf"^(\s*{re.escape(name)}\s*:)\s*.*$", re.MULTILINE)
    if pattern.search(cot):
        return pattern.sub(rf"\1 {value}", cot, count=1)
    suffix = "" if cot.endswith("\n") else "\n"
    return f"{cot}{suffix}{name}: {value}"


def _duration_text(duration_seconds: float) -> str:
    return f"{duration_seconds:g}"


class AceStepPlanner:
    """Deterministic two-phase ACE planner backed by dense Qwen3 decode."""

    def __init__(self, runner: Qwen3CausalLM):
        if int(runner.config["vocab_size"]) < AUDIO_CODE_TOKEN_STOP:
            raise ValueError("Qwen3 vocabulary does not contain ACE audio-code tokens")
        self.runner = runner
        encoded = runner.tokenizer.encode("</think>")
        if len(encoded) != 1:
            raise ValueError("planner tokenizer must encode </think> as one token")
        self.think_end_token = encoded[0]
        self._unconditional_runner = None

    def generate_cot(
        self,
        caption: str,
        lyrics: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.85,
        top_k: int = 0,
        top_p: float = 0.9,
        rng: np.random.Generator | None = None,
    ) -> str:
        """Generate metadata/reasoning until the mandatory ``</think>`` token."""
        if max_tokens <= 0:
            raise ValueError("planner max_tokens must be positive")
        prompt = format_planner_prompt(caption, lyrics)
        logits = self.runner.prefill(self.runner.tokenizer.encode(prompt))
        generated = []
        rng = rng or np.random.default_rng()
        for _ in range(max_tokens):
            token = sample_runner_token(
                self.runner,
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
            )
            if token == self.think_end_token:
                break
            generated.append(token)
            logits = self.runner.decode(token)
        else:
            raise RuntimeError("ACE planner did not close its reasoning phase")
        return self.runner.tokenizer.decode(generated).strip()

    def generate_codes(
        self,
        caption: str,
        lyrics: str,
        cot: str,
        *,
        duration_seconds: float,
        temperature: float = 0.85,
        top_k: int = 0,
        top_p: float = 0.9,
        cfg_scale: float = 2.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, ...]:
        """Generate exactly five grammar-constrained semantic codes per second."""
        if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
            raise ValueError("planner duration must be a positive finite value")
        if not math.isfinite(cfg_scale) or cfg_scale < 0.0:
            raise ValueError("planner CFG scale must be finite and nonnegative")
        count = max(1, int(duration_seconds * 5.0))
        prompt = format_planner_prompt(caption, lyrics, cot=cot)
        logits = self.runner.prefill(self.runner.tokenizer.encode(prompt))
        unconditional = None
        unconditional_logits = None
        if cfg_scale != 1.0:
            if self._unconditional_runner is None:
                self._unconditional_runner = self.runner.fork_state()
            unconditional = self._unconditional_runner
            unconditional_logits = unconditional.prefill(
                unconditional.tokenizer.encode(format_planner_unconditional_prompt())
            )
        result = []
        rng = rng or np.random.default_rng()
        for index in range(count):
            if unconditional is not None:
                positive = logits.flatten()[
                    AUDIO_CODE_TOKEN_BASE:AUDIO_CODE_TOKEN_STOP
                ].numpy()
                negative = unconditional_logits.flatten()[
                    AUDIO_CODE_TOKEN_BASE:AUDIO_CODE_TOKEN_STOP
                ].numpy()
                values = negative + cfg_scale * (positive - negative)
                selected = sample_token(
                    values,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    rng=rng,
                )
                token = AUDIO_CODE_TOKEN_BASE + selected
            elif temperature <= 0.0 or top_k == 1:
                token = self.runner.sample_greedy_range(
                    logits, AUDIO_CODE_TOKEN_BASE, AUDIO_CODE_TOKEN_STOP
                )
            elif 1 < top_k <= 32:
                values, candidates = self.runner.read_top_k_range(
                    logits,
                    AUDIO_CODE_TOKEN_BASE,
                    AUDIO_CODE_TOKEN_STOP,
                    top_k,
                )
                selected = sample_token(
                    values,
                    temperature=temperature,
                    top_p=top_p,
                    rng=rng,
                )
                token = int(candidates[selected])
            else:
                values = logits.flatten()[
                    AUDIO_CODE_TOKEN_BASE:AUDIO_CODE_TOKEN_STOP
                ].reshape((1, 1, AUDIO_CODE_COUNT))
                selected = sample_token(
                    values,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    rng=rng,
                )
                token = AUDIO_CODE_TOKEN_BASE + selected
            result.append(audio_code_from_token_id(token))
            if index + 1 < count:
                logits = self.runner.decode(token)
                if unconditional is not None:
                    unconditional_logits = unconditional.decode(token)
        return tuple(result)

    def generate(
        self,
        caption: str,
        lyrics: str,
        *,
        duration_seconds: float,
        temperature: float = 0.85,
        top_k: int = 0,
        top_p: float = 0.9,
        cfg_scale: float = 2.0,
        seed: int | None = None,
    ) -> AcePlannerResult:
        rng = np.random.default_rng(seed)
        cot = self.generate_cot(
            caption,
            lyrics,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
        )
        cot = _set_planner_metadata(cot, "duration", _duration_text(duration_seconds))
        codes = self.generate_codes(
            caption,
            lyrics,
            cot,
            duration_seconds=duration_seconds,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=cfg_scale,
            rng=rng,
        )
        return AcePlannerResult(cot, parse_planner_metadata(cot), codes)


def audio_code_decoder_weight_names(layers: int = 2) -> tuple[str, ...]:
    """Return the exact FSQ output-projection and detokenizer manifest."""
    names = [
        "tokenizer.quantizer.project_out.weight",
        "tokenizer.quantizer.project_out.bias",
        "detokenizer.embed_tokens.weight",
        "detokenizer.embed_tokens.bias",
        "detokenizer.special_tokens",
        "detokenizer.norm.weight",
        "detokenizer.proj_out.weight",
        "detokenizer.proj_out.bias",
    ]
    suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    for index in range(layers):
        base = f"detokenizer.layers.{index}."
        names.extend(base + suffix for suffix in suffixes)
    return tuple(names)


@dataclass(frozen=True)
class _DecoderConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    layer_types: tuple[str, ...]
    sliding_window: int | None
    attention_bias: bool = False

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layer_types)

    def attention_window(self, layer_index: int) -> int | None:
        if self.layer_types[layer_index] == "sliding_attention":
            return self.sliding_window
        return None


@lru_cache(maxsize=None)
def _patch_kernel(dtype):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def expand(
        code: wp.array2d(dtype=DTYPE),
        special: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
    ):
        token, patch, channel = wp.tid()
        output[token, patch, channel] = DTYPE(
            code[token, channel] + special[0, patch, channel]
        )

    return expand


class _AudioCodeDecodePlan:
    def __init__(self, owner: AceAudioCodeDecoder, count: int):
        self.owner = owner
        self.device = owner.device
        self.dtype = owner.dtype
        self.scalar_codes = wp.empty((count, 6), dtype=self.dtype, device=self.device)
        self.tensors = dict(owner.weights)
        self.tensors["scalar_codes"] = self.scalar_codes
        self.shapes = {name: tuple(value.shape) for name, value in self.tensors.items()}
        self.fsq_projection = Operation(
            "Linear",
            ["scalar_codes", "tokenizer.quantizer.project_out.weight"],
            ["quantized"],
        )
        plan_linear(
            self.fsq_projection,
            self.tensors,
            self.shapes,
            self.device,
            cublas=owner.cublas,
        )
        self.embedding = Operation(
            "Linear", ["quantized", "detokenizer.embed_tokens.weight"], ["embedded"]
        )
        plan_linear(
            self.embedding, self.tensors, self.shapes, self.device, cublas=owner.cublas
        )
        self.hidden = wp.empty(
            (count, owner.pool_window_size, owner.config.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        valid = wp.ones(self.hidden.shape[:2], dtype=wp.bool, device=self.device)
        self.stack = _EncoderStackPlan(
            self.hidden,
            valid,
            owner.weights,
            owner.config,
            "detokenizer",
            len(owner.config.layer_types),
            owner.cublas,
        )
        self.output_tensors = dict(owner.weights)
        self.output_tensors["hidden"] = self.stack.output.reshape(
            (-1, owner.config.hidden_size)
        )
        self.output_shapes = {
            name: tuple(value.shape) for name, value in self.output_tensors.items()
        }
        self.output_projection = Operation(
            "Linear", ["hidden", "detokenizer.proj_out.weight"], ["output"]
        )
        plan_linear(
            self.output_projection,
            self.output_tensors,
            self.output_shapes,
            self.device,
            cublas=owner.cublas,
        )
        self.output = self.output_tensors["output"].reshape(
            (1, count * owner.pool_window_size, owner.output_channels)
        )
        self.graph = None
        self.capture_ready = False

    def execute(self):
        add_bias = _projection_bias_kernel(self.dtype)
        execute_operations(
            (self.fsq_projection,), self.tensors, self.shapes, self.device
        )
        wp.launch(
            add_bias,
            dim=self.tensors["quantized"].shape,
            inputs=[
                self.tensors["quantized"],
                self.owner.weights["tokenizer.quantizer.project_out.bias"],
            ],
            device=self.device,
        )
        execute_operations((self.embedding,), self.tensors, self.shapes, self.device)
        wp.launch(
            add_bias,
            dim=self.tensors["embedded"].shape,
            inputs=[
                self.tensors["embedded"],
                self.owner.weights["detokenizer.embed_tokens.bias"],
            ],
            device=self.device,
        )
        wp.launch(
            _patch_kernel(self.dtype),
            dim=self.hidden.shape,
            inputs=[
                self.tensors["embedded"],
                self.owner.weights["detokenizer.special_tokens"],
                self.hidden,
            ],
            device=self.device,
        )
        self.stack.execute()
        execute_operations(
            (self.output_projection,),
            self.output_tensors,
            self.output_shapes,
            self.device,
        )
        wp.launch(
            add_bias,
            dim=self.output_tensors["output"].shape,
            inputs=[
                self.output_tensors["output"],
                self.owner.weights["detokenizer.proj_out.bias"],
            ],
            device=self.device,
        )
        return self.output

    def run(self):
        if not self.device.is_cuda:
            return self.execute()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        elif self.capture_ready:
            wp.capture_begin(device=self.device)
            try:
                self.execute()
                self.graph = wp.capture_end(device=self.device)
            except Exception:
                wp.capture_end(device=self.device)
                raise
            wp.capture_launch(self.graph)
        else:
            self.execute()
            self.capture_ready = True
        return self.output


class AceAudioCodeDecoder:
    """Decode 5 Hz FSQ indices into 25 Hz, 64-channel semantic latents."""

    def __init__(
        self,
        path: str | Path,
        *,
        dtype=wp.bfloat16,
        device=None,
        use_cublas: bool = True,
    ):
        if dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("ACE audio-code decoder requires FP16 or BF16")
        path = Path(path)
        source = json.loads((path / "config.json").read_text(encoding="utf-8"))
        levels = tuple(int(value) for value in source.get("fsq_input_levels", ()))
        if levels != FSQ_LEVELS or int(source.get("fsq_input_num_quantizers", 0)) != 1:
            raise ValueError("unsupported ACE FSQ geometry")
        layers = int(source["num_attention_pooler_hidden_layers"])
        self.config = _DecoderConfig(
            hidden_size=int(source["encoder_hidden_size"]),
            intermediate_size=int(source["encoder_intermediate_size"]),
            num_attention_heads=int(source["encoder_num_attention_heads"]),
            num_key_value_heads=int(source["encoder_num_key_value_heads"]),
            head_dim=int(source["head_dim"]),
            rms_norm_eps=float(source.get("rms_norm_eps", 1.0e-6)),
            rope_theta=float(source.get("rope_theta", 1_000_000.0)),
            layer_types=tuple(source["layer_types"][:layers]),
            sliding_window=int(source.get("sliding_window", 128)),
        )
        self.pool_window_size = int(source["pool_window_size"])
        self.output_channels = int(source["audio_acoustic_hidden_dim"])
        self.device = wp.get_device(device)
        self.dtype = dtype
        archive = SafeTensorArchive(path)
        names = audio_code_decoder_weight_names(layers)
        missing = set(names) - set(archive.names)
        if missing:
            raise ValueError(f"ACE checkpoint is missing {sorted(missing)[:5]}")
        self.weights = load_cast_weights(archive, names, self.device, dtype)
        self.cublas = (
            try_create_cublas() if use_cublas and self.device.is_cuda else None
        )
        self._plans = {}

    def decode(self, indices: Sequence[int] | np.ndarray) -> wp.array:
        values = fsq_indices_to_codes(indices)
        plan = self._plans.get(len(values))
        if plan is None:
            plan = self._plans[len(values)] = _AudioCodeDecodePlan(self, len(values))
        plan.scalar_codes.assign(values)
        return plan.run()
