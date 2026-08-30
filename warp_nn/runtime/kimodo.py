# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free Kimodo motion-diffusion inference support.

This module owns only Kimodo-specific orchestration and motion representation.
The encoder blocks delegate their dense and attention work to
``warp_nn.runtime.encoder``.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import warp as wp

from .encoder import EncoderStackPlan, _encoder_kernels
from .operators import Operation, execute_operations, plan_linear
from .safetensors import SafeTensorArchive
from .weights import load_cast_weights


@dataclass(frozen=True)
class KimodoConfig:
    """Resolved architecture settings for a Kimodo two-stage denoiser."""

    motion_dim: int
    joints: int
    fps: float
    latent_dim: int
    feedforward_dim: int
    layers: int
    heads: int
    text_dim: int = 4096
    text_tokens: int = 50
    diffusion_steps: int = 1000
    first_heading: bool = True
    concatenate_mask: bool = True

    def __post_init__(self):
        positive = {
            "motion_dim": self.motion_dim,
            "joints": self.joints,
            "fps": self.fps,
            "latent_dim": self.latent_dim,
            "feedforward_dim": self.feedforward_dim,
            "layers": self.layers,
            "heads": self.heads,
            "text_dim": self.text_dim,
            "text_tokens": self.text_tokens,
            "diffusion_steps": self.diffusion_steps,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                f"Kimodo dimensions must be positive: {', '.join(invalid)}"
            )
        if self.latent_dim % self.heads:
            raise ValueError("Kimodo latent_dim must be divisible by heads")
        expected = 12 * self.joints + 9
        if self.motion_dim != expected:
            raise ValueError(
                f"Kimodo motion_dim {self.motion_dim} does not match {self.joints} joints ({expected})"
            )

    @property
    def global_root_dim(self):
        return 5

    @property
    def local_root_dim(self):
        return 4

    @property
    def body_dim(self):
        return self.motion_dim - self.global_root_dim

    @classmethod
    def soma_v1(cls):
        """Architecture of nvidia/Kimodo-SOMA-RP-v1.1."""
        return cls(369, 30, 30.0, 1024, 2048, 16, 8)

    @classmethod
    def from_json(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("Kimodo JSON config must contain an object")
        return cls(**data)


@dataclass(frozen=True)
class KimodoStats:
    """Motion normalization arrays required by the two-stage representation."""

    global_mean: np.ndarray
    global_std: np.ndarray
    local_mean: np.ndarray
    local_std: np.ndarray
    body_mean: np.ndarray
    body_std: np.ndarray
    epsilon: float = 1.0e-5

    def __post_init__(self):
        for group in ("global", "local", "body"):
            mean = np.asarray(getattr(self, f"{group}_mean"), dtype=np.float32)
            std = np.asarray(getattr(self, f"{group}_std"), dtype=np.float32)
            if mean.ndim != 1 or std.shape != mean.shape or not mean.size:
                raise ValueError(f"Kimodo {group} mean/std must be matching vectors")
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
                raise ValueError(f"Kimodo {group} statistics must be finite")
            if np.any(std < 0):
                raise ValueError("Kimodo standard deviations must be nonnegative")
            object.__setattr__(self, f"{group}_mean", mean)
            object.__setattr__(self, f"{group}_std", std)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Kimodo statistics epsilon must be finite and positive")

    @property
    def mean(self):
        return np.concatenate((self.global_mean, self.body_mean))

    @property
    def std(self):
        return np.concatenate((self.global_std, self.body_std))

    @classmethod
    def load(cls, folder: str | Path):
        folder = Path(folder)
        if (folder / "motion").is_dir():
            folder /= "motion"
        return cls(
            *(
                np.load(folder / group / name)
                for group in ("global_root", "local_root", "body")
                for name in ("mean.npy", "std.npy")
            )
        )


def cosine_ddim_schedule(base_steps: int, denoising_steps: int):
    """Return Kimodo's exact subsampled cumulative-alpha DDIM schedule."""
    if base_steps <= 0 or denoising_steps <= 0:
        raise ValueError("diffusion step counts must be positive")

    def alpha_bar(value):
        return math.cos((value + 0.008) / 1.008 * math.pi / 2.0) ** 2

    beta = np.array(
        [
            min(
                1.0
                - alpha_bar((index + 1) / base_steps) / alpha_bar(index / base_steps),
                0.999,
            )
            for index in range(base_steps)
        ],
        dtype=np.float64,
    )
    cumulative = np.cumprod(1.0 - beta)
    stride = (base_steps - 1) / max(1, denoising_steps - 1)
    selected = np.rint(np.arange(denoising_steps) * stride).astype(np.int64)
    selected = np.clip(selected, 0, base_steps - 1)
    cumulative = cumulative[selected]
    previous = np.concatenate(([1.0], cumulative[:-1]))
    return (
        selected.astype(np.int32),
        cumulative.astype(np.float32),
        previous.astype(np.float32),
    )


@lru_cache(maxsize=None)
def _motion_kernels(dtype):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def normalize(
        source: wp.array3d(dtype=DTYPE),
        mean: wp.array1d[wp.float32],
        std: wp.array1d[wp.float32],
        output: wp.array3d(dtype=DTYPE),
        epsilon: wp.float32,
    ):
        batch, frame, feature = wp.tid()
        scale = wp.sqrt(std[feature] * std[feature] + epsilon)
        output[batch, frame, feature] = DTYPE(
            (wp.float32(source[batch, frame, feature]) - mean[feature]) / scale
        )

    @wp.kernel(enable_backward=False, module="unique")
    def apply_condition(
        motion: wp.array3d(dtype=DTYPE),
        observed: wp.array3d(dtype=DTYPE),
        mask: wp.array3d(dtype=wp.bool),
        output: wp.array3d(dtype=DTYPE),
    ):
        batch, frame, feature = wp.tid()
        output[batch, frame, feature] = (
            observed[batch, frame, feature]
            if mask[batch, frame, feature]
            else motion[batch, frame, feature]
        )

    @wp.kernel(enable_backward=False, module="unique")
    def global_root_to_local(
        root: wp.array3d(dtype=DTYPE),
        lengths: wp.array1d[wp.int32],
        global_mean: wp.array1d[wp.float32],
        global_std: wp.array1d[wp.float32],
        local_mean: wp.array1d[wp.float32],
        local_std: wp.array1d[wp.float32],
        output: wp.array3d(dtype=DTYPE),
        fps: wp.float32,
        epsilon: wp.float32,
    ):
        batch, frame = wp.tid()
        length = wp.clamp(lengths[batch], 2, root.shape[1])
        current = wp.min(frame, length - 2)
        following = current + 1
        x0 = (
            wp.float32(root[batch, current, 0])
            * wp.sqrt(global_std[0] * global_std[0] + epsilon)
            + global_mean[0]
        )
        z0 = (
            wp.float32(root[batch, current, 2])
            * wp.sqrt(global_std[2] * global_std[2] + epsilon)
            + global_mean[2]
        )
        x1 = (
            wp.float32(root[batch, following, 0])
            * wp.sqrt(global_std[0] * global_std[0] + epsilon)
            + global_mean[0]
        )
        z1 = (
            wp.float32(root[batch, following, 2])
            * wp.sqrt(global_std[2] * global_std[2] + epsilon)
            + global_mean[2]
        )
        c0 = (
            wp.float32(root[batch, current, 3])
            * wp.sqrt(global_std[3] * global_std[3] + epsilon)
            + global_mean[3]
        )
        s0 = (
            wp.float32(root[batch, current, 4])
            * wp.sqrt(global_std[4] * global_std[4] + epsilon)
            + global_mean[4]
        )
        c1 = (
            wp.float32(root[batch, following, 3])
            * wp.sqrt(global_std[3] * global_std[3] + epsilon)
            + global_mean[3]
        )
        s1 = (
            wp.float32(root[batch, following, 4])
            * wp.sqrt(global_std[4] * global_std[4] + epsilon)
            + global_mean[4]
        )
        angle0 = wp.atan2(s0, c0)
        angle1 = wp.atan2(s1, c1)
        delta = wp.atan2(wp.sin(angle1 - angle0), wp.cos(angle1 - angle0)) * fps
        height = (
            wp.float32(root[batch, frame, 1])
            * wp.sqrt(global_std[1] * global_std[1] + epsilon)
            + global_mean[1]
        )
        values = wp.vec4f(delta, (x1 - x0) * fps, (z1 - z0) * fps, height)
        for feature in range(4):
            scale = wp.sqrt(local_std[feature] * local_std[feature] + epsilon)
            output[batch, frame, feature] = DTYPE(
                (values[feature] - local_mean[feature]) / scale
            )

    @wp.kernel(enable_backward=False, module="unique")
    def guidance(
        predictions: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        text_weight: wp.float32,
        constraint_weight: wp.float32,
        mode: wp.int32,
    ):
        batch, frame, feature = wp.tid()
        batches = output.shape[0]
        if mode == 0:
            output[batch, frame, feature] = predictions[batch, frame, feature]
        elif mode == 1:
            conditional = wp.float32(predictions[batch, frame, feature])
            unconditional = wp.float32(predictions[batch + batches, frame, feature])
            output[batch, frame, feature] = DTYPE(
                unconditional + text_weight * (conditional - unconditional)
            )
        else:
            text = wp.float32(predictions[batch, frame, feature])
            constraint = wp.float32(predictions[batch + batches, frame, feature])
            unconditional = wp.float32(predictions[batch + 2 * batches, frame, feature])
            output[batch, frame, feature] = DTYPE(
                unconditional
                + text_weight * (text - unconditional)
                + constraint_weight * (constraint - unconditional)
            )

    @wp.kernel(enable_backward=False, module="unique")
    def ddim_step(
        noisy: wp.array3d(dtype=DTYPE),
        clean: wp.array3d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
        alpha: wp.float32,
        previous_alpha: wp.float32,
    ):
        batch, frame, feature = wp.tid()
        x = wp.float32(noisy[batch, frame, feature])
        prediction = wp.float32(clean[batch, frame, feature])
        noise = (x / wp.sqrt(alpha) - prediction) / wp.sqrt(
            (wp.float32(1.0) - alpha) / alpha
        )
        output[batch, frame, feature] = DTYPE(
            prediction * wp.sqrt(previous_alpha)
            + wp.sqrt(wp.float32(1.0) - previous_alpha) * noise
        )

    return normalize, apply_condition, global_root_to_local, guidance, ddim_step


class KimodoDiffusionPlan:
    """Fixed-buffer, graph-capturable normalization/CFG/DDIM utilities."""

    def __init__(self, batch, frames, config, stats, *, dtype=wp.float32, device=None):
        if dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError("Kimodo supports FP16, BF16, or FP32 storage")
        self.device = wp.get_device(device)
        self.config = config
        self.dtype = dtype
        shape = (batch, frames, config.motion_dim)
        self.motion = wp.empty(shape, dtype=dtype, device=self.device)
        self.observed = wp.zeros(shape, dtype=dtype, device=self.device)
        self.mask = wp.zeros(shape, dtype=wp.bool, device=self.device)
        self.conditioned = wp.empty_like(self.motion)
        self.clean = wp.empty_like(self.motion)
        self.next_motion = wp.empty_like(self.motion)
        self.lengths = wp.empty((batch,), dtype=wp.int32, device=self.device)
        self.local_root = wp.empty((batch, frames, 4), dtype=dtype, device=self.device)
        self.mean = wp.array(stats.mean, dtype=wp.float32, device=self.device)
        self.std = wp.array(stats.std, dtype=wp.float32, device=self.device)
        self.global_mean = wp.array(
            stats.global_mean, dtype=wp.float32, device=self.device
        )
        self.global_std = wp.array(
            stats.global_std, dtype=wp.float32, device=self.device
        )
        self.local_mean = wp.array(
            stats.local_mean, dtype=wp.float32, device=self.device
        )
        self.local_std = wp.array(stats.local_std, dtype=wp.float32, device=self.device)
        self.epsilon = stats.epsilon
        self._normalize, self._condition, self._root, self._guidance, self._ddim = (
            _motion_kernels(dtype)
        )

    def apply_conditions(self):
        wp.launch(
            self._condition,
            dim=self.motion.shape,
            inputs=[self.motion, self.observed, self.mask, self.conditioned],
            device=self.device,
        )
        return self.conditioned

    def root_to_local(self, root):
        wp.launch(
            self._root,
            dim=root.shape[:2],
            inputs=[
                root,
                self.lengths,
                self.global_mean,
                self.global_std,
                self.local_mean,
                self.local_std,
                self.local_root,
                wp.float32(self.config.fps),
                wp.float32(self.epsilon),
            ],
            device=self.device,
        )
        return self.local_root

    def step(self, alpha, previous_alpha):
        wp.launch(
            self._ddim,
            dim=self.motion.shape,
            inputs=[
                self.motion,
                self.clean,
                self.next_motion,
                wp.float32(alpha),
                wp.float32(previous_alpha),
            ],
            device=self.device,
        )
        self.motion, self.next_motion = self.next_motion, self.motion
        return self.motion


def sinusoidal_encoding(length: int, width: int):
    """Build the exact fixed positional table used by Kimodo/PyTorch."""
    if length <= 0 or width <= 0 or width % 2:
        raise ValueError("sinusoidal encoding requires positive length and even width")
    position = np.arange(length, dtype=np.float32)[:, None]
    frequency = np.power(
        np.float32(10000.0),
        -np.arange(0, width, 2, dtype=np.float32) / np.float32(width),
    )
    output = np.empty((length, width), dtype=np.float32)
    output[:, 0::2] = np.sin(position * frequency)
    output[:, 1::2] = np.cos(position * frequency)
    return output


@lru_cache(maxsize=None)
def _denoiser_kernels(dtype):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def bias_silu(x: wp.array2d(dtype=DTYPE), bias: wp.array1d(dtype=DTYPE)):
        row, column = wp.tid()
        value = wp.float32(x[row, column]) + wp.float32(bias[column])
        x[row, column] = DTYPE(value / (wp.float32(1.0) + wp.exp(-value)))

    @wp.kernel(enable_backward=False, module="unique")
    def timestep_input(
        timesteps: wp.array1d[wp.int32],
        positional: wp.array2d[wp.float32],
        output: wp.array2d(dtype=DTYPE),
    ):
        batch, column = wp.tid()
        step = wp.clamp(timesteps[batch], 0, positional.shape[0] - 1)
        output[batch, column] = DTYPE(positional[step, column])

    @wp.kernel(enable_backward=False, module="unique")
    def heading_input(heading: wp.array1d[wp.float32], output: wp.array2d(dtype=DTYPE)):
        batch, column = wp.tid()
        output[batch, column] = DTYPE(
            wp.cos(heading[batch]) if column == 0 else wp.sin(heading[batch])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def root_input(
        motion: wp.array3d(dtype=DTYPE),
        mask: wp.array3d(dtype=wp.bool),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        width = motion.shape[2]
        batch = row / motion.shape[1]
        frame = row % motion.shape[1]
        if column < width:
            output[row, column] = motion[batch, frame, column]
        else:
            output[row, column] = DTYPE(
                wp.float32(1.0)
                if mask[batch, frame, column - width]
                else wp.float32(0.0)
            )

    @wp.kernel(enable_backward=False, module="unique")
    def body_input(
        motion: wp.array3d(dtype=DTYPE),
        local_root: wp.array3d(dtype=DTYPE),
        mask: wp.array3d(dtype=wp.bool),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        batch = row / motion.shape[1]
        frame = row % motion.shape[1]
        local_width = motion.shape[2] - 1
        if column < 4:
            output[row, column] = local_root[batch, frame, column]
        elif column < local_width:
            output[row, column] = motion[batch, frame, column + 1]
        else:
            output[row, column] = DTYPE(
                wp.float32(1.0)
                if mask[batch, frame, column - local_width]
                else wp.float32(0.0)
            )

    @wp.kernel(enable_backward=False, module="unique")
    def assemble_sequence(
        text: wp.array2d(dtype=DTYPE),
        time: wp.array2d(dtype=DTYPE),
        heading: wp.array2d(dtype=DTYPE),
        motion: wp.array2d(dtype=DTYPE),
        motion_valid: wp.array2d(dtype=wp.bool),
        positional: wp.array2d[wp.float32],
        output: wp.array3d(dtype=DTYPE),
        valid: wp.array2d(dtype=wp.bool),
        text_tokens: wp.int32,
    ):
        batch, token, column = wp.tid()
        motion_token = token - text_tokens - 2
        if token < text_tokens:
            value = wp.float32(text[batch * text_tokens + token, column])
            valid[batch, token] = True
        elif token == text_tokens:
            value = wp.float32(time[batch, column])
            valid[batch, token] = True
        elif token == text_tokens + 1:
            value = wp.float32(heading[batch, column])
            valid[batch, token] = True
        else:
            value = wp.float32(
                motion[batch * motion_valid.shape[1] + motion_token, column]
            )
            valid[batch, token] = motion_valid[batch, motion_token]
        output[batch, token, column] = DTYPE(value + positional[token, column])

    @wp.kernel(enable_backward=False, module="unique")
    def extract_motion(
        sequence: wp.array3d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        prefix: wp.int32,
    ):
        row, column = wp.tid()
        frame_count = output.shape[0] / sequence.shape[0]
        output[row, column] = sequence[
            row / frame_count, prefix + row % frame_count, column
        ]

    @wp.kernel(enable_backward=False, module="unique")
    def combine(
        root: wp.array2d(dtype=DTYPE),
        body: wp.array2d(dtype=DTYPE),
        output: wp.array3d(dtype=DTYPE),
    ):
        batch, frame, column = wp.tid()
        row = batch * output.shape[1] + frame
        output[batch, frame, column] = (
            root[row, column] if column < 5 else body[row, column - 5]
        )

    return (
        bias_silu,
        timestep_input,
        heading_input,
        root_input,
        body_input,
        assemble_sequence,
        extract_motion,
        combine,
    )


class _LinearPlan:
    def __init__(self, x, weight, bias, *, activation=None, cublas=None):
        self.device = x.device
        self.bias = bias
        self.activation = activation
        self.tensors = {"x": x, "weight": weight}
        self.shapes = {name: value.shape for name, value in self.tensors.items()}
        self.operation = Operation("Linear", ["x", "weight"], ["output"])
        plan_linear(self.operation, self.tensors, self.shapes, self.device, cublas)
        self.output = self.tensors["output"]
        self._bias = _encoder_kernels(x.dtype, 1)[0]
        self._silu = _denoiser_kernels(x.dtype)[0]

    def execute(self):
        execute_operations([self.operation], self.tensors, self.shapes, self.device)
        kernel = self._silu if self.activation == "silu" else self._bias
        wp.launch(
            kernel,
            dim=self.output.shape,
            inputs=[self.output, self.bias],
            device=self.device,
        )
        return self.output


class _KimodoStagePlan:
    def __init__(
        self,
        stage,
        stage_input,
        motion_valid,
        text,
        timesteps,
        heading,
        weights,
        config,
        *,
        cublas=None,
    ):
        self.device = stage_input.device
        self.config = config
        self.batch = text.shape[0]
        self.frames = motion_valid.shape[1]
        self.prefix = config.text_tokens + 2
        self.timesteps = timesteps
        self.heading = heading
        self.motion_valid = motion_valid
        dtype = stage_input.dtype
        p = stage
        self.text = _LinearPlan(
            text.reshape((-1, config.text_dim)),
            weights[f"{p}.embed_text.weight"],
            weights[f"{p}.embed_text.bias"],
            cublas=cublas,
        )
        self.motion = _LinearPlan(
            stage_input,
            weights[f"{p}.input_linear.weight"],
            weights[f"{p}.input_linear.bias"],
            cublas=cublas,
        )
        self.time_input = wp.empty(
            (self.batch, config.latent_dim), dtype=dtype, device=self.device
        )
        self.time1 = _LinearPlan(
            self.time_input,
            weights[f"{p}.embed_timestep.time_embed.0.weight"],
            weights[f"{p}.embed_timestep.time_embed.0.bias"],
            activation="silu",
            cublas=cublas,
        )
        self.time2 = _LinearPlan(
            self.time1.output,
            weights[f"{p}.embed_timestep.time_embed.2.weight"],
            weights[f"{p}.embed_timestep.time_embed.2.bias"],
            cublas=cublas,
        )
        self.heading_input = wp.empty((self.batch, 2), dtype=dtype, device=self.device)
        self.heading_projection = _LinearPlan(
            self.heading_input,
            weights[f"{p}.linear_first_heading_angle.weight"],
            weights[f"{p}.linear_first_heading_angle.bias"],
            cublas=cublas,
        )
        sequence_length = self.prefix + self.frames
        self.positional = wp.array(
            sinusoidal_encoding(
                max(config.diffusion_steps, sequence_length), config.latent_dim
            ),
            dtype=wp.float32,
            device=self.device,
        )
        self.sequence = wp.empty(
            (self.batch, sequence_length, config.latent_dim),
            dtype=dtype,
            device=self.device,
        )
        self.valid = wp.empty(
            (self.batch, sequence_length), dtype=wp.bool, device=self.device
        )
        self.encoder = EncoderStackPlan(
            self.sequence,
            self.valid,
            weights,
            f"{p}.seqTransEncoder",
            config.layers,
            config.heads,
            cublas=cublas,
        )
        self.motion_hidden = wp.empty(
            (self.batch * self.frames, config.latent_dim),
            dtype=dtype,
            device=self.device,
        )
        self.output_projection = _LinearPlan(
            self.motion_hidden,
            weights[f"{p}.output_linear.weight"],
            weights[f"{p}.output_linear.bias"],
            cublas=cublas,
        )
        self._kernels = _denoiser_kernels(dtype)

    def execute(self):
        _, time_kernel, heading_kernel, _, _, assemble, extract, _ = self._kernels
        wp.launch(
            time_kernel,
            dim=self.time_input.shape,
            inputs=[self.timesteps, self.positional, self.time_input],
            device=self.device,
        )
        self.time1.execute()
        self.time2.execute()
        wp.launch(
            heading_kernel,
            dim=self.heading_input.shape,
            inputs=[self.heading, self.heading_input],
            device=self.device,
        )
        self.heading_projection.execute()
        self.text.execute()
        self.motion.execute()
        wp.launch(
            assemble,
            dim=self.sequence.shape,
            inputs=[
                self.text.output,
                self.time2.output,
                self.heading_projection.output,
                self.motion.output,
                self.motion_valid,
                self.positional,
                self.sequence,
                self.valid,
                self.config.text_tokens,
            ],
            device=self.device,
        )
        self.encoder.execute()
        wp.launch(
            extract,
            dim=self.motion_hidden.shape,
            inputs=[self.encoder.output, self.motion_hidden, self.prefix],
            device=self.device,
        )
        return self.output_projection.execute()


class KimodoDenoiserPlan:
    """Fixed-shape two-stage Kimodo denoiser, ready for CUDA graph capture."""

    def __init__(
        self,
        motion,
        motion_mask,
        motion_valid,
        text,
        timesteps,
        heading,
        weights,
        config,
        stats,
        *,
        cublas=None,
    ):
        if motion.shape != motion_mask.shape:
            raise ValueError("motion and motion mask shapes must match")
        self.device = motion.device
        self.motion = motion
        self.motion_mask = motion_mask
        self.config = config
        batch, frames, _ = motion.shape
        multiplier = 2 if config.concatenate_mask else 1
        self.root_input = wp.empty(
            (batch * frames, config.motion_dim * multiplier),
            dtype=motion.dtype,
            device=self.device,
        )
        body_width = config.body_dim + 4
        if config.concatenate_mask:
            body_width += config.motion_dim
        self.body_input = wp.empty(
            (batch * frames, body_width), dtype=motion.dtype, device=self.device
        )
        args = (motion_valid, text, timesteps, heading, weights, config)
        self.root = _KimodoStagePlan(
            "root_model", self.root_input, *args, cublas=cublas
        )
        self.body = _KimodoStagePlan(
            "body_model", self.body_input, *args, cublas=cublas
        )
        self.output = wp.empty_like(motion)
        self.local_root = wp.empty(
            (batch, frames, 4), dtype=motion.dtype, device=self.device
        )
        self.lengths = wp.empty((batch,), dtype=wp.int32, device=self.device)
        self.global_mean = wp.array(
            stats.global_mean, dtype=wp.float32, device=self.device
        )
        self.global_std = wp.array(
            stats.global_std, dtype=wp.float32, device=self.device
        )
        self.local_mean = wp.array(
            stats.local_mean, dtype=wp.float32, device=self.device
        )
        self.local_std = wp.array(stats.local_std, dtype=wp.float32, device=self.device)
        self.epsilon = stats.epsilon
        self._kernels = _denoiser_kernels(motion.dtype)
        self._root_kernel = _motion_kernels(motion.dtype)[2]

    def execute(self):
        _, _, _, prepare_root, prepare_body, _, _, combine = self._kernels
        wp.launch(
            prepare_root,
            dim=self.root_input.shape,
            inputs=[self.motion, self.motion_mask, self.root_input],
            device=self.device,
        )
        root = self.root.execute()
        root3 = root.reshape((self.motion.shape[0], self.motion.shape[1], 5))
        wp.launch(
            self._root_kernel,
            dim=root3.shape[:2],
            inputs=[
                root3,
                self.lengths,
                self.global_mean,
                self.global_std,
                self.local_mean,
                self.local_std,
                self.local_root,
                wp.float32(self.config.fps),
                wp.float32(self.epsilon),
            ],
            device=self.device,
        )
        wp.launch(
            prepare_body,
            dim=self.body_input.shape,
            inputs=[
                self.motion,
                self.local_root,
                self.motion_mask,
                self.body_input,
            ],
            device=self.device,
        )
        body = self.body.execute()
        wp.launch(
            combine,
            dim=self.output.shape,
            inputs=[root, body, self.output],
            device=self.device,
        )
        return self.output


def load_kimodo_config(path: str | Path):
    """Read the small official config.yaml without adding a YAML dependency."""
    text = Path(path).read_text(encoding="utf-8")

    def scalar(name, cast, default=None):
        matches = re.findall(rf"(?m)^\s*{re.escape(name)}:\s*([^#\n]+?)\s*$", text)
        if not matches:
            if default is not None:
                return default
            raise ValueError(f"Kimodo config has no '{name}' setting")
        value = matches[-1].strip().strip("'\"")
        return cast(value)

    if "SOMASkeleton77" in text:
        joints = 77
    elif "SOMASkeleton30" in text:
        joints = 30
    elif "G1Skeleton34" in text or "g1skel34" in text:
        joints = 34
    elif "SMPLXSkeleton22" in text or "smplx22" in text.lower():
        joints = 22
    else:
        raise ValueError("unsupported or missing Kimodo skeleton in config")
    llm = re.search(r"(?m)^\s*llm_shape:\s*\[\s*\d+\s*,\s*(\d+)\s*\]", text)
    text_dim = int(llm.group(1)) if llm else 4096
    return KimodoConfig(
        12 * joints + 9,
        joints,
        scalar("fps", float),
        scalar("latent_dim", int),
        scalar("ff_size", int),
        scalar("num_layers", int),
        scalar("num_heads", int),
        text_dim=text_dim,
        text_tokens=scalar("num_text_tokens_override", int, 50),
        diffusion_steps=scalar("num_base_steps", int, 1000),
        first_heading=scalar("input_first_heading_angle", str, "true").lower()
        == "true",
        concatenate_mask=scalar("motion_mask_mode", str, "concat") == "concat",
    )


def kimodo_weight_names(config: KimodoConfig):
    """Return the exact PyTorch state-dict names consumed by the denoiser."""
    names = []
    for stage in ("root_model", "body_model"):
        for projection in (
            "embed_text",
            "input_linear",
            "output_linear",
            "linear_first_heading_angle",
            "embed_timestep.time_embed.0",
            "embed_timestep.time_embed.2",
        ):
            names.extend((f"{stage}.{projection}.weight", f"{stage}.{projection}.bias"))
        for layer in range(config.layers):
            prefix = f"{stage}.seqTransEncoder.layers.{layer}"
            names.extend(
                (
                    f"{prefix}.self_attn.in_proj_weight",
                    f"{prefix}.self_attn.in_proj_bias",
                )
            )
            for projection in ("self_attn.out_proj", "linear1", "linear2"):
                names.extend(
                    (
                        f"{prefix}.{projection}.weight",
                        f"{prefix}.{projection}.bias",
                    )
                )
            for norm in ("norm1", "norm2"):
                names.extend((f"{prefix}.{norm}.weight", f"{prefix}.{norm}.bias"))
    return tuple(names)


def load_kimodo_weights(path, config, device=None, dtype=None):
    """Load a released Kimodo safetensors checkpoint with bounded conversion peak."""
    archive = SafeTensorArchive(path)
    expected = kimodo_weight_names(config)
    source_by_runtime = {}
    for name in expected:
        candidates = (name, f"denoiser.backbone.{name}", f"backbone.{name}")
        found = next(
            (candidate for candidate in candidates if candidate in archive.names), None
        )
        if found is None:
            raise KeyError(f"Kimodo checkpoint is missing '{name}'")
        source_by_runtime[name] = found
    loaded = load_cast_weights(archive, source_by_runtime.values(), device, dtype)
    return {runtime: loaded[source] for runtime, source in source_by_runtime.items()}


@lru_cache(maxsize=None)
def _sampling_kernels(dtype):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def prepare_motion(
        motion: wp.array3d(dtype=DTYPE),
        observed: wp.array3d(dtype=DTYPE),
        mask: wp.array3d(dtype=wp.bool),
        expanded: wp.array3d(dtype=DTYPE),
        expanded_mask: wp.array3d(dtype=wp.bool),
        branches: wp.int32,
    ):
        item, frame, feature = wp.tid()
        batch_count = motion.shape[0]
        branch = item / batch_count
        batch = item % batch_count
        constrained = (
            branches == 1
            or (branches == 2 and branch == 0)
            or (branches == 3 and branch == 1)
        )
        active = constrained and mask[batch, frame, feature]
        expanded[item, frame, feature] = (
            observed[batch, frame, feature] if active else motion[batch, frame, feature]
        )
        expanded_mask[item, frame, feature] = active

    @wp.kernel(enable_backward=False, module="unique")
    def combine_guidance(
        prediction: wp.array3d(dtype=DTYPE),
        clean: wp.array3d(dtype=DTYPE),
        weights: wp.array1d[wp.float32],
        branches: wp.int32,
    ):
        batch, frame, feature = wp.tid()
        batches = clean.shape[0]
        if branches == 1:
            value = wp.float32(prediction[batch, frame, feature])
        elif branches == 2:
            conditional = wp.float32(prediction[batch, frame, feature])
            unconditional = wp.float32(prediction[batch + batches, frame, feature])
            value = unconditional + weights[0] * (conditional - unconditional)
        else:
            text = wp.float32(prediction[batch, frame, feature])
            constraint = wp.float32(prediction[batch + batches, frame, feature])
            unconditional = wp.float32(prediction[batch + 2 * batches, frame, feature])
            value = (
                unconditional
                + weights[0] * (text - unconditional)
                + weights[1] * (constraint - unconditional)
            )
        clean[batch, frame, feature] = DTYPE(value)

    @wp.kernel(enable_backward=False, module="unique")
    def stage_embedding(
        embedding: wp.array2d(dtype=DTYPE), text: wp.array3d(dtype=DTYPE)
    ):
        batch, column = wp.tid()
        text[batch, 0, column] = embedding[batch, column]

    return prepare_motion, combine_guidance, stage_embedding


class KimodoGenerationPlan:
    """A fixed-size GPU generation plan with a captured denoiser graph."""

    def __init__(
        self,
        batch,
        frames,
        config,
        stats,
        weights,
        *,
        cfg_type="separated",
        dtype=wp.bfloat16,
        device=None,
        cublas=None,
    ):
        branches = {"nocfg": 1, "regular": 2, "separated": 3}.get(cfg_type)
        if branches is None:
            raise ValueError("cfg_type must be nocfg, regular, or separated")
        self.device = wp.get_device(device)
        self.batch, self.frames, self.branches = batch, frames, branches
        self.config = config
        self.motion = wp.empty(
            (batch, frames, config.motion_dim), dtype=dtype, device=self.device
        )
        self.clean = wp.empty_like(self.motion)
        self.observed = wp.zeros_like(self.motion)
        self.mask = wp.zeros(self.motion.shape, dtype=wp.bool, device=self.device)
        expanded_batch = batch * branches
        self.expanded_motion = wp.empty(
            (expanded_batch, frames, config.motion_dim),
            dtype=dtype,
            device=self.device,
        )
        self.expanded_mask = wp.empty(
            self.expanded_motion.shape, dtype=wp.bool, device=self.device
        )
        self.valid = wp.empty(
            (expanded_batch, frames), dtype=wp.bool, device=self.device
        )
        self.text = wp.empty(
            (expanded_batch, config.text_tokens, config.text_dim),
            dtype=dtype,
            device=self.device,
        )
        self.timesteps = wp.empty((expanded_batch,), dtype=wp.int32, device=self.device)
        self.heading = wp.empty((expanded_batch,), dtype=wp.float32, device=self.device)
        self.guidance_weights = wp.empty((2,), dtype=wp.float32, device=self.device)
        self.denoiser = KimodoDenoiserPlan(
            self.expanded_motion,
            self.expanded_mask,
            self.valid,
            self.text,
            self.timesteps,
            self.heading,
            weights,
            config,
            stats,
            cublas=cublas,
        )
        self._prepare, self._combine, self._stage_embedding = _sampling_kernels(dtype)
        self._ddim = _motion_kernels(dtype)[4]
        self._graph = None
        self._capture_ready = False

    def stage(self, text, lengths, *, heading=None, observed=None, mask=None, seed=0):
        if isinstance(text, wp.array):
            if text.shape not in (
                (self.batch, self.config.text_dim),
                (self.batch, 1, self.config.text_dim),
            ):
                raise ValueError("text embeddings do not match the generation plan")
            self.text.zero_()
            wp.launch(
                self._stage_embedding,
                dim=(self.batch, self.config.text_dim),
                inputs=[text.reshape((self.batch, self.config.text_dim)), self.text],
                device=self.device,
            )
        else:
            text = np.asarray(text)
            if text.shape != (
                self.batch,
                self.config.text_tokens,
                self.config.text_dim,
            ):
                raise ValueError("text embeddings do not match the generation plan")
            expanded = np.zeros(
                (self.batch * self.branches, *text.shape[1:]), dtype=text.dtype
            )
            expanded[: self.batch] = text
            self.text.assign(expanded)
        lengths = np.asarray(lengths, dtype=np.int32)
        if (
            lengths.shape != (self.batch,)
            or np.any(lengths < 2)
            or np.any(lengths > self.frames)
        ):
            raise ValueError(
                "motion lengths must be between 2 and the fixed frame count"
            )
        heading = (
            np.zeros(self.batch, dtype=np.float32)
            if heading is None
            else np.asarray(heading, dtype=np.float32)
        )
        if heading.shape != (self.batch,):
            raise ValueError("heading must have one value per sample")
        base_valid = np.arange(self.frames)[None, :] < lengths[:, None]
        self.valid.assign(np.tile(base_valid, (self.branches, 1)))
        self.heading.assign(np.tile(heading, self.branches))
        self.denoiser.lengths.assign(np.tile(lengths, self.branches))
        if observed is None:
            self.observed.zero_()
        else:
            self.observed.assign(np.asarray(observed))
        if mask is None:
            self.mask.zero_()
        else:
            self.mask.assign(np.asarray(mask, dtype=bool))
        noise = (
            np.random.default_rng(seed)
            .normal(size=self.motion.shape)
            .astype(np.float32)
        )
        self.motion.assign(noise)

    def _execute_denoiser(self):
        wp.launch(
            self._prepare,
            dim=self.expanded_motion.shape,
            inputs=[
                self.motion,
                self.observed,
                self.mask,
                self.expanded_motion,
                self.expanded_mask,
                self.branches,
            ],
            device=self.device,
        )
        prediction = self.denoiser.execute()
        wp.launch(
            self._combine,
            dim=self.clean.shape,
            inputs=[
                prediction,
                self.clean,
                self.guidance_weights,
                self.branches,
            ],
            device=self.device,
        )

    def denoise(self, denoising_steps=100, *, text_weight=2.0, constraint_weight=2.0):
        selected, alpha, previous = cosine_ddim_schedule(
            self.config.diffusion_steps, denoising_steps
        )
        self.guidance_weights.assign(
            np.array([text_weight, constraint_weight], dtype=np.float32)
        )
        for index in range(len(selected) - 1, -1, -1):
            mapped = int(selected[index])
            self.timesteps.assign(
                np.full(self.batch * self.branches, mapped, dtype=np.int32)
            )
            if self.device.is_cuda and self._graph is not None:
                wp.capture_launch(self._graph)
            elif self.device.is_cuda and self._capture_ready:
                wp.capture_begin(device=self.device)
                self._execute_denoiser()
                self._graph = wp.capture_end(device=self.device)
                wp.capture_launch(self._graph)
            else:
                self._execute_denoiser()
                self._capture_ready = True
            wp.launch(
                self._ddim,
                dim=self.motion.shape,
                inputs=[
                    self.motion,
                    self.clean,
                    self.motion,
                    wp.float32(alpha[index]),
                    wp.float32(previous[index]),
                ],
                device=self.device,
            )
        return self.motion


class KimodoRunner:
    """Load one Kimodo release and cache fixed-size generation plans."""

    def __init__(
        self,
        model_path,
        *,
        text_model_path=None,
        text_adapter_paths=(),
        dtype=wp.bfloat16,
        device=None,
        use_cublas=False,
    ):
        model_path = Path(model_path)
        self.device = wp.get_device(device)
        self.config = load_kimodo_config(model_path / "config.yaml")
        self.stats = KimodoStats.load(model_path / "stats")
        self.weights = load_kimodo_weights(model_path, self.config, self.device, dtype)
        self.dtype = dtype
        self.use_cublas = use_cublas
        self._plans = {}
        self.text_encoder = None
        if text_model_path is not None:
            from .llama_encoder import LLM2VecRunner

            self.text_encoder = LLM2VecRunner(
                text_model_path,
                text_adapter_paths,
                dtype=dtype,
                device=self.device,
                use_cublas=use_cublas,
            )

    def plan(self, frames, batch=1, cfg_type="separated"):
        key = (batch, frames, cfg_type)
        if key not in self._plans:
            cublas = None
            if self.use_cublas:
                from ._cublas import Cublas

                cublas = Cublas(self.device)
            self._plans[key] = KimodoGenerationPlan(
                batch,
                frames,
                self.config,
                self.stats,
                self.weights,
                cfg_type=cfg_type,
                dtype=self.dtype,
                device=self.device,
                cublas=cublas,
            )
        return self._plans[key]

    def generate(
        self,
        prompt,
        frames,
        *,
        denoising_steps=100,
        cfg_type="separated",
        text_weight=2.0,
        constraint_weight=2.0,
        heading=None,
        observed=None,
        mask=None,
        seed=0,
    ):
        """Encode one prompt and generate normalized Kimodo motion features."""
        if self.text_encoder is None:
            raise RuntimeError(
                "generate(prompt) requires a configured LLM2Vec text encoder"
            )
        embedding = self.text_encoder.encode(prompt)
        plan = self.plan(frames, 1, cfg_type)
        plan.stage(
            embedding,
            np.array([frames], dtype=np.int32),
            heading=heading,
            observed=observed,
            mask=mask,
            seed=seed,
        )
        return plan.denoise(
            denoising_steps,
            text_weight=text_weight,
            constraint_weight=constraint_weight,
        )


def decode_motion_features(features, stats: KimodoStats, joints: int):
    """Decode normalized Kimodo features into portable NumPy motion arrays.

    This returns all model-native geometric quantities without depending on the
    Kimodo Python package, PyTorch, einops, or a renderer.
    """
    if isinstance(features, wp.array):
        features = features.numpy()
    features = np.asarray(features, dtype=np.float32)
    expected = 12 * joints + 9
    if features.ndim not in (2, 3) or features.shape[-1] != expected:
        raise ValueError(f"motion features must end in width {expected}")
    unnormalized = features * np.sqrt(stats.std**2 + stats.epsilon) + stats.mean
    offset = 0

    def take(width):
        nonlocal offset
        value = unnormalized[..., offset : offset + width]
        offset += width
        return value

    smooth_root = take(3)
    heading = take(2)
    local_positions = take(joints * 3).reshape(*features.shape[:-1], joints, 3)
    rotation6d = take(joints * 6).reshape(*features.shape[:-1], joints, 6)
    velocities = take(joints * 3).reshape(*features.shape[:-1], joints, 3)
    contacts = take(4) > 0.5
    first = rotation6d[..., :3]
    second = rotation6d[..., 3:]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1.0e-12)
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1.0e-12)
    third = np.cross(first, second)
    global_rotations = np.stack((first, second, third), axis=-1)
    posed = local_positions.copy()
    posed[..., 0] += smooth_root[..., None, 0]
    posed[..., 2] += smooth_root[..., None, 2]
    return {
        "features": unnormalized,
        "smooth_root_pos": smooth_root,
        "global_root_heading": heading,
        "posed_joints": posed,
        "root_positions": posed[..., 0, :],
        "global_rot_mats": global_rotations,
        "velocities": velocities,
        "foot_contacts": contacts,
    }


def save_motion_npz(path, motion, *, fps):
    """Save decoded motion in a dependency-free, Kimodo-friendly NPZ file."""
    required = (
        "posed_joints",
        "root_positions",
        "global_rot_mats",
        "foot_contacts",
    )
    missing = [name for name in required if name not in motion]
    if missing:
        raise ValueError(f"decoded motion is missing: {', '.join(missing)}")
    np.savez_compressed(
        path,
        fps=np.float32(fps),
        **{name: np.asarray(value) for name, value in motion.items()},
    )
