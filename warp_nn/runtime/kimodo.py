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
from typing import Mapping

import numpy as np
import warp as wp


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
        return cls(np.load(folder / "mean.npy"), np.load(folder / "std.npy"))


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
    selected = np.rint(np.arange(base_steps) * stride).astype(np.int64)
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
            stats.mean[:5], dtype=wp.float32, device=self.device
        )
        self.global_std = wp.array(stats.std[:5], dtype=wp.float32, device=self.device)
        # Local-root statistics follow the global motion vector in Kimodo stats.
        self.local_mean = wp.array(
            stats.mean[-4:], dtype=wp.float32, device=self.device
        )
        self.local_std = wp.array(stats.std[-4:], dtype=wp.float32, device=self.device)
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
