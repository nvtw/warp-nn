# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-buffer FP32 AdamW updates for low-precision training parameters."""

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import warp as wp


_PARAMETER_DTYPES = (wp.float16, wp.bfloat16, wp.float32)
_NORM_TILE_SIZE = 256


@wp.func
def _square(value: wp.float32):
    return value * value


@wp.kernel(enable_backward=False)
def _gradient_square_partials(
    gradient: wp.array1d[wp.float32],
    partials: wp.array1d[wp.float32],
    multiplier: wp.array1d[wp.float32],
    output_offset: int,
):
    tile = wp.tid()
    values = wp.tile_load(
        gradient, shape=(_NORM_TILE_SIZE,), offset=(tile * _NORM_TILE_SIZE,)
    )
    values *= multiplier[0]
    wp.tile_store(
        partials, wp.tile_sum(wp.tile_map(_square, values)), offset=output_offset + tile
    )


@wp.kernel(enable_backward=False)
def _reduce_norm_partials(
    source: wp.array1d[wp.float32], destination: wp.array1d[wp.float32]
):
    tile = wp.tid()
    values = wp.tile_load(
        source, shape=(_NORM_TILE_SIZE,), offset=(tile * _NORM_TILE_SIZE,)
    )
    wp.tile_store(destination, wp.tile_sum(values), offset=tile)


@wp.kernel(enable_backward=False)
def _apply_global_norm_clip(
    sum_squares: wp.array1d[wp.float32],
    multiplier: wp.array1d[wp.float32],
    all_finite: wp.array1d[wp.int32],
    step_enabled: wp.array1d[wp.int32],
    global_grad_norm: wp.array1d[wp.float32],
    clip_scale: wp.array1d[wp.float32],
    max_grad_norm: wp.float32,
):
    norm = wp.sqrt(sum_squares[0])
    global_grad_norm[0] = norm
    clip_scale[0] = wp.float32(1.0)
    maximum = wp.float32(3.402823466e38)
    if not (norm <= maximum):
        all_finite[0] = 0
    elif step_enabled[0] != 0:
        if norm > max_grad_norm:
            scale = max_grad_norm / norm
            clip_scale[0] = scale
            multiplier[0] *= scale


@wp.kernel(enable_backward=False)
def _prepare_step(
    valid_token_count: wp.array1d[wp.int32],
    all_finite: wp.array1d[wp.int32],
    step_enabled: wp.array1d[wp.int32],
    multiplier: wp.array1d[wp.float32],
    step: wp.array1d[wp.int32],
    effective_learning_rate: wp.array1d[wp.float32],
    normalize_by_valid_tokens: bool,
    loss_scale: wp.float32,
    legacy_multiplier: wp.float32,
    learning_rate: wp.float32,
    warmup_steps: wp.int32,
    total_steps: wp.int32,
    min_learning_rate_ratio: wp.float32,
):
    all_finite[0] = 1
    step_enabled[0] = 1
    multiplier[0] = legacy_multiplier
    if normalize_by_valid_tokens:
        count = valid_token_count[0]
        if count > 0:
            multiplier[0] = wp.float32(1.0) / (loss_scale * wp.float32(count))
        else:
            step_enabled[0] = 0
    factor = wp.float32(1.0)
    candidate_step = step[0] + 1
    if total_steps > 0:
        if warmup_steps > 0 and candidate_step <= warmup_steps:
            factor = wp.float32(candidate_step) / wp.float32(warmup_steps)
        else:
            progress = wp.clamp(
                wp.float32(candidate_step - warmup_steps)
                / wp.float32(total_steps - warmup_steps),
                wp.float32(0.0),
                wp.float32(1.0),
            )
            cosine = wp.float32(0.5) * (
                wp.float32(1.0) + wp.cos(wp.float32(3.141592653589793) * progress)
            )
            factor = (
                min_learning_rate_ratio
                + (wp.float32(1.0) - min_learning_rate_ratio) * cosine
            )
    effective_learning_rate[0] = learning_rate * factor


@wp.kernel(enable_backward=False)
def _check_finite(
    gradient: wp.array1d[wp.float32],
    multiplier: wp.array1d[wp.float32],
    all_finite: wp.array1d[wp.int32],
):
    value = gradient[wp.tid()] * multiplier[0]
    square = value * value
    maximum = wp.float32(3.402823466e38)
    if not (wp.abs(value) <= maximum and wp.abs(square) <= maximum):
        wp.atomic_min(all_finite, 0, 0)


@wp.kernel(enable_backward=False)
def _advance_step(
    step: wp.array1d[wp.int32],
    all_finite: wp.array1d[wp.int32],
    step_enabled: wp.array1d[wp.int32],
):
    if all_finite[0] != 0 and step_enabled[0] != 0:
        step[0] += 1


@wp.kernel(enable_backward=False)
def _accumulate_valid_tokens(count: wp.array1d[wp.int32], total: wp.array1d[wp.int32]):
    total[0] += count[0]


@wp.kernel(enable_backward=False)
def _zero_int_scalar(value: wp.array1d[wp.int32]):
    value[0] = 0


@wp.kernel(enable_backward=False)
def _zero_gradient(gradient: wp.array1d[wp.float32]):
    gradient[wp.tid()] = wp.float32(0.0)


@dataclass(frozen=True)
class _AdamWKernels:
    initialize: object
    update: object


@lru_cache(maxsize=None)
def _get_adamw_kernels(dtype: type) -> _AdamWKernels:
    if dtype not in _PARAMETER_DTYPES:
        raise TypeError("AdamW parameters must use FP16, BF16, or FP32 storage")
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def initialize(
        parameter: wp.array1d(dtype=DTYPE),
        master: wp.array1d[wp.float32],
    ):
        index = wp.tid()
        master[index] = wp.float32(parameter[index])

    @wp.kernel(enable_backward=False, module="unique")
    def update(
        parameter: wp.array1d(dtype=DTYPE),
        master: wp.array1d[wp.float32],
        gradient: wp.array1d[wp.float32],
        first_moment: wp.array1d[wp.float32],
        second_moment: wp.array1d[wp.float32],
        step: wp.array1d[wp.int32],
        all_finite: wp.array1d[wp.int32],
        step_enabled: wp.array1d[wp.int32],
        gradient_multiplier: wp.array1d[wp.float32],
        learning_rate: wp.array1d[wp.float32],
        beta1: wp.float32,
        beta2: wp.float32,
        epsilon: wp.float32,
        weight_decay: wp.float32,
    ):
        index = wp.tid()
        if all_finite[0] == 0 or step_enabled[0] == 0:
            return
        grad = gradient[index] * gradient_multiplier[0]
        first = beta1 * first_moment[index] + (wp.float32(1.0) - beta1) * grad
        second = beta2 * second_moment[index] + (wp.float32(1.0) - beta2) * grad * grad
        first_moment[index] = first
        second_moment[index] = second
        step_value = wp.float32(step[0])
        corrected_first = first / (wp.float32(1.0) - wp.pow(beta1, step_value))
        corrected_second = second / (wp.float32(1.0) - wp.pow(beta2, step_value))
        value = master[index]
        update = corrected_first / (wp.sqrt(corrected_second) + epsilon)
        updated = value - learning_rate[0] * (update + weight_decay * value)
        master[index] = updated
        parameter[index] = DTYPE(updated)

    initialize.module.options["enable_backward"] = False
    update.module.options["enable_backward"] = False
    return _AdamWKernels(initialize, update)


@dataclass(frozen=True)
class _AdamWSlot:
    parameter: wp.array
    master: wp.array
    gradient: wp.array
    first_moment: wp.array
    second_moment: wp.array


class AdamWPlan:
    """Graph-safe AdamW state bound to fixed parameter and gradient buffers.

    Gradients are expected to include ``loss_scale`` and to be sums over
    ``accumulation_steps`` microbatches. ``step`` unscales and averages them once.
    With ``normalize_by_valid_tokens``, use summed loss gradients and call
    :meth:`accumulate_valid_tokens` after each microbatch instead.
    """

    def __init__(
        self,
        parameters: Sequence[wp.array],
        gradients: Sequence[wp.array],
        *,
        learning_rate: float = 1.0e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1.0e-8,
        weight_decay: float = 0.0,
        loss_scale: float = 1.0,
        accumulation_steps: int = 1,
        normalize_by_valid_tokens: bool = False,
        warmup_steps: int = 0,
        total_steps: int | None = None,
        min_learning_rate_ratio: float = 0.0,
        max_grad_norm: float | None = None,
    ):
        if not parameters or len(parameters) != len(gradients):
            raise ValueError(
                "AdamW requires matching non-empty parameter and gradient lists"
            )
        if learning_rate <= 0.0 or epsilon <= 0.0 or loss_scale <= 0.0:
            raise ValueError("learning_rate, epsilon, and loss_scale must be positive")
        if not all(
            math.isfinite(value)
            for value in (
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                loss_scale,
            )
        ):
            raise ValueError("AdamW hyperparameters must be finite")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("AdamW beta values must be in [0, 1)")
        if weight_decay < 0.0 or accumulation_steps < 1:
            raise ValueError(
                "weight_decay must be non-negative and accumulation_steps positive"
            )
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if total_steps is not None and (
            not isinstance(total_steps, int) or total_steps <= 0
        ):
            raise ValueError("total_steps must be a positive integer or None")
        if total_steps is None:
            if warmup_steps != 0 or min_learning_rate_ratio != 0.0:
                raise ValueError("warmup and minimum learning rate require total_steps")
        elif warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")
        if not math.isfinite(min_learning_rate_ratio) or not (
            0.0 <= min_learning_rate_ratio <= 1.0
        ):
            raise ValueError("min_learning_rate_ratio must be in [0, 1]")
        if max_grad_norm is not None and (
            not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0
        ):
            raise ValueError("max_grad_norm must be positive and finite or None")

        device = parameters[0].device
        slots = []
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            if parameter.dtype not in _PARAMETER_DTYPES:
                raise TypeError(
                    f"parameter {index} must use FP16, BF16, or FP32 storage"
                )
            if gradient.dtype != wp.float32 or gradient.shape != parameter.shape:
                raise TypeError(
                    f"gradient {index} must be FP32 with the parameter shape"
                )
            if parameter.device != device or gradient.device != device:
                raise ValueError(
                    "all AdamW parameters and gradients must share one device"
                )
            if parameter.size == 0:
                raise ValueError(f"parameter and gradient {index} must be non-empty")
            if not parameter.is_contiguous or not gradient.is_contiguous:
                raise ValueError(f"parameter and gradient {index} must be contiguous")
            parameter_flat = parameter.flatten()
            gradient_flat = gradient.flatten()
            master = wp.empty(parameter.size, dtype=wp.float32, device=device)
            wp.launch(
                _get_adamw_kernels(parameter.dtype).initialize,
                dim=parameter.size,
                inputs=[parameter_flat, master],
                device=device,
            )
            slots.append(
                _AdamWSlot(
                    parameter_flat,
                    master,
                    gradient_flat,
                    wp.zeros(parameter.size, dtype=wp.float32, device=device),
                    wp.zeros(parameter.size, dtype=wp.float32, device=device),
                )
            )

        self.device = device
        self._slots = tuple(slots)
        self.step_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.valid_token_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.all_finite = wp.ones(1, dtype=wp.int32, device=device)
        self.step_enabled = wp.ones(1, dtype=wp.int32, device=device)
        self.normalization_multiplier = wp.empty(1, dtype=wp.float32, device=device)
        self.effective_learning_rate = wp.empty(1, dtype=wp.float32, device=device)
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.loss_scale = float(loss_scale)
        self.gradient_multiplier = 1.0 / (self.loss_scale * accumulation_steps)
        self.normalize_by_valid_tokens = bool(normalize_by_valid_tokens)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = 0 if total_steps is None else int(total_steps)
        self.min_learning_rate_ratio = float(min_learning_rate_ratio)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self._gradient_norm_levels: tuple[wp.array, ...] = ()
        self._gradient_norm_offsets: tuple[int, ...] = ()
        self.global_grad_norm: wp.array | None = None
        self.clip_scale: wp.array | None = None
        if self.max_grad_norm is not None:
            partial_counts = tuple(
                (slot.gradient.size + _NORM_TILE_SIZE - 1) // _NORM_TILE_SIZE
                for slot in self._slots
            )
            total_partials = sum(partial_counts)
            levels = [wp.empty(total_partials, dtype=wp.float32, device=device)]
            while levels[-1].size > 1:
                size = (levels[-1].size + _NORM_TILE_SIZE - 1) // _NORM_TILE_SIZE
                levels.append(wp.empty(size, dtype=wp.float32, device=device))
            offsets = []
            offset = 0
            for count in partial_counts:
                offsets.append(offset)
                offset += count
            self._gradient_norm_levels = tuple(levels)
            self._gradient_norm_offsets = tuple(offsets)
            self.global_grad_norm = wp.empty(1, dtype=wp.float32, device=device)
            self.clip_scale = wp.empty(1, dtype=wp.float32, device=device)

    @property
    def first_moments(self) -> tuple[wp.array, ...]:
        return tuple(slot.first_moment for slot in self._slots)

    @property
    def masters(self) -> tuple[wp.array, ...]:
        """Return the authoritative FP32 parameter buffers for inspection."""
        return tuple(slot.master for slot in self._slots)

    @property
    def second_moments(self) -> tuple[wp.array, ...]:
        return tuple(slot.second_moment for slot in self._slots)

    def zero_grad(self) -> None:
        """Zero bound gradients and begin a fresh valid-token accumulation window."""
        for slot in self._slots:
            wp.launch(
                _zero_gradient,
                dim=slot.gradient.size,
                inputs=[slot.gradient],
                device=self.device,
            )
        wp.launch(
            _zero_int_scalar,
            dim=1,
            inputs=[self.valid_token_count],
            device=self.device,
        )

    def accumulate_valid_tokens(self, count: wp.array) -> None:
        """Add one microbatch FP32-loss valid count into device-owned step state."""
        if (
            not isinstance(count, wp.array)
            or count.dtype != wp.int32
            or count.shape != (1,)
        ):
            raise TypeError("valid token count must be an INT32 array with shape (1,)")
        if count.device != self.device:
            raise ValueError("valid token count must be on the optimizer device")
        wp.launch(
            _accumulate_valid_tokens,
            dim=1,
            inputs=[count, self.valid_token_count],
            device=self.device,
        )

    def step(self) -> None:
        """Update only when all gradients are finite and normalization is valid."""
        wp.launch(
            _prepare_step,
            dim=1,
            inputs=[
                self.valid_token_count,
                self.all_finite,
                self.step_enabled,
                self.normalization_multiplier,
                self.step_count,
                self.effective_learning_rate,
                self.normalize_by_valid_tokens,
                self.loss_scale,
                self.gradient_multiplier,
                self.learning_rate,
                self.warmup_steps,
                self.total_steps,
                self.min_learning_rate_ratio,
            ],
            device=self.device,
        )
        if self.max_grad_norm is not None:
            partials = self._gradient_norm_levels[0]
            for slot, offset in zip(self._slots, self._gradient_norm_offsets):
                wp.launch_tiled(
                    _gradient_square_partials,
                    dim=(slot.gradient.size + _NORM_TILE_SIZE - 1) // _NORM_TILE_SIZE,
                    inputs=[
                        slot.gradient,
                        partials,
                        self.normalization_multiplier,
                        offset,
                    ],
                    device=self.device,
                    block_dim=128,
                )
            for source, destination in zip(
                self._gradient_norm_levels, self._gradient_norm_levels[1:]
            ):
                wp.launch_tiled(
                    _reduce_norm_partials,
                    dim=destination.size,
                    inputs=[source, destination],
                    device=self.device,
                    block_dim=128,
                )
            wp.launch(
                _apply_global_norm_clip,
                dim=1,
                inputs=[
                    self._gradient_norm_levels[-1],
                    self.normalization_multiplier,
                    self.all_finite,
                    self.step_enabled,
                    self.global_grad_norm,
                    self.clip_scale,
                    self.max_grad_norm,
                ],
                device=self.device,
            )
        else:
            for slot in self._slots:
                wp.launch(
                    _check_finite,
                    dim=slot.gradient.size,
                    inputs=[
                        slot.gradient,
                        self.normalization_multiplier,
                        self.all_finite,
                    ],
                    device=self.device,
                )
        wp.launch(
            _advance_step,
            dim=1,
            inputs=[self.step_count, self.all_finite, self.step_enabled],
            device=self.device,
        )
        for slot in self._slots:
            wp.launch(
                _get_adamw_kernels(slot.parameter.dtype).update,
                dim=slot.parameter.size,
                inputs=[
                    slot.parameter,
                    slot.master,
                    slot.gradient,
                    slot.first_moment,
                    slot.second_moment,
                    self.step_count,
                    self.all_finite,
                    self.step_enabled,
                    self.normalization_multiplier,
                    self.effective_learning_rate,
                    self.beta1,
                    self.beta2,
                    self.epsilon,
                    self.weight_decay,
                ],
                device=self.device,
            )

    def reset_state(self) -> None:
        """Reset moments, step, and accumulation state without replacing buffers."""
        self.step_count.zero_()
        self.valid_token_count.zero_()
        self.all_finite.fill_(1)
        self.step_enabled.fill_(1)
        for slot in self._slots:
            slot.first_moment.zero_()
            slot.second_moment.zero_()
