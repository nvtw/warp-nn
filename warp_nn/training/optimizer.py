# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-buffer FP32 AdamW updates for low-precision training parameters."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import warp as wp


_PARAMETER_DTYPES = (wp.float16, wp.bfloat16, wp.float32)


@wp.kernel(enable_backward=False)
def _advance_step(step: wp.array1d[wp.int32]):
    step[0] += 1


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
        learning_rate: wp.float32,
        beta1: wp.float32,
        beta2: wp.float32,
        epsilon: wp.float32,
        weight_decay: wp.float32,
        gradient_multiplier: wp.float32,
    ):
        index = wp.tid()
        grad = gradient[index] * gradient_multiplier
        first = beta1 * first_moment[index] + (wp.float32(1.0) - beta1) * grad
        second = beta2 * second_moment[index] + (wp.float32(1.0) - beta2) * grad * grad
        first_moment[index] = first
        second_moment[index] = second
        step_value = wp.float32(step[0])
        corrected_first = first / (wp.float32(1.0) - wp.pow(beta1, step_value))
        corrected_second = second / (wp.float32(1.0) - wp.pow(beta2, step_value))
        value = master[index]
        update = corrected_first / (wp.sqrt(corrected_second) + epsilon)
        updated = value - learning_rate * (update + weight_decay * value)
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
    ):
        if not parameters or len(parameters) != len(gradients):
            raise ValueError(
                "AdamW requires matching non-empty parameter and gradient lists"
            )
        if learning_rate <= 0.0 or epsilon <= 0.0 or loss_scale <= 0.0:
            raise ValueError("learning_rate, epsilon, and loss_scale must be positive")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("AdamW beta values must be in [0, 1)")
        if weight_decay < 0.0 or accumulation_steps < 1:
            raise ValueError(
                "weight_decay must be non-negative and accumulation_steps positive"
            )

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
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.gradient_multiplier = 1.0 / (float(loss_scale) * accumulation_steps)

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
        """Zero every bound FP32 gradient without allocating replacement arrays."""
        for slot in self._slots:
            wp.launch(
                _zero_gradient,
                dim=slot.gradient.size,
                inputs=[slot.gradient],
                device=self.device,
            )

    def step(self) -> None:
        """Advance the device step and update every bound parameter in place."""
        wp.launch(_advance_step, dim=1, inputs=[self.step_count], device=self.device)
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
                    self.learning_rate,
                    self.beta1,
                    self.beta2,
                    self.epsilon,
                    self.weight_decay,
                    self.gradient_multiplier,
                ],
                device=self.device,
            )
