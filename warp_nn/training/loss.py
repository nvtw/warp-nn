# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory-bounded low-precision losses for large language-model vocabularies."""

from dataclasses import dataclass
from functools import lru_cache
import math

import warp as wp

from .primitives import _cross_entropy_reduce_kernel


_PARTITION_SIZE = 256
_STORAGE_DTYPES = (wp.float16, wp.bfloat16)


@dataclass(frozen=True)
class _CrossEntropyKernels:
    partition_maximum: object
    reduce_maximum: object
    partition_sum: object
    finalize: object
    backward: object


@lru_cache(maxsize=None)
def _get_cross_entropy_kernels(dtype: type) -> _CrossEntropyKernels:
    if dtype not in _STORAGE_DTYPES:
        raise TypeError("large-vocabulary cross-entropy supports FP16 and BF16")
    DTYPE = dtype

    @wp.kernel(module="unique")
    def partition_maximum(
        logits: wp.array2d(dtype=DTYPE),
        partials: wp.array2d(dtype=wp.float32),
    ):
        row, partition = wp.tid()
        start = partition * _PARTITION_SIZE
        value = wp.float32(-3.402823466e38)
        for offset in range(_PARTITION_SIZE):
            column = start + offset
            if column < logits.shape[1]:
                value = wp.max(value, wp.float32(logits[row, column]))
        partials[row, partition] = value

    @wp.kernel(module="unique")
    def reduce_maximum(
        partials: wp.array2d(dtype=wp.float32),
        targets: wp.array1d(dtype=wp.int32),
        ignore_index: int,
        maximum: wp.array1d(dtype=wp.float32),
    ):
        row = wp.tid()
        if targets[row] == ignore_index:
            maximum[row] = wp.float32(0.0)
            return
        value = partials[row, 0]
        for partition in range(1, partials.shape[1]):
            value = wp.max(value, partials[row, partition])
        maximum[row] = value

    @wp.kernel(module="unique")
    def partition_sum(
        logits: wp.array2d(dtype=DTYPE),
        targets: wp.array1d(dtype=wp.int32),
        maximum: wp.array1d(dtype=wp.float32),
        ignore_index: int,
        partials: wp.array2d(dtype=wp.float32),
    ):
        row, partition = wp.tid()
        if targets[row] == ignore_index:
            partials[row, partition] = wp.float32(0.0)
            return
        start = partition * _PARTITION_SIZE
        value = wp.float32(0.0)
        for offset in range(_PARTITION_SIZE):
            column = start + offset
            if column < logits.shape[1]:
                value += wp.exp(wp.float32(logits[row, column]) - maximum[row])
        partials[row, partition] = value

    @wp.kernel(module="unique")
    def finalize(
        logits: wp.array2d(dtype=DTYPE),
        targets: wp.array1d(dtype=wp.int32),
        partials: wp.array2d(dtype=wp.float32),
        maximum: wp.array1d(dtype=wp.float32),
        ignore_index: int,
        logsumexp: wp.array1d(dtype=wp.float32),
        losses: wp.array1d(dtype=wp.float32),
        valid: wp.array1d(dtype=wp.int32),
    ):
        row = wp.tid()
        target = targets[row]
        if target == ignore_index:
            logsumexp[row] = wp.float32(0.0)
            losses[row] = wp.float32(0.0)
            valid[row] = 0
            return
        exponential_sum = wp.float32(0.0)
        for partition in range(partials.shape[1]):
            exponential_sum += partials[row, partition]
        shifted_logsumexp = wp.log(exponential_sum)
        logsumexp[row] = shifted_logsumexp
        losses[row] = (
            maximum[row]
            - wp.float32(logits[row, target])
            + shifted_logsumexp
        )
        valid[row] = 1

    @wp.kernel(module="unique")
    def backward(
        logits: wp.array2d(dtype=DTYPE),
        targets: wp.array1d(dtype=wp.int32),
        logsumexp: wp.array1d(dtype=wp.float32),
        maximum: wp.array1d(dtype=wp.float32),
        valid_count: wp.array1d(dtype=wp.int32),
        ignore_index: int,
        loss_scale: wp.float32,
        normalize: bool,
        gradient: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        target = targets[row]
        count = valid_count[0]
        if target == ignore_index or count == 0:
            gradient[row, column] = DTYPE(0.0)
            return
        value = wp.exp(
            wp.float32(logits[row, column]) - maximum[row] - logsumexp[row]
        )
        if column == target:
            value -= wp.float32(1.0)
        divisor = wp.float32(count) if normalize else wp.float32(1.0)
        gradient[row, column] = DTYPE(value * loss_scale / divisor)

    return _CrossEntropyKernels(
        partition_maximum,
        reduce_maximum,
        partition_sum,
        finalize,
        backward,
    )


class LowPrecisionCrossEntropyPlan:
    """Stable parallel cross-entropy without full-vocabulary FP32 storage.

    The plan keeps only ``ceil(classes / 256)`` FP32 partials per row. Logits
    and their gradient use FP16 or BF16. With ``in_place=True``, backward
    overwrites logits after retaining the row statistics it needs, avoiding a
    second large vocabulary buffer. Shapes and storage remain fixed for CUDA
    graph capture.
    """

    def __init__(
        self,
        rows: int,
        classes: int,
        *,
        dtype: type = wp.bfloat16,
        ignore_index: int = -100,
        in_place: bool = False,
        device: object | None = None,
    ):
        if rows <= 0 or classes <= 0:
            raise ValueError("rows and classes must be positive")
        if dtype not in _STORAGE_DTYPES:
            raise TypeError("dtype must be wp.float16 or wp.bfloat16")
        self.device = wp.get_device(device)
        self.rows = rows
        self.classes = classes
        self.dtype = dtype
        self.ignore_index = ignore_index
        self.in_place = bool(in_place)
        self.partitions = (classes + _PARTITION_SIZE - 1) // _PARTITION_SIZE
        self._kernels = _get_cross_entropy_kernels(dtype)
        self.partials = wp.empty(
            (rows, self.partitions), dtype=wp.float32, device=self.device
        )
        self.maximum = wp.empty(rows, dtype=wp.float32, device=self.device)
        self.logsumexp = wp.empty(rows, dtype=wp.float32, device=self.device)
        self.losses = wp.empty(rows, dtype=wp.float32, device=self.device)
        self.valid = wp.empty(rows, dtype=wp.int32, device=self.device)
        self.loss = wp.empty(1, dtype=wp.float32, device=self.device)
        self.valid_count = wp.empty(1, dtype=wp.int32, device=self.device)
        self.gradient = None
        if not self.in_place:
            self.gradient = wp.empty(
                (rows, classes), dtype=dtype, device=self.device
            )

    def _inputs(self, logits: wp.array, targets: wp.array) -> None:
        if not isinstance(logits, wp.array) or logits.ndim != 2:
            raise TypeError("logits must be a 2-D Warp array")
        if tuple(logits.shape) != (self.rows, self.classes):
            raise ValueError(
                f"logits must have shape {(self.rows, self.classes)}, "
                f"got {tuple(logits.shape)}"
            )
        if logits.dtype != self.dtype:
            raise TypeError(f"logits must have dtype {self.dtype}, got {logits.dtype}")
        if logits.device != self.device:
            raise ValueError(f"logits must be on {self.device}, got {logits.device}")
        if not logits.is_contiguous:
            raise ValueError("logits must be contiguous")
        if not isinstance(targets, wp.array) or targets.ndim != 1:
            raise TypeError("targets must be a 1-D Warp array")
        if tuple(targets.shape) != (self.rows,) or targets.dtype != wp.int32:
            raise TypeError("targets must be an int32 array with one item per row")
        if targets.device != self.device:
            raise ValueError(f"targets must be on {self.device}, got {targets.device}")

    @staticmethod
    def _reduction(reduction: str) -> bool:
        if reduction not in ("mean", "sum"):
            raise ValueError("cross-entropy reduction must be mean or sum")
        return reduction == "mean"

    def forward(
        self, logits: wp.array, targets: wp.array, *, reduction: str = "mean"
    ) -> wp.array:
        self._inputs(logits, targets)
        normalize = self._reduction(reduction)
        wp.launch(
            self._kernels.partition_maximum,
            dim=(self.rows, self.partitions),
            inputs=[logits],
            outputs=[self.partials],
            device=self.device,
        )
        wp.launch(
            self._kernels.reduce_maximum,
            dim=self.rows,
            inputs=[self.partials, targets, self.ignore_index],
            outputs=[self.maximum],
            device=self.device,
        )
        wp.launch(
            self._kernels.partition_sum,
            dim=(self.rows, self.partitions),
            inputs=[logits, targets, self.maximum, self.ignore_index],
            outputs=[self.partials],
            device=self.device,
        )
        wp.launch(
            self._kernels.finalize,
            dim=self.rows,
            inputs=[
                logits,
                targets,
                self.partials,
                self.maximum,
                self.ignore_index,
            ],
            outputs=[self.logsumexp, self.losses, self.valid],
            device=self.device,
        )
        wp.launch(
            _cross_entropy_reduce_kernel,
            dim=1,
            inputs=[self.losses, self.valid, normalize],
            outputs=[self.loss, self.valid_count],
            device=self.device,
        )
        return self.loss

    def backward(
        self,
        logits: wp.array,
        targets: wp.array,
        *,
        loss_scale: float = 1.0,
        reduction: str = "mean",
    ) -> wp.array:
        self._inputs(logits, targets)
        normalize = self._reduction(reduction)
        if not math.isfinite(loss_scale) or loss_scale <= 0.0:
            raise ValueError("loss_scale must be positive and finite")
        gradient = logits if self.in_place else self.gradient
        wp.launch(
            self._kernels.backward,
            dim=(self.rows, self.classes),
            inputs=[
                logits,
                targets,
                self.logsumexp,
                self.maximum,
                self.valid_count,
                self.ignore_index,
                float(loss_scale),
                normalize,
            ],
            outputs=[gradient],
            device=self.device,
        )
        return gradient
