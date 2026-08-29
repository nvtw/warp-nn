# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable reference primitives with fixed, reusable storage.

These plans deliberately cover the inexpensive nonlinear and indexing pieces of
transformer training.  Matrix products remain the responsibility of the caller.
The plans are shape-fixed so repeated steps do not allocate temporary arrays.
"""

from dataclasses import dataclass
from functools import lru_cache

import warp as wp


_STORAGE_DTYPES = (wp.float32, wp.float16, wp.bfloat16)


@dataclass(frozen=True)
class _PrimitiveKernels:
    residual: object
    scale: object
    swiglu: object
    sigmoid_gate: object
    rope: object
    mean_square: object
    rms_norm: object
    embedding: object


@lru_cache(maxsize=None)
def _get_primitive_kernels(dtype: type) -> _PrimitiveKernels:
    if dtype not in _STORAGE_DTYPES:
        raise TypeError("training primitives support FP32, FP16, and BF16 storage")
    DTYPE = dtype

    @wp.func
    def stable_sigmoid(value: wp.float32):
        if value >= wp.float32(0.0):
            return wp.float32(1.0) / (wp.float32(1.0) + wp.exp(-value))
        exponential = wp.exp(value)
        return exponential / (wp.float32(1.0) + exponential)

    @wp.kernel(module="unique")
    def residual(
        x: wp.array2d(dtype=DTYPE),
        skip: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        output[row, column] = DTYPE(
            wp.float32(x[row, column]) + wp.float32(skip[row, column])
        )

    @wp.kernel(module="unique")
    def scale(
        x: wp.array2d(dtype=DTYPE),
        multiplier: wp.array1d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        output[row, column] = DTYPE(
            wp.float32(x[row, column]) * wp.float32(multiplier[column])
        )

    @wp.kernel(module="unique")
    def swiglu(
        gate: wp.array2d(dtype=DTYPE),
        up: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        gate_value = wp.float32(gate[row, column])
        output[row, column] = DTYPE(
            gate_value * stable_sigmoid(gate_value) * wp.float32(up[row, column])
        )

    @wp.kernel(module="unique")
    def sigmoid_gate(
        x: wp.array2d(dtype=DTYPE),
        gate: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        output[row, column] = DTYPE(
            wp.float32(x[row, column]) * stable_sigmoid(wp.float32(gate[row, column]))
        )

    @wp.kernel(module="unique")
    def rope(
        x: wp.array2d(dtype=DTYPE),
        cosine: wp.array2d(dtype=DTYPE),
        sine: wp.array2d(dtype=DTYPE),
        rotary_dim: int,
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        if column < rotary_dim:
            half = rotary_dim // 2
            frequency = column % half
            paired_column = column + half
            sign = wp.float32(-1.0)
            if column >= half:
                paired_column = column - half
                sign = wp.float32(1.0)
            output[row, column] = DTYPE(
                wp.float32(x[row, column]) * wp.float32(cosine[row, frequency])
                + sign
                * wp.float32(x[row, paired_column])
                * wp.float32(sine[row, frequency])
            )
        else:
            output[row, column] = x[row, column]

    @wp.kernel(module="unique")
    def mean_square(x: wp.array2d(dtype=DTYPE), output: wp.array1d(dtype=wp.float32)):
        row, column = wp.tid()
        value = wp.float32(x[row, column])
        wp.atomic_add(output, row, value * value / wp.float32(x.shape[1]))

    @wp.kernel(module="unique")
    def rms_norm(
        x: wp.array2d(dtype=DTYPE),
        inverse_rms: wp.array1d(dtype=wp.float32),
        weight: wp.array1d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        output[row, column] = DTYPE(
            wp.float32(x[row, column]) * inverse_rms[row] * wp.float32(weight[column])
        )

    @wp.kernel(module="unique")
    def embedding(
        table: wp.array2d(dtype=DTYPE),
        token_ids: wp.array1d(dtype=wp.int32),
        output: wp.array2d(dtype=DTYPE),
    ):
        token, column = wp.tid()
        output[token, column] = table[token_ids[token], column]

    return _PrimitiveKernels(
        residual, scale, swiglu, sigmoid_gate, rope, mean_square, rms_norm, embedding
    )


@wp.kernel
def _inverse_rms_kernel(
    mean_square: wp.array1d(dtype=wp.float32),
    epsilon: wp.float32,
    inverse_rms: wp.array1d(dtype=wp.float32),
):
    row = wp.tid()
    inverse_rms[row] = wp.float32(1.0) / wp.sqrt(mean_square[row] + epsilon)


@wp.kernel
def _cross_entropy_forward_kernel(
    logits: wp.array2d(dtype=wp.float32),
    targets: wp.array1d(dtype=wp.int32),
    ignore_index: int,
    logsumexp: wp.array1d(dtype=wp.float32),
    maximum: wp.array1d(dtype=wp.float32),
    losses: wp.array1d(dtype=wp.float32),
    valid: wp.array1d(dtype=wp.int32),
):
    row = wp.tid()
    target = targets[row]
    if target == ignore_index:
        logsumexp[row] = wp.float32(0.0)
        maximum[row] = wp.float32(0.0)
        losses[row] = wp.float32(0.0)
        valid[row] = 0
        return

    row_maximum = logits[row, 0]
    for column in range(1, logits.shape[1]):
        row_maximum = wp.max(row_maximum, logits[row, column])
    exponential_sum = wp.float32(0.0)
    for column in range(logits.shape[1]):
        exponential_sum += wp.exp(logits[row, column] - row_maximum)
    shifted_logsumexp = wp.log(exponential_sum)
    logsumexp[row] = shifted_logsumexp
    maximum[row] = row_maximum
    losses[row] = row_maximum - logits[row, target] + shifted_logsumexp
    valid[row] = 1


@wp.kernel
def _cross_entropy_reduce_kernel(
    losses: wp.array1d(dtype=wp.float32),
    valid: wp.array1d(dtype=wp.int32),
    normalize: bool,
    loss: wp.array1d(dtype=wp.float32),
    valid_count: wp.array1d(dtype=wp.int32),
):
    loss_sum = wp.float32(0.0)
    count = int(0)
    for row in range(losses.shape[0]):
        loss_sum += losses[row]
        count += valid[row]
    valid_count[0] = count
    if count > 0 and normalize:
        loss[0] = loss_sum / wp.float32(count)
    else:
        loss[0] = loss_sum


@wp.kernel
def _cross_entropy_backward_kernel(
    logits: wp.array2d(dtype=wp.float32),
    targets: wp.array1d(dtype=wp.int32),
    logsumexp: wp.array1d(dtype=wp.float32),
    maximum: wp.array1d(dtype=wp.float32),
    valid_count: wp.array1d(dtype=wp.int32),
    ignore_index: int,
    loss_scale: wp.float32,
    normalize: bool,
    gradient: wp.array2d(dtype=wp.float32),
):
    row, column = wp.tid()
    target = targets[row]
    count = valid_count[0]
    if target == ignore_index or count == 0:
        gradient[row, column] = wp.float32(0.0)
        return
    value = wp.exp(logits[row, column] - maximum[row] - logsumexp[row])
    if column == target:
        value -= wp.float32(1.0)
    divisor = wp.float32(count) if normalize else wp.float32(1.0)
    gradient[row, column] = value * loss_scale / divisor


class _FixedPlan:
    def __init__(self, device: object | None):
        self.device = wp.get_device(device)

    def _check(
        self, array: wp.array, shape: tuple[int, ...], dtype: type, name: str
    ) -> None:
        if tuple(array.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(array.shape)}"
            )
        if array.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
        if array.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {array.device}")


class TransformerPrimitivePlan(_FixedPlan):
    """Fixed-shape differentiable transformer elementwise primitives.

    RoPE uses the common split-half layout. Each operation owns an output array,
    allowing different operations to coexist on one autodiff tape. A particular
    operation should be launched at most once on a tape because its output is
    intentionally reused across training steps. RMS statistics always use FP32.
    Warp Tape gradients otherwise follow the primal storage dtype.
    """

    def __init__(
        self,
        rows: int,
        width: int,
        *,
        dtype: type = wp.float32,
        rotary_dim: int | None = None,
        epsilon: float = 1.0e-6,
        device: object | None = None,
    ):
        super().__init__(device)
        if dtype not in _STORAGE_DTYPES:
            raise TypeError("dtype must be wp.float32, wp.float16, or wp.bfloat16")
        if rows <= 0 or width <= 0:
            raise ValueError("rows and width must be positive")
        if rotary_dim is None:
            rotary_dim = width
        if rotary_dim <= 0 or rotary_dim > width or rotary_dim % 2:
            raise ValueError(
                "rotary_dim must be positive, even, and no larger than width"
            )
        self.shape = (rows, width)
        self.rows = rows
        self.width = width
        self.dtype = dtype
        self._kernels = _get_primitive_kernels(dtype)
        self.rotary_dim = rotary_dim
        self.epsilon = float(epsilon)
        self._mean_square = wp.empty(
            rows, dtype=wp.float32, device=self.device, requires_grad=True
        )
        self._inverse_rms = wp.empty(
            rows, dtype=wp.float32, device=self.device, requires_grad=True
        )
        self._outputs = {
            name: wp.empty(
                self.shape, dtype=dtype, device=self.device, requires_grad=True
            )
            for name in (
                "residual",
                "scale",
                "swiglu",
                "sigmoid_gate",
                "rope",
                "rms_norm",
            )
        }

    def _matrix(self, array: wp.array, name: str) -> None:
        self._check(array, self.shape, self.dtype, name)

    def residual(self, x: wp.array, residual: wp.array) -> wp.array:
        self._matrix(x, "x")
        self._matrix(residual, "residual")
        output = self._outputs["residual"]
        wp.launch(
            self._kernels.residual,
            dim=self.shape,
            inputs=[x, residual],
            outputs=[output],
            device=self.device,
        )
        return output

    def scale(self, x: wp.array, scale: wp.array) -> wp.array:
        self._matrix(x, "x")
        self._check(scale, (self.width,), self.dtype, "scale")
        output = self._outputs["scale"]
        wp.launch(
            self._kernels.scale,
            dim=self.shape,
            inputs=[x, scale],
            outputs=[output],
            device=self.device,
        )
        return output

    def swiglu(self, gate: wp.array, up: wp.array) -> wp.array:
        self._matrix(gate, "gate")
        self._matrix(up, "up")
        output = self._outputs["swiglu"]
        wp.launch(
            self._kernels.swiglu,
            dim=self.shape,
            inputs=[gate, up],
            outputs=[output],
            device=self.device,
        )
        return output

    def sigmoid_gate(self, x: wp.array, gate: wp.array) -> wp.array:
        self._matrix(x, "x")
        self._matrix(gate, "gate")
        output = self._outputs["sigmoid_gate"]
        wp.launch(
            self._kernels.sigmoid_gate,
            dim=self.shape,
            inputs=[x, gate],
            outputs=[output],
            device=self.device,
        )
        return output

    def rope(self, x: wp.array, cosine: wp.array, sine: wp.array) -> wp.array:
        self._matrix(x, "x")
        frequency_shape = (self.rows, self.rotary_dim // 2)
        self._check(cosine, frequency_shape, self.dtype, "cosine")
        self._check(sine, frequency_shape, self.dtype, "sine")
        output = self._outputs["rope"]
        wp.launch(
            self._kernels.rope,
            dim=self.shape,
            inputs=[x, cosine, sine, self.rotary_dim],
            outputs=[output],
            device=self.device,
        )
        return output

    def rms_norm(self, x: wp.array, weight: wp.array) -> wp.array:
        self._matrix(x, "x")
        self._check(weight, (self.width,), self.dtype, "weight")
        output = self._outputs["rms_norm"]
        self._mean_square.zero_()
        wp.launch(
            self._kernels.mean_square,
            dim=self.shape,
            inputs=[x],
            outputs=[self._mean_square],
            device=self.device,
        )
        wp.launch(
            _inverse_rms_kernel,
            dim=self.rows,
            inputs=[self._mean_square, self.epsilon],
            outputs=[self._inverse_rms],
            device=self.device,
        )
        wp.launch(
            self._kernels.rms_norm,
            dim=self.shape,
            inputs=[x, self._inverse_rms, weight],
            outputs=[output],
            device=self.device,
        )
        return output


class EmbeddingPlan(_FixedPlan):
    """Fixed-shape differentiable embedding gather.

    Warp Tape stores its table gradient in the table storage dtype.
    """

    def __init__(
        self,
        token_count: int,
        vocabulary_size: int,
        width: int,
        *,
        dtype: type = wp.float32,
        device: object | None = None,
    ):
        super().__init__(device)
        if dtype not in _STORAGE_DTYPES:
            raise TypeError("dtype must be wp.float32, wp.float16, or wp.bfloat16")
        if token_count <= 0 or vocabulary_size <= 0 or width <= 0:
            raise ValueError("token_count, vocabulary_size, and width must be positive")
        self.token_count = token_count
        self.vocabulary_size = vocabulary_size
        self.width = width
        self.dtype = dtype
        self._kernel = _get_primitive_kernels(dtype).embedding
        self.output = wp.empty(
            (token_count, width),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )

    def __call__(self, table: wp.array, token_ids: wp.array) -> wp.array:
        self._check(table, (self.vocabulary_size, self.width), self.dtype, "table")
        self._check(token_ids, (self.token_count,), wp.int32, "token_ids")
        wp.launch(
            self._kernel,
            dim=self.output.shape,
            inputs=[table, token_ids],
            outputs=[self.output],
            device=self.device,
        )
        return self.output


class CrossEntropyPlan(_FixedPlan):
    """Stable mean cross-entropy with explicit preallocated backward storage.

    ``backward`` uses the log-sum-exp values produced by the latest ``forward``
    call. Targets must contain class indices or ``ignore_index``. Mean reduction
    remains the default; sum reduction supports exact cross-microbatch token
    normalization by :class:`warp_nn.training.optimizer.AdamWPlan`.
    """

    def __init__(
        self,
        rows: int,
        classes: int,
        *,
        ignore_index: int = -100,
        device: object | None = None,
    ):
        super().__init__(device)
        if rows <= 0 or classes <= 0:
            raise ValueError("rows and classes must be positive")
        self.rows = rows
        self.classes = classes
        self.ignore_index = ignore_index
        self.logsumexp = wp.empty(rows, dtype=wp.float32, device=self.device)
        self.losses = wp.empty(rows, dtype=wp.float32, device=self.device)
        self.valid = wp.empty(rows, dtype=wp.int32, device=self.device)
        self.loss = wp.empty(1, dtype=wp.float32, device=self.device)
        self.maximum = wp.empty(rows, dtype=wp.float32, device=self.device)
        self.valid_count = wp.empty(1, dtype=wp.int32, device=self.device)
        self.gradient = wp.empty((rows, classes), dtype=wp.float32, device=self.device)

    def _inputs(self, logits: wp.array, targets: wp.array) -> None:
        self._check(logits, (self.rows, self.classes), wp.float32, "logits")
        self._check(targets, (self.rows,), wp.int32, "targets")

    def forward(
        self, logits: wp.array, targets: wp.array, *, reduction: str = "mean"
    ) -> wp.array:
        self._inputs(logits, targets)
        if reduction not in ("mean", "sum"):
            raise ValueError("cross-entropy reduction must be mean or sum")
        wp.launch(
            _cross_entropy_forward_kernel,
            dim=self.rows,
            inputs=[logits, targets, self.ignore_index],
            outputs=[self.logsumexp, self.maximum, self.losses, self.valid],
            device=self.device,
        )
        wp.launch(
            _cross_entropy_reduce_kernel,
            dim=1,
            inputs=[self.losses, self.valid, reduction == "mean"],
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
        if reduction not in ("mean", "sum"):
            raise ValueError("cross-entropy reduction must be mean or sum")
        wp.launch(
            _cross_entropy_backward_kernel,
            dim=(self.rows, self.classes),
            inputs=[
                logits,
                targets,
                self.logsumexp,
                self.maximum,
                self.valid_count,
                self.ignore_index,
                loss_scale,
                reduction == "mean",
            ],
            outputs=[self.gradient],
            device=self.device,
        )
        return self.gradient
