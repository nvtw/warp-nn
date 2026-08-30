# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Reusable fixed-shape encoder-transformer inference plans.

The implementation is intentionally model-neutral: Kimodo's motion denoiser and
LLM2Vec both need bidirectional, padding-aware encoder attention.  Dense work is
delegated to the normal warp-nn Linear planner, while the small elementwise and
attention pieces remain allocation-free and CUDA-graph safe.
"""

from functools import lru_cache
import math

import warp as wp

from .operators import Operation, execute_operations, plan_linear


_ENCODER_DTYPES = (wp.float16, wp.bfloat16, wp.float32)


@lru_cache(maxsize=None)
def _encoder_kernels(dtype: type, head_size: int):
    DTYPE = dtype
    HEAD_SIZE = head_size

    @wp.kernel(enable_backward=False, module="unique")
    def add_bias(x: wp.array2d(dtype=DTYPE), bias: wp.array1d(dtype=DTYPE)):
        row, column = wp.tid()
        x[row, column] = DTYPE(wp.float32(x[row, column]) + wp.float32(bias[column]))

    @wp.kernel(enable_backward=False, module="unique")
    def bias_gelu(x: wp.array2d(dtype=DTYPE), bias: wp.array1d(dtype=DTYPE)):
        row, column = wp.tid()
        value = wp.float32(x[row, column]) + wp.float32(bias[column])
        # Exact PyTorch GELU default (erf), rather than the tanh approximation.
        value *= wp.float32(0.5) * (
            wp.float32(1.0) + wp.erf(value * wp.float32(0.7071067811865476))
        )
        x[row, column] = DTYPE(value)

    @wp.kernel(enable_backward=False, module="unique")
    def residual_layer_norm(
        branch: wp.array2d(dtype=DTYPE),
        residual: wp.array2d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
        scale: wp.array1d(dtype=DTYPE),
        shift: wp.array1d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
        epsilon: wp.float32,
    ):
        row = wp.tid()
        width = branch.shape[1]
        mean = wp.float32(0.0)
        for column in range(width):
            mean += (
                wp.float32(branch[row, column])
                + wp.float32(bias[column])
                + wp.float32(residual[row, column])
            )
        mean /= wp.float32(width)
        variance = wp.float32(0.0)
        for column in range(width):
            value = (
                wp.float32(branch[row, column])
                + wp.float32(bias[column])
                + wp.float32(residual[row, column])
                - mean
            )
            variance += value * value
        inverse = wp.float32(1.0) / wp.sqrt(variance / wp.float32(width) + epsilon)
        for column in range(width):
            value = (
                wp.float32(branch[row, column])
                + wp.float32(bias[column])
                + wp.float32(residual[row, column])
            )
            output[row, column] = DTYPE(
                (value - mean) * inverse * wp.float32(scale[column])
                + wp.float32(shift[column])
            )

    @wp.kernel(enable_backward=False, module="unique")
    def split_qkv(
        packed: wp.array2d(dtype=DTYPE),
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
    ):
        batch, head, token, column = wp.tid()
        hidden = query.shape[1] * query.shape[3]
        row = batch * query.shape[2] + token
        offset = head * query.shape[3] + column
        query[batch, head, token, column] = packed[row, offset]
        key[batch, head, token, column] = packed[row, hidden + offset]
        value[batch, head, token, column] = packed[row, hidden * 2 + offset]

    @wp.kernel(enable_backward=False, module="unique")
    def merge_heads(x: wp.array4d(dtype=DTYPE), output: wp.array2d(dtype=DTYPE)):
        batch, head, token, column = wp.tid()
        output[batch * x.shape[2] + token, head * x.shape[3] + column] = x[
            batch, head, token, column
        ]

    @wp.func
    def dot(left: DTYPE, right: DTYPE):
        return wp.float32(left) * wp.float32(right)

    @wp.func
    def update(
        total: wp.float32,
        current: DTYPE,
        old_scale: wp.float32,
        probability: wp.float32,
    ):
        return total * old_scale + wp.float32(current) * probability

    @wp.func
    def normalize(total: wp.float32, denominator: wp.float32):
        return total / denominator

    @wp.kernel(enable_backward=False, module="unique")
    def full_attention(
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
        valid: wp.array2d(dtype=wp.bool),
        output: wp.array4d(dtype=DTYPE),
        scale: wp.float32,
    ):
        item = wp.tid()
        sequence = query.shape[2]
        token = item % sequence
        head = (item / sequence) % query.shape[1]
        batch = item / (query.shape[1] * sequence)
        kv_head = head / (query.shape[1] / key.shape[1])
        accumulator = wp.tile_zeros(shape=(HEAD_SIZE,), dtype=wp.float32)
        q = wp.tile_load(query[batch, head, token], shape=(HEAD_SIZE,))
        maximum = wp.float32(-3.402823466e38)
        denominator = wp.float32(0.0)
        for source in range(sequence):
            if valid[batch, source]:
                k = wp.tile_load(key[batch, kv_head, source], shape=(HEAD_SIZE,))
                score = wp.tile_extract(wp.tile_sum(wp.tile_map(dot, q, k)), 0) * scale
                new_maximum = wp.max(maximum, score)
                old_scale = wp.exp(maximum - new_maximum)
                probability = wp.exp(score - new_maximum)
                denominator = denominator * old_scale + probability
                v = wp.tile_load(value[batch, kv_head, source], shape=(HEAD_SIZE,))
                accumulator = wp.tile_map(
                    update, accumulator, v, old_scale, probability
                )
                maximum = new_maximum
        accumulator = wp.tile_map(normalize, accumulator, denominator)
        wp.tile_store(
            output[batch, head, token], wp.tile_astype(accumulator, dtype=DTYPE)
        )

    return (
        add_bias,
        bias_gelu,
        residual_layer_norm,
        split_qkv,
        merge_heads,
        full_attention,
    )


class EncoderLayerPlan:
    """One fixed-shape PyTorch-compatible post-norm TransformerEncoderLayer."""

    def __init__(
        self,
        x: wp.array,
        valid: wp.array,
        weights: dict[str, wp.array],
        prefix: str,
        heads: int,
        *,
        epsilon: float = 1.0e-5,
        cublas=None,
    ):
        if x.ndim != 3 or valid.shape != x.shape[:2]:
            raise ValueError(
                "encoder input must be [batch, sequence, hidden] with a matching mask"
            )
        if x.dtype not in _ENCODER_DTYPES or valid.dtype != wp.bool:
            raise TypeError("encoder requires FP16/BF16/FP32 input and a boolean mask")
        batch, sequence, hidden = x.shape
        if heads <= 0 or hidden % heads:
            raise ValueError("hidden size must be divisible by the positive head count")
        self.device = x.device
        self.dtype = x.dtype
        self.batch, self.sequence, self.hidden = batch, sequence, hidden
        self.heads, self.head_size = heads, hidden // heads
        self.valid = valid
        self.input = x
        self.output = wp.empty_like(x)
        self._attention_heads = wp.empty(
            (batch, heads, sequence, self.head_size), dtype=x.dtype, device=x.device
        )
        self._query = wp.empty_like(self._attention_heads)
        self._key = wp.empty_like(self._attention_heads)
        self._value = wp.empty_like(self._attention_heads)
        self._attention_flat = wp.empty(
            (batch * sequence, hidden), dtype=x.dtype, device=x.device
        )
        self._norm1 = wp.empty(
            (batch * sequence, hidden), dtype=x.dtype, device=x.device
        )
        self._norm2 = self.output.reshape((batch * sequence, hidden))
        self._epsilon = float(epsilon)
        self._weights = weights
        self._prefix = prefix
        self._tensors = {"x": x.reshape((batch * sequence, hidden))}
        self._shapes = {"x": (batch * sequence, hidden)}
        self._tensors["attention_flat"] = self._attention_flat
        self._shapes["attention_flat"] = self._attention_flat.shape
        self._tensors["norm1"] = self._norm1
        self._shapes["norm1"] = self._norm1.shape
        self._ops = []

        def linear(name, source, weight):
            op = Operation("Linear", [source, weight], [name])
            self._tensors[weight] = weights[weight]
            self._shapes[weight] = weights[weight].shape
            plan_linear(op, self._tensors, self._shapes, self.device, cublas)
            self._ops.append(op)
            return op

        p = prefix
        self._qkv = linear("qkv", "x", f"{p}.self_attn.in_proj_weight")
        self._out = linear(
            "attention_projection", "attention_flat", f"{p}.self_attn.out_proj.weight"
        )
        self._ff1 = linear("ff1", "norm1", f"{p}.linear1.weight")
        self._ff2 = linear("ff2", "ff1", f"{p}.linear2.weight")
        kernels = _encoder_kernels(x.dtype, self.head_size)
        (
            self._add_bias,
            self._bias_gelu,
            self._residual_norm,
            self._split,
            self._merge,
            self._attention,
        ) = kernels

    def _execute(self, op):
        execute_operations([op], self._tensors, self._shapes, self.device)

    def execute(self):
        p = self._prefix
        self._execute(self._qkv)
        qkv = self._tensors["qkv"]
        wp.launch(
            self._add_bias,
            dim=qkv.shape,
            inputs=[qkv, self._weights[f"{p}.self_attn.in_proj_bias"]],
            device=self.device,
        )
        wp.launch(
            self._split,
            dim=self._query.shape,
            inputs=[qkv, self._query, self._key, self._value],
            device=self.device,
        )
        wp.launch_tiled(
            self._attention,
            dim=self.batch * self.heads * self.sequence,
            inputs=[
                self._query,
                self._key,
                self._value,
                self.valid,
                self._attention_heads,
                wp.float32(1.0 / math.sqrt(self.head_size)),
            ],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            self._merge,
            dim=self._attention_heads.shape,
            inputs=[self._attention_heads, self._attention_flat],
            device=self.device,
        )
        self._execute(self._out)
        wp.launch(
            self._residual_norm,
            dim=self.batch * self.sequence,
            inputs=[
                self._tensors["attention_projection"],
                self._tensors["x"],
                self._weights[f"{p}.self_attn.out_proj.bias"],
                self._weights[f"{p}.norm1.weight"],
                self._weights[f"{p}.norm1.bias"],
                self._norm1,
                wp.float32(self._epsilon),
            ],
            device=self.device,
        )
        self._execute(self._ff1)
        wp.launch(
            self._bias_gelu,
            dim=self._tensors["ff1"].shape,
            inputs=[self._tensors["ff1"], self._weights[f"{p}.linear1.bias"]],
            device=self.device,
        )
        self._execute(self._ff2)
        wp.launch(
            self._residual_norm,
            dim=self.batch * self.sequence,
            inputs=[
                self._tensors["ff2"],
                self._norm1,
                self._weights[f"{p}.linear2.bias"],
                self._weights[f"{p}.norm2.weight"],
                self._weights[f"{p}.norm2.bias"],
                self._norm2,
                wp.float32(self._epsilon),
            ],
            device=self.device,
        )
        return self.output


class EncoderStackPlan:
    """A fixed-buffer stack of post-norm encoder layers."""

    def __init__(
        self, x, valid, weights, prefix, layers, heads, *, epsilon=1.0e-5, cublas=None
    ):
        if layers <= 0:
            raise ValueError("encoder stack requires at least one layer")
        self.layers = []
        current = x
        for index in range(layers):
            layer = EncoderLayerPlan(
                current,
                valid,
                weights,
                f"{prefix}.layers.{index}",
                heads,
                epsilon=epsilon,
                cublas=cublas,
            )
            self.layers.append(layer)
            current = layer.output
        self.output = current

    def execute(self):
        for layer in self.layers:
            layer.execute()
        return self.output
