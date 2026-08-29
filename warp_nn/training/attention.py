# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Functional causal grouped-query attention for training.

The routines in this module deliberately use caller-owned arrays.  Forward saves
only a log-sum-exp value per query row; backward reconstructs probabilities from
that bounded state instead of retaining a quadratic attention matrix.
"""

from functools import lru_cache
import math

import warp as wp


_SUPPORTED_DTYPES = (wp.float16, wp.bfloat16)


@lru_cache(maxsize=None)
def _attention_kernels(dtype: type, head_size: int | None = None):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def logsumexp(
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        lengths: wp.array1d[wp.int32],
        lse: wp.array3d[wp.float32],
        scale: wp.float32,
        window: wp.int32,
    ):
        item = wp.tid()
        sequence = query.shape[2]
        query_head = (item / sequence) % query.shape[1]
        batch = item / (query.shape[1] * sequence)
        query_token = item % sequence
        length = wp.clamp(lengths[batch], 0, sequence)
        if query_token >= length:
            lse[batch, query_head, query_token] = wp.float32(-3.402823466e38)
            return

        kv_head = query_head / (query.shape[1] / key.shape[1])
        first_key = wp.max(0, query_token + 1 - window) if window > 0 else 0
        maximum = wp.float32(-3.402823466e38)
        denominator = wp.float32(0.0)
        for key_token in range(first_key, query_token + 1):
            score = wp.float32(0.0)
            for column in range(query.shape[3]):
                score += wp.float32(
                    query[batch, query_head, query_token, column]
                ) * wp.float32(key[batch, kv_head, key_token, column])
            score *= scale
            new_maximum = wp.max(maximum, score)
            denominator = denominator * wp.exp(maximum - new_maximum) + wp.exp(
                score - new_maximum
            )
            maximum = new_maximum
        lse[batch, query_head, query_token] = maximum + wp.log(denominator)

    @wp.kernel(enable_backward=False, module="unique")
    def accumulate_output(
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
        lengths: wp.array1d[wp.int32],
        lse: wp.array3d[wp.float32],
        accumulator: wp.array4d[wp.float32],
        scale: wp.float32,
        window: wp.int32,
    ):
        item = wp.tid()
        sequence = query.shape[2]
        key_token = item % sequence
        query_token = (item / sequence) % sequence
        query_head = (item / (sequence * sequence)) % query.shape[1]
        batch = item / (query.shape[1] * sequence * sequence)
        length = wp.clamp(lengths[batch], 0, sequence)
        first_key = wp.max(0, query_token + 1 - window) if window > 0 else 0
        if query_token >= length or key_token < first_key or key_token > query_token:
            return

        kv_head = query_head / (query.shape[1] / key.shape[1])
        score = wp.float32(0.0)
        for column in range(query.shape[3]):
            score += wp.float32(
                query[batch, query_head, query_token, column]
            ) * wp.float32(key[batch, kv_head, key_token, column])
        probability = wp.exp(score * scale - lse[batch, query_head, query_token])
        for column in range(query.shape[3]):
            wp.atomic_add(
                accumulator,
                batch,
                query_head,
                query_token,
                column,
                probability * wp.float32(value[batch, kv_head, key_token, column]),
            )

    @wp.kernel(enable_backward=False, module="unique")
    def cast_output(
        accumulator: wp.array4d[wp.float32], output: wp.array4d(dtype=DTYPE)
    ):
        batch, head, token, column = wp.tid()
        output[batch, head, token, column] = DTYPE(
            accumulator[batch, head, token, column]
        )

    @wp.kernel(enable_backward=False, module="unique")
    def softmax_delta(
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
        output_grad: wp.array4d(dtype=DTYPE),
        lengths: wp.array1d[wp.int32],
        lse: wp.array3d[wp.float32],
        delta: wp.array3d[wp.float32],
        scale: wp.float32,
        window: wp.int32,
    ):
        item = wp.tid()
        sequence = query.shape[2]
        query_head = (item / sequence) % query.shape[1]
        batch = item / (query.shape[1] * sequence)
        query_token = item % sequence
        length = wp.clamp(lengths[batch], 0, sequence)
        if query_token >= length:
            delta[batch, query_head, query_token] = wp.float32(0.0)
            return

        kv_head = query_head / (query.shape[1] / key.shape[1])
        first_key = wp.max(0, query_token + 1 - window) if window > 0 else 0
        total = wp.float32(0.0)
        for key_token in range(first_key, query_token + 1):
            score = wp.float32(0.0)
            probability_grad = wp.float32(0.0)
            for column in range(query.shape[3]):
                score += wp.float32(
                    query[batch, query_head, query_token, column]
                ) * wp.float32(key[batch, kv_head, key_token, column])
                probability_grad += wp.float32(
                    output_grad[batch, query_head, query_token, column]
                ) * wp.float32(value[batch, kv_head, key_token, column])
            probability = wp.exp(score * scale - lse[batch, query_head, query_token])
            total += probability * probability_grad
        delta[batch, query_head, query_token] = total

    @wp.kernel(enable_backward=False, module="unique")
    def accumulate_gradients(
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
        output_grad: wp.array4d(dtype=DTYPE),
        lengths: wp.array1d[wp.int32],
        lse: wp.array3d[wp.float32],
        delta: wp.array3d[wp.float32],
        query_grad: wp.array4d[wp.float32],
        key_grad: wp.array4d[wp.float32],
        value_grad: wp.array4d[wp.float32],
        scale: wp.float32,
        window: wp.int32,
    ):
        item = wp.tid()
        sequence = query.shape[2]
        key_token = item % sequence
        query_token = (item / sequence) % sequence
        query_head = (item / (sequence * sequence)) % query.shape[1]
        batch = item / (query.shape[1] * sequence * sequence)
        length = wp.clamp(lengths[batch], 0, sequence)
        first_key = wp.max(0, query_token + 1 - window) if window > 0 else 0
        if query_token >= length or key_token < first_key or key_token > query_token:
            return

        kv_head = query_head / (query.shape[1] / key.shape[1])
        score = wp.float32(0.0)
        probability_grad = wp.float32(0.0)
        for column in range(query.shape[3]):
            score += wp.float32(
                query[batch, query_head, query_token, column]
            ) * wp.float32(key[batch, kv_head, key_token, column])
            probability_grad += wp.float32(
                output_grad[batch, query_head, query_token, column]
            ) * wp.float32(value[batch, kv_head, key_token, column])
        probability = wp.exp(score * scale - lse[batch, query_head, query_token])
        score_grad = (
            probability
            * (probability_grad - delta[batch, query_head, query_token])
            * scale
        )
        for column in range(query.shape[3]):
            output_gradient = wp.float32(
                output_grad[batch, query_head, query_token, column]
            )
            wp.atomic_add(
                query_grad,
                batch,
                query_head,
                query_token,
                column,
                score_grad * wp.float32(key[batch, kv_head, key_token, column]),
            )
            wp.atomic_add(
                key_grad,
                batch,
                kv_head,
                key_token,
                column,
                score_grad * wp.float32(query[batch, query_head, query_token, column]),
            )
            wp.atomic_add(
                value_grad,
                batch,
                kv_head,
                key_token,
                column,
                probability * output_gradient,
            )

    if head_size is None:
        return (
            logsumexp,
            accumulate_output,
            cast_output,
            softmax_delta,
            accumulate_gradients,
        )

    HEAD_SIZE = head_size

    @wp.func
    def dot(left: DTYPE, right: DTYPE):
        return wp.float32(left) * wp.float32(right)

    @wp.func
    def update_accumulator(
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
    def streaming_forward(
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
        lengths: wp.array1d[wp.int32],
        output: wp.array4d(dtype=DTYPE),
        lse: wp.array3d[wp.float32],
        workspace: wp.array4d[wp.float32],
        scale: wp.float32,
        window: wp.int32,
    ):
        item = wp.tid()
        sequence = query.shape[2]
        query_token = item % sequence
        query_head = (item / sequence) % query.shape[1]
        batch = item / (query.shape[1] * sequence)
        length = wp.clamp(lengths[batch], 0, sequence)
        accumulator = wp.tile_zeros(shape=(HEAD_SIZE,), dtype=wp.float32)
        if query_token >= length:
            wp.tile_store(workspace[batch, query_head, query_token], accumulator)
            wp.tile_store(
                output[batch, query_head, query_token],
                wp.tile_astype(accumulator, dtype=DTYPE),
            )
            lse[batch, query_head, query_token] = wp.float32(-3.402823466e38)
            return

        kv_head = query_head / (query.shape[1] / key.shape[1])
        query_values = wp.tile_load(
            query[batch, query_head, query_token], shape=(HEAD_SIZE,)
        )
        first_key = wp.max(0, query_token + 1 - window) if window > 0 else 0
        maximum = wp.float32(-3.402823466e38)
        denominator = wp.float32(0.0)
        for key_token in range(first_key, query_token + 1):
            key_values = wp.tile_load(
                key[batch, kv_head, key_token], shape=(HEAD_SIZE,)
            )
            score = wp.tile_extract(
                wp.tile_sum(wp.tile_map(dot, query_values, key_values)), 0
            )
            score *= scale
            new_maximum = wp.max(maximum, score)
            old_scale = wp.exp(maximum - new_maximum)
            probability = wp.exp(score - new_maximum)
            denominator = denominator * old_scale + probability
            value_values = wp.tile_load(
                value[batch, kv_head, key_token], shape=(HEAD_SIZE,)
            )
            accumulator = wp.tile_map(
                update_accumulator,
                accumulator,
                value_values,
                old_scale,
                probability,
            )
            maximum = new_maximum
        normalized = wp.tile_map(normalize, accumulator, denominator)
        wp.tile_store(workspace[batch, query_head, query_token], normalized)
        wp.tile_store(
            output[batch, query_head, query_token],
            wp.tile_astype(normalized, dtype=DTYPE),
        )
        lse[batch, query_head, query_token] = maximum + wp.log(denominator)

    @wp.func
    def add_scaled(total: wp.float32, current: DTYPE, multiplier: wp.float32):
        return total + wp.float32(current) * multiplier

    @wp.kernel(enable_backward=False, module="unique")
    def streaming_delta_query_grad(
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
        output_grad: wp.array4d(dtype=DTYPE),
        lengths: wp.array1d[wp.int32],
        lse: wp.array3d[wp.float32],
        delta: wp.array3d[wp.float32],
        query_grad: wp.array4d[wp.float32],
        scale: wp.float32,
        window: wp.int32,
        accumulate: bool,
    ):
        item = wp.tid()
        sequence = query.shape[2]
        query_token = item % sequence
        query_head = (item / sequence) % query.shape[1]
        batch = item / (query.shape[1] * sequence)
        length = wp.clamp(lengths[batch], 0, sequence)
        query_accumulator = wp.tile_zeros(shape=(HEAD_SIZE,), dtype=wp.float32)
        if accumulate:
            query_accumulator = wp.tile_load(
                query_grad[batch, query_head, query_token], shape=(HEAD_SIZE,)
            )
        if query_token >= length:
            delta[batch, query_head, query_token] = wp.float32(0.0)
            wp.tile_store(query_grad[batch, query_head, query_token], query_accumulator)
            return

        kv_head = query_head / (query.shape[1] / key.shape[1])
        query_values = wp.tile_load(
            query[batch, query_head, query_token], shape=(HEAD_SIZE,)
        )
        output_grad_values = wp.tile_load(
            output_grad[batch, query_head, query_token], shape=(HEAD_SIZE,)
        )
        first_key = wp.max(0, query_token + 1 - window) if window > 0 else 0
        softmax_delta_value = wp.float32(0.0)
        for key_token in range(first_key, query_token + 1):
            key_values = wp.tile_load(
                key[batch, kv_head, key_token], shape=(HEAD_SIZE,)
            )
            value_values = wp.tile_load(
                value[batch, kv_head, key_token], shape=(HEAD_SIZE,)
            )
            score = wp.tile_extract(
                wp.tile_sum(wp.tile_map(dot, query_values, key_values)), 0
            )
            probability_grad = wp.tile_extract(
                wp.tile_sum(wp.tile_map(dot, output_grad_values, value_values)), 0
            )
            probability = wp.exp(score * scale - lse[batch, query_head, query_token])
            softmax_delta_value += probability * probability_grad
        delta[batch, query_head, query_token] = softmax_delta_value

        for key_token in range(first_key, query_token + 1):
            key_values = wp.tile_load(
                key[batch, kv_head, key_token], shape=(HEAD_SIZE,)
            )
            value_values = wp.tile_load(
                value[batch, kv_head, key_token], shape=(HEAD_SIZE,)
            )
            score = wp.tile_extract(
                wp.tile_sum(wp.tile_map(dot, query_values, key_values)), 0
            )
            probability_grad = wp.tile_extract(
                wp.tile_sum(wp.tile_map(dot, output_grad_values, value_values)), 0
            )
            probability = wp.exp(score * scale - lse[batch, query_head, query_token])
            score_grad = probability * (probability_grad - softmax_delta_value) * scale
            query_accumulator = wp.tile_map(
                add_scaled, query_accumulator, key_values, score_grad
            )
        wp.tile_store(query_grad[batch, query_head, query_token], query_accumulator)

    @wp.kernel(enable_backward=False, module="unique")
    def streaming_key_value_grad(
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
        output_grad: wp.array4d(dtype=DTYPE),
        lengths: wp.array1d[wp.int32],
        lse: wp.array3d[wp.float32],
        delta: wp.array3d[wp.float32],
        key_grad: wp.array4d[wp.float32],
        value_grad: wp.array4d[wp.float32],
        scale: wp.float32,
        window: wp.int32,
        accumulate: bool,
    ):
        item = wp.tid()
        sequence = query.shape[2]
        key_token = item % sequence
        kv_head = (item / sequence) % key.shape[1]
        batch = item / (key.shape[1] * sequence)
        length = wp.clamp(lengths[batch], 0, sequence)
        key_accumulator = wp.tile_zeros(shape=(HEAD_SIZE,), dtype=wp.float32)
        value_accumulator = wp.tile_zeros(shape=(HEAD_SIZE,), dtype=wp.float32)
        if accumulate:
            key_accumulator = wp.tile_load(
                key_grad[batch, kv_head, key_token], shape=(HEAD_SIZE,)
            )
            value_accumulator = wp.tile_load(
                value_grad[batch, kv_head, key_token], shape=(HEAD_SIZE,)
            )
        if key_token >= length:
            wp.tile_store(key_grad[batch, kv_head, key_token], key_accumulator)
            wp.tile_store(value_grad[batch, kv_head, key_token], value_accumulator)
            return

        key_values = wp.tile_load(key[batch, kv_head, key_token], shape=(HEAD_SIZE,))
        value_values = wp.tile_load(
            value[batch, kv_head, key_token], shape=(HEAD_SIZE,)
        )
        heads_per_kv = query.shape[1] / key.shape[1]
        query_end = wp.min(length, key_token + window) if window > 0 else length
        for query_head in range(kv_head * heads_per_kv, (kv_head + 1) * heads_per_kv):
            for query_token in range(key_token, query_end):
                query_values = wp.tile_load(
                    query[batch, query_head, query_token], shape=(HEAD_SIZE,)
                )
                output_grad_values = wp.tile_load(
                    output_grad[batch, query_head, query_token], shape=(HEAD_SIZE,)
                )
                score = wp.tile_extract(
                    wp.tile_sum(wp.tile_map(dot, query_values, key_values)), 0
                )
                probability_grad = wp.tile_extract(
                    wp.tile_sum(wp.tile_map(dot, output_grad_values, value_values)),
                    0,
                )
                probability = wp.exp(
                    score * scale - lse[batch, query_head, query_token]
                )
                score_grad = (
                    probability
                    * (probability_grad - delta[batch, query_head, query_token])
                    * scale
                )
                key_accumulator = wp.tile_map(
                    add_scaled, key_accumulator, query_values, score_grad
                )
                value_accumulator = wp.tile_map(
                    add_scaled, value_accumulator, output_grad_values, probability
                )
        wp.tile_store(key_grad[batch, kv_head, key_token], key_accumulator)
        wp.tile_store(value_grad[batch, kv_head, key_token], value_accumulator)

    return streaming_forward, streaming_delta_query_grad, streaming_key_value_grad


def _validate_common(query, key, value, lengths) -> tuple[int, int, int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("GQA query, key, and value must have shape [B, H, S, D]")
    if (
        query.dtype not in _SUPPORTED_DTYPES
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        raise TypeError("GQA query, key, and value must share an FP16 or BF16 dtype")
    batch, query_heads, sequence, head_size = query.shape
    if min(batch, query_heads, sequence, head_size) <= 0:
        raise ValueError("GQA dimensions must be positive")
    if (
        key.shape != value.shape
        or key.shape[0] != batch
        or key.shape[2:] != (sequence, head_size)
    ):
        raise ValueError(
            "GQA key/value shapes must be [B, Hkv, S, D] and match query B, S, D"
        )
    kv_heads = key.shape[1]
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("GQA query heads must be divisible by key/value heads")
    if lengths.ndim != 1 or lengths.dtype != wp.int32 or lengths.shape[0] != batch:
        raise ValueError("GQA lengths must be an INT32 array with shape [B]")
    arrays = (query, key, value, lengths)
    if any(array.device != query.device for array in arrays):
        raise ValueError("GQA arrays must be on the same device")
    return batch, query_heads, sequence, head_size


def _validate_array(array, shape, dtype, device, name: str) -> None:
    if array.shape != shape or array.dtype != dtype:
        raise ValueError(f"GQA {name} must have shape {shape} and dtype {dtype}")
    if array.device != device:
        raise ValueError(f"GQA {name} must be on the query device")


def gqa_attention_forward(
    query,
    key,
    value,
    lengths,
    output,
    lse,
    accumulator,
    *,
    scale: float | None = None,
    window: int = 0,
) -> None:
    """Compute causal/sliding GQA into caller-owned output and saved-state arrays.

    ``window=0`` selects full causal attention.  A positive window is the maximum
    number of keys (including the current token) visible to each query.  ``lse``
    has shape ``[B, Hq, S]`` and is the only forward state needed by backward;
    ``accumulator`` is reusable FP32 workspace with the same shape as ``output``.
    """

    batch, query_heads, sequence, head_size = _validate_common(
        query, key, value, lengths
    )
    if window < 0:
        raise ValueError("GQA window must be non-negative")
    _validate_array(output, query.shape, query.dtype, query.device, "output")
    _validate_array(
        lse, (batch, query_heads, sequence), wp.float32, query.device, "LSE"
    )
    _validate_array(
        accumulator, query.shape, wp.float32, query.device, "forward accumulator"
    )
    effective_scale = head_size**-0.5 if scale is None else float(scale)
    if not math.isfinite(effective_scale):
        raise ValueError("GQA scale must be finite")

    if query.device.is_cuda:
        kernels = _attention_kernels(query.dtype, head_size)
        block_dim = min(1024, max(32, 1 << (head_size - 1).bit_length()))
        wp.launch_tiled(
            kernels[0],
            dim=batch * query_heads * sequence,
            inputs=[
                query,
                key,
                value,
                lengths,
                output,
                lse,
                accumulator,
                effective_scale,
                window,
            ],
            block_dim=block_dim,
            device=query.device,
        )
        return

    kernels = _attention_kernels(query.dtype)
    accumulator.zero_()
    wp.launch(
        kernels[0],
        dim=batch * query_heads * sequence,
        inputs=[query, key, lengths, lse, effective_scale, window],
        device=query.device,
    )
    wp.launch(
        kernels[1],
        dim=batch * query_heads * sequence * sequence,
        inputs=[query, key, value, lengths, lse, accumulator, effective_scale, window],
        device=query.device,
    )
    wp.launch(
        kernels[2], dim=query.shape, inputs=[accumulator, output], device=query.device
    )


def gqa_attention_backward(
    query,
    key,
    value,
    output_grad,
    lengths,
    lse,
    query_grad,
    key_grad,
    value_grad,
    delta,
    *,
    scale: float | None = None,
    window: int = 0,
    accumulate: bool = False,
) -> None:
    """Explicitly compute FP32 ``dQ``, ``dK``, and ``dV`` from saved forward LSE.

    This correctness/reference path saves O(S) LSE and delta state but performs
    exact O(S²) work. Gradient arrays accumulate only when requested; caller-owned
    ``delta`` is always overwritten. The operation is intentionally explicit rather
    than registered with :class:`warp.Tape`.
    """

    batch, query_heads, sequence, head_size = _validate_common(
        query, key, value, lengths
    )
    if window < 0:
        raise ValueError("GQA window must be non-negative")
    _validate_array(
        output_grad, query.shape, query.dtype, query.device, "output gradient"
    )
    _validate_array(
        lse, (batch, query_heads, sequence), wp.float32, query.device, "LSE"
    )
    _validate_array(query_grad, query.shape, wp.float32, query.device, "query gradient")
    _validate_array(key_grad, key.shape, wp.float32, query.device, "key gradient")
    _validate_array(value_grad, value.shape, wp.float32, query.device, "value gradient")
    _validate_array(
        delta, (batch, query_heads, sequence), wp.float32, query.device, "softmax delta"
    )
    effective_scale = head_size**-0.5 if scale is None else float(scale)
    if not math.isfinite(effective_scale):
        raise ValueError("GQA scale must be finite")

    if query.device.is_cuda:
        kernels = _attention_kernels(query.dtype, head_size)
        block_dim = min(1024, max(32, 1 << (head_size - 1).bit_length()))
        wp.launch_tiled(
            kernels[1],
            dim=batch * query_heads * sequence,
            inputs=[
                query,
                key,
                value,
                output_grad,
                lengths,
                lse,
                delta,
                query_grad,
                effective_scale,
                window,
                accumulate,
            ],
            block_dim=block_dim,
            device=query.device,
        )
        wp.launch_tiled(
            kernels[2],
            dim=batch * key.shape[1] * sequence,
            inputs=[
                query,
                key,
                value,
                output_grad,
                lengths,
                lse,
                delta,
                key_grad,
                value_grad,
                effective_scale,
                window,
                accumulate,
            ],
            block_dim=block_dim,
            device=query.device,
        )
        return

    kernels = _attention_kernels(query.dtype)
    if not accumulate:
        query_grad.zero_()
        key_grad.zero_()
        value_grad.zero_()
    wp.launch(
        kernels[3],
        dim=batch * query_heads * sequence,
        inputs=[
            query,
            key,
            value,
            output_grad,
            lengths,
            lse,
            delta,
            effective_scale,
            window,
        ],
        device=query.device,
    )
    wp.launch(
        kernels[4],
        dim=batch * query_heads * sequence * sequence,
        inputs=[
            query,
            key,
            value,
            output_grad,
            lengths,
            lse,
            delta,
            query_grad,
            key_grad,
            value_grad,
            effective_scale,
            window,
        ],
        device=query.device,
    )
