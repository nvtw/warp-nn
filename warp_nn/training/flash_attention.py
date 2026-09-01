# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Tensor-core online attention kernels for dependency-free GPU training."""

from functools import lru_cache
import math

import warp as wp


_QUERY_TILE = 16
_KEY_TILE = 32
_SUPPORTED_DTYPES = (wp.float16, wp.bfloat16)


@lru_cache(maxsize=None)
def _forward_kernel(dtype: type, head_size: int, segmented: bool = False):
    DTYPE = dtype
    HEAD_SIZE = head_size
    SEGMENTED = segmented

    @wp.func
    def maximum(left: wp.float32, right: wp.float32):
        return wp.max(left, right)

    @wp.func
    def minimum(left: wp.int32, right: wp.int32):
        return wp.min(left, right)

    @wp.func
    def maximum_int(left: wp.int32, right: wp.int32):
        return wp.max(left, right)

    @wp.func
    def add_offset(offset: wp.int32, base: wp.int32):
        return offset + base

    @wp.func
    def row_end(offset: wp.int32, query_start: wp.int32):
        return query_start + offset + 1

    @wp.func
    def row_first(end: wp.int32, window: wp.int32):
        return wp.max(0, end - window) if window > 0 else 0

    @wp.func
    def mask_score(
        score: wp.float32,
        key_position: wp.int32,
        first: wp.int32,
        end: wp.int32,
        length: wp.int32,
        scale: wp.float32,
    ):
        return (
            score * scale
            if end <= length and key_position >= first and key_position < end
            else wp.float32(-3.402823466e38)
        )

    @wp.func
    def masked_exp(
        score: wp.float32,
        maximum_value: wp.float32,
        key_position: wp.int32,
        first: wp.int32,
        end: wp.int32,
        length: wp.int32,
    ):
        return (
            wp.exp(score - maximum_value)
            if end <= length and key_position >= first and key_position < end
            else wp.float32(0.0)
        )

    @wp.func
    def exp_difference(old: wp.float32, new: wp.float32):
        return wp.exp(old - new)

    @wp.func
    def safe_inverse(value: wp.float32):
        return wp.float32(1.0) / value if value > 0.0 else wp.float32(0.0)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        query: wp.array2d(dtype=DTYPE),
        key: wp.array2d(dtype=DTYPE),
        value: wp.array2d(dtype=DTYPE),
        lengths: wp.array1d(dtype=wp.int32),
        segment_bounds: wp.array2d(dtype=wp.int32),
        output: wp.array2d(dtype=DTYPE),
        lse: wp.array1d(dtype=wp.float32),
        workspace: wp.array2d(dtype=wp.float32),
        query_heads: wp.int32,
        kv_heads: wp.int32,
        sequence: wp.int32,
        scale: wp.float32,
        window: wp.int32,
    ):
        item = wp.tid()
        query_tiles = (sequence + _QUERY_TILE - 1) / _QUERY_TILE
        tile_index = item % query_tiles
        query_head = (item / query_tiles) % query_heads
        batch = item / (query_tiles * query_heads)
        query_start = tile_index * _QUERY_TILE
        length = wp.clamp(lengths[batch], 0, sequence)
        kv_head = query_head / (query_heads / kv_heads)
        query_base = (batch * query_heads + query_head) * sequence
        cache_base = (batch * kv_heads + kv_head) * sequence

        queries = wp.tile_load(
            query,
            shape=(_QUERY_TILE, HEAD_SIZE),
            offset=(query_base + query_start, 0),
        )
        accumulator = wp.tile_zeros(shape=(_QUERY_TILE, HEAD_SIZE), dtype=wp.float32)
        maximum_values = wp.tile_full(
            shape=(_QUERY_TILE,),
            value=wp.float32(-3.402823466e38),
            dtype=wp.float32,
        )
        denominators = wp.tile_zeros(shape=(_QUERY_TILE,), dtype=wp.float32)
        query_offsets = wp.tile_arange(_QUERY_TILE, dtype=wp.int32)
        row_ends = wp.tile_map(row_end, query_offsets, query_start)
        row_firsts = wp.tile_map(row_first, row_ends, window)
        if wp.static(SEGMENTED):
            bounds = wp.tile_load(
                segment_bounds,
                shape=(_QUERY_TILE, 2),
                offset=(batch * sequence + query_start, 0),
            )
            segment_first_column = wp.tile_zeros(shape=(_QUERY_TILE, 1), dtype=wp.int32)
            wp.tile_assign(
                segment_first_column,
                wp.tile_view(bounds, offset=(0, 0), shape=(_QUERY_TILE, 1)),
            )
            segment_firsts = wp.tile_reshape(segment_first_column, shape=(_QUERY_TILE,))
            row_firsts = wp.tile_map(maximum_int, row_firsts, segment_firsts)
        row_end_group = wp.tile_broadcast(
            wp.tile_reshape(row_ends, shape=(_QUERY_TILE, 1)),
            shape=(_QUERY_TILE, _KEY_TILE),
        )
        row_first_group = wp.tile_broadcast(
            wp.tile_reshape(row_firsts, shape=(_QUERY_TILE, 1)),
            shape=(_QUERY_TILE, _KEY_TILE),
        )
        key_offsets = wp.tile_arange(_KEY_TILE, dtype=wp.int32)
        key_limit = wp.min(length, query_start + _QUERY_TILE)
        key_begin = wp.int32(0)
        if wp.static(SEGMENTED):
            first = wp.tile_extract(wp.tile_reduce(minimum, row_firsts), 0)
            key_begin = (first / _KEY_TILE) * _KEY_TILE

        for key_start in range(key_begin, key_limit, _KEY_TILE):
            keys = wp.tile_load(
                key,
                shape=(_KEY_TILE, HEAD_SIZE),
                offset=(cache_base + key_start, 0),
            )
            scores = wp.tile_zeros(shape=(_QUERY_TILE, _KEY_TILE), dtype=wp.float32)
            wp.tile_matmul(queries, wp.tile_transpose(keys), scores)
            key_positions = wp.tile_map(add_offset, key_offsets, key_start)
            key_position_group = wp.tile_broadcast(
                wp.tile_reshape(key_positions, shape=(1, _KEY_TILE)),
                shape=(_QUERY_TILE, _KEY_TILE),
            )
            scores = wp.tile_map(
                mask_score,
                scores,
                key_position_group,
                row_first_group,
                row_end_group,
                length,
                scale,
            )
            block_maximum = wp.tile_reduce(maximum, scores, axis=1)
            new_maximum = wp.tile_map(maximum, maximum_values, block_maximum)
            old_scale = wp.tile_map(exp_difference, maximum_values, new_maximum)
            maximum_group = wp.tile_broadcast(
                wp.tile_reshape(new_maximum, shape=(_QUERY_TILE, 1)),
                shape=(_QUERY_TILE, _KEY_TILE),
            )
            probabilities = wp.tile_map(
                masked_exp,
                scores,
                maximum_group,
                key_position_group,
                row_first_group,
                row_end_group,
                length,
            )
            denominators = denominators * old_scale + wp.tile_sum(probabilities, axis=1)
            old_scale_group = wp.tile_broadcast(
                wp.tile_reshape(old_scale, shape=(_QUERY_TILE, 1)),
                shape=(_QUERY_TILE, HEAD_SIZE),
            )
            values = wp.tile_load(
                value,
                shape=(_KEY_TILE, HEAD_SIZE),
                offset=(cache_base + key_start, 0),
            )
            contribution = wp.tile_zeros(
                shape=(_QUERY_TILE, HEAD_SIZE), dtype=wp.float32
            )
            wp.tile_matmul(
                wp.tile_astype(probabilities, dtype=DTYPE), values, contribution
            )
            accumulator = accumulator * old_scale_group + contribution
            maximum_values = new_maximum

        inverse = wp.tile_map(safe_inverse, denominators)
        inverse_group = wp.tile_broadcast(
            wp.tile_reshape(inverse, shape=(_QUERY_TILE, 1)),
            shape=(_QUERY_TILE, HEAD_SIZE),
        )
        normalized = accumulator * inverse_group
        for row in range(_QUERY_TILE):
            token = query_start + row
            if token < sequence:
                normalized_row = wp.tile_view(
                    normalized, offset=(row, 0), shape=(1, HEAD_SIZE)
                )
                wp.tile_store(workspace, normalized_row, offset=(query_base + token, 0))
                wp.tile_store(
                    output,
                    wp.tile_astype(normalized_row, dtype=DTYPE),
                    offset=(query_base + token, 0),
                )
                lse[query_base + token] = (
                    wp.tile_extract(maximum_values, row)
                    + wp.log(wp.tile_extract(denominators, row))
                    if token < length
                    else wp.float32(-3.402823466e38)
                )

    kernel.module.options["enable_backward"] = False
    return kernel


def _backward_tile_shape(head_size: int) -> tuple[int, int]:
    """Use smaller tiles when D256 would exceed portable shared-memory limits."""
    return (16, 8) if head_size > 128 else (16, 16)


@lru_cache(maxsize=None)
def _query_backward_kernel(dtype: type, head_size: int, segmented: bool = False):
    """Build the tensor-core softmax-delta and query-gradient pass."""
    DTYPE = dtype
    HEAD_SIZE = head_size
    QUERY_TILE, KEY_TILE = _backward_tile_shape(head_size)
    SEGMENTED = segmented

    @wp.func
    def maximum(left: wp.int32, right: wp.int32):
        return wp.max(left, right)

    @wp.func
    def minimum(left: wp.int32, right: wp.int32):
        return wp.min(left, right)

    @wp.func
    def add_offset(offset: wp.int32, base: wp.int32):
        return offset + base

    @wp.func
    def row_end(offset: wp.int32, query_start: wp.int32):
        return query_start + offset + 1

    @wp.func
    def row_first(end: wp.int32, window: wp.int32):
        return wp.max(0, end - window) if window > 0 else 0

    @wp.func
    def probability(
        score: wp.float32,
        key_position: wp.int32,
        first: wp.int32,
        end: wp.int32,
        length: wp.int32,
        normalizer: wp.float32,
        scale: wp.float32,
    ):
        return (
            wp.exp(score * scale - normalizer)
            if end <= length and key_position >= first and key_position < end
            else wp.float32(0.0)
        )

    @wp.func
    def score_gradient(
        probability_value: wp.float32,
        probability_gradient: wp.float32,
        delta_value: wp.float32,
        scale: wp.float32,
    ):
        return probability_value * (probability_gradient - delta_value) * scale

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        query: wp.array2d(dtype=DTYPE),
        key: wp.array2d(dtype=DTYPE),
        value: wp.array2d(dtype=DTYPE),
        output_grad: wp.array2d(dtype=DTYPE),
        lengths: wp.array1d(dtype=wp.int32),
        segment_bounds: wp.array2d(dtype=wp.int32),
        lse: wp.array1d(dtype=wp.float32),
        delta: wp.array1d(dtype=wp.float32),
        query_grad: wp.array2d(dtype=wp.float32),
        query_heads: wp.int32,
        kv_heads: wp.int32,
        sequence: wp.int32,
        scale: wp.float32,
        window: wp.int32,
        accumulate: bool,
    ):
        item = wp.tid()
        query_tiles = (sequence + QUERY_TILE - 1) / QUERY_TILE
        tile_index = item % query_tiles
        query_head = (item / query_tiles) % query_heads
        batch = item / (query_tiles * query_heads)
        query_start = tile_index * QUERY_TILE
        length = wp.clamp(lengths[batch], 0, sequence)
        kv_head = query_head / (query_heads / kv_heads)
        query_base = (batch * query_heads + query_head) * sequence
        cache_base = (batch * kv_heads + kv_head) * sequence

        queries = wp.tile_load(
            query,
            shape=(QUERY_TILE, HEAD_SIZE),
            offset=(query_base + query_start, 0),
        )
        output_gradients = wp.tile_load(
            output_grad,
            shape=(QUERY_TILE, HEAD_SIZE),
            offset=(query_base + query_start, 0),
        )
        normalizers = wp.tile_load(
            lse, shape=(QUERY_TILE,), offset=(query_base + query_start,)
        )
        normalizer_group = wp.tile_broadcast(
            wp.tile_reshape(normalizers, shape=(QUERY_TILE, 1)),
            shape=(QUERY_TILE, KEY_TILE),
        )
        query_offsets = wp.tile_arange(QUERY_TILE, dtype=wp.int32)
        ends = wp.tile_map(row_end, query_offsets, query_start)
        firsts = wp.tile_map(row_first, ends, window)
        if wp.static(SEGMENTED):
            bounds = wp.tile_load(
                segment_bounds,
                shape=(QUERY_TILE, 2),
                offset=(batch * sequence + query_start, 0),
            )
            segment_first_column = wp.tile_zeros(shape=(QUERY_TILE, 1), dtype=wp.int32)
            wp.tile_assign(
                segment_first_column,
                wp.tile_view(bounds, offset=(0, 0), shape=(QUERY_TILE, 1)),
            )
            segment_firsts = wp.tile_reshape(segment_first_column, shape=(QUERY_TILE,))
            firsts = wp.tile_map(maximum, firsts, segment_firsts)
        end_group = wp.tile_broadcast(
            wp.tile_reshape(ends, shape=(QUERY_TILE, 1)),
            shape=(QUERY_TILE, KEY_TILE),
        )
        first_group = wp.tile_broadcast(
            wp.tile_reshape(firsts, shape=(QUERY_TILE, 1)),
            shape=(QUERY_TILE, KEY_TILE),
        )
        key_offsets = wp.tile_arange(KEY_TILE, dtype=wp.int32)
        key_limit = wp.min(length, query_start + QUERY_TILE)
        key_begin = wp.int32(0)
        if wp.static(SEGMENTED):
            first = wp.tile_extract(wp.tile_reduce(minimum, firsts), 0)
            key_begin = (first / KEY_TILE) * KEY_TILE
        delta_values = wp.tile_zeros(shape=(QUERY_TILE,), dtype=wp.float32)

        for key_start in range(key_begin, key_limit, KEY_TILE):
            keys = wp.tile_load(
                key,
                shape=(KEY_TILE, HEAD_SIZE),
                offset=(cache_base + key_start, 0),
            )
            values = wp.tile_load(
                value,
                shape=(KEY_TILE, HEAD_SIZE),
                offset=(cache_base + key_start, 0),
            )
            scores = wp.tile_zeros(shape=(QUERY_TILE, KEY_TILE), dtype=wp.float32)
            probability_gradients = wp.tile_zeros(
                shape=(QUERY_TILE, KEY_TILE), dtype=wp.float32
            )
            wp.tile_matmul(queries, wp.tile_transpose(keys), scores)
            wp.tile_matmul(
                output_gradients, wp.tile_transpose(values), probability_gradients
            )
            key_positions = wp.tile_map(add_offset, key_offsets, key_start)
            key_group = wp.tile_broadcast(
                wp.tile_reshape(key_positions, shape=(1, KEY_TILE)),
                shape=(QUERY_TILE, KEY_TILE),
            )
            probabilities = wp.tile_map(
                probability,
                scores,
                key_group,
                first_group,
                end_group,
                length,
                normalizer_group,
                scale,
            )
            delta_values += wp.tile_sum(probabilities * probability_gradients, axis=1)

        query_accumulator = wp.tile_zeros(
            shape=(QUERY_TILE, HEAD_SIZE), dtype=wp.float32
        )
        if accumulate:
            query_accumulator = wp.tile_load(
                query_grad,
                shape=(QUERY_TILE, HEAD_SIZE),
                offset=(query_base + query_start, 0),
            )
        delta_group = wp.tile_broadcast(
            wp.tile_reshape(delta_values, shape=(QUERY_TILE, 1)),
            shape=(QUERY_TILE, KEY_TILE),
        )
        for key_start in range(key_begin, key_limit, KEY_TILE):
            keys = wp.tile_load(
                key,
                shape=(KEY_TILE, HEAD_SIZE),
                offset=(cache_base + key_start, 0),
            )
            values = wp.tile_load(
                value,
                shape=(KEY_TILE, HEAD_SIZE),
                offset=(cache_base + key_start, 0),
            )
            scores = wp.tile_zeros(shape=(QUERY_TILE, KEY_TILE), dtype=wp.float32)
            probability_gradients = wp.tile_zeros(
                shape=(QUERY_TILE, KEY_TILE), dtype=wp.float32
            )
            wp.tile_matmul(queries, wp.tile_transpose(keys), scores)
            wp.tile_matmul(
                output_gradients, wp.tile_transpose(values), probability_gradients
            )
            key_positions = wp.tile_map(add_offset, key_offsets, key_start)
            key_group = wp.tile_broadcast(
                wp.tile_reshape(key_positions, shape=(1, KEY_TILE)),
                shape=(QUERY_TILE, KEY_TILE),
            )
            probabilities = wp.tile_map(
                probability,
                scores,
                key_group,
                first_group,
                end_group,
                length,
                normalizer_group,
                scale,
            )
            score_gradients = wp.tile_map(
                score_gradient,
                probabilities,
                probability_gradients,
                delta_group,
                scale,
            )
            contribution = wp.tile_zeros(
                shape=(QUERY_TILE, HEAD_SIZE), dtype=wp.float32
            )
            wp.tile_matmul(
                wp.tile_astype(score_gradients, dtype=DTYPE), keys, contribution
            )
            query_accumulator += contribution

        for row in range(QUERY_TILE):
            token = query_start + row
            if token < sequence:
                delta[query_base + token] = wp.tile_extract(delta_values, row)
                gradient_row = wp.tile_view(
                    query_accumulator, offset=(row, 0), shape=(1, HEAD_SIZE)
                )
                wp.tile_store(query_grad, gradient_row, offset=(query_base + token, 0))

    kernel.module.options["enable_backward"] = False
    return kernel


def flash_gqa_query_backward(
    query,
    key,
    value,
    output_grad,
    lengths,
    lse,
    delta,
    query_grad,
    *,
    segment_bounds=None,
    scale: float,
    window: int,
    accumulate: bool,
) -> None:
    """Recompute softmax tiles and write FP32 delta and dQ without atomics."""
    batch, query_heads, sequence, head_size = query.shape
    kv_heads = key.shape[1]
    query_tile, _ = _backward_tile_shape(head_size)
    query_rows = batch * query_heads * sequence
    kv_rows = batch * kv_heads * sequence
    bounds = _checked_segment_bounds(segment_bounds, lengths, batch, sequence)
    wp.launch_tiled(
        _query_backward_kernel(query.dtype, head_size, segment_bounds is not None),
        dim=batch * query_heads * ((sequence + query_tile - 1) // query_tile),
        inputs=[
            query.reshape((query_rows, head_size)),
            key.reshape((kv_rows, head_size)),
            value.reshape((kv_rows, head_size)),
            output_grad.reshape((query_rows, head_size)),
            lengths,
            bounds,
            lse.flatten(),
            delta.flatten(),
            query_grad.reshape((query_rows, head_size)),
            query_heads,
            kv_heads,
            sequence,
            scale,
            window,
            accumulate,
        ],
        block_dim=128,
        device=query.device,
    )


@lru_cache(maxsize=None)
def _key_value_backward_kernel(dtype: type, head_size: int, segmented: bool = False):
    """Build the unique-writer tensor-core key/value-gradient pass."""
    DTYPE = dtype
    HEAD_SIZE = head_size
    QUERY_TILE, KEY_TILE = _backward_tile_shape(head_size)
    SEGMENTED = segmented

    @wp.func
    def maximum(left: wp.int32, right: wp.int32):
        return wp.max(left, right)

    @wp.func
    def add_offset(offset: wp.int32, base: wp.int32):
        return offset + base

    @wp.func
    def row_end(offset: wp.int32, query_start: wp.int32):
        return query_start + offset + 1

    @wp.func
    def row_first(end: wp.int32, window: wp.int32):
        return wp.max(0, end - window) if window > 0 else 0

    @wp.func
    def probability(
        score: wp.float32,
        key_position: wp.int32,
        first: wp.int32,
        end: wp.int32,
        length: wp.int32,
        normalizer: wp.float32,
        scale: wp.float32,
    ):
        return (
            wp.exp(score * scale - normalizer)
            if end <= length and key_position >= first and key_position < end
            else wp.float32(0.0)
        )

    @wp.func
    def score_gradient(
        probability_value: wp.float32,
        probability_gradient: wp.float32,
        delta_value: wp.float32,
        scale: wp.float32,
    ):
        return probability_value * (probability_gradient - delta_value) * scale

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        query: wp.array2d(dtype=DTYPE),
        key: wp.array2d(dtype=DTYPE),
        value: wp.array2d(dtype=DTYPE),
        output_grad: wp.array2d(dtype=DTYPE),
        lengths: wp.array1d(dtype=wp.int32),
        segment_bounds: wp.array2d(dtype=wp.int32),
        lse: wp.array1d(dtype=wp.float32),
        delta: wp.array1d(dtype=wp.float32),
        key_grad: wp.array2d(dtype=wp.float32),
        value_grad: wp.array2d(dtype=wp.float32),
        query_heads: wp.int32,
        kv_heads: wp.int32,
        sequence: wp.int32,
        scale: wp.float32,
        window: wp.int32,
        accumulate: bool,
    ):
        item = wp.tid()
        key_tiles = (sequence + KEY_TILE - 1) / KEY_TILE
        tile_index = item % key_tiles
        kv_head = (item / key_tiles) % kv_heads
        batch = item / (key_tiles * kv_heads)
        key_start = tile_index * KEY_TILE
        length = wp.clamp(lengths[batch], 0, sequence)
        cache_base = (batch * kv_heads + kv_head) * sequence
        keys = wp.tile_load(
            key,
            shape=(KEY_TILE, HEAD_SIZE),
            offset=(cache_base + key_start, 0),
        )
        values = wp.tile_load(
            value,
            shape=(KEY_TILE, HEAD_SIZE),
            offset=(cache_base + key_start, 0),
        )
        key_accumulator = wp.tile_zeros(shape=(KEY_TILE, HEAD_SIZE), dtype=wp.float32)
        value_accumulator = wp.tile_zeros(shape=(KEY_TILE, HEAD_SIZE), dtype=wp.float32)
        if accumulate:
            key_accumulator = wp.tile_load(
                key_grad,
                shape=(KEY_TILE, HEAD_SIZE),
                offset=(cache_base + key_start, 0),
            )
            value_accumulator = wp.tile_load(
                value_grad,
                shape=(KEY_TILE, HEAD_SIZE),
                offset=(cache_base + key_start, 0),
            )

        key_offsets = wp.tile_arange(KEY_TILE, dtype=wp.int32)
        key_positions = wp.tile_map(add_offset, key_offsets, key_start)
        key_group = wp.tile_broadcast(
            wp.tile_reshape(key_positions, shape=(1, KEY_TILE)),
            shape=(QUERY_TILE, KEY_TILE),
        )
        query_offsets = wp.tile_arange(QUERY_TILE, dtype=wp.int32)
        heads_per_kv = query_heads / kv_heads
        query_begin = wp.int32(0)
        query_limit = length
        if wp.static(SEGMENTED):
            bounds = wp.tile_load(
                segment_bounds,
                shape=(KEY_TILE, 2),
                offset=(batch * sequence + key_start, 0),
            )
            segment_end_column = wp.tile_zeros(shape=(KEY_TILE, 1), dtype=wp.int32)
            wp.tile_assign(
                segment_end_column,
                wp.tile_view(bounds, offset=(0, 1), shape=(KEY_TILE, 1)),
            )
            segment_ends = wp.tile_reshape(segment_end_column, shape=(KEY_TILE,))
            query_begin = (key_start / QUERY_TILE) * QUERY_TILE
            query_limit = wp.min(
                length,
                wp.tile_extract(wp.tile_reduce(maximum, segment_ends), 0),
            )
        for query_head in range(kv_head * heads_per_kv, (kv_head + 1) * heads_per_kv):
            query_base = (batch * query_heads + query_head) * sequence
            for query_start in range(query_begin, query_limit, QUERY_TILE):
                query_end = wp.min(length, query_start + QUERY_TILE)
                common_first = wp.max(0, query_start + 1 - window) if window > 0 else 0
                if key_start < query_end and key_start + KEY_TILE > common_first:
                    queries = wp.tile_load(
                        query,
                        shape=(QUERY_TILE, HEAD_SIZE),
                        offset=(query_base + query_start, 0),
                    )
                    output_gradients = wp.tile_load(
                        output_grad,
                        shape=(QUERY_TILE, HEAD_SIZE),
                        offset=(query_base + query_start, 0),
                    )
                    normalizers = wp.tile_load(
                        lse,
                        shape=(QUERY_TILE,),
                        offset=(query_base + query_start,),
                    )
                    delta_values = wp.tile_load(
                        delta,
                        shape=(QUERY_TILE,),
                        offset=(query_base + query_start,),
                    )
                    ends = wp.tile_map(row_end, query_offsets, query_start)
                    firsts = wp.tile_map(row_first, ends, window)
                    if wp.static(SEGMENTED):
                        query_bounds = wp.tile_load(
                            segment_bounds,
                            shape=(QUERY_TILE, 2),
                            offset=(batch * sequence + query_start, 0),
                        )
                        segment_first_column = wp.tile_zeros(
                            shape=(QUERY_TILE, 1), dtype=wp.int32
                        )
                        wp.tile_assign(
                            segment_first_column,
                            wp.tile_view(
                                query_bounds,
                                offset=(0, 0),
                                shape=(QUERY_TILE, 1),
                            ),
                        )
                        segment_firsts = wp.tile_reshape(
                            segment_first_column, shape=(QUERY_TILE,)
                        )
                        firsts = wp.tile_map(maximum, firsts, segment_firsts)
                    end_group = wp.tile_broadcast(
                        wp.tile_reshape(ends, shape=(QUERY_TILE, 1)),
                        shape=(QUERY_TILE, KEY_TILE),
                    )
                    first_group = wp.tile_broadcast(
                        wp.tile_reshape(firsts, shape=(QUERY_TILE, 1)),
                        shape=(QUERY_TILE, KEY_TILE),
                    )
                    normalizer_group = wp.tile_broadcast(
                        wp.tile_reshape(normalizers, shape=(QUERY_TILE, 1)),
                        shape=(QUERY_TILE, KEY_TILE),
                    )
                    delta_group = wp.tile_broadcast(
                        wp.tile_reshape(delta_values, shape=(QUERY_TILE, 1)),
                        shape=(QUERY_TILE, KEY_TILE),
                    )
                    scores = wp.tile_zeros(
                        shape=(QUERY_TILE, KEY_TILE), dtype=wp.float32
                    )
                    probability_gradients = wp.tile_zeros(
                        shape=(QUERY_TILE, KEY_TILE), dtype=wp.float32
                    )
                    wp.tile_matmul(queries, wp.tile_transpose(keys), scores)
                    wp.tile_matmul(
                        output_gradients,
                        wp.tile_transpose(values),
                        probability_gradients,
                    )
                    probabilities = wp.tile_map(
                        probability,
                        scores,
                        key_group,
                        first_group,
                        end_group,
                        length,
                        normalizer_group,
                        scale,
                    )
                    score_gradients = wp.tile_map(
                        score_gradient,
                        probabilities,
                        probability_gradients,
                        delta_group,
                        scale,
                    )
                    wp.tile_matmul(
                        wp.tile_transpose(wp.tile_astype(score_gradients, dtype=DTYPE)),
                        queries,
                        key_accumulator,
                    )
                    wp.tile_matmul(
                        wp.tile_transpose(wp.tile_astype(probabilities, dtype=DTYPE)),
                        output_gradients,
                        value_accumulator,
                    )

        for row in range(KEY_TILE):
            token = key_start + row
            if token < sequence:
                key_row = wp.tile_view(
                    key_accumulator, offset=(row, 0), shape=(1, HEAD_SIZE)
                )
                value_row = wp.tile_view(
                    value_accumulator, offset=(row, 0), shape=(1, HEAD_SIZE)
                )
                wp.tile_store(key_grad, key_row, offset=(cache_base + token, 0))
                wp.tile_store(value_grad, value_row, offset=(cache_base + token, 0))

    kernel.module.options["enable_backward"] = False
    return kernel


def flash_gqa_key_value_backward(
    query,
    key,
    value,
    output_grad,
    lengths,
    lse,
    delta,
    key_grad,
    value_grad,
    *,
    segment_bounds=None,
    scale: float,
    window: int,
    accumulate: bool,
) -> None:
    """Recompute softmax tiles and uniquely write FP32 dK/dV without atomics."""
    batch, query_heads, sequence, head_size = query.shape
    kv_heads = key.shape[1]
    query_rows = batch * query_heads * sequence
    _, key_tile = _backward_tile_shape(head_size)
    kv_rows = batch * kv_heads * sequence
    bounds = _checked_segment_bounds(segment_bounds, lengths, batch, sequence)
    wp.launch_tiled(
        _key_value_backward_kernel(query.dtype, head_size, segment_bounds is not None),
        dim=batch * kv_heads * ((sequence + key_tile - 1) // key_tile),
        inputs=[
            query.reshape((query_rows, head_size)),
            key.reshape((kv_rows, head_size)),
            value.reshape((kv_rows, head_size)),
            output_grad.reshape((query_rows, head_size)),
            lengths,
            bounds,
            lse.flatten(),
            delta.flatten(),
            key_grad.reshape((kv_rows, head_size)),
            value_grad.reshape((kv_rows, head_size)),
            query_heads,
            kv_heads,
            sequence,
            scale,
            window,
            accumulate,
        ],
        block_dim=128,
        device=query.device,
    )


def _checked_segment_bounds(segment_bounds, lengths, batch: int, sequence: int):
    if segment_bounds is None:
        return lengths.reshape((batch, 1))
    if (
        segment_bounds.shape != (batch, sequence, 2)
        or segment_bounds.dtype != wp.int32
        or segment_bounds.device != lengths.device
        or not segment_bounds.is_contiguous
    ):
        raise ValueError(
            "segment_bounds must be contiguous int32 [batch, sequence, 2] on the input device"
        )
    return segment_bounds.reshape((batch * sequence, 2))


def flash_gqa_forward(
    query,
    key,
    value,
    lengths,
    output,
    lse,
    workspace,
    *,
    segment_bounds=None,
    scale: float | None = None,
    window: int = 0,
) -> None:
    """Launch exact causal/windowed online GQA using tensor-core tiles."""
    if query.dtype not in _SUPPORTED_DTYPES or key.dtype != query.dtype:
        raise TypeError("Flash GQA requires matching FP16/BF16 Q/K/V storage")
    if value.dtype != query.dtype or query.ndim != 4 or key.ndim != 4:
        raise TypeError("Flash GQA requires rank-4 matching Q/K/V arrays")
    batch, query_heads, sequence, head_size = query.shape
    if (
        key.shape != value.shape
        or key.shape[0] != batch
        or key.shape[2:]
        != (
            sequence,
            head_size,
        )
    ):
        raise ValueError("Flash GQA key/value geometry does not match query")
    kv_heads = key.shape[1]
    if min(batch, query_heads, kv_heads, sequence, head_size) <= 0:
        raise ValueError("Flash GQA dimensions must be positive")
    if query_heads % kv_heads:
        raise ValueError("Flash GQA query heads must be divisible by KV heads")
    if not query.device.is_cuda or query.device.arch < (
        80 if query.dtype == wp.bfloat16 else 70
    ):
        raise ValueError("Flash GQA tensor-core path is unsupported on this device")
    if any(
        array.device != query.device or not array.is_contiguous
        for array in (query, key, value, lengths, output, lse, workspace)
    ):
        raise ValueError("Flash GQA arrays must be contiguous on one device")
    if output.shape != query.shape or output.dtype != query.dtype:
        raise ValueError("Flash GQA output geometry does not match query")
    if workspace.shape != query.shape or workspace.dtype != wp.float32:
        raise ValueError("Flash GQA workspace must be FP32 with query shape")
    if lse.shape != (batch, query_heads, sequence) or lse.dtype != wp.float32:
        raise ValueError("Flash GQA LSE geometry is invalid")
    if lengths.shape != (batch,) or lengths.dtype != wp.int32:
        raise ValueError("Flash GQA lengths geometry is invalid")
    if window < 0:
        raise ValueError("Flash GQA window must be non-negative")
    effective_scale = head_size**-0.5 if scale is None else float(scale)
    if not math.isfinite(effective_scale):
        raise ValueError("Flash GQA scale must be finite")

    query_2d = query.reshape((batch * query_heads * sequence, head_size))
    key_2d = key.reshape((batch * kv_heads * sequence, head_size))
    value_2d = value.reshape((batch * kv_heads * sequence, head_size))
    output_2d = output.reshape((batch * query_heads * sequence, head_size))
    lse_1d = lse.flatten()
    workspace_2d = workspace.reshape((batch * query_heads * sequence, head_size))
    bounds = _checked_segment_bounds(segment_bounds, lengths, batch, sequence)
    wp.launch_tiled(
        _forward_kernel(query.dtype, head_size, segment_bounds is not None),
        dim=batch * query_heads * ((sequence + _QUERY_TILE - 1) // _QUERY_TILE),
        inputs=[
            query_2d,
            key_2d,
            value_2d,
            lengths,
            bounds,
            output_2d,
            lse_1d,
            workspace_2d,
            query_heads,
            kv_heads,
            sequence,
            effective_scale,
            window,
        ],
        block_dim=128,
        device=query.device,
    )
