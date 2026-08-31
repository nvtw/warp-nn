# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Chunkwise tensor-core Gated DeltaNet recurrence for training."""

from dataclasses import dataclass
from functools import lru_cache
import math

import warp as wp


_CHUNK = 16
_CUDA_DTYPES = (wp.float16, wp.bfloat16)
_STORAGE_DTYPES = (wp.float32, *_CUDA_DTYPES)


@dataclass(frozen=True)
class _RuleKernels:
    log_decay: object
    scalar: object
    chunkwise: object
    materialize_states: object
    reverse: object


@lru_cache(maxsize=None)
def _kernels(dtype: type, key_size: int, value_size: int, value_tile: int):
    DTYPE = dtype
    KEY_SIZE = key_size
    VALUE_SIZE = value_size
    VALUE_TILE = value_tile
    REVERSE_TILE = min(16, VALUE_TILE)
    MATERIALIZE_TILE = min(8, VALUE_TILE)

    @wp.func
    def to_fp32(value: DTYPE):
        return wp.float32(value)

    @wp.func
    def to_storage(value: wp.float32):
        return DTYPE(value)

    @wp.func
    def scale_storage(value: wp.float32, scale: wp.float32):
        return DTYPE(value * scale)

    @wp.func
    def scale_row(value: DTYPE, factor: wp.float32):
        return DTYPE(wp.float32(value) * factor)

    @wp.kernel(enable_backward=False, module="unique")
    def log_decay(
        decay: wp.array3d(dtype=wp.float32),
        lengths: wp.array1d(dtype=wp.int32),
        output: wp.array3d(dtype=wp.float32),
    ):
        batch, head, chunk = wp.tid()
        start = chunk * _CHUNK
        length = wp.clamp(lengths[batch], 0, decay.shape[1])
        total = wp.float32(0.0)
        for row in range(_CHUNK):
            token = start + row
            if token < decay.shape[1] and token < length:
                total += wp.log(wp.max(decay[batch, token, head], wp.float32(1.0e-30)))
                output[batch, head, token] = total
            elif token < decay.shape[1]:
                output[batch, head, token] = total

    @wp.kernel(enable_backward=False, module="unique")
    def scalar(
        query: wp.array4d(dtype=DTYPE),
        key: wp.array4d(dtype=DTYPE),
        value: wp.array4d(dtype=DTYPE),
        decay: wp.array3d(dtype=wp.float32),
        beta: wp.array3d(dtype=wp.float32),
        lengths: wp.array1d(dtype=wp.int32),
        past: wp.array4d(dtype=wp.float32),
        output: wp.array4d(dtype=DTYPE),
        transformed: wp.array4d(dtype=DTYPE),
        present: wp.array4d(dtype=wp.float32),
        checkpoints: wp.array4d(dtype=wp.float32),
        scale: wp.float32,
    ):
        batch, value_head, value_column = wp.tid()
        key_head = value_head * query.shape[1] // value.shape[1]
        length = wp.clamp(lengths[batch], 0, query.shape[2])
        for token in range(query.shape[2]):
            if token < length:
                decay_value = decay[batch, token, value_head]
                retrieved = wp.float32(0.0)
                for column in range(KEY_SIZE):
                    if token % _CHUNK == 0:
                        checkpoints[
                            batch * value.shape[1] + value_head,
                            token // _CHUNK,
                            column,
                            value_column,
                        ] = past[batch, value_head, column, value_column]
                    retrieved += (
                        wp.float32(key[batch, key_head, token, column])
                        * past[batch, value_head, column, value_column]
                    )
                delta = beta[batch, token, value_head] * (
                    wp.float32(value[batch, value_head, token, value_column])
                    - decay_value * retrieved
                )
                stored_delta = DTYPE(delta)
                transformed[batch, value_head, token, value_column] = stored_delta
                delta = wp.float32(stored_delta)
                result = wp.float32(0.0)
                for column in range(KEY_SIZE):
                    state_value = (
                        decay_value * past[batch, value_head, column, value_column]
                        + wp.float32(key[batch, key_head, token, column]) * delta
                    )
                    past[batch, value_head, column, value_column] = state_value
                    result += (
                        wp.float32(query[batch, key_head, token, column]) * state_value
                    )
                output[batch, value_head, token, value_column] = DTYPE(scale * result)
            else:
                transformed[batch, value_head, token, value_column] = DTYPE(0.0)
                output[batch, value_head, token, value_column] = DTYPE(0.0)
        for column in range(KEY_SIZE):
            state_value = past[batch, value_head, column, value_column]
            present[batch, value_head, column, value_column] = state_value

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def chunkwise(
        query: wp.array2d(dtype=DTYPE),
        key: wp.array2d(dtype=DTYPE),
        value: wp.array2d(dtype=DTYPE),
        local_log_decay: wp.array3d(dtype=wp.float32),
        beta: wp.array3d(dtype=wp.float32),
        lengths: wp.array1d(dtype=wp.int32),
        past: wp.array2d(dtype=wp.float32),
        output: wp.array2d(dtype=DTYPE),
        transformed: wp.array2d(dtype=DTYPE),
        present: wp.array2d(dtype=wp.float32),
        checkpoints: wp.array2d(dtype=wp.float32),
        key_heads: wp.int32,
        value_heads: wp.int32,
        sequence: wp.int32,
        chunks: wp.int32,
        scale: wp.float32,
    ):
        item = wp.tid()
        value_tiles = (VALUE_SIZE + VALUE_TILE - 1) // VALUE_TILE
        value_tile_index = item % value_tiles
        state_item = item // value_tiles
        value_head = state_item % value_heads
        batch = state_item // value_heads
        key_head = value_head * key_heads // value_heads
        value_offset = value_tile_index * VALUE_TILE
        state_base = (batch * value_heads + value_head) * KEY_SIZE
        state = wp.tile_load(
            past,
            shape=(KEY_SIZE, VALUE_TILE),
            offset=(state_base, value_offset),
        )

        for chunk in range(chunks):
            token_start = chunk * _CHUNK
            checkpoint_base = (
                (batch * value_heads + value_head) * chunks + chunk
            ) * KEY_SIZE
            wp.tile_store(
                checkpoints,
                state,
                offset=(checkpoint_base, value_offset),
            )
            key_base = (batch * key_heads + key_head) * sequence + token_start
            value_base = (batch * value_heads + value_head) * sequence + token_start
            queries = wp.tile_load(
                query,
                shape=(_CHUNK, KEY_SIZE),
                offset=(key_base, 0),
            )
            keys = wp.tile_load(
                key,
                shape=(_CHUNK, KEY_SIZE),
                offset=(key_base, 0),
            )
            values = wp.tile_load(
                value,
                shape=(_CHUNK, VALUE_TILE),
                offset=(value_base, value_offset),
            )
            state_storage = wp.tile_map(to_storage, state)
            retrieved = wp.tile_zeros(shape=(_CHUNK, VALUE_TILE), dtype=wp.float32)
            base_output = wp.tile_zeros(shape=(_CHUNK, VALUE_TILE), dtype=wp.float32)
            key_products = wp.tile_zeros(shape=(_CHUNK, _CHUNK), dtype=wp.float32)
            query_products = wp.tile_zeros(shape=(_CHUNK, _CHUNK), dtype=wp.float32)
            wp.tile_matmul(keys, state_storage, retrieved)
            wp.tile_matmul(queries, state_storage, base_output)
            wp.tile_matmul(keys, wp.tile_transpose(keys), key_products)
            wp.tile_matmul(queries, wp.tile_transpose(keys), query_products)
            deltas = wp.tile_zeros(shape=(_CHUNK, VALUE_TILE), dtype=DTYPE)
            outputs = wp.tile_zeros(shape=(_CHUNK, VALUE_TILE), dtype=wp.float32)
            weighted = wp.tile_zeros(shape=(_CHUNK, VALUE_TILE), dtype=DTYPE)
            length = wp.clamp(lengths[batch], 0, sequence)

            for row in range(_CHUNK):
                token = token_start + row
                if token < sequence and token < length:
                    log_value = local_log_decay[batch, value_head, token]
                    beta_value = beta[batch, token, value_head]
                    value_row = wp.tile_view(
                        values, offset=(row, 0), shape=(1, VALUE_TILE)
                    )
                    retrieved_row = wp.tile_view(
                        retrieved, offset=(row, 0), shape=(1, VALUE_TILE)
                    )
                    delta_row = beta_value * (
                        wp.tile_map(to_fp32, value_row)
                        - wp.exp(log_value) * retrieved_row
                    )
                    for prior in range(row):
                        prior_log = local_log_decay[
                            batch, value_head, token_start + prior
                        ]
                        coefficient = (
                            beta_value
                            * wp.tile_extract(key_products, row, prior)
                            * wp.exp(log_value - prior_log)
                        )
                        prior_delta = wp.tile_view(
                            deltas,
                            offset=(prior, 0),
                            shape=(1, VALUE_TILE),
                        )
                        delta_row -= coefficient * wp.tile_map(to_fp32, prior_delta)
                    stored_delta = wp.tile_map(to_storage, delta_row)
                    wp.tile_assign(deltas, stored_delta, offset=(row, 0))
                    output_row = wp.exp(log_value) * wp.tile_view(
                        base_output,
                        offset=(row, 0),
                        shape=(1, VALUE_TILE),
                    )
                    for prior in range(row + 1):
                        prior_log = local_log_decay[
                            batch, value_head, token_start + prior
                        ]
                        coefficient = wp.tile_extract(
                            query_products, row, prior
                        ) * wp.exp(log_value - prior_log)
                        prior_delta = wp.tile_view(
                            deltas,
                            offset=(prior, 0),
                            shape=(1, VALUE_TILE),
                        )
                        output_row += coefficient * wp.tile_map(to_fp32, prior_delta)
                    wp.tile_assign(outputs, output_row, offset=(row, 0))

            valid_end = wp.min(length, wp.min(sequence, token_start + _CHUNK))
            end_log = wp.float32(0.0)
            if valid_end > token_start:
                end_log = local_log_decay[batch, value_head, valid_end - 1]
            for row in range(_CHUNK):
                token = token_start + row
                if token < sequence and token < length:
                    factor = wp.exp(end_log - local_log_decay[batch, value_head, token])
                    wp.tile_assign(
                        weighted,
                        wp.tile_map(
                            scale_row,
                            wp.tile_view(
                                deltas,
                                offset=(row, 0),
                                shape=(1, VALUE_TILE),
                            ),
                            factor,
                        ),
                        offset=(row, 0),
                    )
            state *= wp.exp(end_log)
            wp.tile_matmul(wp.tile_transpose(keys), weighted, state)
            wp.tile_store(
                transformed,
                deltas,
                offset=(value_base, value_offset),
            )
            wp.tile_store(
                output,
                wp.tile_map(scale_storage, outputs, scale),
                offset=(value_base, value_offset),
            )

        wp.tile_store(present, state, offset=(state_base, value_offset))

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def materialize_states(
        key: wp.array2d(dtype=DTYPE),
        decay: wp.array3d(dtype=wp.float32),
        lengths: wp.array1d(dtype=wp.int32),
        transformed: wp.array2d(dtype=DTYPE),
        checkpoints: wp.array2d(dtype=wp.float32),
        states: wp.array2d(dtype=wp.float32),
        key_heads: wp.int32,
        value_heads: wp.int32,
        sequence: wp.int32,
        chunks: wp.int32,
    ):
        item = wp.tid()
        value_tiles = VALUE_SIZE // MATERIALIZE_TILE
        value_tile_index = item % value_tiles
        state_item = item // value_tiles
        value_head = state_item % value_heads
        batch = state_item // value_heads
        key_head = value_head * key_heads // value_heads
        value_offset = value_tile_index * MATERIALIZE_TILE
        length = wp.clamp(lengths[batch], 0, sequence)
        for chunk in range(chunks):
            checkpoint_base = (
                (batch * value_heads + value_head) * chunks + chunk
            ) * KEY_SIZE
            state = wp.tile_load(
                checkpoints,
                shape=(KEY_SIZE, MATERIALIZE_TILE),
                offset=(checkpoint_base, value_offset),
            )
            for row in range(_CHUNK):
                token = chunk * _CHUNK + row
                if token < length:
                    state_base = (
                        (batch * value_heads + value_head) * sequence + token
                    ) * KEY_SIZE
                    wp.tile_store(states, state, offset=(state_base, value_offset))
                    key_row = (batch * key_heads + key_head) * sequence + token
                    value_row = (batch * value_heads + value_head) * sequence + token
                    keys = wp.tile_load(key, shape=(1, KEY_SIZE), offset=(key_row, 0))
                    deltas = wp.tile_load(
                        transformed,
                        shape=(1, MATERIALIZE_TILE),
                        offset=(value_row, value_offset),
                    )
                    state *= decay[batch, token, value_head]
                    wp.tile_matmul(wp.tile_transpose(keys), deltas, state)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def reverse(
        query: wp.array2d(dtype=DTYPE),
        key: wp.array2d(dtype=DTYPE),
        value: wp.array2d(dtype=DTYPE),
        decay: wp.array3d(dtype=wp.float32),
        beta: wp.array3d(dtype=wp.float32),
        lengths: wp.array1d(dtype=wp.int32),
        transformed: wp.array2d(dtype=DTYPE),
        present: wp.array2d(dtype=wp.float32),
        states: wp.array2d(dtype=wp.float32),
        output_grad: wp.array2d(dtype=wp.float32),
        present_grad: wp.array2d(dtype=wp.float32),
        query_grad: wp.array2d(dtype=wp.float32),
        key_grad: wp.array2d(dtype=wp.float32),
        value_grad: wp.array2d(dtype=wp.float32),
        decay_grad: wp.array3d(dtype=wp.float32),
        beta_grad: wp.array3d(dtype=wp.float32),
        past_grad: wp.array2d(dtype=wp.float32),
        key_heads: wp.int32,
        value_heads: wp.int32,
        sequence: wp.int32,
        scale: wp.float32,
        accumulate: wp.bool,
    ):
        item = wp.tid()
        value_tiles = VALUE_SIZE // REVERSE_TILE
        value_tile_index = item % value_tiles
        state_item = item // value_tiles
        value_head = state_item % value_heads
        batch = state_item // value_heads
        key_head = value_head * key_heads // value_heads
        value_offset = value_tile_index * REVERSE_TILE
        state_base = (batch * value_heads + value_head) * KEY_SIZE
        state = wp.tile_load(
            present,
            shape=(KEY_SIZE, REVERSE_TILE),
            offset=(state_base, value_offset),
        )
        state_gradient = wp.tile_load(
            present_grad,
            shape=(KEY_SIZE, REVERSE_TILE),
            offset=(state_base, value_offset),
        )
        length = wp.clamp(lengths[batch], 0, sequence)

        for reverse_token in range(sequence):
            token = sequence - 1 - reverse_token
            if token < length:
                key_row = (batch * key_heads + key_head) * sequence + token
                value_row = (batch * value_heads + value_head) * sequence + token
                queries = wp.tile_load(query, shape=(1, KEY_SIZE), offset=(key_row, 0))
                keys = wp.tile_load(key, shape=(1, KEY_SIZE), offset=(key_row, 0))
                values = wp.tile_load(
                    value,
                    shape=(1, REVERSE_TILE),
                    offset=(value_row, value_offset),
                )
                deltas = wp.tile_load(
                    transformed,
                    shape=(1, REVERSE_TILE),
                    offset=(value_row, value_offset),
                )
                output_gradient = wp.tile_load(
                    output_grad,
                    shape=(1, REVERSE_TILE),
                    offset=(value_row, value_offset),
                )
                output_gradient_storage = wp.tile_map(to_storage, output_gradient)
                state_storage = wp.tile_map(to_storage, state)
                query_partial = wp.tile_zeros(shape=(KEY_SIZE, 1), dtype=wp.float32)
                wp.tile_matmul(
                    state_storage,
                    wp.tile_transpose(output_gradient_storage),
                    query_partial,
                    alpha=scale,
                )
                wp.tile_atomic_add(
                    query_grad,
                    wp.tile_transpose(query_partial),
                    offset=(key_row, 0),
                )
                wp.tile_matmul(
                    wp.tile_transpose(queries),
                    output_gradient_storage,
                    state_gradient,
                    alpha=scale,
                )
                decay_value = wp.max(
                    decay[batch, token, value_head], wp.float32(1.0e-30)
                )
                token_state_base = (
                    (batch * value_heads + value_head) * sequence + token
                ) * KEY_SIZE
                state = wp.tile_load(
                    states,
                    shape=(KEY_SIZE, REVERSE_TILE),
                    offset=(token_state_base, value_offset),
                )
                decay_contribution = wp.tile_extract(
                    wp.tile_sum(state_gradient * state), 0
                )
                gradient_storage = wp.tile_map(to_storage, state_gradient)
                key_partial = wp.tile_zeros(shape=(KEY_SIZE, 1), dtype=wp.float32)
                delta_gradient = wp.tile_zeros(
                    shape=(1, REVERSE_TILE), dtype=wp.float32
                )
                wp.tile_matmul(gradient_storage, wp.tile_transpose(deltas), key_partial)
                wp.tile_matmul(keys, gradient_storage, delta_gradient)
                state_storage = wp.tile_map(to_storage, state)
                retrieved = wp.tile_zeros(shape=(1, REVERSE_TILE), dtype=wp.float32)
                wp.tile_matmul(keys, state_storage, retrieved)
                beta_value = beta[batch, token, value_head]
                residual = wp.tile_map(to_fp32, values) - decay_value * retrieved
                beta_contribution = wp.tile_extract(
                    wp.tile_sum(delta_gradient * residual), 0
                )
                value_gradient = beta_value * delta_gradient
                decay_contribution += wp.tile_extract(
                    wp.tile_sum(delta_gradient * (-beta_value * retrieved)), 0
                )
                retrieved_gradient = -beta_value * decay_value * delta_gradient
                retrieved_gradient_storage = wp.tile_map(to_storage, retrieved_gradient)
                wp.tile_matmul(
                    state_storage,
                    wp.tile_transpose(retrieved_gradient_storage),
                    key_partial,
                )
                wp.tile_atomic_add(
                    key_grad,
                    wp.tile_transpose(key_partial),
                    offset=(key_row, 0),
                )
                state_gradient *= decay_value
                wp.tile_matmul(
                    wp.tile_transpose(keys),
                    retrieved_gradient_storage,
                    state_gradient,
                )
                if accumulate:
                    value_gradient += wp.tile_load(
                        value_grad,
                        shape=(1, REVERSE_TILE),
                        offset=(value_row, value_offset),
                    )
                wp.tile_store(
                    value_grad, value_gradient, offset=(value_row, value_offset)
                )
                wp.tile_atomic_add(
                    decay_grad,
                    wp.tile_full(
                        shape=(1, 1, 1),
                        value=decay_contribution,
                        dtype=wp.float32,
                    ),
                    offset=(batch, token, value_head),
                )
                wp.tile_atomic_add(
                    beta_grad,
                    wp.tile_full(
                        shape=(1, 1, 1),
                        value=beta_contribution,
                        dtype=wp.float32,
                    ),
                    offset=(batch, token, value_head),
                )

        if accumulate:
            state_gradient += wp.tile_load(
                past_grad,
                shape=(KEY_SIZE, REVERSE_TILE),
                offset=(state_base, value_offset),
            )
        wp.tile_store(past_grad, state_gradient, offset=(state_base, value_offset))

    result = _RuleKernels(log_decay, scalar, chunkwise, materialize_states, reverse)
    for kernel in result.__dict__.values():
        kernel.module.options["enable_backward"] = False
    return result


class GatedDeltaRulePlan:
    """Exact fixed-buffer scalar Gated Delta rule with a CUDA chunkwise fast path."""

    def __init__(
        self,
        batch: int,
        sequence: int,
        key_heads: int,
        value_heads: int,
        key_size: int,
        value_size: int,
        dtype: type,
        *,
        scale: float | None = None,
        state_workspace: wp.array | None = None,
        device=None,
    ):
        if min(batch, sequence, key_heads, value_heads, key_size, value_size) <= 0:
            raise ValueError("Gated Delta rule dimensions must be positive")
        if value_heads % key_heads:
            raise ValueError("value heads must be divisible by key heads")
        if dtype not in _STORAGE_DTYPES:
            raise TypeError("Gated Delta storage must use FP32, FP16, or BF16")
        self.batch, self.sequence = batch, sequence
        self.key_heads, self.value_heads = key_heads, value_heads
        self.key_size, self.value_size = key_size, value_size
        self.dtype, self.device = dtype, wp.get_device(device)
        self.scale = key_size**-0.5 if scale is None else float(scale)
        if not math.isfinite(self.scale):
            raise ValueError("Gated Delta scale must be finite")
        self.chunks = (sequence + _CHUNK - 1) // _CHUNK
        self.value_tile = min(32, value_size & -value_size)
        self._kernels = _kernels(dtype, key_size, value_size, self.value_tile)
        self.local_log_decay = wp.empty(
            (batch, value_heads, sequence), dtype=wp.float32, device=self.device
        )
        shape = (batch, value_heads, sequence, value_size)
        state_shape = (batch, value_heads, key_size, value_size)
        self.output = wp.empty(shape, dtype=dtype, device=self.device)
        self.transformed = wp.empty(shape, dtype=dtype, device=self.device)
        self.present = wp.empty(state_shape, dtype=wp.float32, device=self.device)
        self.checkpoints = wp.empty(
            (batch * value_heads, self.chunks, key_size, value_size),
            dtype=wp.float32,
            device=self.device,
        )
        workspace_shape = (
            batch * value_heads * sequence * key_size,
            value_size,
        )
        if state_workspace is None:
            state_workspace = wp.empty(
                workspace_shape, dtype=wp.float32, device=self.device
            )
        elif (
            not isinstance(state_workspace, wp.array)
            or state_workspace.shape != workspace_shape
            or state_workspace.dtype != wp.float32
            or state_workspace.device != self.device
            or not state_workspace.is_contiguous
        ):
            raise ValueError(
                "Gated Delta state workspace must be contiguous FP32 "
                f"{workspace_shape} on {self.device}"
            )
        self.state_workspace = state_workspace
        query_shape = (batch, key_heads, sequence, key_size)
        self.query_grad = wp.empty(query_shape, dtype=wp.float32, device=self.device)
        self.key_grad = wp.empty(query_shape, dtype=wp.float32, device=self.device)
        self.value_grad = wp.empty(shape, dtype=wp.float32, device=self.device)
        self.decay_grad = wp.empty(
            decay_shape := (batch, sequence, value_heads),
            dtype=wp.float32,
            device=self.device,
        )
        self.beta_grad = wp.empty(decay_shape, dtype=wp.float32, device=self.device)
        self.past_grad = wp.empty(state_shape, dtype=wp.float32, device=self.device)
        self._terminal_grad = wp.zeros(
            state_shape, dtype=wp.float32, device=self.device
        )

    @property
    def uses_tensor_cores(self) -> bool:
        return (
            self.device.is_cuda
            and self.dtype in _CUDA_DTYPES
            and self.sequence % _CHUNK == 0
            and self.key_size % 16 == 0
            and self.value_size % self.value_tile == 0
            and (self.dtype != wp.bfloat16 or self.device.arch >= 80)
            and (self.dtype != wp.float16 or self.device.arch >= 70)
        )

    def forward(self, query, key, value, decay, beta, lengths, past):
        """Return output and final state while retaining reversible token deltas."""
        self._validate(query, key, value, decay, beta, lengths, past)
        wp.launch(
            self._kernels.log_decay,
            dim=(self.batch, self.value_heads, self.chunks),
            inputs=[decay, lengths],
            outputs=[self.local_log_decay],
            device=self.device,
        )
        if self.uses_tensor_cores:
            value_tiles = self.value_size // self.value_tile
            wp.launch_tiled(
                self._kernels.chunkwise,
                dim=self.batch * self.value_heads * value_tiles,
                inputs=[
                    query.reshape(
                        (self.batch * self.key_heads * self.sequence, self.key_size)
                    ),
                    key.reshape(
                        (self.batch * self.key_heads * self.sequence, self.key_size)
                    ),
                    value.reshape(
                        (self.batch * self.value_heads * self.sequence, self.value_size)
                    ),
                    self.local_log_decay,
                    beta,
                    lengths,
                    past.reshape(
                        (self.batch * self.value_heads * self.key_size, self.value_size)
                    ),
                    self.output.reshape(
                        (self.batch * self.value_heads * self.sequence, self.value_size)
                    ),
                    self.transformed.reshape(
                        (self.batch * self.value_heads * self.sequence, self.value_size)
                    ),
                    self.present.reshape(
                        (self.batch * self.value_heads * self.key_size, self.value_size)
                    ),
                    self.checkpoints.reshape(
                        (
                            self.batch * self.value_heads * self.chunks * self.key_size,
                            self.value_size,
                        )
                    ),
                    self.key_heads,
                    self.value_heads,
                    self.sequence,
                    self.chunks,
                    self.scale,
                ],
                block_dim=128,
                device=self.device,
            )
        else:
            wp.copy(self.present, past)
            wp.launch(
                self._kernels.scalar,
                dim=(self.batch, self.value_heads, self.value_size),
                inputs=[
                    query,
                    key,
                    value,
                    decay,
                    beta,
                    lengths,
                    self.present,
                    self.output,
                    self.transformed,
                    self.present,
                    self.checkpoints,
                    self.scale,
                ],
                device=self.device,
            )
        return self.output, self.present

    def backward(
        self,
        query,
        key,
        value,
        decay,
        beta,
        lengths,
        past,
        output_grad,
        *,
        present_grad=None,
        accumulate: bool = False,
    ):
        """Reverse the recurrence into FP32 Q/K/V, gate, and past-state gradients."""
        self._validate(query, key, value, decay, beta, lengths, past)
        value_shape = (self.batch, self.value_heads, self.sequence, self.value_size)
        state_shape = (self.batch, self.value_heads, self.key_size, self.value_size)
        self._validate_gradient(output_grad, value_shape, "output")
        if present_grad is None:
            self._terminal_grad.zero_()
            present_grad = self._terminal_grad
        else:
            self._validate_gradient(present_grad, state_shape, "present state")
        if not accumulate:
            self.query_grad.zero_()
            self.key_grad.zero_()
            self.value_grad.zero_()
            self.decay_grad.zero_()
            self.beta_grad.zero_()
            self.past_grad.zero_()

        query_rows = self.batch * self.key_heads * self.sequence
        value_rows = self.batch * self.value_heads * self.sequence
        state_rows = self.batch * self.value_heads * self.key_size
        value_tiles = self.value_size // min(16, self.value_tile)
        materialize_tiles = self.value_size // min(8, self.value_tile)
        wp.launch_tiled(
            self._kernels.materialize_states,
            dim=self.batch * self.value_heads * materialize_tiles,
            inputs=[
                key.reshape((query_rows, self.key_size)),
                decay,
                lengths,
                self.transformed.reshape((value_rows, self.value_size)),
                self.checkpoints.reshape(
                    (
                        self.batch * self.value_heads * self.chunks * self.key_size,
                        self.value_size,
                    )
                ),
                self.state_workspace,
                self.key_heads,
                self.value_heads,
                self.sequence,
                self.chunks,
            ],
            block_dim=128,
            device=self.device,
        )
        wp.launch_tiled(
            self._kernels.reverse,
            dim=self.batch * self.value_heads * value_tiles,
            inputs=[
                query.reshape((query_rows, self.key_size)),
                key.reshape((query_rows, self.key_size)),
                value.reshape((value_rows, self.value_size)),
                decay,
                beta,
                lengths,
                self.transformed.reshape((value_rows, self.value_size)),
                self.present.reshape((state_rows, self.value_size)),
                self.state_workspace,
                output_grad.reshape((value_rows, self.value_size)),
                present_grad.reshape((state_rows, self.value_size)),
                self.query_grad.reshape((query_rows, self.key_size)),
                self.key_grad.reshape((query_rows, self.key_size)),
                self.value_grad.reshape((value_rows, self.value_size)),
                self.decay_grad,
                self.beta_grad,
                self.past_grad.reshape((state_rows, self.value_size)),
                self.key_heads,
                self.value_heads,
                self.sequence,
                self.scale,
                accumulate,
            ],
            block_dim=128,
            device=self.device,
        )
        return (
            self.query_grad,
            self.key_grad,
            self.value_grad,
            self.decay_grad,
            self.beta_grad,
            self.past_grad,
        )

    def _validate_gradient(self, array, shape, name):
        if (
            not isinstance(array, wp.array)
            or array.shape != shape
            or array.dtype != wp.float32
            or array.device != self.device
            or not array.is_contiguous
        ):
            raise ValueError(
                f"Gated Delta {name} gradient must be contiguous FP32 {shape} "
                f"on {self.device}"
            )

    def _validate(self, query, key, value, decay, beta, lengths, past):
        expected = (
            (
                query,
                (self.batch, self.key_heads, self.sequence, self.key_size),
                self.dtype,
            ),
            (
                key,
                (self.batch, self.key_heads, self.sequence, self.key_size),
                self.dtype,
            ),
            (
                value,
                (self.batch, self.value_heads, self.sequence, self.value_size),
                self.dtype,
            ),
            (decay, (self.batch, self.sequence, self.value_heads), wp.float32),
            (beta, (self.batch, self.sequence, self.value_heads), wp.float32),
            (lengths, (self.batch,), wp.int32),
            (
                past,
                (self.batch, self.value_heads, self.key_size, self.value_size),
                wp.float32,
            ),
        )
        for array, shape, dtype in expected:
            if (
                not isinstance(array, wp.array)
                or array.shape != shape
                or array.dtype != dtype
                or array.device != self.device
                or not array.is_contiguous
            ):
                raise ValueError(
                    f"Gated Delta rule input must be contiguous {shape} {dtype} on {self.device}"
                )
