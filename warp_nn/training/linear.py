# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-free BF16/FP16 Linear and LoRA training primitives.

All launch functions overwrite caller-owned output, gradient, and scratch arrays.
The kernels accumulate dot products in FP32, independently of storage dtype.
"""

from dataclasses import dataclass
from functools import lru_cache

import warp as wp

from warp_nn.runtime._cuda import (
    get_prefill_mma_projection,
    get_prefill_mma_split_k_projection,
)


_STORAGE_DTYPES = (wp.float16, wp.bfloat16)


@dataclass(frozen=True)
class _LinearKernels:
    forward: object
    grad_input: object
    grad_weight: object
    lora_down: object
    lora_output: object
    lora_grad_down: object
    lora_grad_input: object
    lora_grad_a: object
    lora_grad_b: object


@lru_cache(maxsize=None)
def _get_linear_kernels(dtype: type) -> _LinearKernels:
    """Create one concrete kernel family for a low-precision storage dtype."""
    if dtype not in _STORAGE_DTYPES:
        raise TypeError("training Linear supports FP16 and BF16 storage")
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def forward(
        x: wp.array2d(dtype=DTYPE),
        weight: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        total = wp.float32(0.0)
        for inner in range(x.shape[1]):
            total += wp.float32(x[row, inner]) * wp.float32(weight[column, inner])
        output[row, column] = DTYPE(total)

    @wp.kernel(enable_backward=False, module="unique")
    def grad_input(
        grad_output: wp.array2d(dtype=DTYPE),
        weight: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, inner = wp.tid()
        total = wp.float32(0.0)
        for column in range(weight.shape[0]):
            total += wp.float32(grad_output[row, column]) * wp.float32(
                weight[column, inner]
            )
        output[row, inner] = DTYPE(total)

    @wp.kernel(enable_backward=False, module="unique")
    def grad_weight(
        x: wp.array2d(dtype=DTYPE),
        grad_output: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=wp.float32),
        accumulate: bool,
    ):
        column, inner = wp.tid()
        total = wp.float32(0.0)
        for row in range(x.shape[0]):
            total += wp.float32(grad_output[row, column]) * wp.float32(x[row, inner])
        if accumulate:
            total += output[column, inner]
        output[column, inner] = total

    @wp.kernel(enable_backward=False, module="unique")
    def lora_down(
        x: wp.array2d(dtype=DTYPE),
        lora_a: wp.array2d(dtype=DTYPE),
        hidden: wp.array2d(dtype=wp.float32),
    ):
        row, rank = wp.tid()
        total = wp.float32(0.0)
        for inner in range(x.shape[1]):
            total += wp.float32(x[row, inner]) * wp.float32(lora_a[rank, inner])
        hidden[row, rank] = total

    @wp.kernel(enable_backward=False, module="unique")
    def lora_output(
        lora_b: wp.array2d(dtype=DTYPE),
        hidden: wp.array2d(dtype=wp.float32),
        scale: wp.float32,
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        adapter = wp.float32(0.0)
        for rank in range(hidden.shape[1]):
            adapter += hidden[row, rank] * wp.float32(lora_b[column, rank])
        output[row, column] = DTYPE(wp.float32(output[row, column]) + scale * adapter)

    @wp.kernel(enable_backward=False, module="unique")
    def lora_grad_down(
        grad_output: wp.array2d(dtype=DTYPE),
        lora_b: wp.array2d(dtype=DTYPE),
        scale: wp.float32,
        output: wp.array2d(dtype=wp.float32),
    ):
        row, rank = wp.tid()
        total = wp.float32(0.0)
        for column in range(grad_output.shape[1]):
            total += wp.float32(grad_output[row, column]) * wp.float32(
                lora_b[column, rank]
            )
        output[row, rank] = scale * total

    @wp.kernel(enable_backward=False, module="unique")
    def lora_grad_input(
        grad_hidden: wp.array2d(dtype=wp.float32),
        lora_a: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, inner = wp.tid()
        total = wp.float32(output[row, inner])
        for rank in range(lora_a.shape[0]):
            total += grad_hidden[row, rank] * wp.float32(lora_a[rank, inner])
        output[row, inner] = DTYPE(total)

    @wp.kernel(enable_backward=False, module="unique")
    def lora_grad_a(
        x: wp.array2d(dtype=DTYPE),
        grad_hidden: wp.array2d(dtype=wp.float32),
        output: wp.array2d(dtype=wp.float32),
        accumulate: bool,
    ):
        rank, inner = wp.tid()
        total = wp.float32(0.0)
        for row in range(x.shape[0]):
            total += grad_hidden[row, rank] * wp.float32(x[row, inner])
        if accumulate:
            total += output[rank, inner]
        output[rank, inner] = total

    @wp.kernel(enable_backward=False, module="unique")
    def lora_grad_b(
        grad_output: wp.array2d(dtype=DTYPE),
        hidden: wp.array2d(dtype=wp.float32),
        scale: wp.float32,
        output: wp.array2d(dtype=wp.float32),
        accumulate: bool,
    ):
        column, rank = wp.tid()
        total = wp.float32(0.0)
        for row in range(grad_output.shape[0]):
            total += wp.float32(grad_output[row, column]) * hidden[row, rank]
        total *= scale
        if accumulate:
            total += output[column, rank]
        output[column, rank] = total

    return _LinearKernels(
        forward=forward,
        grad_input=grad_input,
        grad_weight=grad_weight,
        lora_down=lora_down,
        lora_output=lora_output,
        lora_grad_down=lora_grad_down,
        lora_grad_input=lora_grad_input,
        lora_grad_a=lora_grad_a,
        lora_grad_b=lora_grad_b,
    )


@dataclass(frozen=True)
class _SplitKLoRAKernels:
    transposed_right: object
    regular_right: object
    reduce: object


_TILE_M = 32
_TILE_N = 32
_TILE_K = 32
_TILE_BLOCK_DIM = 128


@lru_cache(maxsize=None)
def _get_tiled_gemm_kernel(
    dtype: type,
    output_dtype: type,
    *,
    transposed_right: bool,
    scale_output: bool,
):
    """Create one boundary-masked tensor-core GEMM specialization."""
    if dtype not in _STORAGE_DTYPES or output_dtype not in (
        dtype,
        wp.float32,
    ):
        raise TypeError("tiled GEMM requires FP16/BF16 input and matching/FP32 output")
    DTYPE = dtype
    OUTPUT_DTYPE = output_dtype
    TRANSPOSE_RIGHT = transposed_right
    SCALE_OUTPUT = scale_output
    FP32_OUTPUT = output_dtype == wp.float32

    @wp.func
    def cast_output(value: wp.float32):
        return OUTPUT_DTYPE(value)

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        left: wp.array2d(dtype=DTYPE),
        right: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=OUTPUT_DTYPE),
        reduction: int,
        scale: wp.float32,
    ):
        tile_row, tile_column = wp.tid()
        accumulator = wp.tile_zeros(shape=(_TILE_M, _TILE_N), dtype=wp.float32)
        for inner_tile in range((reduction + _TILE_K - 1) / _TILE_K):
            inner_offset = inner_tile * _TILE_K
            left_tile = wp.tile_load(
                left,
                shape=(_TILE_M, _TILE_K),
                offset=(tile_row * _TILE_M, inner_offset),
            )
            if wp.static(TRANSPOSE_RIGHT):
                right_tile = wp.tile_transpose(
                    wp.tile_load(
                        right,
                        shape=(_TILE_N, _TILE_K),
                        offset=(tile_column * _TILE_N, inner_offset),
                    )
                )
            else:
                right_tile = wp.tile_load(
                    right,
                    shape=(_TILE_K, _TILE_N),
                    offset=(inner_offset, tile_column * _TILE_N),
                )
            wp.tile_matmul(left_tile, right_tile, accumulator)
        if wp.static(SCALE_OUTPUT):
            accumulator *= scale
        if wp.static(FP32_OUTPUT):
            wp.tile_store(
                output,
                accumulator,
                offset=(tile_row * _TILE_M, tile_column * _TILE_N),
            )
        else:
            wp.tile_store(
                output,
                wp.tile_map(cast_output, accumulator),
                offset=(tile_row * _TILE_M, tile_column * _TILE_N),
            )

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_native_linear_kernel(
    dtype: type,
    tile_m: int,
    tile_n: int,
    *,
    transposed_right: bool = True,
    split_k: bool = False,
    stage_k: int = 32,
):
    """Wrap the shared SM80+ MMA pipeline for direct or split-K output."""
    DTYPE = dtype
    SPLIT_K = split_k
    OUTPUT_DTYPE = wp.float32 if split_k else dtype
    project = (
        get_prefill_mma_split_k_projection(
            dtype, tile_m, tile_n, transposed_right=transposed_right
        )
        if split_k
        else get_prefill_mma_projection(
            dtype,
            tile_m,
            tile_n,
            transposed_right=transposed_right,
            stage_k=stage_k,
        )
    )

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        x: wp.array2d(dtype=DTYPE),
        weight: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=OUTPUT_DTYPE),
        rows: int,
        columns: int,
        inner: int,
        splits: int,
    ):
        typed_zero = DTYPE(0.0)  # noqa: F841 - retain dtype in the Warp closure
        if wp.static(SPLIT_K):
            wp.static(project)(
                x, weight, output, wp.tid(), rows, columns, inner, splits
            )
        else:
            wp.static(project)(x, weight, output, wp.tid(), columns, inner)

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_native_split_k_combine_kernel(dtype: type):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        partial: wp.array2d(dtype=wp.float32),
        output: wp.array2d(dtype=DTYPE),
        splits: int,
    ):
        row, column = wp.tid()
        total = wp.float32(0.0)
        for split in range(splits):
            total += partial[split * output.shape[0] + row, column]
        output[row, column] = DTYPE(total)

    kernel.module.options["enable_backward"] = False
    return kernel


@lru_cache(maxsize=None)
def _get_tiled_weight_gradient_kernel(dtype: type):
    """Create the distinct transposed-output FP32 weight-gradient GEMM."""
    if dtype not in _STORAGE_DTYPES:
        raise TypeError("tiled training Linear supports FP16 and BF16 storage")
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def grad_weight(
        grad_output: wp.array2d(dtype=DTYPE),
        x: wp.array2d(dtype=DTYPE),
        output: wp.array2d(dtype=wp.float32),
        reduction: int,
        accumulate: bool,
    ):
        tile_row, tile_column = wp.tid()
        accumulator = wp.tile_zeros(shape=(_TILE_M, _TILE_N), dtype=wp.float32)
        for inner_tile in range((reduction + _TILE_K - 1) / _TILE_K):
            inner_offset = inner_tile * _TILE_K
            left_tile = wp.tile_transpose(
                wp.tile_load(
                    grad_output,
                    shape=(_TILE_K, _TILE_M),
                    offset=(inner_offset, tile_row * _TILE_M),
                )
            )
            right_tile = wp.tile_load(
                x,
                shape=(_TILE_K, _TILE_N),
                offset=(inner_offset, tile_column * _TILE_N),
            )
            wp.tile_matmul(left_tile, right_tile, accumulator)
        if accumulate:
            accumulator += wp.tile_load(
                output,
                shape=(_TILE_M, _TILE_N),
                offset=(tile_row * _TILE_M, tile_column * _TILE_N),
            )
        wp.tile_store(
            output,
            accumulator,
            offset=(tile_row * _TILE_M, tile_column * _TILE_N),
        )

    grad_weight.module.options["enable_backward"] = False
    return grad_weight


@lru_cache(maxsize=None)
def _get_split_k_lora_kernels(dtype: type) -> _SplitKLoRAKernels:
    """Create a split-K tensor-core kernel for skinny LoRA products."""
    if dtype not in _STORAGE_DTYPES:
        raise TypeError("split-K LoRA supports FP16 and BF16 storage")
    DTYPE = dtype

    def create_kernel(transpose_right: bool):
        TRANSPOSE_RIGHT = transpose_right

        @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
        def kernel(
            left: wp.array2d(dtype=DTYPE),
            right: wp.array2d(dtype=DTYPE),
            partial: wp.array2d(dtype=wp.float32),
            reduction: int,
            splits: int,
            rows: int,
        ):
            split, tile_row, tile_column = wp.tid()
            accumulator = wp.tile_zeros(shape=(_TILE_M, _TILE_N), dtype=wp.float32)
            inner_tiles = (reduction + _TILE_K - 1) / _TILE_K
            tiles_per_split = (inner_tiles + splits - 1) / splits
            for local_tile in range(tiles_per_split):
                inner_tile = split * tiles_per_split + local_tile
                if inner_tile < inner_tiles:
                    inner_offset = inner_tile * _TILE_K
                    left_tile = wp.tile_load(
                        left,
                        shape=(_TILE_M, _TILE_K),
                        offset=(tile_row * _TILE_M, inner_offset),
                    )
                    if wp.static(TRANSPOSE_RIGHT):
                        right_tile = wp.tile_transpose(
                            wp.tile_load(
                                right,
                                shape=(_TILE_N, _TILE_K),
                                offset=(tile_column * _TILE_N, inner_offset),
                            )
                        )
                    else:
                        right_tile = wp.tile_load(
                            right,
                            shape=(_TILE_K, _TILE_N),
                            offset=(inner_offset, tile_column * _TILE_N),
                        )
                    wp.tile_matmul(left_tile, right_tile, accumulator)
            wp.tile_store(
                partial,
                accumulator,
                offset=(split * rows + tile_row * _TILE_M, tile_column * _TILE_N),
            )

        kernel.module.options["enable_backward"] = False
        return kernel

    @wp.kernel(enable_backward=False, module="unique")
    def reduce(
        partial: wp.array2d(dtype=wp.float32),
        output: wp.array2d(dtype=wp.float32),
        splits: int,
        scale: wp.float32,
    ):
        row, column = wp.tid()
        total = wp.float32(0.0)
        for split in range(splits):
            total += partial[split * output.shape[0] + row, column]
        output[row, column] = scale * total

    reduce.module.options["enable_backward"] = False
    return _SplitKLoRAKernels(create_kernel(True), create_kernel(False), reduce)


def _check_array(
    name: str, value: wp.array, shape: tuple[int, int], dtype: type, device
) -> None:
    if not isinstance(value, wp.array) or value.ndim != 2:
        raise TypeError(f"{name} must be a 2-D Warp array")
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")


def _linear_shapes(x: wp.array, weight: wp.array) -> tuple[int, int, int]:
    if (
        not isinstance(x, wp.array)
        or not isinstance(weight, wp.array)
        or x.ndim != 2
        or weight.ndim != 2
    ):
        raise TypeError("x and weight must be 2-D Warp arrays")
    if x.dtype not in _STORAGE_DTYPES or weight.dtype != x.dtype:
        raise TypeError("x and weight must use matching FP16 or BF16 storage")
    rows, inner = x.shape
    columns, weight_inner = weight.shape
    if inner != weight_inner:
        raise ValueError(f"incompatible Linear shapes {x.shape} and {weight.shape}")
    if weight.device != x.device:
        raise ValueError("x and weight must be on the same device")
    return rows, columns, inner


def _native_linear_geometry(
    rows: int,
    columns: int,
    reduction: int,
    dtype: type,
    device,
    *arrays: wp.array,
) -> tuple[int, int] | None:
    # Select a shared pipelined SM80+ MMA geometry without padding.
    if (
        not device.is_cuda
        or device.arch < 80
        or dtype not in _STORAGE_DTYPES
        or reduction % 32
        or not all(array.is_contiguous and array.ptr % 16 == 0 for array in arrays)
    ):
        return None
    if rows >= 64 and rows % 64 == 0 and columns % 32 == 0:
        return 64, 32
    if rows >= 64 and rows % 64 == 0 and columns % 16 == 0:
        return 64, 16
    if rows >= 16 and rows % 16 == 0 and columns % 64 == 0:
        return 16, 64
    return None


def _native_split_k_geometry(
    rows: int,
    columns: int,
    reduction: int,
    dtype: type,
    device,
    *,
    transposed_right: bool,
) -> tuple[int, int, int] | None:
    """Choose a portable SM80+ split-K geometry from matrix dimensions."""
    if (
        not device.is_cuda
        or device.arch < 80
        or dtype not in _STORAGE_DTYPES
        or rows < 64
        or rows % 64
        or reduction % 128
        or columns % 32
    ):
        return None

    blocks_64 = (rows // 64) * (columns // 64)
    tile_n = (
        64
        if columns % 64 == 0 and (columns <= reduction or blocks_64 >= device.sm_count)
        else 32
    )
    output_blocks = (rows // 64) * (columns // tile_n)
    if output_blocks >= 2 * device.sm_count:
        return 64, tile_n, 1

    splits = 4
    if transposed_right and reduction >= 2 * columns:
        # Long row-major weight rows need enough independent K partitions to
        # cover memory latency; retain at least 512 reduction values per CTA.
        limit = min(32, reduction // 512)
        splits = 1 << (limit.bit_length() - 1)
    elif transposed_right and output_blocks >= device.sm_count:
        splits = 8
    while splits > 1 and reduction % (64 * splits):
        splits //= 2
    return 64, tile_n, splits


def _launch_native_split_k(
    left: wp.array,
    right: wp.array,
    output: wp.array,
    workspace: wp.array | None,
    rows: int,
    columns: int,
    reduction: int,
    *,
    transposed_right: bool,
) -> bool:
    geometry = _native_split_k_geometry(
        rows,
        columns,
        reduction,
        left.dtype,
        left.device,
        transposed_right=transposed_right,
    )
    if geometry is None or geometry[2] == 1 or workspace is None:
        return False
    tile_m, tile_n, splits = geometry
    _check_array(
        "matmul_workspace",
        workspace,
        (splits * rows, columns),
        wp.float32,
        left.device,
    )
    if not all(
        array.is_contiguous and array.ptr % 16 == 0 for array in (left, right, output)
    ):
        return False
    block_dim = tile_m * tile_n // 8
    wp.launch(
        _get_native_linear_kernel(
            left.dtype,
            tile_m,
            tile_n,
            transposed_right=transposed_right,
            split_k=True,
        ),
        dim=(rows // tile_m) * (columns // tile_n) * splits * block_dim,
        inputs=[left, right, workspace, rows, columns, reduction, splits],
        block_dim=block_dim,
        device=left.device,
    )
    wp.launch(
        _get_native_split_k_combine_kernel(left.dtype),
        dim=(rows, columns),
        inputs=[workspace, output, splits],
        device=left.device,
    )
    return True


def _use_tiled(
    rows: int,
    columns: int,
    reduction: int,
    dtype: type,
    device,
    *arrays: wp.array,
) -> bool:
    """Select tensor-core tiles only for substantial contiguous CUDA GEMMs."""
    minimum_arch = 80 if dtype == wp.bfloat16 else 70
    return (
        device.is_cuda
        and device.arch >= minimum_arch
        and rows >= 16
        and columns >= 16
        and reduction >= 16
        and all(array.is_contiguous for array in arrays)
    )


def _use_skinny_tiled(
    rows: int,
    reduction: int,
    dtype: type,
    device,
    *arrays: wp.array,
) -> bool:
    """Use boundary-masked tensor-core tiles even when LoRA rank is below 16."""
    minimum_arch = 80 if dtype == wp.bfloat16 else 70
    return (
        device.is_cuda
        and device.arch >= minimum_arch
        and rows >= 16
        and reduction >= 16
        and all(array.is_contiguous for array in arrays)
    )


def _lora_split_count(rows: int, rank: int, reduction: int, dtype: type, device) -> int:
    """Choose bounded split-K parallelism from geometry and available SMs."""
    minimum_arch = 80 if dtype == wp.bfloat16 else 70
    if not device.is_cuda or device.arch < minimum_arch or rows < 16 or reduction < 16:
        return 1
    output_tiles = ((rows + _TILE_M - 1) // _TILE_M) * ((rank + _TILE_N - 1) // _TILE_N)
    inner_tiles = (reduction + _TILE_K - 1) // _TILE_K
    target_splits = (4 * device.sm_count + output_tiles - 1) // output_tiles
    return min(64, inner_tiles, max(1, target_splits))


def _launch_tiled(
    kernel,
    left: wp.array,
    right: wp.array,
    output: wp.array,
    rows: int,
    columns: int,
    reduction: int,
    *extra_inputs,
) -> None:
    wp.launch_tiled(
        kernel,
        dim=(
            (rows + _TILE_M - 1) // _TILE_M,
            (columns + _TILE_N - 1) // _TILE_N,
        ),
        inputs=[left, right, output, reduction, *extra_inputs],
        block_dim=_TILE_BLOCK_DIM,
        device=left.device,
    )


def _launch_split_k_lora(
    kernels: _SplitKLoRAKernels,
    left: wp.array,
    right: wp.array,
    partial: wp.array,
    output: wp.array,
    rows: int,
    columns: int,
    reduction: int,
    splits: int,
    scale: float,
    transposed_right: bool = False,
) -> None:
    wp.launch_tiled(
        kernels.transposed_right if transposed_right else kernels.regular_right,
        dim=(
            splits,
            (rows + _TILE_M - 1) // _TILE_M,
            (columns + _TILE_N - 1) // _TILE_N,
        ),
        inputs=[left, right, partial, reduction, splits, rows],
        block_dim=_TILE_BLOCK_DIM,
        device=left.device,
    )
    wp.launch(
        kernels.reduce,
        dim=(rows, columns),
        inputs=[partial, output, splits, float(scale)],
        device=left.device,
    )


def linear_forward(
    x: wp.array,
    weight: wp.array,
    output: wp.array,
    *,
    cublas=None,
    matmul_workspace: wp.array | None = None,
) -> None:
    """Launch ``output = x @ weight.T`` into a preallocated low-precision array."""
    rows, columns, inner = _linear_shapes(x, weight)
    _check_array("output", output, (rows, columns), x.dtype, x.device)
    native_geometry = _native_linear_geometry(
        rows, columns, inner, x.dtype, x.device, x, weight, output
    )
    if cublas is not None and x.device.is_cuda:
        cublas.gemm(
            x.ptr,
            weight.ptr,
            output.ptr,
            rows,
            columns,
            inner,
            wp.get_stream(x.device).cuda_stream,
            2 if x.dtype == wp.float16 else 14,
        )
    elif _launch_native_split_k(
        x,
        weight,
        output,
        matmul_workspace,
        rows,
        columns,
        inner,
        transposed_right=True,
    ):
        pass
    elif native_geometry is not None:
        tile_m, tile_n = native_geometry
        block_dim = tile_m * tile_n // 8
        wp.launch(
            _get_native_linear_kernel(
                x.dtype,
                tile_m,
                tile_n,
                stage_k=64 if rows >= 128 and inner % 64 == 0 else 32,
            ),
            dim=(rows // tile_m) * (columns // tile_n) * block_dim,
            inputs=[x, weight, output, rows, columns, inner, 1],
            block_dim=block_dim,
            device=x.device,
        )
    elif _use_tiled(rows, columns, inner, x.dtype, x.device, x, weight, output):
        _launch_tiled(
            _get_tiled_gemm_kernel(
                x.dtype,
                x.dtype,
                transposed_right=True,
                scale_output=False,
            ),
            x,
            weight,
            output,
            rows,
            columns,
            inner,
            1.0,
        )
    else:
        wp.launch(
            _get_linear_kernels(x.dtype).forward,
            dim=(rows, columns),
            inputs=[x, weight],
            outputs=[output],
            device=x.device,
        )


def linear_backward(
    x: wp.array,
    weight: wp.array,
    grad_output: wp.array,
    grad_input: wp.array,
    grad_weight: wp.array | None = None,
    *,
    accumulate: bool = False,
    cublas=None,
    matmul_workspace: wp.array | None = None,
) -> None:
    """Overwrite activation gradients and optionally accumulate FP32 weight gradients."""
    rows, columns, inner = _linear_shapes(x, weight)
    _check_array("grad_output", grad_output, (rows, columns), x.dtype, x.device)
    _check_array("grad_input", grad_input, (rows, inner), x.dtype, x.device)
    if grad_weight is not None:
        _check_array("grad_weight", grad_weight, (columns, inner), wp.float32, x.device)

    if cublas is not None and x.device.is_cuda:
        cublas.gemm_nn(
            grad_output.ptr,
            weight.ptr,
            grad_input.ptr,
            rows,
            inner,
            columns,
            wp.get_stream(x.device).cuda_stream,
            2 if x.dtype == wp.float16 else 14,
        )
    elif _launch_native_split_k(
        grad_output,
        weight,
        grad_input,
        matmul_workspace,
        rows,
        inner,
        columns,
        transposed_right=False,
    ):
        pass
    elif (
        native_geometry := _native_linear_geometry(
            rows,
            inner,
            columns,
            x.dtype,
            x.device,
            grad_output,
            weight,
            grad_input,
        )
    ) is not None:
        tile_m, tile_n = native_geometry
        block_dim = tile_m * tile_n // 8
        wp.launch(
            _get_native_linear_kernel(
                x.dtype,
                tile_m,
                tile_n,
                transposed_right=False,
                stage_k=64 if rows >= 128 and columns % 64 == 0 else 32,
            ),
            dim=(rows // tile_m) * (inner // tile_n) * block_dim,
            inputs=[grad_output, weight, grad_input, rows, inner, columns, 1],
            block_dim=block_dim,
            device=x.device,
        )
    elif _use_tiled(
        rows, inner, columns, x.dtype, x.device, grad_output, weight, grad_input
    ):
        _launch_tiled(
            _get_tiled_gemm_kernel(
                x.dtype,
                x.dtype,
                transposed_right=False,
                scale_output=False,
            ),
            grad_output,
            weight,
            grad_input,
            rows,
            inner,
            columns,
            1.0,
        )
    else:
        wp.launch(
            _get_linear_kernels(x.dtype).grad_input,
            dim=(rows, inner),
            inputs=[grad_output, weight],
            outputs=[grad_input],
            device=x.device,
        )

    if grad_weight is None:
        return
    if _use_tiled(columns, inner, rows, x.dtype, x.device, grad_output, x, grad_weight):
        _launch_tiled(
            _get_tiled_weight_gradient_kernel(x.dtype),
            grad_output,
            x,
            grad_weight,
            columns,
            inner,
            rows,
            bool(accumulate),
        )
    else:
        wp.launch(
            _get_linear_kernels(x.dtype).grad_weight,
            dim=(columns, inner),
            inputs=[x, grad_output, grad_weight, bool(accumulate)],
            device=x.device,
        )


def _lora_shapes(
    x: wp.array, weight: wp.array, lora_a: wp.array, lora_b: wp.array
) -> tuple[int, int, int, int]:
    rows, columns, inner = _linear_shapes(x, weight)
    if not isinstance(lora_a, wp.array) or lora_a.ndim != 2:
        raise TypeError("lora_a must be a 2-D Warp array")
    rank = lora_a.shape[0]
    _check_array("lora_a", lora_a, (rank, inner), x.dtype, x.device)
    _check_array("lora_b", lora_b, (columns, rank), x.dtype, x.device)
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    return rows, columns, inner, rank


def lora_forward(
    x: wp.array,
    weight: wp.array,
    lora_a: wp.array,
    lora_b: wp.array,
    hidden: wp.array,
    output: wp.array,
    scale: float,
    *,
    cublas=None,
    base_matmul_workspace: wp.array | None = None,
    matmul_workspace: wp.array | None = None,
    matmul_splits: int = 1,
) -> None:
    """Launch ``x @ W.T + scale * (x @ A.T) @ B.T``.

    ``hidden`` is caller-owned FP32 storage with shape ``(rows, rank)``. It is
    retained for the backward pass and overwritten on every call.
    """
    rows, columns, _, rank = _lora_shapes(x, weight, lora_a, lora_b)
    _check_array("hidden", hidden, (rows, rank), wp.float32, x.device)
    _check_array("output", output, (rows, columns), x.dtype, x.device)
    kernels = _get_linear_kernels(x.dtype)
    if matmul_splits > 1:
        if (
            not isinstance(matmul_workspace, wp.array)
            or matmul_workspace.ndim != 2
            or matmul_workspace.dtype != wp.float32
            or matmul_workspace.device != x.device
            or not matmul_workspace.is_contiguous
            or matmul_workspace.shape[0] < matmul_splits * rows
            or matmul_workspace.shape[1] != rank
        ):
            raise ValueError("split-K LoRA requires a matching FP32 workspace")
        _launch_split_k_lora(
            _get_split_k_lora_kernels(x.dtype),
            x,
            lora_a,
            matmul_workspace,
            hidden,
            rows,
            rank,
            x.shape[1],
            matmul_splits,
            1.0,
            transposed_right=True,
        )
    elif _use_skinny_tiled(rows, x.shape[1], x.dtype, x.device, x, lora_a, hidden):
        _launch_tiled(
            _get_tiled_gemm_kernel(
                x.dtype,
                wp.float32,
                transposed_right=True,
                scale_output=True,
            ),
            x,
            lora_a,
            hidden,
            rows,
            rank,
            x.shape[1],
            1.0,
        )
    else:
        wp.launch(
            kernels.lora_down,
            dim=(rows, rank),
            inputs=[x, lora_a],
            outputs=[hidden],
            device=x.device,
        )
    linear_forward(
        x,
        weight,
        output,
        cublas=cublas,
        matmul_workspace=base_matmul_workspace,
    )
    wp.launch(
        kernels.lora_output,
        dim=(rows, columns),
        inputs=[lora_b, hidden, float(scale), output],
        device=x.device,
    )


def lora_backward(
    x: wp.array,
    weight: wp.array,
    lora_a: wp.array,
    lora_b: wp.array,
    hidden: wp.array,
    grad_output: wp.array,
    grad_hidden: wp.array,
    grad_input: wp.array,
    grad_a: wp.array,
    grad_b: wp.array,
    scale: float,
    grad_weight: wp.array | None = None,
    *,
    accumulate: bool = False,
    base_matmul_workspace: wp.array | None = None,
    matmul_workspace: wp.array | None = None,
    matmul_splits: int = 1,
    cublas=None,
) -> None:
    """Overwrite Linear+LoRA gradients in preallocated arrays.

    ``grad_hidden``, ``grad_a``, ``grad_b``, and optional ``grad_weight`` use
    FP32 storage. ``grad_input`` follows the activation storage dtype.
    """
    rows, columns, inner, rank = _lora_shapes(x, weight, lora_a, lora_b)
    _check_array("hidden", hidden, (rows, rank), wp.float32, x.device)
    _check_array("grad_output", grad_output, (rows, columns), x.dtype, x.device)
    _check_array("grad_hidden", grad_hidden, (rows, rank), wp.float32, x.device)
    _check_array("grad_input", grad_input, (rows, inner), x.dtype, x.device)
    _check_array("grad_a", grad_a, (rank, inner), wp.float32, x.device)
    _check_array("grad_b", grad_b, (columns, rank), wp.float32, x.device)
    kernels = _get_linear_kernels(x.dtype)
    if matmul_splits < 1:
        raise ValueError("matmul_splits must be positive")
    if matmul_splits > 1:
        if (
            not isinstance(matmul_workspace, wp.array)
            or matmul_workspace.ndim != 2
            or matmul_workspace.dtype != wp.float32
            or matmul_workspace.device != x.device
            or not matmul_workspace.is_contiguous
            or matmul_workspace.shape[0] < matmul_splits * rows
            or matmul_workspace.shape[1] != rank
        ):
            raise ValueError("split-K LoRA requires a matching FP32 workspace")
        _launch_split_k_lora(
            _get_split_k_lora_kernels(x.dtype),
            grad_output,
            lora_b,
            matmul_workspace,
            grad_hidden,
            rows,
            rank,
            columns,
            matmul_splits,
            float(scale),
        )
    elif _use_skinny_tiled(
        rows,
        columns,
        x.dtype,
        x.device,
        grad_output,
        lora_b,
        grad_hidden,
    ):
        _launch_tiled(
            _get_tiled_gemm_kernel(
                x.dtype,
                wp.float32,
                transposed_right=False,
                scale_output=True,
            ),
            grad_output,
            lora_b,
            grad_hidden,
            rows,
            rank,
            columns,
            float(scale),
        )
    else:
        wp.launch(
            kernels.lora_grad_down,
            dim=(rows, rank),
            inputs=[grad_output, lora_b, float(scale)],
            outputs=[grad_hidden],
            device=x.device,
        )
    linear_backward(
        x,
        weight,
        grad_output,
        grad_input,
        grad_weight,
        accumulate=accumulate,
        matmul_workspace=base_matmul_workspace,
        cublas=cublas,
    )
    wp.launch(
        kernels.lora_grad_input,
        dim=(rows, inner),
        inputs=[grad_hidden, lora_a, grad_input],
        device=x.device,
    )
    wp.launch(
        kernels.lora_grad_a,
        dim=(rank, inner),
        inputs=[x, grad_hidden, grad_a, bool(accumulate)],
        device=x.device,
    )
    wp.launch(
        kernels.lora_grad_b,
        dim=(columns, rank),
        inputs=[grad_output, hidden, float(scale), grad_b, bool(accumulate)],
        device=x.device,
    )
