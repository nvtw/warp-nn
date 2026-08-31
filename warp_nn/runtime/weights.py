# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Small format-neutral helpers for streaming inference-weight conversion."""

import math

import warp as wp

from warp_nn.runtime.kernels import _cast_kernel_for_dtypes, _merge_lora_kernel


class MappedWeightArchive:
    """Expose an archive through canonical runtime weight names."""

    def __init__(self, archive, names, metadata=None):
        self.archive = archive
        self._names = dict(names)
        self._metadata = metadata or archive.metadata

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._names)

    def metadata(self, name: str):
        return self._metadata(self._names[name])

    def load(self, device=None, names=None) -> dict[str, wp.array]:
        selected = self.names if names is None else tuple(names)
        sources = [self._names[name] for name in selected]
        loaded = self.archive.load(device, sources)
        return {name: loaded[self._names[name]] for name in selected}


def merge_lora_weight(weight, a, b, scale):
    """Merge one low-rank update into a caller-owned inference weight."""
    if (
        weight.ndim != 2
        or a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != weight.shape[1]
        or b.shape != (weight.shape[0], a.shape[0])
    ):
        raise ValueError("LoRA A and B shapes do not match the base weight")
    if a.dtype != weight.dtype or b.dtype != weight.dtype:
        raise TypeError("LoRA and base weight dtypes must match")
    if a.device != weight.device or b.device != weight.device:
        raise ValueError("LoRA and base weight devices must match")
    if not math.isfinite(scale):
        raise ValueError("LoRA scale must be finite")
    wp.launch(
        _merge_lora_kernel(weight.dtype),
        dim=weight.shape,
        inputs=[weight, a, b, wp.float32(scale)],
        device=weight.device,
    )


def load_cast_weights(archive, names, device, dtype=None):
    """Load selected tensors one at a time and optionally cast floating weights.

    The sequential lifecycle bounds peak memory to final weights plus one source
    tensor.  It works with every archive exposing the shared ``metadata/load``
    interface and never creates a CPU numeric conversion copy.
    """
    device = wp.get_device(device)
    names = tuple(dict.fromkeys(names))
    if dtype is None:
        return archive.load(device, names)
    if dtype not in (wp.float16, wp.bfloat16, wp.float32):
        raise TypeError("weight conversion target must be FP16, BF16, or FP32")
    output = {}
    for name in sorted(
        names, key=lambda item: archive.metadata(item).nbytes, reverse=True
    ):
        source = archive.load(device, [name])[name]
        if source.dtype == dtype:
            output[name] = source
            continue
        if source.dtype not in (wp.float16, wp.bfloat16, wp.float32):
            raise TypeError(f"cannot cast non-floating weight '{name}'")
        converted = wp.empty(source.shape, dtype=dtype, device=device)
        wp.launch(
            _cast_kernel_for_dtypes(source.dtype, dtype),
            dim=source.size,
            inputs=[source.flatten(), converted.flatten()],
            device=device,
        )
        if device.is_cuda:
            wp.synchronize_stream(wp.get_stream(device))
        output[name] = converted
    return output
