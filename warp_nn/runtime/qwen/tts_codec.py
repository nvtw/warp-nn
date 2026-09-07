# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Qwen3-TTS 12 Hz speech-tokenizer decoding.

The public code layout is ``[frames, 16]``.  Internally all temporal operators
use Warp's shared channels-last ``[batch, frames, channels]`` convention.  One
codec frame produces exactly 1,920 mono samples at 24 kHz.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import warp as wp

from ...utils.device import parse_device
from .._cublas import try_create_cublas
from ..formats.safetensors import SafeTensorArchive, SafeTensorNamespace
from ..kernels import (
    _add_arrays_kernel,
    _encoder_kernels,
    _gather_sum_embeddings_kernel,
)
from ..operators import (
    BiasedLinearPlan,
    ClampPlan,
    Conv1dPlan,
    LayerNormPlan,
    Operation,
    Snake1dPlan,
    execute_operations,
    plan_linear,
    rotary_cache_values,
)
from ..weights import load_cast_weights
from .encoder import _Qwen3EncoderPlan


@lru_cache(maxsize=None)
def _codec_kernels(dtype):
    DTYPE = dtype

    @wp.kernel(enable_backward=False, module="unique")
    def normalize_codebook(
        embedding_sum: wp.array2d(dtype=DTYPE),
        usage: wp.array1d(dtype=DTYPE),
        output: wp.array2d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        denominator = wp.max(wp.float32(usage[row]), wp.float32(1.0e-5))
        output[row, column] = DTYPE(
            wp.float32(embedding_sum[row, column]) / denominator
        )

    @wp.kernel(enable_backward=False, module="unique")
    def scale_rows(weight: wp.array2d(dtype=DTYPE), scale: wp.array1d(dtype=DTYPE)):
        row, column = wp.tid()
        weight[row, column] = DTYPE(
            wp.float32(weight[row, column]) * wp.float32(scale[row])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def affine(
        values: wp.array2d(dtype=DTYPE),
        scale: wp.array1d(dtype=DTYPE),
        bias: wp.array1d(dtype=DTYPE),
    ):
        row, column = wp.tid()
        values[row, column] = DTYPE(
            wp.float32(values[row, column]) * wp.float32(scale[column])
            + wp.float32(bias[column])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def scaled_residual(
        residual: wp.array1d(dtype=DTYPE),
        branch: wp.array1d(dtype=DTYPE),
        scale: wp.array1d(dtype=DTYPE),
        output: wp.array1d(dtype=DTYPE),
        width: int,
    ):
        index = wp.tid()
        output[index] = DTYPE(
            wp.float32(residual[index])
            + wp.float32(branch[index]) * wp.float32(scale[index % width])
        )

    @wp.kernel(enable_backward=False, module="unique")
    def crop_time(source: wp.array3d(dtype=DTYPE), output: wp.array3d(dtype=DTYPE)):
        batch, position, channel = wp.tid()
        output[batch, position, channel] = DTYPE(source[batch, position, channel])

    return normalize_codebook, scale_rows, affine, scaled_residual, crop_time


class _ContiguousPrefixPlan:
    """Execute an operator and materialize its temporal prefix contiguously."""

    def __init__(self, plan, length):
        self.plan = plan
        self.output = wp.empty(
            (plan.output.shape[0], length, plan.output.shape[2]),
            dtype=plan.output.dtype,
            device=plan.output.device,
        )
        self.kernel = _codec_kernels(plan.output.dtype)[4]

    def execute(self):
        self.plan.execute()
        wp.launch(
            self.kernel,
            dim=self.output.shape,
            inputs=[self.plan.output, self.output],
            device=self.output.device,
        )
        return self.output


class _LinearPlan:
    """Small fixed-shape wrapper around the shared optimized Linear planner."""

    def __init__(self, x, weight, *, cublas=None):
        rows = int(np.prod(x.shape[:-1]))
        self.input = x
        self.tensors = {
            "x": x.reshape((rows, x.shape[-1])),
            "weight": weight.reshape((weight.shape[0], weight.shape[1])),
        }
        self.shapes = {name: value.shape for name, value in self.tensors.items()}
        self.operation = Operation("Linear", ["x", "weight"], ["output"])
        plan_linear(self.operation, self.tensors, self.shapes, x.device, cublas=cublas)
        self.output = self.tensors["output"].reshape((*x.shape[:-1], weight.shape[0]))

    def execute(self):
        execute_operations(
            (self.operation,), self.tensors, self.shapes, self.input.device
        )
        return self.output


class _AffineLayerNormPlan:
    def __init__(self, x, scale, bias, *, epsilon=1.0e-6):
        self.norm = LayerNormPlan(x, epsilon=epsilon)
        self.scale, self.bias = scale, bias
        self.output = self.norm.output
        self.kernel = _codec_kernels(x.dtype)[2]

    def execute(self):
        self.norm.execute()
        rows = self.output.size // self.output.shape[-1]
        values = self.output.reshape((rows, self.output.shape[-1]))
        wp.launch(
            self.kernel,
            dim=values.shape,
            inputs=[values, self.scale, self.bias],
            device=self.output.device,
        )
        return self.output


class _ScaledResidualPlan:
    def __init__(self, residual, branch, scale=None):
        if residual.shape != branch.shape:
            raise ValueError("codec residual shapes do not match")
        self.residual, self.branch, self.scale = residual, branch, scale
        self.output = wp.empty_like(residual)
        self.kernels = _codec_kernels(residual.dtype)

    def execute(self):
        if self.scale is None:
            wp.launch(
                _add_arrays_kernel,
                dim=self.output.size,
                inputs=[
                    self.residual.flatten(),
                    self.branch.flatten(),
                    self.output.flatten(),
                ],
                device=self.output.device,
            )
        else:
            wp.launch(
                self.kernels[3],
                dim=self.output.size,
                inputs=[
                    self.residual.flatten(),
                    self.branch.flatten(),
                    self.scale,
                    self.output.flatten(),
                    self.output.shape[-1],
                ],
                device=self.output.device,
            )
        return self.output


def _causal_conv(x, weight, bias=None, *, stride=1, dilation=1, groups=1):
    effective = (weight.shape[2] - 1) * dilation + 1
    padding = effective - stride
    plan = Conv1dPlan(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    # Symmetric padding produces the exact causal result as its prefix.
    cropped = _ContiguousPrefixPlan(plan, math.ceil(x.shape[1] / stride))
    return cropped, cropped.output


def _causal_transpose(x, weight, bias, stride):
    plan = Conv1dPlan(x, weight, bias, stride=stride, transposed=True, padding=0)
    cropped = _ContiguousPrefixPlan(plan, x.shape[1] * stride)
    return cropped, cropped.output


class _ConvNeXtPlan:
    def __init__(self, x, weights, prefix, *, cublas=None):
        depthwise, current = _causal_conv(
            x,
            weights[f"{prefix}.dwconv.conv.weight"],
            weights[f"{prefix}.dwconv.conv.bias"],
            groups=x.shape[2],
        )
        norm = _AffineLayerNormPlan(
            current,
            weights[f"{prefix}.norm.weight"],
            weights[f"{prefix}.norm.bias"],
            epsilon=1.0e-6,
        )
        first = _LinearPlan(
            norm.output, weights[f"{prefix}.pwconv1.weight"], cublas=cublas
        )
        second = BiasedLinearPlan(
            first.output,
            weights[f"{prefix}.pwconv2.weight"],
            weights[f"{prefix}.pwconv2.bias"],
            cublas=cublas,
        )
        residual = _ScaledResidualPlan(x, second.output, weights[f"{prefix}.gamma"])
        self.plans = (depthwise, norm, first, second, residual)
        self.gelu_bias = weights[f"{prefix}.pwconv1.bias"]
        self.gelu = _encoder_kernels(x.dtype, 64)[1]
        self.output = residual.output

    def execute(self):
        for index, plan in enumerate(self.plans):
            plan.execute()
            if index == 2:
                values = plan.output.reshape(
                    (plan.output.size // plan.output.shape[-1], plan.output.shape[-1])
                )
                wp.launch(
                    self.gelu,
                    dim=values.shape,
                    inputs=[values, self.gelu_bias],
                    device=values.device,
                )
        return self.output


class _ResidualUnitPlan:
    def __init__(self, x, weights, prefix, dilation):
        snake1 = Snake1dPlan(
            x, weights[f"{prefix}.act1.alpha"], weights[f"{prefix}.act1.beta"]
        )
        conv1, current = _causal_conv(
            snake1.output,
            weights[f"{prefix}.conv1.conv.weight"],
            weights[f"{prefix}.conv1.conv.bias"],
            dilation=dilation,
        )
        snake2 = Snake1dPlan(
            current,
            weights[f"{prefix}.act2.alpha"],
            weights[f"{prefix}.act2.beta"],
        )
        conv2, current = _causal_conv(
            snake2.output,
            weights[f"{prefix}.conv2.conv.weight"],
            weights[f"{prefix}.conv2.conv.bias"],
        )
        residual = _ScaledResidualPlan(x, current)
        self.plans = (snake1, conv1, snake2, conv2, residual)
        self.output = residual.output

    def execute(self):
        for plan in self.plans:
            plan.execute()
        return self.output


class _WaveDecoderBlockPlan:
    def __init__(self, x, weights, prefix, stride):
        snake = Snake1dPlan(
            x, weights[f"{prefix}.block.0.alpha"], weights[f"{prefix}.block.0.beta"]
        )
        transpose, current = _causal_transpose(
            snake.output,
            weights[f"{prefix}.block.1.conv.weight"],
            weights[f"{prefix}.block.1.conv.bias"],
            stride,
        )
        self.plans = [snake, transpose]
        for index, dilation in enumerate((1, 3, 9), 2):
            unit = _ResidualUnitPlan(
                current, weights, f"{prefix}.block.{index}", dilation
            )
            self.plans.append(unit)
            current = unit.output
        self.output = current

    def execute(self):
        for plan in self.plans:
            plan.execute()
        return self.output


class _CodecTransformerPlan(_Qwen3EncoderPlan):
    def __init__(self, runner, sequence):
        super().__init__(runner, sequence)
        self.input = wp.empty_like(self.embedding)

    def _stage_embeddings(self):
        wp.copy(self.embedding, self.input)


class _CodecTransformer:
    def __init__(self, config, weights, *, device, dtype, cublas):
        self.config = config
        self.device, self.dtype, self.cublas = device, dtype, cublas
        self.hidden_size = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.query_heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.epsilon = float(config["rms_norm_eps"])
        self.qk_norm = False
        self.attention_bias = False
        self.sliding_window = int(config["sliding_window"])
        self.weights = {
            "model.norm.weight": weights["decoder.pre_transformer.norm.weight"]
        }
        for index in range(self.layers):
            source = f"decoder.pre_transformer.layers.{index}."
            target = f"model.layers.{index}."
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ):
                self.weights[target + suffix] = weights[source + suffix]
            for projection, scale in (
                ("self_attn.o_proj.weight", "self_attn_layer_scale.scale"),
                ("mlp.down_proj.weight", "mlp_layer_scale.scale"),
            ):
                weight = self.weights[target + projection]
                wp.launch(
                    _codec_kernels(dtype)[1],
                    dim=weight.shape,
                    inputs=[weight, weights[source + scale]],
                    device=device,
                )
        cosine, sine = rotary_cache_values(
            int(config["max_position_embeddings"]),
            self.head_dim,
            {"rope_theta": float(config["rope_theta"]), "rope_type": "default"},
        )
        self.cos_cache = wp.array(cosine, dtype=dtype, device=device)
        self.sin_cache = wp.array(sine, dtype=dtype, device=device)


def _decoder_weight_names(config):
    names = []
    quantizers = int(config["num_quantizers"])
    for group in range(quantizers):
        branch = "rvq_first" if group == 0 else "rvq_rest"
        index = 0 if group == 0 else group - 1
        prefix = f"decoder.quantizer.{branch}.vq.layers.{index}._codebook"
        names.extend((f"{prefix}.embedding_sum", f"{prefix}.cluster_usage"))
    names.extend(
        (
            "decoder.quantizer.rvq_first.output_proj.weight",
            "decoder.quantizer.rvq_rest.output_proj.weight",
            "decoder.pre_conv.conv.weight",
            "decoder.pre_conv.conv.bias",
            "decoder.pre_transformer.input_proj.weight",
            "decoder.pre_transformer.input_proj.bias",
            "decoder.pre_transformer.norm.weight",
            "decoder.pre_transformer.output_proj.weight",
            "decoder.pre_transformer.output_proj.bias",
        )
    )
    for index in range(int(config["num_hidden_layers"])):
        prefix = f"decoder.pre_transformer.layers.{index}."
        names.extend(
            prefix + suffix
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn_layer_scale.scale",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "mlp_layer_scale.scale",
            )
        )
    for index in range(2):
        prefix = f"decoder.upsample.{index}."
        names.extend(
            prefix + suffix
            for suffix in (
                "0.conv.weight",
                "0.conv.bias",
                "1.dwconv.conv.weight",
                "1.dwconv.conv.bias",
                "1.norm.weight",
                "1.norm.bias",
                "1.pwconv1.weight",
                "1.pwconv1.bias",
                "1.pwconv2.weight",
                "1.pwconv2.bias",
                "1.gamma",
            )
        )
    names.extend(("decoder.decoder.0.conv.weight", "decoder.decoder.0.conv.bias"))
    for block in range(len(config["upsample_rates"])):
        prefix = f"decoder.decoder.{block + 1}.block."
        names.extend(
            (
                prefix + "0.alpha",
                prefix + "0.beta",
                prefix + "1.conv.weight",
                prefix + "1.conv.bias",
            )
        )
        for unit in range(2, 5):
            for act in ("act1", "act2"):
                names.extend(
                    (f"{prefix}{unit}.{act}.alpha", f"{prefix}{unit}.{act}.beta")
                )
            for conv in ("conv1", "conv2"):
                names.extend(
                    (
                        f"{prefix}{unit}.{conv}.conv.weight",
                        f"{prefix}{unit}.{conv}.conv.bias",
                    )
                )
    final = len(config["upsample_rates"]) + 1
    names.extend(
        (
            f"decoder.decoder.{final}.alpha",
            f"decoder.decoder.{final}.beta",
            f"decoder.decoder.{final + 1}.conv.weight",
            f"decoder.decoder.{final + 1}.conv.bias",
        )
    )
    return tuple(names)


class _DecodePlan:
    def __init__(self, decoder, frames):
        c = decoder.config
        self.codes = wp.empty(
            (frames, c["num_quantizers"]), dtype=wp.int32, device=decoder.device
        )
        quantized = wp.empty(
            (frames, c["codebook_dim"]), dtype=decoder.dtype, device=decoder.device
        )
        self.quantized = quantized
        weights = decoder.weights
        preconv, current = _causal_conv(
            quantized.reshape((1, frames, c["codebook_dim"])),
            weights["decoder.pre_conv.conv.weight"],
            weights["decoder.pre_conv.conv.bias"],
        )
        input_projection = BiasedLinearPlan(
            current,
            weights["decoder.pre_transformer.input_proj.weight"],
            weights["decoder.pre_transformer.input_proj.bias"],
            cublas=decoder.cublas,
        )
        transformer = _CodecTransformerPlan(decoder.transformer, frames)
        output_projection = BiasedLinearPlan(
            transformer.output,
            weights["decoder.pre_transformer.output_proj.weight"],
            weights["decoder.pre_transformer.output_proj.bias"],
            cublas=decoder.cublas,
        )
        self.plans = [preconv, input_projection, transformer, output_projection]
        current = output_projection.output
        for index, ratio in enumerate(c["upsampling_ratios"]):
            transpose, current = _causal_transpose(
                current,
                weights[f"decoder.upsample.{index}.0.conv.weight"],
                weights[f"decoder.upsample.{index}.0.conv.bias"],
                ratio,
            )
            convnext = _ConvNeXtPlan(
                current, weights, f"decoder.upsample.{index}.1", cublas=decoder.cublas
            )
            self.plans.extend((transpose, convnext))
            current = convnext.output
        first, current = _causal_conv(
            current,
            weights["decoder.decoder.0.conv.weight"],
            weights["decoder.decoder.0.conv.bias"],
        )
        self.plans.append(first)
        for index, rate in enumerate(c["upsample_rates"], 1):
            block = _WaveDecoderBlockPlan(
                current, weights, f"decoder.decoder.{index}", rate
            )
            self.plans.append(block)
            current = block.output
        snake_index = len(c["upsample_rates"]) + 1
        snake = Snake1dPlan(
            current,
            weights[f"decoder.decoder.{snake_index}.alpha"],
            weights[f"decoder.decoder.{snake_index}.beta"],
        )
        final, current = _causal_conv(
            snake.output,
            weights[f"decoder.decoder.{snake_index + 1}.conv.weight"],
            weights[f"decoder.decoder.{snake_index + 1}.conv.bias"],
        )
        clamp = ClampPlan(current, -1.0, 1.0)
        self.plans.extend((snake, final, clamp))
        self.transformer = transformer
        self.input_projection = input_projection
        self.output = clamp.output
        self.graph = None

    def _execute(self):
        wp.launch(
            _gather_sum_embeddings_kernel,
            dim=self.quantized.shape,
            inputs=[self.decoder.tables, self.codes, self.quantized],
            device=self.quantized.device,
        )
        for plan in self.plans:
            if plan is self.transformer:
                wp.copy(plan.input, self.input_projection.output)
            plan.execute()

    def execute(self):
        if self.graph is None:
            self._execute()
        else:
            wp.capture_launch(self.graph)
        return self.output

    def capture(self):
        if not self.output.device.is_cuda:
            raise RuntimeError("CUDA graph capture requires a CUDA device")
        wp.capture_begin(device=self.output.device)
        self._execute()
        self.graph = wp.capture_end(device=self.output.device)
        return self.graph


class Qwen3TTSCodecDecoder:
    """Fixed-checkpoint, length-specialized 24 kHz Qwen3-TTS codec decoder."""

    sample_rate = 24_000
    samples_per_frame = 1_920

    def __init__(self, path, *, device=None, dtype=wp.bfloat16, use_cublas=True):
        path = Path(path)
        tokenizer_path = (
            path / "speech_tokenizer" if (path / "speech_tokenizer").is_dir() else path
        )
        document = json.loads(
            (tokenizer_path / "config.json").read_text(encoding="utf-8")
        )
        if document.get("model_type") != "qwen3_tts_tokenizer_12hz":
            raise ValueError("Qwen3-TTS codec requires the 12 Hz tokenizer")
        self.config = dict(document["decoder_config"])
        if (
            math.prod(self.config["upsampling_ratios"] + self.config["upsample_rates"])
            != self.samples_per_frame
        ):
            raise ValueError("Qwen3-TTS codec upsampling geometry is inconsistent")
        if (
            int(self.config["num_quantizers"]) != 16
            or int(self.config["codebook_size"]) != 2048
        ):
            raise ValueError("Qwen3-TTS codec requires 16 groups of 2,048 codes")
        self.device = parse_device(device)
        self.dtype = dtype
        if dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Qwen3-TTS codec activations require FP16 or BF16")
        self.cublas = (
            try_create_cublas() if use_cublas and self.device.is_cuda else None
        )
        archive = SafeTensorNamespace(SafeTensorArchive(tokenizer_path), "")
        names = _decoder_weight_names(self.config)
        missing = set(names) - set(archive.names)
        if missing:
            raise ValueError(
                f"Qwen3-TTS codec checkpoint is missing {sorted(missing)[:5]}"
            )
        self.weights = load_cast_weights(archive, names, self.device, dtype)
        self.tables = self._build_codebooks()
        self.transformer = _CodecTransformer(
            self.config,
            self.weights,
            device=self.device,
            dtype=dtype,
            cublas=self.cublas,
        )
        self._plans = {}

    def _build_codebooks(self):
        c = self.config
        tables = wp.empty(
            (c["num_quantizers"], c["codebook_size"], c["codebook_dim"]),
            dtype=self.dtype,
            device=self.device,
        )
        normalized = wp.empty(
            (c["codebook_size"], c["codebook_dim"] // 2),
            dtype=self.dtype,
            device=self.device,
        )
        for group in range(c["num_quantizers"]):
            branch = "rvq_first" if group == 0 else "rvq_rest"
            index = 0 if group == 0 else group - 1
            prefix = f"decoder.quantizer.{branch}"
            codebook = f"{prefix}.vq.layers.{index}._codebook"
            wp.launch(
                _codec_kernels(self.dtype)[0],
                dim=normalized.shape,
                inputs=[
                    self.weights[f"{codebook}.embedding_sum"],
                    self.weights[f"{codebook}.cluster_usage"],
                    normalized,
                ],
                device=self.device,
            )
            projection = _LinearPlan(
                normalized,
                self.weights[f"{prefix}.output_proj.weight"],
                cublas=self.cublas,
            )
            projection.execute()
            wp.copy(tables[group], projection.output)
        return tables

    def plan(self, frames):
        frames = int(frames)
        if not 1 <= frames <= int(self.config["max_position_embeddings"]):
            raise ValueError("codec frame count is outside the supported range")
        plan = self._plans.get(frames)
        if plan is None:
            plan = self._plans[frames] = _DecodePlan(self, frames)
            plan.decoder = self
        return plan

    def decode(self, codes: Sequence[Sequence[int]] | np.ndarray):
        values = np.asarray(codes, dtype=np.int32)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2 or values.shape[1] != int(self.config["num_quantizers"]):
            raise ValueError("codec codes must have shape [frames, 16]")
        if (
            values.size == 0
            or values.min() < 0
            or values.max() >= int(self.config["codebook_size"])
        ):
            raise ValueError("codec code IDs must be between 0 and 2,047")
        plan = self.plan(values.shape[0])
        plan.codes.assign(values)
        return plan.execute()
