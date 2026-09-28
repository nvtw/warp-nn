# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein VAE decode using shared spatial Warp operators."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from ..formats.safetensors import SafeTensorArchive
from ..operators import (
    ActivationBufferPool,
    AttentionHeadsPlan,
    AttentionMergePlan,
    BiasedLinearPlan,
    BidirectionalGQAPlan,
    Conv2dPlan,
    NearestUpsample2dPlan,
    ResidualAddPlan,
    SpatialGroupNormPlan,
    SpatialPatchPackPlan,
    SpatialPatchUnpackPlan,
    reuse_plan_output,
)
from ..weights import load_cast_weights


class _Residual:
    def __init__(self, x, weights, prefix, *, scratch=None, output_slot=0):
        def share(plan, name):
            if scratch is not None:
                reuse_plan_output(plan, scratch, f"{name}.{plan.output.shape}")

        self.norm1 = SpatialGroupNormPlan(
            x,
            weights[prefix + ".norm1.weight"],
            weights[prefix + ".norm1.bias"],
            silu=True,
        )
        share(self.norm1, "norm1")
        self.conv1 = Conv2dPlan(
            self.norm1.output,
            weights[prefix + ".conv1.weight"],
            weights[prefix + ".conv1.bias"],
            padding=1,
            tensor_cores=True,
        )
        share(self.conv1, "conv1")
        self.norm2 = SpatialGroupNormPlan(
            self.conv1.output,
            weights[prefix + ".norm2.weight"],
            weights[prefix + ".norm2.bias"],
            silu=True,
        )
        share(self.norm2, "norm2")
        self.conv2 = Conv2dPlan(
            self.norm2.output,
            weights[prefix + ".conv2.weight"],
            weights[prefix + ".conv2.bias"],
            padding=1,
            tensor_cores=True,
        )
        share(self.conv2, "conv2")
        shortcut = prefix + ".conv_shortcut.weight"
        self.shortcut = (
            Conv2dPlan(
                x,
                weights[shortcut],
                weights[prefix + ".conv_shortcut.bias"],
                tensor_cores=True,
            )
            if shortcut in weights
            else None
        )
        self.residual = ResidualAddPlan(
            self.conv2.output, x if self.shortcut is None else self.shortcut.output
        )
        share(self.residual, f"hidden.{output_slot}")
        self.output = self.residual.output

    def execute(self):
        self.norm1.execute()
        self.conv1.execute()
        self.norm2.execute()
        self.conv2.execute()
        if self.shortcut is not None:
            self.shortcut.execute()
        return self.residual.execute()


class _MidAttention:
    def __init__(self, x, weights, prefix):
        self.norm = SpatialGroupNormPlan(
            x,
            weights[prefix + ".group_norm.weight"],
            weights[prefix + ".group_norm.bias"],
        )
        self.projections = [
            BiasedLinearPlan(
                self.norm.output.reshape(
                    (x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
                ),
                weights[prefix + f".to_{part}.weight"],
                weights[prefix + f".to_{part}.bias"],
            )
            for part in ("q", "k", "v")
        ]
        heads = [AttentionHeadsPlan(plan.output, 1) for plan in self.projections]
        self.heads = heads
        self.attention = BidirectionalGQAPlan(*(plan.output for plan in heads))
        self.merge = AttentionMergePlan(self.attention.output)
        self.out = BiasedLinearPlan(
            self.merge.output,
            weights[prefix + ".to_out.0.weight"],
            weights[prefix + ".to_out.0.bias"],
        )
        self.residual = ResidualAddPlan(x, self.out.output.reshape(x.shape))
        self.output = self.residual.output

    def execute(self):
        self.norm.execute()
        for plan in self.projections + self.heads:
            plan.execute()
        self.attention.execute()
        self.merge.execute()
        self.out.execute()
        return self.residual.execute()


class Flux2KleinVAEDecoder:
    """One fixed output resolution, with the official post-quant decoder weights."""

    def __init__(
        self, path, latent_height, latent_width, *, dtype=wp.bfloat16, device=None
    ):
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if (
            config.get("latent_channels"),
            config.get("block_out_channels"),
            config.get("norm_num_groups"),
        ) != (32, [128, 256, 512, 512], 32):
            raise ValueError("unsupported FLUX.2 Klein VAE configuration")
        archive = SafeTensorArchive(path / "diffusion_pytorch_model.safetensors")
        names = [
            name
            for name in archive.names
            if name.startswith("decoder.") or name.startswith("post_quant_conv.")
        ]
        weights = load_cast_weights(archive, names, device, dtype)
        self.input = wp.empty(
            (1, 32, latent_height, latent_width), dtype=dtype, device=device
        )
        self.pack = SpatialPatchPackPlan(self.input, 1)
        start = self.pack.output.reshape((1, latent_height, latent_width, 32))
        self.post_quant = Conv2dPlan(
            start,
            weights["post_quant_conv.weight"],
            weights["post_quant_conv.bias"],
            tensor_cores=True,
        )
        self.conv_in = Conv2dPlan(
            self.post_quant.output,
            weights["decoder.conv_in.weight"],
            weights["decoder.conv_in.bias"],
            padding=1,
            tensor_cores=True,
        )
        first = _Residual(self.conv_in.output, weights, "decoder.mid_block.resnets.0")
        attention = _MidAttention(
            first.output, weights, "decoder.mid_block.attentions.0"
        )
        second = _Residual(attention.output, weights, "decoder.mid_block.resnets.1")
        self.mid = (first, attention, second)
        x = second.output
        self.up = []
        for block in range(4):
            residuals = []
            scratch = ActivationBufferPool()
            for index in range(3):
                residual = _Residual(
                    x,
                    weights,
                    f"decoder.up_blocks.{block}.resnets.{index}",
                    scratch=scratch,
                    output_slot=index % 2,
                )
                residuals.append(residual)
                x = residual.output
            if block < 3:
                expanded = NearestUpsample2dPlan(x, 2)
                up = Conv2dPlan(
                    expanded.output,
                    weights[f"decoder.up_blocks.{block}.upsamplers.0.conv.weight"],
                    weights[f"decoder.up_blocks.{block}.upsamplers.0.conv.bias"],
                    padding=1,
                    tensor_cores=True,
                )
                x = up.output
            else:
                expanded = up = None
            self.up.append((residuals, expanded, up))
        self.norm_out = SpatialGroupNormPlan(
            x,
            weights["decoder.conv_norm_out.weight"],
            weights["decoder.conv_norm_out.bias"],
            silu=True,
        )
        self.conv_out = Conv2dPlan(
            self.norm_out.output,
            weights["decoder.conv_out.weight"],
            weights["decoder.conv_out.bias"],
            padding=1,
            tensor_cores=True,
        )
        height, width = latent_height * 8, latent_width * 8
        self.unpack = SpatialPatchUnpackPlan(
            self.conv_out.output.reshape((1, height * width, 3)), height, width, 1
        )
        self.output = self.unpack.output

    def execute(self):
        self.pack.execute()
        self.post_quant.execute()
        self.conv_in.execute()
        for plan in self.mid:
            plan.execute()
        for residuals, expanded, up in self.up:
            for plan in residuals:
                plan.execute()
            if expanded is not None:
                expanded.execute()
                up.execute()
        self.norm_out.execute()
        self.conv_out.execute()
        return self.unpack.execute()


def flux2_klein_latent_batch_norm(path):
    """Read frozen packed-latent BN statistics as FP32 NumPy arrays."""
    archive = SafeTensorArchive(Path(path) / "diffusion_pytorch_model.safetensors")
    stats = archive.load("cpu", ["bn.running_mean", "bn.running_var"])
    mean = stats["bn.running_mean"].numpy().astype(np.float32)
    std = np.sqrt(stats["bn.running_var"].numpy().astype(np.float32) + 1.0e-4)
    if mean.shape != (128,) or std.shape != (128,):
        raise ValueError("FLUX.2 Klein packed-latent BN geometry is incompatible")
    return mean, std
