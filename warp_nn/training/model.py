# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-neutral fixed-buffer causal LoRA training composition."""

from collections.abc import Mapping, Sequence

import warp as wp

from .bridges import cast_from_float32
from .output import CausalLMOutputPlan
from .primitives import EmbeddingPlan
from .stack import LoRATransformerStackPlan


def require_weights(
    weights: Mapping[str, object], names: Sequence[str]
) -> dict[str, wp.array]:
    """Return required unquantized Warp arrays with a compact missing error."""
    missing = [name for name in names if name not in weights]
    if missing:
        raise ValueError(f"training checkpoint is missing {missing[:5]}")
    result = {name: weights[name] for name in names}
    invalid = [
        name for name, value in result.items() if not isinstance(value, wp.array)
    ]
    if invalid:
        raise TypeError(
            "LoRA training requires unquantized FP16/BF16 weights; "
            f"unsupported tensors include {invalid[:5]}"
        )
    return result


class CausalLMTrainingPlan:
    """Compose frozen embeddings, a LoRA stack, and causal output loss.

    All large storage belongs to child plans and is allocated once. ``train_step``
    is safe to capture when its input arrays have stable addresses. Embeddings,
    normalization weights, and the vocabulary head remain frozen; only adapters
    are passed to the optimizer.
    """

    def __init__(
        self,
        embedding_weight: wp.array,
        stack: LoRATransformerStackPlan,
        output: CausalLMOutputPlan,
    ):
        if not isinstance(embedding_weight, wp.array) or embedding_weight.ndim != 2:
            raise TypeError("embedding_weight must be a 2-D Warp array")
        if (
            embedding_weight.dtype != stack.dtype
            or embedding_weight.device != stack.device
            or embedding_weight.shape[1] != stack.hidden
            or not embedding_weight.is_contiguous
        ):
            raise ValueError("embedding weight must match the transformer stack")
        if (
            output.device != stack.device
            or output.dtype != stack.dtype
            or output.rows != stack.rows
            or output.hidden != stack.hidden
        ):
            raise ValueError("causal output plan must match the transformer stack")
        if output.classes != embedding_weight.shape[0]:
            raise ValueError("embedding and output vocabulary sizes must match")
        self.embedding_weight = embedding_weight
        self.stack = stack
        self.output = output
        self.adapters = stack.adapters
        self.device = stack.device
        self.dtype = stack.dtype
        self.rows = stack.rows
        self.hidden = stack.hidden
        self.vocabulary = embedding_weight.shape[0]
        self.embedding = EmbeddingPlan(
            self.rows,
            self.vocabulary,
            self.hidden,
            dtype=self.dtype,
            device=self.device,
        )
        self.stack_output_grad = wp.empty(
            (self.rows, self.hidden), dtype=self.dtype, device=self.device
        )

    def forward(
        self,
        input_ids: wp.array,
        targets: wp.array,
        lengths: wp.array,
        positions=None,
        cosine=None,
        sine=None,
        *,
        reduction: str = "mean",
    ) -> wp.array:
        hidden = self.embedding(self.embedding_weight, input_ids)
        hidden = self.stack.forward(hidden, lengths, positions, cosine, sine)
        return self.output.forward(hidden, targets, reduction=reduction)

    def backward(
        self,
        input_ids: wp.array,
        targets: wp.array,
        lengths: wp.array,
        positions=None,
        cosine=None,
        sine=None,
        *,
        loss_scale: float = 1.0,
        reduction: str = "mean",
        accumulate: bool = False,
    ) -> wp.array:
        del input_ids
        output_gradient = self.output.backward(
            self.stack.output,
            targets,
            loss_scale=loss_scale,
            reduction=reduction,
        )
        cast_from_float32(output_gradient, self.stack_output_grad)
        return self.stack.backward(
            self.embedding.output,
            lengths,
            self.stack_output_grad,
            positions,
            cosine,
            sine,
            accumulate=accumulate,
        )

    def train_step(
        self,
        input_ids: wp.array,
        targets: wp.array,
        lengths: wp.array,
        positions=None,
        cosine=None,
        sine=None,
        *,
        loss_scale: float = 1.0,
        reduction: str = "mean",
    ) -> wp.array:
        """Run zero-grad, forward, backward, and FP32-master AdamW update."""
        self.stack.zero_grad()
        loss = self.forward(
            input_ids,
            targets,
            lengths,
            positions,
            cosine,
            sine,
            reduction=reduction,
        )
        self.backward(
            input_ids,
            targets,
            lengths,
            positions,
            cosine,
            sine,
            loss_scale=loss_scale,
            reduction=reduction,
        )
        self.stack.step()
        return loss
