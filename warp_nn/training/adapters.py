# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Named, model-neutral LoRA adapter parameters and optimizer state."""

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
import warp as wp

from warp_nn.runtime._cublas import try_create_cublas

from .bridges import cast_from_float32
from .optimizer import AdamWPlan
from .step import LoRALinearTrainingPlan


_STORAGE_DTYPES = (wp.float16, wp.bfloat16)


@dataclass(frozen=True)
class LoRAAdapterConfig:
    """Shape-independent configuration for one named LoRA target."""

    rank: int
    alpha: float | None = None
    init_std: float = 0.02

    def __post_init__(self):
        alpha = self.rank if self.alpha is None else self.alpha
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not math.isfinite(self.init_std) or self.init_std <= 0.0:
            raise ValueError("LoRA init_std must be finite and positive")

    @property
    def scale(self) -> float:
        """Return the conventional ``alpha / rank`` adapter multiplier."""
        return float(self.rank if self.alpha is None else self.alpha) / self.rank


@dataclass(frozen=True)
class LoRAAdapterTarget:
    """Fixed buffers associated with one frozen Linear weight."""

    name: str
    config: LoRAAdapterConfig
    weight: wp.array
    lora_a: wp.array
    lora_b: wp.array
    plan: LoRALinearTrainingPlan


class LoRAAdapterCollection:
    """Own named LoRA mirrors, gradients, plans, and one bound AdamW plan.

    The collection does not interpret model names or execute a model. Callers
    select an exact target name and provide that target's input or output
    gradient. Frozen base weights are never passed to the optimizer.
    """

    def __init__(
        self,
        weights: Mapping[str, wp.array],
        rows: int | Mapping[str, int],
        configs: LoRAAdapterConfig | Mapping[str, LoRAAdapterConfig],
        *,
        seed: int = 0,
        optimizer_options: Mapping[str, object] | None = None,
        use_cublas: bool = True,
    ):
        if not weights:
            raise ValueError("LoRA adapters require at least one target weight")
        names = tuple(sorted(weights))
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("LoRA target names must be non-empty strings")
        row_counts = self._expand_rows(rows, names)
        target_configs = self._expand_configs(configs, names)
        rng = np.random.default_rng(seed)
        targets = {}
        parameters = []
        gradients = []
        device = None
        shared_cublas = None
        for name in names:
            weight = weights[name]
            if not isinstance(weight, wp.array) or weight.ndim != 2:
                raise TypeError(f"LoRA weight '{name}' must be a 2-D Warp array")
            if weight.dtype not in _STORAGE_DTYPES:
                raise TypeError(f"LoRA weight '{name}' must use FP16 or BF16 storage")
            if not weight.is_contiguous or weight.size == 0:
                raise ValueError(
                    f"LoRA weight '{name}' must be non-empty and contiguous"
                )
            if device is None:
                device = weight.device
                if use_cublas and device.is_cuda:
                    shared_cublas = try_create_cublas()
            elif weight.device != device:
                raise ValueError("all LoRA target weights must share one device")
            out_features, in_features = weight.shape
            config = target_configs[name]
            a_values = rng.normal(
                0.0, config.init_std, (config.rank, in_features)
            ).astype(np.float32)
            lora_a = wp.array(a_values, dtype=weight.dtype, device=device)
            lora_b = wp.zeros(
                (out_features, config.rank), dtype=weight.dtype, device=device
            )
            plan = LoRALinearTrainingPlan(
                row_counts[name],
                in_features,
                out_features,
                config.rank,
                weight.dtype,
                device=device,
                _defer_base_matmul_workspace=True,
                cublas=shared_cublas,
            )
            targets[name] = LoRAAdapterTarget(
                name, config, weight, lora_a, lora_b, plan
            )
            parameters.extend((lora_a, lora_b))
            gradients.extend((plan.grad_a, plan.grad_b))

        workspace_elements = (
            max(
                target.plan.base_matmul_workspace_elements
                for target in targets.values()
            )
            if shared_cublas is None
            else 0
        )
        self.base_matmul_workspace = (
            wp.empty(workspace_elements, dtype=wp.float32, device=device)
            if workspace_elements
            else None
        )
        if self.base_matmul_workspace is not None:
            for target in targets.values():
                if target.plan.base_matmul_workspace_elements:
                    target.plan.bind_base_matmul_workspace(self.base_matmul_workspace)

        self.device = device
        self.cublas = shared_cublas
        self.configs = MappingProxyType(dict(target_configs))
        self.targets = MappingProxyType(targets)
        options = {"learning_rate": 1.0e-4}
        options.update(optimizer_options or {})
        self.optimizer = AdamWPlan(parameters, gradients, **options)
        parameter_names = tuple(
            key
            for name in names
            for key in (f"{name}.lora_A.weight", f"{name}.lora_B.weight")
        )
        self.named_parameters = MappingProxyType(dict(zip(parameter_names, parameters)))
        self.named_gradients = MappingProxyType(dict(zip(parameter_names, gradients)))
        self.named_masters = MappingProxyType(
            {
                key: master.reshape(parameter.shape)
                for key, parameter, master in zip(
                    parameter_names, parameters, self.optimizer.masters
                )
            }
        )
        self.zero_grad()

    @staticmethod
    def _expand_rows(rows, names: tuple[str, ...]) -> dict[str, int]:
        if isinstance(rows, int):
            result = {name: rows for name in names}
        elif isinstance(rows, Mapping) and set(rows) == set(names):
            result = dict(rows)
        else:
            raise ValueError("LoRA rows must be an integer or an exact target mapping")
        if any(not isinstance(value, int) or value <= 0 for value in result.values()):
            raise ValueError("LoRA row counts must be positive integers")
        return result

    @staticmethod
    def _expand_configs(
        configs, names: tuple[str, ...]
    ) -> dict[str, LoRAAdapterConfig]:
        if isinstance(configs, LoRAAdapterConfig):
            return {name: configs for name in names}
        if not isinstance(configs, Mapping) or set(configs) != set(names):
            raise ValueError(
                "LoRA configs must be one config or an exact target mapping"
            )
        if any(not isinstance(value, LoRAAdapterConfig) for value in configs.values()):
            raise TypeError("LoRA target configs must be LoRAAdapterConfig values")
        return dict(configs)

    def forward(self, name: str, x: wp.array) -> wp.array:
        """Run one named frozen Linear plus its adapter into fixed output storage."""
        target = self.targets[name]
        return target.plan.forward(
            x,
            target.weight,
            target.lora_a,
            target.lora_b,
            scale=target.config.scale,
        )

    def backward(
        self,
        name: str,
        x: wp.array,
        grad_output: wp.array,
        *,
        accumulate: bool = False,
    ) -> wp.array:
        """Run explicit backward for one target and return its fixed input gradient."""
        target = self.targets[name]
        return target.plan.backward(
            x,
            target.weight,
            target.lora_a,
            target.lora_b,
            grad_output,
            scale=target.config.scale,
            accumulate=accumulate,
        )

    def zero_grad(self) -> None:
        """Clear adapter and Tape-boundary gradients without replacing buffers."""
        for target in self.targets.values():
            target.plan.output.grad.zero_()
        self.optimizer.zero_grad()

    def step(self) -> None:
        """Update all adapters through their authoritative FP32 AdamW masters."""
        self.optimizer.step()

    def load_fp32_state(
        self,
        tensors: Mapping[str, np.ndarray | wp.array],
        configs: Mapping[str, LoRAAdapterConfig],
    ) -> None:
        """Restore adapter masters/mirrors and reset optimizer trajectory state."""
        if set(tensors) != set(self.named_masters):
            raise ValueError("checkpoint tensor names do not match adapter targets")
        if set(configs) != set(self.configs):
            raise ValueError("checkpoint configs do not match adapter targets")
        for name, config in configs.items():
            current = self.configs[name]
            if config.rank != current.rank or config.scale != current.scale:
                raise ValueError(f"checkpoint config does not match target {name!r}")
        for name, master in self.named_masters.items():
            source = tensors[name]
            values = source.numpy() if isinstance(source, wp.array) else source
            if not isinstance(values, np.ndarray) or values.dtype != np.float32:
                raise TypeError("checkpoint adapter tensors must use FP32")
            if values.shape != master.shape:
                raise ValueError(f"checkpoint shape does not match {name!r}")
            master.assign(values)
            cast_from_float32(master, self.named_parameters[name])
        self.optimizer.reset_state()
        self.zero_grad()

    def load_training_state(
        self,
        tensors: Mapping[str, np.ndarray],
        configs: Mapping[str, LoRAAdapterConfig],
        optimizer_fingerprint: Mapping[str, object],
        parameter_dtypes: Mapping[str, str],
        backend: str,
    ) -> None:
        """Restore exact AdamW trajectory state without replacing any buffers."""
        current_backend = "cublas" if self.cublas is not None else "warp"
        if backend != current_backend:
            raise ValueError("training-state execution backend does not match")
        if set(configs) != set(self.configs) or any(
            configs[name].rank != current.rank or configs[name].scale != current.scale
            for name, current in self.configs.items()
        ):
            raise ValueError("training-state LoRA configs do not match")
        dtype_names = {wp.float16: "F16", wp.bfloat16: "BF16", wp.float32: "F32"}
        current_dtypes = {
            name: dtype_names[parameter.dtype]
            for name, parameter in self.named_parameters.items()
        }
        if dict(parameter_dtypes) != current_dtypes:
            raise ValueError("training-state parameter dtypes do not match")
        optimizer = self.optimizer
        current_optimizer = {
            name: getattr(optimizer, name)
            for name in (
                "beta1",
                "beta2",
                "epsilon",
                "gradient_multiplier",
                "learning_rate",
                "loss_scale",
                "max_grad_norm",
                "min_learning_rate_ratio",
                "normalize_by_valid_tokens",
                "total_steps",
                "warmup_steps",
                "weight_decay",
            )
        }
        if dict(optimizer_fingerprint) != current_optimizer:
            raise ValueError("training-state optimizer configuration does not match")
        expected = {"optimizer.step_count"}
        for name in self.named_masters:
            expected.update(
                (
                    f"{name}.master",
                    f"{name}.first_moment",
                    f"{name}.second_moment",
                )
            )
        if set(tensors) != expected:
            raise ValueError("training-state tensor names do not match")
        step = tensors["optimizer.step_count"]
        if (
            not isinstance(step, np.ndarray)
            or step.dtype != np.int32
            or step.shape != (1,)
        ):
            raise TypeError("training-state step_count must be INT32 with shape (1,)")
        ordered = []
        for index, (name, master) in enumerate(self.named_masters.items()):
            values = tuple(
                tensors[f"{name}.{suffix}"]
                for suffix in ("master", "first_moment", "second_moment")
            )
            if any(
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or value.shape != master.shape
                for value in values
            ):
                raise ValueError(f"training-state tensors do not match {name!r}")
            ordered.append((index, name, master, values))

        for index, name, master, values in ordered:
            master.assign(values[0])
            optimizer.first_moments[index].assign(values[1].reshape(-1))
            optimizer.second_moments[index].assign(values[2].reshape(-1))
            cast_from_float32(master, self.named_parameters[name])
        optimizer.step_count.assign(step)
        self.zero_grad()
        optimizer.all_finite.fill_(1)
        optimizer.step_enabled.fill_(1)
        optimizer.normalization_multiplier.zero_()
        optimizer.effective_learning_rate.zero_()
        if optimizer.global_grad_norm is not None:
            optimizer.global_grad_norm.zero_()
        if optimizer.clip_scale is not None:
            optimizer.clip_scale.zero_()
