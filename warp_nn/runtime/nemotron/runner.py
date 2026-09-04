# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H and Nemotron Omni runner for Hugging Face safetensors."""

from __future__ import annotations

import json
from pathlib import Path
import weakref

import warp as wp

from warp_nn.runtime._cublas import try_create_cublas
from warp_nn.runtime.kernels import (
    _append_head_cache_kernel,
    _causal_conv_rows_kernel,
    _dequantize_e4m3_kernel,
    _apply_embedding_overrides_kernel,
    _gather_rows_kernel,
    _get_gated_rms_norm_kernel,
    _get_gqa_attention_kernel,
    _get_mamba2_decode_kernel,
    _get_mamba2_prefill_kernel,
    _relu2_kernel,
    _reorder_heads_kernel,
    _split_last_axis_kernel,
    _update_conv_rows_state_kernel,
)
from warp_nn.runtime.operators import (
    Operation,
    SparseExpertPlan,
    execute_operations,
    plan_linear,
    plan_residual_rms_norm,
    plan_rms_norm,
)
from warp_nn.runtime.autoregressive import AutoregressiveRunner
from warp_nn.runtime.formats.safetensors import SafeTensorArchive, SafeTensorNamespace
from warp_nn.utils.device import parse_device


def _language_config(document: dict) -> tuple[dict, str]:
    """Normalize flat Nemotron-H and nested Omni language configurations."""
    if "llm_config" not in document:
        config, prefix = dict(document), ""
    else:
        config, prefix = dict(document["llm_config"]), "language_model."
    if "attention_head_dim" not in config and "head_dim" in config:
        config["attention_head_dim"] = config["head_dim"]
    return config, prefix


def _validate_config(config: dict) -> None:
    required = (
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_hidden_layers",
        "hybrid_override_pattern",
        "mamba_num_heads",
        "mamba_head_dim",
        "n_groups",
        "ssm_state_size",
        "conv_kernel",
        "num_attention_heads",
        "num_key_value_heads",
        "attention_head_dim",
        "max_position_embeddings",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Nemotron-H config is missing {missing}")
    pattern = config["hybrid_override_pattern"]
    if len(pattern) != int(config["num_hidden_layers"]) or set(pattern) - set("M-E*"):
        raise ValueError(
            "Nemotron-H hybrid pattern must contain one M, E, -, or * per layer"
        )
    if int(config["mamba_num_heads"]) % int(config["n_groups"]):
        raise ValueError("Nemotron-H Mamba heads must be divisible by its groups")
    if int(config["num_attention_heads"]) % int(config["num_key_value_heads"]):
        raise ValueError("Nemotron-H query heads must be divisible by KV heads")
    if (
        config.get("mamba_hidden_act", "silu") != "silu"
        or config.get("mlp_hidden_act", "relu2") != "relu2"
    ):
        raise ValueError(
            "Nemotron-H runner requires SiLU Mamba and ReLU-squared MLP blocks"
        )
    if any(
        config.get(name, False)
        for name in ("attention_bias", "mamba_proj_bias", "mlp_bias", "use_bias")
    ):
        raise ValueError("Biased Nemotron-H projections are not supported")
    if "E" in pattern:
        required_moe = (
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "moe_shared_expert_intermediate_size",
            "routed_scaling_factor",
        )
        missing_moe = [name for name in required_moe if name not in config]
        if missing_moe:
            raise ValueError(f"Nemotron-H MoE config is missing {missing_moe}")


def _weight_names(config: dict) -> list[str]:
    names = ["backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"]
    for index, block_type in enumerate(config["hybrid_override_pattern"]):
        prefix = f"backbone.layers.{index}."
        names.append(prefix + "norm.weight")
        if block_type == "M":
            names.extend(
                prefix + "mixer." + suffix
                for suffix in (
                    "norm.weight",
                    "A_log",
                    "D",
                    "dt_bias",
                    "conv1d.weight",
                    "conv1d.bias",
                    "in_proj.weight",
                    "out_proj.weight",
                )
            )
        elif block_type == "-":
            names.extend(
                prefix + "mixer." + suffix
                for suffix in ("up_proj.weight", "down_proj.weight")
            )
        elif block_type == "E":
            mixer = prefix + "mixer."
            names.extend(
                mixer + suffix
                for suffix in (
                    "gate.weight",
                    "gate.e_score_correction_bias",
                    "shared_experts.up_proj.weight",
                    "shared_experts.down_proj.weight",
                )
            )
            for expert in range(int(config["n_routed_experts"])):
                names.extend(
                    mixer + f"experts.{expert}." + suffix
                    for suffix in ("up_proj.weight", "down_proj.weight")
                )
        else:
            names.extend(
                prefix + "mixer." + suffix
                for suffix in (
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                )
            )
    return names


def _load_unpacked_weights(
    archive: SafeTensorArchive, names: list[str], device, dtype: type
) -> dict[str, wp.array]:
    fp8 = [name for name in names if archive.metadata(name).format == "F8_E4M3"]
    scale_names = [name + "_scale" for name in fp8]
    missing_scales = set(scale_names) - set(archive.names)
    if missing_scales:
        raise ValueError(
            f"Nemotron-H checkpoint is missing {sorted(missing_scales)[:5]}"
        )
    weights = archive.load(
        device, [name for name in names if name not in fp8] + scale_names
    )
    for name, scale_name in zip(fp8, scale_names):
        packed = archive.load(device, [name])[name]
        output = wp.empty(packed.shape, dtype=dtype, device=device)
        wp.launch(
            _dequantize_e4m3_kernel,
            dim=packed.size,
            inputs=[packed.flatten(), weights[scale_name].flatten(), output.flatten()],
            device=device,
        )
        wp.synchronize_stream(device)
        weights[name] = output
        del packed
    for name in scale_names:
        del weights[name]
    return weights


def _load_weights(
    archive, names: list[str], device, dtype: type
) -> dict[str, wp.array]:
    """Load ordinary tensors and pack per-expert matrices without duplication."""
    expert_names = [name for name in names if ".experts." in name]
    weights = _load_unpacked_weights(
        archive, [name for name in names if name not in expert_names], device, dtype
    )
    prefixes = sorted({name.split("experts.", 1)[0] for name in expert_names})
    for prefix in prefixes:
        experts = sorted(
            {
                int(name.split("experts.", 1)[1].split(".", 1)[0])
                for name in expert_names
            }
        )
        if experts != list(range(len(experts))):
            raise ValueError(f"Nemotron-H experts under {prefix} are not contiguous")
        packed = {}
        for projection in ("up_proj.weight", "down_proj.weight"):
            first_name = f"{prefix}experts.0.{projection}"
            info = archive.metadata(first_name)
            if info.dtype != dtype:
                raise TypeError("Packed Nemotron-H BF16 experts must match model dtype")
            target = wp.empty((len(experts), *info.shape), dtype=dtype, device=device)
            stride = int(info.nbytes // 2)
            for begin in range(0, len(experts), 8):
                batch_names = [
                    f"{prefix}experts.{expert}.{projection}"
                    for expert in experts[begin : begin + 8]
                ]
                batch = archive.load(device, batch_names)
                for name in batch_names:
                    expert = int(name.split("experts.", 1)[1].split(".", 1)[0])
                    wp.copy(
                        target.flatten(),
                        batch[name].flatten(),
                        dest_offset=expert * stride,
                        count=stride,
                    )
            packed[prefix + "experts." + projection] = target
        weights.update(packed)
    return weights


class _NemotronPlan:
    """Fixed-row Nemotron execution plan sharing persistent model state."""

    def __init__(self, runner: NemotronHRunner, rows: int):
        self.runner = weakref.proxy(runner)
        self.rows = rows
        self.device = runner.device
        self.dtype = runner.dtype
        self.tensors = dict(runner.weights)
        self.shapes = {name: tuple(value.shape) for name, value in self.tensors.items()}
        self.input_ids = wp.zeros((1, rows), dtype=wp.int64, device=self.device)
        self.position_ids = wp.zeros((1, rows), dtype=wp.int64, device=self.device)
        self.embedding = wp.empty(
            (1, rows, runner.hidden_size), dtype=self.dtype, device=self.device
        )
        self.tensors["hidden.0"] = self.embedding.reshape((rows, runner.hidden_size))
        self.shapes["hidden.0"] = (rows, runner.hidden_size)
        self.embedding_overrides = wp.empty_like(self.embedding)
        self.embedding_override_mask = wp.zeros(
            (1, rows), dtype=wp.bool, device=self.device
        )
        self.supports_embedding_overrides = True

        self.layers = []
        self._build()
        self.graph = None

    def _linear(self, name: str, x: str, weight: str) -> Operation:
        op = Operation("Linear", [x, weight], [name])
        plan_linear(
            op, self.tensors, self.shapes, self.device, cublas=self.runner.cublas
        )
        op.attrs["_sequence"] = (op,)
        return op

    def _rms(self, name: str, x: str, scale: str) -> Operation:
        op = Operation(
            "SimplifiedLayerNormalization",
            [x, scale],
            [name],
            {"epsilon": self.runner.epsilon},
        )
        plan_rms_norm(op, self.tensors, self.shapes, self.device)
        op.attrs["_sequence"] = (op,)
        return op

    def _residual_rms(
        self, name: str, x: str, residual: str, scale: str, residual_name: str
    ) -> Operation:
        op = Operation(
            "SkipSimplifiedLayerNormalization",
            [x, residual, scale],
            [name, "", "", residual_name],
            {"epsilon": self.runner.epsilon},
        )
        plan_residual_rms_norm(op, self.tensors, self.shapes, self.device)
        op.attrs["_sequence"] = (op,)
        return op

    def _build(self) -> None:
        hidden_name = "hidden.0"
        normalized_name = "layer.0.input"
        self.first_norm = self._rms(
            normalized_name, hidden_name, "backbone.layers.0.norm.weight"
        )
        for index, block_type in enumerate(self.runner.pattern):
            prefix = f"backbone.layers.{index}."
            layer = {"type": block_type}
            if block_type == "M":
                self._build_mamba(layer, index, prefix, normalized_name)
            elif block_type == "-":
                self._build_mlp(layer, index, prefix, normalized_name)
            elif block_type == "E":
                self._build_moe(layer, index, prefix, normalized_name)
            else:
                self._build_attention(layer, index, prefix, normalized_name)
            output_name = (
                layer["output_name"]
                if block_type == "E"
                else layer["output"].outputs[0]
            )
            if index + 1 < len(self.runner.pattern):
                next_scale = f"backbone.layers.{index + 1}.norm.weight"
                normalized_name = f"layer.{index + 1}.input"
            else:
                next_scale = "backbone.norm_f.weight"
                normalized_name = "final.normalized"
            hidden_next = f"hidden.{index + 1}"
            layer["next_norm"] = self._residual_rms(
                normalized_name,
                output_name,
                hidden_name,
                next_scale,
                hidden_next,
            )
            hidden_name = hidden_next
            self.layers.append(layer)
        self.lm_head = self._linear("logits", normalized_name, "lm_head.weight")
        self.logits = self.tensors["logits"].reshape(
            (1, self.rows, self.runner.config["vocab_size"])
        )

    def _build_mamba(self, layer: dict, index: int, prefix: str, x: str) -> None:
        mixer = prefix + "mixer."
        layer["projection"] = self._linear(
            f"layer.{index}.mamba_projection", x, mixer + "in_proj.weight"
        )
        layer["gate"] = wp.empty(
            (self.rows, self.runner.mamba_width), dtype=self.dtype, device=self.device
        )
        layer["conv_input"] = wp.empty(
            (self.rows, self.runner.conv_dim), dtype=self.dtype, device=self.device
        )
        layer["dt"] = wp.empty(
            (self.rows, self.runner.mamba_heads), dtype=self.dtype, device=self.device
        )
        layer["conv"] = wp.empty_like(layer["conv_input"])
        layer["x"] = wp.empty(
            (self.rows, self.runner.mamba_width), dtype=self.dtype, device=self.device
        )
        bc_shape = (self.rows, self.runner.groups * self.runner.state_size)
        layer["b"] = wp.empty(bc_shape, dtype=self.dtype, device=self.device)
        layer["c"] = wp.empty(bc_shape, dtype=self.dtype, device=self.device)
        layer["core"] = wp.empty_like(layer["x"])
        layer["gated"] = wp.empty_like(layer["x"])
        scale_dtype = self.runner.weights[mixer + "norm.weight"].dtype
        layer["gated_block"], layer["gated_kernel"] = _get_gated_rms_norm_kernel(
            self.runner.group_width, self.dtype, False, scale_dtype
        )
        if self.rows == 1:
            layer["mamba_block"], layer["mamba_kernel"] = _get_mamba2_decode_kernel(
                self.runner.mamba_head_dim,
                self.runner.state_size,
                self.runner.heads_per_group,
                self.dtype,
            )
        else:
            layer["channel_blocks"], layer["mamba_block"], layer["mamba_kernel"] = (
                _get_mamba2_prefill_kernel(
                    self.runner.mamba_head_dim,
                    self.runner.state_size,
                    self.runner.heads_per_group,
                    self.dtype,
                )
            )
        self.tensors[f"layer.{index}.mamba_gated"] = layer["gated"]
        self.shapes[f"layer.{index}.mamba_gated"] = tuple(layer["gated"].shape)
        layer["output"] = self._linear(
            f"layer.{index}.output",
            f"layer.{index}.mamba_gated",
            mixer + "out_proj.weight",
        )

    def _build_mlp(self, layer: dict, index: int, prefix: str, x: str) -> None:
        mixer = prefix + "mixer."
        layer["up"] = self._linear(f"layer.{index}.mlp_up", x, mixer + "up_proj.weight")
        layer["activated"] = wp.empty_like(self.tensors[layer["up"].outputs[0]])
        self.tensors[f"layer.{index}.mlp_activated"] = layer["activated"]
        self.shapes[f"layer.{index}.mlp_activated"] = tuple(layer["activated"].shape)
        layer["output"] = self._linear(
            f"layer.{index}.output",
            f"layer.{index}.mlp_activated",
            mixer + "down_proj.weight",
        )

    def _build_moe(self, layer: dict, index: int, prefix: str, x: str) -> None:
        mixer = prefix + "mixer."
        plan = SparseExpertPlan(
            self.tensors[x],
            self.runner.weights[mixer + "gate.weight"],
            self.runner.weights[mixer + "gate.e_score_correction_bias"],
            self.runner.weights[mixer + "experts.up_proj.weight"],
            self.runner.weights[mixer + "experts.down_proj.weight"],
            self.runner.weights[mixer + "shared_experts.up_proj.weight"],
            self.runner.weights[mixer + "shared_experts.down_proj.weight"],
            top_k=int(self.runner.config["num_experts_per_tok"]),
            scale=float(self.runner.config["routed_scaling_factor"]),
            groups=int(self.runner.config.get("n_group", 1)),
            topk_groups=int(self.runner.config.get("topk_group", 1)),
            cublas=self.runner.cublas,
        )
        output_name = f"layer.{index}.output"
        layer["moe"] = plan
        layer["output_name"] = output_name
        self.tensors[output_name] = plan.output
        self.shapes[output_name] = tuple(plan.output.shape)

    def _build_attention(self, layer: dict, index: int, prefix: str, x: str) -> None:
        mixer = prefix + "mixer."
        for projection in ("q", "k", "v"):
            layer[projection + "_proj"] = self._linear(
                f"layer.{index}.{projection}_projected",
                x,
                mixer + projection + "_proj.weight",
            )
        layer["q"] = wp.empty(
            (self.runner.query_heads * self.rows, self.runner.attention_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        layer["k"] = wp.empty(
            (self.runner.kv_heads * self.rows, self.runner.attention_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        layer["v"] = wp.empty_like(layer["k"])
        layer["core"] = wp.empty(
            (self.rows, self.runner.query_heads * self.runner.attention_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        layer["attention_block"], layer["attention_kernel"] = _get_gqa_attention_kernel(
            self.runner.attention_head_dim, self.dtype
        )
        self.tensors[f"layer.{index}.attention_core"] = layer["core"]
        self.shapes[f"layer.{index}.attention_core"] = tuple(layer["core"].shape)
        layer["output"] = self._linear(
            f"layer.{index}.output",
            f"layer.{index}.attention_core",
            mixer + "o_proj.weight",
        )

    def _execute_op(self, op: Operation) -> None:
        execute_operations(
            op.attrs["_sequence"], self.tensors, self.shapes, self.device
        )

    def _execute_mamba(self, layer: dict, index: int) -> None:
        self._execute_op(layer["projection"])
        projected = self.tensors[layer["projection"].outputs[0]]
        offset = 0
        for output in (layer["gate"], layer["conv_input"], layer["dt"]):
            wp.launch(
                _split_last_axis_kernel,
                dim=output.shape,
                inputs=[projected, output, offset],
                device=self.device,
            )
            offset += output.shape[1]
        mixer = f"backbone.layers.{index}.mixer."
        wp.launch(
            _causal_conv_rows_kernel,
            dim=layer["conv"].shape,
            inputs=[
                layer["conv_input"],
                self.runner.weights[mixer + "conv1d.weight"],
                self.runner.weights[mixer + "conv1d.bias"],
                self.runner.conv_states[index],
                layer["conv"],
                True,
            ],
            device=self.device,
        )
        offset = 0
        for output in (layer["x"], layer["b"], layer["c"]):
            wp.launch(
                _split_last_axis_kernel,
                dim=output.shape,
                inputs=[layer["conv"], output, offset],
                device=self.device,
            )
            offset += output.shape[1]
        wp.launch(
            _update_conv_rows_state_kernel,
            dim=self.runner.conv_dim,
            inputs=[layer["conv_input"], self.runner.conv_states[index]],
            device=self.device,
        )
        a_log = self.runner.weights[mixer + "A_log"]
        dt_bias = self.runner.weights[mixer + "dt_bias"]
        d = self.runner.weights[mixer + "D"]
        state = self.runner.recurrent_states[index]
        if self.rows == 1:
            wp.launch_tiled(
                layer["mamba_kernel"],
                dim=self.runner.mamba_width,
                inputs=[
                    layer["x"].reshape(
                        (self.runner.mamba_heads, self.runner.mamba_head_dim)
                    ),
                    layer["b"].reshape((self.runner.groups, self.runner.state_size)),
                    layer["c"].reshape((self.runner.groups, self.runner.state_size)),
                    layer["dt"].flatten(),
                    a_log,
                    dt_bias,
                    d,
                    state,
                    layer["core"].reshape(
                        (self.runner.mamba_heads, self.runner.mamba_head_dim)
                    ),
                    self.runner.time_step_min,
                    self.runner.time_step_max,
                ],
                block_dim=layer["mamba_block"],
                device=self.device,
            )
        else:
            wp.launch_tiled(
                layer["mamba_kernel"],
                dim=self.runner.mamba_heads * layer["channel_blocks"],
                inputs=[
                    layer["x"],
                    layer["b"],
                    layer["c"],
                    layer["dt"],
                    a_log,
                    dt_bias,
                    d,
                    state,
                    layer["core"],
                    self.rows,
                    self.runner.time_step_min,
                    self.runner.time_step_max,
                ],
                block_dim=layer["mamba_block"],
                device=self.device,
            )
        wp.launch_tiled(
            layer["gated_kernel"],
            dim=self.rows * self.runner.groups,
            inputs=[
                layer["core"].reshape((-1, self.runner.group_width)),
                layer["gate"].reshape((-1, self.runner.group_width)),
                self.runner.weights[mixer + "norm.weight"].reshape(
                    (self.runner.groups, self.runner.group_width)
                ),
                layer["gated"].reshape((-1, self.runner.group_width)),
                self.runner.epsilon,
            ],
            block_dim=layer["gated_block"],
            device=self.device,
        )
        self._execute_op(layer["output"])

    def _execute_mlp(self, layer: dict) -> None:
        self._execute_op(layer["up"])
        up = self.tensors[layer["up"].outputs[0]]
        wp.launch(
            _relu2_kernel,
            dim=up.shape,
            inputs=[up, layer["activated"]],
            device=self.device,
        )
        self._execute_op(layer["output"])

    def _execute_attention(self, layer: dict, index: int) -> None:
        for projection in ("q", "k", "v"):
            self._execute_op(layer[projection + "_proj"])
            output = layer[projection]
            heads = (
                self.runner.query_heads if projection == "q" else self.runner.kv_heads
            )
            wp.launch(
                _reorder_heads_kernel,
                dim=(self.rows, heads, self.runner.attention_head_dim),
                inputs=[
                    self.tensors[layer[projection + "_proj"].outputs[0]],
                    output,
                    self.runner.attention_head_dim,
                ],
                device=self.device,
            )
        key_cache, value_cache = self.runner.kv_caches[index]
        for source, cache in ((layer["k"], key_cache), (layer["v"], value_cache)):
            wp.launch(
                _append_head_cache_kernel,
                dim=(self.runner.kv_heads, self.rows, self.runner.attention_head_dim),
                inputs=[
                    source,
                    self.position_ids,
                    cache,
                    self.runner.kv_heads,
                    self.runner.attention_head_dim,
                ],
                device=self.device,
            )
        wp.launch_tiled(
            layer["attention_kernel"],
            dim=self.runner.query_heads * self.rows,
            inputs=[
                layer["q"],
                key_cache,
                value_cache,
                self.runner.sequence_end,
                layer["core"],
                self.runner.query_heads,
                self.runner.kv_heads,
                self.rows,
                self.runner.cache_capacity,
                self.runner.attention_head_dim**-0.5,
                0,
            ],
            block_dim=layer["attention_block"],
            device=self.device,
        )
        self._execute_op(layer["output"])

    def execute(self) -> wp.array:
        """Execute the fixed plan on its staged token IDs."""
        wp.launch(
            _gather_rows_kernel,
            dim=self.embedding.shape,
            inputs=[
                self.runner.weights["backbone.embeddings.weight"],
                self.input_ids,
                self.embedding,
            ],
            device=self.device,
        )
        wp.launch(
            _apply_embedding_overrides_kernel,
            dim=self.embedding.shape,
            inputs=[
                self.embedding,
                self.embedding_overrides,
                self.embedding_override_mask,
            ],
            device=self.device,
        )
        self._execute_op(self.first_norm)
        for index, layer in enumerate(self.layers):
            if layer["type"] == "M":
                self._execute_mamba(layer, index)
            elif layer["type"] == "-":
                self._execute_mlp(layer)
            elif layer["type"] == "E":
                layer["moe"].execute()
            else:
                self._execute_attention(layer, index)
            self._execute_op(layer["next_norm"])
        self._execute_op(self.lm_head)
        return self.logits


class NemotronHRunner(AutoregressiveRunner):
    """Run Nemotron-H text and lazily loaded Omni media encoders."""

    def __init__(
        self,
        path: str | Path,
        device: str | wp.Device | None = None,
        cache_capacity: int = 4096,
        prefill_chunk_size: int = 16,
        use_cublas: bool = True,
    ):
        path = Path(path)
        self.model_path = path
        document = json.loads((path / "config.json").read_text(encoding="utf-8"))
        self.config, weight_prefix = _language_config(document)
        _validate_config(self.config)
        self.device = parse_device(device)
        self.cache_capacity = int(cache_capacity)
        if not 0 < self.cache_capacity <= int(self.config["max_position_embeddings"]):
            raise ValueError("cache_capacity must be within max_position_embeddings")
        if not 2 <= prefill_chunk_size <= self.cache_capacity:
            raise ValueError("prefill_chunk_size must be between 2 and cache_capacity")
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.pattern = self.config["hybrid_override_pattern"]
        self.video_pruning_rate = float(document.get("video_pruning_rate", 0.0))
        self.hidden_size = int(self.config["hidden_size"])
        self.mamba_heads = int(self.config["mamba_num_heads"])
        self.mamba_head_dim = int(self.config["mamba_head_dim"])
        self.mamba_width = self.mamba_heads * self.mamba_head_dim
        self.groups = int(self.config["n_groups"])
        self.group_width = self.mamba_width // self.groups
        self.heads_per_group = self.mamba_heads // self.groups
        self.state_size = int(self.config["ssm_state_size"])
        self.conv_dim = self.mamba_width + 2 * self.groups * self.state_size
        self.query_heads = int(self.config["num_attention_heads"])
        self.kv_heads = int(self.config["num_key_value_heads"])
        self.attention_head_dim = int(self.config["attention_head_dim"])
        self.epsilon = float(
            self.config.get(
                "layer_norm_epsilon", self.config.get("rms_norm_eps", 1.0e-5)
            )
        )
        time_step_limit = self.config.get("time_step_limit", (0.0, float("inf")))
        self.time_step_min, self.time_step_max = (
            float(value) for value in time_step_limit
        )

        archive = SafeTensorNamespace(SafeTensorArchive(path), weight_prefix)
        names = _weight_names(self.config)
        missing = set(names) - set(archive.names)
        if missing:
            raise ValueError(f"Nemotron-H checkpoint is missing {sorted(missing)[:5]}")
        embedding_dtype = archive.metadata("backbone.embeddings.weight").dtype
        if embedding_dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Nemotron-H embeddings must use FP16 or BF16")
        required_bytes = sum(
            archive.metadata(name).nbytes
            * (2 if archive.metadata(name).format == "F8_E4M3" else 1)
            for name in names
        )
        attention_layers = self.pattern.count("*")
        required_bytes += (
            attention_layers
            * 2
            * self.kv_heads
            * self.cache_capacity
            * self.attention_head_dim
            * 2
        )
        if self.device.is_cuda and required_bytes > self.device.free_memory * 0.95:
            raise MemoryError(
                f"Nemotron-H needs at least {required_bytes / 2**30:.1f} GiB for weights and KV cache; "
                f"{self.device.free_memory / 2**30:.1f} GiB is currently free"
            )
        self.dtype = embedding_dtype
        self.weights = _load_weights(archive, names, self.device, self.dtype)
        self.cublas = (
            try_create_cublas() if use_cublas and self.device.is_cuda else None
        )
        self.sequence_end = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.conv_states = {}
        self.recurrent_states = {}
        self.kv_caches = {}
        for index, block_type in enumerate(self.pattern):
            if block_type == "M":
                self.conv_states[index] = wp.zeros(
                    (self.conv_dim, int(self.config["conv_kernel"]) - 1),
                    dtype=self.dtype,
                    device=self.device,
                )
                self.recurrent_states[index] = wp.zeros(
                    (self.mamba_width, self.state_size),
                    dtype=wp.float32,
                    device=self.device,
                )
            elif block_type == "*":
                shape = (self.kv_heads * self.cache_capacity, self.attention_head_dim)
                self.kv_caches[index] = (
                    wp.empty(shape, dtype=self.dtype, device=self.device),
                    wp.empty(shape, dtype=self.dtype, device=self.device),
                )
        self._decode_plan = _NemotronPlan(self, 1)
        self._chunk_plan = _NemotronPlan(self, self.prefill_chunk_size)
        self._initialize_sampling()
        self.sequence_length = 0

    def _vision_encoder(self):
        encoder = getattr(self, "_vision_encoder_instance", None)
        if encoder is None:
            from .vision import NemotronVisionEncoder

            encoder = self._vision_encoder_instance = NemotronVisionEncoder(
                self.model_path, device=self.device, cublas=self.cublas
            )
        return encoder

    def _audio_encoder(self):
        encoder = getattr(self, "_audio_encoder_instance", None)
        if encoder is None:
            from .audio import NemotronAudioEncoder

            encoder = self._audio_encoder_instance = NemotronAudioEncoder(
                self.model_path, device=self.device, cublas=self.cublas
            )
        return encoder

    def prefill_multimodal(self, prompt) -> wp.array:
        """Encode image/audio/video inputs and prefill compact device overrides."""
        from .omni import NemotronMultimodalPrompt

        if not isinstance(prompt, NemotronMultimodalPrompt):
            raise TypeError("prefill_multimodal expects NemotronMultimodalPrompt")
        if not prompt.images and not prompt.audios and not prompt.videos:
            return self.prefill(prompt.token_ids)
        media_outputs = []
        removed_positions = set()
        total = 0
        for start, media in zip(prompt.image_starts, prompt.images, strict=True):
            output = self._vision_encoder().encode(media)
            if output.shape != (media.tokens, self.hidden_size):
                raise ValueError(
                    "vision encoder output does not match its prompt tokens"
                )
            if output.dtype != self.dtype or output.device != self.device:
                raise TypeError(
                    "vision encoder output must match language dtype and device"
                )
            media_outputs.append((tuple(range(start, start + media.tokens)), output))
            total += media.tokens
        for start, media in zip(prompt.audio_starts, prompt.audios, strict=True):
            output = self._audio_encoder().encode(media)
            tokens = output.shape[0] * output.shape[1]
            output = output.reshape((tokens, self.hidden_size))
            if output.dtype != self.dtype or output.device != self.device:
                raise TypeError(
                    "audio encoder output must match language dtype and device"
                )
            media_outputs.append((tuple(range(start, start + tokens)), output))
            total += tokens
        for starts, media in zip(prompt.video_starts, prompt.videos, strict=True):
            from .video import prune_video_embeddings

            output = self._vision_encoder().encode_video(media)
            if output.shape != (media.tokens, self.hidden_size):
                raise ValueError(
                    "video encoder output does not match its prompt tokens"
                )
            if output.dtype != self.dtype or output.device != self.device:
                raise TypeError(
                    "video encoder output must match language dtype and device"
                )
            positions = []
            for start in starts:
                positions.extend(range(start, start + media.tokens_per_group))
            output, retained = prune_video_embeddings(
                output,
                media.groups,
                media.tokens_per_group,
                self.video_pruning_rate,
            )
            retained_positions = tuple(positions[index] for index in retained)
            removed_positions.update(set(positions) - set(retained_positions))
            media_outputs.append((retained_positions, output))
            total += output.shape[0]
        token_ids = prompt.token_ids
        if removed_positions:
            remap = {}
            compact_ids = []
            for old_position, token_id in enumerate(token_ids):
                if old_position not in removed_positions:
                    remap[old_position] = len(compact_ids)
                    compact_ids.append(token_id)
            token_ids = tuple(compact_ids)
            media_outputs = [
                (tuple(remap[position] for position in positions), output)
                for positions, output in media_outputs
            ]
        media_outputs.sort(key=lambda item: item[0][0])
        positions = []
        embeddings = wp.empty(
            (total, self.hidden_size), dtype=self.dtype, device=self.device
        )
        offset = 0
        for output_positions, output in media_outputs:
            positions.extend(output_positions)
            wp.copy(
                embeddings.flatten(),
                output.flatten(),
                dest_offset=offset * self.hidden_size,
                count=output.size,
            )
            offset += output.shape[0]
        return self.prefill_with_embeddings(token_ids, embeddings, positions)
