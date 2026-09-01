# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-only Nemotron-H runner for Hugging Face safetensors checkpoints."""

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
    execute_operations,
    plan_linear,
    plan_residual_rms_norm,
    plan_rms_norm,
)
from warp_nn.runtime.autoregressive import AutoregressiveRunner
from warp_nn.runtime.formats.safetensors import SafeTensorArchive
from warp_nn.utils.device import parse_device


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
    if len(pattern) != int(config["num_hidden_layers"]) or set(pattern) - set("M-*"):
        raise ValueError("Nemotron-H hybrid pattern must contain one M, -, or * per layer")
    if int(config["mamba_num_heads"]) % int(config["n_groups"]):
        raise ValueError("Nemotron-H Mamba heads must be divisible by its groups")
    if int(config["num_attention_heads"]) % int(config["num_key_value_heads"]):
        raise ValueError("Nemotron-H query heads must be divisible by KV heads")
    if config.get("mamba_hidden_act", "silu") != "silu" or config.get("mlp_hidden_act", "relu2") != "relu2":
        raise ValueError("Nemotron-H runner requires SiLU Mamba and ReLU-squared MLP blocks")
    if any(config.get(name, False) for name in ("attention_bias", "mamba_proj_bias", "mlp_bias", "use_bias")):
        raise ValueError("Biased Nemotron-H projections are not supported")


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
            names.extend(prefix + "mixer." + suffix for suffix in ("up_proj.weight", "down_proj.weight"))
        else:
            names.extend(
                prefix + "mixer." + suffix
                for suffix in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")
            )
    return names


def _load_weights(archive: SafeTensorArchive, names: list[str], device, dtype: type) -> dict[str, wp.array]:
    fp8 = [name for name in names if archive.metadata(name).format == "F8_E4M3"]
    scale_names = [name + "_scale" for name in fp8]
    missing_scales = set(scale_names) - set(archive.names)
    if missing_scales:
        raise ValueError(f"Nemotron-H checkpoint is missing {sorted(missing_scales)[:5]}")
    weights = archive.load(device, [name for name in names if name not in fp8] + scale_names)
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
        self.embedding = wp.empty((1, rows, runner.hidden_size), dtype=self.dtype, device=self.device)
        self.tensors["hidden.0"] = self.embedding.reshape((rows, runner.hidden_size))
        self.shapes["hidden.0"] = (rows, runner.hidden_size)
        self.layers = []
        self._build()
        self.graph = None

    def _linear(self, name: str, x: str, weight: str) -> Operation:
        op = Operation("Linear", [x, weight], [name])
        plan_linear(op, self.tensors, self.shapes, self.device, cublas=self.runner.cublas)
        op.attrs["_sequence"] = (op,)
        return op

    def _rms(self, name: str, x: str, scale: str) -> Operation:
        op = Operation("SimplifiedLayerNormalization", [x, scale], [name], {"epsilon": self.runner.epsilon})
        plan_rms_norm(op, self.tensors, self.shapes, self.device)
        op.attrs["_sequence"] = (op,)
        return op

    def _residual_rms(self, name: str, x: str, residual: str, scale: str, residual_name: str) -> Operation:
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
        self.first_norm = self._rms(normalized_name, hidden_name, "backbone.layers.0.norm.weight")
        for index, block_type in enumerate(self.runner.pattern):
            prefix = f"backbone.layers.{index}."
            layer = {"type": block_type}
            if block_type == "M":
                self._build_mamba(layer, index, prefix, normalized_name)
            elif block_type == "-":
                self._build_mlp(layer, index, prefix, normalized_name)
            else:
                self._build_attention(layer, index, prefix, normalized_name)
            if index + 1 < len(self.runner.pattern):
                next_scale = f"backbone.layers.{index + 1}.norm.weight"
                normalized_name = f"layer.{index + 1}.input"
            else:
                next_scale = "backbone.norm_f.weight"
                normalized_name = "final.normalized"
            hidden_next = f"hidden.{index + 1}"
            layer["next_norm"] = self._residual_rms(
                normalized_name, layer["output"].outputs[0], hidden_name, next_scale, hidden_next
            )
            hidden_name = hidden_next
            self.layers.append(layer)
        self.lm_head = self._linear("logits", normalized_name, "lm_head.weight")
        self.logits = self.tensors["logits"].reshape((1, self.rows, self.runner.config["vocab_size"]))

    def _build_mamba(self, layer: dict, index: int, prefix: str, x: str) -> None:
        mixer = prefix + "mixer."
        layer["projection"] = self._linear(f"layer.{index}.mamba_projection", x, mixer + "in_proj.weight")
        layer["gate"] = wp.empty((self.rows, self.runner.mamba_width), dtype=self.dtype, device=self.device)
        layer["conv_input"] = wp.empty((self.rows, self.runner.conv_dim), dtype=self.dtype, device=self.device)
        layer["dt"] = wp.empty((self.rows, self.runner.mamba_heads), dtype=self.dtype, device=self.device)
        layer["conv"] = wp.empty_like(layer["conv_input"])
        layer["x"] = wp.empty((self.rows, self.runner.mamba_width), dtype=self.dtype, device=self.device)
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
                self.runner.mamba_head_dim, self.runner.state_size, self.runner.heads_per_group, self.dtype
            )
        else:
            layer["channel_blocks"], layer["mamba_block"], layer["mamba_kernel"] = _get_mamba2_prefill_kernel(
                self.runner.mamba_head_dim, self.runner.state_size, self.runner.heads_per_group, self.dtype
            )
        self.tensors[f"layer.{index}.mamba_gated"] = layer["gated"]
        self.shapes[f"layer.{index}.mamba_gated"] = tuple(layer["gated"].shape)
        layer["output"] = self._linear(
            f"layer.{index}.output", f"layer.{index}.mamba_gated", mixer + "out_proj.weight"
        )

    def _build_mlp(self, layer: dict, index: int, prefix: str, x: str) -> None:
        mixer = prefix + "mixer."
        layer["up"] = self._linear(f"layer.{index}.mlp_up", x, mixer + "up_proj.weight")
        layer["activated"] = wp.empty_like(self.tensors[layer["up"].outputs[0]])
        self.tensors[f"layer.{index}.mlp_activated"] = layer["activated"]
        self.shapes[f"layer.{index}.mlp_activated"] = tuple(layer["activated"].shape)
        layer["output"] = self._linear(
            f"layer.{index}.output", f"layer.{index}.mlp_activated", mixer + "down_proj.weight"
        )

    def _build_attention(self, layer: dict, index: int, prefix: str, x: str) -> None:
        mixer = prefix + "mixer."
        for projection in ("q", "k", "v"):
            layer[projection + "_proj"] = self._linear(
                f"layer.{index}.{projection}_projected", x, mixer + projection + "_proj.weight"
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
            f"layer.{index}.output", f"layer.{index}.attention_core", mixer + "o_proj.weight"
        )

    def _execute_op(self, op: Operation) -> None:
        execute_operations(op.attrs["_sequence"], self.tensors, self.shapes, self.device)

    def _execute_mamba(self, layer: dict, index: int) -> None:
        self._execute_op(layer["projection"])
        projected = self.tensors[layer["projection"].outputs[0]]
        offset = 0
        for output in (layer["gate"], layer["conv_input"], layer["dt"]):
            wp.launch(_split_last_axis_kernel, dim=output.shape, inputs=[projected, output, offset], device=self.device)
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
            wp.launch(_split_last_axis_kernel, dim=output.shape, inputs=[layer["conv"], output, offset], device=self.device)
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
                    layer["x"].reshape((self.runner.mamba_heads, self.runner.mamba_head_dim)),
                    layer["b"].reshape((self.runner.groups, self.runner.state_size)),
                    layer["c"].reshape((self.runner.groups, self.runner.state_size)),
                    layer["dt"].flatten(),
                    a_log,
                    dt_bias,
                    d,
                    state,
                    layer["core"].reshape((self.runner.mamba_heads, self.runner.mamba_head_dim)),
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
                self.runner.weights[mixer + "norm.weight"].reshape((self.runner.groups, self.runner.group_width)),
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
        wp.launch(_relu2_kernel, dim=up.shape, inputs=[up, layer["activated"]], device=self.device)
        self._execute_op(layer["output"])

    def _execute_attention(self, layer: dict, index: int) -> None:
        for projection in ("q", "k", "v"):
            self._execute_op(layer[projection + "_proj"])
            output = layer[projection]
            heads = self.runner.query_heads if projection == "q" else self.runner.kv_heads
            wp.launch(
                _reorder_heads_kernel,
                dim=(self.rows, heads, self.runner.attention_head_dim),
                inputs=[self.tensors[layer[projection + "_proj"].outputs[0]], output, self.runner.attention_head_dim],
                device=self.device,
            )
        key_cache, value_cache = self.runner.kv_caches[index]
        for source, cache in ((layer["k"], key_cache), (layer["v"], value_cache)):
            wp.launch(
                _append_head_cache_kernel,
                dim=(self.runner.kv_heads, self.rows, self.runner.attention_head_dim),
                inputs=[source, self.position_ids, cache, self.runner.kv_heads, self.runner.attention_head_dim],
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
            inputs=[self.runner.weights["backbone.embeddings.weight"], self.input_ids, self.embedding],
            device=self.device,
        )
        self._execute_op(self.first_norm)
        for index, layer in enumerate(self.layers):
            if layer["type"] == "M":
                self._execute_mamba(layer, index)
            elif layer["type"] == "-":
                self._execute_mlp(layer)
            else:
                self._execute_attention(layer, index)
            self._execute_op(layer["next_norm"])
        self._execute_op(self.lm_head)
        return self.logits


class NemotronHRunner(AutoregressiveRunner):
    """Run a text-only Nemotron-H FP8 or BF16 safetensors checkpoint."""

    def __init__(
        self,
        path: str | Path,
        device: str | wp.Device | None = None,
        cache_capacity: int = 4096,
        prefill_chunk_size: int = 16,
        use_cublas: bool = True,
    ):
        path = Path(path)
        self.config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        _validate_config(self.config)
        self.device = parse_device(device)
        self.cache_capacity = int(cache_capacity)
        if not 0 < self.cache_capacity <= int(self.config["max_position_embeddings"]):
            raise ValueError("cache_capacity must be within max_position_embeddings")
        if not 2 <= prefill_chunk_size <= self.cache_capacity:
            raise ValueError("prefill_chunk_size must be between 2 and cache_capacity")
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.pattern = self.config["hybrid_override_pattern"]
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
        self.epsilon = float(self.config.get("layer_norm_epsilon", self.config.get("rms_norm_eps", 1.0e-5)))
        time_step_limit = self.config.get("time_step_limit", (0.0, float("inf")))
        self.time_step_min, self.time_step_max = (float(value) for value in time_step_limit)

        archive = SafeTensorArchive(path)
        names = _weight_names(self.config)
        missing = set(names) - set(archive.names)
        if missing:
            raise ValueError(f"Nemotron-H checkpoint is missing {sorted(missing)[:5]}")
        embedding_dtype = archive.metadata("backbone.embeddings.weight").dtype
        if embedding_dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Nemotron-H embeddings must use FP16 or BF16")
        required_bytes = sum(
            archive.metadata(name).nbytes * (2 if archive.metadata(name).format == "F8_E4M3" else 1)
            for name in names
        )
        attention_layers = self.pattern.count("*")
        required_bytes += attention_layers * 2 * self.kv_heads * self.cache_capacity * self.attention_head_dim * 2
        if self.device.is_cuda and required_bytes > self.device.free_memory * 0.95:
            raise MemoryError(
                f"Nemotron-H needs at least {required_bytes / 2**30:.1f} GiB for weights and KV cache; "
                f"{self.device.free_memory / 2**30:.1f} GiB is currently free"
            )
        self.dtype = embedding_dtype
        self.weights = _load_weights(archive, names, self.device, self.dtype)
        self.cublas = try_create_cublas() if use_cublas and self.device.is_cuda else None
        self.sequence_end = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.conv_states = {}
        self.recurrent_states = {}
        self.kv_caches = {}
        for index, block_type in enumerate(self.pattern):
            if block_type == "M":
                self.conv_states[index] = wp.zeros(
                    (self.conv_dim, int(self.config["conv_kernel"]) - 1), dtype=self.dtype, device=self.device
                )
                self.recurrent_states[index] = wp.zeros(
                    (self.mamba_width, self.state_size), dtype=wp.float32, device=self.device
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
