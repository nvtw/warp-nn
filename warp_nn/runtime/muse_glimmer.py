# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-only Muse Glimmer runner for safetensors and unquantized GGUF checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import json
from pathlib import Path
import re

import numpy as np
import warp as wp

from warp_nn.runtime._cublas import try_create_cublas
from warp_nn.runtime.gguf import GGUFArchive
from warp_nn.runtime.kernels import (
    _append_circular_head_cache_kernel,
    _append_head_cache_kernel,
    _binary_broadcast_kernel,
    _gather_rows_kernel,
    _get_gqa_attention_kernel,
    _get_greedy_argmax_kernels,
    _allocate_partitioned_gqa,
    _launch_partitioned_gqa,
    _logit_softcap_kernel,
    _reorder_interleaved_heads_kernel,
    _reorder_heads_kernel,
    _rotary_embedding_kernel_for_dtype,
    _scale_kernel,
    _sigmoid_gate_kernel,
)
from warp_nn.runtime.operators import (
    Operation,
    execute_operations,
    plan_linear,
    plan_rms_norm,
    plan_swiglu,
)
from warp_nn.runtime.qwen35 import Qwen35Runner
from warp_nn.runtime.qwen3 import Qwen3Tokenizer, _pretokenize_o200k
from warp_nn.runtime.safetensors import SafeTensorArchive
from warp_nn.utils.device import parse_device


def _gguf_paths(path: str | Path) -> tuple[Path, ...]:
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    if path.is_dir():
        first_files = sorted(directory.glob("*-00001-of-*.gguf"))
        if not first_files:
            first_files = [
                item
                for item in sorted(directory.glob("*.gguf"))
                if "mmproj" not in item.name.lower()
            ]
        if len(first_files) != 1:
            raise FileNotFoundError(
                f"Expected one GGUF model in '{directory}', found {len(first_files)}"
            )
        path = first_files[0]
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
    if match is None:
        return (path,)
    count = int(match.group(3))
    return tuple(
        directory / f"{match.group(1)}-{index:05d}-of-{count:05d}.gguf"
        for index in range(1, count + 1)
    )


def _gguf_config(metadata: Mapping[str, object]) -> dict:
    if metadata.get("general.architecture") != "muse-glimmer":
        raise ValueError("GGUF checkpoint is not a Muse Glimmer model")
    prefix = "muse-glimmer."
    layers = int(metadata[prefix + "block_count"])
    pattern = int(metadata[prefix + "attention.sliding_window_pattern"])
    rope_theta = float(metadata[prefix + "rope.freq_base"])
    layer_types = [
        "full_attention" if (index + 1) % pattern == 0 else "sliding_attention"
        for index in range(layers)
    ]
    return {
        "model_type": "muse_glimmer_text",
        "hidden_size": int(metadata[prefix + "embedding_length"]),
        "intermediate_size": int(metadata[prefix + "feed_forward_length"]),
        "vocab_size": len(metadata["tokenizer.ggml.tokens"]),
        "num_hidden_layers": layers,
        "layer_types": layer_types,
        "layer_rope_theta": [
            0.0 if kind == "full_attention" else rope_theta for kind in layer_types
        ],
        "num_attention_heads": int(metadata[prefix + "attention.head_count"]),
        "num_key_value_heads": int(metadata[prefix + "attention.head_count_kv"]),
        "head_dim": int(metadata[prefix + "attention.key_length"]),
        "max_position_embeddings": int(metadata[prefix + "context_length"]),
        "sliding_window": int(metadata[prefix + "attention.sliding_window"]),
        "qk_scale_factor": 3.87,
        "rms_norm_eps": float(metadata[prefix + "attention.layer_norm_rms_epsilon"]),
        "post_norm_eps": 1.0e-8,
        "output_multiplier": float(metadata[prefix + "logit_scale"]),
        "final_logit_softcapping": float(metadata[prefix + "final_logit_softcapping"]),
        "hidden_activation": "silu",
        "attention_bias": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": rope_theta},
    }


def _gguf_weight_map(config: Mapping[str, object]) -> dict[str, str]:
    names = {
        "model.language_model.embed_tokens.weight": "token_embd.weight",
        "model.language_model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    suffixes = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight": "ffn_norm.weight",
        "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.gate_proj.weight": "attn_gate.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    for index in range(int(config["num_hidden_layers"])):
        prefix = f"model.language_model.layers.{index}."
        names.update(
            {
                prefix + target: f"blk.{index}.{source}"
                for target, source in suffixes.items()
            }
        )
    return names


class _MappedGGUFArchive:
    def __init__(self, archive: GGUFArchive, names: Mapping[str, str]):
        self.archive = archive
        self._names = dict(names)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._names)

    def metadata(self, name: str):
        return self.archive.tensor(self._names[name])

    def load(self, device=None, names=None) -> dict[str, wp.array]:
        selected = self.names if names is None else tuple(names)
        loaded = self.archive.load(device, [self._names[name] for name in selected])
        return {name: loaded[self._names[name]] for name in selected}


def _validate_config(config: dict) -> None:
    required = (
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_hidden_layers",
        "layer_types",
        "layer_rope_theta",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "sliding_window",
        "qk_scale_factor",
        "rms_norm_eps",
        "post_norm_eps",
        "output_multiplier",
        "final_logit_softcapping",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Muse Glimmer text config is missing {missing}")
    layers = int(config["num_hidden_layers"])
    if (
        len(config["layer_types"]) != layers
        or len(config["layer_rope_theta"]) != layers
    ):
        raise ValueError("Muse Glimmer layer metadata must match num_hidden_layers")
    if set(config["layer_types"]) - {"sliding_attention", "full_attention"}:
        raise ValueError(
            "Muse Glimmer supports sliding_attention and full_attention layers"
        )
    if any(
        (kind == "sliding_attention") != bool(theta)
        for kind, theta in zip(config["layer_types"], config["layer_rope_theta"])
    ):
        raise ValueError(
            "Muse Glimmer expects RoPE on sliding layers and NoPE on full layers"
        )
    if int(config["num_attention_heads"]) % int(config["num_key_value_heads"]):
        raise ValueError("Muse Glimmer query heads must be divisible by KV heads")
    if config.get("hidden_activation", "silu") != "silu" or config.get(
        "attention_bias", False
    ):
        raise ValueError("Muse Glimmer runner requires SiLU and bias-free attention")
    rope = config.get("rope_parameters", {})
    if rope.get("rope_type", "default") != "default":
        raise ValueError("Muse Glimmer runner supports default rotary embeddings")


def _weight_names(config: dict) -> list[str]:
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.gate_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    for index in range(int(config["num_hidden_layers"])):
        prefix = f"model.language_model.layers.{index}."
        names.extend(prefix + suffix for suffix in suffixes)
    return names


def _atem_tool_definitions(tools: Sequence[Mapping[str, object]]) -> str:
    schemas = []
    namespaces = []
    for tool in tools:
        function = tool.get("function", tool)
        name = function["name"]
        namespace = name.split(".", 1)[0]
        if namespace not in namespaces:
            namespaces.append(namespace)
        schemas.append(
            json.dumps(
                {
                    "name": name,
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return (
        "In this environment you have access to a set of tools you can use to answer the user's question.\n\n"
        'Invoke a function with <atem:function_calls><atem:invoke name="$FUNCTION_NAME"> and one '
        '<atem:parameter name="$PARAMETER_NAME">value</atem:parameter> per argument. Lists and objects use JSON.\n\n'
        "// Function schemas\n" + "\n".join(schemas)
    )


def parse_atem_tool_calls(text: str) -> tuple[str, list[dict[str, object]]]:
    """Extract Muse ATEM function calls and return remaining assistant text."""
    calls = []
    invoke_pattern = re.compile(
        r'<atem:invoke\s+name="([^"]+)">(.*?)</atem:invoke>', re.DOTALL
    )
    parameter_pattern = re.compile(
        r'<atem:parameter\s+name="([^"]+)">(.*?)</atem:parameter>', re.DOTALL
    )
    for invoke in invoke_pattern.finditer(text):
        arguments = {}
        for parameter in parameter_pattern.finditer(invoke.group(2)):
            value = parameter.group(2)
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            arguments[parameter.group(1)] = value
        calls.append({"name": invoke.group(1), "arguments": arguments})
    clean = re.sub(
        r"<atem:function_calls>.*?</atem:function_calls>", "", text, flags=re.DOTALL
    ).strip()
    return clean, calls


class _MuseStreamFilter:
    """Stream only Muse's user channel while retaining partial control tags."""

    _end_markers = ("<|eom|>", "<|eot|>")

    def __init__(self):
        self.buffer = ""
        self.visible = False
        self.in_content = False

    def feed(self, text: str, final: bool = False) -> str:
        self.buffer += text
        output = []
        while self.buffer:
            if not self.in_content:
                marker = self.buffer.find("<|message|>")
                if marker < 0:
                    if final:
                        self.buffer = ""
                    break
                header = self.buffer[:marker]
                recipient = re.search(r"(?:^|\s)to=([^<\s]+)", header)
                self.visible = recipient is not None and recipient.group(1) == "user"
                self.buffer = self.buffer[marker + len("<|message|>") :]
                self.in_content = True
                continue
            ends = [(self.buffer.find(marker), marker) for marker in self._end_markers]
            ends = [(position, marker) for position, marker in ends if position >= 0]
            if ends:
                position, marker = min(ends)
                if self.visible:
                    output.append(self.buffer[:position])
                self.buffer = self.buffer[position + len(marker) :]
                self.in_content = False
                self.visible = False
                continue
            keep = max(len(marker) for marker in self._end_markers) - 1
            if final:
                if self.visible:
                    output.append(self.buffer)
                self.buffer = ""
            elif len(self.buffer) > keep:
                if self.visible:
                    output.append(self.buffer[:-keep])
                self.buffer = self.buffer[-keep:]
            break
        return "".join(output)


class MuseGlimmerTokenizer(Qwen3Tokenizer):
    """Dependency-free o200k tokenizer and ATEM text/tool chat formatter."""

    tool_call_start = "<atem:function_calls>"

    def __init__(self, path: str | Path):
        path = Path(path)
        if (path / "tokenizer.json").is_file():
            super().__init__(path)
        else:
            archive = GGUFArchive(_gguf_paths(path))
            metadata = archive.metadata
            if metadata.get("tokenizer.ggml.model") != "gpt2":
                raise ValueError("Muse GGUF requires an embedded GPT-2 BPE tokenizer")
            tokens = metadata["tokenizer.ggml.tokens"]
            token_types = metadata["tokenizer.ggml.token_type"]
            merges = [
                merge.split(" ", 1) for merge in metadata["tokenizer.ggml.merges"]
            ]
            added = [
                {"id": token_id, "content": token, "special": token_type == 3}
                for token_id, (token, token_type) in enumerate(zip(tokens, token_types))
                if token_type != 1
            ]
            eos = int(metadata["tokenizer.ggml.eos_token_id"])
            eot = int(metadata["tokenizer.ggml.eot_token_id"])
            super().__init__(
                {
                    "normalizer": None,
                    "added_tokens": added,
                    "model": {
                        "type": "BPE",
                        "vocab": dict(zip(tokens, range(len(tokens)))),
                        "merges": merges,
                    },
                    "generation_config": {
                        "eos_token_id": [eos, eot],
                        "pad_token_id": int(
                            metadata["tokenizer.ggml.padding_token_id"]
                        ),
                    },
                    "chat_template": metadata.get("tokenizer.chat_template", ""),
                }
            )
        self._pretokenize = _pretokenize_o200k
        self.supports_reasoning_effort = True
        self.default_enable_thinking = True

    @staticmethod
    def _reasoning_strength(enable_thinking: bool, reasoning_effort: str | None) -> str:
        if not enable_thinking:
            return "low"
        return {None: "high", "xhigh": "high", "medium": "medium", "low": "low"}.get(
            reasoning_effort, reasoning_effort
        )

    @staticmethod
    def _tool_call_text(call: Mapping[str, object]) -> str:
        function = call.get("function", call)
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        parameters = []
        for name, value in arguments.items():
            structured = isinstance(value, (dict, list, bool)) or value is None
            encoded = json.dumps(value, ensure_ascii=False) if structured else value
            parameters.append(
                f'<atem:parameter name="{name}">{encoded}</atem:parameter>\n'
            )
        return (
            f'<atem:function_calls>\n<atem:invoke name="{function["name"]}">\n{"".join(parameters)}'
            "</atem:invoke>\n</atem:function_calls>"
        )

    def format_chat(
        self,
        messages: Sequence[Mapping[str, object]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        tools: Sequence[Mapping[str, object]] | None = None,
        reasoning_effort: str | None = None,
        preserve_thinking: bool = True,
    ) -> str:
        """Format text messages and OpenAI function tools using Muse ATEM."""
        del preserve_thinking
        strength = self._reasoning_strength(enable_thinking, reasoning_effort)
        recipients = ['"self"']
        if tools:
            for tool in tools:
                namespace = tool.get("function", tool)["name"].split(".", 1)[0]
                recipient = f'"{namespace}.*"'
                if recipient not in recipients:
                    recipients.append(recipient)
        recipients.append('"user"')
        meta = f"# Valid recipients: {', '.join(recipients)}."
        output = ["<|begin_of_text|>"]
        has_system = any(message.get("role") == "system" for message in messages)
        if not has_system:
            system = (
                "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\n"
                f"Current date: {date.today().isoformat()}.\n\nReasoning strength: {strength}."
            )
            if tools:
                system += "\n\n" + _atem_tool_definitions(tools)
            output.append(f"<|start|>system<|message|>{system}\n\n{meta}<|eot|>")

        call_names = {}
        for prior in messages:
            for call in prior.get("tool_calls") or ():
                if call.get("id"):
                    call_names[call["id"]] = call.get("function", call)["name"]
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("MuseGlimmerTokenizer supports text message content")
            content = "" if content is None else content
            if role == "system":
                system = re.sub(
                    "reasoning effort",
                    "reasoning strength",
                    content,
                    flags=re.IGNORECASE,
                )
                if "reasoning strength" not in system.lower():
                    system += f"\n\nReasoning strength: {strength}."
                if tools:
                    system += "\n\n" + _atem_tool_definitions(tools)
                output.append(f"<|start|>system<|message|>{system}\n\n{meta}<|eot|>")
            elif role == "user":
                output.append(f"<|start|>user<|message|>{content}<|eot|>")
            elif role == "assistant":
                raw_token_ids = message.get("_raw_token_ids")
                if isinstance(raw_token_ids, (list, tuple)):
                    output.append(
                        f"<|start|>assistant{self.decode(raw_token_ids, skip_special_tokens=False)}"
                    )
                    continue
                reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    output.append(
                        f"<|start|>assistant to=self<|message|>{reasoning}<|eom|>"
                    )
                calls = message.get("tool_calls") or ()
                if calls:
                    for call in calls:
                        name = call.get("function", call)["name"]
                        output.append(
                            f"<|start|>assistant to={name}<|message|>{self._tool_call_text(call)}<|eot|>"
                        )
                else:
                    output.append(
                        f"<|start|>assistant to=user<|message|>{content}<|eot|>"
                    )
            elif role == "tool":
                call_id = message.get("tool_call_id", "")
                name = message.get("name") or call_names.get(call_id, call_id)
                output.append(
                    f'<|start|>tool {name}<|message|><tool_output name="{name}">\n{content}\n</tool_output><|eot|>'
                )
            else:
                raise ValueError(f"MuseGlimmerTokenizer does not support role {role!r}")
        if add_generation_prompt:
            output.append("<|start|>assistant")
        return "".join(output)

    def encode_chat(
        self, messages: Sequence[Mapping[str, object]], **kwargs
    ) -> list[int]:
        """Format chat while retaining the model's exact generated token sequence."""
        rendered = self.format_chat(messages, **kwargs)
        raw_messages = [
            message
            for message in messages
            if isinstance(message.get("_raw_token_ids"), (list, tuple))
        ]
        if not raw_messages:
            return self.encode(rendered)
        encoded = []
        cursor = 0
        anchor = "<|start|>assistant"
        for message in raw_messages:
            raw_token_ids = message["_raw_token_ids"]
            raw = self.decode(raw_token_ids, skip_special_tokens=False)
            start = rendered.find(anchor + raw, cursor)
            if start < 0:
                raise ValueError(
                    "Muse raw assistant history is inconsistent with the formatted chat"
                )
            raw_start = start + len(anchor)
            encoded.extend(self.encode(rendered[cursor:raw_start]))
            encoded.extend(int(token_id) for token_id in raw_token_ids)
            cursor = raw_start + len(raw)
        encoded.extend(self.encode(rendered[cursor:]))
        return encoded

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = False
    ) -> str:
        """Decode tokens, reducing generated Muse channels to user/tool content."""
        raw = super().decode(token_ids, skip_special_tokens=False)
        if not skip_special_tokens:
            return raw
        candidate = raw
        if raw.startswith(" to=") or raw.startswith("<|message|>"):
            candidate = "<|start|>assistant" + raw
        channels = re.findall(
            r"<\|start\|>assistant(?: to=([^<]+))?<\|message\|>(.*?)(?:<\|eom\|>|<\|eot\|>|$)",
            candidate,
            re.DOTALL,
        )
        if not channels:
            return super().decode(token_ids, skip_special_tokens=True)
        visible = [content for recipient, content in channels if recipient != "self"]
        return "".join(visible)

    def generation_prefix(self, enable_thinking: bool) -> str:
        """Return the empty suffix after Muse's assistant start anchor."""
        del enable_thinking
        return ""

    def stream_filter(self) -> _MuseStreamFilter:
        """Create a per-response ATEM channel filter for incremental output."""
        return _MuseStreamFilter()

    def parse_tool_calls(self, text: str) -> tuple[str, list[dict[str, object]]]:
        """Extract structured ATEM calls from Muse assistant text."""
        return parse_atem_tool_calls(text)


class _MusePlan:
    """Fixed-row Muse execution plan sharing the runner's persistent caches."""

    def __init__(self, runner: MuseGlimmerRunner, rows: int):
        self.runner = runner
        self.rows = rows
        self.device = runner.device
        self.dtype = runner.dtype
        self.tensors = dict(runner.weights)
        self.tensors.update(runner.unit_scales)
        self.shapes = {name: tuple(value.shape) for name, value in self.tensors.items()}
        self.input_ids = wp.zeros((1, rows), dtype=wp.int64, device=self.device)
        self.position_ids = wp.zeros((1, rows), dtype=wp.int64, device=self.device)
        self.embedding = wp.empty(
            (1, rows, runner.hidden_size), dtype=self.dtype, device=self.device
        )
        self.tensors["hidden.0"] = self.embedding.reshape((rows, runner.hidden_size))
        self.shapes["hidden.0"] = (rows, runner.hidden_size)
        self.layers = []
        self._build()
        self.graphs = {}

    def _linear(self, name: str, x: str, weight: str) -> Operation:
        op = Operation("Linear", [x, weight], [name])
        plan_linear(
            op, self.tensors, self.shapes, self.device, cublas=self.runner.cublas
        )
        op.attrs["_sequence"] = (op,)
        return op

    def _rms(
        self, name: str, x: str, scale: str, epsilon: float, centered: bool = False
    ) -> Operation:
        op = Operation(
            "SimplifiedLayerNormalization", [x, scale], [name], {"epsilon": epsilon}
        )
        if centered:
            op.attrs["_scale_offset"] = 1.0
        plan_rms_norm(op, self.tensors, self.shapes, self.device)
        op.attrs["_sequence"] = (op,)
        return op

    def _swiglu(self, name: str, gate: str, up: str) -> Operation:
        op = Operation("_SwiGLU", [gate, up], [name])
        plan_swiglu(op, self.tensors, self.shapes, self.device)
        op.attrs["_sequence"] = (op,)
        return op

    def _register(self, name: str, value: wp.array) -> str:
        self.tensors[name] = value
        self.shapes[name] = tuple(value.shape)
        return name

    def _build(self) -> None:
        hidden = "hidden.0"
        self.embedding_norm = self._rms(
            "embedding.normalized", hidden, "__unit_hidden", self.runner.rms_epsilon
        )
        hidden = self.embedding_norm.outputs[0]
        for index in range(self.runner.num_layers):
            prefix = f"model.language_model.layers.{index}."
            layer = {"local": self.runner.layer_types[index] == "sliding_attention"}
            layer["input_norm"] = self._rms(
                f"layer.{index}.attention_input",
                hidden,
                prefix + "input_layernorm.weight",
                self.runner.rms_epsilon,
                self.runner.centered_norm_scales,
            )
            self._build_attention(layer, index, prefix, layer["input_norm"].outputs[0])
            layer["post_attention"] = self._rms(
                f"layer.{index}.attention_post",
                layer["attention_output"].outputs[0],
                prefix + "post_attention_layernorm.weight",
                self.runner.post_epsilon,
                self.runner.centered_norm_scales,
            )
            attention_residual = self._register(
                f"layer.{index}.attention_residual",
                wp.empty(
                    (self.rows, self.runner.hidden_size),
                    dtype=self.dtype,
                    device=self.device,
                ),
            )
            layer["attention_residual"] = attention_residual
            layer["feedforward_input"] = self._rms(
                f"layer.{index}.feedforward_input",
                attention_residual,
                prefix + "pre_feedforward_layernorm.weight",
                self.runner.rms_epsilon,
                self.runner.centered_norm_scales,
            )
            layer["mlp_gate"] = self._linear(
                f"layer.{index}.mlp_gate",
                layer["feedforward_input"].outputs[0],
                prefix + "mlp.gate_proj.weight",
            )
            layer["mlp_up"] = self._linear(
                f"layer.{index}.mlp_up",
                layer["feedforward_input"].outputs[0],
                prefix + "mlp.up_proj.weight",
            )
            layer["swiglu"] = self._swiglu(
                f"layer.{index}.swiglu",
                layer["mlp_gate"].outputs[0],
                layer["mlp_up"].outputs[0],
            )
            layer["mlp_down"] = self._linear(
                f"layer.{index}.mlp_down",
                layer["swiglu"].outputs[0],
                prefix + "mlp.down_proj.weight",
            )
            layer["post_feedforward"] = self._rms(
                f"layer.{index}.feedforward_post",
                layer["mlp_down"].outputs[0],
                prefix + "post_feedforward_layernorm.weight",
                self.runner.post_epsilon,
                self.runner.centered_norm_scales,
            )
            hidden = self._register(
                f"hidden.{index + 1}",
                wp.empty(
                    (self.rows, self.runner.hidden_size),
                    dtype=self.dtype,
                    device=self.device,
                ),
            )
            layer["output"] = hidden
            self.layers.append(layer)
        self.final_norm = self._rms(
            "final.normalized",
            hidden,
            "model.language_model.norm.weight",
            self.runner.rms_epsilon,
        )
        self.lm_head = self._linear(
            "logits", self.final_norm.outputs[0], "lm_head.weight"
        )
        self.logits = self.tensors["logits"].reshape(
            (1, self.rows, self.runner.vocab_size)
        )

    def _build_attention(self, layer: dict, index: int, prefix: str, x: str) -> None:
        attention = prefix + "self_attn."
        for projection in ("q", "k", "v", "gate"):
            layer[projection + "_proj"] = self._linear(
                f"layer.{index}.{projection}_projected",
                x,
                attention + projection + "_proj.weight",
            )
        q_shape = (self.runner.query_heads * self.rows, self.runner.head_dim)
        kv_shape = (self.runner.kv_heads * self.rows, self.runner.head_dim)
        layer["q"] = wp.empty(q_shape, dtype=self.dtype, device=self.device)
        layer["k"] = wp.empty(kv_shape, dtype=self.dtype, device=self.device)
        layer["v"] = wp.empty(kv_shape, dtype=self.dtype, device=self.device)
        self._register(f"layer.{index}.q", layer["q"])
        self._register(f"layer.{index}.k", layer["k"])
        layer["q_norm"] = self._rms(
            f"layer.{index}.q_norm",
            f"layer.{index}.q",
            "__unit_head",
            self.runner.rms_epsilon,
        )
        layer["k_norm"] = self._rms(
            f"layer.{index}.k_norm",
            f"layer.{index}.k",
            "__unit_head",
            self.runner.rms_epsilon,
        )
        if layer["local"]:
            layer["q_ready"] = wp.empty_like(layer["q"])
            layer["k_ready"] = wp.empty_like(layer["k"])
        else:
            layer["q_ready"] = self.tensors[layer["q_norm"].outputs[0]]
            layer["k_ready"] = self.tensors[layer["k_norm"].outputs[0]]
        layer["core"] = wp.empty(
            (self.rows, self.runner.query_heads * self.runner.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        layer["gated"] = wp.empty_like(layer["core"])
        core_name = self._register(f"layer.{index}.attention_gated", layer["gated"])
        layer["attention_output"] = self._linear(
            f"layer.{index}.attention_output", core_name, attention + "o_proj.weight"
        )
        layer["attention_block"], layer["attention_kernel"] = _get_gqa_attention_kernel(
            self.runner.head_dim, self.dtype
        )
        if self.rows == 1:
            if not hasattr(self, "partitioned_attention"):
                self.partitioned_attention = {
                    partitions: _allocate_partitioned_gqa(
                        self.runner.query_heads,
                        self.runner.head_dim,
                        self.dtype,
                        self.device,
                        partitions,
                    )
                    for partitions in (256,)
                }
                self.attention_partitions = 256
            layer["partitioned_attention"] = self.partitioned_attention

    def _execute_op(self, op: Operation) -> None:
        execute_operations(
            op.attrs["_sequence"], self.tensors, self.shapes, self.device
        )

    def _execute_attention(self, layer: dict, index: int) -> None:
        for projection in ("q", "k", "v", "gate"):
            self._execute_op(layer[projection + "_proj"])
        for projection in ("q", "k", "v"):
            output = layer[projection]
            heads = (
                self.runner.query_heads if projection == "q" else self.runner.kv_heads
            )
            reorder_kernel = (
                _reorder_interleaved_heads_kernel
                if self.runner.gguf_layout and projection in ("q", "k")
                else _reorder_heads_kernel
            )
            wp.launch(
                reorder_kernel,
                dim=(self.rows, heads, self.runner.head_dim),
                inputs=[
                    self.tensors[layer[projection + "_proj"].outputs[0]],
                    output,
                    self.runner.head_dim,
                ],
                device=self.device,
            )
        self._execute_op(layer["q_norm"])
        self._execute_op(layer["k_norm"])
        q_normalized = self.tensors[layer["q_norm"].outputs[0]]
        wp.launch(
            _scale_kernel,
            dim=q_normalized.shape,
            inputs=[q_normalized, q_normalized, self.runner.qk_scale],
            device=self.device,
        )
        if layer["local"]:
            rotary = _rotary_embedding_kernel_for_dtype(self.dtype)
            for source, output, heads in (
                (q_normalized, layer["q_ready"], self.runner.query_heads),
                (
                    self.tensors[layer["k_norm"].outputs[0]],
                    layer["k_ready"],
                    self.runner.kv_heads,
                ),
            ):
                wp.launch(
                    rotary,
                    dim=(1, heads, self.rows, self.runner.head_dim),
                    inputs=[
                        source.reshape((1, heads, self.rows, self.runner.head_dim)),
                        self.position_ids,
                        self.runner.cos_cache,
                        self.runner.sin_cache,
                        output.reshape((1, heads, self.rows, self.runner.head_dim)),
                        self.runner.head_dim,
                        False,
                        False,
                    ],
                    device=self.device,
                )
        key_cache, value_cache = self.runner.kv_caches[index]
        cache_capacity = self.runner.cache_capacities[index]
        append_kernel = (
            _append_circular_head_cache_kernel
            if layer["local"]
            else _append_head_cache_kernel
        )
        for source, cache in ((layer["k_ready"], key_cache), (layer["v"], value_cache)):
            wp.launch(
                append_kernel,
                dim=(self.runner.kv_heads, self.rows, self.runner.head_dim),
                inputs=[
                    source,
                    self.position_ids,
                    cache,
                    self.runner.kv_heads,
                    self.runner.head_dim,
                ],
                device=self.device,
            )
        window = self.runner.local_window if layer["local"] else 0
        if "partitioned_attention" in layer:
            _launch_partitioned_gqa(
                layer["partitioned_attention"][self.attention_partitions],
                layer["q_ready"],
                key_cache,
                value_cache,
                self.runner.sequence_end,
                layer["core"],
                self.runner.query_heads,
                self.runner.kv_heads,
                cache_capacity,
                self.runner.head_dim**-0.5,
                window,
                self.device,
            )
        else:
            wp.launch_tiled(
                layer["attention_kernel"],
                dim=self.runner.query_heads * self.rows,
                inputs=[
                    layer["q_ready"],
                    key_cache,
                    value_cache,
                    self.runner.sequence_end,
                    layer["core"],
                    self.runner.query_heads,
                    self.runner.kv_heads,
                    self.rows,
                    cache_capacity,
                    self.runner.head_dim**-0.5,
                    window,
                ],
                block_dim=layer["attention_block"],
                device=self.device,
            )
        wp.launch(
            _sigmoid_gate_kernel,
            dim=layer["core"].shape,
            inputs=[
                layer["core"],
                self.tensors[layer["gate_proj"].outputs[0]],
                layer["gated"],
            ],
            device=self.device,
        )
        self._execute_op(layer["attention_output"])

    def execute(self) -> wp.array:
        """Execute the fixed plan on its staged token IDs."""
        wp.launch(
            _gather_rows_kernel,
            dim=self.embedding.shape,
            inputs=[
                self.runner.weights["model.language_model.embed_tokens.weight"],
                self.input_ids,
                self.embedding,
            ],
            device=self.device,
        )
        self._execute_op(self.embedding_norm)
        hidden = self.embedding_norm.outputs[0]
        for index, layer in enumerate(self.layers):
            self._execute_op(layer["input_norm"])
            self._execute_attention(layer, index)
            self._execute_op(layer["post_attention"])
            wp.launch(
                _binary_broadcast_kernel,
                dim=(self.rows, self.runner.hidden_size),
                inputs=[
                    self.tensors[hidden],
                    self.tensors[layer["post_attention"].outputs[0]],
                    0,
                    self.tensors[layer["attention_residual"]],
                ],
                device=self.device,
            )
            for name in (
                "feedforward_input",
                "mlp_gate",
                "mlp_up",
                "swiglu",
                "mlp_down",
                "post_feedforward",
            ):
                self._execute_op(layer[name])
            wp.launch(
                _binary_broadcast_kernel,
                dim=(self.rows, self.runner.hidden_size),
                inputs=[
                    self.tensors[layer["attention_residual"]],
                    self.tensors[layer["post_feedforward"].outputs[0]],
                    0,
                    self.tensors[layer["output"]],
                ],
                device=self.device,
            )
            hidden = layer["output"]
        self._execute_op(self.final_norm)
        self._execute_op(self.lm_head)
        wp.launch(
            _logit_softcap_kernel,
            dim=self.logits.shape,
            inputs=[
                self.logits,
                self.logits,
                self.runner.output_multiplier,
                self.runner.logit_cap,
            ],
            device=self.device,
        )
        return self.logits


class MuseGlimmerRunner(Qwen35Runner):
    """Run text-only Muse Glimmer BF16 safetensors or GGUF checkpoints."""

    def __init__(
        self,
        path: str | Path,
        device: str | wp.Device | None = None,
        cache_capacity: int = 4096,
        prefill_chunk_size: int = 16,
        use_cublas: bool = True,
    ):
        path = Path(path)
        directory = path if path.is_dir() else path.parent
        config_path = directory / "config.json"
        if config_path.is_file():
            outer_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.config = outer_config.get("text_config", outer_config)
            archive = SafeTensorArchive(directory)
            self.gguf_layout = False
            self.centered_norm_scales = True
        else:
            gguf = GGUFArchive(_gguf_paths(path))
            self.config = _gguf_config(gguf.metadata)
            archive = _MappedGGUFArchive(gguf, _gguf_weight_map(self.config))
            self.gguf_layout = True
            self.centered_norm_scales = False
        _validate_config(self.config)
        self.device = parse_device(device)
        self.cache_capacity = int(cache_capacity)
        if not 0 < self.cache_capacity <= int(self.config["max_position_embeddings"]):
            raise ValueError("cache_capacity must be within max_position_embeddings")
        if not 2 <= prefill_chunk_size <= self.cache_capacity:
            raise ValueError("prefill_chunk_size must be between 2 and cache_capacity")
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.hidden_size = int(self.config["hidden_size"])
        self.vocab_size = int(self.config["vocab_size"])
        self.num_layers = int(self.config["num_hidden_layers"])
        self.layer_types = self.config["layer_types"]
        self.query_heads = int(self.config["num_attention_heads"])
        self.kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config["head_dim"])
        self.rms_epsilon = float(self.config["rms_norm_eps"])
        self.post_epsilon = float(self.config["post_norm_eps"])
        self.qk_scale = float(self.config["qk_scale_factor"])
        self.output_multiplier = float(self.config["output_multiplier"])
        self.logit_cap = float(self.config["final_logit_softcapping"])
        self.local_window = min(int(self.config["sliding_window"]), self.cache_capacity)
        self.local_cache_capacity = (
            self.cache_capacity
            if self.cache_capacity <= self.local_window
            else self.local_window + self.prefill_chunk_size - 1
        )

        names = _weight_names(self.config)
        missing = set(names) - set(archive.names)
        if missing:
            raise ValueError(
                f"Muse Glimmer checkpoint is missing {sorted(missing)[:5]}"
            )
        embedding_dtype = archive.metadata(
            "model.language_model.embed_tokens.weight"
        ).dtype
        if embedding_dtype not in (wp.float16, wp.bfloat16):
            raise TypeError("Muse Glimmer embeddings must use FP16 or BF16")
        required_bytes = sum(archive.metadata(name).nbytes for name in names)
        for layer_type in self.layer_types:
            capacity = (
                self.local_cache_capacity
                if layer_type == "sliding_attention"
                else self.cache_capacity
            )
            required_bytes += 2 * self.kv_heads * capacity * self.head_dim * 2
        if self.device.is_cuda and required_bytes > self.device.free_memory * 0.95:
            raise MemoryError(
                f"Muse Glimmer needs at least {required_bytes / 2**30:.1f} GiB for text weights and KV cache; "
                f"{self.device.free_memory / 2**30:.1f} GiB is currently free"
            )
        self.weights = archive.load(self.device, names)
        self.dtype = embedding_dtype
        self.unit_scales = {
            "__unit_hidden": wp.ones(
                self.hidden_size, dtype=wp.float32, device=self.device
            ),
            "__unit_head": wp.ones(self.head_dim, dtype=wp.float32, device=self.device),
        }
        self.cublas = (
            try_create_cublas() if use_cublas and self.device.is_cuda else None
        )
        self.sequence_end = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.conv_states = {}
        self.recurrent_states = {}
        self.kv_caches = {}
        self.cache_capacities = {}
        for index, layer_type in enumerate(self.layer_types):
            capacity = (
                self.local_cache_capacity
                if layer_type == "sliding_attention"
                else self.cache_capacity
            )
            self.cache_capacities[index] = capacity
            shape = (self.kv_heads * capacity, self.head_dim)
            self.kv_caches[index] = (
                wp.empty(shape, dtype=self.dtype, device=self.device),
                wp.empty(shape, dtype=self.dtype, device=self.device),
            )
        positions = np.arange(self.cache_capacity, dtype=np.float32)[:, None]
        theta = float(
            self.config.get("rope_parameters", {}).get("rope_theta", 500000.0)
        )
        frequencies = 1.0 / (
            theta ** (np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim)
        )
        angles = positions * frequencies[None, :]
        self.cos_cache = wp.array(np.cos(angles), dtype=self.dtype, device=self.device)
        self.sin_cache = wp.array(np.sin(angles), dtype=self.dtype, device=self.device)
        self._decode_plan = _MusePlan(self, 1)
        self._chunk_plan = _MusePlan(self, self.prefill_chunk_size)
        self._sample_partial_values = wp.empty(
            128, dtype=wp.float32, device=self.device
        )
        self._sample_partial_tokens = wp.empty(128, dtype=wp.int32, device=self.device)
        self._sampled_token = wp.empty(1, dtype=wp.int32, device=self.device)
        self._sampled_token_host = wp.empty(
            1, dtype=wp.int32, device="cpu", pinned=self.device.is_cuda
        )
        self._sampled_token_host_view = self._sampled_token_host.numpy()
        self._greedy_argmax_kernels = _get_greedy_argmax_kernels(1024, 128, self.dtype)
        self.sequence_length = 0
