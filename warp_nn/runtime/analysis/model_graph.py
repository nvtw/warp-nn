# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create an interactive, standalone HTML map of a model checkpoint.

Only checkpoint headers are inspected. Tensor contents are never read or uploaded.
Run with ``python -m warp_nn.runtime.analysis.model_graph MODEL``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import tempfile
import webbrowser

from warp_nn.runtime.formats.gguf import GGUFArchive, find_gguf_files
from warp_nn.runtime.formats.safetensors import SafeTensorArchive

from .report import render_report


_LAYER_PATTERN = re.compile(
    r"(?:^|\.)(?:layers?|blocks?|blk|transformer_blocks?)\.(\d+)(?:\.|$)"
)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    format: str
    nbytes: int

    @property
    def parameters(self) -> int:
        return math.prod(self.shape)


def _read_config(directory: Path) -> dict:
    path = directory / "config.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid model config '{path}'") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Model config '{path}' must contain an object")
    nested = value.get("text_config")
    if isinstance(nested, dict):
        return {**value, **nested, "_container_model_type": value.get("model_type")}
    return value


def _checkpoint(path: Path) -> tuple[str, dict, list[TensorInfo]]:
    directory = path if path.is_dir() else path.parent
    config = _read_config(directory)
    safe_index = directory / "model.safetensors.index.json"
    safe_single = directory / "model.safetensors"
    is_safe = (
        path.name.endswith(".safetensors")
        or path.name.endswith(".safetensors.index.json")
        or path.is_dir()
        and (safe_index.is_file() or safe_single.is_file())
    )
    if is_safe:
        archive = SafeTensorArchive(path)
        tensors = [
            TensorInfo(name, info.shape, info.format, info.nbytes)
            for name in archive.names
            for info in (archive.metadata(name),)
        ]
        return "safetensors", config, tensors

    archive = GGUFArchive(find_gguf_files(path))
    metadata = archive.metadata
    architecture = metadata.get("general.architecture")
    config = {
        **config,
        "model_type": config.get("model_type", architecture),
        "hidden_size": config.get(
            "hidden_size", metadata.get(f"{architecture}.embedding_length")
        ),
        "num_hidden_layers": config.get(
            "num_hidden_layers", metadata.get(f"{architecture}.block_count")
        ),
        "num_attention_heads": config.get(
            "num_attention_heads", metadata.get(f"{architecture}.attention.head_count")
        ),
        "num_key_value_heads": config.get(
            "num_key_value_heads",
            metadata.get(f"{architecture}.attention.head_count_kv"),
        ),
        "max_position_embeddings": config.get(
            "max_position_embeddings", metadata.get(f"{architecture}.context_length")
        ),
    }
    layers = int(config.get("num_hidden_layers") or 0)
    if not config.get("layer_types") and layers:
        interval = metadata.get(f"{architecture}.full_attention_interval")
        if interval:
            config["layer_types"] = [
                "full_attention"
                if (index + 1) % int(interval) == 0
                else "linear_attention"
                for index in range(layers)
            ]
        pattern = metadata.get(f"{architecture}.attention.sliding_window_pattern")
        if pattern:
            config["layer_types"] = [
                "full_attention"
                if (index + 1) % int(pattern) == 0
                else "sliding_attention"
                for index in range(layers)
            ]
    tensors = [
        TensorInfo(name, info.shape, info.format, info.nbytes)
        for name in archive.names
        for info in (archive.tensor(name),)
    ]
    return "GGUF", config, tensors


def _layer_index(name: str) -> int | None:
    match = _LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def _kind(name: str, layer: int | None) -> str:
    low = name.lower()
    if layer is None:
        if any(
            part in low for part in ("embed_tokens", "token_embd", "word_embeddings")
        ):
            return "embedding"
        if any(part in low for part in ("lm_head", "output.weight", "output_layer")):
            return "output"
        if "norm" in low:
            return "final_norm"
        if "vision" in low or "visual" in low or "patch_embed" in low:
            return "vision"
        if "vae" in low:
            return "vae"
        return "other"
    if "input_layernorm" in low or ".attn_norm" in low:
        return "input_norm"
    if "post_attention_layernorm" in low or "post_attention_norm" in low:
        return "post_attention_norm"
    if "pre_feedforward" in low or ".ffn_norm" in low:
        return "pre_ffn_norm"
    if "post_feedforward" in low or "post_ffw_norm" in low:
        return "post_ffn_norm"
    if any(
        part in low
        for part in ("self_attn", "linear_attn", ".attn", "attention", ".ssm")
    ):
        return "attention"
    if any(part in low for part in (".mlp", ".ffn", "feed_forward", "feedforward")):
        return "mlp"
    if "norm" in low:
        return "normalization"
    if "conv" in low:
        return "convolution"
    return "other"


_KIND_DETAILS = {
    "model": (
        "Model map",
        "The whole learned machine. Follow the solid path to see how one token representation is repeatedly transformed before scores for the next token are produced.",
    ),
    "embedding": (
        "Token embedding",
        "A lookup table turns each discrete token ID into a direction in a high-dimensional space. Similar contexts can then be represented by nearby patterns of numbers.",
    ),
    "input_norm": (
        "Input normalization",
        "Rescales the current representation before the main computation. It keeps signal magnitudes predictable without deciding what the signal means.",
    ),
    "attention": (
        "Attention mixer",
        "Each token gathers useful information from other positions. Query and key vectors decide where to look; value vectors carry the information that is gathered.",
    ),
    "post_attention_norm": (
        "Post-attention normalization",
        "Prepares the residual stream for the feed-forward computation that follows attention.",
    ),
    "pre_ffn_norm": (
        "Feed-forward input normalization",
        "Rescales the residual stream before the layer's independent, per-token transformation.",
    ),
    "mlp": (
        "Feed-forward network",
        "A learned feature detector and writer applied to every token position. It expands the vector, gates useful features, then projects back to the model width.",
    ),
    "post_ffn_norm": (
        "Post-FFN normalization",
        "Controls the scale after the feed-forward transformation before the residual stream continues.",
    ),
    "normalization": (
        "Normalization",
        "Keeps vector magnitudes in a stable range while largely preserving their direction.",
    ),
    "convolution": (
        "Convolution",
        "Mixes nearby positions using the same small learned filter everywhere—a local, translation-like pattern detector.",
    ),
    "final_norm": (
        "Final normalization",
        "The last rescaling before the model turns its internal representation back into token scores.",
    ),
    "output": (
        "Vocabulary projection",
        "Measures how well the final representation aligns with every vocabulary item. The resulting logits become next-token probabilities after sampling.",
    ),
    "vision": (
        "Vision encoder",
        "Turns image patches into vectors that the rest of the model can reason about much like token vectors.",
    ),
    "vae": (
        "Variational autoencoder",
        "Moves between pixels or audio samples and a compact latent space where generation is cheaper.",
    ),
    "other": (
        "Model component",
        "A learned part that did not match a common transformer naming convention. Its exact tensor names and shapes are listed below.",
    ),
    "tensor": (
        "Learned tensor",
        "A multidimensional table of learned numbers. Its shape tells us which spaces it maps between; its dtype tells us how each number is stored.",
    ),
}


def _display_name(kind: str, layer: int | None, layer_type: str | None) -> str:
    title = _KIND_DETAILS[kind][0]
    if kind == "attention" and layer_type:
        title = layer_type.replace("_", " ").title()
    return f"Layer {layer} · {title}" if layer is not None else title


def _compact_number(value: int) -> str:
    for unit, scale in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if value >= scale:
            return f"{value / scale:.3g}{unit}"
    return str(value)


def _component_id(layer: int | None, kind: str) -> str:
    return f"layer-{layer}-{kind}" if layer is not None else f"global-{kind}"


def analyze_model(path: str | Path) -> dict:
    """Inspect checkpoint headers and return a JSON-serializable graph description."""
    path = Path(path).expanduser().resolve()
    format_name, config, tensors = _checkpoint(path)
    if not tensors:
        raise ValueError(f"Checkpoint '{path}' contains no tensors")

    grouped: dict[tuple[int | None, str], list[TensorInfo]] = defaultdict(list)
    for tensor in tensors:
        layer = _layer_index(tensor.name)
        grouped[(layer, _kind(tensor.name, layer))].append(tensor)

    layers_found = sorted({layer for layer, _ in grouped if layer is not None})
    layer_types = config.get("layer_types", [])
    nodes = []
    total_parameters = sum(tensor.parameters for tensor in tensors)
    total_bytes = sum(tensor.nbytes for tensor in tensors)
    model_type = str(
        config.get("_container_model_type")
        or config.get("model_type")
        or "Unknown architecture"
    )
    nodes.append(
        {
            "id": "model",
            "type": "component",
            "kind": "model",
            "label": model_type.replace("_", " ").title(),
            "subtitle": f"{_compact_number(total_parameters)} parameters · {format_name}",
            "parameters": total_parameters,
            "bytes": total_bytes,
            "explanation": _KIND_DETAILS["model"][1],
            "depth": -1,
            "lane": 0,
            "tensorCount": len(tensors),
        }
    )

    component_nodes = {}
    component_order = (
        "input_norm",
        "attention",
        "post_attention_norm",
        "pre_ffn_norm",
        "mlp",
        "post_ffn_norm",
        "normalization",
        "convolution",
        "other",
    )
    global_order = ("embedding", "vision", "vae", "other", "final_norm", "output")
    for (layer, kind), items in sorted(
        grouped.items(),
        key=lambda entry: (
            -1 if entry[0][0] is None else entry[0][0],
            (global_order if entry[0][0] is None else component_order).index(
                entry[0][1]
            )
            if entry[0][1] in (global_order if entry[0][0] is None else component_order)
            else 99,
        ),
    ):
        node_id = _component_id(layer, kind)
        parameters = sum(item.parameters for item in items)
        nbytes = sum(item.nbytes for item in items)
        formats = Counter(item.format for item in items)
        format_label = " + ".join(formats)
        layer_type = (
            str(layer_types[layer])
            if layer is not None and layer < len(layer_types)
            else None
        )
        detail_title, explanation = _KIND_DETAILS[kind]
        node = {
            "id": node_id,
            "type": "component",
            "kind": kind,
            "label": _display_name(kind, layer, layer_type),
            "subtitle": f"{format_label} · {_compact_number(parameters)} params · {len(items)} tensors",
            "parameters": parameters,
            "bytes": nbytes,
            "formats": dict(formats),
            "explanation": explanation,
            "detailTitle": detail_title,
            "layer": layer,
            "layerType": layer_type,
            "depth": (layer + 1 if layer is not None else 0),
            "lane": component_order.index(kind) if kind in component_order else 0,
            "tensorCount": len(items),
        }
        nodes.append(node)
        component_nodes[(layer, kind)] = node
        for tensor_index, tensor in enumerate(
            sorted(items, key=lambda item: item.name)
        ):
            nodes.append(
                {
                    "id": f"tensor-{len(nodes)}",
                    "type": "tensor",
                    "kind": "tensor",
                    "parent": node_id,
                    "label": tensor.name.rsplit(".", 2)[-2]
                    + "."
                    + tensor.name.rsplit(".", 1)[-1]
                    if "." in tensor.name
                    else tensor.name,
                    "fullName": tensor.name,
                    "subtitle": " × ".join(map(str, tensor.shape)) or "scalar",
                    "shape": list(tensor.shape),
                    "format": tensor.format,
                    "parameters": tensor.parameters,
                    "bytes": tensor.nbytes,
                    "explanation": _KIND_DETAILS["tensor"][1],
                    "depth": node["depth"],
                    "lane": node["lane"],
                    "tensorIndex": tensor_index,
                }
            )

    edges = []
    serial = 0

    def connect(source: str, target: str, kind: str = "flow") -> None:
        nonlocal serial
        edges.append(
            {"id": f"edge-{serial}", "source": source, "target": target, "kind": kind}
        )
        serial += 1

    flow = []
    for kind in ("embedding", "vision", "vae", "other"):
        node = component_nodes.get((None, kind))
        if node:
            flow.append(node["id"])
    for layer in layers_found:
        for kind in component_order:
            node = component_nodes.get((layer, kind))
            if node:
                flow.append(node["id"])
    for kind in ("final_norm", "output"):
        node = component_nodes.get((None, kind))
        if node:
            flow.append(node["id"])
    if flow:
        connect("model", flow[0])
        for source, target in zip(flow, flow[1:]):
            connect(source, target)
    for node in nodes:
        if node["type"] == "tensor":
            connect(node["parent"], node["id"], "contains")

    formats = Counter(tensor.format for tensor in tensors)
    inferred_layers = max(layers_found) + 1 if layers_found else 0
    summary = {
        "name": path.name,
        "path": str(path),
        "format": format_name,
        "architecture": model_type,
        "parameters": total_parameters,
        "bytes": total_bytes,
        "tensorCount": len(tensors),
        "layers": int(config.get("num_hidden_layers") or inferred_layers),
        "hiddenSize": config.get("hidden_size"),
        "attentionHeads": config.get("num_attention_heads"),
        "kvHeads": config.get("num_key_value_heads"),
        "contextLength": config.get("max_position_embeddings"),
        "formats": dict(formats),
        "headerOnly": True,
    }
    return {"summary": summary, "nodes": nodes, "edges": edges}


def write_model_graph(model: str | Path, output: str | Path) -> Path:
    """Analyze ``model`` and write a standalone HTML report to ``output``."""
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(analyze_model(model)), encoding="utf-8")
    return output


def _temporary_output(model: Path) -> Path:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", model.name).strip("-.") or "model"
    directory = Path(tempfile.mkdtemp(prefix="warp-nn-model-graph-"))
    return directory / f"{stem}-graph.html"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize a GGUF or safetensors model without loading its tensors."
    )
    parser.add_argument("model", type=Path, help="model directory or checkpoint file")
    parser.add_argument(
        "-o", "--output", type=Path, help="HTML path; defaults to a temporary report"
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="do not open the report in the default browser",
    )
    args = parser.parse_args(argv)
    output = write_model_graph(args.model, args.output or _temporary_output(args.model))
    print(f"Model graph: {output}")
    if not args.no_open and not webbrowser.open(output.as_uri()):
        print("The default browser could not be opened; open the path above manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
