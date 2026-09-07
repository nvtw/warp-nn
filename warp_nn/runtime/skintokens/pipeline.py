# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light SkinTokens inference from an unrigged mesh to a rig.

The published model is split into five ONNX graphs so the runtime only needs
Warp, NumPy, and the optional pure-protobuf ``onnx`` package. PyTorch,
Transformers, SciPy, trimesh, and Blender are not runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import warp as wp

from ..formats.mesh import TriangleMesh
from ..formats.onnx import OnnxInitializerArchive
from ..geometry import farthest_point_indices
from ..chat import sample_candidates
from ..kernels import _transpose_2d_kernel
from ..onnx_runtime import OnnxRuntime
from ..qwen.causal import ExternalEmbeddingQwen3CausalLM
from ..skinning import RiggedMesh
from ..weights import cast_weight
from ...utils.device import parse_device
from .host import (
    DecodedSkeleton,
    SkinTokensGeometry,
    SkinTokensTokenizer,
    prepare_geometry,
    transfer_skin_weights,
)


_GRAPH_FILES = {
    "embed": "embed.onnx",
    "mesh_cond": "mesh_cond.onnx",
    "vae_cond": "vae_cond.onnx",
    "decoder": "decoder.onnx",
    "skin_decode": "skin_decode.onnx",
}


@dataclass(frozen=True)
class SkinTokensCheckpoint:
    """Validated paths and dimensions for a converted SkinTokens checkpoint."""

    root: Path
    graph_paths: dict[str, Path]
    num_points: int
    num_vertex_samples: int
    activation_dtype: type
    external_fps_indices: bool
    fps_seed: int
    mesh_fps_candidates: int
    vae_fps_candidates: int
    tokens_per_skin: int
    mesh_tokens: int
    cond_tokens: int
    hidden_size: int
    intermediate_size: int
    latent_channels: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    vocabulary: int
    max_positions: int
    batched_skin_decode: bool

    @classmethod
    def load(cls, path: str | Path) -> "SkinTokensCheckpoint":
        root = Path(path).expanduser()
        config_path = root if root.suffix == ".json" else root / "skintokens.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"SkinTokens config not found: {config_path}")
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        if config.get("schema") != "qtmesh-skintokens-onnx-v1":
            raise ValueError("unsupported SkinTokens checkpoint schema")
        root = config_path.parent
        graph_paths = {name: root / filename for name, filename in _GRAPH_FILES.items()}
        missing = [str(value) for value in graph_paths.values() if not value.is_file()]
        if missing:
            raise FileNotFoundError(f"SkinTokens graph files are missing: {missing}")
        llm = config["llm"]
        dtype_name = config.get("dtype", "fp32")
        try:
            activation_dtype = {"fp32": wp.float32, "bf16": wp.bfloat16}[dtype_name]
        except KeyError as exc:
            raise ValueError(f"unsupported SkinTokens dtype: {dtype_name}") from exc
        external_fps = bool(config.get("external_fps_indices", False))
        fps = config.get("fps_candidates", {})
        if external_fps and not all(name in fps for name in ("seed", "mesh", "vae")):
            raise ValueError("external FPS checkpoint is missing its candidate contract")
        checkpoint = cls(
            root=root,
            graph_paths=graph_paths,
            num_points=int(config["num_points"]),
            num_vertex_samples=int(config.get("num_vertex_samples", 0)),
            activation_dtype=activation_dtype,
            external_fps_indices=external_fps,
            fps_seed=int(fps.get("seed", 0)),
            mesh_fps_candidates=int(fps.get("mesh", 0)),
            vae_fps_candidates=int(fps.get("vae", 0)),
            tokens_per_skin=int(config["tokens_per_skin"]),
            mesh_tokens=512,
            cond_tokens=int(config["tokens_skin_cond"]),
            hidden_size=int(llm["hidden_size"]),
            intermediate_size=3072,
            latent_channels=int(config["vae_latent_channels"]),
            layers=int(llm["num_hidden_layers"]),
            query_heads=16,
            kv_heads=int(llm["num_key_value_heads"]),
            head_dim=int(llm["head_dim"]),
            vocabulary=int(llm["full_vocab_size"]),
            max_positions=3192,
            batched_skin_decode=bool(config.get("batched_skin_decode", False)),
        )
        if (
            checkpoint.tokens_per_skin != SkinTokensTokenizer.tokens_per_skin
            or checkpoint.vocabulary != SkinTokensTokenizer.full_vocab
        ):
            raise ValueError(
                "checkpoint vocabulary does not match the SkinTokens grammar"
            )
        return checkpoint

    def qwen_config(self) -> dict:
        """Return the native dense-Qwen execution contract."""
        return {
            "model_type": "qwen3",
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.query_heads,
            "num_key_value_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocabulary,
            "max_position_embeddings": self.max_positions,
            "hidden_act": "silu",
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 1_000_000.0,
            "qk_norm": True,
            "attention_bias": False,
            "tie_word_embeddings": False,
            "layer_types": ["full_attention"] * self.layers,
        }


@dataclass(frozen=True)
class SkinTokensResult:
    """Generated token stream, sampled prediction, and transferable final rig."""

    rig: RiggedMesh
    tokens: np.ndarray
    skeleton: DecodedSkeleton
    sampled_weights: np.ndarray
    geometry: SkinTokensGeometry


def _decoder_weight_sources(checkpoint: SkinTokensCheckpoint) -> dict[str, str]:
    """Map canonical Qwen names to the converter's deterministic ONNX names."""
    names = {"model.norm.weight": "transformer.model.layers.0.input_layernorm.weight"}
    suffixes = (
        ("self_attn.q_proj.weight", 0),
        ("self_attn.k_proj.weight", 7),
        ("self_attn.v_proj.weight", 8),
        ("self_attn.o_proj.weight", 24),
        ("mlp.gate_proj.weight", 25),
        ("mlp.up_proj.weight", 26),
        ("mlp.down_proj.weight", 27),
    )
    for layer in range(checkpoint.layers):
        prefix = f"model.layers.{layer}."
        base = 8387 + 28 * layer
        names.update(
            {
                prefix + suffix: f"onnx::MatMul_{base + offset}"
                for suffix, offset in suffixes
            }
        )
        names[prefix + "input_layernorm.weight"] = (
            "transformer.model.layers.0.input_layernorm.weight"
        )
        names[prefix + "post_attention_layernorm.weight"] = (
            "transformer.model.layers.0.input_layernorm.weight"
        )
        names[prefix + "self_attn.q_norm.weight"] = (
            "transformer.model.layers.0.self_attn.q_norm.weight"
        )
        names[prefix + "self_attn.k_norm.weight"] = (
            "transformer.model.layers.0.self_attn.q_norm.weight"
        )
    names["lm_head.weight"] = "onnx::MatMul_9171"
    return names


class _SkinTokensDecoderArchive:
    """Expose transposed ONNX decoder matrices under canonical Qwen names."""

    def __init__(self, checkpoint: SkinTokensCheckpoint):
        self._archive = OnnxInitializerArchive(checkpoint.graph_paths["decoder"])
        self._sources = _decoder_weight_sources(checkpoint)
        missing = set(self._sources.values()) - set(self._archive.names)
        if missing:
            raise ValueError(
                f"SkinTokens decoder is missing weights: {sorted(missing)[:5]}"
            )

    @property
    def names(self):
        return tuple(self._sources)

    def metadata(self, name):
        return self._archive.metadata(self._sources[name])

    def load(self, device=None, names=None):
        selected = self.names if names is None else tuple(names)
        output = {}
        for name in selected:
            source = self._archive.load(device, (self._sources[name],))[
                self._sources[name]
            ]
            if source.ndim == 2:
                transposed = wp.empty(
                    (source.shape[1], source.shape[0]),
                    dtype=source.dtype,
                    device=source.device,
                )
                wp.launch(
                    _transpose_2d_kernel,
                    dim=source.shape,
                    inputs=[source, transposed],
                    device=source.device,
                )
                source = transposed
            output[name] = source
        return output


def sample_token(
    logits,
    allowed,
    sequence,
    *,
    rng: np.random.Generator,
    top_k: int = 5,
    top_p: float = 0.95,
    temperature: float = 1.0,
    repetition_penalty: float = 2.0,
) -> int:
    """Sample one grammar-constrained token with Transformers-compatible order."""
    scores = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    allowed = np.asarray(allowed, dtype=np.int64).reshape(-1)
    if not len(allowed) or np.any(allowed < 0) or np.any(allowed >= len(scores)):
        raise ValueError("allowed token IDs are empty or outside the vocabulary")
    if top_k <= 0 or not 0.0 < top_p <= 1.0 or temperature < 0.0:
        raise ValueError("top_k/top_p/temperature generation settings are invalid")
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be positive")
    repeated = np.unique(np.asarray(sequence, dtype=np.int64))
    repeated = repeated[(repeated >= 0) & (repeated < len(scores))]
    values = scores[repeated]
    scores[repeated] = np.where(
        values < 0.0, values * repetition_penalty, values / repetition_penalty
    )
    candidate_scores = scores[allowed]
    order = np.lexsort((allowed, -candidate_scores))[: min(top_k, len(allowed))]
    candidates = allowed[order]
    candidate_scores = candidate_scores[order]
    if temperature == 0.0:
        return int(candidates[0])
    return sample_candidates(
        candidate_scores, candidates, temperature, top_p, rng
    )


class SkinTokensPipeline:
    """Run the official TokenRig decomposition and return a :class:`RiggedMesh`."""

    def __init__(
        self,
        checkpoint: str | Path | SkinTokensCheckpoint,
        *,
        device: str | wp.Device | None = None,
        use_cublas: bool = True,
        optimized_decoder: bool = True,
        dtype=wp.bfloat16,
        cache_capacity: int = 2048,
        prefill_chunk_size: int = 16,
        skin_decode_batch_size: int = 2,
        runtime_factory: Callable = OnnxRuntime,
    ):
        self.checkpoint = (
            checkpoint
            if isinstance(checkpoint, SkinTokensCheckpoint)
            else SkinTokensCheckpoint.load(checkpoint)
        )
        self.device = parse_device(device)
        self.use_cublas = bool(use_cublas)
        self.optimized_decoder = bool(optimized_decoder)
        self.dtype = dtype
        self.cache_capacity = int(cache_capacity)
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.skin_decode_batch_size = int(skin_decode_batch_size)
        if (
            not 1
            <= self.prefill_chunk_size
            <= self.cache_capacity
            <= self.checkpoint.max_positions
        ):
            raise ValueError("SkinTokens cache/prefill sizes are invalid")
        if self.skin_decode_batch_size <= 0:
            raise ValueError("SkinTokens skin decode batch size must be positive")
        self.tokenizer = SkinTokensTokenizer()
        self._runtime_factory = runtime_factory
        self._runtimes: dict[str, tuple[object, dict[str, tuple[int, ...]]]] = {}
        self._native_decoder = None
        self._skin_batch_states = {}

    def _native(self):
        if self._native_decoder is None:
            archive = _SkinTokensDecoderArchive(self.checkpoint)
            identity = {name: name for name in archive.names}
            self._native_decoder = ExternalEmbeddingQwen3CausalLM(
                self.checkpoint.graph_paths["decoder"],
                self.checkpoint.qwen_config(),
                archive,
                identity,
                cache_capacity=self.cache_capacity,
                prefill_chunk_size=self.prefill_chunk_size,
                device=self.device,
                dtype=self.dtype,
                use_cublas=self.use_cublas,
            )
        return self._native_decoder

    def _runtime(self, name: str, shapes: dict[str, tuple[int, ...]]):
        shapes = {key: tuple(value) for key, value in shapes.items()}
        entry = self._runtimes.get(name)
        if entry is None:
            runtime = self._runtime_factory(
                self.checkpoint.graph_paths[name],
                device=self.device,
                input_shapes=shapes,
                use_cublas=self.use_cublas,
            )
        else:
            runtime, old_shapes = entry
            if shapes != old_shapes:
                runtime.resize_inputs(shapes)
        self._runtimes[name] = (runtime, shapes)
        return runtime

    def _array(self, value, dtype):
        return wp.array(value, dtype=dtype, device=self.device)

    def _run(self, name: str, inputs: dict[str, wp.array]):
        shapes = {key: tuple(value.shape) for key, value in inputs.items()}
        return self._runtime(name, shapes)(inputs)

    def _embed(self, tokens):
        token_ids = np.asarray(tokens, dtype=np.int64).reshape(1, -1)
        return self._run("embed", {"input_ids": self._array(token_ids, wp.int64)})[
            "embeds"
        ]

    def _decoder(self, embeddings, past):
        inputs = {"inputs_embeds": embeddings}
        for layer in range(self.checkpoint.layers):
            for kind in ("key", "value"):
                name = f"past.{layer}.{kind}"
                inputs[name] = past.get(
                    name,
                    wp.empty(
                        (1, self.checkpoint.kv_heads, 0, self.checkpoint.head_dim),
                        dtype=self.checkpoint.activation_dtype,
                        device=self.device,
                    ),
                )
        outputs = self._run("decoder", inputs)
        next_past = {
            f"past.{layer}.{kind}": outputs[f"present.{layer}.{kind}"]
            for layer in range(self.checkpoint.layers)
            for kind in ("key", "value")
        }
        return outputs["logits"], next_past

    def _condition(self, geometry):
        dtype = self.checkpoint.activation_dtype
        vertices = self._array(geometry.sampled_vertices[None], dtype)
        normals = self._array(geometry.sampled_normals[None], dtype)
        cond_np = np.concatenate(
            (geometry.sampled_vertices, geometry.sampled_normals), axis=1
        )[None]
        cond = self._array(cond_np, dtype)
        mesh_inputs = {"vertices": vertices, "normals": normals}
        vae_inputs = {"cond": cond}
        if self.checkpoint.external_fps_indices:
            points = geometry.sampled_vertices

            def fixed_fps(candidate_count, output_count):
                candidates = np.random.default_rng(self.checkpoint.fps_seed).choice(
                    len(points),
                    candidate_count,
                    replace=candidate_count > len(points),
                )
                return farthest_point_indices(points[candidates], output_count)

            mesh_inputs["fps_indices"] = self._array(
                fixed_fps(
                    self.checkpoint.mesh_fps_candidates,
                    self.checkpoint.mesh_tokens,
                ),
                wp.int64,
            )
            vae_inputs["fps_indices"] = self._array(
                fixed_fps(
                    self.checkpoint.vae_fps_candidates,
                    self.checkpoint.cond_tokens,
                ),
                wp.int64,
            )
        mesh = self._run("mesh_cond", mesh_inputs)["cond_embeds"]
        latent = self._run("vae_cond", vae_inputs)["cond_latents"]
        return cond, mesh, latent

    def _join_embeddings(self, first, second):
        total = first.shape[1] + second.shape[1]
        output = wp.empty(
            (1, total, self.checkpoint.hidden_size),
            dtype=first.dtype,
            device=self.device,
        )
        wp.copy(output.flatten(), first.flatten(), count=first.size)
        wp.copy(
            output.flatten(),
            second.flatten(),
            dest_offset=first.size,
            count=second.size,
        )
        return output

    def generate_tokens(
        self,
        mesh_embeddings,
        *,
        cls: str = "articulation",
        skeleton_tokens=None,
        max_new_tokens: int = 1534,
        seed: int = 0,
        top_k: int = 5,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 2.0,
    ) -> np.ndarray:
        """Autoregressively generate a grammar-valid skeleton and SkinTokens."""
        start = (
            self.tokenizer.start_tokens(cls)
            if skeleton_tokens is None
            else np.asarray(skeleton_tokens, dtype=np.int64).reshape(-1)
        )
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if skeleton_tokens is not None:
            forced_skeleton = self.tokenizer.decode_skeleton(start)
            required_tokens = (
                len(forced_skeleton.joints) * self.checkpoint.tokens_per_skin + 1
            )
            if max_new_tokens < required_tokens:
                raise ValueError(
                    f"forced skeleton requires {required_tokens} generated tokens"
                )
            generation_limit = required_tokens
        else:
            generation_limit = max_new_tokens
        start_embeddings = self._embed(start)
        if self.optimized_decoder:
            decoder = self._native()
            prefix = self._join_embeddings(
                cast_weight(mesh_embeddings, self.dtype),
                cast_weight(start_embeddings, self.dtype),
            ).reshape((-1, self.checkpoint.hidden_size))
            generation_limit = min(
                generation_limit, self.cache_capacity - prefix.shape[0]
            )
            if generation_limit <= 0:
                raise ValueError("SkinTokens prefix leaves no KV-cache capacity")
            logits = decoder.prefill_embeddings(prefix)
            past = None
        else:
            logits, past = self._decoder(
                self._join_embeddings(mesh_embeddings, start_embeddings), {}
            )
        generated: list[int] = []
        rng = np.random.default_rng(seed)
        for _ in range(generation_limit):
            sequence = np.concatenate((start, np.asarray(generated, dtype=np.int64)))
            token = sample_token(
                logits.numpy()[0, -1],
                self.tokenizer.allowed_tokens(sequence),
                sequence,
                rng=rng,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            generated.append(token)
            if token == self.tokenizer.global_eos:
                break
            embedding = self._embed((token,))
            if self.optimized_decoder:
                logits = decoder.decode_embedding(
                    cast_weight(embedding, self.dtype).reshape(
                        (1, self.checkpoint.hidden_size)
                    )
                )
            else:
                logits, past = self._decoder(embedding, past)
        else:
            raise RuntimeError("SkinTokens generation did not reach its global EOS")
        return np.concatenate((start, np.asarray(generated, dtype=np.int64)))

    def _decode_weights(self, tokens, cond, cond_latents, skeleton):
        switch = int(np.flatnonzero(tokens == self.tokenizer.switch)[0])
        skin_ids = tokens[switch + 1 : -1]
        expected = len(skeleton.joints) * self.checkpoint.tokens_per_skin
        if len(skin_ids) != expected:
            raise ValueError(f"expected {expected} skin tokens, got {len(skin_ids)}")
        joints = len(skeleton.joints)
        local_ids = skin_ids.reshape(joints, self.checkpoint.tokens_per_skin)
        local_ids = local_ids - self.tokenizer.skeleton_vocab
        if self.checkpoint.batched_skin_decode:
            batch_size = min(self.skin_decode_batch_size, joints)
            weights = np.empty(
                (self.checkpoint.num_points, joints), dtype=np.float32
            )
            for begin in range(0, joints, batch_size):
                end = min(begin + batch_size, joints)
                ids = np.empty(
                    (batch_size, self.checkpoint.tokens_per_skin), dtype=np.int64
                )
                ids[: end - begin] = local_ids[begin:end]
                if end - begin < batch_size:
                    ids[end - begin :] = local_ids[begin]
                decoded = self._decode_weights_batched(ids, cond, cond_latents)
                weights[:, begin:end] = decoded[:, : end - begin]
            return weights
        weights = np.empty((self.checkpoint.num_points, joints), dtype=np.float32)
        for joint in range(joints):
            outputs = self._run(
                "skin_decode",
                {
                    "skin_ids": self._array(local_ids[joint : joint + 1], wp.int64),
                    "cond": cond,
                    "cond_latents": cond_latents,
                },
            )
            weights[:, joint] = outputs["weights"].numpy().reshape(-1)
        return weights

    @staticmethod
    def _copy_repeated_batch(target, source):
        for batch in range(target.shape[0]):
            wp.copy(
                target.flatten(),
                source.flatten(),
                dest_offset=batch * source.size,
                count=source.size,
            )

    def _decode_weights_batched(self, local_ids, cond, cond_latents):
        batches = len(local_ids)
        state = self._skin_batch_states.get(batches)
        if state is None:
            inputs = {
                "skin_ids": wp.empty(
                    (batches, self.checkpoint.tokens_per_skin),
                    dtype=wp.int64,
                    device=self.device,
                ),
                "cond": wp.empty(
                    (batches, *cond.shape[1:]),
                    dtype=cond.dtype,
                    device=self.device,
                ),
                "cond_latents": wp.empty(
                    (batches, *cond_latents.shape[1:]),
                    dtype=cond_latents.dtype,
                    device=self.device,
                ),
            }
            runtime = self._runtime_factory(
                self.checkpoint.graph_paths["skin_decode"],
                device=self.device,
                batch_size=batches,
                input_batch_axes={"skin_ids": 0, "cond": 0, "cond_latents": 0},
                use_cublas=self.use_cublas,
            )
            graph = None
            output = None
            if self.device.is_cuda:
                runtime(inputs)
                wp.capture_begin(device=self.device)
                output = runtime(inputs)["weights"]
                graph = wp.capture_end(device=self.device)
            state = self._skin_batch_states[batches] = (
                runtime,
                inputs,
                graph,
                output,
            )
        runtime, inputs, graph, output = state
        inputs["skin_ids"].assign(np.asarray(local_ids, dtype=np.int64))
        self._copy_repeated_batch(inputs["cond"], cond)
        self._copy_repeated_batch(inputs["cond_latents"], cond_latents)
        if graph is None:
            output = runtime(inputs)["weights"]
        else:
            wp.capture_launch(graph)
        return output.numpy().T.copy()

    def rig(
        self,
        mesh: TriangleMesh,
        *,
        cls: str = "articulation",
        skeleton_tokens=None,
        skeleton_joints=None,
        skeleton_parents=None,
        joint_names=None,
        seed: int = 0,
        max_new_tokens: int = 1534,
        top_k: int = 5,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 2.0,
    ) -> SkinTokensResult:
        """Generate a complete rig for an unrigged triangle mesh."""
        provided_skeleton = skeleton_joints is not None or skeleton_parents is not None
        if provided_skeleton:
            if (
                skeleton_tokens is not None
                or skeleton_joints is None
                or skeleton_parents is None
            ):
                raise ValueError(
                    "provide either skeleton_tokens or both skeleton_joints/skeleton_parents"
                )
            skeleton_joints = np.asarray(skeleton_joints, dtype=np.float32)
        geometry = prepare_geometry(
            mesh,
            points=self.checkpoint.num_points,
            vertex_samples=self.checkpoint.num_vertex_samples,
            seed=seed,
            joints=skeleton_joints if provided_skeleton else None,
        )
        if provided_skeleton:
            normalized_joints = (
                skeleton_joints @ geometry.world_to_model[:3, :3].T
                + geometry.world_to_model[:3, 3]
            )
            skeleton_tokens = self.tokenizer.tokenize_skeleton(
                normalized_joints, skeleton_parents, cls=cls
            )
        cond, mesh_embeddings, cond_latents = self._condition(geometry)
        tokens = self.generate_tokens(
            mesh_embeddings,
            cls=cls,
            skeleton_tokens=skeleton_tokens,
            max_new_tokens=max_new_tokens,
            seed=seed,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        switch = int(np.flatnonzero(tokens == self.tokenizer.switch)[0])
        skeleton = self.tokenizer.decode_skeleton(tokens[: switch + 1])
        sampled_weights = self._decode_weights(tokens, cond, cond_latents, skeleton)
        weights = transfer_skin_weights(
            geometry.normalized_vertices,
            geometry.sampled_vertices,
            sampled_weights,
        )
        rest_joints = (
            skeleton_joints
            if provided_skeleton
            else (
                skeleton.joints @ geometry.model_to_world[:3, :3].T
                + geometry.model_to_world[:3, 3]
            )
        )
        rig = RiggedMesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            rest_joints=rest_joints,
            parents=skeleton.parents,
            weights=weights,
            joint_names=(
                tuple(joint_names) if joint_names is not None else skeleton.joint_names
            ),
        )
        return SkinTokensResult(rig, tokens, skeleton, sampled_weights, geometry)


__all__ = [
    "SkinTokensCheckpoint",
    "SkinTokensPipeline",
    "SkinTokensResult",
    "sample_token",
]
