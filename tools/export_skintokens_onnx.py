#!/usr/bin/env python3
"""Export the official SkinTokens/TokenRig checkpoint for warp-nn.

This offline developer tool is derived from QtMeshEditor's MIT-licensed
SkinTokens ONNX exporter. Runtime inference has no PyTorch dependency. The
quality contract keeps the official 54,000 conditioning points, including up
to 16,384 original mesh vertices.

Outputs are ``mesh_cond.onnx``, ``vae_cond.onnx``, ``embed.onnx``,
``decoder.onnx``, one dynamically batched ``skin_decode.onnx``, and the exact
runtime contract in ``skintokens.json``. Mesh/VAE graphs accept external FPS
indices: Warp computes deterministic greedy FPS in FP32 from manifest-defined
candidate subsets, so no mesh-specific choices are frozen into the graphs.

FP32 is the quality default. BF16 export remains available for numerical
experiments, but must not be published without parity validation: current Warp
BF16 execution accumulates visible conditioner/decoder/skin error.

Create an isolated export environment (versions should follow the official
SkinTokens checkout), install PyTorch, transformers, python-box, einops,
omegaconf, lightning, addict, trimesh, scipy, numpy, and onnx, then run::

    python tools/export_skintokens_onnx.py \
        --repo /path/to/SkinTokens \
        --ckpt experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt \
        --out-dir /path/to/onnx-fp32-54k

The source checkpoint is available from ``VAST-AI/SkinTokens`` on Hugging Face.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn


def log(msg):
    print(f"[export-skintokens] {msg}", flush=True)


def install_flash_attn_stub():
    """skin_vae_model.py imports flash_attn with NO SDPA fallback (unlike
    attention_processor.py, which has one). Inject a pure-torch stub module
    BEFORE importing TokenRig. Contract (flash-attn native): q/k/v are
    (B, L, H, D); returns (out, lse). The Perceiver force-casts q to bf16
    even in an fp32 model, so the stub computes in fp32 and returns in the
    VALUE dtype (the projections' dtype) to avoid a bf16×fp32 matmul error.
    """
    import types
    mod = types.ModuleType("flash_attn_interface")

    def flash_attn_func(q, k, v, *args, **kwargs):
        qt = q.permute(0, 2, 1, 3).float()
        kt = k.permute(0, 2, 1, 3).float()
        vt = v.permute(0, 2, 1, 3).float()
        if qt.shape[1] != kt.shape[1]:
            rep = qt.shape[1] // kt.shape[1]
            kt = kt.repeat_interleave(rep, dim=1)
            vt = vt.repeat_interleave(rep, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt)
        return out.permute(0, 2, 1, 3).to(v.dtype), None

    mod.flash_attn_func = flash_attn_func
    # transformers probes importlib.util.find_spec("flash_attn_interface");
    # a bare injected module has __spec__=None which makes find_spec RAISE.
    # A real (loader-less) spec keeps the probe happy, and the missing dist
    # metadata still makes transformers report flash-attn as unavailable.
    import importlib.machinery
    mod.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn_interface", loader=None)
    sys.modules["flash_attn_interface"] = mod

    # TokenRig.__init__ HARDCODES attn_implementation="flash_attention_2"
    # on AutoModelForCausalLM.from_config — transformers raises when
    # flash-attn is absent. Wrap from_config to force eager (traceable).
    import transformers

    _orig_from_config = transformers.AutoModelForCausalLM.from_config.__func__

    def _eager_from_config(cls, config, **kw):
        kw["attn_implementation"] = "eager"
        return _orig_from_config(cls, config, **kw)

    transformers.AutoModelForCausalLM.from_config = classmethod(
        _eager_from_config)

    # CPU-only torch: michelangelo's FLASH3 helper probes
    # torch.cuda.get_device_name(0) at IMPORT time (crashes without
    # CUDA), and flash_attention() enters torch.backends.cuda.sdp_kernel
    # — stub both so the CPU export can trace the eager/SDPA paths.
    if not torch.cuda.is_available():
        import contextlib
        torch.cuda.get_device_name = lambda *a, **k: "CPU"
        torch.backends.cuda.sdp_kernel = (
            lambda **kw: contextlib.nullcontext())


def build_model(repo: str, ckpt: str, dtype: torch.dtype):
    install_flash_attn_stub()
    sys.path.insert(0, os.path.abspath(repo))
    os.chdir(repo)
    from src.model.tokenrig import TokenRig

    log(f"loading checkpoint {ckpt} as {dtype}…")
    model = TokenRig.load_from_system_checkpoint(checkpoint_path=ckpt)
    model.eval()
    model = model.to(dtype=dtype, device="cpu")
    model.vae = model.vae.to(dtype=dtype, device="cpu")
    model.mesh_encoder = model.mesh_encoder.to(dtype=dtype, device="cpu")
    model.output_proj = model.output_proj.to(dtype=dtype, device="cpu")

    # Swap the LLM to eager attention for a traceable graph.
    try:
        model.transformer.config._attn_implementation = "eager"
        for m in model.transformer.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = "eager"
    except Exception as e:
        log(f"warning: could not force eager attention: {e}")
    return model


class DecomposedRMSNorm(nn.Module):
    """nn.RMSNorm lowers to aten::rms_norm, which the torchscript
    exporter can't map at opset 18 — decompose it into primitive ops
    (numerically identical)."""

    def __init__(self, src: nn.RMSNorm):
        super().__init__()
        self.weight = src.weight
        self.eps = src.eps if src.eps is not None else 1e-6

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def decompose_rmsnorm(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, DecomposedRMSNorm(child))
        else:
            decompose_rmsnorm(child)


class MeshCondWrapper(nn.Module):
    """vertices+normals -> LLM conditioning prefix (encode_mesh_cond)."""

    def __init__(self, mesh_encoder, output_proj, num_points):
        super().__init__()
        self.encoder = mesh_encoder.encoder
        self.output_proj = output_proj
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(num_points, self.encoder.token_num * 4, replace=False)
        self.register_buffer("candidate_indices", torch.from_numpy(indices))

    def forward(self, vertices, normals, fps_indices):
        pc = vertices[:, self.candidate_indices]
        feats = normals[:, self.candidate_indices]
        sampled_pc = pc[:, fps_indices]
        sampled_feats = feats[:, fps_indices]
        data = torch.cat((self.encoder.fourier_embedder(pc), feats), dim=-1)
        data = self.encoder.input_proj(data)
        sampled = torch.cat(
            (self.encoder.fourier_embedder(sampled_pc), sampled_feats), dim=-1
        )
        sampled = self.encoder.input_proj(sampled)
        latents = self.encoder.cross_attn(sampled, data)
        latents = self.encoder.self_attn(latents)
        if self.encoder.ln_post is not None:
            latents = self.encoder.ln_post(latents)
        return self.output_proj(latents)


class VaeCondWrapper(nn.Module):
    """cond [1,N,6] -> cond_latents [1,K,D] (SkinFSQCVAE cond path)."""

    def __init__(self, vae_model, cond_tokens, num_points):
        super().__init__()
        self.vae_model = vae_model
        self.cond_tokens = cond_tokens
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(num_points, cond_tokens * 4, replace=False)
        self.register_buffer("candidate_indices", torch.from_numpy(indices))

    def forward(self, cond, fps_indices):
        candidates = cond[:, self.candidate_indices]
        sampled = candidates[:, fps_indices]
        positions, features = cond[..., :3], cond[..., 3:]
        cond_kv = torch.cat((self.vae_model.embedder(positions), features), dim=-1)
        positions, features = sampled[..., :3], sampled[..., 3:]
        cond_q = torch.cat((self.vae_model.embedder(positions), features), dim=-1)
        return self.vae_model.cond_quant(
            self.vae_model.cond_encoder(cond_q, cond_kv)
        )


class SkinDecodeWrapper(nn.Module):
    """FSQ skin ids (0-based) + cond + cond_latents -> per-point weights.

    Folds FSQ.indices_to_codes and the optional up_perceiver into the
    graph so the runtime never has to reproduce FSQ math.
    """

    def __init__(self, vae, tokens_per_skin):
        super().__init__()
        self.vae = vae
        self.tokens_per_skin = tokens_per_skin

    def forward(self, skin_ids, cond, cond_latents):
        fsq = self.vae.model.FSQ
        z = fsq._indices_to_codes(skin_ids).to(cond.dtype)
        z = fsq.project_out(z)
        batch = skin_ids.shape[0]
        z = z.reshape(batch, self.tokens_per_skin, -1)
        logits = self.vae.decode(z=z, sampled_cond=cond,
                                 cond_tokens=cond_latents)
        return logits.reshape(batch, -1)


class EmbedWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.embed = transformer.get_input_embeddings()

    def forward(self, input_ids):
        return self.embed(input_ids)


class DecoderStepWrapper(nn.Module):
    """Qwen3 causal step with an explicit KV cache."""

    def __init__(self, transformer, num_layers):
        super().__init__()
        self.transformer = transformer
        self.num_layers = num_layers

    def forward(self, inputs_embeds, *past):
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for i in range(self.num_layers):
            k = past[2 * i]
            v = past[2 * i + 1]
            cache.update(k, v, i)
        out = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        # transformers 5.x: DynamicCache holds per-layer objects with
        # .keys/.values (the 4.x key_cache/value_cache lists are gone).
        presents = []
        pkv = out.past_key_values
        for i in range(self.num_layers):
            if hasattr(pkv, "layers"):
                presents.append(pkv.layers[i].keys)
                presents.append(pkv.layers[i].values)
            else:   # transformers 4.x fallback
                presents.append(pkv.key_cache[i])
                presents.append(pkv.value_cache[i])
        return (out.logits, *presents)


def export_onnx(module, args, in_names, out_names, dynamic_axes, path, opset=18):
    log(f"exporting {os.path.basename(path)}…")
    with torch.no_grad():
        torch.onnx.export(
            module, args, path,
            input_names=in_names, output_names=out_names,
            dynamic_axes=dynamic_axes, opset_version=opset,
            do_constant_folding=True, dynamo=False,
        )
    import onnx
    onnx.checker.check_model(onnx.load(path, load_external_data=False))
    log(f"  ok: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="SkinTokens repo checkout")
    ap.add_argument("--ckpt", required=True, help="TokenRig .ckpt path (relative to repo)")
    ap.add_argument("--out-dir", default="dist/skintokens_onnx")
    ap.add_argument("--num-points", type=int, default=54000,
                    help="fixed sampled-point count baked into the traced graphs")
    ap.add_argument(
        "--dtype",
        choices=("bf16", "fp32"),
        default="fp32",
        help="graph dtype; FP32 is the validated quality default",
    )
    ap.add_argument("--skip", default="", help="comma list of graphs to skip")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    skip = set(x for x in args.skip.split(",") if x)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = build_model(args.repo, args.ckpt, dtype)
    tok = model.tokenizer
    N = args.num_points

    llm_cfg = model.llm_config
    num_layers = llm_cfg.num_hidden_layers
    num_kv = llm_cfg.num_key_value_heads
    head_dim = getattr(llm_cfg, "head_dim",
                       llm_cfg.hidden_size // llm_cfg.num_attention_heads)

    # ── Runtime contract ────────────────────────────────────────────
    manifest = {
        "schema": "qtmesh-skintokens-onnx-v1",
        "num_points": N,
        "num_vertex_samples": 16384,
        "dtype": args.dtype,
        "batched_skin_decode": True,
        "external_fps_indices": True,
        "fps_candidates": {"seed": 0, "mesh": 2048, "vae": 1536},
        "tokens_per_skin": model.tokens_per_skin,
        "tokens_skin_cond": model.tokens_skin_cond,
        "vae_latent_channels": model.vae.latent_channels,
        "fsq_codebook_size": model.vae.vocab_size,
        "tokenizer": {
            "num_discrete": tok.num_discrete,
            "continuous_range": list(tok.continuous_range),
            "token_id_branch": tok.token_id_branch,
            "token_id_bos": tok.token_id_bos,
            "token_id_eos": tok.token_id_eos,
            "token_id_pad": tok.token_id_pad,
            "token_id_spring": tok.token_id_spring,
            "token_id_cls_none": tok.token_id_cls_none,
            "cls_token_id": dict(tok.cls_token_id),
            "parts_token_id": dict(tok.parts_token_id),
            "vocab_size": tok.vocab_size,
        },
        "llm": {
            "hidden_size": llm_cfg.hidden_size,
            "num_hidden_layers": num_layers,
            "num_key_value_heads": num_kv,
            "head_dim": head_dim,
            "full_vocab_size": model.vocab_size,
            "global_eos": model.eos,
        },
        "transform_config": model.transform_config.get("predict_transform")
            if isinstance(model.transform_config, dict) else None,
        "license": "MIT (code+weights, VAST-AI/SkinTokens); Qwen3-0.6B Apache-2.0",
    }
    manifest_path = os.path.join(out_dir, "skintokens.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log(f"wrote {manifest_path}")

    torch.manual_seed(0)
    np.random.seed(0)

    # ── mesh_cond.onnx ──────────────────────────────────────────────
    if "mesh_cond" not in skip:
        m = MeshCondWrapper(model.mesh_encoder, model.output_proj, N).eval()
        decompose_rmsnorm(m)
        v = torch.randn(1, N, 3, dtype=dtype)
        n = torch.nn.functional.normalize(torch.randn(1, N, 3, dtype=dtype), dim=-1)
        fps_indices = torch.arange(model.mesh_encoder.encoder.token_num)
        export_onnx(m, (v, n, fps_indices), ["vertices", "normals", "fps_indices"], ["cond_embeds"],
                    {}, os.path.join(out_dir, "mesh_cond.onnx"))

    # ── vae_cond.onnx ───────────────────────────────────────────────
    if "vae_cond" not in skip:
        m = VaeCondWrapper(model.vae.model, model.tokens_skin_cond, N).eval()
        cond = torch.randn(1, N, 6, dtype=dtype)
        fps_indices = torch.arange(model.tokens_skin_cond)
        export_onnx(m, (cond, fps_indices), ["cond", "fps_indices"], ["cond_latents"],
                    {}, os.path.join(out_dir, "vae_cond.onnx"))

    # ── skin_decode.onnx ────────────────────────────────────────────
    if "skin_decode" not in skip:
        m = SkinDecodeWrapper(model.vae, model.tokens_per_skin).eval()
        ids = torch.zeros(2, model.tokens_per_skin, dtype=torch.long)
        cond = torch.randn(2, N, 6, dtype=dtype)
        lat = torch.randn(2, model.tokens_skin_cond,
                          model.vae.latent_channels, dtype=dtype)
        export_onnx(m, (ids, cond, lat),
                    ["skin_ids", "cond", "cond_latents"], ["weights"],
                    {
                        "skin_ids": {0: "batch"},
                        "cond": {0: "batch"},
                        "cond_latents": {0: "batch"},
                        "weights": {0: "batch"},
                    }, os.path.join(out_dir, "skin_decode.onnx"))

    # ── embed.onnx ──────────────────────────────────────────────────
    if "embed" not in skip:
        m = EmbedWrapper(model.transformer).eval()
        ids = torch.zeros(1, 8, dtype=torch.long)
        export_onnx(m, (ids,), ["input_ids"], ["embeds"],
                    {"input_ids": {1: "seq"}, "embeds": {1: "seq"}},
                    os.path.join(out_dir, "embed.onnx"))

    # ── decoder.onnx ────────────────────────────────────────────────
    if "decoder" not in skip:
        m = DecoderStepWrapper(model.transformer, num_layers).eval()
        embeds = torch.randn(1, 4, llm_cfg.hidden_size, dtype=dtype)
        past = []
        in_names = ["inputs_embeds"]
        out_names = ["logits"]
        dyn = {"inputs_embeds": {1: "seq"}, "logits": {1: "seq"}}
        for i in range(num_layers):
            past.append(torch.randn(1, num_kv, 3, head_dim, dtype=dtype))
            past.append(torch.randn(1, num_kv, 3, head_dim, dtype=dtype))
            for kv in ("key", "value"):
                in_names.append(f"past.{i}.{kv}")
                out_names.append(f"present.{i}.{kv}")
                dyn[f"past.{i}.{kv}"] = {2: "past_seq"}
                dyn[f"present.{i}.{kv}"] = {2: "total_seq"}
        export_onnx(m, (embeds, *past), in_names, out_names, dyn,
                    os.path.join(out_dir, "decoder.onnx"))

    log("all requested graphs exported.")


if __name__ == "__main__":
    main()
