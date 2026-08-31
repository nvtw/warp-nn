# Nemotron-H follow-up

The dependency-free BF16 fallback is graph-captured and validated on an RTX 3080. Before calling the FP8 model
production-ready, validate these hardware-dependent pieces on Blackwell:

- Compare logits and generated text with the official 5.3 GB checkpoint and Transformers/vLLM.
- Add a native E4M3 tensor-core GEMM path. The current path dequantizes weights to BF16 once at load time, which is
  correct but roughly doubles weight memory and leaves Blackwell FP8 throughput unused.
- Add optional E4M3 KV-cache storage to match `hf_quant_config.json`; the current BF16 cache favors a validated,
  simple path at twice the checkpoint's intended cache memory.
- Benchmark long-prompt prefill and add a flash/chunked attention kernel if attention dominates. Decode already uses
  bounded preallocated caches, but the general streaming fallback is not an optimized long-context prefill kernel.
- Tune native FP8 kernels only from Blackwell measurements; retain the portable Warp BF16 fallback.
