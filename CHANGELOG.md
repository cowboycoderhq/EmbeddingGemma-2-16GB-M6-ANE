# Changelog

## 0.1.0 (2026-10-23)

First release.

- EmbeddingGemma 2 text tower rebuilt for the Neural Engine: (B, C, 1, S) layout, 1x1 convolutions, per-head
  attention, per-layer embeddings as one projection per layer, host-side token lookup, and a `torch.where` padding
  mask. FP32 output equals sentence-transformers (cosine 0.9999999).
- FP16 on the ANE: residual / 8, attention, MLP and PLE outputs x16, x256 and x16 before their post norms, PLE
  projection weights x1024, and RMSNorm that divides by the row maximum / 8. Cosine to FP32 on the ANE: 0.99998.
- `--gelu sigmoid2` build option: the same GELU written with a sigmoid, cosine 0.999997, about 11% slower.
- One Core AI model with four entry points (128 x 1, 128 x 16, 512 x 1, 512 x 4), FP16 weights, 290 MB.
- `Embedder` runtime (numpy, tokenizers, coreai-core): tokenizes like sentence-transformers, groups texts by length,
  and chooses batch-1 or batched calls by estimated time.
- `eg2_ane.check quality` (PASS/FAIL against FP32 reference embeddings), `eg2_ane.check speed`, and
  `eg2_ane.retrieval` (SciFact nDCG@10 for any engine).
- Benchmark on a 16 GB M6 Mac mini against Core AI GPU, MLX, llama.cpp, FluidInference and Anemll, with energy.

Measured and dropped:
- INT8 weights: about the same speed, lower quality (cosine 0.9998).
- All heads in one matmul and softmax per layer (as FluidInference): 10-22% slower in our graph.
- One fused PLE projection for all layers: 1-6% slower.
- Explicit softmax: no quality change.
