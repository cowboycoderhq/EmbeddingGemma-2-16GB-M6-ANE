# Modifications

All files in this folder are derived from `google/embeddinggemma-2` at revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`. They were made with `eg2_ane.build` from
[github.com/cowboycoderhq/EmbeddingGemma-2-16GB-M6-ANE](https://github.com/cowboycoderhq/EmbeddingGemma-2-16GB-M6-ANE).
Only the text tower is converted. The vision and audio towers are not included.

| File | Change |
|---|---|
| `EmbeddingGemma-2-FP16-M6-ANE.aimodel` | **Modified.** The text transformer (24 layers with per-layer embeddings), mean pooling, the 512 -> 768 embedding projection and the L2 normalization, converted to an Apple Core AI model with four entry points (128 tokens x 1 and x 16, 512 tokens x 1 and x 4) that share one copy of the weights. Weights are FP16. The graph is rewritten for the Neural Engine: (B, C, 1, S) layout, 1x1 convolutions, attention one head at a time. For the FP16 range, the residual stream is scaled by 1/8; the attention, MLP and PLE outputs by 16, 256 and 16 before their post norms; the PLE projection weights by 1024; and each RMSNorm divides by its row maximum / 8 before the square. These changes are exact in FP32 (scale-invariant norms). The padding mask is built with `torch.where`. |
| `embed_table_fp16.npy` | **Modified.** The token embedding matrix (262,144 x 512) multiplied by the embedding scale (sqrt(512)) and by 1/8, stored as FP16. The model looks up tokens on the host. |
| `tokenizer/` | Unchanged copies of `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`. |
| `entries.json` | New: the entry point names and shapes. |
| `release.json` | New: the size and sha256 of every file. |
| `LICENSE`, `NOTICE` | The Apache License 2.0 text, and the notice for this conversion. |

Quality on the Neural Engine of a 16 GB M6 Mac mini: cosine 0.99998 (minimum 0.99994) to the FP32
sentence-transformers embeddings of 64 WikiText-2 texts at 128 and 512 tokens. SciFact (5,183 documents, 300 queries):
nDCG@10 within 0.001 of FP32 (see the GitHub repository).
