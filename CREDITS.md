# Credits

This repository is a small layer on top of other people's work.

## Model

- **[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)** by Google DeepMind, Apache License 2.0.

## Tools and ideas

- **[FluidInference/embeddinggemma-2-coreml](https://huggingface.co/FluidInference/embeddinggemma-2-coreml)**: the
  token lookup on the host, the `x / max|x|` RMSNorm for EmbeddingGemma 2, and the observation that at 512 tokens or
  fewer the sliding window needs no band. Its text model is also the batch-1 reference in our benchmark.
- **[Anemll](https://github.com/Anemll/anemll-embeddings)** and **[ANEMLL Forge](https://github.com/Anemll/anemll-forge)**
  by the ANEMLL contributors, MIT: the Core AI path, the overflow-safe RMSNorm, and the notes on FP16 subnormals on the
  ANE. No ANEMLL code is copied here.
- **Apple [coreai-torch](https://github.com/apple/coreai-torch), coreai-opt and coreai-core**: conversion and the
  runtime.
- **Apple [ml-ane-transformers](https://github.com/apple/ml-ane-transformers)**: the (B, C, 1, S) layout and per-head
  attention for the ANE.
- **[sentence-transformers](https://github.com/UKPLab/sentence-transformers)** and
  **[transformers](https://github.com/huggingface/transformers)**: the FP32 reference.
- **[eeANE](https://github.com/xhighhongo41/eeANE)**: scaling the residual stream down and the MLP output up by powers
  of 2 for EmbeddingGemma on the ANE; we use the same idea before each post norm.

## Comparisons

- [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT, with the
  [ggml-org/embeddinggemma-2-GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) Q8_0 and BF16 files.
- [MLX](https://github.com/ml-explore/mlx) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), MIT, with
  [mlx-community/embeddinggemma-2-bf16](https://huggingface.co/mlx-community/embeddinggemma-2-bf16).
- [Core ML Tools](https://github.com/apple/coremltools), BSD-3-Clause.
- [MTEB SciFact](https://huggingface.co/datasets/mteb/scifact) for the retrieval check.
- [macmon](https://github.com/vladkens/macmon) by vladkens, MIT, for the power measurements.

## Independence

This project is not affiliated with or endorsed by Google, Apple, FluidInference, ANEMLL, or the authors of the tools
above.
