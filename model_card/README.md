---
license: apache-2.0
base_model: google/embeddinggemma-2
base_model_relation: quantized
pipeline_tag: sentence-similarity
library_name: coreai
tags:
- embeddinggemma
- apple-neural-engine
- ane
- core-ai
- apple-silicon
- m6
language:
- multilingual
---

<h1 align="center">EmbeddingGemma-2-FP16-M6-ANE</h1>

<p align="center">
  <sub>by <a href="https://x.com/CowboyCoderHQ">Cowboy Coder</a></sub>
  <br><br>
  <a href="https://x.com/CowboyCoderHQ"><img src="https://img.shields.io/badge/Follow%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X"></a>
  <a href="https://github.com/cowboycoderhq/EmbeddingGemma-2-16GB-M6-ANE"><img src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Code on GitHub"></a>
</p>

The text tower of [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) for the **Apple Neural
Engine** of a 16 GB M6 Mac: **one Core AI file, FP16 weights, the whole model on the ANE, cosine 0.99998 to the FP32
model**. Code, build steps, tests and the full benchmark:
[github.com/cowboycoderhq/EmbeddingGemma-2-16GB-M6-ANE](https://github.com/cowboycoderhq/EmbeddingGemma-2-16GB-M6-ANE).

| | |
|---|---|
| Short texts (128 tokens, 16 per call) | **319 embeddings/s** on the ANE. Fastest GPU path: 270 (Core AI) |
| Long texts (512 tokens, 4 per call) | **81 embeddings/s** on the ANE. Fastest GPU path: 62 (Core AI) |
| One query at a time (128 tokens) | **183 embeddings/s** on the ANE. Core AI GPU: 78. FluidInference (ANE): 207 |
| Energy per long-text embedding, whole Mac | **291 mJ** on the ANE. GPU paths: 548–861 mJ |
| Quality, cosine to FP32 (64 texts, 128 and 512 tokens) | **0.99998** mean, 0.99994 minimum |
| Quality, SciFact retrieval (nDCG@10) | **0.8543** on the ANE. FP32 reference: 0.8542 |

Each engine alone on a 16 GB M6 Mac mini, 30 s of back-to-back calls with the same token ids, energy for the whole
Mac from `macmon`.

## Files

| File | Contents |
|---|---|
| `EmbeddingGemma-2-FP16-M6-ANE.aimodel` | Core AI model, entry points `s128b1`, `s128b16`, `s512b1`, `s512b4` |
| `embed_table_fp16.npy` | Token lookup table (262,144 x 512, FP16, 268 MB), used on the host |
| `tokenizer/` | Tokenizer files (unchanged) |
| `entries.json` | Entry point shapes |
| `release.json` | Size and sha256 of every file |
| `LICENSE`, `NOTICE`, `MODIFICATIONS.md` | Apache License 2.0, the notice, and the list of changes |

## Use

```python
from eg2_ane.runtime import Embedder        # from the GitHub repo (needs only numpy, tokenizers, coreai-core)
emb = Embedder("EmbeddingGemma-2-FP16-M6-ANE")
docs = emb.embed(["Mars is red because of iron oxide.", "Venus is the hottest planet."], prompt="document")
q = emb.embed(["Which planet is red?"], prompt="query")
print(docs @ q.T)    # [[0.795] [0.745]]
```

The first load compiles the model for the Neural Engine (about 2 minutes); later loads use the cache. Inputs longer
than 512 tokens are truncated. Only the text tower is converted.

## Limits

- Measured on one 16 GB M6 Mac mini. Other Macs are not measured.
- For one text per call, the FluidInference Core ML build is faster (13% at 128 tokens, 4% at 512).
- Core AI is beta (coreai-core 1.0.0b2).

## License

Apache License 2.0, the license of the base model. This model is a converted form of EmbeddingGemma 2 by Google
DeepMind; see `NOTICE` and `MODIFICATIONS.md`. As the base model card states, deployments must adhere to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). This is an independent project; it is
not affiliated with or endorsed by Google or Apple.
