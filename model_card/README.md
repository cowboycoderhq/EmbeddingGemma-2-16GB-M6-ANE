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

# EmbeddingGemma-2-FP16-M6-ANE

**EmbeddingGemma 2 (text) on the Apple Neural Engine of a 16 GB M6 Mac: 319 short-text embeddings per second, half the
energy per embedding of the GPU, and cosine 0.99998 to the FP32 model.** A Core AI build of the text tower of
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) with FP16 weights, four entry points in one
290 MB file, and the token lookup on the host.

Code, build steps, tests and the full benchmark:
[github.com/cowboycoderhq/EmbeddingGemma-2-16GB-M6-ANE](https://github.com/cowboycoderhq/EmbeddingGemma-2-16GB-M6-ANE).

## Results on a 16 GB M6 Mac mini

Each engine alone, 30 s of back-to-back calls with the same token ids, energy for the whole Mac from `macmon`.

| Engine | 128 tokens x 16 (emb/s) | 512 tokens x 4 (emb/s) | Energy per 512-token embedding |
|---|---|---|---|
| GPU, llama.cpp (Q8_0, Metal) | 147 | 34 | 861 mJ |
| GPU, MLX (bf16) | 222 | 50 | 548 mJ |
| GPU, Core AI (the same graph) | 270 | 62 | 603 mJ |
| **ANE, this model (FP16)** | **319** | **81** | **291 mJ** |

One text per call (128 tokens): 183 emb/s on the ANE. The FluidInference Core ML build is faster at batch 1
(207 emb/s); this model is faster with batches.

Quality on the ANE: cosine 0.99998 (minimum 0.99994) to the FP32 sentence-transformers embeddings at 128 and 512
tokens (64 WikiText-2 texts). SciFact nDCG@10: 0.8543 (FP32: 0.8542).

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
print(docs @ q.T)
```

The first load compiles the model for the Neural Engine (about 2 minutes); later loads use the cache. Inputs longer
than 512 tokens are truncated. Only the text tower is converted.

## License

Apache License 2.0, the license of the base model. This model is a converted form of EmbeddingGemma 2 by Google
DeepMind; see `NOTICE` and `MODIFICATIONS.md`. As the base model card states, deployments must adhere to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). This is an independent project; it is
not affiliated with or endorsed by Google or Apple.
