# EmbeddingGemma-2-16GB-M6-ANE

**EmbeddingGemma 2 (text) on a 16 GB M6 Mac, on the Neural Engine: 319 embeddings/s, half the energy of the GPU, FP32
quality.** This repository converts the text tower of Google's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) to one Apple Core AI model that runs on the ANE,
with FP16 weights, four input shapes in one file, and the same embeddings as sentence-transformers (cosine 0.99998).

It is an independent project. It is not affiliated with or endorsed by Google or Apple. Much of what makes it work came
from others: see [CREDITS.md](CREDITS.md).

![EmbeddingGemma 2 on a 16 GB M6 Mac mini: GPU vs Neural Engine](docs/embeddinggemma-2-ane-vs-gpu.png)

## Results on a 16 GB M6 Mac mini

EmbeddingGemma 2 text, each engine alone on the same machine, the same token ids, 30 s of back-to-back calls after
10 s idle. Energy is whole-machine power from `macmon`. Raw results: [results/m6-release](results/m6-release).

| Engine | 128 tokens x 16 (emb/s) | 512 tokens x 4 (emb/s) | mJ per 512-token embedding | Watts (512 x 4) |
|---|---|---|---|---|
| GPU, llama.cpp `970b51c` (Q8_0, Metal) | 147 | 34 | 861 | 28.9 |
| GPU, MLX 0.32.4 (bf16, mlx-vlm) | 222 | 50 | 548 | 27.5 |
| GPU, Core AI (the same graph) | 270 | 62 | 603 | 37.3 |
| **ANE, this repo (FP16, Core AI)** | **319** | **81** | **291** | **23.6** |

MLX does not go faster with larger batches: its best was 229 emb/s at 128 x 32 and 54 at 512 x 4
([mlx-larger-batches.json](results/m6-release/mlx-larger-batches.json)). The Core AI GPU row is the faster of two
builds of the same graph on the GPU.

**One text per call.** At batch 1 the ANE build from
[FluidInference](https://huggingface.co/FluidInference/embeddinggemma-2-coreml) is faster than this one: 207 vs
183 emb/s at 128 tokens, 55 vs 53 at 512 tokens. The Core AI GPU path gives 78 and 48. This repo's advantage is the
batched shapes: FluidInference stays at 207 emb/s with 4 parallel requests and with its packed function (two
128-token texts per call), and this model gives 319 at 128 x 16 and 81 at 512 x 4.

**Quality.** Cosine to the FP32 sentence-transformers embeddings of 64 WikiText-2 texts:

| Build | 128 tokens, mean / min | 512 tokens, mean / min |
|---|---|---|
| **This repo (FP16, ANE)** | **0.99998 / 0.99995** | **0.99998 / 0.99994** |
| This repo with `--gelu sigmoid2` (11% slower) | 0.999997 / 0.99999 | 0.999998 / 0.99999 |
| FluidInference (FP16, Core ML, ANE) | 0.99997 / 0.99993 | 0.99997 / 0.99988 |
| [Anemll](https://huggingface.co/anemll/anemll-embeddinggemma-2-ane) (FP16, Core AI, ANE, 320 tokens) | 0.99997 / 0.99991 | 0.99997 / 0.99986 (320) |
| MLX bf16 (GPU) | 0.99994 / 0.99968 | 0.99995 / 0.99978 |
| llama.cpp Q8_0 (GPU) | 0.99991 / 0.99955 | 0.99991 / 0.99971 |

Retrieval, SciFact (5,183 documents at 512 tokens, 300 queries at 128 tokens): nDCG@10 is 0.8543 for this model and
0.8542 for FP32. All engines we measured are within 0.0013 of FP32
([results/m6-retrieval](results/m6-retrieval), [results/m5-retrieval](results/m5-retrieval)).

## What this repository does

- **Rewrites the model for the ANE** ([eg2_ane/model.py](eg2_ane/model.py)): (B, C, 1, S) activations, 1x1
  convolutions, attention one head at a time, the per-layer embeddings (PLE) as one projection per layer, and the
  262,144-row token lookup on the host. In FP32 the output equals sentence-transformers (cosine 0.9999999,
  [tests/test_model.py](tests/test_model.py)).
- **Keeps FP16 in range**: the residual stream is divided by 8, and each RMSNorm divides by its row maximum before it
  squares.
- **Keeps small values out of the FP16 subnormal range.** The ANE sets FP16 values below about 6e-5 to zero
  ([apple/coreai-torch#138](https://github.com/apple/coreai-torch/issues/138)). The attention, MLP and PLE outputs
  are scaled by 16, 256 and 16 before their post norms. With only that, the same FP16 graph gave cosine 0.999999 on
  the GPU and 0.99989 on the ANE. Two more changes, each exact in FP32, bring the ANE to 0.99998:
  - the PLE projection weights are multiplied by 1024 (its outputs had a maximum of 0.05, and 1% were subnormal);
  - each RMSNorm divides by its row maximum / 8, so the values are in [-8, 8] before the square.

  The native GELU in the MLP causes the rest. Its sigmoid form (`--gelu sigmoid2`, the same function) gives 0.999997
  and costs about 11% of the speed.
- **Works around a Core AI mask problem**: a padding mask written as arithmetic plus a `permute` converts to wrong
  attention in coreai-torch 0.4.2 ([apple/coreai-torch#137](https://github.com/apple/coreai-torch/issues/137)). The
  mask is built with `torch.where`.
- **One model, four input shapes** ([eg2_ane/build.py](eg2_ane/build.py)): 128 tokens x 1 and x 16, 512 tokens x 1 and
  x 4, sharing one copy of the FP16 weights (290 MB).
- **Runtime** ([eg2_ane/runtime.py](eg2_ane/runtime.py)): tokenizes like sentence-transformers, groups texts by length,
  and picks the mix of batch-1 and batched calls with the least time. It needs only numpy, tokenizers and the Core AI
  runtime.
- **Checks and benchmarks**: `eg2_ane.check quality` prints PASS or FAIL against FP32 reference embeddings;
  `eg2_ane.retrieval` runs SciFact for any engine; `bench/` measures Core AI, MLX, llama.cpp and FluidInference with
  energy.

## Layout

| Path | Contents |
|---|---|
| `eg2_ane/` | Model rewrite, build, runtime, quality and speed checks, reference data, retrieval check, release tools |
| `bench/` | The comparison engines (MLX, llama.cpp, FluidInference) and energy with `macmon` |
| `tests/` | Tests of the rewrite against sentence-transformers, and runtime tests |
| `reference/eg2_ref.npz` | Token ids and FP32 embeddings of 64 WikiText-2 texts, for `eg2_ane.check quality` |
| `results/` | The benchmark results above, one JSON file per engine and input shape |
| `model_card/` | The Hugging Face model card, NOTICE and MODIFICATIONS.md |
| `scripts/` | Setup, download and the chart |

## Requirements

- An Apple M6 Mac (tested: Mac mini, 16 GB, macOS 27.0.1). Other Macs with Core AI should run it; their speed is not
  measured here. On an M5 Max the same file gives the same quality.
- macOS 27 with the Core AI framework, and Python 3.11 to 3.13 (with [uv](https://docs.astral.sh/uv/),
  `scripts/setup.sh` gets Python 3.13).
- About 600 MB of disk for the model folder.

## Quick start

```sh
git clone https://github.com/cowboycoderhq/EmbeddingGemma-2-16GB-M6-ANE.git ~/src/EmbeddingGemma-2-16GB-M6-ANE
cd ~/src/EmbeddingGemma-2-16GB-M6-ANE
scripts/setup.sh          # makes .venv: numpy, tokenizers, coreai-core
scripts/download.sh       # model files -> ~/Models/EmbeddingGemma-2-FP16-M6-ANE, checked against release.json

# The first load compiles the model for the ANE (about 2 minutes); later loads use the cache
.venv/bin/python -m eg2_ane.check quality ~/Models/EmbeddingGemma-2-FP16-M6-ANE   # QUALITY: PASS
.venv/bin/python -m eg2_ane.check speed ~/Models/EmbeddingGemma-2-FP16-M6-ANE
```

```python
from eg2_ane.runtime import Embedder

emb = Embedder("~/Models/EmbeddingGemma-2-FP16-M6-ANE")
docs = emb.embed(["Mars is red because of iron oxide.", "Venus is the hottest planet."], prompt="document")
q = emb.embed(["Which planet is red?"], prompt="query")
print(docs @ q.T)    # [[0.795] [0.745]]: embeddings are L2-normalized, so the dot product is the cosine
```

`prompt` takes the model's task prompts (`query`, `document`, `qa`, `fact`, `sts`, `classification`, `clustering`,
`code`) or any prefix string. For Matryoshka sizes, keep the first 512, 256 or 128 values and normalize again.

## Build the model yourself

The build runs on any Apple-silicon Mac with macOS 27 (an M5 Max was used, about 1 minute).

```sh
scripts/setup.sh --build                       # makes .venv-build: + torch, coreai-torch 0.4.2, sentence-transformers
.venv-build/bin/python -m pytest -q tests      # the rewrite equals sentence-transformers (11 tests)
.venv-build/bin/python -m eg2_ane.build ~/Models/EmbeddingGemma-2-FP16-M6-ANE              # this release
.venv-build/bin/python -m eg2_ane.build ~/Models/EG2-sigmoid --gelu sigmoid2               # more accurate, slower
```

The build reads `google/embeddinggemma-2` at the revision in [eg2_ane/common.py](eg2_ane/common.py) (Apache 2.0, no
gate). Keep the pinned Core AI versions: coreai-torch 0.4.3 removed `AIProgram.optimize`.

## Limits

- Tested on one machine. The m6 speed numbers come from one 30 s run per engine and shape.
- Text only. The vision and audio towers are not converted (on the ANE they are slower than on the GPU in the
  FluidInference and Anemll measurements).
- Inputs are limited to 512 tokens; longer texts are truncated (EmbeddingGemma 2 supports 8,192).
- Each call has a fixed shape: three texts in the 16-text shape cost as much as sixteen. The runtime uses the batch-1
  shape when that is faster.
- For one text at a time, FluidInference is faster: 13% at 128 tokens, 4% at 512 tokens (see above).
- The ANE output is 0.99998 cosine to FP32, not bit-exact.
- Core AI is a beta (coreai-core 1.0.0b2). The versions are pinned.

## License

Code: MIT, copyright 2026 Cowboy Coder ([LICENSE](LICENSE)). The model files on Hugging Face are a converted form of
EmbeddingGemma 2 by Google DeepMind, under the Apache License 2.0; see [NOTICE](NOTICE). As the base model card states,
deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
