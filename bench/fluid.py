"""FluidInference/embeddinggemma-2-coreml (Core ML, FP16, batch 1), with the same texts and token ids as our runs.
Download the text files first (each file by name: `--include dir/*` misses the nested package files):
  for f in EmbeddingGemma2Text.mlpackage/Manifest.json EmbeddingGemma2Text.mlpackage/Data/com.apple.CoreML/model.mlmodel \\
           EmbeddingGemma2Text.mlpackage/Data/com.apple.CoreML/weights/weight.bin embeddings.bf16 config.json; do
    hf download FluidInference/embeddinggemma-2-coreml $f --local-dir FLUID_DIR; done

  python -m bench.fluid check FLUID_DIR [--ref reference/eg2_ref.npz]       (needs coremltools)
The first ANE compile takes about 3 minutes. Speed and energy: python -m bench.energy fluid FLUID_DIR --fn embed_128"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def table(folder) -> np.ndarray:
    """embeddings.bf16 (raw bfloat16, 262,144 x 512) x sqrt(512), as FP16 (the card's recipe)."""
    raw = np.fromfile(Path(folder) / "embeddings.bf16", np.uint16).reshape(262144, 512)
    return ((raw.astype(np.uint32) << 16).view(np.float32) * np.float32(22.627417)).astype(np.float16)


def load(folder, fns, cu="ane"):
    import coremltools as ct
    pkg = Path(folder) / "EmbeddingGemma2Text.mlpackage"
    mlc = Path(folder) / "EmbeddingGemma2Text.mlmodelc"
    if not mlc.exists():
        ct.utils.compile_model(str(pkg), str(mlc))
    units = {"ane": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU}[cu]
    return {fn: ct.models.CompiledMLModel(str(mlc), compute_units=units, function_name=fn) for fn in fns}


def single_feed(tab, ids, mask, seq):
    return {"inputs_embeds": tab[np.asarray(ids)[None, :seq]].astype(np.float16),
            "attention_mask": np.asarray(mask)[None, :seq].astype(np.float16)}


def check(a):
    tab = table(a.folder)
    ms = load(a.folder, ["embed_128", "embed_512"], a.cu)
    ref = dict(np.load(a.ref))
    res = {}
    for seq in (128, 512):
        ids, mask, r = ref[f"ids_{seq}"], ref[f"mask_{seq}"], ref[f"ref_{seq}"].astype(np.float64)
        e = np.concatenate([np.asarray(ms[f"embed_{seq}"].predict(single_feed(tab, i, m, seq))["embedding"],
                                       np.float64) for i, m in zip(ids, mask)])
        cos = (e * r).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(r, axis=1))
        res[f"embed_{seq}"] = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min())}
        print(f"fluid {a.cu} embed_{seq}: cos mean {cos.mean():.5f} min {cos.min():.5f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("folder")
    ap.add_argument("--ref", default=str(Path(__file__).resolve().parents[1] / "reference" / "eg2_ref.npz"))
    ap.add_argument("--cu", default="ane")
    ap.add_argument("--out")
    check(ap.parse_args(argv))


if __name__ == "__main__":
    main()
