"""Checks on the target Mac (needs only the runtime venv): quality against reference data, and speed.

  python -m eg2_ane.check quality MODEL_DIR [--ref reference/eg2_ref.npz] [--cu ane]
  python -m eg2_ane.check speed   MODEL_DIR [--seconds 20] [--cu ane]

quality: cosine to FP32 sentence-transformers embeddings of 64 WikiText-2 texts at 128 and 512 tokens
(reference/eg2_ref.npz, made with `python -m eg2_ane.reference`). PASS: mean cosine >= 0.9999, min >= 0.999, and every
output vector has length 1 +- 0.01 at every entry point.
speed: back-to-back calls of each entry point with random token ids, median of 5 windows, embeddings per second.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

COS_MEAN, COS_MIN, NORM_TOL = 0.9999, 0.999, 0.01
REF = Path(__file__).resolve().parents[1] / "reference" / "eg2_ref.npz"


def quality(folder, ref_path, cu):
    from eg2_ane.runtime import Embedder
    emb = Embedder(folder, cu)
    ref = dict(np.load(ref_path))
    res, ok = {}, True
    for name, (seq, batch) in emb.entries.items():
        ids, mask, r = ref[f"ids_{seq}"], ref[f"mask_{seq}"], ref[f"ref_{seq}"].astype(np.float64)
        rows = [list(i[: int(m.sum())]) for i, m in zip(ids, mask)]
        e = np.concatenate([emb.run(name, rows[j:j + batch]) for j in range(0, len(rows), batch)]).astype(np.float64)
        cos = (e * r).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(r, axis=1))
        norm = np.linalg.norm(e, axis=1)
        good = cos.mean() >= COS_MEAN and cos.min() >= COS_MIN and np.abs(norm - 1).max() <= NORM_TOL
        ok &= bool(good)
        res[name] = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "norm_min": float(norm.min()),
                     "norm_max": float(norm.max()), "pass": bool(good)}
        print(f"{name:8s} {cu}: cosine to FP32 mean {cos.mean():.5f} min {cos.min():.5f}, "
              f"length {norm.min():.4f}-{norm.max():.4f}  {'PASS' if good else 'FAIL'}", flush=True)
    print(f"QUALITY: {'PASS' if ok else 'FAIL'}")
    return ok, res


def speed(folder, seconds, cu):
    from eg2_ane.runtime import Embedder
    emb = Embedder(folder, cu)
    res = {}
    for name, (seq, batch) in emb.entries.items():
        rows = np.random.default_rng(0).integers(2, 260000, (batch, seq)).tolist()
        for _ in range(5):
            emb.run(name, rows)
        rates = []
        for _ in range(5):
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < seconds / 5:
                emb.run(name, rows)
                n += 1
            rates.append(n * batch / (time.perf_counter() - t0))
        res[name] = statistics.median(rates)
        print(f"{name:8s} {cu}: {res[name]:7.1f} embeddings/s ({seq} tokens, batch {batch})", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["quality", "speed"])
    ap.add_argument("folder")
    ap.add_argument("--ref", default=str(REF))
    ap.add_argument("--cu", default="ane", choices=["ane", "gpu", "cpu"])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "quality":
        ok, res = quality(a.folder, a.ref, a.cu)
    else:
        ok, res = True, speed(a.folder, a.seconds, a.cu)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
