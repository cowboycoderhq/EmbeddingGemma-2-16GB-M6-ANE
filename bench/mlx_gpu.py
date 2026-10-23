"""EmbeddingGemma 2 (text) on the GPU with MLX (mlx-vlm, the commit from the mlx-community model card), the same batch
shapes and token ids as the Core AI runs. Needs its own venv:
  uv venv -p 3.11 .venv-mlx && uv pip install -p .venv-mlx/bin/python \\
    "git+https://github.com/Blaizzy/mlx-vlm.git@3d87e88402f307efbf68e568971aa887ee7d9ed0" "mlx>=0.32.3" numpy

  python -m bench.mlx_gpu check --repo mlx-community/embeddinggemma-2-bf16 --ref reference/eg2_ref.npz
  python -m bench.mlx_gpu bench --repo mlx-community/embeddinggemma-2-bf16 [--extra] --out res.json
Speed: one batch of B fixed-length sequences, timed with mx.eval, 20 s per shape, median of 5 windows."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from bench import common as C

SHAPES = {"s128b16": (128, 16), "s512b4": (512, 4)}
EXTRA = {"s128b32": (128, 32), "s128b64": (128, 64), "s512b8": (512, 8), "s512b16": (512, 16)}


def load(repo):
    from mlx_vlm.embedding_loader import load_embedding_model
    from mlx_vlm.utils import get_model_path
    return load_embedding_model(get_model_path(repo)), None


def embed(model, ids, mask):
    import mlx.core as mx
    e = model(input_ids=mx.array(ids), attention_mask=mx.array(mask)).text_embeds
    mx.eval(e)
    return e


def bench(a):
    model, _ = load(a.repo)
    res = {"host": C.host_info(), "repo": a.repo, "runtime": "mlx-vlm 0.7.4 (3d87e88), mlx 0.32.4", "runs": {}}
    for name, (seq, batch) in (SHAPES | (EXTRA if a.extra else {})).items():
        ids = np.random.default_rng(0).integers(2, 260000, (batch, seq)).astype(np.int32)
        mask = np.ones((batch, seq), np.int32)
        for _ in range(5):
            embed(model, ids, mask)
        rates = []
        with C.PowerSampler(None, 100) as ps:
            for _ in range(5):
                n, t0 = 0, time.perf_counter()
                while time.perf_counter() - t0 < a.seconds / 5:
                    embed(model, ids, mask)
                    n += 1
                rates.append(n * batch / (time.perf_counter() - t0))
        p = ps.summary()
        res["runs"][name] = {"seq": seq, "batch": batch, "metric": C.summarize(rates), "power": p}
        print(f"{name:8s} mlx {a.repo.split('/')[-1]}: {statistics.median(rates):7.1f} emb/s  "
              f"gpu max {p.get('gpu_power_max')} W  mac {p.get('sys_power_mean')} W", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def check(a):
    import mlx.core as mx
    model, _ = load(a.repo)
    ref = dict(np.load(a.ref))
    res = {"repo": a.repo}
    for name, (seq, batch) in SHAPES.items():
        ids, mask, r = ref[f"ids_{seq}"], ref[f"mask_{seq}"], ref[f"ref_{seq}"].astype(np.float64)
        e = np.concatenate([np.array(embed(model, ids[j:j + batch], mask[j:j + batch]).astype(mx.float32), np.float64)
                            for j in range(0, len(ids), batch)])
        cos = (e * r).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(r, axis=1))
        res[name] = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min())}
        print(f"{name:8s} mlx {a.repo.split('/')[-1]}: cos mean {cos.mean():.5f} min {cos.min():.5f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "check"])
    ap.add_argument("--repo", default="mlx-community/embeddinggemma-2-bf16")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--extra", action="store_true", help="also larger batches (GPU throughput ceiling)")
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    bench(a) if a.cmd == "bench" else check(a)


if __name__ == "__main__":
    main()
