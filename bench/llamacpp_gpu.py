"""EmbeddingGemma 2 on the m6 GPU with llama.cpp (llama-server, Metal), through its HTTP /embedding endpoint with
token ids, so it embeds exactly the same token sequences as the Core AI and MLX runs.

  python -m bench.llamacpp_gpu bench --url http://127.0.0.1:8090 --out res.json   # 20 s per shape
  python -m bench.llamacpp_gpu check --url ... --ref reference/eg2_ref.npz            # cosine to the FP32 reference
Each request sends B sequences (the real tokens only: llama.cpp does not need padding). CONC requests run at the
same time so the server can batch across slots. Speed = sequences per second over the window."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import time
import urllib.request
from pathlib import Path

import numpy as np

from bench import common as C

SHAPES = {"s128b16": (128, 16), "s512b4": (512, 4)}


def post(url, seqs):
    body = json.dumps({"content": [[int(t) for t in s] for s in seqs]}).encode()
    req = urllib.request.Request(url + "/embedding", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    out = []
    for item in d if isinstance(d, list) else [d]:
        e = item["embedding"]
        out.append(np.asarray(e[0] if isinstance(e[0], list) else e, np.float64))
    return out


def bench(a):
    res = {"host": C.host_info(), "url": a.url, "concurrency": a.conc, "runs": {}}
    for name, (seq, batch) in SHAPES.items():
        rng = np.random.default_rng(0)
        reqs = [[rng.integers(2, 260000, seq).tolist() for _ in range(batch)] for _ in range(a.conc)]
        for _ in range(3):
            post(a.url, reqs[0])
        rates = []
        with C.PowerSampler(None, 100) as ps, cf.ThreadPoolExecutor(a.conc) as ex:
            for _ in range(5):
                n, t0 = 0, time.perf_counter()
                while time.perf_counter() - t0 < a.seconds / 5:
                    list(ex.map(lambda r: post(a.url, r), reqs))
                    n += a.conc * batch
                rates.append(n / (time.perf_counter() - t0))
        p = ps.summary()
        res["runs"][name] = {"seq": seq, "batch": batch, "metric": C.summarize(rates), "power": p}
        print(f"{name:8s} llama.cpp conc {a.conc}: {statistics.median(rates):7.1f} emb/s  gpu max {p.get('gpu_power_max')} W"
              f"  ane max {p.get('ane_power_max')} W  all {p.get('all_power_mean')} W", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def check(a):
    ref = dict(np.load(a.ref))
    res = {}
    for name, (seq, batch) in SHAPES.items():
        ids, mask, r = ref[f"ids_{seq}"], ref[f"mask_{seq}"], ref[f"ref_{seq}"].astype(np.float64)
        e = np.array(post(a.url, [i[: int(m.sum())] for i, m in zip(ids, mask)]))
        cos = (e * r).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(r, axis=1))
        res[name] = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min())}
        print(f"{name:8s} llama.cpp: cos mean {cos.mean():.5f} min {cos.min():.5f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "check"])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    bench(a) if a.cmd == "bench" else check(a)


if __name__ == "__main__":
    main()
