"""Energy and speed for each EmbeddingGemma 2 engine on the m6, one engine at a time, the same method for all:
10 s idle (macmon), then 30 s of back-to-back calls at one shape (macmon, 100 ms). Reported: emb/s, mean watts
(CPU+GPU+ANE and whole Mac), mJ per embedding, and mJ per embedding above idle.

  python -m bench.energy coreai MODEL.aimodel --table T.npy --entry s128b16=128x16 --cu ane --out r.json
  python -m bench.energy mlx --repo mlx-community/embeddinggemma-2-bf16 --shape 128x16 --out r.json
  python -m bench.energy llamacpp --url http://127.0.0.1:8090 --shape 128x16 --out r.json
  python -m bench.energy fluid FLUID_DIR --fn embed_128 --out r.json        (FluidInference Core ML, batch 1)
Each engine gets random token ids (seed 0) of the full length: no padding, so all engines do the same work."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from bench import common as C

IDLE_S, RUN_S = 10.0, 30.0


def measure(call, items: int, run_s: float = RUN_S, idle_s: float = IDLE_S) -> dict:
    for _ in range(5):
        call()
    with C.PowerSampler(None, 100) as idle:
        time.sleep(idle_s)
    with C.PowerSampler(None, 100) as ps:
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < run_s:
            call()
            n += 1
        dt = time.perf_counter() - t0
    i, p = idle.summary(), ps.summary()
    rate = n * items / dt
    out = {"rate": rate, "seconds": dt, "calls": n, "idle": i, "load": p}
    for k, name in (("all_power_mean", "compute"), ("sys_power_mean", "mac")):
        if p.get(k) is not None:
            out[f"{name}_W"] = p[k]
            out[f"{name}_mJ_per_emb"] = 1000 * p[k] / rate
            if i.get(k) is not None:
                out[f"{name}_mJ_per_emb_above_idle"] = 1000 * (p[k] - i[k]) / rate
    return out


def ids_for(seq, batch):
    return np.random.default_rng(0).integers(2, 260000, (batch, seq)).astype(np.int32)


def coreai_call(a):
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions
    from eg2_ane.common import host_embeds
    name, shape = a.entry.split("=")
    seq, batch = (int(x) for x in shape.split("x"))
    table = np.load(a.table)
    kind = {"gpu": ComputeUnitKind.gpu(), "ane": ComputeUnitKind.neural_engine()}[a.cu]
    loop = asyncio.new_event_loop()
    model = loop.run_until_complete(AIModel.load(
        a.target, specialization_options=SpecializationOptions.from_preferred_compute_unit_kind(kind)))
    fn = model.load_function(name)
    ids = ids_for(seq, batch)
    mask = NDArray(np.ones((batch, seq), np.int32))

    def call():  # the host-side table lookup is part of each call
        loop.run_until_complete(fn(inputs={"inputs_embeds": NDArray(host_embeds(table, ids)), "attention_mask": mask}))
    return call, batch, {"engine": f"Core AI {a.cu.upper()}", "model": Path(a.target).name, "shape": [seq, batch]}


def fluid_call(a):
    from bench.fluid import load, single_feed, table
    fn = a.fn
    m = load(a.target, [fn], a.cu)[fn]
    seq = int(fn.split("_")[1])
    feed = single_feed(table(a.target), ids_for(seq, 1)[0], np.ones(seq, np.int32), seq)
    return (lambda: m.predict(feed)), 1, {"engine": f"Core ML {a.cu.upper()} (FluidInference)",
                                          "model": "FluidInference/embeddinggemma-2-coreml " + fn, "shape": [seq, 1]}


def mlx_call(a):
    from bench.mlx_gpu import embed, load
    seq, batch = (int(x) for x in a.shape.split("x"))
    model, _ = load(a.repo)
    ids, mask = ids_for(seq, batch), np.ones((batch, seq), np.int32)
    return (lambda: embed(model, ids, mask)), batch, {"engine": "MLX GPU (mlx-vlm)", "model": a.repo, "shape": [seq, batch]}


def llamacpp_call(a):
    from bench.llamacpp_gpu import post
    seq, batch = (int(x) for x in a.shape.split("x"))
    seqs = ids_for(seq, batch).tolist()
    return (lambda: post(a.url, seqs)), batch, {"engine": "llama.cpp GPU (Metal)", "model": a.model_name,
                                                "shape": [seq, batch]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["coreai", "fluid", "mlx", "llamacpp"])
    ap.add_argument("target", nargs="?")
    ap.add_argument("--table")
    ap.add_argument("--entry")
    ap.add_argument("--cu", default="ane")
    ap.add_argument("--fn", default="embed_128", help="FluidInference function")
    ap.add_argument("--repo", default="mlx-community/embeddinggemma-2-bf16")
    ap.add_argument("--shape", default="128x16")
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model-name", default="ggml-org/embeddinggemma-2-GGUF Q8_0")
    ap.add_argument("--run-s", type=float, default=RUN_S)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    call, items, meta = {"coreai": coreai_call, "fluid": fluid_call, "mlx": mlx_call, "llamacpp": llamacpp_call}[a.kind](a)
    r = measure(call, items, a.run_s)
    res = {"host": C.host_info()} | meta | r
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{meta['engine']:22s} {meta['shape']}: {r['rate']:7.1f} emb/s | compute {r.get('compute_W', 0):5.2f} W "
          f"{r.get('compute_mJ_per_emb', 0):6.1f} mJ/emb | Mac {r.get('mac_W', 0):5.2f} W {r.get('mac_mJ_per_emb', 0):6.1f} "
          f"mJ/emb ({r.get('mac_mJ_per_emb_above_idle', 0):6.1f} above idle) | ane max {r['load'].get('ane_power_max')} W "
          f"gpu max {r['load'].get('gpu_power_max')} W", flush=True)


if __name__ == "__main__":
    main()
