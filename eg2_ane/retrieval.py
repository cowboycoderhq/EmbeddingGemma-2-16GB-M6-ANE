"""Retrieval check for EmbeddingGemma 2 builds: MTEB SciFact (BEIR), nDCG@10, same token ids for every engine.

  build Mac: python -m eg2_ane.retrieval prep scifact.npz       # tokenize + FP32 reference (build venv, MPS)
  target:    python -m eg2_ane.retrieval embed ENGINE scifact.npz OUT.npz [options]
             ENGINE: ours (--model MODEL_DIR --cu ane|gpu), fluid (--folder FLUID_DIR), mlx, llamacpp (--url)
  any:       python -m eg2_ane.retrieval score scifact.npz OUT.npz [OUT2.npz ...]
Documents: "title: {title} | text: {text}", 512 tokens. Queries: "task: search result | query: {q}", 128 tokens.
The tokenizer call is the sentence-transformers 6.2 one (eg2_model.tokenize)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

DOC_SEQ, Q_SEQ = 512, 128


def prep(out):
    import torch
    from datasets import load_dataset
    from eg2_ane import model as M
    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries = load_dataset("mteb/scifact", "queries", split="queries")
    qrels = load_dataset("mteb/scifact", "default", split="test")
    st = M.load_st()
    tok = st.tokenizer
    test_q = sorted({str(r["query-id"]) for r in qrels})
    qtext = {str(r["_id"]): r["text"] for r in queries}
    docs = [(str(r["_id"]), f"title: {r['title'] or 'none'} | text: {r['text']}") for r in corpus]
    qs = [(q, "task: search result | query: " + qtext[q]) for q in test_q]

    def toks(texts, seq):
        ids, mask = zip(*(M.tokenize(tok, t, seq) for t in texts))
        return np.concatenate(ids), np.concatenate(mask)
    d_ids, d_mask = toks([t for _, t in docs], DOC_SEQ)
    q_ids, q_mask = toks([t for _, t in qs], Q_SEQ)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    st = st.to(dev)

    def ref(ids, mask, bs=32):
        out = []
        with torch.no_grad():
            for j in range(0, len(ids), bs):
                o = st({"input_ids": torch.from_numpy(ids[j:j + bs]).long().to(dev),
                        "attention_mask": torch.from_numpy(mask[j:j + bs]).long().to(dev)})
                out.append(o["sentence_embedding"].float().cpu().numpy())
        return np.concatenate(out)
    t0 = time.time()
    d_ref, q_ref = ref(d_ids, d_mask), ref(q_ids, q_mask)
    print(f"reference on {dev}: {time.time() - t0:.0f} s", flush=True)
    rel = [(test_q.index(str(r["query-id"])), str(r["corpus-id"]), int(r["score"])) for r in qrels]
    doc_index = {d: i for i, (d, _) in enumerate(docs)}
    qrel = np.array([(qi, doc_index[d], s) for qi, d, s in rel if d in doc_index], np.int32)
    np.savez(out, d_ids=d_ids, d_mask=d_mask, q_ids=q_ids, q_mask=q_mask, d_ref=d_ref, q_ref=q_ref, qrel=qrel)
    print(f"{out}: {len(docs)} docs, {len(qs)} queries, {len(qrel)} qrels, "
          f"doc tokens mean {d_mask.sum(1).mean():.0f}, truncated at {DOC_SEQ}: {(d_mask.sum(1) == DOC_SEQ).mean():.1%}")


def ndcg10(q, d, qrel):
    s = q @ d.T
    top = np.argsort(-s, axis=1)[:, :10]
    rel = {}
    for qi, di, sc in qrel:
        rel.setdefault(int(qi), {})[int(di)] = int(sc)
    out = []
    for qi in range(len(q)):
        r = rel.get(qi, {})
        dcg = sum(r.get(int(di), 0) / np.log2(k + 2) for k, di in enumerate(top[qi]))
        ideal = sorted(r.values(), reverse=True)[:10]
        idcg = sum(v / np.log2(k + 2) for k, v in enumerate(ideal))
        out.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(out)), top


def _n(e):
    e = np.asarray(e, np.float64)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def score(data, outs):
    z = dict(np.load(data))
    ref_n, ref_top = ndcg10(_n(z["q_ref"]), _n(z["d_ref"]), z["qrel"])
    print(f"{'FP32 sentence-transformers':34s} nDCG@10 {ref_n:.4f}")
    res = {"fp32": ref_n}
    for o in outs:
        e = dict(np.load(o))
        n, top = ndcg10(_n(e["q"]), _n(e["d"]), z["qrel"])
        cd = (_n(e["d"]) * _n(z["d_ref"])).sum(1)
        top1 = (top[:, 0] == ref_top[:, 0]).mean()
        over = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top, ref_top)])
        name = str(e.get("name", Path(o).stem))
        print(f"{name:34s} nDCG@10 {n:.4f} (delta {n - ref_n:+.4f}) | doc cos mean {cd.mean():.6f} min {cd.min():.5f}"
              f" | top-1 same {top1:.1%} | top-10 overlap {over:.1%}")
        res[name] = {"ndcg10": n, "doc_cos_mean": float(cd.mean()), "doc_cos_min": float(cd.min()),
                     "top1_same": float(top1), "top10_overlap": float(over)}
    return res


# ---------- engines (m6) ----------

def embed_ours(z, a):
    from eg2_ane.runtime import Embedder
    emb = Embedder(a["model"], a.get("cu", "ane"))
    res = {}
    for key, seq in (("d", DOC_SEQ), ("q", Q_SEQ)):
        ids, mask = z[f"{key}_ids"], z[f"{key}_mask"]
        rows = [i[: int(m.sum())].tolist() for i, m in zip(ids, mask)]
        out, start = [], 0
        for name, take in emb.plan(seq, len(rows)):
            out.append(emb.run(name, rows[start:start + take]))
            start += take
        res[key] = np.concatenate(out)
    return res


def embed_fluid(z, a):
    from bench.fluid import load, single_feed, table
    tab = table(a["folder"])
    ms = load(a["folder"], ["embed_128", "embed_512"], a.get("cu", "ane"))
    res = {}
    for key, seq in (("d", DOC_SEQ), ("q", Q_SEQ)):
        m = ms[f"embed_{seq}"]
        res[key] = np.concatenate([np.asarray(m.predict(single_feed(tab, i, k, seq))["embedding"], np.float32)
                                   for i, k in zip(z[f"{key}_ids"], z[f"{key}_mask"])])
    return res


def embed_mlx(z, a):
    import mlx.core as mx
    from bench.mlx_gpu import embed, load
    model, _ = load(a.get("repo", "mlx-community/embeddinggemma-2-bf16"))
    res = {}
    for key, b in (("d", 4), ("q", 16)):
        ids, mask = z[f"{key}_ids"], z[f"{key}_mask"]
        res[key] = np.concatenate([np.array(embed(model, ids[j:j + b], mask[j:j + b]).astype(mx.float32))
                                   for j in range(0, len(ids), b)])
    return res


def embed_llamacpp(z, a):
    from bench.llamacpp_gpu import post
    url = a.get("url", "http://127.0.0.1:8090")
    res = {}
    for key, b in (("d", 4), ("q", 16)):
        ids, mask = z[f"{key}_ids"], z[f"{key}_mask"]
        seqs = [i[: int(m.sum())].tolist() for i, m in zip(ids, mask)]
        res[key] = np.concatenate([np.array(post(url, seqs[j:j + b]), np.float32) for j in range(0, len(seqs), b)])
    return res


def main():
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2])
    elif cmd == "score":
        r = score(sys.argv[2], sys.argv[3:])
        Path(sys.argv[-1]).with_suffix(".score.json").write_text(json.dumps(r, indent=1))
    else:
        engine, data, out = sys.argv[2:5]
        opts = dict(zip([k.lstrip("-") for k in sys.argv[5::2]], sys.argv[6::2]))
        z = dict(np.load(data))
        t0 = time.time()
        e = {"ours": embed_ours, "fluid": embed_fluid, "mlx": embed_mlx, "llamacpp": embed_llamacpp}[engine](z, opts)
        dt = time.time() - t0
        name = opts.get("name", engine)
        np.savez(out, d=e["d"], q=e["q"], name=name, seconds=dt)
        print(f"{name}: {len(e['d'])} docs + {len(e['q'])} queries in {dt:.0f} s", flush=True)


if __name__ == "__main__":
    main()
