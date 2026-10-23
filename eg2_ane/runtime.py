"""Run the Core AI model on the Neural Engine and embed text.

    from eg2_ane.runtime import Embedder
    emb = Embedder("~/Models/EmbeddingGemma-2-FP16-M6-ANE")          # the downloaded or built folder
    vecs = emb.embed(["first text", "second text"], prompt="document")  # (N, 768) float32, L2-normalized
    q = emb.embed(["what is the red planet?"], prompt="query")

Texts are tokenized like sentence-transformers, grouped by the shortest entry point that fits their length, and sent
in the batch size that takes the least time (one text goes through the batch-1 entry point, many texts through the
batched one). The first load compiles the model for the ANE (about 2 minutes); later loads use the cache.
Only numpy, tokenizers and the Core AI runtime (coreai-core) are needed.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np

from eg2_ane.common import DIM, PROMPTS

# milliseconds per call on the ANE of a 16 GB M6 Mac mini (results/m6-release); used only to choose the batch size
MS_PER_CALL = {(128, 1): 5.5, (128, 16): 50.2, (512, 1): 18.9, (512, 4): 49.4}


def _to_numpy(v):
    for attr in ("numpy", "to_numpy"):
        if hasattr(v, attr):
            return np.asarray(getattr(v, attr)())
    return np.asarray(v)


def _ms(seq: int, batch: int) -> float:
    return MS_PER_CALL.get((seq, batch), seq * batch / 40.0)


class Embedder:
    def __init__(self, folder: str | Path, compute_unit: str = "ane"):
        from tokenizers import Tokenizer
        self.dir = Path(folder).expanduser()
        self.model_path = next(self.dir.glob("*.aimodel"))
        self.entries = {k: tuple(v) for k, v in json.loads((self.dir / "entries.json").read_text()).items()}
        self.table = np.load(self.dir / "embed_table_fp16.npy", mmap_mode="r")
        self.tok = Tokenizer.from_file(str(self.dir / "tokenizer" / "tokenizer.json"))
        self.max_seq = max(s for s, _ in self.entries.values())
        self.loop = asyncio.new_event_loop()
        self.model = self.loop.run_until_complete(self._load(compute_unit))
        self.fns = {name: self.model.load_function(name) for name in self.entries}

    async def _load(self, compute_unit: str):
        from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
        kind = {"ane": ComputeUnitKind.neural_engine(), "gpu": ComputeUnitKind.gpu(), "cpu": ComputeUnitKind.cpu()}
        return await AIModel.load(self.model_path, specialization_options=SpecializationOptions
                                  .from_preferred_compute_unit_kind(kind[compute_unit]))

    def ids(self, text: str) -> list[int]:
        """Token ids with BOS/EOS as sentence-transformers makes them: no strip; above max_seq, keep the first
        max_seq - 1 tokens and the end token."""
        ids = self.tok.encode(text).ids
        if len(ids) > self.max_seq:
            ids = ids[: self.max_seq - 1] + [ids[-1]]
        return ids

    def plan(self, n_tokens: int, n_texts: int) -> list[tuple[str, int]]:
        """(entry, texts) calls for n_texts of length <= n_tokens: the shortest sequence that fits, then the mix of
        batch sizes with the least estimated time (a small dynamic program over the number of texts)."""
        seq = min(s for s, _ in self.entries.values() if s >= n_tokens)
        options = [(b, k) for k, (s, b) in self.entries.items() if s == seq]
        best = [(0.0, None)] + [(math.inf, None)] * n_texts  # best[n] = (ms, (batch, entry) of the last call)
        for n in range(1, n_texts + 1):
            for b, k in options:
                t = best[max(0, n - b)][0] + _ms(seq, b)
                if t < best[n][0]:
                    best[n] = (t, (b, k))
        calls, n = [], n_texts
        while n:
            b, k = best[n][1]
            calls.append((k, min(b, n)))
            n -= min(b, n)
        return calls[::-1]

    def run(self, name: str, id_rows: list[list[int]]) -> np.ndarray:
        from coreai.runtime import NDArray
        seq, batch = self.entries[name]
        ids = np.zeros((batch, seq), np.int64)
        mask = np.zeros((batch, seq), np.int32)
        for i, row in enumerate(id_rows):
            ids[i, :len(row)] = row
            mask[i, :len(row)] = 1
        for i in range(len(id_rows), batch):  # unused rows: copy row 0 (results are dropped)
            ids[i], mask[i] = ids[0], mask[0]
        x = np.ascontiguousarray(self.table[ids].transpose(0, 2, 1)[:, :, None, :])
        out = self.loop.run_until_complete(self.fns[name](
            inputs={"inputs_embeds": NDArray(x), "attention_mask": NDArray(mask)}))
        e = _to_numpy(out["embedding"] if isinstance(out, dict) else list(out.values())[0])
        return e.reshape(batch, -1)[: len(id_rows)].astype(np.float32)

    def embed(self, texts: list[str], prompt: str | None = None) -> np.ndarray:
        """prompt: None, a name in PROMPTS ("query", "document", "sts", ...), or a literal prefix string."""
        pre = PROMPTS.get(prompt, prompt or "")
        rows = [self.ids(pre + t) for t in texts]
        out = np.zeros((len(texts), DIM), np.float32)
        seqs = sorted({s for s, _ in self.entries.values()})
        groups: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(min(s for s in seqs if s >= len(r)), []).append(i)
        for seq, idx in groups.items():
            j = 0
            for name, take in self.plan(seq, len(idx)):
                part = idx[j:j + take]
                out[part] = self.run(name, [rows[i] for i in part])
                j += take
        return out
