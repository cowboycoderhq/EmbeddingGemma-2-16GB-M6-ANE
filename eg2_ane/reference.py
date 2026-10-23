"""Make the quality reference data (on a Mac with the build venv): token ids, masks, and FP32 sentence-transformers
embeddings of 64 WikiText-2 test texts at 128 and 512 tokens (query and document prompts in turn).

  python -m eg2_ane.reference reference/eg2_ref.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from eg2_ane import model as M


def wikitext_texts(n: int) -> list[str]:
    from datasets import load_dataset
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    lines = [l.strip() for l in rows if len(l.split()) >= 12 and not l.strip().startswith("=")]
    return [(M.PROMPTS["query"] if i % 2 else M.PROMPTS["document"]) + l for i, l in enumerate(lines[:n])]


def main():
    out = Path(sys.argv[1])
    st = M.load_sentence_transformer()
    texts = wikitext_texts(64)
    data = {}
    for seq in (128, 512):
        st.max_seq_length = seq
        pairs = [M.tokenize(st.tokenizer, t, seq) for t in texts]
        data[f"ids_{seq}"] = np.concatenate([i for i, _ in pairs])
        data[f"mask_{seq}"] = np.concatenate([m for _, m in pairs])
        data[f"ref_{seq}"] = st.encode(texts, normalize_embeddings=True, batch_size=8).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print("saved", out, {k: v.shape for k, v in data.items()})


if __name__ == "__main__":
    main()
