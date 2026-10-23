"""Build the Core AI model: one .aimodel with several entry points that share one copy of the weights.

  python -m eg2_ane.build OUT_DIR [--entry s128b1=128x1 ...] [--gelu tanh|sigmoid2]

OUT_DIR gets:
  EmbeddingGemma-2-FP16-M6-ANE.aimodel   the model (FP16 weights, 279 MB)
  embed_table_fp16.npy                   the host-side token table (268 MB)
  tokenizer/                             the tokenizer files
  entries.json                           entry name -> [seq, batch]
Needs coreai-torch 0.4.2 (requirements-build.txt). Runs on any Apple-silicon Mac with macOS 27; the ANE compile happens
on the target Mac at first load.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from eg2_ane import model as M

DEFAULT_ENTRIES = ["s128b1=128x1", "s128b16=128x16", "s512b1=512x1", "s512b4=512x4"]


def parse_entries(items):
    out = {}
    for it in items:
        name, shape = it.split("=")
        seq, batch = (int(x) for x in shape.split("x"))
        out[name] = (seq, batch)
    return out


def build(out_dir: Path, entries: dict, gelu: str = "tanh", revision: str | None = M.REVISION) -> Path:
    import coreai_torch
    import torch
    from coreai_opt.casting import cast_to_16_bit_precision
    M.GELU = gelu
    st = M.load_sentence_transformer(revision=revision)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for name, (seq, batch) in entries.items():
        model = M.EG2TextANE(st, seq).eval().to(torch.float16)
        x = torch.zeros(batch, M.HIDDEN, 1, seq, dtype=torch.float16)
        m = torch.ones(batch, seq, dtype=torch.int32)
        ep = torch.export.export(model, (x, m), strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=["inputs_embeds", "attention_mask"], output_names=["embedding"],
                                  entrypoint_name=name)
        print(f"exported {name}: {seq} tokens x {batch}", flush=True)
    t0 = time.time()
    prog = conv.to_coreai()
    prog.optimize()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "EmbeddingGemma-2-FP16-M6-ANE.aimodel"
    shutil.rmtree(path, ignore_errors=True)
    prog.save_asset(path)
    np.save(out_dir / "embed_table_fp16.npy", M.embed_table(st))
    st.tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    (out_dir / "entries.json").write_text(json.dumps({k: list(v) for k, v in entries.items()}, indent=1))
    mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    print(f"saved {path} ({mb:.0f} MB, {time.time() - t0:.0f} s after export)", flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--entry", action="append", help="name=SEQxBATCH (repeat); default: " + " ".join(DEFAULT_ENTRIES))
    ap.add_argument("--gelu", choices=["tanh", "sigmoid2"], default="tanh",
                    help="MLP GELU form: tanh (faster) or sigmoid2 (more accurate on the ANE, about 11%% slower)")
    ap.add_argument("--revision", default=M.REVISION, help="google/embeddinggemma-2 revision")
    a = ap.parse_args(argv)
    build(a.out_dir, parse_entries(a.entry or DEFAULT_ENTRIES), a.gelu, a.revision)


if __name__ == "__main__":
    main()
