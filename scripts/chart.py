"""Make the README chart from results/m6-release/*.json (the same style as the Bonsai and EmbeddingGemma-300M charts).
  python scripts/chart.py docs/embeddinggemma-2-ane-vs-gpu.png"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results" / "m6-release"
GREY, LIGHT, BLUE = "#9aa0a6", "#c7cbd1", "#1a73e8"
GPU = [("llamacpp-q8", "GPU\nllama.cpp"), ("mlx-bf16", "GPU\nMLX"), ("coreai-gpu", "GPU\nCore AI")]


def load(prefix, shape):
    p = R / f"{prefix}-{shape}.json"
    return json.loads(p.read_text()) if p.exists() else None


def bars(ax, rows, title, ylabel, fmt, lower_better=False):
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    b = ax.bar(range(len(vals)), vals, color=[r[2] for r in rows], width=0.62)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(v), ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(vals)), labels, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel + (" (lower is better)" if lower_better else ""), fontsize=10)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)


def engine_rows(shape, key):
    rows = [(label, load(p, shape)[key], GREY) for p, label in GPU if load(p, shape)]
    return rows + [("ANE\n(this repo)", load("ane", shape)[key], BLUE)]


def main():
    out = Path(sys.argv[1])
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.2))
    bars(axes[0], engine_rows("s128b16", "rate"), "Short texts: 128 tokens, 16 per call", "embeddings/s", "{:.0f}")
    bars(axes[1], engine_rows("s512b4", "rate"), "Long texts: 512 tokens, 4 per call", "embeddings/s", "{:.0f}")
    bars(axes[2], engine_rows("s512b4", "mac_mJ_per_emb"), "Energy per long-text embedding",
         "millijoules, whole Mac", "{:.0f}", lower_better=True)
    one = [("GPU\nCore AI", load("coreai-gpu", "s128b1")["rate"], GREY),
           ("ANE\nFluidInference", load("fluid", "s128b1")["rate"], LIGHT),
           ("ANE\n(this repo)", load("ane", "s128b1")["rate"], BLUE)]
    bars(axes[3], one, "One text per call: 128 tokens", "embeddings/s", "{:.0f}")
    fig.suptitle("EmbeddingGemma 2 (text) on one 16 GB M6 Mac mini: GPU vs Neural Engine", fontsize=17,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "Each engine alone, 30 s of back-to-back calls after 10 s idle, same token ids; energy = whole "
             "Mac (macmon). github.com/cowboycoderhq/EmbeddingGemma-2-16GB-M6-ANE",
             ha="center", fontsize=10, color="#5f6368")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
