"""Make the README chart from results/m6-release/*.json (the same style as the Bonsai and EmbeddingGemma-300M charts).
  python scripts/chart.py docs/embeddinggemma-2-ane-vs-gpu.png"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results" / "m6-release"
OTHERS = R.parent / "m6-others"
GREY, LIGHT, PALE, BLUE = "#9aa0a6", "#c7cbd1", "#e3e5e8", "#1a73e8"
GPU = [("llamacpp-q8", "GPU\nllama.cpp"), ("mlx-bf16", "GPU\nMLX"), ("coreai-gpu", "GPU\nCore AI")]


def load(prefix, shape):
    p = R / f"{prefix}-{shape}.json"
    return json.loads(p.read_text()) if p.exists() else None


def other(name):
    return json.loads((OTHERS / f"{name}.json").read_text())


def bars(ax, rows, title, ylabel, fmt, lower_better=False):
    """rows: (label, value, colour[, hatch]). A hatch marks a bar measured at a different text length."""
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    b = ax.bar(range(len(vals)), vals, color=[r[2] for r in rows], width=0.66,
               hatch=[r[3] if len(r) > 3 else None for r in rows], edgecolor="#80868b", linewidth=0)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(v), ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(vals)), labels, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel + (" (lower is better)" if lower_better else ""), fontsize=10)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)


def engine_rows(shape, key):
    return [(label, load(p, shape)[key], GREY) for p, label in GPU if load(p, shape)]


ANE = "ANE\nthis repo"
FLUID = "ANE\nFluid-\nInference"
ANEMLL = "ANE\nAnemll"
ANEMLL_320 = "ANE\nAnemll\n(320 tok)"


def main():
    out = Path(sys.argv[1])
    fluid128, fluid512 = load("fluid", "s128b1"), load("fluid", "s512b1")
    anemll128, anemll320 = other("anemll-s320-128tok"), other("anemll-s320-320tok")
    fig, axes = plt.subplots(1, 4, figsize=(25, 6.8))

    short = engine_rows("s128b16", "rate") + [
        (ANEMLL, anemll128["rate"], PALE), (FLUID, fluid128["rate"], LIGHT), (ANE, load("ane", "s128b16")["rate"], BLUE)]
    bars(axes[0], short, "Short texts: 128 tokens (this repo: 16 per call)", "embeddings/s", "{:.0f}")

    long = engine_rows("s512b4", "rate") + [
        (ANEMLL_320, anemll320["rate"], PALE, "//"), (FLUID, fluid512["rate"], LIGHT),
        (ANE, load("ane", "s512b4")["rate"], BLUE)]
    bars(axes[1], long, "Long texts: 512 tokens (this repo: 4 per call)", "embeddings/s", "{:.0f}")

    energy = engine_rows("s512b4", "mac_mJ_per_emb") + [
        (ANEMLL_320, anemll320["mac_mJ_per_emb"], PALE, "//"), (FLUID, fluid512["mac_mJ_per_emb"], LIGHT),
        (ANE, load("ane", "s512b4")["mac_mJ_per_emb"], BLUE)]
    bars(axes[2], energy, "Energy per long-text embedding", "millijoules, whole Mac", "{:.0f}", lower_better=True)

    one = [("GPU\nCore AI", load("coreai-gpu", "s128b1")["rate"], GREY),
           (ANEMLL, anemll128["rate"], PALE), (FLUID, fluid128["rate"], LIGHT),
           (ANE, load("ane", "s128b1")["rate"], BLUE)]
    bars(axes[3], one, "One text per call: 128 tokens", "embeddings/s", "{:.0f}")

    fig.suptitle("EmbeddingGemma 2 (text) on one 16 GB M6 Mac mini: GPU vs Neural Engine", fontsize=17,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.035, "Each engine alone, 30 s of back-to-back calls after 10 s idle, same token ids; energy = whole "
             "Mac (macmon). GPU paths: the same batch as this repo. FluidInference and Anemll: 1 text per call (4 parallel calls "
             "give the same speed; so does FluidInference's packed function).", ha="center", fontsize=10, color="#5f6368")
    fig.text(0.5, 0.01, "Anemll has one fixed 320-token shape: hatched bars = 320-token texts (it has no 512-token shape). github.com/cowboycoderhq/EmbeddingGemma-2-16GB-M6-ANE",
             ha="center", fontsize=10, color="#5f6368")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
