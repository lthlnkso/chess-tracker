"""Figure 2 — Estimated Pre-Training Cost by GPU.

Cost to push one million game-sides through pre-training, which is the number
that decides which card to rent. Two bars per GPU:

  measured    what the run actually costs, dataloader included
  GPU-limited what it would cost if candidate generation never made the GPU wait

A large gap means the run is CPU-bound and the faster card is being wasted.

    python plots/plot_gpu_cost.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from theme import apply  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["RTX 3090", "RTX 4090", "RTX 5090"]


def load(datadir: str) -> list[dict]:
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(datadir, "*.json"))]
    runs.sort(key=lambda r: ORDER.index(r["gpu"]) if r["gpu"] in ORDER else 99)
    return runs


def build(runs: list[dict], mode: str, outfile: str) -> None:
    p = apply(mode)
    labels = [r["gpu"] for r in runs]
    measured = [r["usd_per_m_gamesides_e2e"] for r in runs]
    gpulim = [r["usd_per_m_gamesides_gpu"] for r in runs]

    x = np.arange(len(runs))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    b1 = ax.bar(x - w / 2 - 0.01, measured, w, color=p["series"][0],
                label="measured (dataloader included)", zorder=3)
    b2 = ax.bar(x + w / 2 + 0.01, gpulim, w, color=p["series"][1],
                label="GPU-limited (batch already resident)", zorder=3)

    for bars, vals in ((b1, measured), (b2, gpulim)):
        for rect, v in zip(bars, vals):
            ax.annotate(f"${v:.2f}", xy=(rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11,
                        fontweight="bold", color=p["text"], zorder=4)

    sub = []
    for r in runs:
        sub.append(f"{r['gpu'].replace('RTX ','')}: ${r['price_per_hr']:.2f}/hr · "
                   f"{r['end2end_gamesides_s']:,.0f} game-sides/s · "
                   f"{r['vcpu']} vCPU")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n${r['price_per_hr']:.2f}/hr" for l, r in zip(labels, runs)],
                       fontsize=12, color=p["text"])
    ax.set_ylabel("USD per million game-sides")
    ax.set_ylim(0, max(measured) * 1.26)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    best = min(range(len(runs)), key=lambda i: measured[i])
    fig.suptitle("Estimated Pre-Training Cost by GPU",
                 x=0.055, ha="left", fontsize=17, fontweight="bold", y=0.975)
    ax.set_title(
        f"{runs[0]['params_m']:.1f}M-param successor scorer, batch {runs[0]['batch']} · "
        f"lower is better · cheapest measured: {labels[best]}",
        loc="left", fontsize=11, color=p["muted"], pad=14,
    )
    ax.legend(loc="upper left", fontsize=11, labelcolor=p["muted"], borderaxespad=0.8)

    gap = max(r["dataloader_bound_pct"] for r in runs)
    fig.text(0.055, -0.02,
             f"Prices are what RunPod actually charged at run time; the 3090 was only "
             f"available on community cloud, the 4090/5090 on secure.\n"
             f"Worst dataloader shortfall {gap:.0f}% — where the two bars diverge, "
             f"candidate generation on CPU, not the GPU, is the limit.",
             ha="left", va="top", fontsize=9.5, color=p["muted"])

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    for ext in ("png", "svg"):
        fig.savefig(f"{outfile}.{ext}", bbox_inches="tight")
    print(f"wrote {outfile}.png / .svg")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "bench"))
    ap.add_argument("--name", default="02_pretrain_cost_by_gpu")
    args = ap.parse_args()
    runs = load(args.data)
    if not runs:
        sys.exit(f"no benchmark JSON in {args.data}")
    print(f"{len(runs)} GPUs: {[r['gpu'] for r in runs]}")
    for mode in ("light", "dark"):
        suffix = "" if mode == "light" else "_dark"
        build(runs, mode, os.path.join(HERE, f"{args.name}{suffix}"))


if __name__ == "__main__":
    main()
