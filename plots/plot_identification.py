"""Figure 4 — Identifying a player from their games.

Recall against a gallery of held-out player centroids, as a function of how many
of that player's games the query pools together. One bullet game is thin
evidence; the interesting question is how fast the curve climbs.

Every bar carries a visible value label -- required, because the aqua series
sits below 3:1 contrast on the light surface.

    python plots/plot_identification.py --data plots/data/eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from theme import apply  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KS = (1, 10, 100)
ORDER = ["1", "10", "half", "all"]
LABELS = {"1": "1 game", "10": "10 games", "half": "half their games",
          "all": "all their games"}


def build(payload: dict, mode: str, outfile: str) -> None:
    p = apply(mode)
    res = payload["results"]
    keys = [k for k in ORDER if k in res]
    x = np.arange(len(keys))
    w = 0.26

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    chance = payload["chance_recall@1"]

    for i, k in enumerate(KS):
        vals = [res[q][f"recall@{k}"] for q in keys]
        bars = ax.bar(x + (i - 1) * (w + 0.015), vals, w, color=p["series"][i],
                      label=f"true player in top {k}", zorder=3)
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.1%}" if v >= 0.001 else f"{v:.2%}",
                        xy=(rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10,
                        fontweight="bold", color=p["text"], zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(k, k) for k in keys], fontsize=12, color=p["text"])
    ax.set_xlabel("evidence pooled into the query")
    ax.set_ylabel("recall against held-out player centroids")
    top = max(res[q][f"recall@{k}"] for q in keys for k in KS)
    # Headroom above the tallest bar for its label; ticks still stop at 100%.
    ax.set_ylim(0, max(top * 1.15, 0.02))
    ax.set_yticks([t for t in np.arange(0, 1.01, 0.2) if t <= top * 1.15 + 0.02])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    fig.suptitle("Identifying a Player From Their Games", x=0.055, ha="left",
                 fontsize=17, fontweight="bold", y=0.975)
    ax.set_title(
        f"{payload['n_gallery_players']:,} held-out players never seen in training · "
        f"centroids built from {payload['centroid_frac']:.0%} of each player's games",
        loc="left", fontsize=11, color=p["muted"], pad=14)
    ax.legend(loc="upper left", fontsize=11, labelcolor=p["muted"], borderaxespad=0.8)

    fig.text(0.055, -0.02,
             f"Chance for top-1 is 1 in {payload['n_gallery_players']:,} "
             f"({chance:.3%}) — too small to draw, so it is stated rather than plotted. "
             f"{payload['n_test_gamesides']:,} game-sides embedded.",
             ha="left", va="top", fontsize=9.5, color=p["muted"])

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    for ext in ("png", "svg"):
        fig.savefig(f"{outfile}.{ext}", bbox_inches="tight")
    print(f"wrote {outfile}.png / .svg")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "eval.json"))
    ap.add_argument("--name", default="04_player_identification")
    args = ap.parse_args()
    with open(args.data) as f:
        payload = json.load(f)
    for mode in ("light", "dark"):
        suffix = "" if mode == "light" else "_dark"
        build(payload, mode, os.path.join(HERE, f"{args.name}{suffix}"))


if __name__ == "__main__":
    main()
