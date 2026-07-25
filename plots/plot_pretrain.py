"""Figure 1 — Pre-Training: Predicting Successor States.

Cross-entropy loss against training step, for the two board encodings. The
dashed reference is the loss of guessing uniformly among a ply's legal
candidates; without it a falling curve says nothing about whether the model is
any good.

    python plots/plot_pretrain.py            # reads plots/data/*/history.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from theme import apply  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

RUNS = [
    ("succ_p13", "13 planes — with castling + en passant"),
    ("succ_p8", "8 planes — pieces only"),
]


def load(name: str) -> dict:
    with open(os.path.join(HERE, "data", name, "history.json")) as f:
        return json.load(f)


def build(mode: str, outfile: str) -> None:
    p = apply(mode)
    runs = [(label, load(name)) for name, label in RUNS]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    chance = max(r["curve"][-1]["chance"] for _, r in runs)
    y_chance = -math.log(chance)
    ax.axhline(y_chance, color=p["baseline"], lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.annotate(
        f"guessing uniformly among legal moves  ({y_chance:.2f})",
        xy=(0.015, y_chance), xycoords=("axes fraction", "data"),
        xytext=(0, 8), textcoords="offset points",
        ha="left", va="bottom", fontsize=10, color=p["muted"],
    )

    finals = []
    for i, (label, r) in enumerate(runs):
        colour = p["series"][i]
        steps = [c["step"] for c in r["curve"]]
        loss = [c["loss"] for c in r["curve"]]
        ax.plot(steps, loss, color=colour, lw=2.0, zorder=3, label=label)

        vs = [h["step"] for h in r["history"]]
        vl = [h["loss"] for h in r["history"]]
        ax.plot(vs, vl, color=colour, lw=0, marker="o", ms=8,
                mfc=p["surface"], mec=colour, mew=2.0, zorder=4)
        finals.append((vl[-1], vs[-1]))

    # The curves land ~0.02 apart, so a shared anchor would stack the labels.
    # Push the higher one up and the lower one down off their own endpoints.
    order = sorted(range(len(finals)), key=lambda k: finals[k][0])
    dy = {order[0]: -12, order[-1]: 12}
    for i, (fv, fs) in enumerate(finals):
        ax.annotate(
            f"{fv:.3f}", xy=(fs, fv), xytext=(12, dy[i]), textcoords="offset points",
            va="center", ha="left", fontsize=11, color=p["text"],
            fontweight="bold", zorder=5,
            arrowprops=dict(arrowstyle="-", color=p["grid"], lw=1.0,
                            shrinkA=0, shrinkB=4),
        )

    # One legend entry for the validation marker, in ink rather than a series colour.
    ax.plot([], [], lw=0, marker="o", ms=8, mfc=p["surface"],
            mec=p["muted"], mew=2.0, label="held-out validation")

    ax.set_xlabel("training step  (batch of 128 games, 12 supervised plies each)")
    ax.set_ylabel("cross-entropy loss")
    ax.set_xlim(0, max(c["step"] for _, r in runs for c in r["curve"]) * 1.16)
    ax.set_ylim(1.52, y_chance + 0.40)
    ax.grid(axis="y")
    ax.grid(axis="x", alpha=0.45)
    ax.set_axisbelow(True)

    minutes = max(r["minutes"] for _, r in runs)
    params = runs[0][1]["params"] / 1e6
    fig.suptitle("Pre-Training: Predicting Successor States",
                 x=0.055, ha="left", fontsize=17, fontweight="bold", y=0.975)
    ax.set_title(
        f"{params:.1f}M-parameter transformer scoring candidate next positions · "
        f"118,734 lichess games · {minutes:.0f} min on one RTX 4090",
        loc="left", fontsize=11, color=p["muted"], pad=14,
    )
    ax.legend(loc="upper right", fontsize=11, labelcolor=p["muted"],
              handlelength=2.4, borderaxespad=1.0)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "svg"):
        fig.savefig(f"{outfile}.{ext}", bbox_inches="tight")
    print(f"wrote {outfile}.png / .svg")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="01_pretrain_successor_states")
    args = ap.parse_args()
    for mode in ("light", "dark"):
        suffix = "" if mode == "light" else "_dark"
        build(mode, os.path.join(HERE, f"{args.name}{suffix}"))


if __name__ == "__main__":
    main()
