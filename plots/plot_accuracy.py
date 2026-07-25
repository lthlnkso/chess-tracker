"""Figure 3 — How often the model picks the state that actually happened.

Accuracy on held-out games, assessed periodically through training. The dashed
line is what guessing among the same candidate set would score, so the gap is
the signal.

Note the scale: the periodic assessment scores against 16 sampled legal
candidates (what training sees). The full-legal-move number is stricter and is
annotated separately where available.

    python plots/plot_accuracy.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from theme import apply  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RUNS = [
    ("succ_p13", "13 planes — with castling + en passant", 0.370),
    ("succ_p8", "8 planes — pieces only", 0.368),
]


def load(name: str) -> dict:
    with open(os.path.join(HERE, "data", name, "history.json")) as f:
        return json.load(f)


def build(runs, mode: str, outfile: str, xkey: str = "step") -> None:
    p = apply(mode)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    chance = runs[0][1]["history"][-1]["chance"]
    ax.axhline(chance, color=p["baseline"], lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.annotate(
        f"guessing among the same candidates  ({chance:.1%})",
        xy=(0.015, chance), xycoords=("axes fraction", "data"),
        xytext=(0, 8), textcoords="offset points",
        ha="left", va="bottom", fontsize=10, color=p["muted"],
    )

    finals = []
    for i, (label, r, full_legal) in enumerate(runs):
        colour = p["series"][i]
        xs = [h[xkey] for h in r["history"]]
        ys = [h["acc"] for h in r["history"]]
        ax.plot(xs, ys, color=colour, lw=2.0, marker="o", ms=8, mfc=p["surface"],
                mec=colour, mew=2.0, zorder=3, label=label)
        finals.append((ys[-1], xs[-1], full_legal))

    order = sorted(range(len(finals)), key=lambda k: finals[k][0])
    dy = {order[0]: -13, order[-1]: 13} if len(finals) > 1 else {0: 0}
    for i, (fv, fx, full_legal) in enumerate(finals):
        txt = f"{fv:.1%}"
        if full_legal:
            txt += f"\n{full_legal:.1%} vs all legal"
        ax.annotate(txt, xy=(fx, fv), xytext=(12, dy[i]), textcoords="offset points",
                    va="center", ha="left", fontsize=10.5, color=p["text"],
                    fontweight="bold", zorder=5,
                    arrowprops=dict(arrowstyle="-", color=p["grid"], lw=1.0,
                                    shrinkA=0, shrinkB=4))

    ax.set_xlabel("training step" if xkey == "step" else "wall-clock minutes")
    ax.set_ylabel("next-state accuracy on held-out games")
    ax.set_ylim(0, max(f[0] for f in finals) * 1.55)
    ax.set_xlim(0, max(h[xkey] for _, r, _ in runs for h in r["history"]) * 1.22)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y")
    ax.grid(axis="x", alpha=0.45)
    ax.set_axisbelow(True)

    fig.suptitle("Predicting the Next Board State", x=0.055, ha="left",
                 fontsize=17, fontweight="bold", y=0.975)
    ax.set_title(
        f"held-out accuracy, assessed every {runs[0][1]['history'][0]['step']:,} steps · "
        f"{runs[0][1]['params_m'] if 'params_m' in runs[0][1] else runs[0][1]['params']/1e6:.1f}M "
        f"parameters · 118,734 lichess games",
        loc="left", fontsize=11, color=p["muted"], pad=14,
    )
    ax.legend(loc="upper left", fontsize=11, labelcolor=p["muted"],
              handlelength=2.4, borderaxespad=0.8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "svg"):
        fig.savefig(f"{outfile}.{ext}", bbox_inches="tight")
    print(f"wrote {outfile}.png / .svg")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="03_next_state_accuracy")
    ap.add_argument("--run", action="append", default=None,
                    help="dir:label[:full_legal_acc] — repeatable; overrides defaults")
    ap.add_argument("--x", default="step", choices=["step", "minutes"])
    args = ap.parse_args()

    if args.run:
        spec = []
        for s in args.run:
            parts = s.split(":")
            spec.append((parts[1], load(parts[0]),
                         float(parts[2]) if len(parts) > 2 else None))
    else:
        spec = [(label, load(d), fl) for d, label, fl in DEFAULT_RUNS]

    for mode in ("light", "dark"):
        suffix = "" if mode == "light" else "_dark"
        build(spec, mode, os.path.join(HERE, f"{args.name}{suffix}"), args.x)


if __name__ == "__main__":
    main()
