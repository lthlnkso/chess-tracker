"""Figure 7 — Identification accuracy against the player's own rating.

Difference from the mean rather than the level, because the level is not
trustworthy: the games sampled here sit inside the gallery's own January-June
window, so some helped build the very centroids being queried and every absolute
number is inflated. What survives that is the SHAPE across bands, and plotting
each band's distance from the mean is the honest way to show a shape.

The +/- 1 s.e. band is drawn, not just quoted. At 120 players per band a single
bar has a standard error of 4.5 points, so most individual bars say nothing on
their own -- the result is the run of bands above 2100, and a reader can only
see that if the noise floor is on the chart.

    python plots/plot_strength_bands.py --data plots/data/bands_5game.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from theme import apply  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KNEE = 2100


def build(rows: list, mode: str, outfile: str, bundle: int) -> None:
    p = apply(mode)
    n = sum(r["n"] for r in rows)
    mean = sum(r["r10"] * r["n"] for r in rows) / n
    per_band = rows[0]["n"]
    se = math.sqrt(mean * (1 - mean) / per_band) * 100

    diff = np.array([(r["r10"] - mean) * 100 for r in rows])
    lows = np.array([r["lo"] for r in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    # series[0] is the blue, series[1] the orange -- above and below the mean.
    # Direction is carried by position around zero as well as by hue, so the
    # chart still reads without colour.
    colours = [p["series"][0] if d >= 0 else p["series"][1] for d in diff]
    ax.bar(x, diff, 0.68, color=colours, zorder=3)

    ax.axhspan(-se, se, color=p["muted"], alpha=0.13, zorder=1,
               label=f"within one standard error (±{se:.1f} pts)")
    ax.axhline(0, color=p["baseline"], linewidth=1.4, zorder=2)

    knee_x = next((i for i, r in enumerate(rows) if r["lo"] >= KNEE), None)
    if knee_x is not None:
        ax.axvline(knee_x - 0.5, color=p["muted"], linewidth=1.1,
                   linestyle=(0, (3, 3)), zorder=2)
        ax.annotate(f"{KNEE}+", xy=(knee_x - 0.4, ax.get_ylim()[1]),
                    xytext=(3, -12), textcoords="offset points",
                    fontsize=10.5, fontweight="bold", color=p["muted"], va="top")

    for xi, d in zip(x, diff):
        ax.annotate(f"{d:+.1f}", xy=(xi, d),
                    xytext=(0, 4 if d >= 0 else -13), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold",
                    color=p["text"], zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lo}" for lo in lows], fontsize=10, rotation=45,
                       ha="right", color=p["text"])
    ax.set_xlabel("player rating band (lichess bullet, 100-point buckets)")
    ax.set_ylabel(f"points of top-10 recall vs the {mean:.1%} mean")
    ax.set_xlim(-0.8, len(rows) - 0.2)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=10, labelcolor=p["muted"],
              borderaxespad=0.8)

    fig.suptitle("Top 10 Identification Accuracy in "
                 f"{bundle} Games Degrades With Player Strength",
                 x=0.045, ha="left", fontsize=17, fontweight="bold", y=0.98)

    lo_rows = [r for r in rows if r["hi"] <= KNEE]
    hi_rows = [r for r in rows if r["lo"] >= KNEE]
    nl = sum(r["n"] for r in lo_rows); pl = sum(r["r10"] * r["n"] for r in lo_rows) / nl
    nh = sum(r["n"] for r in hi_rows); ph = sum(r["r10"] * r["n"] for r in hi_rows) / nh
    pooled = (pl * nl + ph * nh) / (nl + nh)
    z = (pl - ph) / math.sqrt(pooled * (1 - pooled) * (1 / nl + 1 / nh))
    ax.set_title(
        f"{per_band} sampled players per band · {bundle} real games per query · "
        f"558,735-player gallery\n"
        f"below {KNEE}: {pl:.1%}   ·   {KNEE} and up: {ph:.1%}   ·   "
        f"{(pl-ph)*100:.1f} points, p = {math.erfc(abs(z)/math.sqrt(2)):.1e}",
        loc="left", fontsize=11, color=p["muted"], pad=14)

    fig.text(0.045, -0.03,
             "Measured on players' ordinary games against their own peers, so this is not the bot's doing — "
             "strong players are simply harder to fingerprint.\n"
             "Absolute levels are inflated because these games fall inside the gallery's own window; "
             "the shape across bands is the result, not the height.",
             ha="left", va="top", fontsize=9.5, color=p["muted"])

    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    for ext in ("png", "svg"):
        fig.savefig(f"{outfile}.{ext}", bbox_inches="tight")
    print(f"wrote {outfile}.png / .svg")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "bands_5game.json"))
    ap.add_argument("--bundle", type=int, default=5)
    ap.add_argument("--name", default="07_strength_vs_identification")
    args = ap.parse_args()
    with open(args.data) as f:
        rows = json.load(f)
    for mode in ("light", "dark"):
        suffix = "" if mode == "light" else "_dark"
        build(rows, mode, os.path.join(HERE, args.name + suffix), args.bundle)


if __name__ == "__main__":
    main()
