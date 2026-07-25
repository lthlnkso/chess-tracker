"""Figures 5 & 6 — architecture diagrams.

Drawn in the style of the original transformer paper's block diagram, but for
what we actually built. The repeated block is shown once with an "x N layers"
bracket rather than stacked eight times.

    python plots/plot_architecture.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

sys.path.insert(0, os.path.dirname(__file__))
from theme import apply  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SKIN = {
    "light": {
        "norm": "#cfe9d6", "attn": "#c5dcf5", "ffn": "#fbe6bd", "masked": "#f6ccc8",
        "io": "#ffffff", "proj": "#e6e4df", "score": "#d9d2ee",
        "edge": "#3f3f3d", "ink": "#0b0b0b", "muted": "#52514e", "dash": "#8a8880",
    },
    "dark": {
        "norm": "#2f5d40", "attn": "#2a4d70", "ffn": "#6b5326", "masked": "#6d3733",
        "io": "#26262a", "proj": "#3a3a38", "score": "#453c63",
        "edge": "#c9c8c0", "ink": "#ffffff", "muted": "#c3c2b7", "dash": "#7d7b73",
    },
}


def box(ax, cx, cy, w, h, text, fill, s, fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.5",
        linewidth=1.2, edgecolor=s["edge"], facecolor=fill, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=s["ink"], zorder=4, fontweight=weight, linespacing=1.35)
    return cy + h / 2, cy - h / 2


def arrow(ax, x1, y1, x2, y2, s, style="-|>", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
        linewidth=lw, color=s["edge"], zorder=2,
        connectionstyle=f"arc3,rad={rad}"))


def plus(ax, cx, cy, s, r=0.85):
    ax.add_patch(Circle((cx, cy), r, facecolor=s["io"], edgecolor=s["edge"],
                        linewidth=1.2, zorder=4))
    ax.text(cx, cy, "+", ha="center", va="center", fontsize=11,
            color=s["ink"], zorder=5)


def group(ax, x, y, w, h, s, label=None, dashed=True, colour=None):
    ax.add_patch(Rectangle(
        (x, y), w, h, fill=False, linewidth=1.3,
        edgecolor=colour or s["dash"],
        linestyle=(0, (5, 4)) if dashed else "solid", zorder=1))
    if label:
        ax.text(x + w + 0.6, y + h / 2, label, ha="left", va="center",
                fontsize=10, color=s["muted"], fontweight="bold", linespacing=1.3)


def residual(ax, cx, y_from, y_to, s, side=1, off=10.5):
    """Skip connection: out and around the sublayer, into the + node."""
    x = cx + side * off
    arrow(ax, cx, y_from, x, y_from, s, style="-")
    arrow(ax, x, y_from, x, y_to, s, style="-")
    arrow(ax, x, y_to, cx + side * 0.85, y_to, s)


def trunk(ax, cx, y0, s, masked_label, n_layers=8):
    """The shared encoder stack; returns the y of its top output."""
    y = y0
    top, _ = box(ax, cx, y, 20, 3.4, "Board sequence, player's view\n(T x 8 x 8 x 8)", s["io"], s, 10)
    y = top + 3.0
    arrow(ax, cx, top, cx, y - 1.7, s)
    top, _ = box(ax, cx, y, 17, 2.8, "Linear projection  512 -> 256", s["proj"], s, 10)

    y = top + 3.4
    arrow(ax, cx, top, cx, y - 0.85, s)
    plus(ax, cx, y, s)
    ax.text(cx - 6.2, y, "positional\nencoding", ha="right", va="center",
            fontsize=9.5, color=s["muted"], linespacing=1.3)
    arrow(ax, cx - 6.0, y, cx - 0.85, y, s)
    y_after_pos = y

    # ---- repeated block ----
    gx0, gy0 = cx - 13.5, y_after_pos + 2.2
    y = gy0 + 2.6
    arrow(ax, cx, y_after_pos + 0.85, cx, y - 1.4, s)
    top, bot = box(ax, cx, y, 9, 2.4, "Norm", s["norm"], s, 10)
    y = top + 3.0
    arrow(ax, cx, top, cx, y - 2.0, s)
    a_top, a_bot = box(ax, cx, y, 21, 4.0, masked_label, s["masked"], s, 10)
    y_plus1 = a_top + 2.6
    arrow(ax, cx, a_top, cx, y_plus1 - 0.85, s)
    plus(ax, cx, y_plus1, s)
    residual(ax, cx, bot - 1.4, y_plus1, s, side=-1, off=10.5)

    y = y_plus1 + 3.0
    arrow(ax, cx, y_plus1 + 0.85, cx, y - 1.4, s)
    top2, bot2 = box(ax, cx, y, 9, 2.4, "Norm", s["norm"], s, 10)
    y = top2 + 3.2
    arrow(ax, cx, top2, cx, y - 1.8, s)
    f_top, _ = box(ax, cx, y, 18, 3.6, "Feed-Forward Network\n256 -> 1024 -> 256  (GELU)",
                   s["ffn"], s, 9.5)
    y_plus2 = f_top + 2.6
    arrow(ax, cx, f_top, cx, y_plus2 - 0.85, s)
    plus(ax, cx, y_plus2, s)
    residual(ax, cx, y_plus1 + 0.85, y_plus2, s, side=-1, off=10.5)

    gy1 = y_plus2 + 2.2
    group(ax, gx0, gy0, 27, gy1 - gy0, s, label=f"x {n_layers}\nlayers")

    y = gy1 + 3.0
    arrow(ax, cx, y_plus2 + 0.85, cx, y - 1.2, s)
    top, _ = box(ax, cx, y, 9, 2.4, "Norm", s["norm"], s, 10)
    return top


def fig_pretrain(mode: str, outfile: str) -> None:
    s = SKIN[mode]
    p = apply(mode)
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    ax.set_xlim(2, 98)
    ax.set_ylim(1, 69)
    ax.axis("off")

    LX, RX = 30, 74
    top_l = trunk(ax, LX, 6, s, "Masked Multi-Head\nSelf-Attention")
    ax.text(LX, 2.2, "the game so far", ha="center", va="center",
            fontsize=11, color=p["muted"], fontweight="bold")

    # right column: candidate successor encoder
    y = 6
    top_r, _ = box(ax, RX, y, 20, 3.4,
                   "Candidate next position\n(8 x 8 x 8)", s["io"], s, 10)
    ax.text(RX, 2.2, "one legal successor", ha="center", va="center",
            fontsize=11, color=p["muted"], fontweight="bold")
    y = top_r + 4.2
    arrow(ax, RX, top_r, RX, y - 2.6, s)
    top_r, _ = box(ax, RX, y, 21, 5.2,
                   "Board Encoder (MLP)\n512 -> 1024 -> 256 -> 256\n(GELU)", s["proj"], s, 9.5)
    y = top_r + 3.2
    arrow(ax, RX, top_r, RX, y - 1.2, s)
    top_r, _ = box(ax, RX, y, 9, 2.4, "Norm", s["norm"], s, 10)
    ax.text(RX + 13.5, y, "shared across\nall C candidates", ha="left", va="center",
            fontsize=9.5, color=s["muted"], linespacing=1.3)

    # converge
    y_dot = max(top_l, top_r) + 7.5
    box(ax, (LX + RX) / 2, y_dot, 30, 4.2,
        "Scaled dot product   <h, e> / sqrt(256)", s["score"], s, 10.5)
    arrow(ax, LX, top_l, LX, y_dot - 2.1, s, rad=0.0)
    arrow(ax, LX, y_dot - 2.1, (LX + RX) / 2 - 8, y_dot - 2.1, s, style="-")
    arrow(ax, RX, top_r, RX, y_dot - 2.1, s)
    arrow(ax, RX, y_dot - 2.1, (LX + RX) / 2 + 8, y_dot - 2.1, s, style="-")
    ax.text(LX - 2, y_dot - 5.0, "h", ha="right", va="center", fontsize=11,
            color=s["muted"], style="italic")
    ax.text(RX + 2, y_dot - 5.0, "e", ha="left", va="center", fontsize=11,
            color=s["muted"], style="italic")

    cx = (LX + RX) / 2
    y = y_dot + 6.0
    arrow(ax, cx, y_dot + 2.1, cx, y - 2.0, s)
    top, _ = box(ax, cx, y, 32, 4.0,
                 "Softmax over the C legal candidates", s["attn"], s, 10.5)
    y = top + 5.0
    arrow(ax, cx, top, cx, y - 2.0, s)
    box(ax, cx, y, 26, 4.0, "Which position came next?", s["io"], s, 11, weight="bold")

    fig.suptitle("Pre-Training: Scoring Candidate Next Positions",
                 x=0.06, ha="left", fontsize=17, fontweight="bold", y=0.972)
    ax.set_title("7.4M parameters · d_model 256 · 8 heads · causal (each position sees "
                 "only earlier ones)",
                 loc="left", fontsize=11, color=p["muted"], pad=16)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    for ext in ("png", "svg"):
        fig.savefig(f"{outfile}.{ext}", bbox_inches="tight")
    print(f"wrote {outfile}.png / .svg")
    plt.close(fig)


def fig_embed(mode: str, outfile: str) -> None:
    s = SKIN[mode]
    p = apply(mode)
    fig, ax = plt.subplots(figsize=(8.4, 9.4))
    ax.set_xlim(2, 70)
    ax.set_ylim(1, 70)
    ax.axis("off")

    CX = 30
    top = trunk(ax, CX, 6, s, "Masked Multi-Head\nSelf-Attention")
    ax.text(CX, 2.2, "one player's game", ha="center", va="center",
            fontsize=11, color=p["muted"], fontweight="bold")
    ax.text(CX + 15.5, 46, "same trunk as\npre-training,\nweights carried over",
            ha="left", va="center", fontsize=9.5, color=s["muted"],
            fontweight="bold", linespacing=1.4)

    y = top + 5.4
    arrow(ax, CX, top, CX, y - 2.3, s)
    top, _ = box(ax, CX, y, 27, 4.6,
                 "Mean-pool over the plies\nthis player actually moved", s["attn"], s, 10)
    y = top + 5.4
    arrow(ax, CX, top, CX, y - 2.3, s)
    top, _ = box(ax, CX, y, 24, 4.6,
                 "Projection head\n256 -> 256 -> 128  (GELU)", s["ffn"], s, 10)
    y = top + 4.4
    arrow(ax, CX, top, CX, y - 1.4, s)
    top, _ = box(ax, CX, y, 16, 2.8, "L2 normalise", s["norm"], s, 10)
    y = top + 5.0
    arrow(ax, CX, top, CX, y - 2.0, s)
    top, _ = box(ax, CX, y, 26, 4.0, "128-dim player vector", s["io"], s, 11, weight="bold")

    y = top + 5.6
    arrow(ax, CX, top, CX, y - 2.4, s)
    box(ax, CX, y, 34, 4.8,
        "Supervised contrastive loss\n32 players x 4 games per batch", s["score"], s, 10)

    fig.suptitle("Identification: Turning a Game Into a Player Vector",
                 x=0.06, ha="left", fontsize=16.5, fontweight="bold", y=0.972)
    ax.set_title("a player is the centroid of their game vectors",
                 loc="left", fontsize=11, color=p["muted"], pad=16)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    for ext in ("png", "svg"):
        fig.savefig(f"{outfile}.{ext}", bbox_inches="tight")
    print(f"wrote {outfile}.png / .svg")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=("light", "dark", "both"))
    args = ap.parse_args()
    modes = ("light", "dark") if args.mode == "both" else (args.mode,)
    for m in modes:
        sfx = "" if m == "light" else "_dark"
        fig_pretrain(m, os.path.join(HERE, f"05_architecture_pretrain{sfx}"))
        fig_embed(m, os.path.join(HERE, f"06_architecture_embedding{sfx}"))


if __name__ == "__main__":
    main()
