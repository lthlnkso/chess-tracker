"""Shared chart styling for presentation figures.

Two-slot categorical palette (blue, orange), validated for colour-vision
deficiency and contrast against both surfaces. Text never wears a series colour;
identity comes from the mark plus a direct label.
"""

from __future__ import annotations

import matplotlib as mpl

LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e3e2df",
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    "baseline": "#8a8880",
}

DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "muted": "#c3c2b7",
    "grid": "#333331",
    "series": ["#3987e5", "#d95926", "#199e70"],
    "baseline": "#7d7b73",
}


def palette(mode: str) -> dict:
    return LIGHT if mode == "light" else DARK


def apply(mode: str) -> dict:
    p = palette(mode)
    mpl.rcParams.update({
        "figure.facecolor": p["surface"],
        "axes.facecolor": p["surface"],
        "savefig.facecolor": p["surface"],
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "text.color": p["text"],
        "axes.labelcolor": p["muted"],
        "axes.edgecolor": p["grid"],
        "xtick.color": p["muted"],
        "ytick.color": p["muted"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": p["grid"],
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "figure.dpi": 140,
    })
    return p
