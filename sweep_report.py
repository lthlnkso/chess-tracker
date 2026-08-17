"""Summarise an embedding-width sweep into one comparable table.

Recall is the number that decides this, but it is reported alongside the
contrastive diagnostics: a wider embedding that scores the same recall while
separating classes less is a worse bet at the next scale up, and recall alone
would hide that.

    python sweep_report.py --glob '/workspace/ckpt/sweep_d*/eval.json'
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os


def load(path):
    with open(path) as f:
        d = json.load(f)
    r = d.get("result") or {}
    curve = d.get("curve") or []
    tail = curve[-5:] if curve else []
    return {
        "d_embed": d.get("d_embed") or d.get("args", {}).get("d_embed"),
        "recall@1": r.get("recall@1"), "recall@10": r.get("recall@10"),
        "recall@100": r.get("recall@100"), "median_rank": r.get("median_rank"),
        "gallery": r.get("gallery"), "queries": r.get("queries"),
        "chance@1": r.get("chance@1"), "elo_mae": r.get("elo_mae"),
        "steps": curve[-1]["step"] if curve else None,
        "minutes": curve[-1].get("minutes") if curve else None,
        "final_loss": sum(c["loss"] for c in tail) / len(tail) if tail else None,
        "final_gap": sum(c["gap"] for c in tail) / len(tail) if tail else None,
        "pos_cos": tail[-1].get("pos_cos") if tail else None,
        "neg_cos": tail[-1].get("neg_cos") if tail else None,
        "path": path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [load(p) for p in sorted(globmod.glob(args.glob))]
    rows = [r for r in rows if r["recall@1"] is not None]
    rows.sort(key=lambda r: r["d_embed"] or 0)
    if not rows:
        print("no completed arms found")
        return

    print(f"\n{'d_embed':>8}{'recall@1':>10}{'recall@10':>11}{'recall@100':>12}"
          f"{'med rank':>10}{'gap':>8}{'loss':>8}{'steps':>9}{'min':>7}")
    for r in rows:
        print(f"{r['d_embed']:>8}{r['recall@1']:>10.4f}{r['recall@10']:>11.4f}"
              f"{r['recall@100']:>12.4f}{r['median_rank']:>10.0f}"
              f"{r['final_gap']:>8.3f}{r['final_loss']:>8.3f}"
              f"{r['steps']:>9,}{r['minutes']:>7.0f}")

    g = rows[0]["gallery"]
    print(f"\ngallery {g:,} players, chance@1 {rows[0]['chance@1']:.5f}"
          f"  ({rows[0]['queries']:,} queries)")

    best = max(rows, key=lambda r: r["recall@1"])
    base = next((r for r in rows if r["d_embed"] == 128), None)
    print(f"best recall@1: d_embed {best['d_embed']} at {best['recall@1']:.4f}")
    if base and base is not best:
        d = best["recall@1"] - base["recall@1"]
        print(f"  vs d_embed 128: {d:+.4f} ({100*d/base['recall@1']:+.1f}% relative)")

    # Steps differ if a wider model runs slower in the same wall clock, which
    # would confound "wider is better" with "trained longer".
    steps = [r["steps"] for r in rows]
    if steps and max(steps) / max(min(steps), 1) > 1.15:
        print(f"\nCAUTION: step counts vary {min(steps):,}-{max(steps):,} across arms "
              f"(equal wall clock, not equal steps) -- part of any gap is training "
              f"length, not width.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
