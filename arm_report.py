"""One comparable table across the arms of a sweep.

Works for either sweep shape, because both answer the same question with the
same evidence: an identification eval from finetune_mt.py, optionally paired
with the pre-training history that produced its trunk.

    python arm_report.py --title "contrastive loss" \
        --arm supcon=/workspace/ckpt/ws2_supcon \
        --arm ms=/workspace/ckpt/ws2_ms --out closs.json

    python arm_report.py --title "candidates" \
        --arm 4=/workspace/ckpt/c_c4_id,/workspace/ckpt/c_c4 \
        --arm curr=/workspace/ckpt/c_curr_id,/workspace/ckpt/c_curr --out cand.json

Two guards, because both are ways a sweep quietly lies:
  - unequal step counts across arms at equal wall clock, which turns "better
    objective" into "more training";
  - unequal galleries, which makes recall numbers incomparable outright.
"""

from __future__ import annotations

import argparse
import json
import os


def _tail_mean(curve, key, n=5):
    vals = [c[key] for c in curve[-n:] if key in c and c[key] == c[key]]
    return sum(vals) / len(vals) if vals else None


def load_arm(name, ft_dir, pre_dir):
    out = {"arm": name, "ft_dir": ft_dir, "pre_dir": pre_dir}
    ev = os.path.join(ft_dir, "eval.json")
    if not os.path.exists(ev):
        out["status"] = "no eval.json"
        return out
    with open(ev) as f:
        d = json.load(f)
    r = d.get("result") or {}
    curve = d.get("curve") or []
    out.update({
        "status": "collapsed" if d.get("collapsed") else "ok",
        "loss_name": d.get("loss"), "pk": d.get("pk"),
        "recall@1": r.get("recall@1"), "recall@10": r.get("recall@10"),
        "recall@100": r.get("recall@100"), "median_rank": r.get("median_rank"),
        "gallery": r.get("gallery"), "queries": r.get("queries"),
        "chance@1": r.get("chance@1"), "elo_mae": r.get("elo_mae"),
        "ft_steps": d.get("steps") or (curve[-1]["step"] if curve else None),
        "ft_minutes": d.get("minutes") or (curve[-1].get("minutes") if curve else None),
        "gap": _tail_mean(curve, "gap"), "pos_cos": _tail_mean(curve, "pos_cos"),
        "neg_cos": _tail_mean(curve, "neg_cos"),
        "probes": d.get("probes") or [],
    })
    if out["ft_steps"] and out["ft_minutes"]:
        out["ft_it_s"] = out["ft_steps"] / (out["ft_minutes"] * 60)

    if pre_dir and os.path.exists(os.path.join(pre_dir, "history.json")):
        with open(os.path.join(pre_dir, "history.json")) as f:
            h = json.load(f)
        hist = h.get("history") or []
        out.update({
            "pre_steps": h.get("steps"), "pre_minutes": h.get("minutes"),
            "pre_n_cand": h.get("args", {}).get("n_cand"),
            "pre_curriculum": bool(h.get("args", {}).get("cand_curriculum")),
            "pre_stages": h.get("stages") or [],
            # Validation is pinned to --eval-n-cand for every arm, so this one is
            # comparable even when the arms trained at different difficulties.
            "val_move_acc": hist[-1].get("move_acc") if hist else None,
            "val_elo_mae": hist[-1].get("elo_mae") if hist else None,
        })
        if out.get("pre_steps") and out.get("pre_minutes"):
            out["pre_it_s"] = out["pre_steps"] / (out["pre_minutes"] * 60)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="name=ft_dir[,pre_dir]")
    ap.add_argument("--title", default="sweep")
    ap.add_argument("--baseline", default="", help="arm to report deltas against")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    for spec in args.arm:
        name, _, dirs = spec.partition("=")
        ft, _, pre = dirs.partition(",")
        rows.append(load_arm(name, ft, pre or ""))

    done = [r for r in rows if r.get("recall@1") is not None]
    for r in rows:
        if r not in done:
            print(f"  !! {r['arm']}: {r.get('status', 'missing')}")
    if not done:
        print("no completed arms")
        return

    has_pre = any(r.get("val_move_acc") is not None for r in done)
    print(f"\n=== {args.title} ===")
    head = (f"{'arm':>12}{'recall@1':>10}{'recall@10':>11}{'recall@100':>12}"
            f"{'med rk':>8}{'gap':>8}{'elo mae':>9}{'ft steps':>10}{'it/s':>7}")
    if has_pre:
        head += f"{'val acc':>9}{'pre it/s':>10}"
    print(head)
    for r in sorted(done, key=lambda x: -x["recall@1"]):
        line = (f"{r['arm']:>12}{r['recall@1']:>10.4f}{r['recall@10']:>11.4f}"
                f"{r['recall@100']:>12.4f}{r['median_rank']:>8.0f}"
                f"{(r['gap'] or float('nan')):>8.3f}{(r['elo_mae'] or float('nan')):>9.0f}"
                f"{r['ft_steps']:>10,}{r.get('ft_it_s', 0):>7.1f}")
        if has_pre:
            line += (f"{(r.get('val_move_acc') or float('nan')):>9.4f}"
                     f"{r.get('pre_it_s', float('nan')):>10.2f}")
        print(line + ("   COLLAPSED" if r["status"] == "collapsed" else ""))

    g = {r["gallery"] for r in done}
    print(f"\ngallery {'/'.join(f'{x:,}' for x in sorted(g))} players, "
          f"chance@1 {done[0]['chance@1']:.5f}")
    if len(g) > 1:
        print("CAUTION: arms were evaluated against different gallery sizes; "
              "recall is not comparable across them.")

    base = next((r for r in done if r["arm"] == args.baseline), None)
    if base:
        print(f"\nvs {base['arm']} (recall@1 {base['recall@1']:.4f}):")
        for r in sorted(done, key=lambda x: -x["recall@1"]):
            if r is base:
                continue
            d = r["recall@1"] - base["recall@1"]
            print(f"  {r['arm']:>12} {d:+.4f}  ({100*d/base['recall@1']:+.1f}% relative)")

    steps = [r["ft_steps"] for r in done if r["ft_steps"]]
    if steps and max(steps) / max(min(steps), 1) > 1.15:
        print(f"\nCAUTION: fine-tune steps vary {min(steps):,}-{max(steps):,}. At equal "
              f"wall clock that is a throughput difference, and part of any recall "
              f"gap is training length rather than the arm itself.")

    for r in done:
        if len(r.get("probes") or []) >= 2:
            p = r["probes"]
            print(f"\n{r['arm']} probe curve (gallery {p[0]['gallery']:,}): " +
                  " -> ".join(f"{x['hours']}h {x['recall@1']:.4f}" for x in p))
            lift = p[-1]["recall@1"] - p[-2]["recall@1"]
            print(f"  last interval {lift:+.4f} "
                  f"({'still climbing' if lift > 0.002 else 'flat -- saturated'})")
        if r.get("pre_stages"):
            print(f"\n{r['arm']} curriculum: " + " | ".join(
                f"C{s['n_cand']} {s['to_step']-s['from_step']} steps ({s['why']})"
                for s in r["pre_stages"]))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"title": args.title, "rows": rows}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
