"""Repeatable single-process loader throughput, in samples/minute.

The optimisation loop needs a number that moves only when the code does, so:
fixed sample set, fixed seed, several repeats, and the *median* repeat reported
rather than the mean (one GC pause or one background process should not look
like a regression).

    python bench_loader.py --shard data/2026-06-big --cands 16,32 --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from successor_data import MultiTaskDataset


def measure(ds, idxs, repeats):
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in idxs:
            ds[i]
        runs.append(len(idxs) / (time.perf_counter() - t0))
    return statistics.median(runs), min(runs), max(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cands", default="16,32")
    ap.add_argument("--plies", type=int, default=12)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    print(f"{args.label or 'current'}  |  {args.n} samples x {args.repeats} repeats, "
          f"median of repeats\n")
    print(f"{'n_cand':>7}{'samples/s':>12}{'samples/min':>14}{'ms/sample':>12}{'spread':>10}")
    out = {}
    for c in [int(x) for x in args.cands.split(",")]:
        ds = MultiTaskDataset(args.shard, max_len=160, plies_per_game=args.plies,
                              n_cand=c, with_rights=True)
        idxs = np.random.default_rng(0).choice(len(ds), args.n, replace=False)
        for i in idxs[:20]:                      # warm the page cache
            ds[i]
        med, lo, hi = measure(ds, idxs, args.repeats)
        out[c] = med
        print(f"{c:>7}{med:>12.1f}{med*60:>14,.0f}{1000/med:>12.3f}"
              f"{100*(hi-lo)/med:>9.1f}%")
    return out


if __name__ == "__main__":
    main()
