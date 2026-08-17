"""Per-player rating for every name in the gallery.

The gallery stores centroids and names but no rating, so the one strong signal
we hold about a visitor that the embedding does NOT encode -- their estimated
Elo, good to MAE 156 -- has nowhere to be compared against. This builds
username -> (median Elo, games) across every ingested month.

Only meta.npy is needed, never the move or clock streams, so each month is a
775 MB download that is deleted before the next one starts.

    python build_elo_table.py --out play/elo_table.npz
"""

from __future__ import annotations

import argparse
import os

import boto3
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+",
                    default=["2026-01", "2026-02", "2026-03",
                             "2026-04", "2026-05", "2026-06"])
    ap.add_argument("--out", default="play/elo_table.npz")
    ap.add_argument("--tmp", default="/tmp/elo_meta")
    args = ap.parse_args()

    env = dict(l.strip().split("=", 1) for l in open(".env")
               if "=" in l and not l.startswith("#"))
    s3 = boto3.client("s3", endpoint_url="https://s3api-eu-cz-1.runpod.io",
                      aws_access_key_id=env["RUNPOD_S3_ACCESS_KEY"],
                      aws_secret_access_key=env["RUNPOD_S3_SECRET_KEY"],
                      region_name="EU-CZ-1")
    os.makedirs(args.tmp, exist_ok=True)

    # Accumulate sum/count per username rather than every rating: a player with
    # 400 games would otherwise hold 400 ints, and there are 1.28M usernames.
    tot, cnt = {}, {}
    for mth in args.months:
        mp = os.path.join(args.tmp, "meta.npy")
        pp = os.path.join(args.tmp, "players.txt")
        s3.download_file("shusq6ritt", f"data/mt/{mth}/meta.npy", mp)
        s3.download_file("shusq6ritt", f"data/mt/{mth}/players.txt", pp)
        meta = np.load(mp, mmap_mode="r")
        names = open(pp, encoding="utf-8").read().split("\n")
        for side, elo_key in (("white_pid", "white_elo"), ("black_pid", "black_elo")):
            p = np.asarray(meta[side])
            e = np.asarray(meta[elo_key]).astype(np.int64)
            good = e > 0
            p, e = p[good], e[good]
            order = np.argsort(p, kind="stable")
            p, e = p[order], e[order]
            uniq, starts = np.unique(p, return_index=True)
            sums = np.add.reduceat(e, starts)
            lens = np.diff(np.r_[starts, len(e)])
            for u, sm, ln in zip(uniq, sums, lens):
                if u >= len(names):
                    continue
                n = names[u].lower()
                tot[n] = tot.get(n, 0) + int(sm)
                cnt[n] = cnt.get(n, 0) + int(ln)
        print(f"  {mth}: {len(meta):,} games -> {len(tot):,} rated usernames", flush=True)
        os.remove(mp); os.remove(pp)

    names = np.array(sorted(tot), dtype=object)
    elo = np.array([tot[n] / cnt[n] for n in names], dtype=np.float32)
    games = np.array([cnt[n] for n in names], dtype=np.int32)
    np.savez_compressed(args.out, names=names, elo=elo, games=games)
    print(f"\nwrote {args.out}: {len(names):,} players, "
          f"elo mean {elo.mean():.0f} sd {elo.std():.0f}, "
          f"{os.path.getsize(args.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
