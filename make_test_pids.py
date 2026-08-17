"""Recover the held-out player split used by an earlier `identify_eval_mt` run.

The single-game fine-tune never stored which players it held out; both the
fine-tune and the eval re-derived the split from a shared seed. To score that
checkpoint under a *different* eval harness without silently including players
it trained on, the split has to be reproduced exactly.

Reproduction is checked rather than assumed: the same procedure also determines
how many gallery players have enough query games, and the earlier run reported
that number (`matched_players`). If the reconstruction disagrees, the assumed
arguments were wrong and the split must not be used.

    python make_test_pids.py --shard data/mt/2026-01 --out mt_test_pids.npy \
        --expect-matched 10675
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--centroid-frac", type=float, default=0.8)
    ap.add_argument("--eval-players", type=int, default=20000)
    ap.add_argument("--biggest-pool", type=int, default=10)
    ap.add_argument("--expect-matched", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = np.load(os.path.join(args.shard, "meta.npy"), mmap_mode="r")
    clocks = np.memmap(os.path.join(args.shard, "clocks.u16"), dtype=np.uint16, mode="r")
    first = np.asarray(meta["offset"], dtype=np.int64)
    clocked = np.asarray(clocks[first]) != 0xFFFF

    # MultiTaskDataset keeps both seats of every fully-clocked game.
    pid = np.concatenate([np.asarray(meta["white_pid"])[clocked],
                          np.asarray(meta["black_pid"])[clocked]])
    u, c = np.unique(pid, return_counts=True)
    players = u[c >= args.min_games]            # np.unique -> sorted, so order is fixed
    counts = c[c >= args.min_games]
    print(f"{len(players):,} players with >= {args.min_games} clocked game-sides")

    rng = np.random.default_rng(args.seed)
    test_p = rng.choice(players, int(len(players) * args.test_frac), replace=False)
    print(f"held out {len(test_p):,} players ({args.test_frac:.0%})")

    # Mirror the gallery cap and the matched-player rule of identify_eval_mt.
    gal = np.sort(test_p)
    if len(gal) > args.eval_players:
        gal = rng.choice(gal, args.eval_players, replace=False)
    cnt = dict(zip(players.tolist(), counts.tolist()))
    matched = 0
    for p in gal:
        n = cnt[int(p)]
        if n < 3:
            continue
        nc = min(max(1, int(round(args.centroid_frac * n))), n - 1)
        if n - nc >= args.biggest_pool:
            matched += 1
    print(f"gallery {len(gal):,} | matched {matched:,}")

    if args.expect_matched:
        if matched == args.expect_matched:
            print(f"MATCH: reproduces the recorded matched_players="
                  f"{args.expect_matched:,} -- split is trustworthy")
        else:
            raise SystemExit(
                f"MISMATCH: got {matched:,}, recorded {args.expect_matched:,}. "
                "The assumed split arguments are wrong; using this split would "
                "score the baseline on players it trained on.")

    np.save(args.out, np.sort(test_p))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
