"""Recover the held-out usernames from a checkpoint's test_pids.

The pids are not a hash and not a per-shard row number: MultiGameDataset walks
each shard's players.txt in the order the shards were given, assigns a new
sequential id to each lowercased username it has not seen, and stores that index
as gpid. Replaying the same walk over the same shards in the same order
reconstructs the table exactly, so a pid becomes a name.

This matters because without it nothing about a trained model can be MEASURED --
only bounded. ctx10_ft.pt's test_pids top out at 1,284,883 and the union over
mt/2026-01..06 is 1,284,947 names, which is the match that confirms the order.

    python recover_heldout.py --ckpt ctx10_ft.pt --shards /workspace/data/mt/2026-0{1,2,3,4,5,6} \
        --out heldout_names.npy
"""
import argparse, os
import numpy as np
import torch


def union_players(shards):
    players, seen = [], set()
    for sp in shards:
        with open(os.path.join(sp, "players.txt"), encoding="utf-8") as f:
            for nm in f.read().split("\n"):
                k = nm.lower()
                if k not in seen:
                    seen.add(k); players.append(nm)
    return players


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    tp = np.asarray(ck["test_pids"])
    players = union_players(a.shards)
    if tp.max() >= len(players):
        raise SystemExit(f"shard list is wrong or out of order: max pid {tp.max():,} "
                         f"but the union holds only {len(players):,} names")
    names = np.array([players[int(i)] for i in tp], dtype=object)
    np.save(a.out, names, allow_pickle=True)
    print(f"{len(names):,} held-out usernames -> {a.out}")


if __name__ == "__main__":
    main()
