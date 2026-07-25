"""Build a filtered training index across per-month shards.

Does NOT copy move data. Usernames are per-shard ids, so this maps them to a
global id space, counts each player's games across every month, and writes an
index of the (shard, game, seat) triples worth training on. The shards stay
where they are and get memory-mapped.

    python combine.py --shards /workspace/data/2026-* --out /workspace/data/combined \
        --min-games 100
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

INDEX_DTYPE = np.dtype([
    ("shard", np.uint8),
    ("game", np.uint32),
    ("seat", np.uint8),      # 0 = White, 1 = Black
    ("gpid", np.uint32),     # global player id
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-games", type=int, default=100)
    args = ap.parse_args()

    shards = sorted(args.shards)
    if len(shards) > 255:
        raise SystemExit("shard id is uint8; too many shards")
    os.makedirs(args.out, exist_ok=True)

    names: dict[str, int] = {}
    per_shard = []

    for si, sh in enumerate(shards):
        meta = np.load(os.path.join(sh, "meta.npy"), mmap_mode="r")
        with open(os.path.join(sh, "players.txt"), encoding="utf-8") as f:
            local = f.read().split("\n")
        # local pid -> global pid
        remap = np.empty(len(local), dtype=np.uint32)
        for i, nm in enumerate(local):
            g = names.get(nm)
            if g is None:
                g = len(names)
                names[nm] = g
            remap[i] = g
        w = remap[np.asarray(meta["white_pid"])]
        b = remap[np.asarray(meta["black_pid"])]
        per_shard.append((si, w, b))
        print(f"  {os.path.basename(sh):<12} {len(meta):>12,} games  "
              f"{len(local):>10,} players  (global vocab {len(names):,})", flush=True)

    counts = np.zeros(len(names), dtype=np.int64)
    for _, w, b in per_shard:
        counts += np.bincount(w, minlength=len(names))
        counts += np.bincount(b, minlength=len(names))

    keep = counts >= args.min_games
    print(f"\nplayers: {len(names):,} total, {int(keep.sum()):,} with >= {args.min_games} games")

    parts = []
    for si, w, b in per_shard:
        for seat, pid in ((0, w), (1, b)):
            sel = np.flatnonzero(keep[pid])
            if not len(sel):
                continue
            arr = np.empty(len(sel), dtype=INDEX_DTYPE)
            arr["shard"] = si
            arr["game"] = sel.astype(np.uint32)
            arr["seat"] = seat
            arr["gpid"] = pid[sel]
            parts.append(arr)

    index = np.concatenate(parts)
    # Compact the surviving players into a dense id space so an embedding table
    # over them is sized by the players we keep, not the millions we discard.
    kept_ids = np.flatnonzero(keep)
    dense = np.full(len(names), np.iinfo(np.uint32).max, dtype=np.uint32)
    dense[kept_ids] = np.arange(len(kept_ids), dtype=np.uint32)
    index["gpid"] = dense[index["gpid"]]

    np.save(os.path.join(args.out, "index.npy"), index)

    inv = [None] * len(names)
    for nm, g in names.items():
        inv[g] = nm
    with open(os.path.join(args.out, "players.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(inv[g] for g in kept_ids))

    with open(os.path.join(args.out, "shards.json"), "w") as f:
        json.dump({"shards": [os.path.abspath(s) for s in shards],
                   "min_games": args.min_games,
                   "game_sides": int(len(index)),
                   "players": int(keep.sum())}, f, indent=2)

    gs = len(index)
    print(f"game-sides kept: {gs:,}")
    print(f"median games/kept player: {int(np.median(counts[keep]))}")
    print(f"wrote {args.out}/index.npy ({index.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
