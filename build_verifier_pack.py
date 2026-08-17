"""Ship the gallery's games, small enough to sit next to the server.

The verifier reads a CANDIDATE'S actual games, but the gallery spans six months
of shards (~39 GB) and the demo machine holds one partial month -- 8.1% of the
roster. Without this, 92% of any shortlist is unscoreable.

Two measurements make a local pack possible:

  the verifier ignores most of a game   AUC is flat from 160 plies down to 40
                                        (0.8297 vs 0.8270), so 60 plies is all
                                        that needs shipping
  it only needs a few games per player  evidence compounds across a candidate's
                                        games, but 4 is already most of it

4 games x 60 plies x (2B move + 2B clock) is under 500 bytes per player, so the
whole 558,735-player roster fits in ~270 MB instead of 39 GB.

Run on a pod (fast S3, disk to spare); download the single output file.

    python build_verifier_pack.py --gallery play/gallery_2026.npz \
        --shards /data/2026-01 ... --out /data/verifier_pack.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=4, help="games kept per player")
    ap.add_argument("--plies", type=int, default=60, help="plies kept per game")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = np.load(args.gallery, allow_pickle=True)
    want = {str(n).lower(): i for i, n in enumerate(g["names"])}
    P = len(want)
    print(f"gallery {P:,} players | keeping {args.games} games x {args.plies} plies",
          flush=True)

    L = args.plies
    moves = np.zeros((P, args.games, L), np.uint16)
    clocks = np.full((P, args.games, L), 0xFFFF, np.uint16)
    nply = np.zeros((P, args.games), np.uint16)
    seat = np.zeros((P, args.games), np.uint8)
    tcb = np.zeros((P, args.games), np.uint16)
    tci = np.zeros((P, args.games), np.uint16)
    have = np.zeros(P, np.uint8)

    rng = np.random.default_rng(args.seed)
    for sh in args.shards:
        meta = np.load(os.path.join(sh, "meta.npy"), mmap_mode="r")
        mv = np.memmap(os.path.join(sh, "moves.u16"), dtype=np.uint16, mode="r")
        ck = np.memmap(os.path.join(sh, "clocks.u16"), dtype=np.uint16, mode="r")
        with open(os.path.join(sh, "players.txt"), encoding="utf-8") as f:
            names = f.read().split("\n")
        # Map this shard's local pid -> gallery row, once. pid is per shard, so
        # this is the only safe join.
        p2row = np.full(len(names), -1, np.int64)
        for pid, nm in enumerate(names):
            r = want.get(nm.lower())
            if r is not None:
                p2row[pid] = r

        off = np.asarray(meta["offset"], np.int64)
        npl = np.asarray(meta["nply"], np.int64)
        clocked = np.asarray(ck[off]) != 0xFFFF
        added = 0
        for side, key in ((0, "white_pid"), (1, "black_pid")):
            pids = np.asarray(meta[key])
            rows = p2row[pids]
            cand = np.flatnonzero((rows >= 0) & clocked & (npl >= 10))
            rng.shuffle(cand)
            for gi in cand:
                r = rows[gi]
                h = have[r]
                if h >= args.games:
                    continue
                o, n = int(off[gi]), min(int(npl[gi]), L)
                moves[r, h, :n] = np.asarray(mv[o:o + n])
                clocks[r, h, :n] = np.asarray(ck[o:o + n])
                nply[r, h] = n
                seat[r, h] = side
                tcb[r, h] = int(meta[gi]["tc_base"])
                tci[r, h] = int(meta[gi]["tc_inc"])
                have[r] = h + 1
                added += 1
        print(f"  {os.path.basename(sh)}: +{added:,} games | "
              f"{int((have >= args.games).sum()):,}/{P:,} players full", flush=True)

    np.savez_compressed(args.out, moves=moves, clocks=clocks, nply=nply,
                        seat=seat, tc_base=tcb, tc_inc=tci, have=have,
                        names=g["names"], plies=L, games=args.games)
    mb = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {args.out}: {mb:.0f} MB | "
          f"{int((have > 0).sum()):,}/{P:,} players have at least one game")
    print("PACK_DONE")


if __name__ == "__main__":
    main()
