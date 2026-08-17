"""How fast can we label positions with per-move centipawn loss?

The CPL loss needs, for every supervised ply, the eval of each candidate move.
Doing that as one search per candidate is 6.2G searches on a full shard. One
`go depth D multipv N` returns all N at once from a single shared tree, which is
the only version of this that can exist.

So the question this answers is: what does one multipv search cost, and where is
the knee? Two things get measured separately because they are confusable --

  engine time     the search itself, which depth controls
  UCI overhead    per-call round-trip through python-chess, which is FIXED and
                  starts to dominate at shallow depths. If it does, the fix is a
                  different transport, not a different depth.

    python profile_cpl.py --positions 60
"""

from __future__ import annotations

import argparse
import time

import chess
import chess.engine
import numpy as np

from bitboards import decode_move


def sample_positions(shard, n, seed=0, skip_opening=8):
    """Real mid-game positions, not the opening book Stockfish answers instantly."""
    meta = np.load(f"{shard}/meta.npy", mmap_mode="r")
    mv = np.memmap(f"{shard}/moves.u16", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        gi = int(rng.integers(0, len(meta)))
        row = meta[gi]
        o, k = int(row["offset"]), int(row["nply"])
        if k < skip_opening + 4:
            continue
        t = int(rng.integers(skip_opening, k))
        b = chess.Board()
        for c in np.asarray(mv[o:o + t]):
            b.push(decode_move(int(c)))
        if b.is_game_over():
            continue
        out.append(b.copy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--multipv", type=int, default=32)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    boards = sample_positions(args.shard, args.positions)
    legal = np.array([b.legal_moves.count() for b in boards])
    print(f"{len(boards)} real mid-game positions | legal moves: "
          f"mean {legal.mean():.1f}  median {np.median(legal):.0f}  max {legal.max()}")
    print(f"positions with >= {args.multipv} legal moves: "
          f"{100*(legal >= args.multipv).mean():.0f}%\n")

    eng = chess.engine.SimpleEngine.popen_uci(args.engine)
    eng.configure({"Threads": args.threads, "Hash": 64})

    # Baseline the transport itself. If a depth-1 multipv search costs about the
    # same as a depth-8 one, we are paying for python-chess round-trips rather
    # than for search, and deepening is free until that changes.
    print(f"{'limit':>14} {'ms/pos':>9} {'pos/s':>9} {'pos/s x24':>11} {'note':>26}")
    print("-" * 74)
    results = {}
    for label, limit in (("depth 1", chess.engine.Limit(depth=1)),
                         ("depth 4", chess.engine.Limit(depth=4)),
                         ("depth 6", chess.engine.Limit(depth=6)),
                         ("depth 8", chess.engine.Limit(depth=8)),
                         ("depth 10", chess.engine.Limit(depth=10)),
                         ("depth 12", chess.engine.Limit(depth=12)),
                         ("nodes 10k", chess.engine.Limit(nodes=10_000)),
                         ("nodes 50k", chess.engine.Limit(nodes=50_000))):
        t0 = time.perf_counter()
        for b in boards:
            eng.analyse(b, limit, multipv=args.multipv)
        dt = time.perf_counter() - t0
        ms = dt / len(boards) * 1000
        results[label] = ms
        print(f"{label:>14} {ms:>9.1f} {1000/ms:>9.1f} {24000/ms:>11.0f}")
    eng.quit()

    base = results["depth 1"]
    print(f"\ndepth-1 floor is {base:.1f} ms/pos -- that is transport, not search.")
    for k in ("depth 8", "depth 10", "depth 12"):
        print(f"  {k}: {results[k]:.1f} ms, of which ~{base:.1f} is overhead "
              f"({100*base/results[k]:.0f}%)")

    print("\n--- what that buys, at 24 parallel engines ---")
    for k in ("depth 6", "depth 8", "depth 10"):
        pps = 24000 / results[k]
        for n, tag in ((5_000_000, "5M positions"), (194_000_000, "full shard 8/game")):
            h = n / pps / 3600
            print(f"  {k:>9} {tag:<22} {h:>7.1f} GPU-free CPU-hours"
                  f"  (${h*0.27:>6.2f} at pod rates)")
    print("PROFILE_DONE")


if __name__ == "__main__":
    main()
