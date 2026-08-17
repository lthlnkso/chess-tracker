"""Round 2: where the multipv cost actually goes, and what we can take back.

Round 1 established the shape (depth 8, multipv 32 = 53.8 ms/pos, transport only
4% of it). This measures the three levers that could move it:

  multipv width   MultiPV suppresses alpha-beta cutoffs between root moves, so
                  cost should climb steeply with N. If it does, the candidate
                  count is the expensive knob, not the depth.
  hash reuse      consecutive plies of one game share almost the whole tree.
                  python-chess sends `ucinewgame` per analyse() by default via
                  `game=`; holding the same game token keeps the TT warm.
  threads         24 single-threaded engines vs fewer SMP engines. SMP scales
                  badly for throughput work; one engine per core usually wins.
"""

from __future__ import annotations

import argparse
import time

import chess
import chess.engine
import numpy as np

from profile_cpl import sample_positions
from bitboards import decode_move


def sequential_game(shard, min_plies=30, seed=3):
    """Consecutive positions from ONE game, to test transposition-table reuse."""
    meta = np.load(f"{shard}/meta.npy", mmap_mode="r")
    mv = np.memmap(f"{shard}/moves.u16", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    while True:
        gi = int(rng.integers(0, len(meta)))
        row = meta[gi]; o, k = int(row["offset"]), int(row["nply"])
        if k < min_plies:
            continue
        b = chess.Board(); out = []
        for c in np.asarray(mv[o:o + min_plies]):
            out.append(b.copy())
            b.push(decode_move(int(c)))
        return [x for x in out if not x.is_game_over()]


def timed(eng, boards, limit, multipv, game_token):
    t0 = time.perf_counter()
    for b in boards:
        eng.analyse(b, limit, multipv=multipv, game=game_token)
    return (time.perf_counter() - t0) / len(boards) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--positions", type=int, default=30)
    args = ap.parse_args()

    boards = sample_positions(args.shard, args.positions)
    eng = chess.engine.SimpleEngine.popen_uci(args.engine)
    eng.configure({"Threads": 1, "Hash": 64})
    D = chess.engine.Limit(depth=8)

    print("=== lever 1: multipv width, at depth 8 ===")
    print(f"{'multipv':>8} {'ms/pos':>9} {'vs mpv32':>10}")
    base = None
    for n in (1, 2, 4, 8, 16, 32):
        ms = timed(eng, boards, D, n, object())
        if n == 32:
            base = ms
        print(f"{n:>8} {ms:>9.1f}" + (f" {base/ms:>9.2f}x" if base else ""))
    m32 = timed(eng, boards, D, 32, object())
    print(f"  (multipv 32 re-measured: {m32:.1f} ms)\n")

    print("=== lever 2: transposition-table reuse across a game ===")
    seq = sequential_game(args.shard)[:args.positions]
    fresh = timed(eng, seq, D, 32, None)          # None = new game each call
    tok = object()
    warm = timed(eng, seq, D, 32, tok)            # same token = TT kept warm
    warm2 = timed(eng, seq, D, 32, tok)
    print(f"  consecutive plies, fresh TT each call : {fresh:>7.1f} ms/pos")
    print(f"  consecutive plies, shared game token  : {warm:>7.1f} ms/pos")
    print(f"  same again (fully warm)               : {warm2:>7.1f} ms/pos")
    print(f"  -> reuse is worth {fresh/max(warm2,1e-9):.2f}x on same-game plies\n")
    eng.quit()

    print("=== lever 3: threads per engine (throughput, not latency) ===")
    for th in (1, 2, 4):
        e = chess.engine.SimpleEngine.popen_uci(args.engine)
        e.configure({"Threads": th, "Hash": 64})
        ms = timed(e, boards, D, 32, object())
        e.quit()
        print(f"  Threads={th}: {ms:>7.1f} ms/pos | "
              f"per-core throughput {1000/ms/th:>6.1f} pos/s/core")
    print("PROFILE2_DONE")


if __name__ == "__main__":
    main()
