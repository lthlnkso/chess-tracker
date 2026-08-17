"""Where does a training step actually spend its time?

The claim that this pipeline is dataloader-bound was inferred from the fact that
four concurrent jobs on one GPU produced no more aggregate throughput than one.
That is suggestive, not a measurement. This measures it directly:

  1. wall-clock per sample from MultiTaskDataset, at several candidate counts
  2. a cProfile breakdown of where that time goes
  3. a synthetic split of the two things _build does -- replaying the game to get
     the position sequence, versus generating and encoding candidate successors
  4. the implied samples/s for W workers, compared against the it/s actually
     observed on the training pods

    python profile_loader.py --shard data/2026-06-big --n 200 --workers 24
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np
import chess

from successor_data import MultiTaskDataset
from bitboards import board_to_planes8, decode_move
from fastboard import N_BB, snapshot, encode_batch

# Measured on the training pods, batch 128.
OBSERVED = {"pre-train (24 workers)": 11.1, "fine-tune (24 workers)": 16.7}
BATCH = 128


def time_samples(ds, idxs):
    t0 = time.perf_counter()
    for i in idxs:
        ds[i]
    return (time.perf_counter() - t0) / len(idxs)


def split_cost(ds, idxs, n_cand):
    """Replay-only cost vs replay+candidates, by re-running _build's two halves.

    Done by hand rather than by instrumenting _build so the measurement cannot
    perturb the thing it measures.
    """
    replay = cand = 0.0
    for i in idxs:
        gi, seat = (int(x) for x in ds.index[i])
        row = ds.meta[gi]
        o, n = int(row["offset"]), int(row["nply"])
        codes = np.asarray(ds.moves[o:o + n])
        pov = chess.WHITE if seat == 0 else chess.BLACK
        T = min(len(codes), ds.max_len)
        buf = np.zeros((ds.n_planes, 8, 8), dtype=np.uint8)

        # half 1: replay the game, snapshotting the position before each ply
        snaps = np.zeros((T + ds.plies_per_game * n_cand, N_BB), dtype=np.uint64)
        t0 = time.perf_counter()
        board = chess.Board()
        for t in range(T):
            snapshot(board, pov, snaps[t])
            board.push(decode_move(int(codes[t])))
        replay += time.perf_counter() - t0

        # half 2: at plies_per_game sampled plies, generate the legal successors
        # and encode each one
        rng = np.random.default_rng(0)
        chosen = np.sort(rng.choice(np.arange(T), min(ds.plies_per_game, T), replace=False))
        t0 = time.perf_counter()
        board = chess.Board()
        k = 0
        for t in range(T):
            if k < len(chosen) and chosen[k] == t:
                true_move = decode_move(int(codes[t]))
                others = [m for m in board.legal_moves if m != true_move]
                if len(others) > n_cand - 1:
                    others = others[:n_cand - 1]
                base = T + k * n_cand
                for j, pick in enumerate([true_move] + others):
                    board.push(pick)
                    snapshot(board, pov, snaps[base + j])
                    board.pop()
                k += 1
            board.push(decode_move(int(codes[t])))
        encode_batch(snaps, ds.n_planes,
                     np.full((len(snaps), 5), -1, np.int16) if ds.with_rights else None)
        cand += time.perf_counter() - t0
    n = len(idxs)
    return replay / n, cand / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--cands", default="4,16,32,64")
    ap.add_argument("--plies", type=int, default=12)
    args = ap.parse_args()

    cands = [int(c) for c in args.cands.split(",")]
    rng = np.random.default_rng(0)

    print(f"shard {args.shard} | {args.n} samples/point | {args.plies} plies/game\n")
    print(f"{'n_cand':>7}{'ms/sample':>12}{'replay ms':>12}{'cand ms':>10}"
          f"{'cand %':>9}{'samples/s':>12}{'it/s @128':>11}")
    base = None
    for c in cands:
        ds = MultiTaskDataset(args.shard, max_len=160, plies_per_game=args.plies,
                              n_cand=c, with_rights=True)
        idxs = rng.choice(len(ds), args.n, replace=False)
        per = time_samples(ds, idxs)
        r, cd = split_cost(ds, idxs[:max(20, args.n // 5)], c)
        sps = args.workers / per
        print(f"{c:>7}{per*1000:>12.2f}{r*1000:>12.2f}{cd*1000:>10.2f}"
              f"{100*cd/(r+cd):>8.0f}%{sps:>12.0f}{sps/BATCH:>11.1f}")
        if c == 16:
            base = (ds, idxs)

    print(f"\nimplied it/s assumes {args.workers} workers each running one sample at a "
          f"time,\nwith zero collate/transfer overhead -- so it is an UPPER bound on "
          f"what the\nloader can feed. Compare against what the pods actually did:")
    for k, v in OBSERVED.items():
        print(f"    {k:<26}{v:>6.1f} it/s")

    ds, idxs = base
    pr = cProfile.Profile()
    pr.enable()
    for i in idxs[:args.n // 2]:
        ds[i]
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14)
    print("\n--- cProfile, n_cand=16, sorted by tottime ---")
    print("\n".join(s.getvalue().splitlines()[4:26]))


if __name__ == "__main__":
    main()
