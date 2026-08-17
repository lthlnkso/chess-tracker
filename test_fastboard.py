"""Verify fastboard produces byte-identical planes to the reference encoder.

A wrong encoder does not crash -- it silently trains the model on a corrupted
board, and the loss curve looks fine. So this compares every plane of every
position of real games, including the positions reached by pushing each legal
move (the candidate successors, which is where most encodings happen), for both
the 8- and 13-plane variants and both points of view.

    python test_fastboard.py --shard data/2026-06-big --games 300
"""

from __future__ import annotations

import argparse

import numpy as np
import chess

from bitboards import board_to_planes8, decode_move
from fastboard import snapshot, rights, encode_batch
from successor_data import MultiTaskDataset


def check_positions(boards, povs, n_planes):
    ref = np.stack([board_to_planes8(b, p, None, n_planes == 13)
                    for b, p in zip(boards, povs)])
    snaps = np.stack([snapshot(b, p) for b, p in zip(boards, povs)])
    ra = (np.array([rights(b, p) for b, p in zip(boards, povs)], dtype=np.int16)
          if n_planes == 13 else None)
    got = encode_batch(snaps, n_planes, ra)
    if not np.array_equal(ref, got):
        bad = np.flatnonzero((ref != got).any(axis=(1, 2, 3)))
        i = int(bad[0])
        planes = np.flatnonzero((ref[i] != got[i]).any(axis=(1, 2)))
        raise AssertionError(
            f"{len(bad)}/{len(ref)} positions differ; first is index {i}, "
            f"planes {planes.tolist()}, fen {boards[i].fen()} pov "
            f"{'W' if povs[i] else 'B'}\nref:\n{ref[i][planes[0]]}\n"
            f"got:\n{got[i][planes[0]]}")
    return len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = MultiTaskDataset(args.shard, max_len=160, plies_per_game=12, n_cand=16)
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(ds), args.games, replace=False)

    total = 0
    for n_planes in (8, 13):
        checked = 0
        for i in picks:
            gi, seat = (int(x) for x in ds.index[i])
            row = ds.meta[gi]
            o, n = int(row["offset"]), int(row["nply"])
            codes = np.asarray(ds.moves[o:o + n])
            pov = chess.WHITE if seat == 0 else chess.BLACK

            boards, povs = [], []
            board = chess.Board()
            for t in range(min(len(codes), 160)):
                boards.append(board.copy(stack=False)); povs.append(pov)
                # every legal successor, which is what the candidate sets encode
                for m in board.legal_moves:
                    board.push(m)
                    boards.append(board.copy(stack=False)); povs.append(pov)
                    board.pop()
                board.push(decode_move(int(codes[t])))
            checked += check_positions(boards, povs, n_planes)
        print(f"  {n_planes:>2} planes: {checked:,} positions identical "
              f"across {args.games} games")
        total += checked

    # Positions the games themselves rarely reach.
    edge = [
        chess.Board(),                                                   # start
        chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1"),                      # bare kings
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),             # all castling
        chess.Board("rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3"),
        chess.Board("8/P7/8/8/8/8/7p/K6k w - - 0 1"),                    # promotions
        chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1"),                   # one right
    ]
    for b in edge:
        for pov in (chess.WHITE, chess.BLACK):
            for n_planes in (8, 13):
                check_positions([b], [pov], n_planes)
    print(f"  edge cases: {len(edge)*4} encodings identical")
    print(f"\nOK - {total:,} positions verified byte-identical")


if __name__ == "__main__":
    main()
