"""Correctness checks and summary stats for an ingested shard.

    python verify.py /workspace/data/2013-01
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import chess

from bitboards import (
    board_to_planes, to_pov, decode_move, _flip_move_codes, game_to_bitboards,
)
from ingest import META_DTYPE, TERMINATIONS


def load(shard: str):
    meta = np.load(os.path.join(shard, "meta.npy"))
    moves = np.memmap(os.path.join(shard, "moves.u16"), dtype=np.uint16, mode="r")
    with open(os.path.join(shard, "players.txt"), encoding="utf-8") as f:
        players = f.read().split("\n")
    return meta, moves, players


def game_moves(meta_row, moves) -> np.ndarray:
    o = int(meta_row["offset"])
    return np.asarray(moves[o:o + int(meta_row["nply"])])


def check_pov_against_mirror(n: int = 400) -> None:
    """to_pov(planes, BLACK) must equal python-chess's own board.mirror() encoding."""
    rng = random.Random(0)
    checked = 0
    for _ in range(n):
        board = chess.Board()
        for _ in range(rng.randint(0, 60)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        ours = to_pov(board_to_planes(board), chess.BLACK)
        theirs = board_to_planes(board.mirror())
        assert np.array_equal(ours, theirs), f"POV mismatch at {board.fen()}"
        checked += 1
    print(f"  POV flip vs board.mirror():        {checked} random positions OK")


def check_move_flip(n: int = 2000) -> None:
    rng = random.Random(1)
    for _ in range(n):
        frm, to = rng.randrange(64), rng.randrange(64)
        promo = rng.choice([0, 2, 3, 4, 5])
        code = frm | (to << 6) | (promo << 12)
        flipped = int(_flip_move_codes(np.array([code], dtype=np.uint16))[0])
        m = decode_move(flipped)
        assert m.from_square == frm ^ 56 and m.to_square == to ^ 56
        assert (m.promotion or 0) == promo
    assert int(_flip_move_codes(np.array([0xFFFF], dtype=np.uint16))[0]) == 0xFFFF
    print(f"  move-code rank mirror:            {n} random codes OK (+ padding sentinel)")


def check_replay(meta, moves, n: int = 300) -> None:
    rng = random.Random(2)
    idx = rng.sample(range(len(meta)), min(n, len(meta)))
    for i in idx:
        codes = game_moves(meta[i], moves)
        board = chess.Board()
        for c in codes:
            mv = decode_move(int(c))
            assert board.is_legal(mv), f"illegal move in game {i}: {mv} at {board.fen()}"
            board.push(mv)
    print(f"  packed moves replay legally:      {len(idx)} games OK")


def check_bitboard_shapes(meta, moves, n: int = 50) -> None:
    rng = random.Random(3)
    for i in rng.sample(range(len(meta)), min(n, len(meta))):
        codes = game_moves(meta[i], moves)
        for pov in (chess.WHITE, chess.BLACK):
            planes, mv = game_to_bitboards(codes, pov)
            assert planes.shape == (len(codes) + 1, 18, 8, 8), planes.shape
            assert planes.dtype == np.uint8 and set(np.unique(planes)) <= {0, 1}
            assert mv.shape == (len(codes) + 1,) and mv[-1] == 0xFFFF
        # From the mover's own seat, the "my turn" plane must be 1 on the plies
        # they actually moved: White on even plies, Black on odd.
        wp, _ = game_to_bitboards(codes, chess.WHITE)
        bp, _ = game_to_bitboards(codes, chess.BLACK)
        assert wp[0, 12].all() and not bp[0, 12].any()
        assert bp[1, 12].all() and not wp[1, 12].any()
        # Each side starts with 8 pawns in its own plane 0.
        assert wp[0, 0].sum() == 8 and bp[0, 0].sum() == 8
    print(f"  POV bitboard tensors:             {n} games OK (shape/dtype/turn/parity)")


def stats(meta, moves, players, shard: str) -> None:
    nply = meta["nply"].astype(np.int64)
    elos = np.concatenate([meta["white_elo"], meta["black_elo"]])
    elos = elos[elos > 0]
    pids = np.concatenate([meta["white_pid"], meta["black_pid"]])
    counts = np.bincount(pids, minlength=len(players))

    print(f"\n  games                {len(meta):,}")
    print(f"  players              {len(players):,}")
    print(f"  plies                {nply.sum():,}")
    print(f"  moves.u16 on disk    {os.path.getsize(os.path.join(shard,'moves.u16'))/1e6:,.1f} MB")
    print(f"  plies/game           min {nply.min()} / med {int(np.median(nply))} / "
          f"mean {nply.mean():.1f} / p99 {int(np.percentile(nply,99))} / max {nply.max()}")
    if len(elos):
        print(f"  Elo                  p5 {int(np.percentile(elos,5))} / med {int(np.median(elos))} "
              f"/ p95 {int(np.percentile(elos,95))}")
    res = meta["result"]
    print(f"  result W/D/B         {(res==1).sum():,} / {(res==0).sum():,} / {(res==-1).sum():,}")
    term = np.bincount(meta["termination"], minlength=len(TERMINATIONS))
    print("  termination          " + ", ".join(
        f"{TERMINATIONS[i]} {c:,}" for i, c in enumerate(term) if c))

    print(f"\n  games per player     med {int(np.median(counts))} / mean {counts.mean():.1f} "
          f"/ max {counts.max():,}")
    for thresh in (5, 10, 20, 50, 100):
        n = int((counts >= thresh).sum())
        print(f"    players with >={thresh:>3} games   {n:>8,}  "
              f"({int(counts[counts>=thresh].sum()):,} game-sides)")
    top = np.argsort(-counts)[:5]
    print("  most active          " + ", ".join(f"{players[i]} ({counts[i]:,})" for i in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    args = ap.parse_args()

    with open(os.path.join(args.shard, "manifest.json")) as f:
        print(json.dumps(json.load(f), indent=2))

    meta, moves, players = load(args.shard)
    assert meta.dtype == META_DTYPE
    assert int(meta["offset"][-1]) + int(meta["nply"][-1]) == len(moves), \
        "meta offsets do not cover moves.u16 exactly"

    print("\nchecks:")
    check_pov_against_mirror()
    check_move_flip()
    check_replay(meta, moves)
    check_bitboard_shapes(meta, moves)
    stats(meta, moves, players, args.shard)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
