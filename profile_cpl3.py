"""Round 3: how shallow can the CPL labels be before they stop being CPL?

Rounds 1-2 said multipv width dominates cost and depth is the other big term:
depth 6 is 3.6x cheaper than depth 8, which decides whether labelling a useful
corpus costs $0.56 or $4. But cheap labels are worthless if they rank moves
differently from good ones, so this measures AGREEMENT, not speed.

Depth 12 is the reference. For each position we take the full CPL vector over
candidates and ask three things of each cheaper depth:

  rank agreement    Spearman over the candidate CPLs. This is what the loss
                    actually consumes -- it compares candidates to each other.
  blunder agreement does it agree on which moves are >=100cp mistakes? For
                    modelling a human's error profile this matters far more than
                    resolving 20cp from 40cp.
  absolute error    RMS centipawn difference, which bounds how much noise the
                    (CPL_i - CPL_j)^2 term inherits.

Bullet players are not finding depth-12 tactics, so faithful-to-the-human may
well be shallower than faithful-to-the-engine. This says how much shallower.
"""

from __future__ import annotations

import argparse
import time

import chess
import chess.engine
import numpy as np

from profile_cpl import sample_positions

MATE_CP = 2000          # mate scores clipped; unbounded values would dominate


def cpl_vector(eng, board, depth, multipv):
    """CPL per root move, in centipawns, from the mover's point of view."""
    info = eng.analyse(board, chess.engine.Limit(depth=depth),
                       multipv=multipv, game=object())
    out = {}
    for e in info:
        pv = e.get("pv")
        if not pv:
            continue
        sc = e["score"].pov(board.turn).score(mate_score=MATE_CP)
        out[pv[0].uci()] = float(np.clip(sc, -MATE_CP, MATE_CP))
    if not out:
        return {}
    best = max(out.values())
    return {m: best - v for m, v in out.items()}      # CPL >= 0, 0 = best move


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--positions", type=int, default=25)
    ap.add_argument("--multipv", type=int, default=32)
    ap.add_argument("--ref-depth", type=int, default=12)
    args = ap.parse_args()

    boards = sample_positions(args.shard, args.positions, seed=7)
    eng = chess.engine.SimpleEngine.popen_uci(args.engine)
    eng.configure({"Threads": 1, "Hash": 128})

    print(f"reference: depth {args.ref_depth}, multipv {args.multipv}, "
          f"{len(boards)} positions\n")
    ref = [cpl_vector(eng, b, args.ref_depth, args.multipv) for b in boards]

    print(f"{'depth':>6} {'ms/pos':>8} {'rank rho':>10} {'blunder agree':>14} "
          f"{'RMS cp':>9} {'best-move same':>15}")
    print("-" * 68)
    for d in (2, 4, 6, 8, 10):
        t0 = time.perf_counter()
        got = [cpl_vector(eng, b, d, args.multipv) for b in boards]
        ms = (time.perf_counter() - t0) / len(boards) * 1000
        rhos, blun, errs, same = [], [], [], []
        for r, g in zip(ref, got):
            keys = [k for k in r if k in g]
            if len(keys) < 3:
                continue
            a = np.array([r[k] for k in keys]); b_ = np.array([g[k] for k in keys])
            rhos.append(spearman(a, b_))
            blun.append(float(((a >= 100) == (b_ >= 100)).mean()))
            errs.append(float(np.sqrt(((a - b_) ** 2).mean())))
            same.append(float(min(r, key=r.get) == min(g, key=g.get)))
        print(f"{d:>6} {ms:>8.1f} {np.nanmean(rhos):>10.3f} "
              f"{np.mean(blun):>13.1%} {np.mean(errs):>9.0f} {np.mean(same):>14.1%}")
    eng.quit()
    print("PROFILE3_DONE")


if __name__ == "__main__":
    main()
