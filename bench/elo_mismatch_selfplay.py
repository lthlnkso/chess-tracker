"""Harness 3 -- internal consistency of the rating conditioning.

The bot plays itself. Both sides are the same weights; the only difference is
the rating each side is TOLD to play at. If the conditioning means anything, the
side told it is stronger should score more, and score more as the gap widens.

This needs no engine and no human data, which is the point: it isolates the
conditioning from every question about whether our Elo scale matches lichess's.
A bot that ignores the rating produces the same game whichever numbers are
passed, so its curve is flat by construction -- that is the null this is read
against.

Mismatch tops out at 1800, not the 2000 asked for: conditioning is trained over
800-2600 in 100-point bins, so 700 vs 2700 lands in the same two edge bins as
800 vs 2600 and measures nothing new.
"""
import argparse, json, os, random, sys, time
import numpy as np

REPO = "/Users/inteoryx/investigations/chess_tracker"
SF = "/opt/homebrew/bin/stockfish"


def load_openings(n, plies, seed):
    import chess
    sys.path.insert(0, REPO)
    from bitboards import decode_move
    meta = np.load(f"{REPO}/data/2026-06-big/meta.npy", allow_pickle=True)
    mv = np.memmap(f"{REPO}/data/2026-06-big/moves.u16", dtype=np.uint16, mode="r")
    rng = random.Random(seed)
    out, tries = [], 0
    while len(out) < n and tries < n * 30:
        tries += 1
        g = meta[rng.randrange(len(meta))]
        if g["nply"] < plies + 20: continue
        b, ok, h = chess.Board(), True, []
        for c in mv[g["offset"]:g["offset"] + plies]:
            try: m = decode_move(int(c))
            except Exception: ok = False; break
            if m not in b.legal_moves: ok = False; break
            h.append(m.uci()); b.push(m)
        if ok and not b.is_game_over(): out.append(h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ctx5_pre_elo.pt")
    ap.add_argument("--centre", type=int, default=1700)
    ap.add_argument("--mismatches", default="0,100,200,300,400,600,800,1000,1200,1400,1600,1800")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--openings", type=int, default=30)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=180)
    ap.add_argument("--out", default="bench/elo_mismatch_selfplay.json")
    a = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "2"
    sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/play")
    import chess, chess.engine, torch, server

    openings = load_openings(a.openings, a.opening_plies, seed=42)
    server.load(f"{REPO}/ckpt/final/{a.ckpt}", "", "")
    torch.set_num_threads(2)
    cond = getattr(server.MODEL["move"], "elo_cond", None) is not None and \
           bool(server.MODEL["move"].elo_cond)
    judge = chess.engine.SimpleEngine.popen_uci(SF)
    judge.configure({"Threads": 1, "Hash": 16})

    print(f"{a.ckpt}   elo_conditioned={cond}   centre {a.centre}\n")
    print(f"  {'mismatch':>9} {'pairing':>13} {'high scores':>12} {'W':>4} {'D':>4} {'L':>4} {'plies':>6}")
    rows = []
    for m in [int(x) for x in a.mismatches.split(",")]:
        lo, hi = a.centre - m // 2, a.centre + m // 2
        score = 0.0; w = d = l = 0; tot_plies = 0
        for gi in range(a.games):
            opening = openings[gi % len(openings)]
            high_white = (gi % 2 == 0)
            board = chess.Board(); hist = []
            for u in opening:
                board.push(chess.Move.from_uci(u)); hist.append(u)
            times = [server.sample_think_ms(i) for i in range(len(hist))]
            while not board.is_game_over(claim_draw=True) and len(hist) < a.max_plies:
                is_high = (board.turn == chess.WHITE) == high_white
                elo = hi if is_high else lo
                uci, _ = server.think(hist, 0.0, times, elo)
                if uci is None: break
                board.push(chess.Move.from_uci(uci)); hist.append(uci)
                times.append(server.sample_think_ms(len(hist) - 1))
            o = board.outcome(claim_draw=True)
            if o is not None:
                pt = 0.5 if o.winner is None else (1.0 if (o.winner == chess.WHITE) == high_white else 0.0)
            else:
                cp = judge.analyse(board, chess.engine.Limit(depth=12))["score"]\
                          .white().score(mate_score=10000)
                if cp is None: pt = 0.5
                elif cp > 150: pt = 1.0 if high_white else 0.0
                elif cp < -150: pt = 0.0 if high_white else 1.0
                else: pt = 0.5
            score += pt; tot_plies += len(hist)
            w += pt == 1.0; d += pt == 0.5; l += pt == 0.0
        rate = score / a.games
        rows.append({"mismatch": m, "lo": lo, "hi": hi, "rate": rate,
                     "w": w, "d": d, "l": l, "avg_plies": tot_plies / a.games})
        print(f"  {m:>9} {f'{lo} v {hi}':>13} {rate*100:>11.1f}% {w:>4} {d:>4} {l:>4} "
              f"{tot_plies/a.games:>6.0f}", flush=True)
        json.dump({"ckpt": a.ckpt, "elo_conditioned": cond, "rows": rows},
                  open(a.out, "w"), indent=1)
    judge.quit()
    rates = [r["rate"] for r in rows]
    span = max(rates) - min(rates)
    print(f"\n  spread {span*100:.1f} points across the mismatch range"
          f"   (flat = conditioning does nothing)")
    if len(rates) > 2:
        c = np.corrcoef([r["mismatch"] for r in rows], rates)[0, 1]
        print(f"  correlation(mismatch, score) = {c:+.2f}   (want strongly positive)")


if __name__ == "__main__":
    main()
