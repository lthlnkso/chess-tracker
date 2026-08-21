"""Mean centipawn loss by rating band. One random move from each of 100 games.

One move per game on purpose. Taking 100 moves out of a handful of games gives
100 numbers but nowhere near 100 independent samples -- moves inside one game
share a player, an opening and a mood, so the error bars come out far too
narrow. One move from each of 100 distinct games is what the sample size claims.
"""
import argparse, json, sys, time
import numpy as np

REPO = "/Users/inteoryx/investigations/chess_tracker"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=1000)
    ap.add_argument("--hi", type=int, default=2400)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--n", type=int, default=100, help="moves per band, 1 per game")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--first-ply", type=int, default=8)
    ap.add_argument("--last-ply", type=int, default=60)
    ap.add_argument("--decided-cp", type=int, default=400)
    ap.add_argument("--out", default="bench/band_cpl.json")
    a = ap.parse_args()

    sys.path.insert(0, REPO)
    import chess, chess.engine
    from bitboards import decode_move

    meta = np.load(f"{REPO}/data/2026-06-big/meta.npy", allow_pickle=True)
    mv = np.memmap(f"{REPO}/data/2026-06-big/moves.u16", dtype=np.uint16, mode="r")
    eng = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    eng.configure({"Threads": 1, "Hash": 16})
    ev = lambda b: eng.analyse(b, chess.engine.Limit(depth=a.depth))["score"]\
                      .white().score(mate_score=2000)

    rng = np.random.default_rng(17)
    rows = []
    t0 = time.perf_counter()
    print(f"one random move from each of {a.n} games per band, "
          f"Stockfish depth {a.depth}\n")
    print(f"  {'band':>11} {'n':>5} {'mean CPL':>9} {'±SE':>6} {'median':>7} "
          f"{'>100cp':>7} {'games':>7}")
    for lo in range(a.lo, a.hi, a.width):
        hi = lo + a.width
        sel = np.flatnonzero((meta["white_elo"] >= lo) & (meta["white_elo"] < hi) &
                             (meta["nply"] > a.first_ply + 12))
        rng.shuffle(sel)
        cpl, used = [], 0
        for gi in sel:
            if len(cpl) >= a.n:
                break
            used += 1
            g = meta[gi]; n = int(g["nply"])
            hiply = min(n - 1, a.last_ply)
            if hiply <= a.first_ply:
                continue
            # one White move, chosen uniformly from this game's eligible plies
            cands = [t for t in range(a.first_ply, hiply) if t % 2 == 0]
            if not cands:
                continue
            want = int(rng.choice(cands))
            board = chess.Board(); ok = True
            for t in range(want):
                try: m = decode_move(int(mv[g["offset"] + t]))
                except Exception: ok = False; break
                if m not in board.legal_moves: ok = False; break
                board.push(m)
            if not ok or board.turn != chess.WHITE:
                continue
            try: m = decode_move(int(mv[g["offset"] + want]))
            except Exception: continue
            if m not in board.legal_moves:
                continue
            e0 = ev(board)
            if e0 is None or abs(e0) > a.decided_cp:
                continue
            board.push(m); eh = ev(board)
            if eh is None: continue
            cpl.append(max(0, e0 - eh))
        if len(cpl) < 20:
            continue
        c = np.array(cpl, float)
        rec = {"band": f"{lo}-{hi}", "lo": lo, "n": len(c),
               "mean_cpl": float(c.mean()),
               "se": float(c.std(ddof=1) / np.sqrt(len(c))),
               "median_cpl": float(np.median(c)),
               "blunder_pct": float((c > 100).mean() * 100), "games_scanned": used}
        rows.append(rec)
        print(f"  {lo:>4}-{hi:<5} {len(c):>5} {c.mean():>9.1f} {rec['se']:>6.1f} "
              f"{np.median(c):>7.0f} {rec['blunder_pct']:>6.0f}% {used:>7}", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)
    eng.quit()
    json.dump(rows, open(a.out, "w"), indent=1)
    x = np.array([r["lo"] + 50 for r in rows]); y = np.array([r["mean_cpl"] for r in rows])
    sl = np.polyfit(x, y, 1)[0]
    print(f"\n  {time.perf_counter()-t0:.0f}s total")
    print(f"  trend: {sl*100:+.1f} cp per 100 rating points  "
          f"(correlation {np.corrcoef(x, y)[0,1]:+.2f})")


if __name__ == "__main__":
    main()
