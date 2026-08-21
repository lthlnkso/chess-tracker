"""Human centipawn loss per rating band, and what it costs to measure.

Sampling is NESTED, not repeated: 100 positions per band are drawn once and the
running mean is reported at 10, 50 and 100. So the three columns are the same
experiment seen at three sample sizes rather than three unrelated draws, which
is what makes the convergence readable.

Timing is cumulative wall clock for the first N of each band, including the cost
of SCANNING for qualifying positions -- games are rejected for undecodable
moves, and positions for falling outside the ply window or being already decided.
That scan is a real part of the bill and extrapolating without it understates.

The standard error is reported alongside, because "how long for 1,000" is only
worth knowing next to "how much better is the answer".
"""
import argparse, json, os, sys, time
import numpy as np

REPO = "/Users/inteoryx/investigations/chess_tracker"
SF = "/opt/homebrew/bin/stockfish"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=1000)
    ap.add_argument("--hi", type=int, default=2400)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--max-n", type=int, default=100)
    ap.add_argument("--marks", default="10,50,100")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--first-ply", type=int, default=8)
    ap.add_argument("--last-ply", type=int, default=60)
    ap.add_argument("--decided-cp", type=int, default=400)
    ap.add_argument("--target", type=int, default=1000, help="extrapolate to this")
    ap.add_argument("--out", default="bench/band_cpl_timing.json")
    a = ap.parse_args()

    sys.path.insert(0, REPO)
    import chess, chess.engine
    from bitboards import decode_move

    meta = np.load(f"{REPO}/data/2026-06-big/meta.npy", allow_pickle=True)
    mv = np.memmap(f"{REPO}/data/2026-06-big/moves.u16", dtype=np.uint16, mode="r")
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 16})

    def ev(b):
        return eng.analyse(b, chess.engine.Limit(depth=a.depth))["score"]\
                  .white().score(mate_score=2000)

    marks = [int(x) for x in a.marks.split(",")]
    rng = np.random.default_rng(3)
    rows = []
    print(f"judge: Stockfish depth {a.depth}, 1 thread\n")
    hdr = "  " + f"{'band':>11}" + "".join(f"{'n=' + str(m):>19}" for m in marks) + f"{'scan':>8}"
    sub = "  " + " " * 11 + "".join(f"{'ACPL':>9}{'±SE':>6}{'s':>4}" for m in marks) + f"{'eff%':>8}"
    print(hdr); print(sub, flush=True)

    for lo in range(a.lo, a.hi, a.width):
        hi = lo + a.width
        sel = np.flatnonzero((meta["white_elo"] >= lo) & (meta["white_elo"] < hi) &
                             (meta["nply"] > a.first_ply + 12))
        if len(sel) == 0:
            continue
        rng.shuffle(sel)
        cpl, tstamp = [], []
        looked = 0
        t0 = time.perf_counter()
        for gi in sel:
            if len(cpl) >= a.max_n:
                break
            g = meta[gi]; n = int(g["nply"])
            board = chess.Board()
            for t in range(n):
                if len(cpl) >= a.max_n:
                    break
                try:
                    m = decode_move(int(mv[g["offset"] + t]))
                except Exception:
                    break
                if m not in board.legal_moves:
                    break
                if (a.first_ply <= t < min(n - 1, a.last_ply)
                        and board.turn == chess.WHITE):
                    looked += 1
                    e0 = ev(board)
                    if e0 is not None and abs(e0) <= a.decided_cp:
                        board.push(m); eh = ev(board); board.pop()
                        if eh is not None:
                            cpl.append(max(0, e0 - eh))
                            tstamp.append(time.perf_counter() - t0)
                board.push(m)
        if len(cpl) < max(marks):
            continue
        c = np.array(cpl, float)
        cells = []
        rec = {"band": f"{lo}-{hi}", "lo": lo, "looked": looked, "kept": len(cpl)}
        for m in marks:
            s = c[:m]
            se = float(s.std(ddof=1) / np.sqrt(m))
            rec[f"n{m}"] = {"acpl": float(s.mean()), "se": se, "sec": tstamp[m - 1]}
            cells.append(f"{s.mean():>9.0f}{se:>6.0f}{tstamp[m-1]:>4.0f}")
        eff = 100.0 * len(cpl) / max(looked, 1)
        rec["scan_efficiency"] = eff
        rows.append(rec)
        print(f"  {lo:>4}-{hi:<5}" + "".join(cells) + f"{eff:>7.0f}%", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    eng.quit()
    json.dump(rows, open(a.out, "w"), indent=1)

    big = max(marks)
    per_band = np.array([r[f"n{big}"]["sec"] for r in rows])
    per_move = per_band / big
    tot_big = per_band.sum()
    scale = a.target / big
    print(f"\n  {len(rows)} bands, {big} moves each = {len(rows)*big:,} moves "
          f"in {tot_big/60:.1f} min")
    print(f"  per move: mean {per_move.mean():.2f}s  "
          f"(fastest band {per_move.min():.2f}s, slowest {per_move.max():.2f}s)")
    print(f"\n  EXTRAPOLATION to {a.target} moves/band "
          f"({len(rows)*a.target:,} moves total):")
    print(f"    {tot_big*scale/60:.0f} min  = {tot_big*scale/3600:.1f} h  "
          f"single-threaded at depth {a.depth}")
    print(f"    on 6 cores in parallel: ~{tot_big*scale/3600/6:.1f} h")
    se_big = np.array([r[f"n{big}"]["se"] for r in rows])
    print(f"\n  precision: SE is now {se_big.mean():.0f} cp at n={big}; "
          f"at n={a.target} it would be ~{se_big.mean()/np.sqrt(scale):.0f} cp "
          f"(SE falls as 1/sqrt(n))")


if __name__ == "__main__":
    main()
