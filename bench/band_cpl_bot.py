"""Mean centipawn loss by band for the BOT, measured the same way as for humans.

The human number samples a move from a game two players of that rating actually
played. The bot analogue has to match that: play the bot against ITSELF with both
sides told the band's rating, then score one move from that game. Scoring the bot
on human positions would answer a different question -- how it does in someone
else's game -- and would not be comparable to the human curve.

The game is played only as far as the sampled ply, not to a finish. The position
is all that is needed and stopping early roughly halves the cost.

Openings come from real games, 8 plies. The bot is deterministic at temperature
0, so without them every game at a given rating would be the same game; the
sampled think times add some variety but not enough to rely on. Plies past the
opening are bot-generated, which is what is being measured.

Run the control (a checkpoint that ignores the rating) and the treatment in the
same invocation so they see identical openings and identical sampled plies.
"""
import argparse, json, os, random, sys, time
import numpy as np

REPO = "/Users/inteoryx/investigations/chess_tracker"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=1000)
    ap.add_argument("--hi", type=int, default=2400)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--n", type=int, default=250, help="sampled moves per band")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--first-ply", type=int, default=8)
    ap.add_argument("--last-ply", type=int, default=60)
    ap.add_argument("--decided-cp", type=int, default=400)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--openings", type=int, default=400)
    ap.add_argument("--bots", default="ctx5_pre.pt,ctx5_pre_elo.pt")
    ap.add_argument("--out", default="bench/band_cpl_bot.json")
    a = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "2"
    sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/play")
    import chess, chess.engine, torch, server
    from bitboards import decode_move

    meta = np.load(f"{REPO}/data/2026-06-big/meta.npy", allow_pickle=True)
    mv = np.memmap(f"{REPO}/data/2026-06-big/moves.u16", dtype=np.uint16, mode="r")

    # openings, drawn once and reused by every bot and band
    rr = random.Random(99)
    openings = []
    tries = 0
    while len(openings) < a.openings and tries < a.openings * 40:
        tries += 1
        g = meta[rr.randrange(len(meta))]
        if g["nply"] < a.opening_plies + 30: continue
        b, ok, h = chess.Board(), True, []
        for c in mv[g["offset"]:g["offset"] + a.opening_plies]:
            try: m = decode_move(int(c))
            except Exception: ok = False; break
            if m not in b.legal_moves: ok = False; break
            h.append(m.uci()); b.push(m)
        if ok and not b.is_game_over(): openings.append(h)
    print(f"{len(openings)} real openings of {a.opening_plies} plies\n", flush=True)

    eng = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    eng.configure({"Threads": 1, "Hash": 16})
    ev = lambda b: eng.analyse(b, chess.engine.Limit(depth=a.depth))["score"]\
                      .white().score(mate_score=2000)

    out = {}
    for ckpt in a.bots.split(","):
        server.MODEL.clear()
        server.load(f"{REPO}/ckpt/final/{ckpt}", "", "")
        torch.set_num_threads(2)
        cond = getattr(server.MODEL["move"], "elo_cond", None) is not None and \
               bool(server.MODEL["move"].elo_cond)
        print(f"{ckpt}  elo_conditioned={cond}", flush=True)
        print(f"  {'band':>11} {'n':>5} {'mean CPL':>9} {'±SE':>6} {'median':>7} "
              f"{'>100cp':>7} {'s':>6}")
        rows = []
        for lo in range(a.lo, a.hi, a.width):
            hi = lo + a.width
            elo = lo + a.width // 2                 # both sides told the same
            rng = random.Random(4242 + lo)          # same plies/openings per bot
            cpl = []
            t0 = time.perf_counter()
            attempts = 0
            while len(cpl) < a.n and attempts < a.n * 6:
                attempts += 1
                op = openings[rng.randrange(len(openings))]
                want = rng.randrange(a.first_ply, a.last_ply)
                want += (want % 2)                   # land on a White move
                board = chess.Board(); hist = []
                for u in op:
                    board.push(chess.Move.from_uci(u)); hist.append(u)
                times = [server.sample_think_ms(i) for i in range(len(hist))]
                bad = False
                while len(hist) < want:
                    if board.is_game_over(claim_draw=True): bad = True; break
                    uci, _ = server.think(hist, 0.0, times, elo)
                    if uci is None: bad = True; break
                    board.push(chess.Move.from_uci(uci)); hist.append(uci)
                    times.append(server.sample_think_ms(len(hist) - 1))
                if bad or board.turn != chess.WHITE or board.is_game_over(claim_draw=True):
                    continue
                e0 = ev(board)
                if e0 is None or abs(e0) > a.decided_cp:
                    continue
                uci, _ = server.think(hist, 0.0, times, elo)   # the move being scored
                if uci is None: continue
                board.push(chess.Move.from_uci(uci)); eb = ev(board)
                if eb is None: continue
                cpl.append(max(0, e0 - eb))
            if len(cpl) < min(20, a.n): continue
            c = np.array(cpl, float)
            rec = {"band": f"{lo}-{hi}", "lo": lo, "elo_fed": elo, "n": len(c),
                   "mean_cpl": float(c.mean()),
                   "se": float(c.std(ddof=1) / np.sqrt(len(c))),
                   "median_cpl": float(np.median(c)),
                   "blunder_pct": float((c > 100).mean() * 100)}
            rows.append(rec)
            print(f"  {lo:>4}-{hi:<5} {len(c):>5} {c.mean():>9.1f} {rec['se']:>6.1f} "
                  f"{np.median(c):>7.0f} {rec['blunder_pct']:>6.0f}% "
                  f"{time.perf_counter()-t0:>6.0f}", flush=True)
            out[ckpt] = {"elo_conditioned": cond, "rows": rows}
            json.dump(out, open(a.out, "w"), indent=1)
        if rows:
            x = np.array([r["lo"] + 50 for r in rows])
            y = np.array([r["mean_cpl"] for r in rows])
            print(f"  trend {np.polyfit(x, y, 1)[0]*100:+.1f} cp / 100 rating "
                  f"(r {np.corrcoef(x, y)[0,1]:+.2f})\n", flush=True)
    eng.quit()
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
