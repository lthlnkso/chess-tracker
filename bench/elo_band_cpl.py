"""Harness 2 -- does the bot's move quality track the rating it is asked to imitate?

For each 100-point band, take real positions from real games of players in that
band and ask the bot "what would a player rated X do here", with X the mover's
ACTUAL rating. Score its move and the human's move on the same position, in
centipawns.

The property being tested is not "is the bot good". It is whether the GAP between
the bot and the human it is imitating stays roughly constant across bands. A bot
that ignores the rating plays one fixed strength, so it looks better than weak
players and worse than strong ones, and the gap swings from strongly negative to
strongly positive. A bot that tracks its instruction holds a flat gap.

Two stages, because the engine evaluations dominate the cost and do not depend on
which bot is being tested:
    build   -- sample positions per band, judge the position and the human move
    score   -- load one bot at a time, judge only its move

Stage `build` is cached to disk so a new bot costs one cheap pass.
"""
import argparse, io, json, os, subprocess, sys, time
import numpy as np

REPO = "/Users/inteoryx/investigations/chess_tracker"
SF = "/opt/homebrew/bin/stockfish"


def free_mb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    head, *lines = out.splitlines()
    psz = int(head.split("page size of")[1].split("bytes")[0].strip())
    d = {}
    for ln in lines:
        if ":" in ln:
            k, v = ln.split(":", 1); v = v.strip().rstrip(".")
            if v.isdigit(): d[k.strip()] = int(v)
    return (d.get("Pages free", 0) + d.get("Pages inactive", 0)) * psz / (1 << 20)


def build(a):
    sys.path.insert(0, REPO)
    import chess, chess.engine
    from bitboards import decode_move
    from timefeat import ms_used_per_ply

    meta = np.load(f"{REPO}/data/2026-06-big/meta.npy", allow_pickle=True)
    mv = np.memmap(f"{REPO}/data/2026-06-big/moves.u16", dtype=np.uint16, mode="r")
    ck = np.memmap(f"{REPO}/data/2026-06-big/clocks.u16", dtype=np.uint16, mode="r")
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 16})
    def ev(b):
        return eng.analyse(b, chess.engine.Limit(depth=a.depth))["score"].white()\
                  .score(mate_score=2000)

    rng = np.random.default_rng(7)
    out = []
    print(f"{'band':>11} {'positions':>10} {'human ACPL':>11}", flush=True)
    for lo in range(a.lo, a.hi, a.width):
        hi = lo + a.width
        sel = np.flatnonzero((meta["white_elo"] >= lo) & (meta["white_elo"] < hi) &
                             (meta["nply"] > a.first_ply + 12))
        rng.shuffle(sel)
        got = []
        for gi in sel:
            if len(got) >= a.positions: break
            if free_mb() < a.min_free_mb:
                print(f"  stopping: {free_mb():.0f} MB free"); raise SystemExit(1)
            g = meta[gi]; n = int(g["nply"])
            clocks = np.asarray(ck[g["offset"]:g["offset"]+n], dtype=np.float64)
            durs = np.nan_to_num(ms_used_per_ply(clocks, int(g["tc_base"]),
                                                 int(g["tc_inc"])), nan=0.0).tolist()
            board = chess.Board(); hist = []
            for t in range(n):
                if len(got) >= a.positions: break
                try: m = decode_move(int(mv[g["offset"] + t]))
                except Exception: break
                if m not in board.legal_moves: break
                if (a.first_ply <= t < min(n - 1, a.last_ply)
                        and board.turn == chess.WHITE):
                    e0 = ev(board)
                    if e0 is not None and abs(e0) <= a.decided_cp:
                        board.push(m); eh = ev(board); board.pop()
                        if eh is not None:
                            got.append({"band": f"{lo}-{hi}", "lo": lo,
                                        "elo": int(g["white_elo"]),
                                        "history": list(hist),
                                        "times": durs[:len(hist)],
                                        "human_uci": m.uci(),
                                        "e0": e0, "e_human": eh})
                board.push(m); hist.append(m.uci())
        if got:
            h = float(np.mean([max(0, x["e0"] - x["e_human"]) for x in got]))
            print(f"  {lo:>4}-{hi:<5} {len(got):>10} {h:>11.0f}", flush=True)
            out += got
            json.dump(out, open(a.cache, "w"))
    eng.quit()
    json.dump(out, open(a.cache, "w"))
    print(f"\ncached {len(out):,} positions -> {a.cache}")


def score(a):
    sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/play")
    os.environ["OMP_NUM_THREADS"] = "2"
    import chess, chess.engine, torch, server
    pos = json.load(io.open(a.cache))
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 16})
    def ev(b):
        return eng.analyse(b, chess.engine.Limit(depth=a.depth))["score"].white()\
                  .score(mate_score=2000)

    results = {}
    for ckpt in a.bots.split(","):
        server.MODEL.clear()
        server.load(f"{REPO}/ckpt/final/{ckpt}", "", "")
        torch.set_num_threads(2)
        cond = getattr(server.MODEL["move"], "elo_cond", None) is not None and \
               bool(server.MODEL["move"].elo_cond)
        per = {}
        t0 = time.perf_counter()
        for i, p in enumerate(pos):
            board = chess.Board()
            for u in p["history"]: board.push(chess.Move.from_uci(u))
            uci, _ = server.think(list(p["history"]), 0.0, p["times"], p["elo"])
            if uci is None: continue
            board.push(chess.Move.from_uci(uci)); eb = ev(board)
            if eb is None: continue
            d = per.setdefault(p["band"], {"h": [], "b": [], "agree": 0, "n": 0})
            d["h"].append(max(0, p["e0"] - p["e_human"]))
            d["b"].append(max(0, p["e0"] - eb))
            d["agree"] += (uci == p["human_uci"]); d["n"] += 1
        def stats(v):
            h = np.array(v["h"], float); b = np.array(v["b"], float)
            # PAIRED: bot and human played the same position, so differencing
            # per position cancels how sharp that position was. Mean-minus-mean
            # would carry all of that variance into the gap.
            d = b - h
            se = float(d.std(ddof=1) / max(np.sqrt(len(d)), 1)) if len(d) > 1 else 0.0
            return {"n": v["n"], "human": float(h.mean()), "bot": float(b.mean()),
                    "gap": float(d.mean()), "gap_se": se,
                    "gap_med": float(np.median(d)),
                    "human_blunder": float((h > 100).mean() * 100),
                    "bot_blunder": float((b > 100).mean() * 100),
                    "agree": 100.0 * v["agree"] / max(v["n"], 1)}
        results[ckpt] = {"elo_conditioned": cond, "bands": {
            k: stats(v)
            for k, v in sorted(per.items(), key=lambda kv: int(kv[0].split("-")[0]))}}
        print(f"\n{ckpt}   elo_conditioned={cond}   ({time.perf_counter()-t0:.0f}s)")
        print(f"  {'band':>11} {'n':>5} {'human':>7} {'bot':>7} "
              f"{'gap (paired)':>16} {'blunder h/b':>13} {'agree':>7}")
        for k, v in results[ckpt]["bands"].items():
            print(f"  {k:>11} {v['n']:>5} {v['human']:>7.0f} {v['bot']:>7.0f} "
                  f"{v['gap']:>+9.0f} +-{v['gap_se']:<4.0f} "
                  f"{v['human_blunder']:>5.0f}%/{v['bot_blunder']:<5.0f}% "
                  f"{v['agree']:>6.1f}%")
        gaps = [v["gap"] for v in results[ckpt]["bands"].values()]
        print(f"  TRACKING: gap spread {max(gaps)-min(gaps):.0f} cp, "
              f"sd {np.std(gaps):.0f}  <- flat gap = the bot follows the rating it is told")
    eng.quit()
    json.dump(results, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "score"])
    ap.add_argument("--lo", type=int, default=1000)
    ap.add_argument("--hi", type=int, default=2400)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--first-ply", type=int, default=8)
    ap.add_argument("--last-ply", type=int, default=60)
    ap.add_argument("--decided-cp", type=int, default=400)
    ap.add_argument("--min-free-mb", type=float, default=600)
    ap.add_argument("--cache", default="bench/elo_band_positions.json")
    ap.add_argument("--bots", default="ctx5_pre.pt,ctx10_pre.pt,ctx5_pre_elo.pt")
    ap.add_argument("--out", default="bench/elo_band_cpl.json")
    a = ap.parse_args()
    (build if a.stage == "build" else score)(a)


if __name__ == "__main__":
    main()
