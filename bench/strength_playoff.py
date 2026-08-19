"""Our bot against Stockfish across a strength range.

Tests one link in the chain: is the bot weak enough that a strong human would
play differently against it than against their own peers?

FOOTPRINT. An earlier version ran nine strength levels in parallel, each its own
Python process with torch and the model loaded and each opening TWO Stockfish
engines -- eighteen engines and ~4.3 GB on an 8 GB machine that was already
holding three identify workers. It took the machine down. This one is strictly
sequential: one process, one model load, one engine reconfigured per level,
about 350 MB total, and it checks free memory between games and stops rather
than swapping.

Design notes that matter for the number to mean anything:

* The bot reads think time at every ply and leans on it hard. Feeding zeros
  would measure a bot that never ships. Both sides' plies get durations sampled
  from think_times.json -- the same real-game table the demo uses.
* Temperature 0, matching production. The bot is therefore deterministic, so
  game variety comes from real opening positions drawn from the lichess dump.
* Colours alternate; every opening is played from both sides.
* No clock forfeits. Time is a feature here, not a rule -- this measures chess
  strength, not who flags first.
"""
import argparse, json, os, random, subprocess, sys, time

REPO = "/Users/inteoryx/investigations/chess_tracker"
SF = "/opt/homebrew/bin/stockfish"


def free_mb():
    """Free + inactive pages -- what the kernel can actually hand out.

    The page size is READ, not assumed: Apple Silicon reports 16384-byte pages
    and hardcoding 4096 understates free memory by 4x, which would make this
    guard abort on a perfectly healthy machine.
    """
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        head, *lines = out.splitlines()
        psz = int(head.split("page size of")[1].split("bytes")[0].strip())
        d = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                v = v.strip().rstrip(".")
                if v.isdigit():
                    d[k.strip()] = int(v)
        return (d.get("Pages free", 0) + d.get("Pages inactive", 0)) * psz / (1 << 20)
    except Exception:                                          # noqa: BLE001
        return 0.0        # unreadable -> refuse to run, never "assume plenty"


def load_openings(n_openings, plies, seed):
    """Opening positions from real 1+0 lichess games, not a synthetic book."""
    import numpy as np, chess
    sys.path.insert(0, REPO)
    from bitboards import decode_move
    meta = np.load(f"{REPO}/data/2026-06-big/meta.npy", allow_pickle=True)
    mv = np.memmap(f"{REPO}/data/2026-06-big/moves.u16", dtype=np.uint16, mode="r")
    rng = random.Random(seed)
    out, tries = [], 0
    while len(out) < n_openings and tries < n_openings * 30:
        tries += 1
        g = meta[rng.randrange(len(meta))]
        if g["nply"] < plies + 20:
            continue
        b, ok, hist = chess.Board(), True, []
        for c in mv[g["offset"]:g["offset"] + plies]:
            try:
                m = decode_move(int(c))
                if m not in b.legal_moves:
                    ok = False; break
                hist.append(m.uci()); b.push(m)
            except Exception:                                  # noqa: BLE001
                ok = False; break
        if ok and not b.is_game_over():
            out.append(hist)
    del mv, meta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", default="0,1,2,3,5,8")
    ap.add_argument("--elos", default="1320,1500,1700")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--sf-time", type=float, default=0.08)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--openings", type=int, default=30)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--min-free-mb", type=float, default=900)
    ap.add_argument("--out", default="playoff.json")
    a = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/play")
    import chess, chess.engine, torch, server

    print(f"free memory at start: {free_mb():.0f} MB", flush=True)
    openings = load_openings(a.openings, a.opening_plies, seed=99)
    print(f"{len(openings)} real openings of {a.opening_plies} plies", flush=True)

    server.load(f"{REPO}/ckpt/final/ctx5_pre.pt", "", "")
    torch.set_num_threads(1)

    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 8})

    levels = ([(int(x), "skill") for x in a.skills.split(",") if x != ""] +
              [(int(x), "elo") for x in a.elos.split(",") if x != ""])
    rows = []
    print(f"\n{'opponent':>14} {'score':>7} {'W':>4} {'D':>4} {'L':>4} "
          f"{'plies':>6} {'adj':>4} {'freeMB':>7}", flush=True)
    try:
        for level, kind in levels:
            if kind == "elo":
                eng.configure({"Skill Level": 20, "UCI_LimitStrength": True,
                               "UCI_Elo": int(level)})
            else:
                eng.configure({"UCI_LimitStrength": False,
                               "Skill Level": int(level)})
            rng = random.Random(1234 + level)
            score = 0.0; w = d = l = 0; plies_tot = 0; adj = 0; played = 0
            for gi in range(a.games):
                if free_mb() < a.min_free_mb:
                    print(f"  stopping: only {free_mb():.0f} MB free", flush=True)
                    raise SystemExit(1)
                opening = openings[gi % len(openings)]
                bot_white = (gi % 2 == 0)
                board = chess.Board(); hist = []
                for u in opening:
                    board.push(chess.Move.from_uci(u)); hist.append(u)
                times = [server.sample_think_ms(i) for i in range(len(hist))]

                while not board.is_game_over(claim_draw=True) and len(hist) < a.max_plies:
                    if (board.turn == chess.WHITE) == bot_white:
                        choice, _ = server.think(hist, 0.0, times, None)
                        if choice is None:
                            break
                        board.push(chess.Move.from_uci(choice)); hist.append(choice)
                    else:
                        res = eng.play(board, chess.engine.Limit(time=a.sf_time))
                        if res.move is None:
                            break
                        board.push(res.move); hist.append(res.move.uci())
                    times.append(server.sample_think_ms(len(hist) - 1))

                outcome = board.outcome(claim_draw=True)
                if outcome is not None:
                    pt = 0.5 if outcome.winner is None else (
                        1.0 if (outcome.winner == chess.WHITE) == bot_white else 0.0)
                else:
                    # Ply cap. Adjudicate with a full-strength eval from the SAME
                    # engine -- a second engine per level is what made the old
                    # version 18 processes -- then restore the handicap.
                    adj += 1
                    eng.configure({"UCI_LimitStrength": False, "Skill Level": 20})
                    info = eng.analyse(board, chess.engine.Limit(depth=12))
                    cp = info["score"].white().score(mate_score=10000)
                    if kind == "elo":
                        eng.configure({"UCI_LimitStrength": True, "UCI_Elo": int(level)})
                    else:
                        eng.configure({"Skill Level": int(level)})
                    if cp is None: pt = 0.5
                    elif cp > 150: pt = 1.0 if bot_white else 0.0
                    elif cp < -150: pt = 0.0 if bot_white else 1.0
                    else: pt = 0.5
                score += pt; plies_tot += len(hist); played += 1
                w += pt == 1.0; d += pt == 0.5; l += pt == 0.0

            name = f"Skill {level}" if kind == "skill" else f"SF Elo {level}"
            row = {"level": level, "kind": kind, "name": name, "games": played,
                   "score": score, "rate": score / max(played, 1), "w": w, "d": d,
                   "l": l, "avg_plies": plies_tot / max(played, 1), "adjudicated": adj}
            rows.append(row)
            print(f"{name:>14} {row['rate']*100:>6.1f}% {w:>4} {d:>4} {l:>4} "
                  f"{row['avg_plies']:>6.0f} {adj:>4} {free_mb():>7.0f}", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    finally:
        eng.quit()
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
