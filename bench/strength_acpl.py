"""Where does our bot sit on the LICHESS rating scale?

The playoff answers "how does the bot do against Stockfish", but Stockfish's
UCI_Elo is not lichess bullet rating -- lichess ratings are inflated relative to
engine Elo and the two scales cannot be compared by eye. So use real players as
the ruler instead.

For each rating band, walk real 1+0 games and at every position measure two
things on the SAME position:

    the centipawn loss of the move the human actually played
    the centipawn loss of the move OUR BOT would have played

Matched positions matter: centipawn loss depends heavily on how sharp the
position is, so comparing our bot's average loss on its own games against
humans' average loss on theirs would mostly measure which games were wilder.
Here every comparison is on identical positions.

Evals chain: the position after the human's move is the next position in the
game, so walking a game costs one eval per ply plus one extra for the bot's
alternative.

The bot gets the game's REAL clock track through time_features -- it reads think
time at every ply and leans on it hard, so feeding zeros would measure a
different bot from the one that ships.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

REPO = "/Users/inteoryx/investigations/chess_tracker"
SF = "/opt/homebrew/bin/stockfish"


def free_mb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    head, *lines = out.splitlines()
    psz = int(head.split("page size of")[1].split("bytes")[0].strip())
    d = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1); v = v.strip().rstrip(".")
            if v.isdigit(): d[k.strip()] = int(v)
    return (d.get("Pages free", 0) + d.get("Pages inactive", 0)) * psz / (1 << 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default="1000,1200,1400,1600,1800,2000,2200,2400,2600")
    ap.add_argument("--games", type=int, default=8, help="games per band")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--first-ply", type=int, default=8)
    ap.add_argument("--last-ply", type=int, default=60)
    ap.add_argument("--decided-cp", type=int, default=400,
                    help="skip positions already this lopsided")
    ap.add_argument("--min-free-mb", type=float, default=600)
    ap.add_argument("--out", default="acpl.json")
    a = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/play")
    import chess, chess.engine, torch, server
    from bitboards import decode_move
    from timefeat import ms_used_per_ply

    meta = np.load(f"{REPO}/data/2026-06-big/meta.npy", allow_pickle=True)
    mv = np.memmap(f"{REPO}/data/2026-06-big/moves.u16", dtype=np.uint16, mode="r")
    ck = np.memmap(f"{REPO}/data/2026-06-big/clocks.u16", dtype=np.uint16, mode="r")

    server.load(f"{REPO}/ckpt/final/ctx5_pre.pt", "", "")
    torch.set_num_threads(1)
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 16})

    edges = [int(x) for x in a.bands.split(",")]
    bands = list(zip(edges[:-1], edges[1:]))
    rng = np.random.default_rng(7)
    rows = []
    print(f"free at start: {free_mb():.0f} MB\n")
    print(f"{'band':>12} {'gms':>5} {'plies':>6} "
          f"{'hACPL':>7} {'bACPL':>6} {'hMED':>7} {'bMED':>6} "
          f"{'h>100':>8} {'b>100':>7} {'agree':>7}", flush=True)

    def ev(board):
        """Eval in centipawns from WHITE's perspective."""
        info = eng.analyse(board, chess.engine.Limit(depth=a.depth))
        return info["score"].white().score(mate_score=2000)

    for lo, hi in bands:
        # Games where the side to move we sample is inside the band, and the
        # two players are close enough that the game is a normal one.
        okw = (meta["white_elo"] >= lo) & (meta["white_elo"] < hi) & (meta["nply"] > a.first_ply + 12)
        idx = np.flatnonzero(okw)
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        hcpl, bcpl, agree, tot, ngames = [], [], 0, 0, 0
        for gi in idx[: a.games]:
            if free_mb() < a.min_free_mb:
                print(f"  stopping: {free_mb():.0f} MB free"); raise SystemExit(1)
            g = meta[gi]
            n = int(g["nply"])
            codes = mv[g["offset"]: g["offset"] + n]
            clocks = np.asarray(ck[g["offset"]: g["offset"] + n], dtype=np.float64)
            # think() wants per-ply DURATIONS in ms, the same thing the browser
            # measures. Derived from the real clock trace so the bot sees the
            # time signal it actually leans on rather than zeros.
            durs = np.nan_to_num(
                ms_used_per_ply(clocks, int(g["tc_base"]), int(g["tc_inc"])),
                nan=0.0).tolist()
            board = chess.Board(); hist = []
            ok = True
            for t in range(n):
                try:
                    m = decode_move(int(codes[t]))
                except Exception:                               # noqa: BLE001
                    ok = False; break
                if m not in board.legal_moves:
                    ok = False; break
                # Only white's moves (band was selected on white_elo), only the
                # middlegame, only positions that are not already decided.
                if (a.first_ply <= t < min(n - 1, a.last_ply)
                        and board.turn == chess.WHITE):
                    e_before = ev(board)
                    if e_before is not None and abs(e_before) <= a.decided_cp:
                        bot_uci, _ = server.think(hist, 0.0, durs[:len(hist)], None)
                        board.push(m); e_h = ev(board); board.pop()
                        if bot_uci is not None:
                            bm = chess.Move.from_uci(bot_uci)
                            board.push(bm); e_b = ev(board); board.pop()
                        else:
                            e_b = None
                        if e_h is not None and e_b is not None:
                            hcpl.append(max(0, e_before - e_h))
                            bcpl.append(max(0, e_before - e_b))
                            agree += (bm == m); tot += 1
                board.push(m); hist.append(m.uci())
            if ok:
                ngames += 1
        if tot == 0:
            continue
        h, b = np.array(hcpl, float), np.array(bcpl, float)
        row = {"band": f"{lo}-{hi}", "lo": lo, "hi": hi, "games": ngames,
               "plies": tot,
               "human_acpl": float(h.mean()), "bot_acpl": float(b.mean()),
               "human_med": float(np.median(h)), "bot_med": float(np.median(b)),
               "human_blunder": float((h > 100).mean() * 100),
               "bot_blunder": float((b > 100).mean() * 100),
               "agree_pct": 100.0 * agree / tot}
        rows.append(row)
        print(f"{row['band']:>12} {ngames:>5} {tot:>6} "
              f"{row['human_acpl']:>7.0f} {row['bot_acpl']:>6.0f} "
              f"{row['human_med']:>7.0f} {row['bot_med']:>6.0f} "
              f"{row['human_blunder']:>7.0f}% {row['bot_blunder']:>6.0f}% "
              f"{row['agree_pct']:>6.1f}%", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)
    eng.quit()
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
