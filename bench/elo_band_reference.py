"""Harness 1 -- can an engine be handicapped to play like each lichess band?

Needed because "Elo" is not one scale. Stockfish's UCI_Elo is not lichess bullet
rating, and neither is directly comparable to a policy net's move quality. The
common currency this project already uses is centipawn loss, so calibrate on it:
for each band, measure what a real player of that rating loses per move, then
find the engine handicap that loses the same amount ON THE SAME POSITIONS.

Matched positions are the whole point. Centipawn loss depends heavily on how
sharp a position is, and 2300 games are sharper than 1100 games, so measuring
the engine on a pooled position set and the humans on their own would compare
the engine's difficulty to the humans' difficulty rather than their skill.

Two handicap axes, because UCI_Elo bottoms out at 1320 and the weakest bands may
sit below it:
  * UCI_Elo   -- calibrated, but floored
  * node cap  -- a 1-node search is barely more than the raw eval, and it goes
                 as weak as we like. Skill Level is deliberately NOT used: it
                 keeps full search depth and only randomises move choice, which
                 measured non-monotonic against this bot (Skill 0 beat it harder
                 than Skill 3).
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
    for ln in lines:
        if ":" in ln:
            k, v = ln.split(":", 1); v = v.strip().rstrip(".")
            if v.isdigit(): d[k.strip()] = int(v)
    return (d.get("Pages free", 0) + d.get("Pages inactive", 0)) * psz / (1 << 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=1000)
    ap.add_argument("--hi", type=int, default=2400)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--positions", type=int, default=40, help="per band")
    ap.add_argument("--depth", type=int, default=10, help="judge depth")
    ap.add_argument("--first-ply", type=int, default=8)
    ap.add_argument("--last-ply", type=int, default=60)
    ap.add_argument("--decided-cp", type=int, default=400)
    ap.add_argument("--nodes", default="1,10,50,200,1000")
    ap.add_argument("--elos", default="1320,1600,2000,2400")
    ap.add_argument("--min-free-mb", type=float, default=600)
    ap.add_argument("--out", default="bench/elo_band_reference.json")
    a = ap.parse_args()

    sys.path.insert(0, REPO)
    import chess, chess.engine
    from bitboards import decode_move

    meta = np.load(f"{REPO}/data/2026-06-big/meta.npy", allow_pickle=True)
    mv = np.memmap(f"{REPO}/data/2026-06-big/moves.u16", dtype=np.uint16, mode="r")
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 16})

    def judge(board):
        eng.configure({"UCI_LimitStrength": False, "Skill Level": 20})
        i = eng.analyse(board, chess.engine.Limit(depth=a.depth))
        return i["score"].white().score(mate_score=2000)

    settings = ([("nodes", int(x)) for x in a.nodes.split(",") if x] +
                [("elo", int(x)) for x in a.elos.split(",") if x])

    def multipv_losses(board, k=16):
        """Every candidate move with what it costs, cheapest first.

        This is the handicap axis that actually reaches human bands. UCI_Elo and
        node caps both bottom out far above a 1000-rated player because they
        still choose with Stockfish's NNUE evaluation, which is superhuman even
        at one node. Picking the Nth-best move instead controls the error
        directly, so a reference opponent can be built to lose whatever a band
        loses rather than whatever the engine's floor happens to be.
        """
        eng.configure({"UCI_LimitStrength": False, "Skill Level": 20,
                       "MultiPV": k})
        try:
            info = eng.analyse(board, chess.engine.Limit(depth=a.depth), multipv=k)
        finally:
            eng.configure({"MultiPV": 1})
        best = None; out = []
        for e in info:
            pv = e.get("pv")
            if not pv: continue
            sc = e["score"].pov(board.turn).score(mate_score=2000)
            if sc is None: continue
            if best is None: best = sc
            out.append((pv[0], max(0, best - sc)))
        return out

    def play(board, kind, val):
        if kind == "elo":
            eng.configure({"UCI_LimitStrength": True, "UCI_Elo": val})
            lim = chess.engine.Limit(time=0.05)
        else:
            eng.configure({"UCI_LimitStrength": False})
            lim = chess.engine.Limit(nodes=val)
        return eng.play(board, lim).move

    rng = np.random.default_rng(11)
    rows = []
    band_losses = {}          # band -> the human loss distribution, for sampling
    hdr = f"{'band':>11} {'pos':>5} {'human':>7} " + " ".join(
        f"{('n=' + str(v)) if k == 'nodes' else ('E' + str(v)):>7}" for k, v in settings)
    print(f"centipawn loss on the SAME positions\n\n{hdr}", flush=True)
    print(f"free at start: {free_mb():.0f} MB\n", flush=True)

    for lo in range(a.lo, a.hi, a.width):
        hi = lo + a.width
        sel = np.flatnonzero((meta["white_elo"] >= lo) & (meta["white_elo"] < hi) &
                             (meta["nply"] > a.first_ply + 12))
        if len(sel) == 0: continue
        rng.shuffle(sel)
        hcpl = []; scpl = {s: [] for s in settings}
        for gi in sel:
            if len(hcpl) >= a.positions: break
            if free_mb() < a.min_free_mb:
                print(f"  stopping: {free_mb():.0f} MB free"); raise SystemExit(1)
            g = meta[gi]; n = int(g["nply"])
            board = chess.Board(); ok = True
            for t in range(n):
                if len(hcpl) >= a.positions: break
                try: m = decode_move(int(mv[g["offset"] + t]))
                except Exception: ok = False; break
                if m not in board.legal_moves: ok = False; break
                if (a.first_ply <= t < min(n - 1, a.last_ply)
                        and board.turn == chess.WHITE):
                    e0 = judge(board)
                    if e0 is not None and abs(e0) <= a.decided_cp:
                        board.push(m); eh = judge(board); board.pop()
                        if eh is not None:
                            got = {}
                            for s in settings:
                                bm = play(board, *s)
                                if bm is None: continue
                                board.push(bm); eb = judge(board); board.pop()
                                if eb is not None: got[s] = max(0, e0 - eb)
                            if len(got) == len(settings):
                                hcpl.append(max(0, e0 - eh))
                                for s in settings: scpl[s].append(got[s])
                board.push(m)
            if not ok: continue
        if len(hcpl) < 10: continue
        h = float(np.mean(hcpl))
        band_losses[f"{lo}-{hi}"] = [int(x) for x in hcpl]
        cells = {f"{k}={v}": float(np.mean(scpl[(k, v)])) for k, v in settings}
        best = min(cells, key=lambda c: abs(cells[c] - h))
        rows.append({"band": f"{lo}-{hi}", "lo": lo, "n": len(hcpl),
                     "human_acpl": h, "engine": cells, "closest": best})
        print(f"  {lo:>4}-{hi:<5} {len(hcpl):>5} {h:>7.0f} "
              + " ".join(f"{cells[f'{k}={v}']:>7.0f}" for k, v in settings)
              + f"   -> {best}", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)
    eng.quit()
    json.dump(rows, open(a.out, "w"), indent=1)
    print("\nmapping (band -> closest engine handicap by centipawn loss):")
    for r in rows:
        print(f"  {r['band']:>11} -> {r['closest']}")


if __name__ == "__main__":
    main()
