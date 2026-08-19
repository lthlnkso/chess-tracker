"""Identification recall against the player's own rating, in 100-point bands.

Every band is sampled from real June 2026 games and queried against the
production 558,735 gallery, exactly as the demo would.

Read the SHAPE, not the level. These June games sit inside the gallery's Jan-Jun
window, so some of them helped build the centroid being queried and every
absolute number is inflated. Centroid depth is not a confound: the gallery caps
it at 60 games and the per-band mean only moves from 55.7 to 59.6, so a trend
across bands is about identifiability rather than how well each centroid was
built.
"""
import argparse, io, json, os, random, subprocess, sys, time
from collections import defaultdict
import numpy as np

REPO = "/Users/inteoryx/investigations/chess_tracker"


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
    ap.add_argument("--lo", type=int, default=900)
    ap.add_argument("--hi", type=int, default=2700)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--bundle", type=int, default=3)
    ap.add_argument("--per-band", type=int, default=120)
    ap.add_argument("--min-free-mb", type=float, default=500)
    ap.add_argument("--out", default="band_eval.json")
    a = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
    sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/play")
    import chess, torch, server
    from bitboards import decode_move
    from timefeat import ms_used_per_ply

    meta = np.load(f"{REPO}/data/2026-06-big/meta.npy", allow_pickle=True)
    names = [l.rstrip("\n") for l in
             io.open(f"{REPO}/data/2026-06-big/players.txt", encoding="utf-8")]
    mv = np.memmap(f"{REPO}/data/2026-06-big/moves.u16", dtype=np.uint16, mode="r")
    ck = np.memmap(f"{REPO}/data/2026-06-big/clocks.u16", dtype=np.uint16, mode="r")

    print(f"free at start: {free_mb():.0f} MB", flush=True)
    server.load(f"{REPO}/ckpt/final/ctx5_pre.pt",
                f"{REPO}/ckpt/final/ctx10_ft.pt",
                f"{REPO}/play/gallery_ctx10.npz")
    torch.set_num_threads(2)
    gal = server.MODEL["name2idx"]

    G = len(meta)
    rows_pid = np.concatenate([meta["white_pid"], meta["black_pid"]])
    rows_me = np.concatenate([meta["white_elo"], meta["black_elo"]]).astype(int)
    rows_gi = np.concatenate([np.arange(G), np.arange(G)])
    rows_wh = np.concatenate([np.ones(G, bool), np.zeros(G, bool)])
    keep = (rows_me > 0) & (meta["nply"][rows_gi] >= 16)
    rows_pid, rows_me = rows_pid[keep], rows_me[keep]
    rows_gi, rows_wh = rows_gi[keep], rows_wh[keep]

    by_player = defaultdict(list)
    for i in range(len(rows_pid)):
        by_player[int(rows_pid[i])].append(i)
    # a player's rating is the median across their own games, not per-game
    prate = {p: float(np.median(rows_me[idx])) for p, idx in by_player.items()}

    def make_game(ri):
        gi = int(rows_gi[ri]); g = meta[gi]; n = int(g["nply"])
        hist, b = [], chess.Board()
        for c in mv[g["offset"]: g["offset"] + n]:
            try:
                m = decode_move(int(c))
            except Exception:                                  # noqa: BLE001
                break
            if m not in b.legal_moves:
                break
            hist.append(m.uci()); b.push(m)
        if len(hist) < 8:
            return None
        clocks = np.asarray(ck[g["offset"]: g["offset"] + n], dtype=np.float64)
        durs = np.nan_to_num(ms_used_per_ply(clocks, int(g["tc_base"]),
                                             int(g["tc_inc"])), nan=0.0)
        return {"history": hist, "human_white": bool(rows_wh[ri]),
                "times": durs[:len(hist)].tolist()}

    rng = random.Random(23)
    out = []
    t0 = time.perf_counter()
    print(f"\n{'band':>11} {'n':>5} {'r@1':>7} {'r@10':>7} {'r@100':>7} "
          f"{'median rank':>12}", flush=True)
    for lo in range(a.lo, a.hi, a.width):
        hi = lo + a.width
        cand = [p for p, r in prate.items()
                if lo <= r < hi and len(by_player[p]) >= a.bundle
                and p < len(names) and names[p].lower() in gal]
        rng.shuffle(cand)
        ranks = []
        for p in cand:
            if len(ranks) >= a.per_band:
                break
            if free_mb() < a.min_free_mb:
                print(f"  stopping: {free_mb():.0f} MB free"); break
            idx = by_player[p][:]; rng.shuffle(idx)
            games = [g for g in (make_game(i) for i in idx[: a.bundle * 3]) if g]
            games = games[: a.bundle]
            if len(games) < a.bundle:
                continue
            r = server.identify(games, target=names[p])
            pr = (r or {}).get("probe") or {}
            if pr.get("rank"):
                ranks.append(int(pr["rank"]))
        if len(ranks) < 20:
            continue
        rk = np.array(ranks)
        row = {"lo": lo, "hi": hi, "n": len(rk),
               "r1": float((rk == 1).mean()), "r10": float((rk <= 10).mean()),
               "r100": float((rk <= 100).mean()), "median": float(np.median(rk)),
               "pool": len(cand)}
        out.append(row)
        print(f"  {lo:>4}-{hi:<5} {len(rk):>5} {row['r1']*100:>6.1f}% "
              f"{row['r10']*100:>6.1f}% {row['r100']*100:>6.1f}% "
              f"{row['median']:>12,.0f}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n{time.perf_counter()-t0:.0f}s")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
