"""Does identification of a strong player survive a weak opponent?

The worry: our bot is far weaker than a strong visitor, so their games against
it look different from their games against peers -- and the gallery centroid was
built from the peer games. If that shift breaks identification, strong players
are unidentifiable no matter how good the model is.

Design: PAIRED, within player. For each strong player who has enough June 2026
games in more than one opponent-gap bin, query the production gallery once per
bin using the same number of games, and compare the rank of that same player.

Paired matters because of leakage: these June games are inside the gallery's
Jan-Jun window, so some of them helped build the very centroid being queried.
That inflates absolute recall in every bin. It does not explain a DIFFERENCE
between bins for one player, which is what this measures.

Gap bins are chosen to bracket what our bot actually presents. It scores about
10% against a 2200, which is an implied gap near 380 -- so the 300+ bins are
the analogue, and the peer bin is the control.
"""
import argparse, io, json, os, random, subprocess, sys, time
from collections import Counter, defaultdict
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
    ap.add_argument("--min-elo", type=int, default=2000)
    ap.add_argument("--bundle", type=int, default=3)
    ap.add_argument("--bins", default="peer:-100:100,mod:150:300,big:300:9999")
    ap.add_argument("--max-players", type=int, default=120)
    ap.add_argument("--min-free-mb", type=float, default=500)
    ap.add_argument("--synth-clocks", action="store_true",
                    help="give EVERY game the same synthetic clock track, so the "
                         "arms differ only in their moves. Zeroing the clocks "
                         "instead drives both arms to the floor and "
                         "discriminates nothing.")
    ap.add_argument("--no-clocks", action="store_true",
                    help="zero the clock features in BOTH arms, to see whether "
                         "the peer-vs-gap difference is carried by the clock "
                         "track or by the moves themselves")
    ap.add_argument("--out", default="gap_eval.json")
    a = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
    sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/play")
    import chess, torch, server
    from bitboards import decode_move
    from timefeat import ms_used_per_ply

    bins = []
    for spec in a.bins.split(","):
        nm, lo, hi = spec.split(":")
        bins.append((nm, int(lo), int(hi)))

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
    gal_lower = server.MODEL["name2idx"]

    # (player, game) rows, both colours
    G = len(meta)
    rows_pid = np.concatenate([meta["white_pid"], meta["black_pid"]])
    rows_me = np.concatenate([meta["white_elo"], meta["black_elo"]]).astype(int)
    rows_op = np.concatenate([meta["black_elo"], meta["white_elo"]]).astype(int)
    rows_gi = np.concatenate([np.arange(G), np.arange(G)])
    rows_wh = np.concatenate([np.ones(G, bool), np.zeros(G, bool)])
    keep = (rows_me > 0) & (rows_op > 0) & (meta["nply"][rows_gi] >= 16)
    rows_pid, rows_me, rows_op = rows_pid[keep], rows_me[keep], rows_op[keep]
    rows_gi, rows_wh = rows_gi[keep], rows_wh[keep]
    gap = rows_me - rows_op

    strong = rows_me >= a.min_elo
    per_bin = {}
    for nm, lo, hi in bins:
        sel = strong & ((np.abs(gap) <= hi) if nm == "peer" else ((gap >= lo) & (gap < hi)))
        d = defaultdict(list)
        for i in np.flatnonzero(sel):
            d[int(rows_pid[i])].append(i)
        per_bin[nm] = d

    first = bins[0][0]
    cand = [p for p in per_bin[first]
            if all(len(per_bin[nm].get(p, [])) >= a.bundle for nm, _, _ in bins)
            and p < len(names) and names[p].lower() in gal_lower]
    rng = random.Random(17); rng.shuffle(cand)
    cand = cand[: a.max_players]
    print(f"{len(cand)} strong players with >={a.bundle} games in every bin\n", flush=True)

    def make_game(ri):
        gi = int(rows_gi[ri]); g = meta[gi]; n = int(g["nply"])
        codes = mv[g["offset"]: g["offset"] + n]
        hist, b = [], chess.Board()
        for c in codes:
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
        if a.no_clocks:
            t = []
        elif a.synth_clocks:
            # Same seed for every game, so a game of length L always gets the
            # identical track regardless of which arm it came from.
            np.random.seed(4242)
            t = [float(server.sample_think_ms(i)) for i in range(len(hist))]
        else:
            t = durs[:len(hist)].tolist()
        return {"history": hist, "human_white": bool(rows_wh[ri]), "times": t}

    res = {nm: [] for nm, _, _ in bins}
    t0 = time.perf_counter()
    for k, pid in enumerate(cand):
        if free_mb() < a.min_free_mb:
            print(f"  stopping: {free_mb():.0f} MB free"); break
        nm_player = names[pid]
        for nm, _, _ in bins:
            idxs = per_bin[nm][pid][:]
            rng.shuffle(idxs)
            games = [g for g in (make_game(i) for i in idxs[: a.bundle * 2]) if g]
            games = games[: a.bundle]
            if len(games) < a.bundle:
                continue
            r = server.identify(games, target=nm_player)
            pr = (r or {}).get("probe") or {}
            if pr.get("rank"):
                res[nm].append({"player": nm_player, "rank": int(pr["rank"])})
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(cand)} players, {time.perf_counter()-t0:.0f}s",
                  flush=True)

    print(f"\n{'bin':>8} {'n':>5} {'r@1':>7} {'r@10':>7} {'r@100':>7} {'median rank':>12}")
    out = {}
    for nm, _, _ in bins:
        rk = np.array([x["rank"] for x in res[nm]])
        if len(rk) == 0:
            continue
        out[nm] = {"n": len(rk), "r1": float((rk == 1).mean()),
                   "r10": float((rk <= 10).mean()), "r100": float((rk <= 100).mean()),
                   "median": float(np.median(rk)),
                   "ranks": {x["player"]: x["rank"] for x in res[nm]}}
        print(f"{nm:>8} {len(rk):>5} {out[nm]['r1']*100:>6.1f}% "
              f"{out[nm]['r10']*100:>6.1f}% {out[nm]['r100']*100:>6.1f}% "
              f"{np.median(rk):>12,.0f}")

    # Paired: same player, peer bin vs each other bin.
    base = out.get(first, {}).get("ranks", {})
    print(f"\npaired against '{first}' (same player, same bundle size):")
    for nm, _, _ in bins[1:]:
        rr = out.get(nm, {}).get("ranks", {})
        common = [p for p in base if p in rr]
        if not common:
            continue
        b10 = np.array([base[p] <= 10 for p in common])
        o10 = np.array([rr[p] <= 10 for p in common])
        worse = sum(1 for p in common if rr[p] > base[p])
        better = sum(1 for p in common if rr[p] < base[p])
        print(f"  {nm:>6}: n={len(common):>3}  top10 {b10.mean()*100:5.1f}% -> "
              f"{o10.mean()*100:5.1f}%   rank worse for {worse}, better for {better}")
        out.setdefault("paired", {})[nm] = {
            "n": len(common), "peer_top10": float(b10.mean()),
            "bin_top10": float(o10.mean()), "worse": worse, "better": better}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
