"""Does a bullet-trained identifier recognise a player from their BLITZ games?

The decisive cheap experiment for the multi-time-control direction. If a model
that has only ever seen 1+0 already ranks a player from their 3+0 games well
above chance, then supporting other time controls is a refinement of something
that partly works. If it is at chance, the retrain has to build the capability
from nothing and we have learned that for the price of an ingest.

Paired within player: the same person is queried once with bullet games (the
control, matching what the gallery was built from) and once with blitz games.
The difference between those two numbers is cross-time-control transfer; the
absolute level of either is inflated by the usual leakage, since both months
sit inside the gallery's own window.
"""
import argparse, io, json, os, random, subprocess, sys, time
from collections import defaultdict
import numpy as np

REPO = "/Users/inteoryx/investigations/chess_tracker"
SHARDS = {"bullet": f"{REPO}/data/2026-06-big",
          "blitz":  f"{REPO}/data/2026-06-blitz"}


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


class Shard:
    def __init__(self, path):
        self.meta = np.load(f"{path}/meta.npy", allow_pickle=True)
        self.names = [l.rstrip("\n") for l in
                      io.open(f"{path}/players.txt", encoding="utf-8")]
        self.mv = np.memmap(f"{path}/moves.u16", dtype=np.uint16, mode="r")
        self.ck = np.memmap(f"{path}/clocks.u16", dtype=np.uint16, mode="r")
        G = len(self.meta)
        self.pid = np.concatenate([self.meta["white_pid"], self.meta["black_pid"]])
        self.gi = np.concatenate([np.arange(G), np.arange(G)])
        self.wh = np.concatenate([np.ones(G, bool), np.zeros(G, bool)])
        keep = self.meta["nply"][self.gi] >= 16
        self.pid, self.gi, self.wh = self.pid[keep], self.gi[keep], self.wh[keep]
        self.by_name = defaultdict(list)
        for i in range(len(self.pid)):
            self.by_name[self.names[self.pid[i]].lower()].append(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=int, default=3)
    ap.add_argument("--players", type=int, default=400)
    ap.add_argument("--min-free-mb", type=float, default=500)
    ap.add_argument("--synth-clocks", action="store_true",
                    help="give every game the same synthetic clock track, so the "
                         "two arms differ only in their moves")
    ap.add_argument("--norm-clocks", action="store_true",
                    help="rescale think times by the time control before feeding "
                         "them, the normalisation a multi-TC model would learn")
    ap.add_argument("--out", default="xtc.json")
    a = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
    sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/play")
    import chess, torch, server
    from bitboards import decode_move
    from timefeat import ms_used_per_ply

    print(f"free at start: {free_mb():.0f} MB", flush=True)
    sh = {k: Shard(v) for k, v in SHARDS.items()}
    server.load(f"{REPO}/ckpt/final/ctx5_pre.pt",
                f"{REPO}/ckpt/final/ctx10_ft.pt",
                f"{REPO}/play/gallery_ctx10.npz")
    torch.set_num_threads(2)
    gal = server.MODEL["name2idx"]

    common = [n for n in sh["blitz"].by_name
              if n in gal
              and len(sh["blitz"].by_name[n]) >= a.bundle
              and len(sh["bullet"].by_name.get(n, [])) >= a.bundle]
    rng = random.Random(31); rng.shuffle(common)
    common = common[: a.players]
    print(f"{len(common)} players with >={a.bundle} games in BOTH time controls "
          f"and present in the gallery\n", flush=True)

    def make_game(s, ri):
        gi = int(s.gi[ri]); g = s.meta[gi]; n = int(g["nply"])
        hist, b = [], chess.Board()
        for c in s.mv[g["offset"]: g["offset"] + n]:
            try:
                m = decode_move(int(c))
            except Exception:                                  # noqa: BLE001
                break
            if m not in b.legal_moves:
                break
            hist.append(m.uci()); b.push(m)
        if len(hist) < 8:
            return None
        clocks = np.asarray(s.ck[g["offset"]: g["offset"] + n], dtype=np.float64)
        durs = np.nan_to_num(ms_used_per_ply(clocks, int(g["tc_base"]),
                                             int(g["tc_inc"])), nan=0.0)
        t = durs[:len(hist)]
        if a.synth_clocks:
            # identical track for every game of a given length, both arms
            np.random.seed(4242)
            t = np.array([float(server.sample_think_ms(i)) for i in range(len(hist))])
        elif a.norm_clocks:
            # Scale to what the same fraction of the clock would have been in a
            # 60 s game. This is the cheap stand-in for the feature change --
            # if it recovers most of the loss, normalising time is the fix.
            t = t * (60.0 / max(int(g["tc_base"]), 1))
        return {"history": hist, "human_white": bool(s.wh[ri]),
                "times": t.tolist(),
                "tc_base": int(g["tc_base"]), "tc_inc": int(g["tc_inc"])}

    res = {"bullet": {}, "blitz": {}}
    t0 = time.perf_counter()
    for i, nm in enumerate(common):
        if free_mb() < a.min_free_mb:
            print(f"  stopping: {free_mb():.0f} MB free"); break
        for arm in ("bullet", "blitz"):
            idx = sh[arm].by_name[nm][:]
            rng.shuffle(idx)
            games = [g for g in (make_game(sh[arm], j) for j in idx[: a.bundle * 3]) if g]
            games = games[: a.bundle]
            if len(games) < a.bundle:
                continue
            r = server.identify(games, target=nm)
            pr = (r or {}).get("probe") or {}
            if pr.get("rank"):
                res[arm][nm] = int(pr["rank"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(common)}  {time.perf_counter()-t0:.0f}s", flush=True)

    n_gal = server.MODEL["gal_n"]
    print(f"\n{'arm':>8} {'n':>5} {'r@1':>7} {'r@10':>7} {'r@100':>8} {'median rank':>12}")
    out = {"gallery": n_gal, "bundle": a.bundle}
    for arm in ("bullet", "blitz"):
        rk = np.array(list(res[arm].values()))
        if not len(rk):
            continue
        out[arm] = {"n": int(len(rk)), "r1": float((rk == 1).mean()),
                    "r10": float((rk <= 10).mean()), "r100": float((rk <= 100).mean()),
                    "median": float(np.median(rk)), "ranks": res[arm]}
        print(f"{arm:>8} {len(rk):>5} {out[arm]['r1']*100:>6.1f}% "
              f"{out[arm]['r10']*100:>6.1f}% {out[arm]['r100']*100:>7.1f}% "
              f"{np.median(rk):>12,.0f}")
    print(f"\nchance r@10 = {10/n_gal*100:.5f}%  (1 in {n_gal:,})")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
