"""Gallery-size curve for a multi-game-context model, up to the whole player base.

identify_eval_ctx.py answers "how good is this model on a 20k gallery". It cannot
answer "how good on 200k", for two reasons this script fixes:

  1. It restricts the gallery to held-out players. Only the QUERIES have to be
     held out -- a distractor is just a wrong answer, and the deployed gallery
     really is everyone. Restricting both caps the gallery at the size of the
     test split.
  2. It embeds one player at a time (`e[0]`) and encodes boards square by square,
     so a 200k gallery is millions of batch-1 forward passes behind a CPU-bound
     encoder. Measured: ~1.8h for 20k. Batched + fastboard, the same work is
     minutes.

Like sweep_gallery.py, every smaller gallery is then derived EXACTLY rather than
re-measured: for a query beaten by `r` of the `M-1` distractors, a random
sub-gallery of size N keeps a hypergeometric number of them, so recall@k(N) is a
closed form. One embedding pass gives the whole curve.

    python gallery_ctx.py --ckpt ckpt/final/ctx5_ft2.pt --shard data/mt/2026-01 \
        --out ctx5_curve.json --ks 5 --gallery-players 200000 --query-players 5000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import chess
from bitboards import decode_move, n_planes_compact
from fastboard import N_BB, snapshot_bb, rights_bb, encode_batch
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import time_features, N_TIME_FEATS, N_TIME_BINS
from sweep_gallery import recall_at_size

KS_REPORT = (1, 10, 100)


class Bundles(Dataset):
    """One item = one player's k-game bundle, as raw bitboards.

    Encoding is deferred to collate so a whole batch goes through one
    unpackbits, and the per-square Python path is gone entirely.
    """

    def __init__(self, shard, bundles, mlpg, with_rights):
        self.meta = np.load(os.path.join(shard, "meta.npy"), mmap_mode="r")
        self.moves = np.memmap(os.path.join(shard, "moves.u16"), dtype=np.uint16, mode="r")
        self.clocks = np.memmap(os.path.join(shard, "clocks.u16"), dtype=np.uint16, mode="r")
        self.bundles = bundles          # list of [(game_idx, seat), ...]
        self.mlpg = mlpg
        self.with_rights = with_rights
        self.n_planes = n_planes_compact(with_rights)

    def __len__(self):
        return len(self.bundles)

    def _game(self, gi, seat):
        row = self.meta[gi]
        o, n = int(row["offset"]), int(row["nply"])
        codes = np.asarray(self.moves[o:o + n])
        clk = np.asarray(self.clocks[o:o + n])
        T = min(len(codes), self.mlpg)
        pov = chess.WHITE if seat == 0 else chess.BLACK
        pos = np.zeros((T, N_BB), np.uint64)
        rgt = np.zeros((T, 5), np.int16); rgt[:, 4] = -1
        b = chess.Board()
        for t in range(T):
            pos[t] = snapshot_bb(b, pov)
            if self.with_rights:
                rgt[t] = rights_bb(b, pov)
            b.push(decode_move(int(codes[t])))
        if pov == chess.BLACK:                 # POV mirror, per game
            pos = pos.byteswap()
            e = rgt[:, 4]; rgt[:, 4] = np.where(e >= 0, e ^ 56, e)
        fe, _, _ = time_features(clk, int(row["tc_base"]), int(row["tc_inc"]))
        mt = np.zeros(T, bool); mt[seat::2] = True
        return pos, rgt, fe[:T], mt

    def __getitem__(self, i):
        games = [self._game(gi, s) for gi, s in self.bundles[i]]
        return games


class Collate:
    """A class, not a closure: DataLoader workers pickle the collate_fn, and a
    nested function is unpicklable."""

    def __init__(self, n_planes, n_slots, with_rights):
        self.n_planes, self.n_slots, self.with_rights = n_planes, n_slots, with_rights

    def __call__(self, batch):
        n_planes, n_slots, with_rights = self.n_planes, self.n_slots, self.with_rights
        B = len(batch)
        T = max(sum(g[0].shape[0] for g in games) for games in batch)
        snaps = np.concatenate([g[0] for games in batch for g in games])
        rgts = np.concatenate([g[1] for games in batch for g in games]) if with_rights else None
        enc = encode_batch(snaps, n_planes, rgts)
        planes = np.zeros((B, T, n_planes, 8, 8), np.uint8)
        extra = np.zeros((B, T, N_TIME_FEATS), np.float32)
        pad = np.ones((B, T), bool)
        mine = np.zeros((B, T), bool)
        slot = np.zeros((B, T), np.int64)
        ppos = np.zeros((B, T), np.int64)
        at = 0
        for i, games in enumerate(batch):
            o = 0
            for s, (pos, _r, fe, mt) in enumerate(games):
                t = pos.shape[0]
                planes[i, o:o + t] = enc[at:at + t]
                extra[i, o:o + t] = fe
                mine[i, o:o + t] = mt
                slot[i, o:o + t] = min(s, n_slots - 1)
                ppos[i, o:o + t] = np.arange(t)   # ply index restarts per game
                at += t; o += t
            pad[i, :o] = False
        return (torch.from_numpy(planes), torch.from_numpy(extra),
                torch.from_numpy(pad), torch.from_numpy(mine),
                torch.from_numpy(slot), torch.from_numpy(ppos))


@torch.no_grad()
def embed_bundles(model, ds, n_slots, device, batch, workers, label=""):
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    collate_fn=Collate(ds.n_planes, n_slots, ds.with_rights),
                    pin_memory=True)
    out, t0 = [], time.time()
    for i, t in enumerate(dl):
        t = [x.to(device, non_blocking=True) for x in t]
        e, _ = model.embed(t[0], t[1], t[2], t[3], t[4], t[5])
        out.append(e.float().cpu())
        if i % 200 == 0:
            done = (i + 1) * batch
            print(f"    {label} {done:,}/{len(ds):,} "
                  f"({done / max(time.time() - t0, 1e-9):.0f}/s)", flush=True)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", required=True,
                    help="shard the GALLERY centroids are built from")
    ap.add_argument("--query-shard", default="",
                    help="shard the QUERIES come from; defaults to --shard. Set it "
                         "to a different time control to measure cross-control "
                         "identification -- a visitor hands us bullet games and "
                         "their history is blitz.")
    ap.add_argument("--allow-contaminated", action="store_true",
                    help="proceed when held-out status cannot be verified. Required "
                         "rather than assumed, because the failure is silent.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="5")
    ap.add_argument("--gallery-games", type=int, default=64,
                    help="CAP on games per centroid, not a requirement. The old "
                         "default of 12 was undocumented and cost ~11%% top-10 at "
                         "k=5: deployment has a player's whole history, so a "
                         "12-game centroid measures a handicap we do not have. "
                         "Requiring N would also shrink the pool to hyper-active "
                         "players and flatter the result, hence a cap.")
    ap.add_argument("--min-gallery-games", type=int, default=8,
                    help="floor, so centroids are not degenerately thin")
    ap.add_argument("--gallery-players", type=int, default=200_000)
    ap.add_argument("--query-players", type=int, default=5_000)
    ap.add_argument("--sizes", default="1000,5000,10000,20000,50000,100000,200000")
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--colour-split", action="store_true",
                    help="also build separate white/black centroids and score "
                         "colour-matched queries fused by SCORE SUM. Fusion, not "
                         "top-N intersection: intersection is a hard AND on list "
                         "membership and measured 0.815 vs 0.895 against fusion "
                         "on a 3k gallery, because it discards the ranking inside "
                         "each list.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # Clamp here, not just in the launcher. This job is CPU-bound -- the GPU
    # waits on bundle assembly -- so oversubscribing the loader is not free the
    # way it is during training. nproc reports the HOST's cores inside a RunPod
    # container, so a launcher that trusts it asks for 24 workers on ~8 real
    # ones: measured 2026-08-17, the k=10 gallery ran at 28 bundles/s against
    # k=5's 503/s with the GPU pinned at 0%. Clamping at the point of use means
    # a stale launcher script on a long-running pod cannot reintroduce it.
    from cpuquota import cpu_quota
    if args.workers > cpu_quota():
        print(f"workers {args.workers} -> {cpu_quota()} (cgroup quota)", flush=True)
        args.workers = cpu_quota()
    ks = [int(x) for x in args.ks.split(",")]
    sizes = [int(x) for x in args.sizes.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    n_slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len)
    wr = ck["n_planes"] == 13
    model = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS, n_game_slots=n_slots).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"step {ck['step']}, {n_slots} slots, {mlpg} plies/game, loss {ck.get('loss')}",
          flush=True)

    q_shard = args.query_shard or args.shard
    cross = q_shard != args.shard

    def load_shard(path, need):
        """-> {username: (gid array, seat array)} for players with enough games.

        Keyed by NAME, not pid. Player ids are per-shard row numbers, so the same
        person is a different integer in every shard and matching on pid across
        two of them silently pairs strangers.
        """
        meta = np.load(os.path.join(path, "meta.npy"), mmap_mode="r")
        clocks = np.memmap(os.path.join(path, "clocks.u16"), dtype=np.uint16, mode="r")
        with open(os.path.join(path, "players.txt"), encoding="utf-8") as f:
            names = [ln.rstrip("\n") for ln in f]
        pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
        gid = np.concatenate([np.arange(len(meta))] * 2)
        seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
        ok = np.concatenate([np.asarray(clocks[np.asarray(meta["offset"], np.int64)]) != 0xFFFF] * 2)
        order = np.argsort(pid, kind="stable")
        pid, gid, seat, ok = pid[order], gid[order], seat[order], ok[order]
        bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
        out = {}
        for i in range(len(bnd) - 1):
            sl = slice(bnd[i], bnd[i + 1]); m = ok[sl]
            if m.sum() >= need:
                j = int(pid[sl][0])
                if j < len(names):
                    out[names[j].lower()] = (gid[sl][m], seat[sl][m])
        return out

    need_gal = args.min_gallery_games + (0 if cross else max(ks))
    gal_by_name = load_shard(args.shard, need_gal)
    print(f"{len(gal_by_name):,} players with >= {need_gal} clocked games in "
          f"{os.path.basename(args.shard)}", flush=True)
    if cross:
        qry_by_name = load_shard(q_shard, max(ks))
        print(f"{len(qry_by_name):,} players with >= {max(ks)} clocked games in "
              f"{os.path.basename(q_shard)}", flush=True)
        eligible = [n for n in gal_by_name if n in qry_by_name]
        print(f"{len(eligible):,} players present in BOTH shards", flush=True)
    else:
        qry_by_name = gal_by_name
        eligible = list(gal_by_name)

    # Held-out status. test_pids index the player table of the shard the model was
    # TRAINED on, so they are only meaningful against that same shard. Applying
    # them to a different one selects an arbitrary integer subset and prints a
    # reassuring number that means nothing -- which is exactly what happened on
    # 2026-08-19 and produced an inflated 94% that had to be retracted.
    tp = ck.get("test_pids")
    trained_shard = ck.get("shard") or ck.get("shards")
    can_verify = tp is not None and trained_shard and str(trained_shard) in (args.shard, q_shard)
    if not can_verify:
        msg = (f"cannot verify held-out status: checkpoint's test_pids index "
               f"{trained_shard or 'an unrecorded shard'}, queries come from {q_shard}")
        if not args.allow_contaminated:
            raise SystemExit(msg + "\n  pass --allow-contaminated to proceed; "
                             "absolute recall will be an upper bound, and only "
                             "comparisons between arms measured the same way hold")
        print(f"WARNING: {msg}", flush=True)
        print("WARNING: absolute recall below is an UPPER BOUND, not a measurement",
              flush=True)

    rng = np.random.default_rng(args.seed)
    eligible.sort()
    n_q = min(args.query_players, len(eligible))
    qsel = [eligible[i] for i in rng.choice(len(eligible), n_q, replace=False)]
    qset = set(qsel)
    # Distractors need not be held out -- they are only ever wrong answers, and
    # the deployed gallery is everyone. This is what lifts the ceiling from the
    # test-split size to the whole player base.
    dpool = [n for n in gal_by_name if n not in qset]
    n_d = min(args.gallery_players - n_q, len(dpool))
    dsel = [dpool[i] for i in rng.choice(len(dpool), n_d, replace=False)]
    gal_players = [(n, *gal_by_name[n]) for n in qsel + dsel]
    M = len(gal_players)
    sizes = [s for s in sizes if s <= M]
    print(f"gallery {M:,} ({n_q:,} queries + {n_d:,} distractors)"
          + (f" | queries from {os.path.basename(q_shard)}" if cross else ""), flush=True)

    res = {"ckpt": args.ckpt, "gallery_built": M, "query_players": n_q,
           "gallery_games": args.gallery_games, "sizes": sizes, "pools": {}}

    for k in ks:
        # Gallery: non-overlapping k-game chunks per player, averaged, matching
        # identify_eval_ctx's `matched` mode so query and gallery are the same
        # kind of object.
        gal_bundles, gal_owner = [], []
        qry_bundles, cent_sizes = [], []
        for idx, (nm, g, s) in enumerate(gal_players):
            perm = rng.permutation(len(g))
            # Use as many games as the player HAS, capped -- a player with 20
            # games gets a 20-game centroid, which is exactly what deployment
            # does. When the query comes from the SAME shard, reserve k games for
            # it so the two never overlap; across shards there is nothing to
            # reserve, since no game can appear on both sides.
            n_gal = min(args.gallery_games, len(g) - (0 if cross else k))
            gal = perm[:n_gal]
            chunks = [gal[j:j + k] for j in range(0, len(gal) - k + 1, k)] or [gal[:k]]
            cent_sizes.append(len(chunks) * k)
            for c in chunks:
                gal_bundles.append([(int(g[j]), int(s[j])) for j in c])
                gal_owner.append(idx)
            if idx < n_q:                       # queries are the first n_q entries
                if cross:
                    qg, qs = qry_by_name[nm]
                    qp = rng.permutation(len(qg))[:k]
                    qry_bundles.append([(int(qg[j]), int(qs[j])) for j in qp])
                else:
                    q = perm[n_gal:n_gal + k]
                    qry_bundles.append([(int(g[j]), int(s[j])) for j in q])

        cs = np.asarray(cent_sizes)
        print(f"  k={k}: {len(gal_bundles):,} gallery bundles, {len(qry_bundles):,} queries "
              f"| centroid games: mean {cs.mean():.1f} median {np.median(cs):.0f} "
              f"min {cs.min()} max {cs.max()}", flush=True)
        res.setdefault("centroid_games_stats", {})[str(k)] = {
            "mean": float(cs.mean()), "median": float(np.median(cs)),
            "min": int(cs.min()), "max": int(cs.max())}
        GE = embed_bundles(model, Bundles(args.shard, gal_bundles, mlpg, wr),
                           n_slots, device, args.batch, args.workers, f"k{k} gallery")
        QE = embed_bundles(model, Bundles(q_shard, qry_bundles, mlpg, wr),
                           n_slots, device, args.batch, args.workers, f"k{k} query")
        owner = torch.tensor(gal_owner)
        C = torch.zeros(M, GE.shape[1])
        C.index_add_(0, owner, GE)
        C = C / C.norm(dim=1, keepdim=True).clamp(min=1e-8)
        Q = QE / QE.norm(dim=1, keepdim=True).clamp(min=1e-8)

        C_d, Q_d = C.to(device), Q.to(device)
        beat = []
        for s0 in range(0, len(Q_d), 1024):
            q = Q_d[s0:s0 + 1024]
            sim = q @ C_d.T
            true = sim.gather(1, torch.arange(s0, s0 + len(q), device=device)[:, None])
            beat.append((sim > true).sum(1).cpu())
        beat = torch.cat(beat).numpy()
        entry = {"direct_at_full_gallery": {
                    f"recall@{j}": float((beat < j).mean()) for j in KS_REPORT},
                 "median_rank": float(np.median(beat)), "by_gallery_size": {}}
        for N in sizes:
            entry["by_gallery_size"][str(N)] = {
                f"recall@{j}": recall_at_size(beat, M, N, j) for j in KS_REPORT}
        res["pools"][str(k)] = entry
        print(f"  k={k} @ {M:,}: " + "  ".join(
            f"r@{j} {entry['direct_at_full_gallery'][f'recall@{j}']:.4f}" for j in KS_REPORT),
            flush=True)

    if args.colour_split:
        k = max(ks)
        # Same gallery games as the combined arm -- only the GROUPING differs,
        # so any gap is the split itself and not a different sample.
        wc, bc, owner_w, owner_b = [], [], [], []
        qw_b, qb_b, qmix_b, keep = [], [], [], []
        for idx, (nm, g, s_) in enumerate(gal_players):
            perm = rng.permutation(len(g))
            w = [int(j) for j in perm if s_[j] == 0]
            b = [int(j) for j in perm if s_[j] == 1]
            if len(w) < 2 * k or len(b) < 2 * k:
                continue                       # needs query + centroid of both
            keep.append(idx)
            qw, qb = w[:k], b[:k]
            gw, gb = w[k:k + args.gallery_games], b[k:k + args.gallery_games]
            for j in range(0, len(gw) - k + 1, k):
                wc.append([(int(g[x]), int(s_[x])) for x in gw[j:j + k]]); owner_w.append(len(keep) - 1)
            for j in range(0, len(gb) - k + 1, k):
                bc.append([(int(g[x]), int(s_[x])) for x in gb[j:j + k]]); owner_b.append(len(keep) - 1)
            if idx < n_q:
                qw_b.append([(int(g[x]), int(s_[x])) for x in qw])
                qb_b.append([(int(g[x]), int(s_[x])) for x in qb])
                mix = (qw[:k // 2 + k % 2] + qb[:k // 2])[:k]
                qmix_b.append([(int(g[x]), int(s_[x])) for x in mix])
        Mc = len(keep)
        nq2 = len(qw_b)
        print(f"  colour-split: {Mc:,} of {M:,} players have >= {2*k} games of BOTH "
              f"colours | {nq2:,} queries", flush=True)

        def cent(bundles, owner, n):
            E = embed_bundles(model, Bundles(args.shard, bundles, mlpg, wr),
                              n_slots, device, args.batch, args.workers, "split")
            C = torch.zeros(n, E.shape[1]); C.index_add_(0, torch.tensor(owner), E)
            return C / C.norm(dim=1, keepdim=True).clamp(min=1e-8)

        CW, CB = cent(wc, owner_w, Mc), cent(bc, owner_b, Mc)
        QW = embed_bundles(model, Bundles(args.shard, qw_b, mlpg, wr), n_slots,
                           device, args.batch, args.workers, "qw")
        QB = embed_bundles(model, Bundles(args.shard, qb_b, mlpg, wr), n_slots,
                           device, args.batch, args.workers, "qb")
        QM = embed_bundles(model, Bundles(args.shard, qmix_b, mlpg, wr), n_slots,
                           device, args.batch, args.workers, "qmix")
        nrm = lambda X: X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)
        QW, QB, QM = nrm(QW), nrm(QB), nrm(QM)
        # combined centroid restricted to the same kept players, for a fair base
        CC = C[torch.tensor(keep)]
        CC = CC / CC.norm(dim=1, keepdim=True).clamp(min=1e-8)

        out = {}
        for name, Q, G in (("combined", QM, [CC]), ("split_fused", (QW, QB), [CW, CB])):
            beat = []
            for s0 in range(0, nq2, 512):
                if name == "combined":
                    sim = Q[s0:s0 + 512].to(device) @ G[0].to(device).T
                else:
                    sim = (Q[0][s0:s0 + 512].to(device) @ G[0].to(device).T +
                           Q[1][s0:s0 + 512].to(device) @ G[1].to(device).T)
                tr = sim.gather(1, torch.arange(s0, s0 + sim.shape[0], device=device)[:, None])
                beat.append((sim > tr).sum(1).cpu())
            beat = torch.cat(beat).numpy()
            out[name] = {"direct": {f"recall@{j}": float((beat < j).mean()) for j in KS_REPORT},
                         "by_gallery_size": {str(N): {f"recall@{j}": recall_at_size(beat, Mc, N, j)
                                                      for j in KS_REPORT}
                                             for N in sizes if N <= Mc}}
            print(f"  {name} ({k if name=='combined' else 2*k} games) @ {Mc:,}: " +
                  "  ".join(f"r@{j} {out[name]['direct'][f'recall@{j}']:.4f}" for j in KS_REPORT),
                  flush=True)
        res["colour_split"] = {"players": Mc, "queries": nq2, "k": k, "arms": out}

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("GALLERY_CTX_DONE", flush=True)


if __name__ == "__main__":
    main()
