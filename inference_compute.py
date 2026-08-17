"""Spend more compute at query time, without retraining anything.

Three knobs, all measured against the same shortlists so the comparison is
honest:

  subsets   Instead of chopping the visitor's games into fixed consecutive
            bundles, sample many random 5-game subsets and sum their scores.
            Same evidence, more views of it -- the analogue of self-consistency
            sampling.

  prf       Pseudo-relevance feedback. Take the top-k matches, blend their
            centroids back into the query, search again. A classic IR trick:
            if the top hits are mostly right, they sharpen the query; the risk
            is that a wrong top hit drags the query toward itself, so alpha
            stays small and the original query keeps most of the weight.

  both      prf on top of subset ensembling.

Costs one extra 5 ms search (prf) or a few extra encodes (subsets) -- against a
visitor who is sitting through a 60-second game anyway.

    python inference_compute.py --gallery play/gallery_2026.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from gallery_ctx import Bundles, embed_bundles
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import N_TIME_BINS
from verify import player_index


def ranks_of(sims, truth):
    return np.array([int((sims[i] > sims[i, truth[i]]).sum()) + 1
                     for i in range(len(truth))])


def report(name, rk, n_extra=""):
    print(f"  {name:<34}{(rk <= 1).mean():>8.3f}{(rk <= 10).mean():>8.3f}"
          f"{np.median(rk):>9.0f}{n_extra:>12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--gallery", default="play/gallery_2026.npz")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--queries", type=int, default=250)
    ap.add_argument("--games", type=int, default=10, help="games the visitor has")
    ap.add_argument("--subsets", type=int, default=8)
    ap.add_argument("--prf-k", type=int, default=5)
    ap.add_argument("--prf-alpha", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len)
    wr = ck["n_planes"] == 13
    model = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS, n_game_slots=slots,
                           elo_cond=bool(ck.get("elo_cond"))).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    g = np.load(args.gallery, allow_pickle=True)
    C = F.normalize(torch.from_numpy(g["centroids"].astype(np.float32)), dim=-1)
    names = [str(n).lower() for n in g["names"]]
    idx = {n: i for i, n in enumerate(names)}
    print(f"gallery {len(names):,} players", flush=True)

    shard_names = open(f"{args.shard}/players.txt", encoding="utf-8").read().split("\n")
    meta = np.load(f"{args.shard}/meta.npy", mmap_mode="r")
    clocks = np.memmap(f"{args.shard}/clocks.u16", dtype=np.uint16, mode="r")
    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
    ok = np.concatenate([np.asarray(clocks[np.asarray(meta["offset"], np.int64)]) != 0xFFFF] * 2)
    o = np.argsort(pid, kind="stable")
    pid, gid, seat, ok = pid[o], gid[o], seat[o], ok[o]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])

    rng = np.random.default_rng(args.seed)
    order = list(range(len(bnd) - 1)); rng.shuffle(order)
    picks, truth = [], []
    for i in order:
        if len(picks) >= args.queries:
            break
        sl = slice(bnd[i], bnd[i + 1]); m = ok[sl]
        g_, s_ = gid[sl][m], seat[sl][m]
        if len(g_) < args.games:
            continue
        p = int(pid[sl][0])
        if p >= len(shard_names):
            continue
        j = idx.get(shard_names[p].lower())
        if j is None:
            continue
        sel = rng.permutation(len(g_))[:args.games]
        picks.append([(int(g_[x]), int(s_[x])) for x in sel])
        truth.append(j)
    truth = np.array(truth)
    Q = len(picks)
    print(f"{Q} query players, {args.games} games each\n", flush=True)

    # --- baseline: consecutive bundles, the deployed behaviour ---------------
    base_b, base_owner = [], []
    for qi, gs in enumerate(picks):
        for c in range(0, len(gs) - slots + 1, slots):
            base_b.append(gs[c:c + slots]); base_owner.append(qi)
    # --- subsets: random 5-game views of the same games ----------------------
    sub_b, sub_owner = [], []
    for qi, gs in enumerate(picks):
        for _ in range(args.subsets):
            sel = rng.permutation(len(gs))[:slots]
            sub_b.append([gs[x] for x in sel]); sub_owner.append(qi)

    def scores(bundles, owner, label):
        E = embed_bundles(model, Bundles(args.shard, bundles, mlpg, wr), slots,
                          device, args.batch, args.workers, label)
        E = F.normalize(E.float(), dim=-1)
        S = torch.zeros(Q, C.shape[0])
        sims = E @ C.T
        S.index_add_(0, torch.tensor(owner), sims)
        return S

    S_base = scores(base_b, base_owner, "baseline")
    S_sub = scores(sub_b, sub_owner, f"subsets x{args.subsets}")

    def prf(S):
        """Blend the top-k centroids back into the query and search again."""
        top = S.topk(args.prf_k, dim=1).indices
        fb = F.normalize(C[top].mean(dim=1), dim=-1)          # (Q, D)
        # The original query keeps most of the weight: a wrong top hit would
        # otherwise pull the query onto itself and lock the error in.
        return S + args.prf_alpha * (fb @ C.T)

    print(f"{'method':<36}{'r@1':>8}{'r@10':>8}{'median':>9}{'cost':>12}")
    report("baseline (consecutive bundles)", ranks_of(S_base, truth),
           f"{len(base_b)//Q} enc")
    report(f"subsets x{args.subsets}", ranks_of(S_sub, truth), f"{args.subsets} enc")
    report("baseline + prf", ranks_of(prf(S_base), truth), "+1 search")
    report(f"subsets x{args.subsets} + prf", ranks_of(prf(S_sub), truth), "+1 search")

    # prf is sensitive to how much feedback is trusted; show the curve rather
    # than a single tuned number.
    print(f"\n  prf alpha sweep (on subsets x{args.subsets}):")
    for a in (0.0, 0.1, 0.25, 0.5, 1.0):
        args.prf_alpha = a
        rk = ranks_of(prf(S_sub), truth)
        print(f"    alpha {a:<5} r@1 {(rk<=1).mean():.3f}  r@10 {(rk<=10).mean():.3f}")
    print("\nINFERENCE_COMPUTE_DONE")


if __name__ == "__main__":
    main()
