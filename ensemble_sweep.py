"""Does ensembling independently-trained models actually help identification?

Ensembling only pays if the models' *errors* decorrelate. That is an empirical
question, and it is much cheaper to answer with checkpoints we already have than
to fund a fleet of new ones first.

Combination rule: each model contributes an L2-normalised embedding, and scores
are averaged across models. Note that averaging cosine similarities is exactly
equivalent to concatenating the unit embeddings and renormalising --

    cos([e_1..e_M]/sqrt(M), [g_1..g_M]/sqrt(M)) = (1/M) * sum_i cos(e_i, g_i)

-- so "concatenate the embeddings" and "average the scores" are the same
experiment, and only one needs running. Reciprocal-rank fusion is reported too
because it is genuinely different: it discards score calibration and uses only
each model's ordering, which helps when models disagree about scale.

Every query player is held out by EVERY member. Models trained with different
splits have different held-out sets, so the usable query pool is their
intersection; anything else would let one member score players it memorised.

    python ensemble_sweep.py --ckpts a.pt,b.pt --shard data/mt/2026-01 \
        --out ens.json --pools 1,2,3,4,5
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from successor_data import MultiTaskDataset
from model import MultiTaskModel, Config, N_ELO_BINS
from identify_eval_mt import embed_all
from sweep_gallery import recall_at_size, _unit

KS = (1, 10, 100)


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    n_slots = ck.get("n_game_slots", 1)
    m = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                       d_embed=ck["d_embed"], n_time_bins=ck["n_time_bins"],
                       n_elo_bins=ck.get("n_elo_bins", N_ELO_BINS),
                       n_game_slots=n_slots).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck, cfg, n_slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoints")
    ap.add_argument("--names", default="", help="comma-separated display names")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gallery-players", type=int, default=100_000)
    ap.add_argument("--query-players", type=int, default=5_000)
    ap.add_argument("--centroid-games", type=int, default=12)
    ap.add_argument("--pools", default="1,2,3,4,5")
    ap.add_argument("--sizes", default="1000,5000,10000,20000,30000,50000,100000")
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--extra-pids", default="",
                    help="held-out ids for checkpoints that stored none "
                         "(applies to every such checkpoint)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pools = [int(x) for x in args.pools.split(",")]
    sizes = [int(x) for x in args.sizes.split(",")]
    paths = args.ckpts.split(",")
    names = args.names.split(",") if args.names else [
        os.path.basename(os.path.dirname(p)) for p in paths]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extra = np.load(args.extra_pids) if args.extra_pids else None
    metas, held_sets = [], []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        tp = ck.get("test_pids")
        if tp is None:
            if extra is None:
                raise SystemExit(f"{p} stored no test_pids and --extra-pids not given")
            tp = extra
        held_sets.append(set(int(x) for x in np.asarray(tp)))
        metas.append((int(ck["step"]), Config(**ck["cfg"]).max_len,
                      ck.get("n_game_slots", 1)))
        del ck

    common = set.intersection(*held_sets)
    for n, h, (st, ml, sl) in zip(names, held_sets, metas):
        print(f"  {n:<16} step {st:>7,}  max_len {ml:>4}  slots {sl}  held-out {len(h):,}",
              flush=True)
    print(f"common held-out across all {len(paths)} models: {len(common):,}", flush=True)
    if not common:
        raise SystemExit("no player is held out by every model")

    # The row index is the set of fully-clocked game-sides and does not depend
    # on max_len, so the same rows mean the same games for every model.
    ds0 = MultiTaskDataset(args.shard, max_len=metas[0][1], plies_per_game=1,
                           n_cand=2, with_rights=True)
    _idx = np.asarray(ds0.index)
    _gi, _seat = _idx[:, 0], _idx[:, 1]
    pid = np.where(_seat == 0,
                   np.asarray(ds0.meta["white_pid"])[_gi],
                   np.asarray(ds0.meta["black_pid"])[_gi]).astype(np.int64)
    order = np.argsort(pid, kind="stable")
    spid = pid[order]
    bnd = np.flatnonzero(np.r_[True, spid[1:] != spid[:-1], True])
    groups = {int(spid[bnd[i]]): order[bnd[i]:bnd[i + 1]] for i in range(len(bnd) - 1)}
    del ds0

    need_q = args.centroid_games + max(pools)
    rng = np.random.default_rng(args.seed)
    q_pool = [p for p, g in groups.items() if p in common and len(g) >= need_q]
    d_pool = [p for p, g in groups.items()
              if p not in common and len(g) >= args.centroid_games]
    print(f"{len(q_pool):,} eligible queries | {len(d_pool):,} distractors", flush=True)

    n_q = min(args.query_players, len(q_pool))
    qsel = [q_pool[i] for i in rng.choice(len(q_pool), n_q, replace=False)]
    n_d = min(args.gallery_players - n_q, len(d_pool))
    dsel = [d_pool[i] for i in rng.choice(len(d_pool), n_d, replace=False)]
    M = n_q + n_d
    sizes = [s for s in sizes if s <= M]
    print(f"gallery {M:,} ({n_q:,} queries + {n_d:,} distractors)", flush=True)

    rows, spans, qspans = [], {}, {}
    for p in qsel:
        g = groups[p]
        perm = rng.permutation(len(g))
        c = g[perm[:args.centroid_games]]
        q = g[perm[args.centroid_games:args.centroid_games + max(pools)]]
        spans[p] = (len(rows), len(rows) + len(c)); rows.extend(c.tolist())
        qspans[p] = (len(rows), len(rows) + len(q)); rows.extend(q.tolist())
    for p in dsel:
        g = groups[p]
        c = g[rng.permutation(len(g))[:args.centroid_games]]
        spans[p] = (len(rows), len(rows) + len(c)); rows.extend(c.tolist())
    rows = np.asarray(rows)
    gal_players = qsel + dsel
    print(f"embedding {len(rows):,} game-sides x {len(paths)} models", flush=True)

    # Per-model centroid banks and query banks, all on the same rows.
    banks = []
    for path, name in zip(paths, names):
        model, ck, cfg, n_slots = load_model(path, device)
        ds = MultiTaskDataset(args.shard, max_len=cfg.max_len, plies_per_game=1,
                              n_cand=2, with_rights=ck["n_planes"] == 13)
        E, _, elo_mae = embed_all(model, ds, rows, device, args.batch, args.workers,
                                  use_slots=n_slots > 1)
        E = torch.nn.functional.normalize(E, dim=-1)
        C = torch.stack([_unit(E[a:b].mean(0)) for p in gal_players
                         for a, b in [spans[p]]]).to(device)
        Q = {pool: torch.stack([_unit(E[qspans[p][0]:qspans[p][0] + pool].mean(0))
                                for p in qsel]).to(device) for pool in pools}
        banks.append({"name": name, "C": C, "Q": Q, "elo_mae": elo_mae})
        print(f"  {name}: centroids built, elo_mae {elo_mae:.0f}", flush=True)
        del model, ds, E
        torch.cuda.empty_cache()

    tidx = torch.arange(n_q, device=device)
    res = {"gallery_built": int(M), "query_players": int(n_q), "sizes": sizes,
           "models": [b["name"] for b in banks], "results": {}}

    def record(tag, pool, r):
        res["results"].setdefault(tag, {})[str(pool)] = {
            "direct_at_full_gallery": {f"recall@{k}": float((r < k).mean()) for k in KS},
            "median_rank": float(np.median(r)),
            "by_gallery_size": {str(N): {f"recall@{k}": recall_at_size(r, M, N, k)
                                         for k in KS} for N in sizes}}

    for pool in pools:
        per_model_r, rank_accum, score_accum = [], None, None
        for b in banks:
            rs, ranks_for_fusion = [], []
            for s in range(0, n_q, 512):
                sim = b["Q"][pool][s:s + 512] @ b["C"].T
                true = sim.gather(1, tidx[s:s + 512, None])
                rs.append((sim > true).sum(1).cpu())
                ranks_for_fusion.append(sim)
            per_model_r.append(torch.cat(rs).numpy().astype(np.float64))
            record(b["name"], pool, per_model_r[-1])

        # mean-score ensemble (== concatenated unit embeddings)
        rs = []
        for s in range(0, n_q, 512):
            sim = sum(b["Q"][pool][s:s + 512] @ b["C"].T for b in banks) / len(banks)
            true = sim.gather(1, tidx[s:s + 512, None])
            rs.append((sim > true).sum(1).cpu())
        record("ENSEMBLE_mean", pool, torch.cat(rs).numpy().astype(np.float64))

        # reciprocal-rank fusion: ordering only, no score calibration
        rs = []
        for s in range(0, n_q, 512):
            fused = 0
            for b in banks:
                sim = b["Q"][pool][s:s + 512] @ b["C"].T
                rk = sim.argsort(dim=1, descending=True).argsort(dim=1).float()
                fused = fused + 1.0 / (60.0 + rk)
            true = fused.gather(1, tidx[s:s + 512, None])
            rs.append((fused > true).sum(1).cpu())
        record("ENSEMBLE_rrf", pool, torch.cat(rs).numpy().astype(np.float64))

        line = " | ".join(
            f"{tag}:{res['results'][tag][str(pool)]['by_gallery_size'][str(max(sizes))]['recall@10']:.3f}"
            for tag in list(res["results"]))
        print(f"pool {pool} top-10 @{max(sizes)//1000}k  {line}", flush=True)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("ENSEMBLE_DONE", flush=True)


if __name__ == "__main__":
    main()
