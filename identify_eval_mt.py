"""Identification eval for the multi-task model, with query pooling.

The fine-tune's built-in eval only scores single-game queries. The product
question is different: a visitor plays a handful of games, so what matters is
recall when several of their games are averaged into one query.

Every pool size is scored on the SAME players -- those holding at least
`--max-pool` query games. Otherwise "3 pooled" would quietly drop the players
with the least evidence, who are also the hardest to identify, and look better
than "1" for the wrong reason.

    python identify_eval_mt.py --ckpt ckpt/mt_id/last.pt --shard data/mt/2026-01 \
        --out eval_pooled.json --pools 1,2,3,5,10
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from successor_data import MultiTaskDataset, collate_multitask
from model import MultiTaskModel, Config, elo_expectation

KS = (1, 10, 100)


@torch.no_grad()
def embed_all(model, ds, rows, device, batch, workers, log_every=300,
              use_slots=False):
    """use_slots: a context model saw a game-slot embedding on every training
    sample, so embedding a lone game without one is a train/inference mismatch.
    Slot 0 is the faithful choice -- it is what 'the first game' looked like."""
    dl = DataLoader(Subset(ds, rows.tolist()), batch_size=batch, shuffle=False,
                    num_workers=workers, collate_fn=collate_multitask, pin_memory=True)
    E, L, EL, TR = [], [], [], []
    model.eval()
    for i, b in enumerate(dl):
        b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
             for k, v in b.items()}
        gs = None
        if use_slots:
            B, T = b["planes"].shape[:2]
            gs = torch.zeros((B, T), dtype=torch.long, device=b["planes"].device)
        e, el = model.embed(b["planes"], b["extra"], b["pad_mask"], b["my_turn"], gs)
        E.append(e.float().cpu()); L.append(b["player_id"].cpu())
        EL.append(elo_expectation(el).cpu()); TR.append(b["elo"].cpu())
        if i % log_every == 0:
            print(f"  embedded {i*batch:,}/{len(rows):,}", flush=True)
    return (torch.cat(E), torch.cat(L),
            float((torch.cat(EL) - torch.cat(TR).float()).abs().mean()))


def recall(Q, QL, C, CL, device, chunk=2048):
    C = C.to(device); CL = CL.to(device)
    maxk = min(max(KS), len(CL))
    hits = {k: 0 for k in KS}
    ranks = []
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk].to(device); ql = QL[s:s + chunk].to(device)
        sim = q @ C.T
        top = sim.topk(maxk, dim=1).indices
        m = CL[top] == ql[:, None]
        for k in hits:
            if k <= maxk:
                hits[k] += int(m[:, :k].any(1).sum())
        tc = (CL[None, :] == ql[:, None]).float().argmax(1)
        ranks.append((sim > sim.gather(1, tc[:, None])).sum(1).cpu())
    r = torch.cat(ranks).float()
    n = len(Q)
    return {**{f"recall@{k}": hits[k] / n for k in KS},
            "median_rank": float(r.median()), "n_queries": n,
            "mean_percentile": float((1 - r / max(len(CL) - 1, 1)).mean() * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pools", default="1,2,3,5,10")
    ap.add_argument("--centroid-frac", type=float, default=0.8)
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--eval-players", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--test-pids-file", default="",
                    help="explicit held-out player ids (see make_test_pids.py)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pools = [int(x) for x in args.pools.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    ds = MultiTaskDataset(args.shard, max_len=cfg.max_len, plies_per_game=8,
                          n_cand=8, with_rights=ck["n_planes"] == 13)
    # A context checkpoint carries game_emb; build with the same slot count or
    # load_state_dict rejects it outright.
    model = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=ck["d_embed"], n_time_bins=ck["n_time_bins"],
                           n_elo_bins=ck["n_elo_bins"],
                           n_game_slots=ck.get("n_game_slots", 1)).to(device)
    model.load_state_dict(ck["model"])
    print(f"ckpt step {ck['step']}, d_embed {ck['d_embed']}", flush=True)

    pid = np.array([int(ds.meta[g]["white_pid"] if s == 0 else ds.meta[g]["black_pid"])
                    for g, s in ds.index])
    u, c = np.unique(pid, return_counts=True)
    keep = set(u[c >= args.min_games].tolist())
    rows = np.flatnonzero(np.fromiter((p in keep for p in pid), bool, len(pid)))
    rng = np.random.default_rng(args.seed)
    players = np.unique(pid[rows])
    if args.test_pids_file:
        # Two models trained with different splits have no common clean test set
        # unless one is supplied. Passing the intersection of both held-out sets
        # is the only way to score them on players NEITHER model trained on.
        test_p = set(int(x) for x in np.load(args.test_pids_file))
        print(f"held-out ids from {args.test_pids_file} ({len(test_p):,})", flush=True)
    else:
        test_p = set(rng.choice(players, int(len(players) * args.test_frac),
                                replace=False).tolist())
    te = rows[np.fromiter((p in test_p for p in pid[rows]), bool, len(rows))]
    gal = np.unique(pid[te])
    if len(gal) > args.eval_players:
        gal = set(rng.choice(gal, args.eval_players, replace=False).tolist())
        te = te[np.fromiter((p in gal for p in pid[te]), bool, len(te))]
    print(f"gallery target {len(np.unique(pid[te])):,} players, {len(te):,} rows",
          flush=True)

    E, L, elo_mae = embed_all(model, ds, te, device, args.batch, args.workers)

    ln = L.numpy(); order = np.argsort(ln, kind="stable"); sl = ln[order]
    bnd = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1], True])
    cent, clab, qgroups = [], [], []
    for i in range(len(bnd) - 1):
        g = order[bnd[i]:bnd[i + 1]]
        if len(g) < 3:
            continue
        perm = rng.permutation(len(g))
        nc = min(max(1, int(round(args.centroid_frac * len(g)))), len(g) - 1)
        v = E[g[perm[:nc]]].mean(0)
        cent.append(v / v.norm().clamp(min=1e-8))
        clab.append(int(sl[bnd[i]]))
        qgroups.append(g[perm[nc:]])
    C = torch.stack(cent); CL = torch.tensor(clab)
    print(f"gallery {len(CL):,} centroids", flush=True)

    # Matched set: only players with enough query games for the largest pool.
    biggest = max(pools)
    matched = [i for i, q in enumerate(qgroups) if len(q) >= biggest]
    print(f"{len(matched):,}/{len(qgroups):,} players have >= {biggest} query games "
          f"-- all pool sizes scored on those", flush=True)

    out = {"gallery": int(len(CL)), "chance@1": 1.0 / len(CL), "elo_mae": elo_mae,
           "matched_players": len(matched), "pools": {}}
    for p in pools:
        qs, qls = [], []
        for i in matched:
            g = qgroups[i]
            sel = rng.choice(g, size=p, replace=False)
            v = E[sel].mean(0)
            qs.append(v / v.norm().clamp(min=1e-8))
            qls.append(CL[i])
        r = recall(torch.stack(qs), torch.tensor(qls), C, CL, device)
        out["pools"][str(p)] = r
        print(f"  pool {p:>3}: top-1 {r['recall@1']:.4f}  top-10 {r['recall@10']:.4f}  "
              f"top-100 {r['recall@100']:.4f}  med rank {r['median_rank']:.0f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}\nPOOLED_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
