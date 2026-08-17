"""Top-10 recall as a function of gallery size, from a single embedding pass.

The product's gallery is every lichess player, not the 20k we happen to score.
Running seven separate evals to get seven gallery sizes would be seven times the
cost and would also vary the query set between points, which is exactly the
comparison you do not want to be noisy.

Instead this embeds once against the largest gallery we can afford and derives
every smaller size exactly. For a query whose true centroid is beaten by `r` of
the `M-1` distractors, a random sub-gallery of size `N` containing the true
player draws `N-1` distractors without replacement, so the number of beating
distractors that survive is hypergeometric:

    recall@k(N) = P(X < k),  X ~ Hypergeometric(M-1, r, N-1)

That is exact, not a fit -- and at N = M it reduces to the directly measured
recall, which the script asserts as a self-check.

Two deliberate choices:

- **Distractors may include players the model trained on.** Only the *queries*
  must be held out. The deployed gallery really is everyone, and restricting
  distractors to held-out players would both shrink the reachable gallery and
  misrepresent the product.
- **Queries are pooled by averaging**, matching the original single-game
  protocol, so a context model is judged on whether context-conditioned
  pre-training produced better per-game embeddings.

    python sweep_gallery.py --ckpt ckpt/ctx3_160_ft/last.pt --shard data/mt/2026-01 \
        --out sweep.json --gallery-players 100000 --query-players 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from successor_data import MultiTaskDataset
from model import MultiTaskModel, Config, N_ELO_BINS
from identify_eval_mt import embed_all

KS = (1, 10, 100)


try:                                    # scipy is present in the pod image
    from scipy.special import gammaln as _gammaln
except ImportError:                     # keep the script runnable without it
    _gammaln = np.vectorize(math.lgamma)


def _logC(n, k):
    """log C(n, k), vectorised, -inf where the choice is impossible."""
    n = np.asarray(n, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    bad = (k < 0) | (k > n) | (n < 0)
    safe_k = np.where(bad, 0.0, k)
    out = _gammaln(n + 1) - _gammaln(safe_k + 1) - _gammaln(n - safe_k + 1)
    return np.where(bad, -np.inf, out)


def recall_at_size(r, M, N, k):
    """P(fewer than k of the r beating distractors survive in a gallery of N)."""
    r = np.asarray(r, dtype=np.float64)
    tot = np.zeros_like(r)
    denom = _logC(M - 1, N - 1)
    for x in range(k):
        lp = _logC(r, x) + _logC(M - 1 - r, N - 1 - x) - denom
        tot += np.exp(np.clip(lp, -700, 0))
    return float(np.mean(np.clip(tot, 0.0, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gallery-players", type=int, default=100_000)
    ap.add_argument("--query-players", type=int, default=5_000)
    ap.add_argument("--centroid-games", type=int, default=12)
    ap.add_argument("--pools", default="1,2,3")
    ap.add_argument("--sizes", default="1000,5000,10000,20000,30000,50000,100000")
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--test-pids-file", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pools = [int(x) for x in args.pools.split(",")]
    sizes = [int(x) for x in args.sizes.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    n_slots = ck.get("n_game_slots", 1)
    model = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=ck["d_embed"], n_time_bins=ck["n_time_bins"],
                           n_elo_bins=ck.get("n_elo_bins", N_ELO_BINS),
                           n_game_slots=n_slots).to(device)
    model.load_state_dict(ck["model"])
    print(f"ckpt step {ck['step']}, max_len {cfg.max_len}, slots {n_slots}", flush=True)

    # plies_per_game/n_cand are minimal: candidate successor states are pure
    # waste here, and at a million games that waste dominates the wall clock.
    ds = MultiTaskDataset(args.shard, max_len=cfg.max_len, plies_per_game=1,
                          n_cand=2, with_rights=ck["n_planes"] == 13)
    # Vectorised: a Python comprehension over ~48M game-sides takes minutes and
    # allocates 48M boxed ints.
    _idx = np.asarray(ds.index)
    _gi, _seat = _idx[:, 0], _idx[:, 1]
    pid = np.where(_seat == 0,
                   np.asarray(ds.meta["white_pid"])[_gi],
                   np.asarray(ds.meta["black_pid"])[_gi]).astype(np.int64)

    order = np.argsort(pid, kind="stable")
    spid = pid[order]
    bnd = np.flatnonzero(np.r_[True, spid[1:] != spid[:-1], True])
    groups = {int(spid[bnd[i]]): order[bnd[i]:bnd[i + 1]]
              for i in range(len(bnd) - 1)}

    need_q = args.centroid_games + max(pools)
    rng = np.random.default_rng(args.seed)

    held = None
    if args.test_pids_file:
        held = set(int(x) for x in np.load(args.test_pids_file))
        print(f"held-out ids: {len(held):,}", flush=True)
    elif ck.get("test_pids") is not None:
        held = set(int(x) for x in np.asarray(ck["test_pids"]))
        print(f"held-out ids from checkpoint: {len(held):,}", flush=True)
    if held is None:
        raise SystemExit("no held-out set: queries would include trained-on players")

    q_pool = [p for p, g in groups.items() if p in held and len(g) >= need_q]
    d_pool = [p for p, g in groups.items() if p not in held and len(g) >= args.centroid_games]
    print(f"{len(q_pool):,} eligible query players | {len(d_pool):,} distractors",
          flush=True)

    n_q = min(args.query_players, len(q_pool))
    qsel = [q_pool[i] for i in rng.choice(len(q_pool), n_q, replace=False)]
    n_d = min(args.gallery_players - n_q, len(d_pool))
    dsel = [d_pool[i] for i in rng.choice(len(d_pool), n_d, replace=False)]
    M = n_q + n_d
    print(f"gallery {M:,} players ({n_q:,} queries + {n_d:,} distractors)", flush=True)
    sizes = [s for s in sizes if s <= M]

    # Assemble every row we need, once, then embed in one batched pass.
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
    print(f"embedding {len(rows):,} game-sides", flush=True)

    E, _, elo_mae = embed_all(model, ds, rows, device, args.batch, args.workers,
                              use_slots=n_slots > 1)
    E = torch.nn.functional.normalize(E, dim=-1)

    gal_players = qsel + dsel
    C = torch.stack([_unit(E[a:b].mean(0)) for p in gal_players
                     for a, b in [spans[p]]]).to(device)
    print(f"built {len(C):,} centroids | elo_mae {elo_mae:.0f}", flush=True)

    res = {"gallery_built": int(M), "query_players": int(n_q),
           "centroid_games": args.centroid_games, "elo_mae": elo_mae,
           "sizes": sizes, "pools": {}}
    for pool in pools:
        Q, tidx = [], []
        for i, p in enumerate(qsel):
            a, b = qspans[p]
            Q.append(_unit(E[a:a + pool].mean(0)))
            tidx.append(i)
        Q = torch.stack(Q).to(device)
        tidx = torch.tensor(tidx, device=device)

        ranks = []
        for s in range(0, len(Q), 1024):
            sim = Q[s:s + 1024] @ C.T
            true = sim.gather(1, tidx[s:s + 1024, None])
            ranks.append((sim > true).sum(1).cpu())
        r = torch.cat(ranks).numpy().astype(np.float64)

        direct = {f"recall@{k}": float((r < k).mean()) for k in KS}
        curve = {str(N): {f"recall@{k}": recall_at_size(r, M, N, k) for k in KS}
                 for N in sizes}
        # At N == M the formula must reproduce the directly measured value.
        if M in sizes:
            d, e = direct["recall@10"], curve[str(M)]["recall@10"]
            assert abs(d - e) < 1e-6, f"extrapolation self-check failed: {d} vs {e}"
        res["pools"][str(pool)] = {"direct_at_full_gallery": direct,
                                   "median_rank": float(np.median(r)),
                                   "by_gallery_size": curve}
        lbl = lambda N: f"{N//1000}k" if N >= 1000 else str(N)
        line = "  ".join(f"{lbl(N)}:{curve[str(N)]['recall@10']:.3f}" for N in sizes)
        print(f"pool {pool}: top-10 by gallery  {line}", flush=True)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("SWEEP_DONE", flush=True)


def _unit(v):
    return v / v.norm().clamp(min=1e-8)


if __name__ == "__main__":
    main()
