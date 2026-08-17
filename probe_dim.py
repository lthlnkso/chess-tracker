"""How many embedding dimensions does identification actually need?

Projects the SHIPPED 128-d gallery down to k dimensions with PCA and re-measures
top-10. This is free -- no training -- and it brackets the question:

  if PCA-k already matches 128    a model trained at k will almost certainly
                                  match too, because PCA is a worse compressor
                                  than a network given the same width
  if PCA-k collapses              the answer is not yet no. PCA optimises
                                  variance, not separability, and the directions
                                  it discards last are not the ones that tell
                                  two similar players apart. Only a trained run
                                  settles that case.

So a pass here is decisive and a fail is only suggestive -- which is exactly the
right shape for a screening test that costs nothing.

Absolute numbers here run high: the 2026-06 queries are inside the gallery's own
centroids, so this is not the product metric. The COMPARISON across k is valid,
because every arm carries the same leak.

    python probe_dim.py --players 400
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from bayesfeat import shard_queries
from gallery_ctx import Bundles, embed_bundles
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import N_TIME_BINS


class _Names:
    """shard_queries only needs the roster, not a feature table."""

    def __init__(self, names):
        self.names = names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="play/gallery_2026.npz")
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--players", type=int, default=400)
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[8, 16, 24, 32, 48, 64, 96, 128])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = np.load(args.gallery, allow_pickle=True)
    names = [str(x).lower() for x in g["names"]]
    C = torch.tensor(np.asarray(g["centroids"], np.float32))
    C = F.normalize(C, dim=-1)
    print(f"gallery {C.shape[0]:,} x {C.shape[1]}", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"]); slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len); wr = ck["n_planes"] == 13
    m = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                       d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                       n_elo_bins=N_ELO_BINS, n_game_slots=slots,
                       elo_cond=bool(ck.get("elo_cond")))
    m.load_state_dict(ck["model"]); m.eval()

    qs = shard_queries(args.shard, _Names(names), args.players, args.games,
                       args.seed)
    print(f"{len(qs):,} queries", flush=True)
    bundles, owner = [], []
    for qi, (_, _, _, picks) in enumerate(qs):
        for c in range(0, max(len(picks) - slots + 1, 1), slots):
            if picks[c:c + slots]:
                bundles.append(picks[c:c + slots]); owner.append(qi)
    E = F.normalize(embed_bundles(m, Bundles(args.shard, bundles, mlpg, wr),
                                  slots, "cpu", 64, 0, "dim").float(), dim=-1)
    Q = torch.zeros(len(qs), E.shape[1])
    Q.index_add_(0, torch.tensor(owner), E)
    Q = F.normalize(Q, dim=-1)
    truth = np.array([r for r, _, _, _ in qs])

    # PCA basis from the gallery, which is the population the centroids live in.
    mu = C.mean(0, keepdim=True)
    Cc = C - mu
    cov = (Cc.T @ Cc) / len(Cc)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evecs, evals = evecs[:, order], evals[order]

    print(f"\n{'dims':>6} {'cum var':>9} {'r@1':>8} {'r@10':>8} {'r@100':>8} "
          f"{'median':>8} {'gallery MB':>11}")
    print("-" * 62)
    base = None
    for k in args.dims:
        V = evecs[:, :k]
        Ck = F.normalize((C - mu) @ V, dim=-1)
        Qk = F.normalize((Q - mu) @ V, dim=-1)
        r = []
        for i in range(0, len(Qk), 64):
            s = Qk[i:i + 64] @ Ck.T
            for j in range(len(s)):
                t = truth[i + j]
                r.append(int((s[j] > s[j, t]).sum()) + 1)
        r = np.asarray(r)
        cv = float(evals[:k].sum() / evals.sum())
        mb = C.shape[0] * k * 2 / 1e6
        tag = "" if base is None else f"  ({(r <= 10).mean() - base:+.4f})"
        if base is None and k == args.dims[-1]:
            base = (r <= 10).mean()
        print(f"{k:>6} {cv:>9.4f} {(r<=1).mean():>8.4f} {(r<=10).mean():>8.4f} "
              f"{(r<=100).mean():>8.4f} {np.median(r):>8.0f} {mb:>11.1f}")
    print("\nPROBE_DONE")


if __name__ == "__main__":
    main()
