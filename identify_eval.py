"""Can we identify a player from their games?

Test players only (never seen in training). For each: 80% of their game-sides
build a centroid, 20% become queries. Every query is matched against every test
player's centroid by cosine similarity, and we report how often the true player
is in the top 1 / 10 / 100.

Then the same thing with queries pooled: 1, 10, half, all. Pooling is the
interesting axis -- one bullet game is thin evidence, twenty may not be.

    python identify_eval.py --ckpt /workspace/ckpt/id/last.pt \
        --combined /workspace/data/combined --out /workspace/ckpt/id/eval.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from id_data import EmbedDataset, collate, split_players
from model import PlayerEncoder, Config
from bitboards import N_PLANES13

KS = (1, 10, 100)


@torch.no_grad()
def embed_rows(model, combined, rows, max_len, batch, workers, device,
               with_rights=True, log_every=200):
    ds = EmbedDataset(combined, rows=rows, max_len=max_len, with_rights=with_rights)
    assert ds.n_planes == model.n_planes, f"loader {ds.n_planes} != model {model.n_planes}"
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    collate_fn=collate, pin_memory=True)
    out = torch.zeros((len(ds), model.d_embed), dtype=torch.float32)
    labels = torch.zeros(len(ds), dtype=torch.long)
    at = 0
    for i, b in enumerate(dl):
        n = b["planes"].shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model(b["planes"].to(device, non_blocking=True),
                      b["pad_mask"].to(device, non_blocking=True),
                      b["my_turn"].to(device, non_blocking=True))
        out[at:at + n] = e.float().cpu()
        labels[at:at + n] = b["player_id"]
        at += n
        if i % log_every == 0:
            print(f"  embedded {at:,}/{len(ds):,}", flush=True)
    return out[:at], labels[:at]


def recall_at(queries: torch.Tensor, q_labels: torch.Tensor,
              centroids: torch.Tensor, c_labels: torch.Tensor,
              device: str, chunk: int = 2048) -> dict:
    """Cosine kNN of queries against player centroids -> recall@k and mean rank."""
    C = centroids.to(device)
    cl = c_labels.to(device)
    maxk = min(max(KS), len(cl))
    hits = {k: 0 for k in KS}
    ranks = []
    n = len(queries)
    for s in range(0, n, chunk):
        q = queries[s:s + chunk].to(device)
        ql = q_labels[s:s + chunk].to(device)
        sim = q @ C.T                                     # both L2-normalised
        top = sim.topk(maxk, dim=1).indices
        match = cl[top] == ql[:, None]
        for k in KS:
            if k <= maxk:
                hits[k] += int(match[:, :k].any(1).sum())
        # exact rank of the true centroid, for a scale-free summary
        true_col = (cl[None, :] == ql[:, None]).float().argmax(1)
        true_sim = sim.gather(1, true_col[:, None])
        ranks.append((sim > true_sim).sum(1).cpu())
    ranks = torch.cat(ranks).float()
    return {
        **{f"recall@{k}": hits[k] / n for k in KS},
        "median_rank": float(ranks.median()),
        "mean_percentile": float((1 - ranks / max(len(cl) - 1, 1)).mean() * 100),
        "n_queries": n,
    }


def pool(queries: torch.Tensor, labels: torch.Tensor, size: int | str, seed: int = 0):
    """Average `size` queries per player into one vector ('half'/'all' allowed)."""
    rng = np.random.default_rng(seed)
    lab = labels.numpy()
    order = np.argsort(lab, kind="stable")
    sl = lab[order]
    bounds = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1], True])
    vecs, out_lab = [], []
    for i in range(len(bounds) - 1):
        g = order[bounds[i]:bounds[i + 1]]
        m = len(g)
        take = m if size == "all" else (max(1, m // 2) if size == "half" else min(int(size), m))
        if size not in ("all", "half") and m < int(size):
            continue                                   # not enough evidence to honour the ask
        sel = rng.choice(g, size=take, replace=False)
        v = queries[sel].mean(0)
        vecs.append(v / v.norm().clamp(min=1e-8))
        out_lab.append(sl[bounds[i]])
    if not vecs:
        return None, None
    return torch.stack(vecs), torch.tensor(out_lab, dtype=torch.long)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--combined", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--centroid-frac", type=float, default=0.8)
    ap.add_argument("--max-test-players", type=int, default=20000,
                    help="cap the gallery; 0 = all")
    ap.add_argument("--min-games", type=int, default=10,
                    help="test players need at least this many game-sides")
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = PlayerEncoder(cfg, n_planes=ck["n_planes"], d_embed=ck["d_embed"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint step {ck['step']}, d_embed {ck['d_embed']}", flush=True)

    full = np.load(os.path.join(args.combined, "index.npy"), mmap_mode="r")
    with open(os.path.join(args.combined, "players.txt"), encoding="utf-8") as f:
        n_players = len(f.read().split("\n"))
    # Same split rule and seed as training, or "held-out" means nothing.
    _, test_rows, _ = split_players(full, n_players, ck["test_frac"], ck["seed"])
    print(f"test game-sides: {len(test_rows):,}", flush=True)

    gp = np.asarray(full["gpid"])[test_rows]
    uniq, counts = np.unique(gp, return_counts=True)
    eligible = uniq[counts >= args.min_games]
    print(f"test players with >= {args.min_games} game-sides: {len(eligible):,}", flush=True)
    if args.max_test_players and len(eligible) > args.max_test_players:
        rng = np.random.default_rng(args.seed)
        eligible = rng.choice(eligible, size=args.max_test_players, replace=False)
        print(f"  capped gallery to {len(eligible):,} players", flush=True)

    keep = np.isin(gp, eligible)
    rows = test_rows[keep]
    print(f"embedding {len(rows):,} game-sides", flush=True)
    emb, lab = embed_rows(model, args.combined, rows, cfg.max_len,
                          args.batch, args.workers, device,
                          with_rights=ck["n_planes"] == N_PLANES13)

    # Per player: split their game-sides into centroid vs query.
    rng = np.random.default_rng(args.seed)
    lab_np = lab.numpy()
    order = np.argsort(lab_np, kind="stable")
    sl = lab_np[order]
    bounds = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1], True])

    cent, cent_lab, q_idx = [], [], []
    for i in range(len(bounds) - 1):
        g = order[bounds[i]:bounds[i + 1]]
        perm = rng.permutation(len(g))
        ncent = max(1, int(round(args.centroid_frac * len(g))))
        if len(g) - ncent < 1:
            ncent = len(g) - 1
        ci, qi = g[perm[:ncent]], g[perm[ncent:]]
        v = emb[ci].mean(0)
        cent.append(v / v.norm().clamp(min=1e-8))
        cent_lab.append(sl[bounds[i]])
        q_idx.append(qi)

    centroids = torch.stack(cent)
    c_labels = torch.tensor(cent_lab, dtype=torch.long)
    q_idx = np.concatenate(q_idx)
    queries, q_labels = emb[q_idx], lab[q_idx]
    # Every pooling size must be scored on the SAME players, or the comparison is
    # confounded: "10 pooled" would silently drop players with <10 query games --
    # exactly the players with the least evidence and the hardest to identify --
    # and then look better than "all" purely from an easier population.
    pool_max = max(s for s in (1, 10) if isinstance(s, int))
    qlab_np = q_labels.numpy()
    uq, cnt = np.unique(qlab_np, return_counts=True)
    matched = uq[cnt >= pool_max]
    keep_q = np.isin(qlab_np, matched)
    m_queries, m_labels = queries[keep_q], q_labels[keep_q]
    print(f"gallery: {len(centroids):,} centroids | "
          f"queries {len(queries):,} -> {len(m_queries):,} on the matched set "
          f"({len(matched):,} players with >= {pool_max} query games)", flush=True)

    results = {}
    for size in (1, 10, "half", "all"):
        if size == 1:
            q, ql = m_queries, m_labels
        else:
            q, ql = pool(m_queries, m_labels, size, args.seed)
        if q is None:
            continue
        r = recall_at(q, ql, centroids, c_labels, device)
        r["n_query_players"] = int(len(np.unique(ql.numpy())))
        results[str(size)] = r
        print(f"\nquery = {size} game(s) pooled   ({r['n_queries']:,} queries over "
              f"{r['n_query_players']:,} players, {len(centroids):,} candidates)")
        for k in KS:
            print(f"  top-{k:<4} {r[f'recall@{k}']:.4f}")
        print(f"  median rank {r['median_rank']:.0f} | mean percentile "
              f"{r['mean_percentile']:.2f}")

    # Headline number: one game, every eligible player, no matching restriction.
    r1 = recall_at(queries, q_labels, centroids, c_labels, device)
    r1["n_query_players"] = int(len(np.unique(q_labels.numpy())))
    results["1_all_players"] = r1
    print(f"\n[unmatched] query = 1 game over ALL {r1['n_query_players']:,} test "
          f"players: top-1 {r1['recall@1']:.4f}  top-10 {r1['recall@10']:.4f}  "
          f"top-100 {r1['recall@100']:.4f}")

    payload = {
        "ckpt": args.ckpt, "step": ck["step"],
        "n_gallery_players": int(len(centroids)),
        "n_test_gamesides": int(len(rows)),
        "centroid_frac": args.centroid_frac,
        "matched_pool_min": 10,
        "chance_recall@1": 1.0 / len(centroids),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
