"""Re-rank the top-N candidates with features cosine cannot see.

The embedding was trained with a metric loss so that cosine IS the right
comparison, which is why a model fed only cosine would add nothing. The value is
in a second stage over a shortlist, using signals that live outside the vector:

  elo gap        we estimate a visitor's rating to MAE 156 and every gallery
                 player has a known one. Today that costs nothing and is used
                 for nothing. As a soft feature it can only reorder a shortlist,
                 so a bad estimate loses a little ranking quality instead of
                 eliminating the right answer -- which is why this is safe where
                 a hard Elo filter would not be.
  centroid size  a 10-game centroid is noisier than a 60-game one; a slightly
                 lower cosine against a rich centroid can be worth more than a
                 high one against a thin one.
  score shape    gap to the top, margin over the rest, z-score inside the
                 shortlist. Absolute cosine means little; standing out does.

Trained on players whose identity we know, with the true match as the positive.

    python rerank.py train --shard data/2026-06-big --gallery play/gallery_2026.npz
    python rerank.py eval  --shard data/2026-06-big --gallery play/gallery_2026.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import chess

from bitboards import decode_move, board_to_planes8, N_PLANES13
from timefeat import time_features, ms_used_per_ply, N_TIME_FEATS, N_TIME_BINS
from model import MultiTaskModel, Config, N_ELO_BINS, ELO_CENTRES

TOPN = 100


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    m = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                       d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                       n_elo_bins=N_ELO_BINS, n_game_slots=ck.get("n_game_slots", 1),
                       elo_cond=bool(ck.get("elo_cond"))).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck


@torch.no_grad()
def embed_query(model, ck, shard_arrays, picks, device):
    """One joint embedding plus the rating estimate, for k game-sides."""
    meta, moves, clocks = shard_arrays
    npl = ck["n_planes"]; wr = npl == N_PLANES13
    mlpg = ck.get("max_len_per_game", 160)
    slots = ck.get("n_game_slots", 1)
    blocks = []
    for gi, seat in picks:
        row = meta[gi]; o, n = int(row["offset"]), int(row["nply"])
        codes = np.asarray(moves[o:o + n]); clk = np.asarray(clocks[o:o + n])
        T = min(len(codes), mlpg)
        pov = chess.WHITE if seat == 0 else chess.BLACK
        pl = np.zeros((T, npl, 8, 8), np.uint8); b = chess.Board()
        for t in range(T):
            board_to_planes8(b, pov, pl[t], wr)
            b.push(decode_move(int(codes[t])))
        fe, _, _ = time_features(clk, int(row["tc_base"]), int(row["tc_inc"]))
        mt = np.zeros(T, bool); mt[seat::2] = True
        blocks.append((pl, fe[:T], mt))
    T = sum(x[0].shape[0] for x in blocks)
    P = np.zeros((1, T, npl, 8, 8), np.uint8); E = np.zeros((1, T, N_TIME_FEATS), np.float32)
    M = np.zeros((1, T), bool); S = np.zeros((1, T), np.int64); Q = np.zeros((1, T), np.int64)
    at = 0
    for si, (pl, fe, mt) in enumerate(blocks):
        t = pl.shape[0]
        P[0, at:at + t] = pl; E[0, at:at + t] = fe; M[0, at:at + t] = mt
        S[0, at:at + t] = min(si, slots - 1); Q[0, at:at + t] = np.arange(t); at += t
    e, elo_logits = model.embed(torch.from_numpy(P).to(device), torch.from_numpy(E).to(device),
                                torch.zeros((1, T), dtype=torch.bool, device=device),
                                torch.from_numpy(M).to(device), torch.from_numpy(S).to(device),
                                torch.from_numpy(Q).to(device))
    q = torch.nn.functional.normalize(e.float(), dim=-1)[0].cpu()
    p = torch.softmax(elo_logits.float(), -1)[0].cpu()
    c = ELO_CENTRES
    mean = float((p * c).sum()); sd = float(((p * (c - mean) ** 2).sum()).sqrt())
    return q, mean, sd


def features(sims, order, cent_games, cand_elo, q_elo, q_sd, n_games):
    """Per-candidate feature rows for one query's shortlist."""
    s = sims[order].numpy().astype(np.float64)
    top = s[0]
    rank = np.arange(len(s), dtype=np.float64)
    mu, sd = s.mean(), s.std() + 1e-9
    ce = cand_elo[order.numpy()]
    known = ce > 0
    gap_elo = np.where(known, np.abs(ce - q_elo), 400.0)          # 400 = "no info"
    return np.column_stack([
        s,                                   # cosine
        top - s,                             # gap to the best candidate
        (s - mu) / sd,                       # z within the shortlist
        rank,
        cent_games[order.numpy()],           # richness of the candidate centroid
        gap_elo,
        gap_elo / max(q_sd, 1.0),            # gap in units of our own uncertainty
        np.where(known, ce, q_elo),
        np.full(len(s), q_elo),
        np.full(len(s), q_sd),
        np.full(len(s), n_games),
        np.full(len(s), float(len(s))),
    ])


FEATURE_NAMES = ["cos", "gap_to_top", "z", "rank", "cent_games", "elo_gap",
                 "elo_gap_over_sd", "cand_elo", "q_elo", "q_sd", "n_games", "shortlist"]


def build_dataset(args, device):
    g = np.load(args.gallery, allow_pickle=True)
    C = torch.from_numpy(g["centroids"].astype(np.float32))
    names = [str(n).lower() for n in g["names"]]
    idx = {n: i for i, n in enumerate(names)}
    cent_games = g["centroid_games"].astype(np.float64)

    cand_elo = np.zeros(len(names), np.float64)
    if os.path.exists(args.elo_table):
        et = np.load(args.elo_table, allow_pickle=True)
        emap = {str(n).lower(): float(e) for n, e in zip(et["names"], et["elo"])}
        hit = 0
        for i, n in enumerate(names):
            v = emap.get(n)
            if v:
                cand_elo[i] = v; hit += 1
        print(f"  rating known for {hit:,}/{len(names):,} gallery players", flush=True)
    else:
        print(f"  {args.elo_table} missing -- elo features will be inert", flush=True)

    model, ck = load_model(args.ckpt, device)
    slots = ck.get("n_game_slots", 1)
    meta = np.load(f"{args.shard}/meta.npy", mmap_mode="r")
    moves = np.memmap(f"{args.shard}/moves.u16", dtype=np.uint16, mode="r")
    clocks = np.memmap(f"{args.shard}/clocks.u16", dtype=np.uint16, mode="r")
    shard_names = open(f"{args.shard}/players.txt", encoding="utf-8").read().split("\n")

    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
    ok = np.concatenate([np.asarray(clocks[np.asarray(meta["offset"], np.int64)]) != 0xFFFF] * 2)
    o = np.argsort(pid, kind="stable")
    pid, gid, seat, ok = pid[o], gid[o], seat[o], ok[o]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])

    rng = np.random.default_rng(args.seed)
    rows, labels, groups, base_hit, n_used = [], [], [], [], 0
    order_players = list(range(len(bnd) - 1))
    rng.shuffle(order_players)
    for i in order_players:
        if n_used >= args.players:
            break
        sl = slice(bnd[i], bnd[i + 1]); m = ok[sl]
        g_, s_ = gid[sl][m], seat[sl][m]
        if len(g_) < args.k:
            continue
        p = int(pid[sl][0])
        if p >= len(shard_names):
            continue
        who = shard_names[p].lower()
        j = idx.get(who)
        if j is None:                       # not in the gallery: no label
            continue
        sel = rng.permutation(len(g_))[:args.k]
        picks = [(int(g_[x]), int(s_[x])) for x in sel]
        try:
            q, q_elo, q_sd = embed_query(model, ck, (meta, moves, clocks), picks, device)
        except Exception:
            continue
        sims = q @ C.T
        top = torch.topk(sims, TOPN)
        if j not in set(top.indices.tolist()):
            n_used += 1
            base_hit.append(0)
            continue                        # unreachable: re-ranking cannot save it
        base_hit.append(1)
        F = features(sims, top.indices, cent_games, cand_elo, q_elo, q_sd, args.k)
        y = (top.indices.numpy() == j).astype(np.int32)
        rows.append(F); labels.append(y); groups.append(len(y))
        n_used += 1
        if n_used % 200 == 0:
            print(f"    {n_used:,}/{args.players:,} queries", flush=True)

    X = np.vstack(rows); Y = np.concatenate(labels)
    print(f"  {len(groups):,} usable queries | true match inside top-{TOPN}: "
          f"{np.mean(base_hit)*100:.1f}%", flush=True)
    return X, Y, np.array(groups), np.array(base_hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "train"))
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--gallery", default="play/gallery_2026.npz")
    ap.add_argument("--elo-table", default="play/elo_table.npz")
    ap.add_argument("--players", type=int, default=1500)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="play/reranker.json")
    ap.add_argument("--cache", default="play/rerank_data.npz",
                    help="feature cache. Built in one process and trained in "
                         "another: torch and xgboost each bring their own "
                         "OpenMP runtime and loading both segfaults on macOS.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "build":
        X, Y, groups, base_hit = build_dataset(args, device)
        np.savez_compressed(args.cache, X=X, Y=Y, groups=groups, base_hit=base_hit)
        print(f"wrote {args.cache}: {X.shape[0]:,} rows, {len(groups):,} queries")
        print("RERANK_BUILD_DONE")
        return

    d = np.load(args.cache)
    X, Y, groups, base_hit = d["X"], d["Y"], d["groups"], d["base_hit"]
    print(f"  {X.shape[0]:,} rows | {len(groups):,} queries | "
          f"reachable {base_hit.mean()*100:.1f}%")

    import xgboost as xgb
    # Split by QUERY, never by row: rows from one shortlist share a query and
    # leak into each other trivially.
    nq = len(groups)
    cut = int(nq * 0.75)
    starts = np.r_[0, np.cumsum(groups)]
    tr_rows = np.arange(starts[cut])
    te_rows = np.arange(starts[cut], starts[-1])

    dtr = xgb.DMatrix(X[tr_rows], label=Y[tr_rows], feature_names=FEATURE_NAMES)
    dtr.set_group(groups[:cut])
    dte = xgb.DMatrix(X[te_rows], label=Y[te_rows], feature_names=FEATURE_NAMES)
    dte.set_group(groups[cut:])

    params = {"objective": "rank:pairwise", "eta": 0.08, "max_depth": 5,
              "subsample": 0.9, "colsample_bytree": 0.9, "eval_metric": "ndcg@10",
              "min_child_weight": 5}
    bst = xgb.train(params, dtr, num_boost_round=250,
                    evals=[(dte, "test")], verbose_eval=50)

    # Compare like with like: cosine ordering vs re-ranked ordering, on the
    # SAME shortlists, over the held-out queries only.
    pred = bst.predict(dte)
    at = 0
    b1 = b10 = r1 = r10 = 0
    for gi, n in enumerate(groups[cut:]):
        blk = slice(at, at + n); at += n
        y = Y[te_rows][blk]
        base_pos = int(np.argmax(y))                 # shortlist is cosine-ordered
        new_pos = int(np.where(np.argsort(-pred[blk]) == base_pos)[0][0])
        b1 += base_pos == 0; b10 += base_pos < 10
        r1 += new_pos == 0;  r10 += new_pos < 10
    n = len(groups[cut:])
    print(f"\nheld-out queries where the answer was reachable: {n}")
    print(f"  cosine only : r@1 {b1/n:.4f}  r@10 {b10/n:.4f}")
    print(f"  re-ranked   : r@1 {r1/n:.4f}  r@10 {r10/n:.4f}")
    imp = bst.get_score(importance_type="gain")
    print("\n  feature gain:")
    for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {k:<18}{v:8.1f}")
    bst.save_model(args.out)
    print(f"\nwrote {args.out}")
    print("RERANK_DONE")


if __name__ == "__main__":
    main()
