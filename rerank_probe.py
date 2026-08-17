"""What does the reranker actually do to the true player's placement?

The aggregate r@10 numbers say reranking loses, but they do not say WHY, and
"the verifier has AUC 0.61" is not a placement. This answers the question in the
form it was asked: take players whose true account is already inside the
depth-100 cosine shortlist, and report where each signal puts them.

Three ranks per query, all within the same 100 candidates:

  cosine rank     where the embedding puts the true player
  verifier rank   where the VERIFIER ALONE puts them, ignoring cosine
  fused rank      where the combination puts them

If the verifier were adding information, its own ranking would be respectable
and fusing would beat cosine. If its ranking is near-random (median ~50 of 100),
then every fusion is cosine plus noise, and the aggregate loss is explained.

    python rerank_probe.py --verifier ckpt/final/verifier2_sat.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from gallery_ctx import Bundles, embed_bundles
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import N_TIME_BINS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--verifier", default="ckpt/final/verifier2_sat.pt")
    ap.add_argument("--pack", default="play/verifier_pack.npz")
    ap.add_argument("--gallery", default="play/gallery_2026.npz")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--out", default="plots/data/rerank_probe.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "play"))
    import server

    server.load(args.ckpt, args.ckpt, args.gallery)
    server.load_verifier(args.verifier, args.pack)
    C = server.MODEL["cent"]
    idx = {str(n).lower(): i for i, n in enumerate(server.MODEL["names"])}

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"]); slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len); wr = ck["n_planes"] == 13
    emb = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                         d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                         n_elo_bins=N_ELO_BINS, n_game_slots=slots,
                         elo_cond=bool(ck.get("elo_cond")))
    emb.load_state_dict(ck["model"]); emb.eval()

    meta = np.load(f"{args.shard}/meta.npy", mmap_mode="r")
    clocks = np.memmap(f"{args.shard}/clocks.u16", dtype=np.uint16, mode="r")
    moves_mm = np.memmap(f"{args.shard}/moves.u16", dtype=np.uint16, mode="r")
    sn = open(f"{args.shard}/players.txt", encoding="utf-8").read().split("\n")
    npl_ = ck["n_planes"]

    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
    ok = np.concatenate([np.asarray(clocks[np.asarray(meta["offset"], np.int64)]) != 0xFFFF] * 2)
    o = np.argsort(pid, kind="stable")
    pid, gid, seat, ok = pid[o], gid[o], seat[o], ok[o]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])

    def shard_blocks(picks):
        import chess
        from bitboards import decode_move, board_to_planes8
        from timefeat import time_features
        out = []
        for gi, st in picks:
            row = meta[gi]; off, n = int(row["offset"]), int(row["nply"])
            codes = np.asarray(moves_mm[off:off + n]); clk = np.asarray(clocks[off:off + n])
            T = min(len(codes), mlpg)
            pov = chess.WHITE if st == 0 else chess.BLACK
            pl = np.zeros((T, npl_, 8, 8), np.uint8); b = chess.Board()
            for t in range(T):
                board_to_planes8(b, pov, pl[t], wr)
                b.push(decode_move(int(codes[t])))
            fe, _, _ = time_features(clk, int(row["tc_base"]), int(row["tc_inc"]))
            mt = np.zeros(T, bool); mt[st::2] = True
            out.append((pl, fe[:T], mt))
        return out

    rng = np.random.default_rng(args.seed)
    order = list(range(len(bnd) - 1)); rng.shuffle(order)
    qsets, truth = [], []
    for i in order:
        if len(qsets) >= args.queries:
            break
        sl = slice(bnd[i], bnd[i + 1]); m = ok[sl]
        g_, s_ = gid[sl][m], seat[sl][m]
        if len(g_) < args.games:
            continue
        p = int(pid[sl][0])
        if p >= len(sn):
            continue
        j = idx.get(sn[p].lower())
        if j is None:
            continue
        sel = rng.permutation(len(g_))[:args.games]
        qsets.append([(int(g_[x]), int(s_[x])) for x in sel]); truth.append(j)
    print(f"{len(qsets)} queries x {args.games} games", flush=True)

    bundles, owner = [], []
    for qi, gs in enumerate(qsets):
        for c in range(0, len(gs) - slots + 1, slots):
            bundles.append(gs[c:c + slots]); owner.append(qi)
    E = F.normalize(embed_bundles(emb, Bundles(args.shard, bundles, mlpg, wr),
                                  slots, "cpu", 64, 0, "probe").float(), dim=-1)
    S = torch.zeros(len(qsets), C.shape[0])
    S.index_add_(0, torch.tensor(owner), E @ C.T)

    K = server.MODEL["ver_k"]
    cal = json.load(open("play/bayes_calib.json"))
    rows_out = []
    for qi in range(len(qsets)):
        sim = S[qi]
        rows = torch.topk(sim, args.depth).indices.tolist()
        if truth[qi] not in rows:
            continue                       # cosine already lost them; nothing to rerank
        per = server.verifier_scores(shard_blocks(qsets[qi][-(K - 1):]), rows,
                                     per_game=True)
        cs = sim[rows].numpy().astype(np.float64)
        vv = np.array([np.mean(per[r]) if per.get(r) else -1e9 for r in rows])
        z = (cs - cs.mean()) / (cs.std() + 1e-9)
        fused = cal["cos_a"] * z + cal["cos_b"] + np.array(
            [sum(cal["ver_a"] * s + cal["ver_b"] for s in per.get(r, ())) for r in rows])
        t = rows.index(truth[qi])
        rows_out.append({
            "cos": int((cs > cs[t]).sum()) + 1,
            "ver": int((vv > vv[t]).sum()) + 1,
            "fused": int((fused > fused[t]).sum()) + 1,
        })
        if len(rows_out) % 20 == 0:
            print(f"  {len(rows_out)} scored", flush=True)

    c = np.array([r["cos"] for r in rows_out])
    v = np.array([r["ver"] for r in rows_out])
    f = np.array([r["fused"] for r in rows_out])
    print(f"\n{len(c)} queries where the true player IS inside the top {args.depth}\n")
    print(f"{'ranking of the true player':<32}{'median':>8}{'mean':>8}"
          f"{'top1':>8}{'top10':>8}")
    for nm, a in (("cosine (what we ship)", c), ("verifier ALONE", v),
                  ("cosine + verifier fused", f)):
        print(f"  {nm:<30}{np.median(a):>8.0f}{a.mean():>8.1f}"
              f"{(a <= 1).mean():>8.3f}{(a <= 10).mean():>8.3f}")
    print(f"\n  verifier beats cosine on {(v < c).mean():.1%} of these queries")
    print(f"  a coin flip would put the true player at median {args.depth // 2}")
    json.dump({"rows": rows_out, "depth": args.depth}, open(args.out, "w"))
    print(f"\nwrote {args.out}")
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
