"""Identification eval for multi-game-context models, 1..N games per query.

Two ways to turn k games into one query vector:

    joint    feed all k games to the trunk as one sequence, pool once
    average  embed each game separately and average the vectors

The second is what every earlier model did, because it was the only option. A
multi-game-context model can do the first, and the comparison is the whole point
of the branch -- if joint does not beat average, the extra context is not
earning its cost.

Both are reported for every k, on the same players and the same games, so the
only difference is the aggregation.

    python identify_eval_ctx.py --ckpt ckpt/ctx3_pre/last.pt --shard data/mt/2026-01 \
        --out eval_ctx.json --ks 1,2,3,5,10
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import chess
from bitboards import board_to_planes8, decode_move, n_planes_compact
from timefeat import time_features, N_TIME_FEATS, N_TIME_BINS
from model import MultiTaskModel, Config, N_ELO_BINS

KS_REPORT = (1, 10, 100)


def build_games(meta, moves, clocks, picks, max_len_per_game, n_planes, with_rights):
    """Encode a list of (game, seat) into per-game plane/feature blocks."""
    out = []
    for gi, seat in picks:
        row = meta[gi]
        o, n = int(row["offset"]), int(row["nply"])
        codes = np.asarray(moves[o:o + n])
        clk = np.asarray(clocks[o:o + n])
        T = min(len(codes), max_len_per_game)
        pov = chess.WHITE if seat == 0 else chess.BLACK
        pl = np.zeros((T, n_planes, 8, 8), dtype=np.uint8)
        b = chess.Board()
        for t in range(T):
            board_to_planes8(b, pov, pl[t], with_rights)
            b.push(decode_move(int(codes[t])))
        fe, _, _ = time_features(clk, int(row["tc_base"]), int(row["tc_inc"]))
        mt = np.zeros(T, dtype=bool)
        mt[seat::2] = True
        out.append((pl, fe[:T], mt))
    return out


def pack(blocks, device, n_slots):
    """Concatenate game blocks into one padded batch row set."""
    T = sum(b[0].shape[0] for b in blocks)
    P = blocks[0][0].shape[1]
    planes = np.zeros((1, T, P, 8, 8), np.uint8)
    extra = np.zeros((1, T, N_TIME_FEATS), np.float32)
    mine = np.zeros((1, T), bool)
    slot = np.zeros((1, T), np.int64)
    ppos = np.zeros((1, T), np.int64)
    at = 0
    for s, (pl, fe, mt) in enumerate(blocks):
        t = pl.shape[0]
        planes[0, at:at + t] = pl
        extra[0, at:at + t] = fe
        mine[0, at:at + t] = mt
        slot[0, at:at + t] = min(s, n_slots - 1)
        ppos[0, at:at + t] = np.arange(t)
        at += t
    dev = lambda x, d=None: torch.as_tensor(x, dtype=d).to(device)
    return (dev(planes), dev(extra), torch.zeros((1, T), dtype=torch.bool, device=device),
            dev(mine), dev(slot), dev(ppos))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="1,2,3,5,10")
    ap.add_argument("--gallery-games", type=int, default=12,
                    help="games per player used to build their centroid")
    ap.add_argument("--eval-players", type=int, default=5000)
    ap.add_argument("--max-len-per-game", type=int, default=0,
                    help="0 = use the checkpoint's value; set to compare a "
                         "long-context model against a truncated one")
    ap.add_argument("--gallery-mode", choices=("single", "matched"), default="single",
                    help="single: centroid = mean of single-game embeddings. "
                         "matched: for joint queries, build the centroid from "
                         "joint k-game embeddings too, so query and gallery are "
                         "the same kind of object")
    ap.add_argument("--test-pids-file", default="",
                    help="held-out player ids for a checkpoint that did not "
                         "store its own (see make_test_pids.py)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    n_slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len - 8)
    model = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS, n_game_slots=n_slots).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    wr = ck["n_planes"] == 13
    if args.max_len_per_game:
        mlpg = args.max_len_per_game
    print(f"ckpt step {ck['step']}, {n_slots} game slots, {mlpg} plies/game, "
          f"gallery-mode {args.gallery_mode}", flush=True)

    meta = np.load(os.path.join(args.shard, "meta.npy"), mmap_mode="r")
    moves = np.memmap(os.path.join(args.shard, "moves.u16"), dtype=np.uint16, mode="r")
    clocks = np.memmap(os.path.join(args.shard, "clocks.u16"), dtype=np.uint16, mode="r")

    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
    first = np.asarray(meta["offset"], dtype=np.int64)
    ok = np.concatenate([np.asarray(clocks[first]) != 0xFFFF] * 2)

    need = args.gallery_games + max(ks)
    order = np.argsort(pid, kind="stable")
    pid, gid, seat, ok = pid[order], gid[order], seat[order], ok[order]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    players = []
    for i in range(len(bnd) - 1):
        sl = slice(bnd[i], bnd[i + 1])
        k = ok[sl]
        if k.sum() >= need:
            players.append((int(pid[sl][0]), gid[sl][k], seat[sl][k]))
    print(f"{len(players):,} players with >= {need} clocked games", flush=True)

    # A SupCon fine-tune memorises the players it trained on, so scoring them
    # here would measure recall of the training set, not identification. The
    # fine-tune writes its held-out ids into the checkpoint; honour them.
    tp = ck.get("test_pids")
    if args.test_pids_file:
        tp = np.load(args.test_pids_file)
        print(f"held-out ids from {args.test_pids_file} ({len(tp):,})", flush=True)
    if tp is not None:
        keep = set(int(x) for x in np.asarray(tp))
        players = [p for p in players if p[0] in keep]
        print(f"restricted to {len(players):,} held-out players", flush=True)
    else:
        print("WARNING: checkpoint has no test_pids -- scoring ALL players, so "
              "any fine-tuned model's numbers are contaminated", flush=True)

    rng = np.random.default_rng(args.seed)
    if len(players) > args.eval_players:
        idx = rng.choice(len(players), args.eval_players, replace=False)
        players = [players[i] for i in idx]

    def embed_group(picks):
        blocks = build_games(meta, moves, clocks, picks, mlpg, ck["n_planes"], wr)
        e, _ = model.embed(*pack(blocks, device, n_slots))
        return e[0].float().cpu()

    # Split each player's games into a gallery part and a disjoint query part.
    galsets, held, clab = [], [], []
    for p, g, s in players:
        perm = rng.permutation(len(g))
        gal, qry = perm[:args.gallery_games], perm[args.gallery_games:]
        galsets.append([(int(g[j]), int(s[j])) for j in gal])
        held.append((g[qry], s[qry]))
        clab.append(p)
    CL = torch.tensor(clab).to(device)

    def build_gallery(group_k):
        """Centroid from group_k-game embeddings (group_k=1 -> single games)."""
        cent = []
        for gs in galsets:
            if group_k == 1:
                vs = [embed_group([x]) for x in gs]
            else:
                vs = [embed_group(gs[j:j + group_k])
                      for j in range(0, len(gs) - group_k + 1, group_k)]
                if not vs:                       # fewer gallery games than k
                    vs = [embed_group(gs)]
            v = torch.stack(vs).mean(0)
            cent.append(v / v.norm().clamp(min=1e-8))
            if len(cent) % 2000 == 0:
                print(f"    centroids {len(cent):,}/{len(galsets):,}", flush=True)
        return torch.stack(cent).to(device)

    C_single = build_gallery(1)
    print(f"gallery {len(CL):,} centroids (single-game)", flush=True)

    res = {"gallery": int(len(CL)), "chance@1": 1.0 / len(CL),
           "gallery_games": args.gallery_games, "mlpg": mlpg,
           "gallery_mode": args.gallery_mode, "modes": {}}
    for mode in ("joint", "average"):
        res["modes"][mode] = {}
        for k in ks:
            if mode == "joint" and k > n_slots:
                continue          # model has no slot embedding for game k
            # A joint query is one embedding of a k-game sequence; a single-game
            # gallery is a different kind of object, and comparing across that
            # mismatch penalises joint for reasons unrelated to context.
            if mode == "joint" and k > 1 and args.gallery_mode == "matched":
                print(f"  rebuilding gallery as joint {k}-game embeddings", flush=True)
                C = build_gallery(k)
            else:
                C = C_single
            qs, ql = [], []
            for (p, _, _), (gq, sq) in zip(players, held):
                if len(gq) < k:
                    continue
                sel = rng.choice(len(gq), size=k, replace=False)
                picks = [(int(gq[j]), int(sq[j])) for j in sel]
                blocks = build_games(meta, moves, clocks, picks, mlpg,
                                     ck["n_planes"], wr)
                if mode == "joint":
                    e, _ = model.embed(*pack(blocks, device, n_slots))
                    v = e[0].float().cpu()
                else:
                    vv = []
                    for bl in blocks:
                        e, _ = model.embed(*pack([bl], device, n_slots))
                        vv.append(e[0].float().cpu())
                    v = torch.stack(vv).mean(0)
                qs.append(v / v.norm().clamp(min=1e-8))
                ql.append(p)
            if not qs:
                continue
            Q = torch.stack(qs).to(device)
            QL = torch.tensor(ql).to(device)
            sim = Q @ C.T
            maxk = min(max(KS_REPORT), len(CL))
            top = sim.topk(maxk, 1).indices
            m = CL[top] == QL[:, None]
            tc = (CL[None, :] == QL[:, None]).float().argmax(1)
            rank = (sim > sim.gather(1, tc[:, None])).sum(1).float()
            r = {f"recall@{j}": float(m[:, :j].any(1).float().mean()) for j in KS_REPORT}
            r.update(median_rank=float(rank.median()), n=len(QL))
            # Normalised-rank quantiles, so this run can be compared against a
            # different gallery size after the fact: recall@k at gallery N is
            # P(normalised rank < k/N), and that CDF is a property of the
            # embedding rather than of N.
            q = torch.linspace(0, 1, 1001, device=sim.device)
            r["rank_pct_quantiles"] = [round(float(x), 6) for x in
                                       (rank / len(CL)).quantile(q)]
            res["modes"][mode][str(k)] = r
            print(f"  {mode:<8} k={k:>2}: top-1 {r['recall@1']:.4f}  "
                  f"top-10 {r['recall@10']:.4f}  top-100 {r['recall@100']:.4f}  "
                  f"med {r['median_rank']:.0f}  (n={r['n']:,})", flush=True)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("CTX_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
