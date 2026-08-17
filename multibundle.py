"""Does 10 games beat 5, if we stop halving the centroid?

The colour-split arm lost twice, but it changed two things at once: it matched
colours AND it split each centroid in half. This isolates the other half of that
idea -- more query evidence, same full centroid.

The model has five game slots, so a single query bundle caps at five games. Two
bundles of five scored against the SAME combined centroid and summed uses ten
games of evidence with no centroid penalty at all. If the colour arm's loss was
the halving rather than the fusion, this should win.

    python multibundle.py --ckpt ckpt/final/ctx5_ft2.pt --shard data/2026-06-big
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from gallery_ctx import Bundles, embed_bundles, Collate  # noqa: F401
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import N_TIME_BINS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--players", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cap", type=int, default=30, help="gallery games per player")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
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

    meta = np.load(f"{args.shard}/meta.npy", mmap_mode="r")
    clocks = np.memmap(f"{args.shard}/clocks.u16", dtype=np.uint16, mode="r")
    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
    ok = np.concatenate([np.asarray(clocks[np.asarray(meta["offset"], np.int64)]) != 0xFFFF] * 2)
    o = np.argsort(pid, kind="stable")
    pid, gid, seat, ok = pid[o], gid[o], seat[o], ok[o]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])

    k = args.k
    need = 2 * k + k                      # two query bundles + a centroid chunk
    players = []
    for i in range(len(bnd) - 1):
        sl = slice(bnd[i], bnd[i + 1]); m = ok[sl]
        g, s = gid[sl][m], seat[sl][m]
        if len(g) >= need:
            players.append((g, s))
    print(f"{len(players):,} players with >= {need} clocked games", flush=True)
    rng = np.random.default_rng(args.seed)
    if len(players) > args.players:
        players = [players[i] for i in rng.choice(len(players), args.players, replace=False)]
    P = len(players)

    gal, owner, q1, q2 = [], [], [], []
    for idx, (g, s) in enumerate(players):
        perm = rng.permutation(len(g))
        qa = perm[:k]                     # first query bundle
        qb = perm[k:2 * k]                # second, disjoint
        rest = perm[2 * k:2 * k + args.cap]
        chunks = [rest[j:j + k] for j in range(0, len(rest) - k + 1, k)] or [rest[:k]]
        for c in chunks:
            gal.append([(int(g[j]), int(s[j])) for j in c]); owner.append(idx)
        q1.append([(int(g[j]), int(s[j])) for j in qa])
        q2.append([(int(g[j]), int(s[j])) for j in qb])
    print(f"{len(gal):,} gallery bundles | {P:,} players", flush=True)

    def emb(bundles, label):
        return embed_bundles(model, Bundles(args.shard, bundles, mlpg, wr),
                             slots, device, args.batch, args.workers, label)

    GE = emb(gal, "gallery")
    C = torch.zeros(P, GE.shape[1]); C.index_add_(0, torch.tensor(owner), GE)
    C = C / C.norm(dim=1, keepdim=True).clamp(min=1e-8)

    nq = min(args.queries, P)
    sel = rng.choice(P, nq, replace=False)
    A = emb([q1[i] for i in sel], "query A")
    B = emb([q2[i] for i in sel], "query B")
    nrm = lambda X: X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    A, B = nrm(A), nrm(B)

    def report(sim, label, games):
        true = torch.tensor(sel)
        beat = (sim > sim.gather(1, true[:, None])).sum(1).numpy()
        r1 = float((beat < 1).mean()); r10 = float((beat < 10).mean())
        print(f"  {label:<28} ({games:>2} games)  r@1 {r1:.4f}  r@10 {r10:.4f}  "
              f"median rank {np.median(beat):.0f}")
        return r10

    print(f"\ngallery {P:,} players | {nq} queries\n")
    sA = A @ C.T
    sB = B @ C.T
    report(sA, "one bundle", k)
    report((sA + sB) / 2, "two bundles, score sum", 2 * k)
    # Averaging the query VECTORS before scoring is the other way to fuse, and
    # is not the same operation -- it blurs two embeddings into one point rather
    # than adding two independent votes.
    report(nrm(A + B) @ C.T, "two bundles, mean vector", 2 * k)
    print("\nMULTIBUNDLE_DONE")


if __name__ == "__main__":
    main()
