"""Mine each training player's negatives from the DEPLOYED gallery, not the shard.

This exists because of a measured failure. The verifier reached val AUC 0.861
against in-batch negatives and then placed a real visitor at rank 50 of 100 --
exactly a coin flip -- on the shortlist the product actually serves.

The cause is negative density. verify2 mines hard negatives from the training
shard, and that shard holds 16,070 players with enough games. Deployment ranks
against 558,735. The 64th-nearest neighbour out of 16k is a far easier negative
than the 64th out of 558k, so the model was trained on a question the product
never asks: "is this the same player, versus someone unrelated?" rather than "of
100 people who ALL look like you, which one is you?"

This script computes, for each training player, their top-N nearest GALLERY
centroids -- the same shortlist `identify()` builds at serving time. Training
against those is training on the real question.

Self is excluded from every shortlist: the player's own centroid is built from
their own games, so leaving it in would hand the model a positive labelled
negative.

    python build_shortlist.py --out play/shortlist.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from gallery_ctx import Bundles, embed_bundles
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import N_TIME_BINS


def players_with_pid(shard, min_games):
    """Like verify.player_index, but keeps the pid so names can be joined.

    player_index() drops players with no clocked games, so its list position is
    NOT the pid -- joining gallery names by that position would silently pair
    players with other players' names.
    """
    meta = np.load(os.path.join(shard, "meta.npy"), mmap_mode="r")
    clocks = np.memmap(os.path.join(shard, "clocks.u16"), dtype=np.uint16, mode="r")
    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
    ok = np.concatenate([np.asarray(clocks[np.asarray(meta["offset"], np.int64)]) != 0xFFFF] * 2)
    o = np.argsort(pid, kind="stable")
    pid, gid, seat, ok = pid[o], gid[o], seat[o], ok[o]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    out = []
    for i in range(len(bnd) - 1):
        sl = slice(bnd[i], bnd[i + 1]); m = ok[sl]
        if m.sum() >= min_games:
            out.append((int(pid[sl][0]), gid[sl][m], seat[sl][m]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--gallery", default="play/gallery_2026.npz")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--min-games", type=int, default=6)
    ap.add_argument("--players", type=int, default=0,
                    help="cap the training pool (0 = every eligible player)")
    ap.add_argument("--topn", type=int, default=128)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="play/shortlist.npz")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"]); slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len); wr = ck["n_planes"] == 13
    model = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS, n_game_slots=slots,
                           elo_cond=bool(ck.get("elo_cond"))).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    g = np.load(args.gallery, allow_pickle=True)
    C = F.normalize(torch.from_numpy(g["centroids"].astype(np.float32)), dim=-1)
    grow = {str(n).lower(): i for i, n in enumerate(g["names"])}
    print(f"gallery {C.shape[0]:,} players", flush=True)

    sn = open(os.path.join(args.shard, "players.txt"), encoding="utf-8").read().split("\n")
    ps = players_with_pid(args.shard, args.min_games)
    print(f"{len(ps):,} shard players with >= {args.min_games} clocked games", flush=True)
    rng = np.random.default_rng(0)
    if args.players and len(ps) > args.players:
        ps = [ps[i] for i in rng.choice(len(ps), args.players, replace=False)]
        print(f"  capped to {len(ps):,}", flush=True)

    bundles, pids, rows = [], [], []
    for p, gg, ss in ps:
        sel = rng.permutation(len(gg))[:args.k]
        bundles.append([(int(gg[j]), int(ss[j])) for j in sel])
        pids.append(p)
        rows.append(grow.get(sn[p].lower(), -1) if p < len(sn) else -1)
    rows = np.array(rows, np.int64)
    print(f"  {(rows >= 0).sum():,} of them are in the gallery", flush=True)

    E = embed_bundles(model, Bundles(args.shard, bundles, mlpg, wr), slots, device,
                      args.batch, args.workers, "shortlist")
    E = F.normalize(E.float(), dim=-1)

    # Chunked: the full matrix would be 16k x 559k floats = 36 GB.
    N = args.topn
    short = np.zeros((len(E), N), np.int32)
    Cd = C.to(device)
    step = 256
    for i in range(0, len(E), step):
        sim = E[i:i + step].to(device) @ Cd.T
        # Drop self BEFORE topk, so every stored row is a genuine negative.
        for r in range(sim.shape[0]):
            if rows[i + r] >= 0:
                sim[r, rows[i + r]] = -2.0
        short[i:i + step] = sim.topk(N, dim=1).indices.cpu().numpy()
        if i % 4096 == 0:
            print(f"  shortlists {i:,}/{len(E):,}", flush=True)

    np.savez_compressed(args.out, shortlist=short, pid=np.array(pids, np.int64),
                        gal_row=rows, topn=N, min_games=args.min_games)
    print(f"\nwrote {args.out}: {len(short):,} players x {N} negatives "
          f"({os.path.getsize(args.out)/1e6:.1f} MB)")
    print("SHORTLIST_DONE")


if __name__ == "__main__":
    main()
