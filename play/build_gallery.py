"""Build a local centroid gallery for the demo's identification panel.

One row per player: the L2-normalised mean of their joint k-game embeddings,
exactly the "matched" gallery the eval scores against, so the demo's numbers are
the ones the sweep measured rather than a different protocol that happens to look
similar.

Centroid size is a CAP, not a requirement -- a player with 20 games gets a
20-game centroid. Requiring N would silently restrict the gallery to hyper-active
players and flatter the demo.

With --colour the file also carries separate white and black centroids built from
the SAME games as the combined ones, so the demo can score a colour-matched query
against the matching bank and fuse by score sum. Players without enough games of
a colour get a zero row and a zero count; the server must skip those rather than
score them, since a zero vector has cosine 0 with everything and would rank
ahead of genuinely negative matches.

    python play/build_gallery.py --ckpt ckpt/final/ctx5_ft2.pt \
        --shard data/2026-06-big --out play/gallery.npz --players 8000 --colour
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gallery_ctx import Bundles, embed_bundles          # noqa: E402
from model import MultiTaskModel, Config, N_ELO_BINS    # noqa: E402
from timefeat import N_TIME_BINS                        # noqa: E402


def chunk(sel, k):
    """Non-overlapping k-game chunks; a short tail is dropped, never padded."""
    out = [sel[j:j + k] for j in range(0, len(sel) - k + 1, k)]
    return out or ([sel[:k]] if len(sel) else [])


def centroids_from(model, shard, mlpg, wr, n_slots, device, batch, workers,
                   bundles, owner, n_players, label):
    """Mean of each owner's bundle embeddings, L2-normalised. Zero row if none."""
    if not bundles:
        return torch.zeros(n_players, 1), np.zeros(n_players, np.int64)
    E = embed_bundles(model, Bundles(shard, bundles, mlpg, wr),
                      n_slots, device, batch, workers, label)
    own = torch.tensor(owner)
    C = torch.zeros(n_players, E.shape[1])
    C.index_add_(0, own, E)
    n = torch.zeros(n_players)
    n.index_add_(0, own, torch.ones(len(owner)))
    C = C / C.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return C, n.numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", default="play/gallery.npz")
    ap.add_argument("--players", type=int, default=8000)
    ap.add_argument("--k", type=int, default=5, help="games per joint chunk")
    ap.add_argument("--gallery-games", type=int, default=64, help="cap per centroid")
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--colour", action="store_true",
                    help="also emit white/black centroid banks from the same games")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    n_slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len)
    wr = ck["n_planes"] == 13
    model = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS, n_game_slots=n_slots).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    meta = np.load(os.path.join(args.shard, "meta.npy"), mmap_mode="r")
    clocks = np.memmap(os.path.join(args.shard, "clocks.u16"), dtype=np.uint16, mode="r")
    with open(os.path.join(args.shard, "players.txt"), encoding="utf-8") as f:
        names = f.read().split("\n")

    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
    ok = np.concatenate([np.asarray(clocks[np.asarray(meta["offset"], np.int64)]) != 0xFFFF] * 2)

    order = np.argsort(pid, kind="stable")
    pid, gid, seat, ok = pid[order], gid[order], seat[order], ok[order]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    players = []
    for i in range(len(bnd) - 1):
        sl = slice(bnd[i], bnd[i + 1]); m = ok[sl]
        if m.sum() >= args.min_games:
            players.append((int(pid[sl][0]), gid[sl][m], seat[sl][m]))
    print(f"{len(players):,} players with >= {args.min_games} clocked games", flush=True)

    rng = np.random.default_rng(args.seed)
    if len(players) > args.players:
        players = [players[i] for i in rng.choice(len(players), args.players, replace=False)]
    P = len(players)

    bundles, owner, sizes = [], [], []
    wb, wo, bb, bo = [], [], [], []
    for idx, (p, g, s) in enumerate(players):
        perm = rng.permutation(len(g))
        if args.colour:
            # Cap PER COLOUR, then let the combined bank use the union, so the
            # two banks see the same games and any gap is the split itself.
            w = [int(j) for j in perm if s[j] == 0][:args.gallery_games]
            b = [int(j) for j in perm if s[j] == 1][:args.gallery_games]
            for c in chunk(w, args.k):
                wb.append([(int(g[j]), int(s[j])) for j in c]); wo.append(idx)
            for c in chunk(b, args.k):
                bb.append([(int(g[j]), int(s[j])) for j in c]); bo.append(idx)
            gal = w + b
        else:
            gal = [int(j) for j in perm[:min(args.gallery_games, len(g))]]
        ch = chunk(gal, args.k)
        for c in ch:
            bundles.append([(int(g[j]), int(s[j])) for j in c]); owner.append(idx)
        sizes.append(len(ch) * args.k)
    sizes = np.asarray(sizes)
    print(f"{P:,} players -> {len(bundles):,} bundles | centroid games "
          f"mean {sizes.mean():.1f} median {np.median(sizes):.0f} max {sizes.max()}", flush=True)

    C, _ = centroids_from(model, args.shard, mlpg, wr, n_slots, device,
                          args.batch, args.workers, bundles, owner, P, "gallery")

    pids = np.array([p for p, _, _ in players], dtype=np.int64)
    who = np.array([names[p] if p < len(names) else f"player{p}" for p in pids], dtype=object)
    blob = dict(centroids=C.numpy().astype(np.float16), pids=pids, names=who,
                k=args.k, centroid_games=sizes, ckpt=os.path.basename(args.ckpt))

    if args.colour:
        print(f"colour banks: {len(wb):,} white bundles, {len(bb):,} black", flush=True)
        Cw, nw = centroids_from(model, args.shard, mlpg, wr, n_slots, device,
                                args.batch, args.workers, wb, wo, P, "white")
        Cb, nb = centroids_from(model, args.shard, mlpg, wr, n_slots, device,
                                args.batch, args.workers, bb, bo, P, "black")
        blob.update(centroids_w=Cw.numpy().astype(np.float16),
                    centroids_b=Cb.numpy().astype(np.float16),
                    n_white=nw * args.k, n_black=nb * args.k)
        print(f"  players with a white bank {int((nw > 0).sum()):,} | "
              f"black {int((nb > 0).sum()):,} | both {int(((nw > 0) & (nb > 0)).sum()):,}",
              flush=True)

    np.savez_compressed(args.out, **blob)
    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}: {P:,} centroids x {C.shape[1]}d, {mb:.1f} MB")


if __name__ == "__main__":
    main()
