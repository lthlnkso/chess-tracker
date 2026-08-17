"""Build ONE centroid per player from every month of 2026 at once.

A single-month gallery is not a smaller version of the product, it is a biased
one. Player ids are per-shard, so each month is its own namespace and a player
only exists in the months they happened to play. Measured on the account we
tested with: 3 games in January but 274 across the year, which put them below
the 13-game eligibility bar in the January gallery and made them unfindable at
any gallery size. The fix is not more sampling, it is the union.

Two things follow from unioning, both of which help:

  reach      the roster becomes everyone who played 1+0 in 2026, not everyone
             who played it in one arbitrary month
  richness   a player's games accumulate across months, so far more of them
             reach the centroid cap -- and centroid richness is worth a lot
             (+11% top-10 going 12 -> 64 games, measured)

Identity is the username string, which is stable across dumps; the per-shard
integer pid is not, and treating it as stable would silently merge unrelated
players. Case is normalised because lichess usernames are case-insensitive.

    python union_gallery.py --ckpt ckpt/final/ctx5_ft2.pt \
        --shards data/mt/2026-01 data/mt/2026-02 ... --out gallery_2026.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from bitboards import decode_move, n_planes_compact
from fastboard import N_BB, snapshot_bb, rights_bb
from timefeat import time_features, N_TIME_BINS
from model import MultiTaskModel, Config, N_ELO_BINS
from gallery_ctx import Collate

import chess


class UnionBundles(Dataset):
    """One item = one player's k-game bundle, drawn from ANY of the shards.

    Each bundle entry is (shard_idx, game_idx, seat). Memmaps are opened lazily
    per worker: a DataLoader forks after construction, and sharing an open
    memmap across processes is what turns a fast loader into a mysterious slow
    one.
    """

    def __init__(self, shards, bundles, mlpg, with_rights):
        self.shards = list(shards)
        self.bundles = bundles
        self.mlpg = mlpg
        self.with_rights = with_rights
        self.n_planes = n_planes_compact(with_rights)
        self._open = None

    def _maps(self, si):
        if self._open is None:
            self._open = {}
        if si not in self._open:
            s = self.shards[si]
            self._open[si] = (
                np.load(os.path.join(s, "meta.npy"), mmap_mode="r"),
                np.memmap(os.path.join(s, "moves.u16"), dtype=np.uint16, mode="r"),
                np.memmap(os.path.join(s, "clocks.u16"), dtype=np.uint16, mode="r"))
        return self._open[si]

    def __len__(self):
        return len(self.bundles)

    def _game(self, si, gi, seat):
        meta, moves, clocks = self._maps(si)
        row = meta[gi]
        o, n = int(row["offset"]), int(row["nply"])
        codes = np.asarray(moves[o:o + n])
        clk = np.asarray(clocks[o:o + n])
        T = min(len(codes), self.mlpg)
        pov = chess.WHITE if seat == 0 else chess.BLACK
        pos = np.zeros((T, N_BB), np.uint64)
        rgt = np.zeros((T, 5), np.int16); rgt[:, 4] = -1
        b = chess.Board()
        for t in range(T):
            pos[t] = snapshot_bb(b, pov)
            if self.with_rights:
                rgt[t] = rights_bb(b, pov)
            b.push(decode_move(int(codes[t])))
        if pov == chess.BLACK:
            pos = pos.byteswap()
            e = rgt[:, 4]; rgt[:, 4] = np.where(e >= 0, e ^ 56, e)
        fe, _, _ = time_features(clk, int(row["tc_base"]), int(row["tc_inc"]))
        mt = np.zeros(T, bool); mt[seat::2] = True
        return pos, rgt, fe[:T], mt

    def __getitem__(self, i):
        b = self.bundles[i]
        return [self._game(int(b[j, 0]), int(b[j, 1]), int(b[j, 2]))
                for j in range(b.shape[0])]


def _cap_per_group(gid, cap):
    """Boolean mask keeping at most `cap` rows per gid. Input must be gid-sorted."""
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    lens = np.diff(np.r_[starts, len(gid)])
    rank = np.arange(len(gid), dtype=np.int64) - np.repeat(starts, lens)
    return rank < cap


def build_roster(shards, rng, cap, min_games):
    """Global username -> up to `cap` (shard, game, seat) rows, pooled over shards.

    Vectorised deliberately. The obvious Python loop visits every game-side --
    ~290M of them across six months of 1+0 -- and materialises a tuple per row,
    which is both minutes of interpreter time and tens of GB of small objects.
    Everything here stays in numpy and is capped PER SHARD before being pooled,
    so peak memory is bounded by cap x players rather than by total games.
    """
    name2gid, shard_names = {}, []
    for sh in shards:
        with open(os.path.join(sh, "players.txt"), encoding="utf-8") as f:
            names = f.read().split("\n")
        shard_names.append(names)
        for n in names:
            k = n.lower()
            if k not in name2gid:
                name2gid[k] = len(name2gid)
    G = len(name2gid)
    print(f"  {G:,} distinct usernames across {len(shards)} shard(s)", flush=True)

    gid_l, gidx_l, si_l, seat_l = [], [], [], []
    for si, sh in enumerate(shards):
        meta = np.load(os.path.join(sh, "meta.npy"), mmap_mode="r")
        clocks = np.memmap(os.path.join(sh, "clocks.u16"), dtype=np.uint16, mode="r")
        first = np.asarray(meta["offset"], dtype=np.int64)
        gi = np.flatnonzero(np.asarray(clocks[first]) != 0xFFFF).astype(np.int32)
        p2g = np.fromiter((name2gid[n.lower()] for n in shard_names[si]),
                          dtype=np.int32, count=len(shard_names[si]))
        wp = np.asarray(meta["white_pid"])[gi]
        bp = np.asarray(meta["black_pid"])[gi]
        gid = np.concatenate([p2g[wp], p2g[bp]])
        gidx = np.concatenate([gi, gi])
        seat = np.concatenate([np.zeros(len(gi), np.int8), np.ones(len(gi), np.int8)])
        order = np.argsort(gid, kind="stable")
        gid, gidx, seat = gid[order], gidx[order], seat[order]
        keep = _cap_per_group(gid, cap)
        gid_l.append(gid[keep]); gidx_l.append(gidx[keep]); seat_l.append(seat[keep])
        si_l.append(np.full(int(keep.sum()), si, np.int8))
        print(f"  {os.path.basename(sh)}: {len(meta):,} games, "
              f"{int(keep.sum()):,} rows kept", flush=True)

    gid = np.concatenate(gid_l); gidx = np.concatenate(gidx_l)
    seat = np.concatenate(seat_l); si_a = np.concatenate(si_l)
    del gid_l, gidx_l, seat_l, si_l

    order = np.argsort(gid, kind="stable")
    gid, gidx, seat, si_a = gid[order], gidx[order], seat[order], si_a[order]
    counts = np.bincount(gid, minlength=G)
    # Cap again across the pooled set, and drop anyone under the floor.
    keep = _cap_per_group(gid, cap) & (counts[gid] >= min_games)
    gid, gidx, seat, si_a = gid[keep], gidx[keep], seat[keep], si_a[keep]

    gid2name = np.empty(G, dtype=object)
    for n, g in name2gid.items():
        gid2name[g] = n
    return gid, si_a, gidx, seat, gid2name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gallery-games", type=int, default=64,
                    help="CAP per player, pooled across every month")
    ap.add_argument("--min-games", type=int, default=13,
                    help="floor across the WHOLE year, not per month")
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # Same clamp as gallery_ctx.py, and it matters more here: this is the
    # longest CPU-bound job we run (558k players, millions of bundles), and
    # nproc reports the HOST's cores inside a RunPod container. Trusting it asks
    # for 24 workers on ~10 real ones and starves the GPU -- measured 28
    # bundles/s against 556/s once clamped.
    from cpuquota import cpu_quota
    if args.workers > cpu_quota():
        print(f"workers {args.workers} -> {cpu_quota()} (cgroup quota)", flush=True)
        args.workers = cpu_quota()
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
    print(f"model step {ck['step']}, {n_slots} slots, {mlpg} plies/game", flush=True)

    rng = np.random.default_rng(args.seed)
    print(f"building roster over {len(args.shards)} shard(s)...", flush=True)
    gid, si_a, gidx, seat, gid2name = build_roster(
        args.shards, rng, args.gallery_games, args.min_games)
    uniq, starts = np.unique(gid, return_index=True)
    lens = np.diff(np.r_[starts, len(gid)])
    names = [gid2name[g] for g in uniq]
    print(f"{len(names):,} players with >= {args.min_games} clocked games in the union",
          flush=True)

    k = args.k
    # Whole k-game chunks per player; a short tail is dropped, and a player with
    # fewer than k games still gets one padded-out bundle rather than vanishing.
    n_chunks = np.maximum(lens // k, 1)
    total = int(n_chunks.sum())
    bundles = np.zeros((total, k, 3), np.int32)
    owner = np.zeros(total, np.int64)
    sizes = np.zeros(len(names), np.int64)
    at = 0
    for p in range(len(names)):
        s0, n = int(starts[p]), int(lens[p])
        for c in range(int(n_chunks[p])):
            lo = s0 + c * k
            hi = min(lo + k, s0 + n)
            m = hi - lo
            bundles[at, :m, 0] = si_a[lo:hi]
            bundles[at, :m, 1] = gidx[lo:hi]
            bundles[at, :m, 2] = seat[lo:hi]
            if m < k:                      # repeat the last game to fill
                bundles[at, m:] = bundles[at, m - 1]
            owner[at] = p
            at += 1
        sizes[p] = int(n_chunks[p]) * k
    print(f"{total:,} bundles | games per centroid: mean {sizes.mean():.1f} "
          f"median {np.median(sizes):.0f} max {sizes.max()}", flush=True)

    ds = UnionBundles(args.shards, bundles, mlpg, wr)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    collate_fn=Collate(ds.n_planes, n_slots, wr), pin_memory=True)
    out, done = [], 0
    import time as _t
    t0 = _t.time()
    with torch.no_grad():
        for planes, extra, pad, mine, slot, ppos in dl:
            e, _ = model.embed(planes.to(device), extra.to(device), pad.to(device),
                               mine.to(device), slot.to(device), ppos.to(device))
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
            done += planes.shape[0]
            if done % (args.batch * 200) < args.batch:
                print(f"    {done:,}/{len(ds):,} ({done/max(_t.time()-t0,1e-9):.0f}/s)",
                      flush=True)
    E = torch.cat(out)

    C = torch.zeros(len(names), E.shape[1])
    C.index_add_(0, torch.from_numpy(owner), E)
    C = C / C.norm(dim=1, keepdim=True).clamp(min=1e-8)

    np.savez_compressed(args.out, centroids=C.numpy().astype(np.float16),
                        names=np.array(names, dtype=object),
                        pids=np.arange(len(names), dtype=np.int64),
                        k=args.k, centroid_games=sizes,
                        ckpt=os.path.basename(args.ckpt),
                        shards=np.array([os.path.basename(s) for s in args.shards],
                                        dtype=object))
    print(f"wrote {args.out}: {len(names):,} centroids x {C.shape[1]}d, "
          f"{os.path.getsize(args.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
