"""Where does verifier training time actually go?

Splits a training step into its three real costs -- replaying boards, assembling
the batch tensor, and the model's forward/backward -- so the fix targets the one
that dominates instead of the one that looks slow.

    python profile_verify.py --shard data/2026-06-big
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from gallery_ctx import Bundles, Collate
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import N_TIME_BINS
from verify import Verifier, player_index, make_epoch


def timeit(fn, n, warmup=1):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--items", type=int, default=64)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len)
    wr = ck["n_planes"] == 13

    players = [p for p in player_index(args.shard) if len(p[0]) >= args.k + 2]
    rng = np.random.default_rng(0)
    items, labels = make_epoch(players[:4000], None, args.k, rng, args.items, 0.0)
    ds = Bundles(args.shard, items, mlpg, wr)
    coll = Collate(ck["n_planes"], slots, wr)

    print(f"device {device} | batch {args.batch} | k {args.k} | "
          f"{mlpg} plies/game cap\n")

    # 1. board replay: python-chess push + bitboard snapshot, per training item
    i = [0]
    def one_item():
        ds[i[0] % len(ds)]; i[0] += 1
    t_item = timeit(one_item, 24)
    plies = np.mean([sum(g[0].shape[0] for g in ds[j]) for j in range(8)])
    print(f"  1. board replay      {t_item*1000:7.2f} ms/item   "
          f"({plies:.0f} plies -> {plies/t_item/1000:.0f}k plies/s)")

    # 2. collate: unpackbits + tensor assembly for a whole batch
    raw = [ds[j % len(ds)] for j in range(args.batch)]
    t_coll = timeit(lambda: coll(raw), 10)
    print(f"  2. collate           {t_coll*1000:7.2f} ms/batch  "
          f"({t_coll/args.batch*1000:.2f} ms/item)")

    # 3. model forward + backward
    trunk = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS, n_game_slots=slots,
                           elo_cond=bool(ck.get("elo_cond")))
    trunk.load_state_dict(ck["model"])
    ver = Verifier(trunk, cfg.d_model, args.k - 1).to(device)
    opt = torch.optim.AdamW(ver.parameters(), lr=1e-5)
    planes, extra, pad, mine, slot, ppos = coll(raw)
    planes, extra, pad = planes.to(device), extra.to(device), pad.to(device)
    slot, ppos = slot.to(device), ppos.to(device)
    tgt = torch.zeros(planes.shape[0], device=device)

    def fwd_only():
        with torch.no_grad():
            ver(planes, extra, pad, slot, ppos)
    def fwd_bwd():
        lo = ver(planes, extra, pad, slot, ppos)
        loss = F.binary_cross_entropy_with_logits(lo.float(), tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    t_fwd = timeit(fwd_only, 5)
    t_step = timeit(fwd_bwd, 5)
    print(f"  3. model forward     {t_fwd*1000:7.2f} ms/batch")
    print(f"     forward+backward  {t_step*1000:7.2f} ms/batch  "
          f"({t_step/args.batch*1000:.2f} ms/item)")

    print(f"\n  sequence length: {planes.shape[1]} positions "
          f"({args.k} games x up to {mlpg})")

    data_per_item = t_item + t_coll / args.batch
    print(f"\n  per item: data {data_per_item*1000:6.2f} ms | "
          f"model {t_step/args.batch*1000:6.2f} ms")
    print(f"  data:model ratio {data_per_item/(t_step/args.batch):.2f}:1")
    for w in (8, 16, 32):
        supply = w / data_per_item
        demand = args.batch / t_step
        print(f"    {w:>2} workers -> data supplies {supply:6.0f} items/s, "
              f"GPU wants {demand:6.0f} -> {'DATA-bound' if supply < demand else 'model-bound'}")

    # What fraction of each item is the query, which is identical for every
    # candidate scored against it?
    print(f"\n  of the {args.k} games per item, {args.k-1} are the query "
          f"({(args.k-1)/args.k*100:.0f}% of the work) and are re-encoded for "
          f"every candidate.")


if __name__ == "__main__":
    main()
