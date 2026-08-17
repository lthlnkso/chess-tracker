"""Separate the three things that can limit a training step.

Neither probe_scale.py nor the training logs answer "is the GPU or the loader the
limit?", because both measure the two together. This times them apart:

    GPU-only    one batch cached on the device, replayed -- the ceiling a
                perfect loader could reach
    loader-only iterate the DataLoader, never touch the GPU
    end-to-end  what training actually gets

The comparison is the point. If end-to-end tracks loader-only, more dataloader
work pays; if it tracks GPU-only, it cannot, and a faster move generator buys
nothing but a hotter CPU.

    python bench_split.py --shard /workspace/data/mt/2026-01 --workers 24
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from successor_data import MultiTaskDataset, collate_multitask
from model import MultiTaskModel, Config, multitask_loss, N_ELO_BINS
from timefeat import N_TIME_FEATS, N_TIME_BINS, TIME_CENTRES


def make_loader(ds, rows, batch, workers):
    return DataLoader(Subset(ds, rows.tolist()), batch_size=batch, shuffle=True,
                      drop_last=True, num_workers=workers,
                      collate_fn=collate_multitask, pin_memory=True,
                      persistent_workers=workers > 0, prefetch_factor=4 if workers else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cands", default="16,32")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default="/workspace/ckpt/bench_split.json")
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    lossk = dict(time_centres=torch.as_tensor(TIME_CENTRES))
    results = []

    for C in [int(x) for x in args.cands.split(",")]:
        ds = MultiTaskDataset(args.shard, max_len=160, plies_per_game=12, n_cand=C)
        rows = np.random.default_rng(0).choice(len(ds), 200_000, replace=False)
        cfg = Config(d_model=args.d_model, n_layers=args.layers,
                     n_heads=max(8, args.d_model // 64), d_ff=args.d_model * 4)
        m = MultiTaskModel(cfg, n_planes=ds.n_planes, n_extra=N_TIME_FEATS,
                           n_time_bins=N_TIME_BINS, n_elo_bins=N_ELO_BINS).to(dev)
        if args.compile:
            # dynamic=True: T varies per batch (games differ in length), and
            # static shapes would recompile on every new sequence length.
            m = torch.compile(m, dynamic=True)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def step(b):
            # bf16 rather than fp16: no loss scaler, and the successor-scoring
            # logits are a scaled dot product whose range fp16 handles poorly.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                out = m(b["planes"], b["extra"], b["cands"], b["ply_idx"],
                        b["pad_mask"], b["my_turn"])
                loss, _ = multitask_loss(out[0], out[1], out[2], b, **lossk)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # --- GPU-only: one batch, already resident, replayed
        dl = make_loader(ds, rows, args.batch, args.workers)
        it = iter(dl)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in next(it).items()}
        for _ in range(15 if args.compile else 8):
            step(b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            step(b)
        torch.cuda.synchronize()
        gpu_only = args.steps / (time.perf_counter() - t0)
        del it, dl

        # --- loader-only: never touch the GPU
        dl = make_loader(ds, rows, args.batch, args.workers)
        it = iter(dl)
        for _ in range(5):
            next(it)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            next(it)
        loader_only = args.steps / (time.perf_counter() - t0)
        del it, dl

        # --- end-to-end
        dl = make_loader(ds, rows, args.batch, args.workers)
        it = iter(dl)
        for _ in range(5):
            bb = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                  for k, v in next(it).items()}
            step(bb)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            bb = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                  for k, v in next(it).items()}
            step(bb)
        torch.cuda.synchronize()
        e2e = args.steps / (time.perf_counter() - t0)

        r = {"n_cand": C, "batch": args.batch, "workers": args.workers, "amp": args.amp, "compiled": args.compile,
             "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
             "gpu_only_it_s": round(gpu_only, 2),
             "loader_only_it_s": round(loader_only, 2),
             "end_to_end_it_s": round(e2e, 2),
             "gpu_util_pct": round(100 * e2e / gpu_only, 1),
             "headroom_if_loader_free": round(gpu_only / e2e, 2)}
        results.append(r)
        print(json.dumps(r), flush=True)
        del m, opt, it, dl
        torch.cuda.empty_cache()

    print(f"\n{'n_cand':>7}{'GPU-only':>10}{'loader':>9}{'actual':>9}"
          f"{'GPU util':>10}{'max speedup':>13}")
    for r in results:
        print(f"{r['n_cand']:>7}{r['gpu_only_it_s']:>10.1f}{r['loader_only_it_s']:>9.1f}"
              f"{r['end_to_end_it_s']:>9.1f}{r['gpu_util_pct']:>9.1f}%"
              f"{r['headroom_if_loader_free']:>13.2f}x")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("BENCH_SPLIT_DONE")


if __name__ == "__main__":
    main()
