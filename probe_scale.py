"""Measure real GPU throughput and VRAM for candidate model sizes.

Sizing an overnight run from a guess is how you discover at 06:00 that the big
model saw a third of the samples the small one did. This measures steps/sec and
peak memory for each candidate so the time split can be chosen from data.

    python probe_scale.py --shard /workspace/data/mt/2026-01 --steps 40
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from successor_data import MultiTaskDataset, collate_multitask
from model import MultiTaskModel, Config, multitask_loss, supcon_loss, N_ELO_BINS
from timefeat import N_TIME_FEATS, N_TIME_BINS, TIME_CENTRES

CANDIDATES = [(256, 8, "1x  current"), (512, 12, "5x"), (768, 12, "12x"), (640, 12, "8x")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--out", default="/workspace/ckpt/scale_probe.json")
    args = ap.parse_args()
    dev = "cuda"

    ds = MultiTaskDataset(args.shard, max_len=160, plies_per_game=12, n_cand=16)
    rows = np.random.default_rng(0).choice(len(ds), 200_000, replace=False)
    dl = DataLoader(Subset(ds, rows.tolist()), batch_size=args.batch, shuffle=True,
                    drop_last=True, num_workers=args.workers,
                    collate_fn=collate_multitask, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    lossk = dict(time_centres=torch.as_tensor(TIME_CENTRES))

    results = []
    for d, L, lbl in CANDIDATES:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        cfg = Config(d_model=d, n_layers=L, n_heads=max(8, d // 64), d_ff=d * 4,
                     max_len=160)
        m = MultiTaskModel(cfg, n_planes=ds.n_planes, n_extra=N_TIME_FEATS,
                           d_embed=128, n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS).to(dev)
        n = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        it = iter(dl)
        try:
            # pre-train step
            for i in range(args.steps + 5):
                if i == 5:
                    torch.cuda.synchronize(); t0 = time.time()
                b = next(it)
                b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in b.items()}
                ml, tl, el, _, _ = m(b["planes"], b["extra"], b["cands"],
                                     b["ply_idx"], b["pad_mask"], b["my_turn"])
                loss, _ = multitask_loss(ml, tl, el, b, 1.0, 0.3, 0.3, **lossk)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            torch.cuda.synchronize()
            pre_its = args.steps / (time.time() - t0)

            # fine-tune step (embed + supcon only -- cheaper per step)
            for i in range(args.steps + 5):
                if i == 5:
                    torch.cuda.synchronize(); t0 = time.time()
                b = next(it)
                b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in b.items()}
                emb, _ = m.embed(b["planes"], b["extra"], b["pad_mask"], b["my_turn"])
                # Timing only: random rows almost never share a player, so use
                # synthetic P x K labels. Throughput is what we are measuring.
                lbls = (torch.arange(emb.shape[0], device=dev) //
                        4).clamp(max=emb.shape[0] // 4 - 1)
                l2, _ = supcon_loss(emb, lbls)
                opt.zero_grad(set_to_none=True); l2.backward(); opt.step()
            torch.cuda.synchronize()
            ft_its = args.steps / (time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 2**30
            results.append({"label": lbl, "d_model": d, "layers": L,
                            "params_m": round(n / 1e6, 1),
                            "pretrain_it_s": round(pre_its, 2),
                            "finetune_it_s": round(ft_its, 2),
                            "peak_vram_gb": round(peak, 1)})
            print(f"  {lbl:<12} d{d} L{L}  {n/1e6:>6.1f}M params  "
                  f"pretrain {pre_its:>5.2f} it/s  finetune {ft_its:>5.2f} it/s  "
                  f"peak {peak:.1f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  {lbl:<12} d{d} L{L}  OOM at batch {args.batch}", flush=True)
            results.append({"label": lbl, "d_model": d, "layers": L, "oom": True})
        del m, opt
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"batch": args.batch, "results": results}, f, indent=2)

    base = next((r for r in results if r.get("params_m") and r["d_model"] == 256), None)
    if base:
        print(f"\n{'model':<12}{'params':>9}{'pre it/s':>10}{'ft it/s':>9}"
              f"{'slowdown':>10}{'VRAM':>8}")
        for r in results:
            if r.get("oom"):
                continue
            sd = base["pretrain_it_s"] / r["pretrain_it_s"]
            print(f"{r['label']:<12}{r['params_m']:>8.1f}M{r['pretrain_it_s']:>10.2f}"
                  f"{r['finetune_it_s']:>9.2f}{sd:>9.1f}x{r['peak_vram_gb']:>7.1f}G")
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
