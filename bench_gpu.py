"""Benchmark one GPU for successor-scorer pre-training.

Measures two numbers, because they can differ a lot and only one of them is
about the GPU:

  gpu_only   -- steps/s replaying a cached batch. Pure device throughput, no
                dataloader in the loop. This is what a GPU comparison means.
  end2end    -- steps/s with the real dataloader feeding it. What you actually
                pay for. If this is far below gpu_only the run is CPU-bound and
                the GPU choice barely matters -- which is a finding, not a bug.

Writes a JSON blob so the plotting step never has to parse logs.

    python bench_gpu.py --shard /root/data/2013-01 --gpu "RTX 4090" \
        --price 0.34 --out /root/bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from successor_data import SuccessorDataset, collate
from model import SuccessorScorer, Config, successor_loss


def train_step(model, opt, b):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(b["planes"], b["cands"], b["ply_idx"])
        loss, _ = successor_loss(logits, b["label"], b["cand_mask"], b["ply_mask"])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--gpu", required=True, help="label for the plot")
    ap.add_argument("--price", type=float, required=True, help="$/hr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--plies-per-game", type=int, default=12)
    ap.add_argument("--n-cand", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--e2e-steps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=0, help="0 = min(16, vCPU-2)")
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    # Inside a container os.cpu_count() reports the host's cores, so cap it.
    vcpu = os.cpu_count() or 8
    workers = args.workers or min(16, max(2, vcpu - 2))

    ds = SuccessorDataset(args.shard, max_len=args.max_len,
                          plies_per_game=args.plies_per_game, n_cand=args.n_cand)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=workers, collate_fn=collate, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)

    cfg = Config(d_model=args.d_model, n_layers=args.layers, max_len=args.max_len)
    model = SuccessorScorer(cfg, n_planes=ds.n_planes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    model.train()

    it = iter(dl)
    cached = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in next(it).items()}

    # --- gpu_only: same batch, over and over ---
    for _ in range(args.warmup):
        train_step(model, opt, cached)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.steps):
        train_step(model, opt, cached)
    torch.cuda.synchronize()
    gpu_dt = time.time() - t0
    gpu_ips = args.steps / gpu_dt

    # --- end2end: real dataloader ---
    for _ in range(10):                       # let workers spin up
        b = next(it)
        train_step(model, opt, {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                                for k, v in b.items()})
    torch.cuda.synchronize()
    t0 = time.time()
    done = 0
    for b in it:
        train_step(model, opt, {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                                for k, v in b.items()})
        done += 1
        if done >= args.e2e_steps:
            break
    torch.cuda.synchronize()
    e2e_dt = time.time() - t0
    e2e_ips = done / e2e_dt

    gs = args.batch                            # game-sides per step
    res = {
        "gpu": args.gpu,
        "device_name": torch.cuda.get_device_name(0),
        "price_per_hr": args.price,
        "vcpu": vcpu,
        "workers": workers,
        "batch": args.batch,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "gpu_only_it_s": round(gpu_ips, 3),
        "end2end_it_s": round(e2e_ips, 3),
        "gpu_only_gamesides_s": round(gpu_ips * gs, 1),
        "end2end_gamesides_s": round(e2e_ips * gs, 1),
        "dataloader_bound_pct": round(100 * (1 - e2e_ips / gpu_ips), 1),
        # headline for the plot: $ to push 1M game-sides through pre-training
        "usd_per_m_gamesides_e2e": round(args.price / (e2e_ips * gs * 3600 / 1e6), 4),
        "usd_per_m_gamesides_gpu": round(args.price / (gpu_ips * gs * 3600 / 1e6), 4),
        "torch": torch.__version__,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
