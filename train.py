"""Pretrain the encoder on next-move prediction.

    /workspace/venv/bin/python train.py --shard /workspace/data/2013-01 \
        --out /workspace/ckpt/nextmove --steps 6000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import PlayerGameDataset, collate
from model import ChessTransformer, Config, loss_and_stats


def split_by_game(ds, val_frac: float, seed: int = 0):
    """Hold out whole games: both seats of a game must land on the same side."""
    n_games = len(ds.meta)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_games)
    n_val = int(n_games * val_frac)
    val_games = np.zeros(n_games, dtype=bool)
    val_games[perm[:n_val]] = True
    is_val = val_games[ds.index[:, 0]]
    return (
        Subset(ds, np.flatnonzero(~is_val).tolist()),
        Subset(ds, np.flatnonzero(is_val).tolist()),
    )


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * base + 0.9 * base * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> dict:
    model.eval()
    tot_loss = tot_acc = tot_top5 = 0.0
    tot_n = 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        planes = b["planes"].to(device, non_blocking=True)
        moves = b["moves"].to(device, non_blocking=True)
        pad = b["pad_mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lm, lp = model(planes)
            loss, st = loss_and_stats(lm, lp, moves, pad)
        tot_loss += loss.item() * st["n"]
        tot_acc += st["acc"] * st["n"]
        tot_top5 += st["top5"] * st["n"]
        tot_n += st["n"]
    model.train()
    return {"loss": tot_loss / tot_n, "acc": tot_acc / tot_n, "top5": tot_top5 / tot_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    ds = PlayerGameDataset(args.shard, max_len=args.max_len)
    train_ds, val_ds = split_by_game(ds, args.val_frac, args.seed)
    print(f"train {len(train_ds):,} game-sides | val {len(val_ds):,} | players {len(ds.players):,}")

    common = dict(batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
                  pin_memory=True, persistent_workers=args.workers > 0, prefetch_factor=4)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(val_ds, shuffle=False, **common)

    cfg = Config(d_model=args.d_model, n_layers=args.layers, max_len=args.max_len)
    model = ChessTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params  (d_model={cfg.d_model}, layers={cfg.n_layers})")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    history = []
    step = 0
    t0 = time.time()
    run_loss = run_acc = 0.0
    log_every = 100

    model.train()
    while step < args.steps:
        for b in train_dl:
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)

            planes = b["planes"].to(device, non_blocking=True)
            moves = b["moves"].to(device, non_blocking=True)
            pad = b["pad_mask"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                lm, lp = model(planes)
                loss, st = loss_and_stats(lm, lp, moves, pad)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run_loss += loss.item()
            run_acc += st["acc"]
            step += 1

            if step % log_every == 0:
                dt = time.time() - t0
                print(f"step {step:>6}/{args.steps} | loss {run_loss/log_every:6.3f} | "
                      f"acc {run_acc/log_every:6.3f} | lr {opt.param_groups[0]['lr']:.2e} | "
                      f"{step/dt:5.1f} it/s | {dt/60:5.1f} min", flush=True)
                run_loss = run_acc = 0.0

            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(model, val_dl, device, args.eval_batches)
                print(f"  >> val @ {step}: loss {ev['loss']:.3f}  acc {ev['acc']:.3f}  "
                      f"top5 {ev['top5']:.3f}", flush=True)
                history.append({"step": step, **ev})
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                            "step": step, "val": ev}, os.path.join(args.out, "last.pt"))

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"args": vars(args), "params": n_params, "history": history,
                   "minutes": round((time.time() - t0) / 60, 1)}, f, indent=2)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.out}/last.pt")


if __name__ == "__main__":
    main()
