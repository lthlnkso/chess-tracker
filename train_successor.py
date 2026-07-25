"""Train the successor-state scorer.

    /workspace/venv/bin/python train_successor.py --shard /workspace/data/2013-01 \
        --out /workspace/ckpt/successor --steps 6000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from successor_data import SuccessorDataset, MultiShardSuccessorDataset, collate
from model import SuccessorScorer, Config, successor_loss
from train import split_by_game, lr_at
import numpy as np
from torch.utils.data import Subset


def to_dev(b, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> dict:
    model.eval()
    tot = {"loss": 0.0, "acc": 0.0, "chance": 0.0, "cands": 0.0}
    tot_n = 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        b = to_dev(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["planes"], b["cands"], b["ply_idx"])
            loss, st = successor_loss(logits, b["label"], b["cand_mask"], b["ply_mask"])
        n = st["n"]
        tot["loss"] += loss.item() * n
        for k in ("acc", "chance", "cands"):
            tot[k] += st[k] * n
        tot_n += n
    model.train()
    return {k: v / tot_n for k, v in tot.items()}


def split_multishard(ds, val_frac: float, seed: int = 0):
    """Hold out whole games across shards: both seats must land on the same side.

    Hashes (shard, game) rather than permuting, because the index spans hundreds
    of millions of rows and the pair -- not the row -- is the unit of leakage.
    """
    idx = ds.idx
    key = (np.asarray(idx["shard"], dtype=np.uint64) << np.uint64(32)) | \
        np.asarray(idx["game"], dtype=np.uint64)
    h = (key * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(seed)
    h ^= h >> np.uint64(29)
    is_val = (h % np.uint64(100000)) < np.uint64(int(val_frac * 100000))
    # Keep the indices as numpy arrays, NOT .tolist(). A Python list of 248M ints
    # is ~9 GB of individually refcounted objects; every forked dataloader worker
    # that reads one dirties its page, so copy-on-write silently multiplies that
    # by the worker count. Measured at 0.39 GB/min anon growth -- enough to blow a
    # 116 GB cgroup partway through a 5-hour run. A numpy array has no per-element
    # refcount, so reads stay shared.
    return (Subset(ds, np.flatnonzero(~is_val)),
            Subset(ds, np.flatnonzero(is_val)))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shard", help="single month shard")
    src.add_argument("--combined", help="combine.py output spanning several months")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop after this much wall clock (0 = run all --steps)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--plies-per-game", type=int, default=12)
    ap.add_argument("--n-cand", type=int, default=16)
    ap.add_argument("--no-rights", action="store_true",
                    help="8-plane encoding: drop castling rights and en passant")
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=30)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    with_rights = not args.no_rights
    kw = dict(max_len=args.max_len, plies_per_game=args.plies_per_game,
              n_cand=args.n_cand, with_rights=with_rights)
    if args.combined:
        ds = MultiShardSuccessorDataset(args.combined, **kw)
        train_ds, val_ds = split_multishard(ds, args.val_frac, args.seed)
    else:
        ds = SuccessorDataset(args.shard, **kw)
        train_ds, val_ds = split_by_game(ds, args.val_frac, args.seed)
    print(f"train {len(train_ds):,} game-sides | val {len(val_ds):,} | "
          f"players {len(ds.players):,}")

    common = dict(batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
                  pin_memory=True, persistent_workers=args.workers > 0, prefetch_factor=4)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(val_ds, shuffle=False, **common)

    cfg = Config(d_model=args.d_model, n_layers=args.layers, max_len=args.max_len)
    model = SuccessorScorer(cfg, n_planes=ds.n_planes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params  (d_model={cfg.d_model}, layers={cfg.n_layers}, "
          f"planes={ds.n_planes}, cands={args.n_cand}, plies/game={args.plies_per_game})")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    history = []
    curve = []
    stop = False
    best_acc, best_step = -1.0, 0
    step = 0
    t0 = time.time()
    run_loss = run_acc = run_chance = 0.0
    log_every = 100

    model.train()
    while step < args.steps and not stop:
        for b in train_dl:
            if step >= args.steps:
                break
            if args.max_hours and (time.time() - t0) >= args.max_hours * 3600:
                print(f"reached --max-hours {args.max_hours} at step {step}", flush=True)
                stop = True
                break
            # With a wall-clock budget the step count is not known up front, so
            # drive the cosine off elapsed time instead -- otherwise the schedule
            # is sized to a guessed --steps and the LR never finishes decaying.
            if args.max_hours:
                budget = args.max_hours * 3600
                frac = min(1.0, (time.time() - t0) / budget)
                virt_total = 100_000
                virt_step = int(frac * virt_total)
                lr_now = lr_at(max(virt_step, min(step, args.warmup)),
                               virt_total, args.lr, args.warmup)
            else:
                lr_now = lr_at(step, args.steps, args.lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr_now

            b = to_dev(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b["planes"], b["cands"], b["ply_idx"])
                loss, st = successor_loss(logits, b["label"], b["cand_mask"], b["ply_mask"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run_loss += loss.item()
            run_acc += st["acc"]
            run_chance += st["chance"]
            step += 1

            if step % log_every == 0:
                dt = time.time() - t0
                print(f"step {step:>6}/{args.steps} | loss {run_loss/log_every:6.3f} | "
                      f"acc {run_acc/log_every:6.3f} (chance {run_chance/log_every:5.3f}) | "
                      f"{step/dt:5.1f} it/s | {dt/60:5.1f} min", flush=True)
                curve.append({"step": step, "loss": run_loss / log_every,
                              "acc": run_acc / log_every, "chance": run_chance / log_every,
                              "minutes": round(dt / 60, 3)})
                run_loss = run_acc = run_chance = 0.0

            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(model, val_dl, device, args.eval_batches)
                best = ev["acc"] > best_acc
                print(f"  >> val @ {step}: loss {ev['loss']:.3f}  acc {ev['acc']:.3f}  "
                      f"chance {ev['chance']:.3f}  mean cands {ev['cands']:.1f}"
                      f"{'  *best*' if best else f'  (best {best_acc:.3f} @ {best_step})'}",
                      flush=True)
                history.append({"step": step, **ev})
                blob = {"model": model.state_dict(), "cfg": cfg.__dict__,
                        "n_cand": args.n_cand, "n_planes": ds.n_planes,
                        "step": step, "val": ev}
                torch.save(blob, os.path.join(args.out, "last.pt"))
                # Keep the best-by-val checkpoint separately. `last.pt` is
                # whatever the run happened to end on, which is the wrong thing
                # to hand downstream if the tail of the run degraded.
                if best:
                    best_acc, best_step = ev["acc"], step
                    torch.save(blob, os.path.join(args.out, "best.pt"))

    if not history or history[-1]["step"] != step:
        ev = evaluate(model, val_dl, device, args.eval_batches)
        print(f"  >> val @ {step} (final): loss {ev['loss']:.3f}  acc {ev['acc']:.3f}",
              flush=True)
        history.append({"step": step, **ev})
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                    "n_cand": args.n_cand, "n_planes": ds.n_planes,
                    "step": step, "val": ev}, os.path.join(args.out, "last.pt"))

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"args": vars(args), "params": n_params, "n_planes": ds.n_planes,
                   "best_acc": best_acc, "best_step": best_step,
                   "history": history, "curve": curve,
                   "minutes": round((time.time() - t0) / 60, 1)}, f, indent=2)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.out}/last.pt")


if __name__ == "__main__":
    main()
