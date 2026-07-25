"""Fine-tune the pre-trained encoder for player identification.

Metric learning: batch-hard triplet loss over P players x K games per batch,
starting from the successor-prediction trunk.

    python finetune_id.py --combined /workspace/data/combined \
        --pretrained /workspace/ckpt/prod/last.pt --out /workspace/ckpt/id \
        --max-hours 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from id_data import EmbedDataset, PKSampler, collate, split_players
from model import PlayerEncoder, Config, batch_hard_triplet, supcon_loss
from bitboards import n_planes_compact, N_PLANES13


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.05 * base + 0.95 * base * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", required=True)
    ap.add_argument("--pretrained", default="", help="successor-scorer checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=100_000_000)
    ap.add_argument("--p", type=int, default=32, help="players per batch")
    ap.add_argument("--k", type=int, default=4, help="games per player per batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--margin", type=float, default=0.0, help="triplet only; 0 = soft margin")
    ap.add_argument("--loss", choices=("supcon", "triplet"), default="supcon",
                    help="triplet collapses: its minimum IS the constant-output solution")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--collapse-gap", type=float, default=0.02,
                    help="abort if pos_cos-neg_cos stays under this after --collapse-after")
    ap.add_argument("--collapse-after", type=int, default=2000)
    ap.add_argument("--d-embed", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--strict-split", action="store_true")
    ap.add_argument("--no-rights", action="store_true",
                    help="8-plane encoding; ignored when --pretrained sets it")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    full = np.load(os.path.join(args.combined, "index.npy"), mmap_mode="r")
    with open(os.path.join(args.combined, "players.txt"), encoding="utf-8") as f:
        n_players = len(f.read().split("\n"))
    train_rows, test_rows, is_test = split_players(
        full, n_players, args.test_frac, args.seed, args.strict_split)
    print(f"players {n_players:,}  ->  train {int((~is_test).sum()):,} / "
          f"test {int(is_test.sum()):,}", flush=True)
    print(f"game-sides: train {len(train_rows):,}  test(both-seats) {len(test_rows):,}",
          flush=True)
    np.save(os.path.join(args.out, "test_rows.npy"), test_rows)
    np.save(os.path.join(args.out, "is_test_player.npy"), is_test)

    # Resolve the encoding from the checkpoint FIRST: the loader must emit the
    # same plane count the pre-trained trunk expects, or every batch is a shape
    # error. Deciding this after building the dataset is how that bug happens.
    cfg = Config(max_len=args.max_len)
    n_planes = n_planes_compact(not args.no_rights)
    ck = None
    if args.pretrained:
        ck = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        cfg = Config(**ck["cfg"])
        n_planes = ck.get("n_planes", n_planes)

    ds = EmbedDataset(args.combined, rows=train_rows, max_len=args.max_len,
                      with_rights=n_planes == N_PLANES13)
    assert ds.n_planes == n_planes, f"loader {ds.n_planes} != model {n_planes}"
    sampler = PKSampler(ds.labels, p=args.p, k=args.k, seed=args.seed)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers,
                    collate_fn=collate, pin_memory=True, persistent_workers=True,
                    prefetch_factor=4)

    model = PlayerEncoder(cfg, n_planes=n_planes, d_embed=args.d_embed).to(device)
    if ck is not None:
        took, total = model.load_pretrained(ck["model"])
        print(f"loaded {took}/{total} tensors from {args.pretrained} "
              f"(step {ck.get('step')}, {n_planes} planes) — projection head is fresh",
              flush=True)
    else:
        print(f"training from scratch, {n_planes} planes", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    curve = []
    step = 0
    t0 = time.time()
    keys = (("pos_cos", "neg_cos", "gap", "emb_std") if args.loss == "supcon"
            else ("d_pos", "d_neg", "frac_violating"))
    run = {"loss": 0.0, **{k: 0.0 for k in keys}}
    budget = args.max_hours * 3600
    stop = False

    model.train()
    while step < args.steps and not stop:
        for b in dl:
            if step >= args.steps:
                break
            if budget and (time.time() - t0) >= budget:
                print(f"reached --max-hours {args.max_hours} at step {step}", flush=True)
                stop = True
                break

            # Time-driven cosine: the step count is not known ahead of a wall-clock run.
            frac = min(1.0, (time.time() - t0) / budget) if budget else step / args.steps
            virt = 100_000
            lr_now = lr_at(max(int(frac * virt), min(step, args.warmup)),
                           virt, args.lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr_now

            planes = b["planes"].to(device, non_blocking=True)
            pad = b["pad_mask"].to(device, non_blocking=True)
            mine = b["my_turn"].to(device, non_blocking=True)
            lbl = b["player_id"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = model(planes, pad, mine)
            if args.loss == "supcon":
                loss, st = supcon_loss(emb.float(), lbl, args.temperature)
            else:
                loss, st = batch_hard_triplet(emb.float(), lbl, args.margin)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run["loss"] += loss.item()
            for k in keys:
                run[k] += st[k]
            step += 1

            # Fail fast on embedding collapse rather than burning hours on a
            # constant-output model, which is what triplet did here.
            if args.loss == "supcon" and step == args.collapse_after:
                if st["gap"] < args.collapse_gap:
                    raise SystemExit(
                        f"COLLAPSE: pos/neg cosine gap {st['gap']:.4f} < "
                        f"{args.collapse_gap} at step {step}; aborting")
                print(f"collapse check passed at step {step}: gap {st['gap']:.3f}",
                      flush=True)

            if step % args.log_every == 0:
                dt = time.time() - t0
                m = {k: v / args.log_every for k, v in run.items()}
                if args.loss == "supcon":
                    detail = (f"pos {m['pos_cos']:+.3f} neg {m['neg_cos']:+.3f} "
                              f"gap {m['gap']:5.3f}")
                else:
                    detail = (f"d+ {m['d_pos']:5.3f} d- {m['d_neg']:5.3f} "
                              f"viol {m['frac_violating']:5.3f}")
                print(f"step {step:>7} | loss {m['loss']:6.4f} | {detail} | "
                      f"lr {lr_now:.2e} | {step/dt:5.1f} it/s | {dt/60:6.1f} min",
                      flush=True)
                curve.append({"step": step, "minutes": round(dt / 60, 3),
                              "lr": lr_now, **{k: round(v, 5) for k, v in m.items()}})
                run = {k: 0.0 for k in run}

            if step % args.ckpt_every == 0 or stop:
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                            "n_planes": n_planes, "d_embed": args.d_embed,
                            "step": step, "combined": args.combined,
                            "test_frac": args.test_frac, "seed": args.seed},
                           os.path.join(args.out, "last.pt"))

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "n_planes": n_planes, "d_embed": args.d_embed, "step": step,
                "combined": args.combined, "test_frac": args.test_frac,
                "seed": args.seed}, os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"args": vars(args), "curve": curve,
                   "minutes": round((time.time() - t0) / 60, 1)}, f, indent=2)
    print(f"done: {step:,} steps in {(time.time()-t0)/60:.1f} min -> {args.out}/last.pt")


if __name__ == "__main__":
    main()
