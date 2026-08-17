"""Multi-task pre-training: move + time + Elo from one trunk.

    python train_multitask.py --shard data/2026-06-sample --out ckpt/mt_smoke \
        --steps 300 --batch 32 --d-model 128 --layers 4 --workers 4

The point of the extra heads is not the auxiliary metrics themselves. Time and
rating are cheap supervision that the board alone does not provide, and both feed
the embedding at deployment -- the Elo estimate also prunes the retrieval gallery.
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

from successor_data import (MultiTaskDataset, MultiShardMultiTaskDataset,
                            collate_multitask)
from model import MultiTaskModel, Config, multitask_loss, N_ELO_BINS
from timefeat import N_TIME_FEATS, N_TIME_BINS, TIME_CENTRES
from balance import EloBalancedSampler


def _row_elo(ds, combined):
    """Per-row Elo of the attributed player, for either dataset flavour."""
    if combined:
        return np.array([int(ds.metas[int(r["shard"])][int(r["game"])]
                             ["white_elo" if int(r["seat"]) == 0 else "black_elo"])
                         for r in ds.idx])
    return np.array([int(ds.meta[g]["white_elo" if s == 0 else "black_elo"])
                     for g, s in ds.index])


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.05 * base + 0.95 * base * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, device, max_batches, w, LOSSK):
    model.eval()
    acc = {}
    n = 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        logits, tpred, epred, _, _ = model(
            b["planes"], b["extra"], b["cands"], b["ply_idx"],
            b["pad_mask"], b["my_turn"])
        _, st = multitask_loss(logits, tpred, epred, b, *w, **LOSSK)
        for k, v in st.items():
            if v == v:                                   # skip NaN
                acc[k] = acc.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shard", help="single month shard")
    src.add_argument("--combined", help="combine.py index across months")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--plies-per-game", type=int, default=12)
    ap.add_argument("--n-cand", type=int, default=16)
    # --- candidate-set curriculum ---
    # Scoring the true successor against 2 candidates is nearly free supervision;
    # against 64 it is a much harder discrimination. The question is whether
    # starting easy and hardening buys anything over training at the hard setting
    # throughout, at equal wall-clock.
    ap.add_argument("--cand-curriculum", action="store_true",
                    help="start at --cand-start and double on promotion")
    ap.add_argument("--cand-start", type=int, default=2)
    ap.add_argument("--cand-max", type=int, default=64)
    ap.add_argument("--cand-promote", type=float, default=0.45,
                    help="promote when the chance-normalised skill score "
                         "(acc - 1/C)/(1 - 1/C) reaches this")
    ap.add_argument("--cand-min-steps", type=int, default=2000)
    ap.add_argument("--cand-patience", type=int, default=6000,
                    help="promote anyway if the stage stops improving -- a "
                         "difficulty that has stopped teaching is the other "
                         "reason to move on")
    ap.add_argument("--eval-n-cand", type=int, default=16,
                    help="candidates used for VALIDATION, held fixed across "
                         "arms; move_acc is meaningless to compare otherwise "
                         "because chance is 1/C")
    ap.add_argument("--d-embed", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=10)
    ap.add_argument("--w-move", type=float, default=1.0)
    ap.add_argument("--w-time", type=float, default=0.3)
    ap.add_argument("--w-elo", type=float, default=0.3)
    ap.add_argument("--no-rights", action="store_true")
    ap.add_argument("--balance-elo", action="store_true",
                    help="sample training rows so Elo bands are equally represented")
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast. +49%% end-to-end on a 3090, but unlike the "
                         "loader work it CHANGES NUMERICS -- A/B a short run before "
                         "trusting it for a real one")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (+19%% on top of --amp)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _kw = dict(max_len=args.max_len, plies_per_game=args.plies_per_game,
               n_cand=args.n_cand, with_rights=not args.no_rights)
    ds = (MultiShardMultiTaskDataset(args.combined, **_kw) if args.combined
          else MultiTaskDataset(args.shard, **_kw))
    # A second instance, pinned to --eval-n-cand. It has to be a separate object:
    # the curriculum mutates ds.n_cand as it runs, and validation that moved with
    # it would report a different task at every stage.
    ds_val = (MultiShardMultiTaskDataset(args.combined, **{**_kw, "n_cand": args.eval_n_cand})
              if args.combined
              else MultiTaskDataset(args.shard, **{**_kw, "n_cand": args.eval_n_cand}))
    print(f"{len(ds):,} game-sides from {ds.n_clocked_games:,} clocked games "
          f"| {ds.n_planes} planes + {N_TIME_FEATS} time feats", flush=True)

    if args.cand_curriculum:
        ds.n_cand = args.cand_start
        print(f"curriculum: C {args.cand_start} -> {args.cand_max}, promote at "
              f"skill {args.cand_promote}, min {args.cand_min_steps} steps/stage, "
              f"patience {args.cand_patience} | val fixed at C={args.eval_n_cand}",
              flush=True)

    # hold out whole games, both seats together
    gkey = (np.asarray(ds.idx["shard"], np.int64) << 32 | np.asarray(ds.idx["game"], np.int64)
            if args.combined else ds.index[:, 0])
    games = np.unique(gkey)
    rng = np.random.default_rng(args.seed)
    val_games = set(rng.choice(games, max(1, int(len(games) * args.val_frac)),
                               replace=False).tolist())
    is_val = np.fromiter((g in val_games for g in gkey), bool, len(gkey))
    train_ds = Subset(ds, np.flatnonzero(~is_val))
    val_ds = Subset(ds_val, np.flatnonzero(is_val))
    print(f"train {len(train_ds):,} | val {len(val_ds):,}", flush=True)

    common = dict(batch_size=args.batch, num_workers=args.workers,
                  collate_fn=collate_multitask, pin_memory=device == "cuda")
    if args.balance_elo:
        # Weight TRAIN rows only; val stays on the natural distribution so the
        # reported Elo MAE describes the population that actually exists.
        tr_idx = np.flatnonzero(~is_val)
        tr_elo = _row_elo(ds, args.combined)[tr_idx]
        # NOT WeightedRandomSampler: it routes through torch.multinomial, which
        # hard-fails above 2^24 categories. One month is 48M game-sides.
        sampler = EloBalancedSampler(tr_elo, num_samples=len(tr_idx), seed=args.seed)
        print(f"Elo balancing: {sampler.describe()}", flush=True)
        train_dl = DataLoader(train_ds, sampler=sampler, drop_last=True, **common)
    else:
        train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(val_ds, shuffle=False, **common)

    cfg = Config(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                 d_ff=args.d_model * 4, max_len=args.max_len, d_embed=args.d_embed)
    model = MultiTaskModel(cfg, n_planes=ds.n_planes, n_extra=N_TIME_FEATS,
                           d_embed=args.d_embed, n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS).to(device)
    # `core` stays the uncompiled module. torch.compile returns a wrapper whose
    # state_dict() keys are prefixed with "_orig_mod.", which every loader in
    # this repo would silently fail to match -- so saving always goes through
    # `core`, never through the compiled handle.
    core = model
    if args.compile:
        # dynamic=True because T varies with game length; static shapes would
        # recompile on nearly every batch.
        model = torch.compile(model, dynamic=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model {n_par/1e6:.2f}M params on {device}", flush=True)

    _e = _row_elo(ds, args.combined).astype(np.float64)
    _e = _e[_e > 0]
    print(f"Elo: mean {_e.mean():.0f} sd {_e.std():.0f} | "
          f"BASELINE MAE (predict mean) {np.abs(_e-_e.mean()).mean():.0f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    w = (args.w_move, args.w_time, args.w_elo)
    lossk = dict(time_centres=torch.as_tensor(TIME_CENTRES))

    hist, curve = [], []
    step = 0
    t0 = time.time()
    model.train()
    stop = False
    # Curriculum state. `ema` smooths the noisy per-batch accuracy; `best_at` is
    # the last step at which the stage was still learning anything.
    stages, stage_start, ema, best_ema, best_at = [], 0, None, -1.0, 0
    while step < args.steps and not stop:
        for b in train_dl:
            if step >= args.steps:
                break
            if args.max_hours and (time.time() - t0) >= args.max_hours * 3600:
                print(f"reached --max-hours {args.max_hours} at step {step}", flush=True)
                stop = True
                break
            if args.max_hours:
                frac = min(1.0, (time.time() - t0) / (args.max_hours * 3600))
                virt = 100_000
                lr_now = lr_at(max(int(frac * virt), min(step, args.warmup)),
                               virt, args.lr, args.warmup)
            else:
                lr_now = lr_at(step, args.steps, args.lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr_now
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                logits, tpred, epred, _, _ = model(
                    b["planes"], b["extra"], b["cands"], b["ply_idx"],
                    b["pad_mask"], b["my_turn"])
                loss, st = multitask_loss(logits, tpred, epred, b, *w, **lossk)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            C = ds.n_cand
            ema = st["move_acc"] if ema is None else 0.98 * ema + 0.02 * st["move_acc"]
            skill = (ema - 1.0 / C) / (1.0 - 1.0 / C)
            if ema > best_ema + 1e-4:
                best_ema, best_at = ema, step

            if step % 25 == 0:
                dt = time.time() - t0
                print(f"step {step:>5} | C {C:>2} skill {skill:+.3f} | "
                      f"total {st['total']:6.3f} | move {st['move']:6.3f} "
                      f"(acc {st['move_acc']:.3f}) | time {st['time']:6.4f} "
                      f"(acc {st.get('time_acc', float('nan')):.3f}, "
                      f"mae {st.get('time_mae_s', float('nan')):.2f}s) | "
                      f"elo_mae {st.get('elo_mae', float('nan')):5.0f} | "
                      f"{step/dt:4.1f} it/s", flush=True)
                curve.append({"step": step, "n_cand": C, "skill": skill, **st})

            if step % args.eval_every == 0 or step == args.steps or stop:
                ev = evaluate(model, val_dl, device, args.eval_batches, w, lossk)
                print(f"  >> val @ {step}: move_acc {ev.get('move_acc', 0):.3f} | "
                      f"time_acc {ev.get('time_acc', float('nan')):.3f} "
                      f"(mae {ev.get('time_mae_s', float('nan')):.2f}s) | "
                      f"elo_mae {ev.get('elo_mae', float('nan')):.0f} Elo", flush=True)
                hist.append({"step": step, **ev})
                torch.save({"model": core.state_dict(), "cfg": cfg.__dict__,
                            "n_planes": ds.n_planes, "n_extra": N_TIME_FEATS,
                            "d_embed": args.d_embed, "step": step, "val": ev,
                            "n_time_bins": N_TIME_BINS, "n_elo_bins": N_ELO_BINS},
                           os.path.join(args.out, "last.pt"))

            if (args.cand_curriculum and C < args.cand_max
                    and step - stage_start >= args.cand_min_steps):
                why = ("skill" if skill >= args.cand_promote else
                       "plateau" if step - best_at >= args.cand_patience else None)
                if why:
                    stages.append({"n_cand": C, "from_step": stage_start,
                                   "to_step": step, "skill": skill, "why": why,
                                   "minutes": round((time.time() - t0) / 60, 2)})
                    ds.n_cand = min(C * 2, args.cand_max)
                    print(f"  ** promote C {C} -> {ds.n_cand} at step {step} "
                          f"({why}, skill {skill:+.3f})", flush=True)
                    stage_start, ema, best_ema, best_at = step, None, -1.0, step
                    # Break so the outer loop builds a fresh iterator: the worker
                    # processes fork the dataset, so a live n_cand change only
                    # reaches them when they are respawned.
                    break

    # A --max-hours run almost never stops on a multiple of --eval-every, and the
    # in-loop save is the only one there was. Without this the last stretch of a
    # time-boxed run -- which is exactly the part a longer run is buying -- is
    # thrown away.
    ev = evaluate(model, val_dl, device, args.eval_batches, w, lossk)
    print(f"  >> FINAL val @ {step}: move_acc {ev.get('move_acc', 0):.3f} | "
          f"time_acc {ev.get('time_acc', float('nan')):.3f} | "
          f"elo_mae {ev.get('elo_mae', float('nan')):.0f} Elo", flush=True)
    hist.append({"step": step, **ev})
    torch.save({"model": core.state_dict(), "cfg": cfg.__dict__,
                "n_planes": ds.n_planes, "n_extra": N_TIME_FEATS,
                "d_embed": args.d_embed, "step": step, "val": ev,
                "n_time_bins": N_TIME_BINS, "n_elo_bins": N_ELO_BINS},
               os.path.join(args.out, "last.pt"))

    if args.cand_curriculum:
        stages.append({"n_cand": ds.n_cand, "from_step": stage_start,
                       "to_step": step, "why": "end",
                       "minutes": round((time.time() - t0) / 60, 2)})
        print("curriculum stages: " + " | ".join(
            f"C{s['n_cand']}:{s['to_step']-s['from_step']}steps({s['why']})"
            for s in stages), flush=True)

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"args": vars(args), "params": n_par, "curve": curve,
                   "history": hist, "stages": stages, "steps": step,
                   "final_n_cand": ds.n_cand,
                   "minutes": round((time.time() - t0) / 60, 2)}, f,
                  indent=2)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
