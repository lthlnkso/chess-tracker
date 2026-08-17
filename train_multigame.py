"""Pre-train with multi-game context.

The change that matters is to the *objective*, not the plumbing. With one game
per sample the task is "predict the next move in this game". With several of the
same player's games concatenated it becomes "predict the next move in this game,
given this player's other games" -- so the trunk is rewarded for extracting
whatever is stable about the player. That is the quantity identification needs,
and previously nothing in pre-training asked for it.

    python train_multigame.py --shard data/mt/2026-01 --out ckpt/ctx3 \
        --max-hours 3 --max-games 3
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

from multigame_data import MultiGameDataset, collate_multigame
from model import elo_to_bin, MultiTaskModel, Config, multitask_loss, N_ELO_BINS
from timefeat import N_TIME_FEATS, N_TIME_BINS, TIME_CENTRES
from balance import EloBalancedSampler


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.05 * base + 0.95 * base * 0.5 * (1 + math.cos(math.pi * p))


def ply_positions(game_slot, pad_mask):
    """Per-game ply index, so 'ply 3 of game 2' does not collide with 'ply 83'.

    Computed from run lengths of the slot ids rather than assumed, because games
    have different lengths and padding sits on the right.

    Vectorised. The obvious per-row loop over torch.unique cost ~2,500 eager
    dispatches and ~740 device syncs per step -- `s[valid]` and `pos[r][m] = ...`
    are bool-mask index/index_put_, which lower to nonzero() and stall the CPU on
    a device round trip. Profiling put it at roughly a third of a training step,
    and because it is HOST work its share grows on a faster GPU rather than
    shrinking.

    A position's index within its game is just its distance from the start of its
    run, and a run starts wherever the slot id changes. cummax over the run-start
    markers gives that start for every position at once, with no sync at all.

    Preconditions, both guaranteed by collate_multigame and probe_pack: each slot
    id occupies ONE contiguous run, and padding is right-side only.
    """
    B, T = game_slot.shape
    idx = torch.arange(T, device=game_slot.device).expand(B, T)
    valid = ~pad_mask

    cont = torch.zeros_like(valid)
    cont[:, 1:] = (game_slot[:, 1:] == game_slot[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    # A run start marks itself with its own index; continuations mark 0, so the
    # running maximum is always the most recent run start.
    run_start = torch.cummax(torch.where(cont, torch.zeros_like(idx), idx), dim=1).values
    return (idx - run_start) * valid


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, device, max_batches, w, lossk, elo_cond=False):
    model.eval()
    acc, n = {}, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        b = to_dev(b, device)
        pp = ply_positions(b["game_slot"], b["pad_mask"])
        eb = elo_to_bin(b["elo"]) if elo_cond else None
        ml, tl, el, _, _ = model(b["planes"], b["extra"], b["cands"], b["ply_idx"],
                                 b["pad_mask"], b["my_turn"], b["game_slot"], pp,
                                 elo_bin=eb)
        _, st = multitask_loss(ml, tl, el, b, *w, **lossk)
        for k, v in st.items():
            if v == v:
                acc[k] = acc.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs="+", required=True,
                    help="one path, or several to train across months. "
                         "Multiple shards are joined on the lowercased "
                         "USERNAME -- pid is per shard.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=100_000_000)
    ap.add_argument("--max-games", type=int, default=3)
    ap.add_argument("--max-len-per-game", type=int, default=160,
                    help="160 truncates 0.09%% of games and 0.02%% of plies; the "
                         "old 80 discarded 25%% of every player's evidence")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--min-games", type=int, default=0,
                    help="eligibility threshold, default 0 = use --max-games. "
                         "Decoupling these matters: min_games==max_games makes "
                         "the SLOT COUNT an eligibility filter, so going 5->10 "
                         "slots silently halves the player pool (18,856 -> 9,308 "
                         "on a real shard) and keeps only the most active "
                         "accounts. The loader already samples bundles of "
                         "1..max_games and clamps to what a player has, so a "
                         "lower threshold costs nothing.")
    ap.add_argument("--plies-per-game", type=int, default=8)
    ap.add_argument("--n-cand", type=int, default=16)
    ap.add_argument("--d-embed", type=int, default=128)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-batches", type=int, default=25)
    ap.add_argument("--w-time", type=float, default=0.3)
    ap.add_argument("--w-elo", type=float, default=0.3)
    ap.add_argument("--cpl-dir", default="",
                    help="CPL corpus from cpl_label.py. Adds the graded "
                         "win-probability term; without it the run is the "
                         "ordinary cross-entropy baseline.")
    ap.add_argument("--cpl-only", action="store_true",
                    help="restrict training to games the corpus labelled. The "
                         "corpus covers ~0.2%% of a shard, so without this the "
                         "CPL term almost never fires and the arms are "
                         "indistinguishable. BOTH arms must set it.")
    ap.add_argument("--w-cpl", type=float, default=0.0,
                    help="weight on the CPL term. 0 disables it even with a "
                         "corpus attached, which is how the control arm runs.")
    ap.add_argument("--balance-elo", action="store_true")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many evals with no val move_acc "
                         "improvement. 0 = run the full clock. This is what "
                         "'pre-train until it stops improving' means operationally: "
                         "every previous run was stopped by --max-hours with the "
                         "curve still climbing.")
    ap.add_argument("--min-delta", type=float, default=0.0005)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--init", default="",
                    help="initialise from an existing checkpoint instead of "
                         "random. Training the trunk from scratch costs ~25h; "
                         "adding rating conditioning to a trained one is hours.")
    ap.add_argument("--elo-cond", action="store_true",
                    help="condition the trunk on the player's rating, making "
                         "the style settable at inference")
    ap.add_argument("--elo-drop", type=float, default=0.1,
                    help="fraction of samples trained with the rating hidden, "
                         "so the model still works when none is supplied")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    cpl = None
    if args.cpl_dir:
        from cplcorpus import CplCorpus
        cpl = CplCorpus(args.cpl_dir)
        print(f"CPL corpus {args.cpl_dir}: {len(cpl):,} labelled plies "
              f"(depth {cpl.manifest.get('depth')}, "
              f"multipv {cpl.manifest.get('multipv')}) | w_cpl={args.w_cpl}",
              flush=True)

    ds = MultiGameDataset(args.shard, max_games=args.max_games,
                          max_len_per_game=args.max_len_per_game,
                          plies_per_game=args.plies_per_game, n_cand=args.n_cand,
                          min_games=args.min_games or args.max_games,
                          seed=args.seed, cpl=cpl,
                          cpl_only=args.cpl_only)
    print(f"{len(ds):,} players with >= {args.min_games or args.max_games} "
          f"clocked games | "
          f"context up to {ds.max_len} plies ({args.max_games} x "
          f"{args.max_len_per_game})", flush=True)

    # Split by PLAYER, not by game: the whole point is cross-game context, so a
    # player appearing in both splits would leak directly.
    rng = np.random.default_rng(args.seed)
    n_val = max(1, int(len(ds) * args.val_frac))
    val_ids = set(rng.choice(len(ds), n_val, replace=False).tolist())
    tr = np.array([i for i in range(len(ds)) if i not in val_ids])
    va = np.array(sorted(val_ids))
    print(f"train {len(tr):,} players | val {len(va):,} players", flush=True)

    common = dict(batch_size=args.batch, num_workers=args.workers,
                  collate_fn=collate_multigame, pin_memory=device == "cuda")
    if args.balance_elo:
        elo = np.array([int(np.median([
            ds.meta[sh][g]["white_elo" if s == 0 else "black_elo"]
            for sh, g, s in zip(*ds.groups[i])][:8])) for i in tr])
        sampler = EloBalancedSampler(elo, num_samples=len(tr), seed=args.seed)
        print(f"Elo balancing: {sampler.describe()}", flush=True)
        train_dl = DataLoader(Subset(ds, tr.tolist()), sampler=sampler,
                              drop_last=True, **common)
    else:
        train_dl = DataLoader(Subset(ds, tr.tolist()), shuffle=True,
                              drop_last=True, **common)
    val_dl = DataLoader(Subset(ds, va.tolist()), shuffle=False, **common)

    cfg = Config(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                 d_ff=args.d_model * 4, max_len=args.max_len_per_game + 8,
                 d_embed=args.d_embed)
    model = MultiTaskModel(cfg, n_planes=ds.n_planes, n_extra=N_TIME_FEATS,
                           d_embed=args.d_embed, n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS,
                           n_game_slots=args.max_games,
                           elo_cond=args.elo_cond).to(device)
    print(f"model {sum(p.numel() for p in model.parameters())/1e6:.2f}M params "
          f"on {device}", flush=True)

    if args.init:
        ick = torch.load(args.init, map_location="cpu", weights_only=False)
        # strict=False: elo_cond.weight is new and stays at its zero init, so the
        # model starts bit-identical to the checkpoint and learns the
        # conditioning from there.
        miss = model.load_state_dict(ick["model"], strict=False)
        print(f"init from {args.init} (step {ick.get('step')}) | "
              f"missing {miss.missing_keys} unexpected {miss.unexpected_keys}",
              flush=True)

    # core stays uncompiled: a compiled handle's state_dict() keys are prefixed
    # "_orig_mod." and nothing downstream matches them.
    core = model
    if args.compile:
        model = torch.compile(model, dynamic=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    w = (1.0, args.w_time, args.w_elo)
    lossk = dict(time_centres=torch.as_tensor(TIME_CENTRES), w_cpl=args.w_cpl)

    hist, curve = [], []
    step, t0, stop = 0, time.time(), False
    budget = args.max_hours * 3600
    model.train()
    best_acc, bad_evals = -1.0, 0
    while step < args.steps and not stop:
        for b in train_dl:
            if step >= args.steps:
                break
            if budget and (time.time() - t0) >= budget:
                print(f"reached --max-hours {args.max_hours} at step {step}", flush=True)
                stop = True
                break
            frac = min(1.0, (time.time() - t0) / budget) if budget else step / args.steps
            lr_now = lr_at(max(int(frac * 100_000), min(step, args.warmup)), 100_000,
                           args.lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr_now

            b = to_dev(b, device)
            pp = ply_positions(b["game_slot"], b["pad_mask"])
            eb = None
            if args.elo_cond:
                eb = elo_to_bin(b["elo"])
                # Hide the rating on a fraction of samples so the model still
                # plays sensibly when none is given. Without this it would only
                # ever have seen a rating and the unconditioned path -- which is
                # what any caller that does not set one gets -- would be
                # untrained.
                if args.elo_drop > 0:
                    drop = torch.rand(eb.shape, device=eb.device) < args.elo_drop
                    eb = torch.where(drop, torch.full_like(eb, N_ELO_BINS), eb)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                ml, tl, el, _, _ = model(b["planes"], b["extra"], b["cands"], b["ply_idx"],
                                         b["pad_mask"], b["my_turn"], b["game_slot"], pp,
                                         elo_bin=eb)
                loss, st = multitask_loss(ml, tl, el, b, *w, **lossk)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % 100 == 0:
                dt = time.time() - t0
                print(f"step {step:>7} | total {st['total']:6.3f} | move {st['move']:6.3f} "
                      f"(acc {st['move_acc']:.3f}) | time_acc {st.get('time_acc', 0):.3f} "
                      f"| elo_mae {st.get('elo_mae', float('nan')):5.0f} | "
                      f"{step/dt:5.2f} it/s | {dt/60:6.1f} min", flush=True)
                curve.append({"step": step, "minutes": round(dt / 60, 2), **st})

            if step % args.eval_every == 0 or stop:
                ev = evaluate(model, val_dl, device, args.eval_batches, w, lossk, args.elo_cond)
                print(f"  >> val @ {step}: move_acc {ev.get('move_acc',0):.3f} | "
                      f"time_acc {ev.get('time_acc',0):.3f} | "
                      f"elo_mae {ev.get('elo_mae',0):.0f}", flush=True)
                hist.append({"step": step, **ev})
                acc = ev.get("move_acc", 0.0)
                if acc > best_acc + args.min_delta:
                    best_acc, bad_evals = acc, 0
                else:
                    bad_evals += 1
                    if args.patience and bad_evals >= args.patience:
                        print(f"  no val gain for {bad_evals} evals "
                              f"(best {best_acc:.4f}) -- pre-training has stopped "
                              f"improving, stopping at step {step}", flush=True)
                        stop = True
                        break
                torch.save({"model": core.state_dict(), "cfg": cfg.__dict__,
                            "n_planes": ds.n_planes, "n_extra": N_TIME_FEATS,
                            "d_embed": args.d_embed, "n_time_bins": N_TIME_BINS,
                            "n_elo_bins": N_ELO_BINS, "n_game_slots": args.max_games,
                "elo_cond": args.elo_cond,
                            "max_len_per_game": args.max_len_per_game,
                            "step": step, "val": ev},
                           os.path.join(args.out, "last.pt"))

    # Save unconditionally: the budget check breaks out of the loop before the
    # periodic save, so without this a run whose step count never lands on a
    # multiple of --eval-every produces no checkpoint at all, and an ordinary
    # run silently discards everything since the last one.
    ev = evaluate(model, val_dl, device, args.eval_batches, w, lossk, args.elo_cond)
    hist.append({"step": step, **ev})
    torch.save({"model": core.state_dict(), "cfg": cfg.__dict__,
                "n_planes": ds.n_planes, "n_extra": N_TIME_FEATS,
                "d_embed": args.d_embed, "n_time_bins": N_TIME_BINS,
                "n_elo_bins": N_ELO_BINS, "n_game_slots": args.max_games,
                "elo_cond": args.elo_cond,
                "max_len_per_game": args.max_len_per_game,
                "step": step, "val": ev},
               os.path.join(args.out, "last.pt"))
    print(f"final @ {step}: move_acc {ev.get('move_acc',0):.3f} | "
          f"elo_mae {ev.get('elo_mae',0):.0f}", flush=True)

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"args": vars(args), "curve": curve, "history": hist,
                   "minutes": round((time.time() - t0) / 60, 1)}, f, indent=2)
    print("MULTIGAME_PRETRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
