"""SupCon fine-tune for multi-game-context models.

`finetune_mt.py` cannot do this job: it feeds the single-game `MultiTaskDataset`
and never passes `game_slot`/`ply_pos`, so a context model fine-tuned with it
would silently lose the per-game position encoding it was pre-trained with.

Two things differ from the single-game fine-tune:

- **Positives come for free.** A sample is "a random subset of player P's games",
  so drawing the same player index twice yields two *different* game subsets of
  the same player -- a genuine positive pair, not the same row twice.
- **The split is by player and is written into the checkpoint.** Pre-training
  and eval previously each re-derived the held-out set from a shared seed, which
  only stays correct while every filtering step upstream matches exactly. Here
  the held-out player ids are persisted, so the eval cannot drift onto players
  the contrastive head has already memorised.

    python finetune_ctx.py --ckpt ckpt/ctx3_pre/last.pt --shard data/mt/2026-01 \
        --out ckpt/ctx3_ft --max-hours 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from multigame_data import MultiGameDataset, collate_multigame
from model import MultiTaskModel, Config, N_ELO_BINS
from contrastive import ALL_LOSSES, make_loss, needs_proxies, default_pk
from timefeat import N_TIME_FEATS, N_TIME_BINS
from train_multigame import ply_positions, to_dev


class PKPlayers(Sampler):
    """Each batch is P players x K draws, every draw a different game subset."""

    def __init__(self, pool, p=24, k=4, batches=10**9, seed=0):
        self.pool = np.asarray(pool)
        self.p, self.k, self.batches = p, k, batches
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.batches):
            pick = self.rng.choice(len(self.pool), size=self.p, replace=False)
            yield [int(self.pool[i]) for i in pick for _ in range(self.k)]

    def __len__(self):
        return self.batches


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.05 * base + 0.95 * base * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--from-scratch", action="store_true",
                    help="random init instead of a pre-trained trunk -- the "
                         "control for whether pre-training earns its compute")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-embed", type=int, default=128)
    ap.add_argument("--new-embed", action="store_true",
                    help="take --d-embed from the flag rather than the "
                         "checkpoint, and re-initialise the projection head. "
                         "Opt-in on purpose: silently honouring --d-embed on a "
                         "resume would throw away a trained head whenever the "
                         "flag was left at its default.")
    ap.add_argument("--max-games", type=int, default=0,
                    help="from-scratch only; 1 = single-game")
    ap.add_argument("--max-len-per-game", type=int, default=160)
    ap.add_argument("--eval-every", type=int, default=5000,
                    help="steps between held-out SupCon evaluations")
    ap.add_argument("--eval-batches", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4,
                    help="stop after this many evals with no val improvement")
    ap.add_argument("--collapse-step", type=int, default=2000,
                    help="from random init the loss legitimately sits near the "
                         "ceiling for a while; check later than a fine-tune would")
    ap.add_argument("--shard", nargs="+", required=True,
                    help="one path, or several. The deployed gallery spans "
                         "six months, so single-month training never shows "
                         "the model that Jan-you and Jun-you are one person.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=100_000_000)
    ap.add_argument("--p", type=int, default=24)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--min-games", type=int, default=0,
                    help="0 = use the checkpoint's game count")
    ap.add_argument("--loss", default="ms", choices=ALL_LOSSES,
                    help="ms by default, not supcon: measured +52%% on an "
                         "identical trunk and budget, replicated across two seeds")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast; carries most of the fine-tune speedup "
                         "but is the one change that is not numerically exact")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same-colour", action="store_true",
                    help="every training bundle is ONE colour, matching how a "
                         "colour-split gallery is queried. Uniform sampling makes "
                         "an all-one-colour bundle a ~6%% case, so the mixed-trained "
                         "model meets an unfamiliar input shape at eval and lost "
                         "2.3 pts of top-10 despite twice the query games.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.from_scratch:
        n_slots = max(1, args.max_games)
        mlpg = args.max_len_per_game
        cfg = Config(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                     d_ff=args.d_model * 4, max_len=mlpg + 8, d_embed=args.d_embed)
        ck = None
        print(f"FROM SCRATCH: {n_slots} game slots, {mlpg} plies/game", flush=True)
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt is required unless --from-scratch")
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = Config(**ck["cfg"])
        n_slots = ck.get("n_game_slots", 1)
        mlpg = ck.get("max_len_per_game", cfg.max_len - 8)
        print(f"ckpt step {ck['step']}, {n_slots} game slots, {mlpg} plies/game", flush=True)

    # Enough games to build a query AND a disjoint gallery at eval time.
    need = args.min_games or (n_slots + 2)
    ds = MultiGameDataset(args.shard, max_games=n_slots, max_len_per_game=mlpg,
                          plies_per_game=4, n_cand=4, min_games=need, seed=args.seed,
                          same_colour=args.same_colour)
    print(f"{len(ds):,} players with >= {need} clocked games", flush=True)

    rng = np.random.default_rng(args.seed)
    n_test = max(1, int(len(ds) * args.test_frac))
    test_idx = rng.choice(len(ds), n_test, replace=False)
    is_test = np.zeros(len(ds), bool)
    is_test[test_idx] = True
    train_pool = np.flatnonzero(~is_test)
    test_pids = ds.gpid[is_test]
    print(f"train {len(train_pool):,} players | held out {int(is_test.sum()):,}", flush=True)

    n_planes = ds.n_planes if ck is None else ck["n_planes"]
    n_extra = N_TIME_FEATS if ck is None else ck["n_extra"]
    d_embed = args.d_embed if (ck is None or args.new_embed) else ck["d_embed"]
    model = MultiTaskModel(cfg, n_planes=n_planes, n_extra=n_extra,
                           d_embed=d_embed, n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS, n_game_slots=n_slots).to(device)
    if ck is not None:
        sd = ck["model"]
        # Widen the clock-feature inputs if the trunk was trained with fewer of
        # them. encode() does cat([planes, extra]) and embed_head takes
        # cat([pooled, elo_p, time_summary]), so in BOTH layers the extra
        # features are the LAST columns -- appending zeros there leaves the model
        # bit-identical to the checkpoint and lets it learn the new input from
        # zero, the same way elo_cond is introduced.
        old_ne = int(ck.get("n_extra", n_extra))
        if old_ne < n_extra:
            grow = n_extra - old_ne
            for key in ("in_proj.weight", "embed_head.0.weight"):
                w = sd[key]
                pad = torch.zeros(w.shape[0], grow, dtype=w.dtype)
                sd = dict(sd)
                sd[key] = torch.cat([w, pad], dim=1)
            print(f"clock features widened {old_ne} -> {n_extra}; new columns "
                  f"zero-init so the trunk starts unchanged", flush=True)
        elif old_ne > n_extra:
            raise SystemExit(f"checkpoint has {old_ne} clock features, this build "
                             f"expects {n_extra}; refusing to silently drop one")
        if args.new_embed and ck["d_embed"] != d_embed:
            # Only the head's OUTPUT width changes, so drop exactly that layer
            # and load everything else strictly -- a blanket strict=False would
            # hide a genuinely mismatched trunk.
            drop = [k for k in sd if k.startswith("embed_head.2.")]
            sd = {k: v for k, v in sd.items() if k not in drop}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            assert not unexpected, unexpected
            assert set(missing) == set(drop), (missing, drop)
            print(f"embed head re-initialised: {ck['d_embed']} -> {d_embed} "
                  f"(dropped {', '.join(drop)})", flush=True)
        else:
            model.load_state_dict(sd)
    print(f"model {sum(q.numel() for q in model.parameters())/1e6:.2f}M params",
          flush=True)
    model.train()

    pk = PKPlayers(train_pool, p=args.p, k=args.k, batches=args.steps, seed=args.seed)
    dl = DataLoader(ds, batch_sampler=pk, num_workers=args.workers,
                    collate_fn=collate_multigame, pin_memory=device == "cuda")
    # Held-out players, fixed seed: the val batches are the SAME players in the
    # same order at every evaluation, so the curve reflects the model changing
    # rather than the sample changing.
    val_pool = np.flatnonzero(is_test)
    val_pk = PKPlayers(val_pool, p=args.p, k=args.k,
                       batches=args.eval_batches, seed=12345)
    val_dl = DataLoader(ds, batch_sampler=val_pk, num_workers=max(4, args.workers // 4),
                        collate_fn=collate_multigame, pin_memory=device == "cuda")

    @torch.no_grad()
    def validate():
        model.eval()
        tot, n = 0.0, 0
        for vb in val_dl:
            vb = to_dev(vb, device)
            vpp = ply_positions(vb["game_slot"], vb["pad_mask"])
            ve, _ = model.embed(vb["planes"], vb["extra"], vb["pad_mask"],
                                vb["my_turn"], vb["game_slot"], vpp)
            vl, _ = loss_fn(ve.float(), vb["player_id"])
            tot += float(vl); n += 1
        model.train()
        return tot / max(n, 1)
    loss_fn = make_loss(args.loss).to(device)
    if needs_proxies(args.loss):
        raise SystemExit("proxy losses need a class bank keyed on train players; "
                         "they also collapsed in every configuration tested here")
    # `core` is the uncompiled module; a compiled handle's state_dict() carries
    # an "_orig_mod." prefix that no loader in this repo matches.
    core = model
    if args.compile:
        model.embed = torch.compile(model.embed, dynamic=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))

    curve, step, t0 = [], 0, time.time()
    budget = args.max_hours * 3600
    ema = None
    val_hist, best_val, bad_evals = [], float("inf"), 0

    def save(path, extra=None):
        d = {"model": core.state_dict(), "cfg": cfg.__dict__,
             "n_planes": n_planes, "n_extra": n_extra, "d_embed": d_embed,
             "n_time_bins": N_TIME_BINS, "n_elo_bins": N_ELO_BINS,
             "n_game_slots": n_slots, "max_len_per_game": mlpg,
             "step": step, "test_pids": test_pids, "loss_ema": ema,
             "loss": args.loss}
        if extra:
            d.update(extra)
        torch.save(d, os.path.join(args.out, path))
    for b in dl:
        if step >= args.steps or (time.time() - t0) >= budget:
            break
        frac = min(1.0, (time.time() - t0) / budget)
        for g in opt.param_groups:
            g["lr"] = lr_at(max(int(frac * 20_000), min(step, args.warmup)),
                            20_000, args.lr, args.warmup)

        b = to_dev(b, device)
        pp = ply_positions(b["game_slot"], b["pad_mask"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            emb, _ = model.embed(b["planes"], b["extra"], b["pad_mask"], b["my_turn"],
                                 b["game_slot"], pp)
            loss, st = loss_fn(emb.float(), b["player_id"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        lv = float(loss.detach())
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv

        if step % 100 == 0:
            dt = time.time() - t0
            print(f"step {step:>6} | {args.loss} {lv:6.3f} (ema {ema:6.3f}) | "
                  f"{step/dt:5.2f} it/s | {dt/60:6.1f} min", flush=True)
            curve.append({"step": step, "loss": lv, "ema": ema})

        # Collapse is the failure mode that killed the triplet run. Any
        # in-batch softmax contrastive loss (MS included) tops out at ln(B-1),
        # so an EMA parked there means no signal at all.
        if step == args.collapse_step and ema > 0.97 * math.log(args.p * args.k - 1):
            raise SystemExit(f"collapsed: ema {ema:.3f} ~ ln(B-1) "
                             f"{math.log(args.p*args.k-1):.3f}")

        if step % args.eval_every == 0:
            v = validate()
            val_hist.append({"step": step, "val_loss": v})
            better = v < best_val - 1e-4
            print(f"  >> val @ {step}: {args.loss} {v:.4f} "
                  f"({'best' if better else f'no gain ({bad_evals+1}/{args.patience})'})",
                  flush=True)
            if better:
                best_val, bad_evals = v, 0
                save("best.pt", {"val_loss": v})
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    print(f"early stop: {args.patience} evals without improvement "
                          f"(best val {best_val:.4f})", flush=True)
                    break

        if step % 1000 == 0 or (time.time() - t0) >= budget:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "n_planes": n_planes, "n_extra": n_extra,
                        "d_embed": d_embed, "n_time_bins": N_TIME_BINS,
                        "n_elo_bins": N_ELO_BINS, "n_game_slots": n_slots,
                        "max_len_per_game": mlpg, "step": step,
                        "test_pids": test_pids, "loss_ema": ema,
                        "loss": args.loss, "max_games": n_slots},
                       os.path.join(args.out, "last.pt"))

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "n_planes": n_planes, "n_extra": n_extra,
                "d_embed": d_embed, "n_time_bins": N_TIME_BINS,
                "n_elo_bins": N_ELO_BINS, "n_game_slots": n_slots,
                "max_len_per_game": mlpg, "step": step,
                "test_pids": test_pids, "loss_ema": ema,
                        "loss": args.loss, "max_games": n_slots},
               os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"args": vars(args), "curve": curve, "final_ema": ema,
                   "val_history": val_hist, "best_val": best_val,
                   "minutes": round((time.time() - t0) / 60, 1)}, f, indent=2)
    print("CTX_FINETUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
