"""Does clock data actually buy identification accuracy?

Three arms, identical in every other respect -- same data, same split, same
seed, same model size, same step counts:

    A  board                     the current system
    B  board + time inputs       timing visible to the trunk
    C  board + time + aux heads  plus time-bucket and Elo supervision

A vs B isolates the *input* feature; B vs C isolates the *auxiliary
supervision*. Reporting only A vs C would confound the two.

Each arm runs pre-train -> SupCon fine-tune -> identification eval, and the
only number that matters is the last one: recall against held-out player
centroids.

    python ablation_time_elo.py --shard data/2026-06-big --minutes 55
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Sampler

from successor_data import MultiTaskDataset, collate_multitask
from model import (MultiTaskModel, Config, multitask_loss, supcon_loss,
                   N_ELO_BINS, elo_expectation)
from timefeat import N_TIME_FEATS, N_TIME_BINS, TIME_CENTRES

ARMS = [
    ("A_board",        dict(use_time=False, aux=False)),
    ("B_time_input",   dict(use_time=True,  aux=False)),
    ("C_time_plus_aux", dict(use_time=True, aux=True)),
]


class PK(Sampler):
    """P players x K games per batch -- SupCon needs positives in-batch."""

    def __init__(self, labels, p=16, k=4, batches=10**9, seed=0):
        self.p, self.k, self.batches = p, k, batches
        self.rng = np.random.default_rng(seed)
        order = np.argsort(labels, kind="stable")
        sl = labels[order]
        b = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1], True])
        self.groups = [order[b[i]:b[i + 1]] for i in range(len(b) - 1)]
        self.groups = [g for g in self.groups if len(g) >= k]

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            out = []
            for gi in self.rng.choice(len(self.groups), min(self.p, len(self.groups)),
                                      replace=False):
                g = self.groups[gi]
                out.extend(self.rng.choice(g, self.k, replace=False).tolist())
            yield out


def lr_at(step, total, base, warmup=50):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.05 * base + 0.95 * base * 0.5 * (1 + math.cos(math.pi * p))


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def zero_time(b):
    """Blank the clock channels -- arm A must not see them."""
    b = dict(b)
    b["extra"] = torch.zeros_like(b["extra"])
    return b


@torch.no_grad()
def identify(model, ds, rows, device, use_time, cfg, batch=64, workers=4,
             centroid_frac=0.6, seed=0):
    """Embed held-out game-sides, build centroids, score recall@k."""
    dl = DataLoader(Subset(ds, rows.tolist()), batch_size=batch, shuffle=False,
                    num_workers=workers, collate_fn=collate_multitask)
    embs, labs = [], []
    model.eval()
    for b in dl:
        b = to_dev(b if use_time else zero_time(b), device)
        e, _ = model.embed(b["planes"], b["extra"], b["pad_mask"], b["my_turn"])
        embs.append(e.float().cpu())
        labs.append(b["player_id"].cpu())
    model.train()
    E = torch.cat(embs)
    L = torch.cat(labs)

    rng = np.random.default_rng(seed)
    ln = L.numpy()
    order = np.argsort(ln, kind="stable")
    sl = ln[order]
    bnd = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1], True])

    cent, clab, qidx = [], [], []
    for i in range(len(bnd) - 1):
        g = order[bnd[i]:bnd[i + 1]]
        if len(g) < 3:
            continue
        perm = rng.permutation(len(g))
        nc = max(1, int(round(centroid_frac * len(g))))
        nc = min(nc, len(g) - 1)
        v = E[g[perm[:nc]]].mean(0)
        cent.append(v / v.norm().clamp(min=1e-8))
        clab.append(sl[bnd[i]])
        qidx.append(g[perm[nc:]])
    if not cent:
        return {}
    C = torch.stack(cent)
    CL = torch.tensor(clab)
    qi = np.concatenate(qidx)
    Q, QL = E[qi], L[qi]

    sim = Q @ C.T
    maxk = min(10, len(CL))
    top = sim.topk(maxk, dim=1).indices
    match = CL[top] == QL[:, None]
    true_col = (CL[None, :] == QL[:, None]).float().argmax(1)
    rank = (sim > sim.gather(1, true_col[:, None])).sum(1).float()
    return {
        "gallery": int(len(CL)), "queries": int(len(QL)),
        "recall@1": float(match[:, :1].any(1).float().mean()),
        "recall@10": float(match[:, :maxk].any(1).float().mean()),
        "median_rank": float(rank.median()),
        "chance@1": 1.0 / len(CL),
    }


def run_arm(name, opts, ds, splits, args, device):
    torch.manual_seed(args.seed)
    use_time, aux = opts["use_time"], opts["aux"]
    tr_rows, te_rows, pid_tr = splits

    cfg = Config(d_model=args.d_model, n_layers=args.layers, n_heads=4,
                 d_ff=args.d_model * 4, max_len=args.max_len, d_embed=128)
    model = MultiTaskModel(cfg, n_planes=ds.n_planes, n_extra=N_TIME_FEATS,
                           d_embed=128, n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS).to(device)

    t0 = time.time()

    # ---- stage 1: pre-train ----
    dl = DataLoader(Subset(ds, tr_rows.tolist()), batch_size=args.batch, shuffle=True,
                    drop_last=True, num_workers=args.workers,
                    collate_fn=collate_multitask)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    lossk = dict(time_centres=torch.as_tensor(TIME_CENTRES))
    step = 0
    model.train()
    while step < args.pre_steps:
        for b in dl:
            if step >= args.pre_steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.pre_steps, args.lr)
            b = to_dev(b if use_time else zero_time(b), device)
            ml, tl, el, _, _ = model(b["planes"], b["extra"], b["cands"],
                                     b["ply_idx"], b["pad_mask"], b["my_turn"])
            loss, st = multitask_loss(ml, tl, el, b, 1.0,
                                      args.w_time if aux else 0.0,
                                      args.w_elo if aux else 0.0, **lossk)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                print(f"    [{name}] pre {step}/{args.pre_steps} "
                      f"move_acc {st['move_acc']:.3f} "
                      f"elo_mae {st.get('elo_mae', float('nan')):.0f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    pre_stats = dict(move_acc=st["move_acc"], elo_mae=st.get("elo_mae", float("nan")),
                     time_acc=st.get("time_acc", float("nan")))

    # ---- stage 2: SupCon fine-tune ----
    sub = Subset(ds, tr_rows.tolist())
    pk = PK(pid_tr, p=args.p, k=args.k, batches=args.ft_steps, seed=args.seed)
    dl2 = DataLoader(sub, batch_sampler=pk, num_workers=args.workers,
                     collate_fn=collate_multitask)
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=0.05,
                             betas=(0.9, 0.95))
    step = 0
    for b in dl2:
        for g in opt2.param_groups:
            g["lr"] = lr_at(step, args.ft_steps, args.ft_lr)
        b = to_dev(b if use_time else zero_time(b), device)
        emb, _ = model.embed(b["planes"], b["extra"], b["pad_mask"], b["my_turn"])
        loss, st2 = supcon_loss(emb, b["player_id"])
        st2["loss"] = float(loss)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt2.step()
        step += 1
        if step % 100 == 0:
            print(f"    [{name}] ft {step}/{args.ft_steps} loss {st2['loss']:.3f} "
                  f"gap {st2['gap']:+.3f} ({time.time()-t0:.0f}s)", flush=True)
        if step >= args.ft_steps:
            break

    res = identify(model, ds, te_rows, device, use_time, cfg,
                   workers=args.workers, seed=args.seed)
    res.update(pretrain=pre_stats, minutes=round((time.time() - t0) / 60, 2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--out", default="ckpt/ablation.json")
    ap.add_argument("--pre-steps", type=int, default=700)
    ap.add_argument("--ft-steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--p", type=int, default=12)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=120)
    ap.add_argument("--plies-per-game", type=int, default=8)
    ap.add_argument("--n-cand", type=int, default=12)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--w-time", type=float, default=0.3)
    ap.add_argument("--w-elo", type=float, default=0.3)
    ap.add_argument("--eval-players", type=int, default=1200,
                    help="cap the held-out gallery so eval does not dominate")
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MultiTaskDataset(args.shard, max_len=args.max_len,
                          plies_per_game=args.plies_per_game, n_cand=args.n_cand)

    pid = np.asarray([int(ds.meta[g]["white_pid"] if s == 0 else ds.meta[g]["black_pid"])
                      for g, s in ds.index])
    u, c = np.unique(pid, return_counts=True)
    keep = set(u[c >= args.min_games].tolist())
    rows = np.flatnonzero(np.array([p in keep for p in pid]))
    kept_pid = pid[rows]

    rng = np.random.default_rng(args.seed)
    players = np.unique(kept_pid)
    test_players = set(rng.choice(players, int(len(players) * args.test_frac),
                                  replace=False).tolist())
    is_test = np.array([p in test_players for p in kept_pid])
    tr_rows = rows[~is_test]
    pid_tr = kept_pid[~is_test]
    te_rows_all, te_pid = rows[is_test], kept_pid[is_test]
    # Cap the gallery: embedding every held-out row would cost more than the
    # training it is meant to compare. Same players for every arm.
    gal = np.unique(te_pid)
    if len(gal) > args.eval_players:
        gal = set(rng.choice(gal, args.eval_players, replace=False).tolist())
        te_rows = te_rows_all[np.array([p in gal for p in te_pid])]
    else:
        te_rows = te_rows_all
    print(f"{len(players):,} players with >={args.min_games} game-sides | "
          f"train {len(tr_rows):,} rows / test {len(te_rows):,} rows "
          f"({len(test_players):,} held-out players)", flush=True)

    out = {}
    for name, opts in ARMS:
        print(f"\n=== arm {name}  {opts} ===", flush=True)
        out[name] = run_arm(name, opts, ds, (tr_rows, te_rows, pid_tr), args, device)
        print(f"  -> {json.dumps(out[name])}", flush=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "arms": out}, f, indent=2)

    print(f"\n{'arm':<16}{'recall@1':>10}{'recall@10':>11}{'med rank':>10}"
          f"{'gallery':>9}{'min':>7}")
    for k, v in out.items():
        if v:
            print(f"{k:<16}{v['recall@1']:>10.4f}{v['recall@10']:>11.4f}"
                  f"{v['median_rank']:>10.0f}{v['gallery']:>9,}{v['minutes']:>7.1f}")
    if out.get("A_board"):
        print(f"\nchance@1 = {out['A_board']['chance@1']:.5f}")


if __name__ == "__main__":
    main()
