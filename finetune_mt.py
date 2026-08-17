"""Stage 2 for the multi-task trunk: metric-learning fine-tune, then identification eval.

Fine-tune and evaluation live in one script so an overnight run has one fewer
checkpoint hand-off to get wrong. The eval is the only number that matters.

`--loss` selects the objective from contrastive.py; everything else about the
run is held fixed, which is what makes the arms of the sweep comparable.

    python finetune_mt.py --combined data/combined --pretrained ckpt/mt/last.pt \
        --out ckpt/mt_id --max-hours 3 --balance-elo --loss supcon
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Sampler

from successor_data import (MultiTaskDataset, MultiShardMultiTaskDataset,
                            collate_multitask)
from model import MultiTaskModel, Config, supcon_loss, N_ELO_BINS, elo_expectation
from timefeat import N_TIME_FEATS, N_TIME_BINS
from balance import balanced_weights, describe, elo_bins
from contrastive import ALL_LOSSES, make_loss, needs_proxies, default_pk


def lr_at(step, total, base, warmup=300):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.05 * base + 0.95 * base * 0.5 * (1 + math.cos(math.pi * p))


class PK(Sampler):
    """P players x K game-sides. Players may be drawn Elo-balanced."""

    def __init__(self, pid, p=32, k=4, batches=10**9, seed=0, weights=None):
        self.p, self.k, self.batches = p, k, batches
        self.rng = np.random.default_rng(seed)
        order = np.argsort(pid, kind="stable")
        sp = pid[order]
        b = np.flatnonzero(np.r_[True, sp[1:] != sp[:-1], True])
        groups, gw = [], []
        for i in range(len(b) - 1):
            g = order[b[i]:b[i + 1]]
            if len(g) >= k:
                groups.append(g)
                gw.append(weights[g[0]] if weights is not None else 1.0)
        if not groups:
            raise ValueError(f"no player has >= {k} game-sides")
        self.groups = groups
        w = np.asarray(gw, dtype=np.float64)
        self.pw = w / w.sum()

    def __len__(self):
        return self.batches

    def __iter__(self):
        n = len(self.groups)
        for _ in range(self.batches):
            picks = self.rng.choice(n, size=min(self.p, n), replace=False, p=self.pw)
            out = []
            for gi in picks:
                g = self.groups[gi]
                out.extend(self.rng.choice(g, self.k, replace=False).tolist())
            yield out


def embed_bucketed(model, b, n_buckets: int):
    """`model.embed` over length-sorted sub-batches.

    A PK batch is 128 unrelated games, and collate pads all of them to the
    longest one: measured mean length is 65 plies against a 160 cap, so 2.16x of
    the trunk's tokens are padding. Splitting the batch by length and trimming
    each chunk to its own maximum removes most of that.

    Exact, not approximate: attention is per-game, so no game's hidden states
    depend on any other game's presence in the batch. The embeddings are
    identical to the unbucketed ones and the contrastive loss still sees all 128
    at once -- only the padding is gone.

    MEASURED AND REJECTED, default 1. On a 3090 this made things dramatically
    WORSE -- 2 buckets 0.80x, 4 buckets 0.49x, 8 buckets 0.27x -- with and
    without torch.compile. Removing 41% of the FLOPs lost to 4x the kernel
    launches at a quarter the occupancy, because a 7.9M model over 20k tokens
    does not fill the GPU to begin with. The padding was never costing anything.
    Kept, disabled, so the negative result is not rediscovered.
    """
    if n_buckets <= 1:
        return model.embed(b["planes"], b["extra"], b["pad_mask"], b["my_turn"])[0]
    lens = (~b["pad_mask"]).sum(1)
    order = torch.argsort(lens, descending=True)
    out = None
    for idx in torch.chunk(order, n_buckets):
        L = int(lens[idx].max())
        e, _ = model.embed(b["planes"][idx, :L], b["extra"][idx, :L],
                           b["pad_mask"][idx, :L], b["my_turn"][idx, :L])
        if out is None:
            out = torch.empty((len(order),) + e.shape[1:], dtype=e.dtype,
                              device=e.device)
        out[idx] = e
    return out


def pad_static(b, length: int):
    """Right-pad a batch to a fixed length so torch.compile can use CUDA graphs.

    Exact for the same reason collate's variable padding is: padding sits on the
    right and attention is causal, so no real position ever attends to it.
    """
    T = b["planes"].shape[1]
    if T >= length:
        return b
    n = length - T
    F = torch.nn.functional
    o = dict(b)
    o["planes"] = F.pad(b["planes"], (0, 0, 0, 0, 0, 0, 0, n))
    o["extra"] = F.pad(b["extra"], (0, 0, 0, n))
    o["pad_mask"] = F.pad(b["pad_mask"], (0, n), value=True)
    o["my_turn"] = F.pad(b["my_turn"], (0, n), value=False)
    return o


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in b.items()}


@torch.no_grad()
def identify(model, ds, rows, device, batch, workers, centroid_frac=0.8,
             seed=0, log_every=400):
    dl = DataLoader(Subset(ds, rows.tolist()), batch_size=batch, shuffle=False,
                    num_workers=workers, collate_fn=collate_multitask, pin_memory=True)
    E, L, EL, TRUE = [], [], [], []
    model.eval()
    for i, b in enumerate(dl):
        b = to_dev(b, device)
        e, elo_logits = model.embed(b["planes"], b["extra"], b["pad_mask"], b["my_turn"])
        E.append(e.float().cpu()); L.append(b["player_id"].cpu())
        EL.append(elo_expectation(elo_logits).cpu()); TRUE.append(b["elo"].cpu())
        if i % log_every == 0:
            print(f"    embedded {i*batch:,}/{len(rows):,}", flush=True)
    model.train()
    E = torch.cat(E); L = torch.cat(L)
    elo_mae = float((torch.cat(EL) - torch.cat(TRUE).float()).abs().mean())

    rng = np.random.default_rng(seed)
    ln = L.numpy(); order = np.argsort(ln, kind="stable"); sl = ln[order]
    bnd = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1], True])
    cent, clab, qidx = [], [], []
    for i in range(len(bnd) - 1):
        g = order[bnd[i]:bnd[i + 1]]
        if len(g) < 3:
            continue
        perm = rng.permutation(len(g))
        nc = min(max(1, int(round(centroid_frac * len(g)))), len(g) - 1)
        v = E[g[perm[:nc]]].mean(0)
        cent.append(v / v.norm().clamp(min=1e-8)); clab.append(sl[bnd[i]])
        qidx.append(g[perm[nc:]])
    C = torch.stack(cent).to(device); CL = torch.tensor(clab).to(device)
    qi = np.concatenate(qidx); Q, QL = E[qi], L[qi]

    hits = {1: 0, 10: 0, 100: 0}; ranks = []
    maxk = min(100, len(CL))
    for s in range(0, len(Q), 2048):
        q = Q[s:s + 2048].to(device); ql = QL[s:s + 2048].to(device)
        sim = q @ C.T
        top = sim.topk(maxk, dim=1).indices
        m = CL[top] == ql[:, None]
        for k in hits:
            if k <= maxk:
                hits[k] += int(m[:, :k].any(1).sum())
        tc = (CL[None, :] == ql[:, None]).float().argmax(1)
        ranks.append((sim > sim.gather(1, tc[:, None])).sum(1).cpu())
    ranks = torch.cat(ranks).float()
    n = len(Q)
    return {"gallery": int(len(CL)), "queries": n,
            **{f"recall@{k}": hits[k] / n for k in hits},
            "median_rank": float(ranks.median()),
            "chance@1": 1.0 / len(CL), "elo_mae": elo_mae}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shard", help="single month shard")
    src.add_argument("--combined", help="combine.py index across months")
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=100_000_000)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--eval-players", type=int, default=20000)
    ap.add_argument("--eval-batch", type=int, default=192)
    ap.add_argument("--balance-elo", action="store_true")
    ap.add_argument("--collapse-after", type=int, default=2000)
    ap.add_argument("--loss", default="supcon", choices=ALL_LOSSES)
    ap.add_argument("--probe-every-hours", type=float, default=0.0,
                    help="periodic identification probe on a small gallery. The "
                         "loss curve cannot tell you whether recall has stopped "
                         "improving; only this can. 0 disables.")
    ap.add_argument("--probe-players", type=int, default=2000)
    ap.add_argument("--ft-n-cand", type=int, default=1)
    ap.add_argument("--ft-plies", type=int, default=1)
    ap.add_argument("--proxy-warmup", type=int, default=1500,
                    help="steps with the trunk frozen while the proxy bank "
                         "aligns to the existing embedding space. Without it a "
                         "randomly-initialised bank drags the trunk to a point "
                         "before it carries any identity information.")
    ap.add_argument("--proxy-lr-mult", type=float, default=10.0,
                    help="LR multiplier on the proxy bank; a proxy is touched "
                         "far less often than a trunk weight, so it needs one")
    ap.add_argument("--d-embed", type=int, default=0,
                    help="override the checkpoint's embedding width (0 = keep)")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast. +49%% end-to-end on a 3090, but unlike the "
                         "loader work it CHANGES NUMERICS -- A/B a short run before "
                         "trusting it for a real one")
    ap.add_argument("--static-len", type=int, default=0,
                    help="pad every batch to this length so the shape never "
                         "varies, letting --compile use CUDA graphs. Padding is "
                         "free here (the step is launch-bound, not FLOP-bound) "
                         "and exact (causal attention never reads right-padding). "
                         "160 matches the model's max_len. ~+5%%, near the noise.")
    ap.add_argument("--embed-buckets", type=int, default=1,
                    help="length-sorted sub-batches per step. Batches pad to the "
                         "longest game (2.16x waste measured); 4-8 buckets removes "
                         "most of it. Exact -- attention never crosses games.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (+19%% on top of --amp)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    ck = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    # 1 candidate at 1 ply, not the 8x8 the pre-training loader uses. Nothing in
    # this stage reads them -- `model.embed` takes planes/extra/masks only -- and
    # generating them is ~45% of the per-sample board encoding, which is the
    # bottleneck here (python-chess move generation on the CPU workers, not the
    # GPU). They cannot be dropped entirely because collate_multitask expects the
    # keys.
    _kw = dict(max_len=cfg.max_len, plies_per_game=args.ft_plies,
               n_cand=args.ft_n_cand, with_rights=ck["n_planes"] == 13)
    ds = (MultiShardMultiTaskDataset(args.combined, **_kw) if args.combined
          else MultiTaskDataset(args.shard, **_kw))
    assert ds.n_planes == ck["n_planes"], "loader/model plane mismatch"

    d_embed = args.d_embed or ck["d_embed"]
    model = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=d_embed, n_time_bins=ck["n_time_bins"],
                           n_elo_bins=ck["n_elo_bins"]).to(device)
    # Only the final embed_head layer depends on d_embed, and it is untrained
    # during pre-training anyway (no gradient reaches it), so dropping shape
    # mismatches loses nothing. Everything else -- the whole trunk, the time and
    # Elo heads -- must load, and we assert that it did.
    own = model.state_dict()
    take = {k: v for k, v in ck["model"].items()
            if k in own and own[k].shape == v.shape}
    dropped = sorted(set(ck["model"]) - set(take))
    model.load_state_dict(take, strict=False)
    assert all(k.startswith("embed_head") for k in dropped), \
        f"unexpected weights dropped: {dropped}"
    print(f"loaded trunk from {args.pretrained} (step {ck['step']}) "
          f"| d_embed {ck['d_embed']} -> {d_embed} | "
          f"{len(take)}/{len(own)} tensors, dropped {dropped}", flush=True)

    if args.combined:
        pid = np.asarray(ds.idx["gpid"]).astype(np.int64)
        elo = np.array([int(ds.metas[int(r["shard"])][int(r["game"])]
                            ["white_elo" if int(r["seat"]) == 0 else "black_elo"])
                        for r in ds.idx])
    else:
        pid = np.array([int(ds.meta[g]["white_pid"] if s == 0 else ds.meta[g]["black_pid"])
                        for g, s in ds.index])
        elo = np.array([int(ds.meta[g]["white_elo"] if s == 0 else ds.meta[g]["black_elo"])
                        for g, s in ds.index])
    u, c = np.unique(pid, return_counts=True)
    keep = set(u[c >= args.min_games].tolist())
    rows = np.flatnonzero(np.fromiter((p in keep for p in pid), bool, len(pid)))

    rng = np.random.default_rng(args.seed)
    players = np.unique(pid[rows])
    test_p = set(rng.choice(players, int(len(players) * args.test_frac),
                            replace=False).tolist())
    is_test = np.fromiter((p in test_p for p in pid[rows]), bool, len(rows))
    tr_rows, te_rows = rows[~is_test], rows[is_test]
    print(f"{len(players):,} players >= {args.min_games} games | "
          f"train {len(tr_rows):,} rows | test {len(te_rows):,} rows "
          f"({len(test_p):,} held-out players)", flush=True)

    wts = None
    if args.balance_elo:
        w_all = np.zeros(len(pid))
        w_all[tr_rows] = balanced_weights(elo[tr_rows])
        wts = w_all
        print(f"Elo balancing (train): {describe(elo[tr_rows], w_all[tr_rows])}", flush=True)

    # Batch shape is loss-dependent but P*K is not, so every arm sees the same
    # number of game-sides per step and the wall-clock comparison stays fair.
    p_eff, k_eff = default_pk(args.loss, args.p, args.k)
    if (p_eff, k_eff) != (args.p, args.k):
        print(f"{args.loss}: PK {args.p}x{args.k} -> {p_eff}x{k_eff} "
              f"(same {p_eff*k_eff} game-sides/step)", flush=True)
    pk = PK(pid[tr_rows], p=p_eff, k=k_eff, batches=args.steps, seed=args.seed,
            weights=(wts[tr_rows] if wts is not None else None))
    dl = DataLoader(Subset(ds, tr_rows.tolist()), batch_sampler=pk,
                    num_workers=args.workers, collate_fn=collate_multitask,
                    pin_memory=True, persistent_workers=args.workers > 0)

    # Proxy losses need one vector per *train* player, so raw lichess ids get
    # remapped onto a contiguous range. Test players deliberately have no proxy:
    # the gallery is built from embeddings, never from the bank.
    n_classes, cls_lut = 0, None
    if needs_proxies(args.loss):
        train_players = np.unique(pid[tr_rows])
        n_classes = len(train_players)
        lut = np.full(int(pid.max()) + 1, -1, dtype=np.int64)
        lut[train_players] = np.arange(n_classes)
        cls_lut = torch.from_numpy(lut).to(device)
        print(f"{args.loss}: {n_classes:,} proxies x {d_embed} "
              f"= {n_classes*d_embed/1e6:.1f}M params", flush=True)
    loss_fn = make_loss(args.loss, n_classes=n_classes, d_embed=d_embed).to(device)

    # See train_multitask.py: `core` is the uncompiled module and is the only
    # thing state_dict() is ever taken from. This stage calls model.embed(),
    # not forward(), and torch.compile only traces forward -- so the method
    # itself is what gets compiled.
    core = model
    if args.compile:
        # reduce-overhead == CUDA graphs, which need a shape that never changes;
        # only safe to ask for when --static-len pins it.
        model.embed = (torch.compile(model.embed, mode="reduce-overhead")
                       if args.static_len else
                       torch.compile(model.embed, dynamic=True))

    groups = [{"params": list(model.parameters()), "lr": args.lr, "mult": 1.0}]
    proxy_params = list(loss_fn.parameters())
    if proxy_params:
        groups.append({"params": proxy_params, "lr": args.lr * args.proxy_lr_mult,
                       "mult": args.proxy_lr_mult})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))

    def save(step):
        # The proxy bank is deliberately not saved: nothing downstream reads it
        # (the gallery is built from embeddings) and at 250k x 128 it would
        # quadruple the checkpoint for no consumer.
        torch.save({"model": core.state_dict(), "cfg": cfg.__dict__,
                    "n_planes": ck["n_planes"], "n_extra": ck["n_extra"],
                    "d_embed": d_embed, "n_time_bins": ck["n_time_bins"],
                    "n_elo_bins": ck["n_elo_bins"], "step": step,
                    "loss": args.loss},
                   os.path.join(args.out, "last.pt"))

    # A fixed subset of the held-out players, so successive probes measure the
    # model changing and not the gallery changing.
    probe_rows = np.array([], dtype=np.int64)
    if args.probe_every_hours:
        pg = np.unique(pid[te_rows])
        if len(pg) > args.probe_players:
            pg = set(rng.choice(pg, args.probe_players, replace=False).tolist())
            probe_rows = te_rows[np.fromiter((p in pg for p in pid[te_rows]),
                                             bool, len(te_rows))]
        else:
            probe_rows = te_rows
        print(f"probe: {len(probe_rows):,} rows / {min(len(pg), args.probe_players):,} "
              f"players every {args.probe_every_hours}h", flush=True)

    budget = args.max_hours * 3600
    curve = []
    probes = []
    t0 = time.time()
    next_probe = args.probe_every_hours * 3600
    step = 0
    collapsed = False
    model.train()
    for b in dl:
        if budget and (time.time() - t0) >= budget:
            print(f"reached --max-hours {args.max_hours} at step {step}", flush=True)
            break
        frac = min(1.0, (time.time() - t0) / budget) if budget else step / args.steps
        lr_now = lr_at(max(int(frac * 100_000), min(step, args.warmup)), 100_000,
                       args.lr, args.warmup)
        # lr 0 rather than requires_grad=False: AdamW applies decoupled weight
        # decay as p -= lr*wd*p, so a zero lr freezes the trunk completely,
        # whereas detaching the graph would still let decay shrink it.
        warming = bool(proxy_params) and step < args.proxy_warmup
        for i, g in enumerate(opt.param_groups):
            g["lr"] = 0.0 if (warming and i == 0) else lr_now * g["mult"]
        b = to_dev(b, device)
        if args.static_len:
            b = pad_static(b, args.static_len)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            emb = embed_bucketed(model, b, args.embed_buckets)
            tgt = b["player_id"] if cls_lut is None else cls_lut[b["player_id"]]
            loss, st = loss_fn(emb.float(), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % 200 == 0:
            dt = time.time() - t0
            print(f"step {step:>7} | loss {float(loss):.4f} | pos {st['pos_cos']:+.3f} "
                  f"neg {st['neg_cos']:+.3f} gap {st['gap']:+.4f} | lr {lr_now:.2e} | "
                  f"{step/dt:5.1f} it/s | {dt/60:6.1f} min", flush=True)
            curve.append({"step": step, "loss": float(loss), **st,
                          "minutes": round(dt / 60, 2)})
        # Counted from the end of any warmup -- during warmup the trunk is frozen
        # by construction, so a flat gap there is expected, not a collapse.
        if (step == args.collapse_after + (args.proxy_warmup if proxy_params else 0)
                and st["gap"] < 0.02):
            # Break rather than exit: a collapsed arm is a result the sweep needs
            # recorded, and the eval that records it costs minutes while the rest
            # of the fine-tune budget would cost hours.
            collapsed = True
            print(f"COLLAPSE: gap {st['gap']:.4f} at step {step} -- "
                  f"stopping early, going straight to eval", flush=True)
            break
        if step % 4000 == 0:
            save(step)

        if args.probe_every_hours and (time.time() - t0) >= next_probe:
            next_probe += args.probe_every_hours * 3600
            pr = identify(model, ds, probe_rows, device, args.eval_batch,
                          args.workers, seed=args.seed, log_every=10**9)
            pr.update(step=step, hours=round((time.time() - t0) / 3600, 2))
            probes.append(pr)
            print(f"  >> probe @ {pr['hours']}h step {step}: "
                  f"r@1 {pr['recall@1']:.4f} r@10 {pr['recall@10']:.4f} "
                  f"median_rank {pr['median_rank']:.0f} "
                  f"(gallery {pr['gallery']:,})", flush=True)
            with open(os.path.join(args.out, "probes.json"), "w") as f:
                json.dump(probes, f, indent=2)

    save(step)
    print(f"fine-tune done: {step:,} steps in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- identification eval on the natural (unbalanced) held-out set ----
    gal = np.unique(pid[te_rows])
    if len(gal) > args.eval_players:
        gal = set(rng.choice(gal, args.eval_players, replace=False).tolist())
        te_rows = te_rows[np.fromiter((p in gal for p in pid[te_rows]), bool, len(te_rows))]
    print(f"evaluating on {len(te_rows):,} rows", flush=True)
    res = identify(model, ds, te_rows, device, args.eval_batch, args.workers,
                   seed=args.seed)
    print(json.dumps(res, indent=2), flush=True)
    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump({"args": vars(args), "loss": args.loss, "d_embed": d_embed,
                   "pk": [p_eff, k_eff], "n_classes": n_classes,
                   "collapsed": collapsed, "steps": step,
                   "minutes": round((time.time() - t0) / 60, 2),
                   "result": res, "probes": probes, "curve": curve}, f, indent=2)
    print("IDENTIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
