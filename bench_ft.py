"""Bottleneck split for the FINE-TUNE path specifically.

bench_split.py times the pre-training graph: trunk plus candidate scoring, fed by
a 12-ply/32-candidate loader. The fine-tune runs neither of those. It calls
`model.embed` -- trunk, masked pooling, embed head -- against a loader configured
`--ft-n-cand 1 --ft-plies 1`, and its batches come from the PK sampler rather
than a shuffle. Those are different enough that the pre-training numbers do not
transfer, which is why fine-tune speedups were being extrapolated instead of
measured.

Also times the ORIGINAL loader (orig_loader.py) so the end-to-end baseline is a
reading rather than a division.

    python bench_ft.py --shard /workspace/data/2026-06-big --workers 24
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from successor_data import MultiTaskDataset, collate_multitask
from orig_loader import OriginalMultiTaskDataset
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import N_TIME_FEATS, N_TIME_BINS
from contrastive import make_loss
from finetune_mt import PK, embed_bucketed


def build(shard, orig, ft_n_cand, ft_plies):
    cls = OriginalMultiTaskDataset if orig else MultiTaskDataset
    return cls(shard, max_len=160, plies_per_game=ft_plies, n_cand=ft_n_cand)


def loader_for(ds, pid, rows, p, k, workers, seed=0):
    pk = PK(pid[rows], p=p, k=k, batches=10 ** 9, seed=seed)
    return DataLoader(Subset(ds, rows.tolist()), batch_sampler=pk,
                      num_workers=workers, collate_fn=collate_multitask,
                      pin_memory=True, persistent_workers=workers > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--ft-n-cand", type=int, default=1)
    ap.add_argument("--ft-plies", type=int, default=1)
    ap.add_argument("--variants", default="orig,cur,cur+amp,cur+amp+compile")
    ap.add_argument("--out", default="/workspace/bench_ft.json")
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    base = MultiTaskDataset(args.shard, max_len=160, plies_per_game=1, n_cand=1)
    pid = np.array([int(base.meta[g]["white_pid"] if s == 0 else base.meta[g]["black_pid"])
                    for g, s in base.index])
    u, c = np.unique(pid, return_counts=True)
    keep = set(u[c >= args.min_games].tolist())
    rows = np.flatnonzero(np.fromiter((x in keep for x in pid), bool, len(pid)))
    print(f"{len(keep):,} players >= {args.min_games} games | {len(rows):,} rows", flush=True)
    del base

    results = []
    for variant in args.variants.split(","):
        orig = variant == "orig"
        amp = "amp" in variant
        comp = "compile" in variant
        nb = next((int(x[1:]) for x in variant.split("+") if x.startswith("b")), 1)
        static = "static" in variant
        p_mult = next((int(x[1:]) for x in variant.split("+") if x.startswith("p")), 1)

        ds = build(args.shard, orig, args.ft_n_cand, args.ft_plies)
        cfg = Config(d_model=args.d_model, n_layers=args.layers,
                     n_heads=max(8, args.d_model // 64), d_ff=args.d_model * 4)
        m = MultiTaskModel(cfg, n_planes=ds.n_planes, n_extra=N_TIME_FEATS,
                           n_time_bins=N_TIME_BINS, n_elo_bins=N_ELO_BINS).to(dev)
        if comp:
            # Static shapes let CUDA graphs capture the whole step, which is what
            # "reduce-overhead" turns on -- the right mode when the limit is
            # launches rather than arithmetic.
            mode = ("max-autotune" if "auto" in variant else
                    "reduce-overhead" if static else None)
            m.embed = (torch.compile(m.embed, mode=mode) if static
                       else torch.compile(m.embed, dynamic=True))
        # fused=True runs the whole AdamW update as one kernel instead of a few
        # hundred small ones. Irrelevant when a step is arithmetic-bound; this
        # one is launch-bound, so it is exactly the right shape of fix.
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused="fused" in variant)
        loss_fn = make_loss("ms").to(dev)

        MAXLEN = 160

        def pad_static(b):
            """Pad every batch to the same length so the shape never changes.

            Costs FLOPs we have spare and buys a single reusable kernel graph.
            Exact: padding is right-side and causal attention means no real
            position ever attends to it, which is the same reason collate can
            pad at all.
            """
            T = b["planes"].shape[1]
            if T >= MAXLEN:
                return b
            n = MAXLEN - T
            o = dict(b)
            o["planes"] = torch.nn.functional.pad(b["planes"], (0,0,0,0,0,0,0,n))
            o["extra"] = torch.nn.functional.pad(b["extra"], (0, 0, 0, n))
            o["pad_mask"] = torch.nn.functional.pad(b["pad_mask"], (0, n), value=True)
            o["my_turn"] = torch.nn.functional.pad(b["my_turn"], (0, n), value=False)
            return o

        def step(b):
            if static:
                b = pad_static(b)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                emb = embed_bucketed(m, b, nb)
                loss, _ = loss_fn(emb.float(), b["player_id"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        P_eff = args.p * p_mult
        dl = loader_for(ds, pid, rows, P_eff, args.k, args.workers)
        it = iter(dl)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in next(it).items()}
        for _ in range(15 if comp else 8):
            step(b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            step(b)
        torch.cuda.synchronize()
        gpu_only = args.steps / (time.perf_counter() - t0)
        del it, dl

        dl = loader_for(ds, pid, rows, P_eff, args.k, args.workers)
        it = iter(dl)
        for _ in range(5):
            next(it)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            next(it)
        loader_only = args.steps / (time.perf_counter() - t0)
        del it, dl

        dl = loader_for(ds, pid, rows, P_eff, args.k, args.workers)
        it = iter(dl)
        for _ in range(5):
            step({k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                  for k, v in next(it).items()})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            step({k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                  for k, v in next(it).items()})
        torch.cuda.synchronize()
        e2e = args.steps / (time.perf_counter() - t0)

        r = {"variant": variant, "gpu_only_it_s": round(gpu_only, 2),
             "loader_only_it_s": round(loader_only, 2),
             "end_to_end_it_s": round(e2e, 2),
             "batch": P_eff * args.k,
             "samples_per_min": round(e2e * P_eff * args.k * 60),
             "bound_by": "loader" if loader_only < gpu_only else "gpu"}
        results.append(r)
        print(json.dumps(r), flush=True)
        del m, opt, it, dl, ds
        torch.cuda.empty_cache()

    print(f"\n{'variant':>20}{'GPU-only':>10}{'loader':>9}{'actual':>9}"
          f"{'samples/min':>13}{'bound by':>10}{'vs orig':>9}")
    b0 = results[0]["end_to_end_it_s"]
    for r in results:
        print(f"{r['variant']:>20}{r['gpu_only_it_s']:>10.1f}{r['loader_only_it_s']:>9.1f}"
              f"{r['end_to_end_it_s']:>9.1f}{r['samples_per_min']:>13,}"
              f"{r['bound_by']:>10}{r['end_to_end_it_s']/b0:>8.2f}x")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("BENCH_FT_DONE")


if __name__ == "__main__":
    main()
