"""Evaluate the successor scorer against the FULL legal move set.

Training samples 16 candidates per ply, so its accuracy is measured against ~15
distractors. That number flatters the model. This scores every legal successor,
which is the honest "did it pick the move that was actually played" figure.

    /workspace/venv/bin/python eval_successor.py --ckpt /workspace/ckpt/successor/last.pt \
        --shard /workspace/data/2013-01
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from successor_data import SuccessorDataset, MultiShardSuccessorDataset, collate
from model import SuccessorScorer, Config
from train import split_by_game
from train_successor import split_multishard


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shard", help="single month shard")
    src.add_argument("--combined", help="combine.py output spanning several months")
    ap.add_argument("--n-cand", type=int, default=96, help="cap on legal successors scored")
    ap.add_argument("--plies-per-game", type=int, default=8)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--batches", type=int, default=80)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = Config(**ck["cfg"])
    n_planes = ck.get("n_planes", 8)
    model = SuccessorScorer(cfg, n_planes=n_planes).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"checkpoint step {ck['step']}, {n_planes} planes, "
          f"{ck.get('n_cand')} candidates/ply in training")

    kw = dict(max_len=cfg.max_len, plies_per_game=args.plies_per_game,
              n_cand=args.n_cand, with_rights=n_planes == 13)
    if args.combined:
        # Must reuse training's exact hold-out rule, or "held-out" is a lie.
        ds = MultiShardSuccessorDataset(args.combined, **kw)
        _, val_ds = split_multishard(ds, args.val_frac, args.seed)
    else:
        ds = SuccessorDataset(args.shard, **kw)
        _, val_ds = split_by_game(ds, args.val_frac, args.seed)
    dl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers,
                    collate_fn=collate, pin_memory=True)

    hits1 = hits3 = 0
    n = 0
    ranks, ncands = [], []
    for i, b in enumerate(dl):
        if i >= args.batches:
            break
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["planes"], b["cands"], b["ply_idx"])
        logits = logits.float().masked_fill(~b["cand_mask"], float("-inf"))
        sel = b["ply_mask"] & b["cand_mask"].any(-1)
        lg, tg = logits[sel], b["label"][sel]

        true_score = lg.gather(1, tg[:, None])
        rank = (lg > true_score).sum(1)          # 0 = model's top pick
        hits1 += int((rank == 0).sum())
        hits3 += int((rank < 3).sum())
        n += len(tg)
        ranks.append(rank.cpu().numpy())
        ncands.append(b["cand_mask"][sel].sum(-1).cpu().numpy())

    ranks = np.concatenate(ranks)
    ncands = np.concatenate(ncands)
    chance = float(np.mean(1.0 / ncands))

    print(f"\nscored {n:,} plies against all legal successors")
    print(f"  legal moves/ply    mean {ncands.mean():.1f}  median {int(np.median(ncands))}  "
          f"max {ncands.max()}  (hit the {args.n_cand} cap: "
          f"{100*float((ncands>=args.n_cand).mean()):.2f}%)")
    print(f"  top-1              {hits1/n:.3f}   (uniform-over-legal chance {chance:.3f}, "
          f"{(hits1/n)/chance:.1f}x)")
    print(f"  top-3              {hits3/n:.3f}")
    print(f"  median rank of played move   {int(np.median(ranks))}")
    print(f"  mean percentile of played move  "
          f"{100*float(np.mean(1 - ranks/np.maximum(ncands-1,1))):.1f}")


if __name__ == "__main__":
    main()
