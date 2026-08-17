"""Verifier v2: same idea, ~200x the training signal per GPU-hour.

Profiling the v1 trainer found three things, in increasing order of importance:

  1. it never called torch.compile, unlike every other trainer here (~1.5x)
  2. AUC is FLAT from 160 plies per game down to 40 -- 0.8297 vs 0.8270 -- while
     cost falls 3x. The identity signal is in the opening and early middlegame;
     the endgame is paid for and ignored.
  3. every training pair encoded 5 games, 4 of which were the query, re-encoded
     from scratch for each candidate. One label cost five game encodings.

(3) is the structural one. Encode B queries and B candidate games -- the same
5B game encodings v1 already paid for -- then score ALL B x B combinations
instead of B. At batch 48 that is 2,304 pairs per step instead of 48.

The catch is that in-batch negatives are normally RANDOM players, and hard
negatives are exactly what this task needs: at inference every candidate has
already been ranked into a shortlist by cosine, so easy negatives are a
distribution the model will never meet. So batches are built from ONE
neighbourhood -- B players that cosine already considers similar -- which makes
every off-diagonal pair a hard negative for free, and harder than v1's sampler
produced.

What this gives up: query and candidate no longer share a sequence, so there is
no full cross-attention between them. The interaction is recovered cheaply in
the head, which runs over pooled vectors rather than 800 positions. That is a
real loss of expressiveness, and whether it costs accuracy is the thing this
script exists to measure.

    python verify2.py train --shard data/mt/2026-01 --ckpt ckpt/final/ctx5_ft2.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from gallery_ctx import Bundles, Collate, embed_bundles
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import N_TIME_BINS
from verify import player_index


class DualVerifier(nn.Module):
    """Encode each side once; score every pair in the batch."""

    def __init__(self, trunk: MultiTaskModel, d_model: int):
        super().__init__()
        self.trunk = trunk
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_c = nn.Linear(d_model, d_model)
        # Runs over pooled vectors, so B^2 rows cost almost nothing. Fed the
        # elementwise product and absolute difference rather than the raw
        # concatenation: those are the interactions a dot product cannot express,
        # which is the whole reason this is not just cosine again.
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # FIXED temperature, not learnable. A learnable scale multiplying the
        # logits is an unbounded path to overflow: under bf16 the first run
        # diverged to loss 4.9e6 and then collapsed to exactly ln(B) -- dead,
        # uniform logits -- for the rest of two hours.
        self.temp = 4.0

    @staticmethod
    def _pool(h, keep):
        w = keep.unsqueeze(-1).to(h.dtype)
        return (h * w).sum(1) / w.sum(1).clamp(min=1)

    def encode(self, planes, extra, pad, slot, ppos, proj):
        h = self.trunk.encode(planes, extra, slot, ppos)
        return proj(self._pool(h, ~pad))

    def pair_logits(self, q, c):
        """(B, D) x (B, D) -> (B, B) logits for every query/candidate pair.

        Both sides are L2-normalised first, so the product and difference are
        bounded regardless of what the trunk emits. Unnormalised, their
        magnitudes are free to grow and bf16 has ~3 decimal digits to hold them.
        """
        q = F.normalize(q.float(), dim=-1)
        c = F.normalize(c.float(), dim=-1)
        prod = q[:, None, :] * c[None, :, :]
        diff = (q[:, None, :] - c[None, :, :]).abs()
        z = torch.cat([prod, diff], dim=-1)
        return self.head(z).squeeze(-1) * self.temp

    def forward(self, qp, qe, qpad, qs, qpp, cp, ce, cpad, cs, cpp):
        q = self.encode(qp, qe, qpad, qs, qpp, self.proj_q)
        c = self.encode(cp, ce, cpad, cs, cpp, self.proj_c)
        with torch.autocast("cuda", enabled=False):
            return self.pair_logits(q, c)


class PairSet(Dataset):
    """Item i = (that player's query games, one more game of theirs)."""

    def __init__(self, shard, qb, cb, mlpg, wr):
        self.q = Bundles(shard, qb, mlpg, wr)
        self.c = Bundles(shard, cb, mlpg, wr)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        return self.q[i], self.c[i]


class PairCollate:
    def __init__(self, n_planes, n_slots, with_rights):
        self.inner = Collate(n_planes, n_slots, with_rights)

    def __call__(self, batch):
        return self.inner([b[0] for b in batch]), self.inner([b[1] for b in batch])


def neighbourhood_batches(nn_idx, batch, rng, n_batches):
    """Batches drawn from one neighbourhood, so off-diagonals are hard."""
    n = nn_idx.shape[0]
    out = []
    for _ in range(n_batches):
        seed = int(rng.integers(n))
        pool = [seed] + [int(x) for x in nn_idx[seed]]
        pool = list(dict.fromkeys(pool))            # keep order, drop repeats
        if len(pool) < batch:
            pool += [int(x) for x in rng.choice(n, batch - len(pool), replace=False)]
        out.append(pool[:batch])
    return out


def build_pairs(players, groups, k, rng):
    qb, cb = [], []
    for grp in groups:
        for p in grp:
            g, s = players[p]
            perm = rng.permutation(len(g))
            qb.append([(int(g[j]), int(s[j])) for j in perm[:k - 1]])
            j = int(perm[k - 1])
            cb.append([(int(g[j]), int(s[j]))])
    return qb, cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train",))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mlpg", type=int, default=60,
                    help="plies per game. AUC measured flat from 160 down to 40 "
                         "while cost falls 3x, so the default is short.")
    ap.add_argument("--players", type=int, default=40000)
    ap.add_argument("--min-games", type=int, default=6)
    ap.add_argument("--neighbours", type=int, default=64)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--batches-per-epoch", type=int, default=400)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--max-hours", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=800)
    ap.add_argument("--patience", type=int, default=12,
                    help="evals without an AUC improvement before stopping. The "
                         "point of a saturation run is to stop when it plateaus, "
                         "not to burn the clock.")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    slots = ck.get("n_game_slots", 1)
    wr = ck["n_planes"] == 13
    trunk = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS, n_game_slots=slots,
                           elo_cond=bool(ck.get("elo_cond")))
    trunk.load_state_dict(ck["model"])
    model = DualVerifier(trunk, cfg.d_model).to(device)
    core = model
    if args.compile:
        model = torch.compile(model, dynamic=True)
    print(f"trunk step {ck.get('step')} | {sum(p.numel() for p in core.parameters())/1e6:.2f}M "
          f"params | mlpg {args.mlpg} | batch {args.batch} -> "
          f"{args.batch**2:,} pairs/step", flush=True)

    allp = [p for p in player_index(args.shard) if len(p[0]) >= args.min_games]
    rng = np.random.default_rng(args.seed)
    if len(allp) > args.players:
        allp = [allp[i] for i in rng.choice(len(allp), args.players, replace=False)]
    cut = int(len(allp) * 0.95)
    train_p, val_p = allp[:cut], allp[cut:]
    print(f"{len(train_p):,} train | {len(val_p):,} val players", flush=True)

    class A:
        batch = args.batch; workers = args.workers; neighbours = args.neighbours
    print("neighbour tables...", flush=True)
    trunk.eval()
    def nn_for(ps):
        bs = []
        r = np.random.default_rng(0)
        for g, s in ps:
            sel = r.permutation(len(g))[:args.k - 1]
            bs.append([(int(g[j]), int(s[j])) for j in sel])
        E = embed_bundles(trunk, Bundles(args.shard, bs, args.mlpg, wr), slots,
                          device, args.batch, args.workers, "nn")
        E = F.normalize(E.float(), dim=-1)
        # A small split can hold fewer players than the requested neighbourhood.
        nb = min(args.neighbours, max(1, len(E) - 1))
        out = torch.zeros(len(E), nb, dtype=torch.long)
        for i in range(0, len(E), 2048):
            sim = E[i:i + 2048].to(device) @ E.to(device).T
            sim[torch.arange(sim.shape[0]), torch.arange(i, min(i + 2048, len(E)))] = -2
            out[i:i + 2048] = sim.topk(nb, dim=1).indices.cpu()
        return out
    nn_tr, nn_va = nn_for(train_p), nn_for(val_p)

    coll = PairCollate(ck["n_planes"], slots, wr)
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=0.05)
    t0 = time.time(); budget = args.max_hours * 3600
    step, best, hist, bad = 0, -1.0, [], 0
    os.makedirs(args.out, exist_ok=True)

    def loader_for(ps, nn_idx, nb, seed):
        r = np.random.default_rng(seed)
        groups = neighbourhood_batches(nn_idx, args.batch, r, nb)
        qb, cb = build_pairs(ps, groups, args.k, r)
        ds = PairSet(args.shard, qb, cb, args.mlpg, wr)
        return DataLoader(ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, collate_fn=coll, drop_last=True)

    def run_eval():
        core.eval()
        aucs = []
        with torch.no_grad():
            for (qp, qe, qpad, qm, qs, qpp), (cp, ce, cpad, cm, cs, cpp) in \
                    loader_for(val_p, nn_va, args.eval_batches, 999):
                lo = core(qp.to(device), qe.to(device), qpad.to(device), qs.to(device),
                          qpp.to(device), cp.to(device), ce.to(device), cpad.to(device),
                          cs.to(device), cpp.to(device)).float().cpu().numpy()
                B = lo.shape[0]
                pos = np.diag(lo)
                neg = lo[~np.eye(B, dtype=bool)]
                aucs.append(float((pos[:, None] > neg[None, :]).mean()))
        core.train()
        return float(np.mean(aucs))

    while time.time() - t0 < budget:
        for (qp, qe, qpad, qm, qs, qpp), (cp, ce, cpad, cm, cs, cpp) in \
                loader_for(train_p, nn_tr, args.batches_per_epoch, step):
            if time.time() - t0 >= budget:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                lo = model(qp.to(device), qe.to(device), qpad.to(device), qs.to(device),
                           qpp.to(device), cp.to(device), ce.to(device), cpad.to(device),
                           cs.to(device), cpp.to(device))
                B = lo.shape[0]
                tgt = torch.arange(B, device=device)
                # Symmetric InfoNCE: each query must pick its own candidate out of
                # the batch, and each candidate its own query. Directly optimises
                # ranking, which is what a shortlist re-order needs.
                loss = 0.5 * (F.cross_entropy(lo.float(), tgt)
                              + F.cross_entropy(lo.float().T, tgt))
            lv = float(loss.item())
            # The first run sat at exactly ln(B) for two hours after diverging.
            # Stop instead of paying for a dead model.
            if not np.isfinite(lv) or lv > 50.0:
                print(f"DIVERGED at step {step}: loss {lv:.4g} -- stopping", flush=True)
                raise SystemExit(3)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                pairs = step * args.batch ** 2
                print(f"step {step:>6} | loss {loss.item():.4f} | "
                      f"{pairs/1e6:.1f}M pairs | {(time.time()-t0)/60:.1f} min", flush=True)
            if step % args.eval_every == 0:
                auc = run_eval()
                hist.append({"step": step, "auc": auc,
                             "pairs": step * args.batch ** 2})
                flag = ""
                if auc > best:
                    best = auc; flag = " (best)"; bad = 0
                    torch.save({"model": core.state_dict(), "cfg": cfg.__dict__,
                                "n_planes": ck["n_planes"], "n_extra": ck["n_extra"],
                                "d_embed": ck["d_embed"], "n_game_slots": slots,
                                "max_len_per_game": args.mlpg, "k": args.k,
                                "dual": True, "elo_cond": bool(ck.get("elo_cond")),
                                "step": step},
                               os.path.join(args.out, "best.pt"))
                else:
                    bad += 1
                print(f"  >> val @ {step}: auc {auc:.4f}{flag}"
                      f"{'' if not bad else f' ({bad}/{args.patience})'}", flush=True)
                if bad >= args.patience:
                    print(f"saturated: {args.patience} evals without improvement",
                          flush=True)
                    budget = 0
                json.dump({"args": vars(args), "history": hist},
                          open(os.path.join(args.out, "history.json"), "w"))

    auc = run_eval()
    print(f"final auc {auc:.4f} | best {max(best, auc):.4f} | "
          f"{step * args.batch**2/1e6:.1f}M pairs seen")
    torch.save({"model": core.state_dict(), "cfg": cfg.__dict__,
                "n_planes": ck["n_planes"], "n_extra": ck["n_extra"],
                "d_embed": ck["d_embed"], "n_game_slots": slots,
                "max_len_per_game": args.mlpg, "k": args.k, "dual": True,
                "elo_cond": bool(ck.get("elo_cond")), "step": step},
               os.path.join(args.out, "last.pt"))
    json.dump({"args": vars(args), "history": hist},
              open(os.path.join(args.out, "history.json"), "w"))
    print("VERIFY2_TRAIN_DONE")


if __name__ == "__main__":
    main()
