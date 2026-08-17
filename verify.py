"""A verifier: given some of a player's games and ONE more game, same person?

Retrieval by cosine compresses five games into 128 numbers on each side and
compares two summaries. Measured on the real 558,735-player gallery, that puts
the right player at rank 1 for 51.6% of ten-game queries -- but inside the top
1000 for 94.4%. So the shortlist almost always contains the answer and the
ordering is what fails. That gap, 51.6 -> 94.4, is what a verifier is for.

Structure exploits the trunk being CAUSAL. The visitor's games occupy slots
0..K-2 and the candidate game goes last, so candidate plies attend to the
visitor's play while the visitor's plies never see the candidate. Pooling the
candidate's positions therefore reads "this game, in the light of those games",
which is the comparison we actually want to score.

Negatives are the whole game. A random opponent is trivially separable and
teaches nothing: at inference this model only ever sees candidates that cosine
already ranked in the top 1000, i.e. players who ALREADY look alike. So
negatives are drawn from each anchor's nearest neighbours in embedding space.

    python verify.py train --shard data/mt/2026-01 --ckpt ckpt/final/ctx5_ft2.pt
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
from torch.utils.data import DataLoader

from gallery_ctx import Bundles, Collate, embed_bundles
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import N_TIME_BINS


class Verifier(nn.Module):
    """Trunk + a binary head over (candidate pooled, query pooled)."""

    def __init__(self, trunk: MultiTaskModel, d_model: int, cand_slot: int):
        super().__init__()
        self.trunk = trunk
        self.cand_slot = cand_slot
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def _pool(h, keep):
        w = keep.unsqueeze(-1).to(h.dtype)
        return (h * w).sum(1) / w.sum(1).clamp(min=1)

    def forward(self, planes, extra, pad, slot, ppos):
        h = self.trunk.encode(planes, extra, slot, ppos)
        live = ~pad
        cand = live & (slot == self.cand_slot)
        qry = live & (slot < self.cand_slot)
        z = torch.cat([self._pool(h, cand), self._pool(h, qry)], dim=-1)
        return self.head(z).squeeze(-1)


def player_index(shard):
    """(games, seats) per player, clocked games only."""
    meta = np.load(os.path.join(shard, "meta.npy"), mmap_mode="r")
    clocks = np.memmap(os.path.join(shard, "clocks.u16"), dtype=np.uint16, mode="r")
    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
    ok = np.concatenate([np.asarray(clocks[np.asarray(meta["offset"], np.int64)]) != 0xFFFF] * 2)
    o = np.argsort(pid, kind="stable")
    pid, gid, seat, ok = pid[o], gid[o], seat[o], ok[o]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    out = []
    for i in range(len(bnd) - 1):
        sl = slice(bnd[i], bnd[i + 1]); m = ok[sl]
        if m.sum() >= 1:
            out.append((gid[sl][m], seat[sl][m]))
    return out


def neighbour_table(model, shard, players, k, mlpg, wr, slots, device, args):
    """Nearest neighbours in embedding space, for hard negatives.

    One bundle per player is enough to place them roughly; the point is to find
    plausible confusions, not to build a gallery.
    """
    rng = np.random.default_rng(0)
    bundles = []
    for g, s in players:
        sel = rng.permutation(len(g))[:k]
        bundles.append([(int(g[j]), int(s[j])) for j in sel])
    E = embed_bundles(model, Bundles(shard, bundles, mlpg, wr), slots, device,
                      args.batch, args.workers, "neighbours")
    E = F.normalize(E.float(), dim=-1)
    # Chunked: a full 50k x 50k similarity matrix is 10 GB and unnecessary.
    nn_idx = torch.zeros(len(E), args.neighbours, dtype=torch.long)
    step = 2048
    for i in range(0, len(E), step):
        sim = E[i:i + step].to(device) @ E.to(device).T
        sim[torch.arange(sim.shape[0]), torch.arange(i, min(i + step, len(E)))] = -2
        nn_idx[i:i + step] = sim.topk(args.neighbours, dim=1).indices.cpu()
    return nn_idx


def make_epoch(players, nn_idx, k, rng, n_items, hard_frac):
    """Bundles of k game-sides: k-1 from the anchor, the last one the candidate."""
    items, labels = [], []
    n = len(players)
    for _ in range(n_items):
        a = int(rng.integers(n))
        g, s = players[a]
        if len(g) < k:
            continue
        perm = rng.permutation(len(g))
        q = perm[:k - 1]
        pos = bool(rng.integers(2))
        if pos:
            c = int(perm[k - 1])
            cg, cs = int(g[c]), int(s[c])
        else:
            if rng.random() < hard_frac and nn_idx is not None:
                b = int(nn_idx[a][int(rng.integers(nn_idx.shape[1]))])
            else:
                b = int(rng.integers(n))
            if b == a:
                continue
            g2, s2 = players[b]
            j = int(rng.integers(len(g2)))
            cg, cs = int(g2[j]), int(s2[j])
        bundle = [(int(g[j]), int(s[j])) for j in q] + [(cg, cs)]
        items.append(bundle)
        labels.append(1.0 if pos else 0.0)
    return items, np.array(labels, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train",))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--ckpt", required=True, help="trunk to initialise from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5, help="games per bundle incl. candidate")
    ap.add_argument("--players", type=int, default=40000)
    ap.add_argument("--min-games", type=int, default=6)
    ap.add_argument("--neighbours", type=int, default=50)
    ap.add_argument("--hard-frac", type=float, default=0.8)
    ap.add_argument("--items-per-epoch", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--eval-items", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len)
    wr = ck["n_planes"] == 13
    if args.k > slots:
        raise SystemExit(f"k={args.k} exceeds the checkpoint's {slots} game slots")
    trunk = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS, n_game_slots=slots,
                           elo_cond=bool(ck.get("elo_cond")))
    trunk.load_state_dict(ck["model"])
    model = Verifier(trunk, cfg.d_model, args.k - 1).to(device)
    print(f"trunk step {ck.get('step')} | verifier "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params", flush=True)

    allp = player_index(args.shard)
    allp = [p for p in allp if len(p[0]) >= args.min_games]
    rng = np.random.default_rng(args.seed)
    if len(allp) > args.players:
        allp = [allp[i] for i in rng.choice(len(allp), args.players, replace=False)]
    cut = int(len(allp) * 0.95)
    train_p, val_p = allp[:cut], allp[cut:]
    print(f"{len(train_p):,} train players | {len(val_p):,} val", flush=True)

    print("building neighbour table for hard negatives...", flush=True)
    trunk.to(device).eval()
    nn_tr = neighbour_table(trunk, args.shard, train_p, args.k - 1, mlpg, wr,
                            slots, device, args)
    nn_va = neighbour_table(trunk, args.shard, val_p, args.k - 1, mlpg, wr,
                            slots, device, args)
    print(f"  neighbours: {nn_tr.shape}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    coll = Collate(trunk.n_planes if hasattr(trunk, "n_planes") else ck["n_planes"],
                   slots, wr)
    budget = args.max_hours * 3600
    t0 = time.time()
    step, best = 0, -1.0
    hist = []
    os.makedirs(args.out, exist_ok=True)

    def run_eval():
        model.eval()
        items, y = make_epoch(val_p, nn_va, args.k, np.random.default_rng(1234),
                              args.eval_items, args.hard_frac)
        ds = Bundles(args.shard, items, mlpg, wr)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=coll)
        P, at = [], 0
        with torch.no_grad():
            for planes, extra, pad, mine, slot, ppos in dl:
                lo = model(planes.to(device), extra.to(device), pad.to(device),
                           slot.to(device), ppos.to(device))
                P.append(torch.sigmoid(lo.float()).cpu())
        p = torch.cat(P).numpy()[:len(y)]
        yy = y[:len(p)]
        acc = float(((p > 0.5) == (yy > 0.5)).mean())
        # AUC matters more than accuracy: at inference this ranks a shortlist,
        # it does not make yes/no calls.
        order = np.argsort(-p)
        ranks = np.empty_like(order); ranks[order] = np.arange(len(p))
        npos, nneg = yy.sum(), len(yy) - yy.sum()
        auc = float((np.sum(len(p) - ranks[yy > 0.5]) - npos * (npos + 1) / 2)
                    / max(npos * nneg, 1))
        model.train()
        return acc, auc

    while time.time() - t0 < budget:
        items, y = make_epoch(train_p, nn_tr, args.k, rng,
                              args.items_per_epoch, args.hard_frac)
        ds = Bundles(args.shard, items, mlpg, wr)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=coll, drop_last=True)
        yt = torch.from_numpy(y)
        at = 0
        for planes, extra, pad, mine, slot, ppos in dl:
            if time.time() - t0 >= budget:
                break
            tgt = yt[at:at + planes.shape[0]].to(device); at += planes.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                lo = model(planes.to(device), extra.to(device), pad.to(device),
                           slot.to(device), ppos.to(device))
                loss = F.binary_cross_entropy_with_logits(lo.float(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step:>6} | loss {loss.item():.4f} | "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
            if step % 2000 == 0:
                acc, auc = run_eval()
                hist.append({"step": step, "acc": acc, "auc": auc})
                flag = ""
                if auc > best:
                    best = auc; flag = " (best)"
                    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                                "n_planes": ck["n_planes"], "n_extra": ck["n_extra"],
                                "d_embed": ck["d_embed"], "n_game_slots": slots,
                                "max_len_per_game": mlpg, "k": args.k,
                                "elo_cond": bool(ck.get("elo_cond")), "step": step},
                               os.path.join(args.out, "best.pt"))
                print(f"  >> val @ {step}: acc {acc:.4f} auc {auc:.4f}{flag}", flush=True)
                json.dump({"args": vars(args), "history": hist},
                          open(os.path.join(args.out, "history.json"), "w"))

    acc, auc = run_eval()
    print(f"final: acc {acc:.4f} auc {auc:.4f} | best auc {max(best, auc):.4f}")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "n_planes": ck["n_planes"], "n_extra": ck["n_extra"],
                "d_embed": ck["d_embed"], "n_game_slots": slots,
                "max_len_per_game": mlpg, "k": args.k,
                "elo_cond": bool(ck.get("elo_cond")), "step": step},
               os.path.join(args.out, "last.pt"))
    json.dump({"args": vars(args), "history": hist},
              open(os.path.join(args.out, "history.json"), "w"))
    print("VERIFY_TRAIN_DONE")


if __name__ == "__main__":
    main()
