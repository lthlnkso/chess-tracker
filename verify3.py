"""Verifier v3: negatives from the gallery shortlist, not from the shard.

v2 trained to saturation (val AUC 0.861) and then placed a real visitor at rank
50 of 100 on the shortlist the product actually serves -- a coin flip. The two
numbers are not in conflict; they answer different questions. v2's negatives are
the nearest players inside a 16,070-player training shard, so it learned "is
this the same player, versus someone unrelated?". The product asks "of 100
people who ALL look like you, which one is you?" and never asked v2 that.

Two changes, both aimed at that gap:

  negatives come from the deployed gallery   build_shortlist.py precomputes each
      training player's top-N nearest centroids out of 558,735. Those are ~35x
      denser than the shard's nearest neighbours, and they are literally the
      candidates identify() will rank.

  candidates come from the verifier pack     which is what serving scores. It
      also removes a trap: if positives came from the shard and shortlist
      negatives from the pack, "pack" alone would predict "negative" and the
      model could score well by reading the source rather than the player.

Batch shape is B queries x (B + M) candidates. The B diagonal candidates are the
queries' own games; the M extras are shortlist players who appear only as
negatives, which is what lets negative density exceed the number of players we
can build queries for. Cost is M extra single-game encodes for B*M extra pairs.

    python verify3.py train --shard data/2026-06-big --ckpt ckpt/final/ctx5_ft2.pt \
        --shortlist play/shortlist.npz --pack play/verifier_pack.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from bitboards import decode_move
from fastboard import N_BB, snapshot_bb, rights_bb
from gallery_ctx import Bundles, Collate, embed_bundles
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import time_features, N_TIME_BINS
from verify2 import DualVerifier
from build_shortlist import players_with_pid


def pack_game(pack, r, j, mlpg, with_rights):
    """One pack game in the same raw form Bundles._game emits.

    Returns None for a game whose moves do not replay -- the pack stores
    truncated games, and a bad push would otherwise poison a whole batch.
    """
    n = int(pack["nply"][r, j])
    if n < 4:
        return None
    codes = pack["moves"][r, j, :n]
    clk = np.asarray(pack["clocks"][r, j, :n])
    seat = int(pack["seat"][r, j])
    T = min(n, mlpg)
    pov = chess.WHITE if seat == 0 else chess.BLACK
    pos = np.zeros((T, N_BB), np.uint64)
    rgt = np.zeros((T, 5), np.int16); rgt[:, 4] = -1
    b = chess.Board()
    for t in range(T):
        pos[t] = snapshot_bb(b, pov)
        if with_rights:
            rgt[t] = rights_bb(b, pov)
        try:
            b.push(decode_move(int(codes[t])))
        except (AssertionError, ValueError):
            return None
    if pov == chess.BLACK:
        pos = pos.byteswap()
        e = rgt[:, 4]; rgt[:, 4] = np.where(e >= 0, e ^ 56, e)
    fe, _, _ = time_features(clk, int(pack["tc_base"][r, j]), int(pack["tc_inc"][r, j]))
    mt = np.zeros(T, bool); mt[seat::2] = True
    return pos, rgt, fe[:T], mt


class BatchSet(Dataset):
    """Item i = one COMPLETE batch: B queries and B+M candidates.

    Batching happens here rather than in the DataLoader because the M extra
    negatives are a property of the batch as a whole -- they are sampled from
    the union of its members' shortlists -- and cannot be expressed as
    independent per-item samples.
    """

    def __init__(self, shard, players, gal_row, shortlist, pack, groups, k, mlpg,
                 with_rights, n_extra, seed):
        self.q = Bundles(shard, [[(0, 0)]], mlpg, with_rights)   # reuse _game only
        self.players = players
        self.gal_row = gal_row
        self.shortlist = shortlist
        self.pack = pack
        self.groups = groups
        self.k, self.mlpg, self.wr = k, mlpg, with_rights
        self.n_extra = n_extra
        self.seed = seed
        self.n_planes = self.q.n_planes
        self.with_rights = with_rights

    def __len__(self):
        return len(self.groups)

    def _codes(self, gi):
        row = self.q.meta[gi]
        o, n = int(row["offset"]), int(row["nply"])
        return np.asarray(self.q.moves[o:o + min(n, self.mlpg)])

    def __getitem__(self, i):
        grp = self.groups[i]
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        q_items, c_items, own = [], [], set()

        for p in grp:
            gg, ss = self.players[p]
            r = int(self.gal_row[p])
            own.add(r)
            have = int(self.pack["have"][r])
            pos_blk, pos_codes = None, None
            for j in rng.permutation(have):
                blk = pack_game(self.pack, r, int(j), self.mlpg, self.wr)
                if blk is not None:
                    pos_blk = blk
                    pos_codes = self.pack["moves"][r, int(j), :int(self.pack["nply"][r, int(j)])]
                    break
            if pos_blk is None:
                continue

            # The query must not contain the positive's own game. The pack is
            # built from the same six months as this shard, so a pack game CAN
            # be one of these shard games -- and a query containing its own
            # answer is a free positive that teaches nothing.
            cand = []
            for idx in rng.permutation(len(gg)):
                c = self._codes(int(gg[idx]))
                if len(c) == len(pos_codes) and np.array_equal(c, pos_codes):
                    continue
                cand.append(int(idx))
                if len(cand) == self.k - 1:
                    break
            if len(cand) < self.k - 1:
                continue
            q_items.append([self.q._game(int(gg[j]), int(ss[j])) for j in cand])
            c_items.append([pos_blk])

        if not q_items:
            return None
        B = len(q_items)

        # Extras: shortlist players who appear ONLY as candidates. Excluding the
        # batch's own gallery rows keeps a query's true match from being handed
        # to it as a labelled negative.
        pool = np.unique(self.shortlist[grp].ravel())
        pool = pool[~np.isin(pool, list(own))]
        rng.shuffle(pool)
        for r in pool:
            if len(c_items) - B >= self.n_extra:
                break
            r = int(r)
            have = int(self.pack["have"][r])
            if have <= 0:
                continue
            blk = pack_game(self.pack, r, int(rng.integers(have)), self.mlpg, self.wr)
            if blk is not None:
                c_items.append([blk])
        return q_items, c_items, B


class BatchCollate:
    def __init__(self, n_planes, n_slots, with_rights):
        self.inner = Collate(n_planes, n_slots, with_rights)

    def __call__(self, batch):
        item = batch[0]
        if item is None:
            return None
        q, c, B = item
        return self.inner(q), self.inner(c), B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train",))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mlpg", type=int, default=60)
    ap.add_argument("--min-games", type=int, default=6)
    ap.add_argument("--neighbours", type=int, default=64)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--extra", type=int, default=96,
                    help="candidate-only negatives per batch, from the shortlists")
    ap.add_argument("--batches-per-epoch", type=int, default=600)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--max-hours", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=800)
    ap.add_argument("--patience", type=int, default=12)
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

    sl = np.load(args.shortlist)
    short_all, pid_all, row_all = sl["shortlist"], sl["pid"], sl["gal_row"]
    if int(sl["min_games"]) != args.min_games:
        raise SystemExit(f"shortlist built with --min-games {int(sl['min_games'])}, "
                         f"not {args.min_games}; rebuild it or match the flag")

    ps = players_with_pid(args.shard, args.min_games)
    # Join by pid rather than trusting position: a mismatch here would pair
    # every player with a stranger's shortlist and still train happily.
    want = {int(p): i for i, p in enumerate(pid_all)}
    keep = [(g, s, want[p]) for p, g, s in ps if p in want and row_all[want[p]] >= 0]
    players = [(g, s) for g, s, _ in keep]
    gal_row = np.array([row_all[j] for _, _, j in keep], np.int64)
    short = np.stack([short_all[j] for _, _, j in keep])
    print(f"{len(players):,} players joined to shortlists "
          f"({short.shape[1]} negatives each, from a {558735:,}-player gallery)",
          flush=True)

    p = np.load(args.pack, allow_pickle=True)
    pack = {k: p[k] for k in ("moves", "clocks", "nply", "seat", "tc_base",
                              "tc_inc", "have")}
    print(f"pack loaded: {int((pack['have'] > 0).sum()):,} players", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(players))
    cut = int(len(idx) * 0.95)
    tr_i, va_i = idx[:cut], idx[cut:]
    print(f"{len(tr_i):,} train | {len(va_i):,} val players", flush=True)
    print(f"batch {args.batch} x {args.batch + args.extra} candidates = "
          f"{args.batch * (args.batch + args.extra):,} pairs/step", flush=True)

    trunk.eval()

    def nn_for(sub):
        """Intra-batch hardness, as in v2 -- the extras add density on top."""
        bs = []
        r = np.random.default_rng(0)
        for j in sub:
            g, s = players[j]
            sel = r.permutation(len(g))[:args.k - 1]
            bs.append([(int(g[x]), int(s[x])) for x in sel])
        E = embed_bundles(trunk, Bundles(args.shard, bs, args.mlpg, wr), slots,
                          device, args.batch, args.workers, "nn")
        E = F.normalize(E.float(), dim=-1)
        nb = min(args.neighbours, max(1, len(E) - 1))
        out = torch.zeros(len(E), nb, dtype=torch.long)
        for i in range(0, len(E), 2048):
            sim = E[i:i + 2048].to(device) @ E.to(device).T
            sim[torch.arange(sim.shape[0]), torch.arange(i, min(i + 2048, len(E)))] = -2
            out[i:i + 2048] = sim.topk(nb, dim=1).indices.cpu()
        return out

    print("neighbour tables...", flush=True)
    nn_tr, nn_va = nn_for(tr_i), nn_for(va_i)

    def groups_for(sub, nn_idx, r, n):
        """Batches from one neighbourhood, mapped back to global player ids."""
        out = []
        for _ in range(n):
            seed = int(r.integers(len(sub)))
            pool = [seed] + [int(x) for x in nn_idx[seed]]
            pool = list(dict.fromkeys(pool))
            if len(pool) < args.batch:
                pool += [int(x) for x in r.choice(len(sub), args.batch - len(pool),
                                                  replace=False)]
            out.append([int(sub[x]) for x in pool[:args.batch]])
        return out

    coll = BatchCollate(ck["n_planes"], slots, wr)

    def loader_for(sub, nn_idx, n, seed):
        r = np.random.default_rng(seed)
        ds = BatchSet(args.shard, players, gal_row, short, pack,
                      groups_for(sub, nn_idx, r, n), args.k, args.mlpg, wr,
                      args.extra, seed)
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                          collate_fn=coll)

    def run_eval():
        """AUC on the REAL task: positives against shortlist candidates."""
        core.eval()
        aucs = []
        with torch.no_grad():
            for item in loader_for(va_i, nn_va, args.eval_batches, 999):
                if item is None:
                    continue
                (qp, qe, qpad, qm, qs, qpp), (cp, ce, cpad, cm, cs, cpp), B = item
                lo = core(qp.to(device), qe.to(device), qpad.to(device), qs.to(device),
                          qpp.to(device), cp.to(device), ce.to(device), cpad.to(device),
                          cs.to(device), cpp.to(device)).float().cpu().numpy()
                pos = np.diag(lo[:, :B])
                mask = np.ones(lo.shape, bool)
                mask[np.arange(B), np.arange(B)] = False
                neg = lo[mask]
                aucs.append(float((pos[:, None] > neg[None, :]).mean()))
        core.train()
        return float(np.mean(aucs)) if aucs else float("nan")

    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=0.05)
    t0 = time.time(); budget = args.max_hours * 3600
    step, best, hist, bad = 0, -1.0, [], 0
    os.makedirs(args.out, exist_ok=True)
    per_step = args.batch * (args.batch + args.extra)

    def save(name):
        torch.save({"model": core.state_dict(), "cfg": cfg.__dict__,
                    "n_planes": ck["n_planes"], "n_extra": ck["n_extra"],
                    "d_embed": ck["d_embed"], "n_game_slots": slots,
                    "max_len_per_game": args.mlpg, "k": args.k, "dual": True,
                    "elo_cond": bool(ck.get("elo_cond")), "step": step,
                    "shortlist_negatives": True},
                   os.path.join(args.out, name))

    while time.time() - t0 < budget:
        for item in loader_for(tr_i, nn_tr, args.batches_per_epoch, step):
            if time.time() - t0 >= budget:
                break
            if item is None:
                continue
            (qp, qe, qpad, qm, qs, qpp), (cp, ce, cpad, cm, cs, cpp), B = item
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                lo = model(qp.to(device), qe.to(device), qpad.to(device), qs.to(device),
                           qpp.to(device), cp.to(device), ce.to(device), cpad.to(device),
                           cs.to(device), cpp.to(device))
                tgt = torch.arange(B, device=device)
                # Query -> candidate runs over ALL B+M candidates, so the extras
                # are exactly the negatives the query must beat. The reverse
                # direction can only use the B candidates that have a query.
                loss = 0.5 * (F.cross_entropy(lo.float(), tgt)
                              + F.cross_entropy(lo.float()[:, :B].T, tgt))
            lv = float(loss.item())
            if not np.isfinite(lv) or lv > 50.0:
                print(f"DIVERGED at step {step}: loss {lv:.4g} -- stopping", flush=True)
                raise SystemExit(3)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step:>6} | loss {lv:.4f} | "
                      f"{step*per_step/1e6:.1f}M pairs | "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
            if step % args.eval_every == 0:
                auc = run_eval()
                hist.append({"step": step, "auc": auc, "pairs": step * per_step})
                flag = ""
                if auc > best:
                    best = auc; flag = " (best)"; bad = 0
                    save("best.pt")
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
          f"{step*per_step/1e6:.1f}M pairs seen")
    save("last.pt")
    json.dump({"args": vars(args), "history": hist},
              open(os.path.join(args.out, "history.json"), "w"))
    print("VERIFY3_DONE")


if __name__ == "__main__":
    main()
