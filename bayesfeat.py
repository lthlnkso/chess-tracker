"""Hand features as a likelihood-ratio term in the Bayesian identifier.

Fits a logistic on standardised query-vs-candidate feature differences, over
BALANCED same/different pairs. Fit that way, the output logit is the
log-likelihood ratio directly, so it can be added to the cosine prior:

    logit P(you) = cos_logit + feat_llr + SUM_games verifier_llr

Two reasons this is a logistic rather than one Gaussian LLR per feature:

  the features are correlated   the six piece fractions sum to 1, so summing
                                per-feature LLRs would count the same evidence
                                up to six times. A joint fit learns the real
                                weighting.
  the weights are the report    a coefficient near zero says a feature carries
                                nothing the others don't. That is the answer to
                                "is this worth shipping", and it comes free.

Unlike the verifier, this scores the WHOLE gallery -- it is a single matvec over
558k rows -- so it can rescue a player cosine ranked outside the shortlist.

    python bayesfeat.py fit  --out play/feat_llr.json
    python bayesfeat.py eval --coef play/feat_llr.json
"""

from __future__ import annotations

import argparse
import json

import chess
import numpy as np
import torch
import torch.nn.functional as F

from bitboards import decode_move
from handfeat import N_FEATS, NAMES, game_features, opening_hash

PACK_PLIES = 60          # the gallery profile only ever saw this many


def profile_from_uci(games, plies=PACK_PLIES):
    """Visitor-side profile, matched to how the gallery profile was built.

    Two alignments matter and both are easy to get wrong:

      truncate to `plies`   the gallery profile was computed from 60-ply
                            prefixes. Comparing a full-length query mean against
                            it would bias every length-sensitive feature.
      quantise think time   lichess [%clk] ticks once per second, so every
                            gallery think time is a whole number of seconds. The
                            browser measures real milliseconds. Left alone, the
                            query's mean_think sits systematically off the
                            gallery's and the timing features -- the strongest
                            ones we have -- turn into noise.
    """
    rows, opens = [], []
    for g in games:
        hist = list(g.get("history", []))[:plies]
        if len(hist) < 16:
            continue
        seat = 0 if bool(g.get("human_white", True)) else 1
        codes, b = [], chess.Board()
        for u in hist:
            mv = chess.Move.from_uci(u)
            codes.append(mv)
            b.push(mv)
        times = list(g.get("times") or [])[:len(hist)]
        # Rebuild a whole-second clock track, then let game_features derive
        # think times from it exactly as it does for shard games.
        clk = np.full(len(hist), 0xFFFF, np.uint16)
        rem = [60.0, 60.0]
        for t in range(len(hist)):
            side = t % 2
            used = (times[t] / 1000.0) if t < len(times) else 0.0
            rem[side] = max(rem[side] - round(used), 0.0)
            clk[t] = int(min(rem[side] * 100, 0xFFFE))
        f = _features_from_moves(codes, clk, seat)
        if f is None:
            continue
        rows.append(f)
        opens.append(_opening_hash_moves(codes, seat))
    if not rows:
        return None, []
    return np.stack(rows).mean(0), [o for o in opens if o]


def _features_from_moves(moves, clk, seat):
    """game_features works from packed codes; the demo has chess.Move objects."""
    from bitboards import encode_move
    codes = np.array([encode_move(m) for m in moves], np.uint16)
    return game_features(codes, clk, seat)


def _opening_hash_moves(moves, seat, plies=6):
    from bitboards import encode_move
    if len(moves) < plies:
        return 0
    codes = np.array([encode_move(m) for m in moves[:plies]], np.uint16)
    return opening_hash(codes, seat, plies)


class Table:
    """The gallery-side hand-feature profile, plus an opening index."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.F = np.asarray(z["feats"], np.float32)
        self.O = np.asarray(z["opens"], np.uint64)
        self.n = np.asarray(z["n"], np.uint8)
        self.names = [str(x).lower() for x in z["names"]]
        ok = self.n > 0
        self.mu = self.F[ok].mean(0)
        self.sd = self.F[ok].std(0) + 1e-6
        # Flat (hash, row) index so an opening match is a sorted lookup rather
        # than a scan of 2.2M entries per query game.
        flat = self.O.reshape(-1)
        rows = np.repeat(np.arange(len(self.F)), self.O.shape[1])
        m = flat != 0
        self.oh, self.orow = flat[m], rows[m]
        o = np.argsort(self.oh, kind="stable")
        self.oh, self.orow = self.oh[o], self.orow[o]

    def overlap(self, q_opens):
        """How many of the query's openings each gallery player also plays."""
        c = np.zeros(len(self.F), np.float32)
        for h in set(int(x) for x in q_opens if x):
            lo = np.searchsorted(self.oh, np.uint64(h), "left")
            hi = np.searchsorted(self.oh, np.uint64(h), "right")
            if hi > lo:
                np.add.at(c, self.orow[lo:hi], 1.0)
        return c

    def design(self, qfeat, q_opens, rows=None):
        """|z-difference| per feature, plus opening overlap. One row per player."""
        Fc = self.F if rows is None else self.F[rows]
        z = np.abs((Fc - qfeat[None, :]) / self.sd[None, :])
        ov = self.overlap(q_opens)
        ov = ov if rows is None else ov[rows]
        nn = (self.n if rows is None else self.n[rows]).astype(np.float32)
        return np.c_[z, ov, (nn == 0).astype(np.float32)].astype(np.float32)


N_DESIGN = N_FEATS + 2


def fit_logistic_mv(X, y, l2=1e-3, iters=1500, lr=0.05):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(iters):
        loss = F.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach().numpy(), float(b.item())


def shard_queries(shard, table, n_players, k_games, seed=0,
                  plies=PACK_PLIES, offset=0):
    """Query sets from the shard, with pack games excluded.

    The gallery profile was built from these same six months, so a query game
    can be one of the four games already in the candidate's profile. Scoring a
    game against a profile that contains it is leakage and would make every
    number below look better than the product ever will.
    """
    meta = np.load(f"{shard}/meta.npy", mmap_mode="r")
    mv = np.memmap(f"{shard}/moves.u16", dtype=np.uint16, mode="r")
    ck = np.memmap(f"{shard}/clocks.u16", dtype=np.uint16, mode="r")
    sn = open(f"{shard}/players.txt", encoding="utf-8").read().split("\n")
    idx = {n: i for i, n in enumerate(table.names)}

    off = np.asarray(meta["offset"], np.int64)
    npl = np.asarray(meta["nply"], np.int64)
    ok = np.asarray(ck[off]) != 0xFFFF
    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
    keep = np.concatenate([ok & (npl >= 16)] * 2)
    pid, gid, seat = pid[keep], gid[keep], seat[keep]
    o = np.argsort(pid, kind="stable")
    pid, gid, seat = pid[o], gid[o], seat[o]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])

    z = np.load("play/verifier_pack.npz", allow_pickle=True)
    pk_moves, pk_nply = np.asarray(z["moves"]), np.asarray(z["nply"])

    rng = np.random.default_rng(seed)
    out, skipped = [], 0
    for i in rng.permutation(len(bnd) - 1):
        if len(out) >= n_players:
            break
        s = slice(bnd[i], bnd[i + 1])
        p = int(pid[s][0])
        if p >= len(sn):
            continue
        row = idx.get(sn[p].lower())
        if row is None:
            continue
        g_, s_ = gid[s], seat[s]
        picks, feats, opens = [], [], []
        for x in rng.permutation(len(g_)):
            if len(feats) >= k_games:
                break
            gi, st = int(g_[x]), int(s_[x])
            o_, n_ = int(off[gi]), min(int(npl[gi]), plies)
            codes = np.asarray(mv[o_:o_ + n_])
            dup = False
            for j in range(pk_moves.shape[1]):
                m = min(int(pk_nply[row, j]), n_)
                if m == n_ and m > 0 and np.array_equal(pk_moves[row, j, :m], codes):
                    dup = True
                    break
            if dup:
                continue
            f = game_features(codes, np.asarray(ck[o_:o_ + n_]), st,
                              int(meta[gi]["tc_base"]), int(meta[gi]["tc_inc"]))
            if f is None:
                continue
            feats.append(f); opens.append(opening_hash(codes, st))
            picks.append((gi, st))
        if len(feats) == k_games:
            # `offset` skips ACCEPTED players, so fit and eval draw disjoint
            # sets from the same shuffle rather than two overlapping samples.
            if skipped < offset:
                skipped += 1
                continue
            out.append((row, np.stack(feats).mean(0),
                        [o for o in opens if o], picks))
    return out


def cmd_fit(args):
    t = Table(args.table)
    qs = shard_queries(args.shard, t, args.players, args.games, args.seed)
    print(f"{len(qs):,} leakage-free query sets", flush=True)
    rng = np.random.default_rng(args.seed)
    X, y = [], []
    for row, qf, qo, _ in qs:
        neg = rng.integers(0, len(t.F), args.negatives)
        neg = neg[neg != row]
        rows = np.r_[row, neg]
        d = t.design(qf, qo, rows)
        X.append(d[:1]); y.append(np.ones(1, np.float32))
        # Balance to one negative per positive so the fitted logit is an LLR,
        # not an LLR plus the shortlist's base rate.
        pick = rng.integers(0, len(neg))
        X.append(d[1 + pick:2 + pick]); y.append(np.zeros(1, np.float32))
    X = np.concatenate(X); y = np.concatenate(y)
    w, b = fit_logistic_mv(X, y)
    print(f"\nfit on {len(y):,} balanced pairs")
    print(f"{'term':<18} {'weight':>8}")
    print("-" * 28)
    lbl = NAMES + ["opening_overlap", "no_profile"]
    for j in np.argsort(-np.abs(w)):
        print(f"{lbl[j]:<18} {w[j]:8.3f}")
    p = 1 / (1 + np.exp(-(X @ w + b)))
    auc = float((p[y > .5][:, None] > p[y < .5][None, :]).mean())
    print(f"\npairwise AUC (train) {auc:.4f}   bias {b:.3f}")
    json.dump({"w": w.tolist(), "b": b, "mu": t.mu.tolist(),
               "sd": t.sd.tolist(), "labels": lbl, "auc_train": auc},
              open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    print("FIT_DONE")


def cmd_eval(args):
    """The only question that matters: does this move top-10 on the real gallery?

    ICC says how much signal a feature has in isolation. It says nothing about
    whether the embedding already has it. Everything here is measured against
    the full 558,735-player gallery, with cosine as the baseline.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "play"))
    from gallery_ctx import Bundles, embed_bundles
    from model import MultiTaskModel, Config, N_ELO_BINS
    from timefeat import N_TIME_BINS

    t = Table(args.table)
    cf = json.load(open(args.coef))
    w, b = np.asarray(cf["w"], np.float32), float(cf["b"])

    g = np.load(args.gallery, allow_pickle=True)
    gnames = [str(x).lower() for x in g["names"]]
    if gnames != t.names:
        raise SystemExit("gallery and feature table are not row-aligned")
    C = torch.tensor(np.asarray(g["centroids"], np.float32))
    C = F.normalize(C, dim=-1)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"]); slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len); wr = ck["n_planes"] == 13
    m = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                       d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                       n_elo_bins=N_ELO_BINS, n_game_slots=slots,
                       elo_cond=bool(ck.get("elo_cond")))
    m.load_state_dict(ck["model"]); m.eval()

    qs = shard_queries(args.shard, t, args.players, args.games, args.seed,
                       offset=args.offset)
    print(f"{len(qs):,} eval queries (offset {args.offset}) | "
          f"gallery {len(t.names):,}", flush=True)

    bundles, owner = [], []
    for qi, (_, _, _, picks) in enumerate(qs):
        for c in range(0, max(len(picks) - slots + 1, 1), slots):
            sel = picks[c:c + slots]
            if sel:
                bundles.append(sel); owner.append(qi)
    E = F.normalize(embed_bundles(m, Bundles(args.shard, bundles, mlpg, wr),
                                  slots, "cpu", 64, 0, "eval").float(), dim=-1)
    S = torch.zeros(len(qs), C.shape[0])
    S.index_add_(0, torch.tensor(owner), E @ C.T)

    # Collect per-query scores first; the combination weight is chosen on one
    # half and reported on the other, so the headline is never the weight's
    # own training score.
    Z, L, TR = [], [], []
    for qi, (row, qf, qo, _) in enumerate(qs):
        sim = S[qi].numpy()
        Z.append((sim - sim.mean()) / (sim.std() + 1e-9))
        L.append(t.design(qf, qo) @ w + b)
        TR.append(row)
        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{len(qs)}", flush=True)
    TR = np.asarray(TR)

    def rk(scores, idx):
        return np.array([int((scores[i] > scores[i][TR[i]]).sum()) + 1
                         for i in idx])

    n = len(qs); half = n // 2
    dev, test = np.arange(half), np.arange(half, n)
    grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
    best_l, best_v = 0.0, -1.0
    print(f"\nweight sweep on {len(dev)} dev queries (r@10):")
    for lam in grid:
        v = (rk([Z[i] + lam * L[i] for i in range(n)], dev) <= 10).mean()
        print(f"  lambda {lam:5.2f}  r@10 {v:.4f}")
        if v > best_v:
            best_v, best_l = v, lam
    print(f"  -> lambda {best_l:.2f}")

    print(f"\n{len(test)} held-out queries | gallery {len(t.names):,}")
    print(f"{'scorer':<16} {'r@1':>7} {'r@10':>7} {'r@100':>7} {'r@1000':>8} "
          f"{'median':>8}")
    print("-" * 58)
    rows = (("cosine", [Z[i] for i in range(n)]),
            ("features", [L[i] for i in range(n)]),
            (f"cos+{best_l:.2f}*feat", [Z[i] + best_l * L[i] for i in range(n)]))
    keep = {}
    for name, sc in rows:
        r = rk(sc, test); keep[name] = r
        print(f"{name:<16} {(r<=1).mean():7.4f} {(r<=10).mean():7.4f} "
              f"{(r<=100).mean():7.4f} {(r<=1000).mean():8.4f} "
              f"{np.median(r):8.0f}")
    rc = keep["cosine"]; rb = keep[f"cos+{best_l:.2f}*feat"]
    d = (rb <= 10).mean() - (rc <= 10).mean()
    disc = ((rb <= 10) ^ (rc <= 10)).mean()
    se = np.sqrt(max(disc, 1e-9) / len(rc))
    print(f"\ntop-10 delta {d:+.4f}  (+-{1.96*se:.4f} at 95%)")
    print(f"true-player feature LLR above median: "
          f"{np.mean([L[i][TR[i]] - np.median(L[i]) for i in range(n)]):+.2f} nats")
    # Openings alone, since that is the one term with a large weight.
    ov = np.array([t.overlap(qs[i][2])[TR[i]] for i in range(n)])
    print(f"true player shares an opening with the query in "
          f"{(ov > 0).mean():.1%} of queries (mean {ov.mean():.2f} of "
          f"{len(qs[0][2])})")
    print("EVAL_DONE")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--table", default="play/handfeat_pack.npz")
    f.add_argument("--shard", default="data/2026-06-big")
    f.add_argument("--players", type=int, default=3000)
    f.add_argument("--games", type=int, default=5)
    f.add_argument("--negatives", type=int, default=64)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", default="play/feat_llr.json")
    f.set_defaults(fn=cmd_fit)
    e = sub.add_parser("eval")
    e.add_argument("--table", default="play/handfeat_pack.npz")
    e.add_argument("--coef", default="play/feat_llr.json")
    e.add_argument("--gallery", default="play/gallery_2026.npz")
    e.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")  # the gallery's model
    e.add_argument("--shard", default="data/2026-06-big")
    e.add_argument("--players", type=int, default=300)
    e.add_argument("--games", type=int, default=5)
    e.add_argument("--offset", type=int, default=3000)
    e.add_argument("--cos-scale", type=float, default=1.0)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(fn=cmd_eval)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
