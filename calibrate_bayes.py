"""Turn cosine and the verifier into evidence that can legitimately be added.

The demo currently combines them by z-scoring each inside the shortlist and
summing. That is a heuristic: it gives both signals equal weight by construction,
throws away how confident each one is, and cannot express "this candidate has
twelve games agreeing" versus "this one has three".

The Bayesian version is:

    logit P(candidate is you | everything)
        = logit P(you | cosine)                 <- prior, one per candidate
        + SUM_i log [ P(s_i | same) / P(s_i | diff) ]   <- one term per game

and the trick that makes it cheap is Platt scaling. Fit a logistic on a BALANCED
set of same/different pairs and the fitted `a*s + b` IS the log-likelihood ratio,
because logit P(same | s) = LLR + logit(prior) and a balanced prior contributes
logit(0.5) = 0.

Two properties fall out that the z-score sum cannot give:

  evidence accumulates   a candidate with 12 games contributes 12 LLR terms.
                         Their per-game scores are near-independent (measured
                         within-player correlation -0.27), so this is real
                         accumulation, not double counting.
  scores stay comparable across queries, because they are absolute
                         probabilities rather than a rank within one shortlist.

    python calibrate_bayes.py --out play/bayes_calib.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from gallery_ctx import Bundles, embed_bundles
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import N_TIME_BINS
from verify import player_index


def fit_logistic(x, y, iters=400, lr=0.1):
    """1-D logistic by gradient descent; returns (a, b) for logit p = a*x + b."""
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    a = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(iters):
        loss = F.binary_cross_entropy_with_logits(a * x + b, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(a.item()), float(b.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--verifier", default="ckpt/final/verifier2_best.pt")
    ap.add_argument("--pack", default="play/verifier_pack.npz")
    ap.add_argument("--gallery", default="play/gallery_2026.npz")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--out", default="play/bayes_calib.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cpu"

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "play"))
    import server

    server.load(args.ckpt, args.ckpt, args.gallery)
    server.load_verifier(args.verifier, args.pack)
    if "ver" not in server.MODEL:
        raise SystemExit("verifier not loaded")
    C = server.MODEL["cent"]
    names = [str(n).lower() for n in server.MODEL["names"]]
    idx = {n: i for i, n in enumerate(names)}

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"]); slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len); wr = ck["n_planes"] == 13
    emb = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                         d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                         n_elo_bins=N_ELO_BINS, n_game_slots=slots,
                         elo_cond=bool(ck.get("elo_cond")))
    emb.load_state_dict(ck["model"]); emb.eval()

    meta = np.load(f"{args.shard}/meta.npy", mmap_mode="r")
    clocks = np.memmap(f"{args.shard}/clocks.u16", dtype=np.uint16, mode="r")
    sn = open(f"{args.shard}/players.txt", encoding="utf-8").read().split("\n")
    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
    ok = np.concatenate([np.asarray(clocks[np.asarray(meta["offset"], np.int64)]) != 0xFFFF] * 2)
    o = np.argsort(pid, kind="stable")
    pid, gid, seat, ok = pid[o], gid[o], seat[o], ok[o]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])

    moves_mm = np.memmap(f"{args.shard}/moves.u16", dtype=np.uint16, mode="r")
    npl_ = ck["n_planes"]

    def shard_blocks(picks):
        """Encode shard games the way identify() encodes the visitor's games."""
        import chess
        from bitboards import decode_move, board_to_planes8
        from timefeat import time_features, N_TIME_FEATS
        out = []
        for gi, st in picks:
            row = meta[gi]; o, n = int(row["offset"]), int(row["nply"])
            codes = np.asarray(moves_mm[o:o + n]); clk = np.asarray(clocks[o:o + n])
            T = min(len(codes), mlpg)
            pov = chess.WHITE if st == 0 else chess.BLACK
            pl = np.zeros((T, npl_, 8, 8), np.uint8); b = chess.Board()
            for t in range(T):
                board_to_planes8(b, pov, pl[t], wr)
                b.push(decode_move(int(codes[t])))
            fe, _, _ = time_features(clk, int(row["tc_base"]), int(row["tc_inc"]))
            mt = np.zeros(T, bool); mt[st::2] = True
            out.append((pl, fe[:T], mt))
        return out

    rng = np.random.default_rng(args.seed)
    order = list(range(len(bnd) - 1)); rng.shuffle(order)
    qsets, truth = [], []
    for i in order:
        if len(qsets) >= args.queries:
            break
        sl = slice(bnd[i], bnd[i + 1]); m = ok[sl]
        g_, s_ = gid[sl][m], seat[sl][m]
        if len(g_) < args.games:
            continue
        p = int(pid[sl][0])
        if p >= len(sn):
            continue
        j = idx.get(sn[p].lower())
        if j is None:
            continue
        sel = rng.permutation(len(g_))[:args.games]
        qsets.append([(int(g_[x]), int(s_[x])) for x in sel]); truth.append(j)
    print(f"{len(qsets)} calibration queries", flush=True)

    # Cosine side: embed each query as bundles, sum similarities (deployed path).
    bundles, owner = [], []
    for qi, gs in enumerate(qsets):
        for c in range(0, len(gs) - slots + 1, slots):
            bundles.append(gs[c:c + slots]); owner.append(qi)
    E = F.normalize(embed_bundles(emb, Bundles(args.shard, bundles, mlpg, wr),
                                  slots, device, 64, 0, "cal").float(), dim=-1)
    S = torch.zeros(len(qsets), C.shape[0])
    S.index_add_(0, torch.tensor(owner), E @ C.T)

    # Collect once, then fit and evaluate on DISJOINT halves. Fitting and
    # scoring the same queries would report the calibration's training fit,
    # which is exactly the number that cannot tell us whether to deploy this.
    K = server.MODEL["ver_k"]
    recs = []
    for qi in range(len(qsets)):
        sim = S[qi]
        rows = torch.topk(sim, args.depth).indices.tolist()
        per = server.verifier_scores(shard_blocks(qsets[qi][-(K - 1):]), rows,
                                     per_game=True)
        recs.append({"rows": rows, "cs": sim[rows].numpy().astype(np.float64),
                     "per": per, "truth": truth[qi]})
        if (qi + 1) % 25 == 0:
            print(f"  {qi+1}/{len(qsets)}", flush=True)

    cut = len(recs) // 2
    fit, ev = recs[:cut], recs[cut:]
    print(f"\nfit on {len(fit)} queries, evaluate on {len(ev)} held out\n")

    def zscore(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    cos_x, cos_y, ver_x, ver_y, ngames = [], [], [], [], []
    for r in fit:
        z = zscore(r["cs"])
        for i, row in enumerate(r["rows"]):
            cos_x.append(z[i]); cos_y.append(1.0 if row == r["truth"] else 0.0)
        for row, scores in r["per"].items():
            lab = 1.0 if row == r["truth"] else 0.0
            ver_x.extend(scores); ver_y.extend([lab] * len(scores))
            if lab:
                ngames.append(len(scores))

    out = {"depth": args.depth, "games": args.games,
           "n_cos": len(cos_x), "n_ver": len(ver_x)}
    a, b = fit_logistic(np.array(cos_x), np.array(cos_y))
    out["cos_a"], out["cos_b"] = a, b
    print(f"  cosine prior : logit p = {a:.4f}*z {b:+.4f}   (n={len(cos_x):,})")

    if not ver_x:
        raise SystemExit("no verifier scores -- nothing to calibrate")
    # BALANCE before fitting: the shortlist has 1 positive per ~100 negatives,
    # and an unbalanced fit would bake that prior into b, which then
    # double-counts against the cosine prior.
    vx, vy = np.array(ver_x), np.array(ver_y)
    pos, neg = np.flatnonzero(vy > 0.5), np.flatnonzero(vy < 0.5)
    take = min(len(pos), len(neg))
    sub = np.r_[rng.choice(pos, take, replace=False),
                rng.choice(neg, take, replace=False)]
    c, d = fit_logistic(vx[sub], vy[sub])
    out["ver_a"], out["ver_b"] = c, d
    out["n_ver_balanced"] = int(2 * take)
    print(f"  verifier LLR : llr = {c:.4f}*s {d:+.4f}   (balanced n={2*take:,})")
    print(f"  games per true candidate: mean {np.mean(ngames):.1f}")

    # Two things decide whether per-game accumulation is legitimate here.
    #
    # AUC says whether a single game carries signal at all. The game-count
    # spread says whether summing is FAIR: each term contributes the intercept
    # `d` as well as the slope, so a candidate with 4 packed games and one with
    # 1 are shifted by different constants for a reason that has nothing to do
    # with whether they are the visitor. If the counts are near-uniform that
    # bias cancels; if they are spread, summing quietly ranks by pack coverage.
    o_ = np.argsort(vx); r_ = np.empty(len(vx)); r_[o_] = np.arange(len(vx)) + 1
    npos, nneg = len(pos), len(neg)
    auc = (r_[vy > 0.5].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    cnt = np.array([len(v) for r in fit for v in r["per"].values()])
    print(f"  per-game AUC : {auc:.4f}  ({npos:,} pos / {nneg:,} neg)")
    print(f"  games per candidate: mean {cnt.mean():.2f}, "
          f"{(cnt == cnt.max()).mean():.1%} at the max of {cnt.max()}")
    out["per_game_auc"] = float(auc)

    # --- held-out comparison against the deployed heuristic -------------------
    MISS = args.depth + 1        # truth outside the shortlist; no rerank can fix

    def ranks(score_fn):
        rk = []
        for r in ev:
            if r["truth"] not in r["rows"]:
                rk.append(MISS); continue
            sc = score_fn(r)
            t = r["rows"].index(r["truth"])
            rk.append(int((sc > sc[t]).sum()) + 1)
        return np.array(rk)

    def cosine_only(r):
        return r["cs"]

    def zsum(r):
        """The deployed heuristic: z-score each signal, add."""
        vv = np.array([np.mean(r["per"][x]) if r["per"].get(x) else np.nan
                       for x in r["rows"]])
        seen = ~np.isnan(vv)
        zv = np.zeros_like(vv)
        if seen.sum() >= 2:
            zv[seen] = zscore(vv[seen])
        return zscore(r["cs"]) + zv

    def bayes(r):
        z = zscore(r["cs"])
        llr = np.array([sum(c * s + d for s in r["per"].get(x, ()))
                        for x in r["rows"]])
        return a * z + b + llr

    def bayes_mean(r):
        """One LLR term per CANDIDATE, not per game -- isolates whether the
        accumulation across games is what helps, or just the calibration."""
        z = zscore(r["cs"])
        llr = np.array([c * np.mean(r["per"][x]) + d if r["per"].get(x) else 0.0
                        for x in r["rows"]])
        return a * z + b + llr

    print(f"\n{'method':<28}{'r@1':>8}{'r@10':>8}{'median':>9}")
    res, allrk = {}, {}
    for nm, fn in (("cosine only", cosine_only), ("z-score sum (deployed)", zsum),
                   ("bayes (per-game LLR)", bayes), ("bayes (per-candidate)", bayes_mean)):
        rk = ranks(fn)
        allrk[nm] = rk
        res[nm] = {"r1": float((rk <= 1).mean()), "r10": float((rk <= 10).mean()),
                   "median": float(np.median(rk))}
        print(f"  {nm:<26}{(rk<=1).mean():>8.3f}{(rk<=10).mean():>8.3f}"
              f"{np.median(rk):>9.0f}")

    # PAIRED comparison against cosine. The same queries run through both
    # methods, so the only queries carrying information are the ones where they
    # DISAGREE -- comparing two proportions as if they were independent samples
    # throws that pairing away and badly overstates the noise floor. McNemar on
    # the discordant pairs is the right test, and it is what stops a two-query
    # difference from being read as a win.
    base = allrk["cosine only"]
    print(f"\n  paired vs cosine (n={len(base)}), top-10:")
    for nm, rk in allrk.items():
        if nm == "cosine only":
            continue
        win = int(((rk <= 10) & (base > 10)).sum())
        lose = int(((rk > 10) & (base <= 10)).sum())
        n = win + lose
        # Two-sided exact binomial on the discordant pairs, p=0.5 under the null.
        from math import comb
        p = 1.0 if n == 0 else min(1.0, 2 * sum(
            comb(n, i) for i in range(min(win, lose) + 1)) / 2 ** n)
        res[nm]["win"], res[nm]["lose"], res[nm]["p"] = win, lose, float(p)
        print(f"    {nm:<26} +{win} -{lose}   p={p:.3f}"
              f"{'' if p < 0.05 else '   (not significant)'}")
    print(f"\n  ({(ranks(cosine_only) < MISS).mean():.3f} of queries have the true "
          f"player inside the depth-{args.depth} shortlist at all -- that is the "
          f"ceiling for every rerank above)")
    out["heldout"] = res
    out["n_eval"] = len(ev)

    # The server reads this rather than being told which fusion to use, so the
    # deployed behaviour follows the held-out measurement instead of whichever
    # method we happened to build last. r@10 is the product metric; median rank
    # breaks ties because it moves even when r@10 is saturated.
    # "none" is a real option and has to be on the ballot. A second stage that
    # reorders the shortlist can only help if it is better than the cosine
    # ordering it overwrites -- and a verifier with weak discrimination on THIS
    # distribution actively destroys a good ranking. Leaving it off the list
    # would guarantee we deploy a reranker whether or not it earns its place.
    key = {"cosine only": "none", "z-score sum (deployed)": "zsum",
           "bayes (per-game LLR)": "bayes", "bayes (per-candidate)": "bayes_mean"}
    best = max(key, key=lambda n: (res[n]["r10"], -res[n]["median"]))
    # A rerank has to EARN the swap, not merely win the sample. Ranking by r@10
    # alone will happily promote a two-query difference on 150 queries, which is
    # well inside the noise -- and swapping the product's ranking on that is how
    # you ship a regression that no later measurement can explain. Cosine keeps
    # the slot unless the paired test says the difference is real.
    if best != "cosine only" and res[best].get("p", 1.0) >= 0.05:
        print(f"\n  {key[best]} led on r@10 but the paired test says p="
              f"{res[best]['p']:.3f} -- inside the noise. Keeping cosine.")
        best = "cosine only"
    out["recommended"] = key[best]
    print(f"\n  recommended fusion: {key[best]}  ({best})")
    if key[best] == "none":
        print("  -> NO rerank beat plain cosine by more than noise. The verifier "
              "does not earn its place; the server will skip the second stage.")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("CALIBRATE_DONE")


if __name__ == "__main__":
    main()
