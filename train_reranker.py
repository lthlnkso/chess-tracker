"""Fit the shortlist re-ranker on cached features.

Deliberately free of torch. Loading torch and xgboost into one process
segfaults on macOS -- both ship their own OpenMP runtime and the second one
loaded wins in a way neither survives -- so feature extraction (rerank.py build)
and fitting live in separate processes with an npz between them.

    python train_reranker.py --cache play/rerank_data.npz --out play/reranker.json
"""

from __future__ import annotations

import argparse

import numpy as np
import xgboost as xgb

FEATURE_NAMES = ["cos", "gap_to_top", "z", "rank", "cent_games", "elo_gap",
                 "elo_gap_over_sd", "cand_elo", "q_elo", "q_sd", "n_games", "shortlist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="play/rerank_data.npz")
    ap.add_argument("--out", default="play/reranker.json")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--drop", nargs="*", default=[],
                    help="feature names to zero out, for ablation")
    args = ap.parse_args()

    d = np.load(args.cache)
    X, Y, groups = d["X"], d["Y"], d["groups"]
    base_hit = d["base_hit"]
    print(f"{X.shape[0]:,} rows | {len(groups):,} queries | "
          f"answer reachable in the shortlist {base_hit.mean()*100:.1f}%")

    if args.drop:
        for name in args.drop:
            X[:, FEATURE_NAMES.index(name)] = 0.0
        print(f"  ablated: {', '.join(args.drop)}")

    # Split by QUERY. Rows from one shortlist share a query; splitting by row
    # would put near-duplicates on both sides and report a fantasy.
    nq = len(groups)
    cut = int(nq * 0.75)
    starts = np.r_[0, np.cumsum(groups)]
    tr = np.arange(starts[cut])
    te = np.arange(starts[cut], starts[-1])

    dtr = xgb.DMatrix(X[tr], label=Y[tr], feature_names=FEATURE_NAMES)
    dtr.set_group(groups[:cut])
    dte = xgb.DMatrix(X[te], label=Y[te], feature_names=FEATURE_NAMES)
    dte.set_group(groups[cut:])

    params = {"objective": "rank:pairwise", "eta": 0.08, "max_depth": 5,
              "subsample": 0.9, "colsample_bytree": 0.9, "eval_metric": "ndcg@10",
              "min_child_weight": 5, "nthread": 4}
    bst = xgb.train(params, dtr, num_boost_round=args.rounds,
                    evals=[(dte, "test")], verbose_eval=100)

    pred = bst.predict(dte)
    Yte = Y[te]
    at = 0
    b1 = b10 = r1 = r10 = 0
    for n in groups[cut:]:
        blk = slice(at, at + n); at += n
        y = Yte[blk]
        base_pos = int(np.argmax(y))          # shortlist arrives cosine-ordered
        new_pos = int(np.where(np.argsort(-pred[blk]) == base_pos)[0][0])
        b1 += base_pos == 0; b10 += base_pos < 10
        r1 += new_pos == 0;  r10 += new_pos < 10
    n = len(groups[cut:])
    print(f"\nheld-out queries (answer reachable): {n}")
    print(f"  cosine only : r@1 {b1/n:.4f}   r@10 {b10/n:.4f}")
    print(f"  re-ranked   : r@1 {r1/n:.4f}   r@10 {r10/n:.4f}")
    print(f"  delta       : r@1 {(r1-b1)/n:+.4f}  r@10 {(r10-b10)/n:+.4f}")

    imp = bst.get_score(importance_type="gain")
    print("\n  feature gain:")
    for k, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<18}{v:9.1f}")

    bst.save_model(args.out)
    print(f"\nwrote {args.out}")
    print("RERANK_TRAIN_DONE")


if __name__ == "__main__":
    main()
