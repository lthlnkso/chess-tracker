"""Run ONE real visitor's games through the reranker and report the placement.

The aggregate probe averages over 134 held-out players whose queries are real
shard games, where cosine is already at median rank 1 and there is little for a
second stage to add. A demo visitor is a harder and more interesting case: the
games are against a bot, so cosine is weaker and the reranker has more room.

Reports the same three rankings, for this visitor:

  cosine rank     where the embedding puts them in the full gallery
  verifier rank   where the VERIFIER ALONE puts them inside the shortlist
  fused rank      where the calibrated combination puts them

    python my_rerank.py --games play/saved/session_12games.json --name YOURNAME
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "play"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="play/saved/session_12games.json")
    ap.add_argument("--name", required=True,
                    help="lichess username to score against")
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--verifier", default="ckpt/final/verifier2_sat.pt")
    ap.add_argument("--pack", default="play/verifier_pack.npz")
    ap.add_argument("--gallery", default="play/gallery_2026.npz")
    ap.add_argument("--calib", default="play/bayes_calib.json")
    ap.add_argument("--depth", type=int, default=100)
    args = ap.parse_args()

    import server
    server.load(args.ckpt, args.ckpt, args.gallery)
    server.load_verifier(args.verifier, args.pack)

    games = json.load(open(args.games))
    if isinstance(games, dict):
        games = games.get("games", [])
    print(f"\n{len(games)} games from {os.path.basename(args.games)}")

    me = server.MODEL["name2idx"].get(args.name.lower())
    if me is None:
        raise SystemExit(f"{args.name} is not in the gallery")

    # Reuse the server's OWN identify() for the cosine stage rather than
    # reimplementing the encoding -- a subtle mismatch here would produce a
    # confident wrong answer. verify_depth=0 gives the pure cosine ranking.
    base = server.identify(games, topn=args.depth, target=args.name, verify_depth=0)
    cos_rank = base["probe"]["rank"]
    print(f"  cosine rank: {cos_rank:,} of {base['gallery']:,}  "
          f"(top {base['probe']['top_pct']}%)  [{base['mode']}, "
          f"{base['games_used']} games used]")

    if cos_rank > args.depth:
        print(f"\n  {args.name} is NOT inside the top {args.depth}, so no rerank of "
              f"that depth can reach them. Nothing further to measure.")
        return

    # Capture the shortlist and the per-game verifier scores from the real path.
    grab = {}
    real = server.verifier_scores

    def spy(q_blocks, rows, per_game=False):
        out = real(q_blocks, rows, per_game=True)
        grab["rows"], grab["per"] = list(rows), out
        return out

    server.verifier_scores = spy
    server.MODEL.pop("bayes", None)          # force the stage to run
    server.identify(games, topn=10, target=args.name, verify_depth=args.depth)
    server.verifier_scores = real

    rows, per = grab["rows"], grab["per"]
    # rows comes from topk on the cosine scores, so its order IS the cosine order.
    assert rows[0] != me or cos_rank == 1
    ci = rows.index(me)
    vv = np.array([np.mean(per[r]) if per.get(r) else -1e9 for r in rows])
    ver_rank = int((vv > vv[ci]).sum()) + 1

    cal = json.load(open(args.calib))
    # The REAL cosine scores, not a rank-order stand-in: the fusion adds an
    # absolute LLR to a scaled cosine, so substituting ranks would silently
    # change the weight between the two terms and report a fused rank that the
    # server would never produce.
    byname = {c["name"]: c["score"] for c in base["top"]}
    cs = np.array([byname[server.MODEL["names"][r]] for r in rows], dtype=np.float64)
    z = (cs - cs.mean()) / (cs.std() + 1e-9)
    llr = np.array([sum(cal["ver_a"] * s + cal["ver_b"] for s in per.get(r, ()))
                    for r in rows])
    fused = cal["cos_a"] * z + cal["cos_b"] + llr
    fused_rank = int((fused > fused[ci]).sum()) + 1

    print(f"\n  inside the {len(rows)}-candidate shortlist:")
    print(f"    cosine puts you at        #{ci + 1}")
    print(f"    the verifier ALONE at     #{ver_rank}")
    print(f"    the two fused at          #{fused_rank}")
    print(f"    (random would be ~#{len(rows) // 2})")

    mine = per.get(me, [])
    print(f"\n  your per-game verifier logits: "
          f"{', '.join(f'{s:+.2f}' for s in mine)}"
          f"   (mean {np.mean(mine):+.3f})" if mine else
          "\n  you have NO games in the pack -- unscoreable")
    top = np.argsort(-vv)[:5]
    print("  verifier's own top 5 of the shortlist:")
    for r in top:
        nm = server.MODEL["names"][rows[r]]
        print(f"    {vv[r]:+.3f}  {nm}{'   <-- you' if rows[r] == me else ''}")
    print("\nMY_RERANK_DONE")


if __name__ == "__main__":
    main()
