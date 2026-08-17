"""End to end: does the verifier move the right player up a real shortlist?

AUC on sampled pairs is not the product question. This builds an actual cosine
shortlist, scores every candidate's games with the verifier, combines the two
signals, and reports where the true player lands.

Three things get measured that the training AUC cannot tell us:

  aggregate AUC   per-game AUC compounds over a candidate's games only if those
                  scores are independent. They are not -- same player, same
                  repertoire -- so the sqrt(N) arithmetic is an upper bound and
                  this measures the real thing.
  correlation     how much each extra game of a candidate actually adds.
  P(top 10)       cosine alone vs cosine + verifier, on the same shortlists.

The gallery here is built locally and is far smaller than the deployed 558,735,
so absolute recall is not comparable to production numbers. The COMPARISON
between the two rankers on identical shortlists is the point.

    python verify_eval.py --verifier ckpt/final/verifier_partial.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gallery_ctx import Bundles, Collate, embed_bundles
from model import MultiTaskModel, Config, N_ELO_BINS
from timefeat import N_TIME_BINS
from verify import Verifier, player_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="ckpt/final/verifier_partial.pt")
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--gallery-players", type=int, default=3000)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--shortlist", type=int, default=100)
    ap.add_argument("--cand-games", type=int, default=4)
    ap.add_argument("--gallery-games", type=int, default=12)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"]); slots = ck.get("n_game_slots", 1)
    mlpg = ck.get("max_len_per_game", cfg.max_len); wr = ck["n_planes"] == 13
    emb_model = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                               d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                               n_elo_bins=N_ELO_BINS, n_game_slots=slots,
                               elo_cond=bool(ck.get("elo_cond"))).to(device)
    emb_model.load_state_dict(ck["model"]); emb_model.eval()

    vk = torch.load(args.verifier, map_location="cpu", weights_only=False)
    vtrunk = MultiTaskModel(Config(**vk["cfg"]), n_planes=vk["n_planes"],
                            n_extra=vk["n_extra"], d_embed=vk["d_embed"],
                            n_time_bins=N_TIME_BINS, n_elo_bins=N_ELO_BINS,
                            n_game_slots=vk["n_game_slots"],
                            elo_cond=bool(vk.get("elo_cond")))
    ver = Verifier(vtrunk, Config(**vk["cfg"]).d_model, vk["k"] - 1).to(device)
    ver.load_state_dict(vk["model"]); ver.eval()
    K = vk["k"]
    print(f"verifier step {vk['step']} | k={K}", flush=True)

    players = [p for p in player_index(args.shard) if len(p[0]) >= K + args.cand_games]
    rng = np.random.default_rng(args.seed)
    if len(players) > args.gallery_players:
        players = [players[i] for i in rng.choice(len(players), args.gallery_players,
                                                  replace=False)]
    P = len(players)
    print(f"{P:,} gallery players", flush=True)

    # Reserve the first K-1 games of each player as their possible query, the
    # next `cand_games` as the games a verifier may inspect, and build the
    # centroid from what is left. Nothing is shared between the three roles.
    perms = [rng.permutation(len(g)) for g, _ in players]
    gal_bundles, owner = [], []
    for i, (g, s) in enumerate(players):
        rest = perms[i][K - 1 + args.cand_games:][:args.gallery_games]
        ch = [rest[j:j + slots] for j in range(0, len(rest) - slots + 1, slots)] or [rest[:slots]]
        for c in ch:
            gal_bundles.append([(int(g[j]), int(s[j])) for j in c]); owner.append(i)
    GE = embed_bundles(emb_model, Bundles(args.shard, gal_bundles, mlpg, wr),
                       slots, device, args.batch, args.workers, "gallery")
    C = torch.zeros(P, GE.shape[1]); C.index_add_(0, torch.tensor(owner), GE)
    C = F.normalize(C, dim=-1)

    qsel = rng.choice(P, min(args.queries, P), replace=False)
    qb = [[(int(players[i][0][j]), int(players[i][1][j]))
           for j in perms[i][:K - 1]] for i in qsel]
    QE = F.normalize(embed_bundles(emb_model, Bundles(args.shard, qb, mlpg, wr),
                                   slots, device, args.batch, args.workers, "query"), dim=-1)

    coll = Collate(ck["n_planes"], slots, wr)
    base_ranks, comb_ranks, ver_ranks = [], [], []
    within_corr, agg_pos, agg_neg = [], [], []

    for qi, gi in enumerate(qsel):
        sims = QE[qi] @ C.T
        top = torch.topk(sims, min(args.shortlist, P))
        cand = top.indices.tolist()
        if gi not in cand:
            base_ranks.append(int((sims > sims[gi]).sum()) + 1)
            comb_ranks.append(base_ranks[-1]); ver_ranks.append(base_ranks[-1])
            continue

        # One bundle per (candidate, one of their games): the query occupies
        # slots 0..K-2 and the candidate's game goes last.
        bundles, who = [], []
        for c in cand:
            g2, s2 = players[c]
            for j in perms[c][K - 1:K - 1 + args.cand_games]:
                bundles.append(qb[qi] + [(int(g2[j]), int(s2[j]))])
                who.append(c)
        dl = DataLoader(Bundles(args.shard, bundles, mlpg, wr), batch_size=args.batch,
                        shuffle=False, num_workers=args.workers, collate_fn=coll)
        out = []
        with torch.no_grad():
            for planes, extra, pad, mine, slot, ppos in dl:
                lo = ver(planes.to(device), extra.to(device), pad.to(device),
                         slot.to(device), ppos.to(device))
                out.append(lo.float().cpu())
        sc = torch.cat(out).numpy()
        who = np.array(who)

        per = {}
        for c in cand:
            v = sc[who == c]
            per[c] = float(np.mean(v))
            if c == gi and len(v) > 1:
                # Correlation between a player's own game scores: how much each
                # extra game really adds beyond the first.
                within_corr.append(float(np.corrcoef(v[:-1], v[1:])[0, 1])
                                   if len(v) > 2 else np.nan)
        agg_pos.append(per[gi])
        agg_neg.extend([per[c] for c in cand if c != gi])

        cs = sims[cand].numpy()
        vs = np.array([per[c] for c in cand])
        # Standardise each signal inside the shortlist before adding: cosine and
        # a logit are on different scales and summing raw values would silently
        # weight one of them to nothing.
        zc = (cs - cs.mean()) / (cs.std() + 1e-9)
        zv = (vs - vs.mean()) / (vs.std() + 1e-9)
        pos = cand.index(gi)
        base_ranks.append(int((cs > cs[pos]).sum()) + 1)
        ver_ranks.append(int((vs > vs[pos]).sum()) + 1)
        comb_ranks.append(int(((zc + zv) > (zc[pos] + zv[pos])).sum()) + 1)
        if (qi + 1) % 10 == 0:
            print(f"  {qi+1}/{len(qsel)} queries", flush=True)

    b = np.array(base_ranks); v = np.array(ver_ranks); c2 = np.array(comb_ranks)
    pos = np.array(agg_pos); neg = np.array(agg_neg)
    auc = float((pos[:, None] > neg[None, :]).mean())
    wc = np.array([x for x in within_corr if x == x])
    print(f"\n{len(b)} queries | shortlist {args.shortlist} of {P:,} | "
          f"{args.cand_games} games inspected per candidate\n")
    print(f"  aggregate AUC (true player vs shortlist impostors): {auc:.4f}")
    if len(wc):
        print(f"  within-player score correlation: {wc.mean():.3f} "
              f"(1.0 = extra games add nothing)")
    print(f"\n{'ranker':<22}{'r@1':>8}{'r@10':>8}{'median':>9}")
    for name, r in (("cosine only", b), ("verifier only", v), ("cosine + verifier", c2)):
        print(f"  {name:<20}{(r<=1).mean():>8.3f}{(r<=10).mean():>8.3f}{np.median(r):>9.0f}")
    print("\nVERIFY_EVAL_DONE")


if __name__ == "__main__":
    main()
