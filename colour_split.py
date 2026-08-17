"""Does a separate white/black centroid identify better than one combined centroid?

Two effects pull against each other and only measurement settles it:

  sharper  a player's white and black repertoires are different objects, and one
           centroid averages them into something that is neither
  noisier   splitting halves the games behind each centroid, and richer centroids
           measurably help (+11% top-10 at k=5 going 12 -> 64 games)

There is also a free side effect: the model has 5 game slots, so a combined query
caps at 5 games. Colour-split queries are two bundles of up to 5, which is 10
games of evidence from a 5-slot model.

Combination is by SCORE FUSION, not by intersecting top-N lists. Intersection is
a hard AND on list membership: a player ranked 1st on white and 101st on black
drops out entirely, and the ranking *within* each list -- which is most of the
information -- is discarded. Summing the two cosines keeps both.

    python colour_split.py --ckpt ckpt/final/ctx5_ft2.pt --shard data/2026-06-big
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import chess

from bitboards import decode_move, board_to_planes8
from timefeat import time_features, ms_used_per_ply, N_TIME_FEATS, N_TIME_BINS
from model import MultiTaskModel, Config, N_ELO_BINS


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    m = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                       d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                       n_elo_bins=N_ELO_BINS, n_game_slots=ck.get("n_game_slots", 1))
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--players", type=int, default=1500)
    ap.add_argument("--queries", type=int, default=250)
    ap.add_argument("--per-colour", type=int, default=5, help="query games per colour")
    ap.add_argument("--cap", type=int, default=40, help="centroid games cap, per colour")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m, ck, cfg = load_model(args.ckpt)
    slots = ck.get("n_game_slots", 5)
    mlpg = ck.get("max_len_per_game", cfg.max_len)
    npl, wr = ck["n_planes"], ck["n_planes"] == 13

    meta = np.load(f"{args.shard}/meta.npy", mmap_mode="r")
    moves = np.memmap(f"{args.shard}/moves.u16", dtype=np.uint16, mode="r")
    clocks = np.memmap(f"{args.shard}/clocks.u16", dtype=np.uint16, mode="r")

    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8), np.ones(len(meta), np.int8)])
    ok = np.concatenate([np.asarray(clocks[np.asarray(meta["offset"], np.int64)]) != 0xFFFF] * 2)
    o = np.argsort(pid, kind="stable")
    pid, gid, seat, ok = pid[o], gid[o], seat[o], ok[o]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])

    need = args.per_colour + 5           # per colour: query games + a centroid chunk
    players = []
    for i in range(len(bnd) - 1):
        sl = slice(bnd[i], bnd[i + 1]); mask = ok[sl]
        g, s = gid[sl][mask], seat[sl][mask]
        w, b = g[s == 0], g[s == 1]
        if len(w) >= need and len(b) >= need:
            players.append((int(pid[sl][0]), w, b))
    print(f"{len(players):,} players with >= {need} clocked games of BOTH colours")
    rng = np.random.default_rng(args.seed)
    if len(players) > args.players:
        players = [players[i] for i in rng.choice(len(players), args.players, replace=False)]
    P = len(players)

    @torch.no_grad()
    def embed(picks):
        """picks: [(game_idx, seat)] -> one joint embedding, with real clocks."""
        blocks = []
        for gi, st in picks:
            row = meta[gi]; o0, nn = int(row["offset"]), int(row["nply"])
            codes = np.asarray(moves[o0:o0 + nn]); clk = np.asarray(clocks[o0:o0 + nn])
            T = min(len(codes), mlpg)
            pov = chess.WHITE if st == 0 else chess.BLACK
            pl = np.zeros((T, npl, 8, 8), np.uint8); bd = chess.Board()
            for t in range(T):
                board_to_planes8(bd, pov, pl[t], wr)
                bd.push(decode_move(int(codes[t])))
            fe, _, _ = time_features(clk, int(row["tc_base"]), int(row["tc_inc"]))
            mt = np.zeros(T, bool); mt[st::2] = True
            blocks.append((pl, fe[:T], mt))
        T = sum(b[0].shape[0] for b in blocks)
        pp = np.zeros((1, T, npl, 8, 8), np.uint8); ee = np.zeros((1, T, N_TIME_FEATS), np.float32)
        mm = np.zeros((1, T), bool); ss = np.zeros((1, T), np.int64); qq = np.zeros((1, T), np.int64)
        at = 0
        for si, (pl, fe, mt) in enumerate(blocks):
            t = pl.shape[0]
            pp[0, at:at + t] = pl; ee[0, at:at + t] = fe; mm[0, at:at + t] = mt
            ss[0, at:at + t] = min(si, slots - 1); qq[0, at:at + t] = np.arange(t); at += t
        e, _ = m.embed(torch.from_numpy(pp), torch.from_numpy(ee),
                       torch.zeros((1, T), dtype=torch.bool), torch.from_numpy(mm),
                       torch.from_numpy(ss), torch.from_numpy(qq))
        return torch.nn.functional.normalize(e.float(), dim=-1)[0]

    def centroid(games, st, cap):
        picks = [(int(x), st) for x in games[:cap]]
        chunks = [picks[i:i + 5] for i in range(0, len(picks) - 4, 5)] or [picks[:5]]
        v = torch.stack([embed(c) for c in chunks]).mean(0)
        return v / v.norm().clamp(min=1e-8)

    print(f"building centroids for {P:,} players (combined / white / black)…")
    Cc, Cw, Cb, held = [], [], [], []
    for i, (p, w, b) in enumerate(players):
        pw, pb = rng.permutation(len(w)), rng.permutation(len(b))
        qw = [int(w[j]) for j in pw[:args.per_colour]]
        qb = [int(b[j]) for j in pb[:args.per_colour]]
        gw = [int(w[j]) for j in pw[args.per_colour:]]
        gb = [int(b[j]) for j in pb[args.per_colour:]]
        Cw.append(centroid(gw, 0, args.cap))
        Cb.append(centroid(gb, 1, args.cap))
        # combined uses the SAME gallery games, just not split by colour
        mixed = [(x, 0) for x in gw[:args.cap]] + [(x, 1) for x in gb[:args.cap]]
        ch = [mixed[j:j + 5] for j in range(0, len(mixed) - 4, 5)] or [mixed[:5]]
        v = torch.stack([embed(c) for c in ch]).mean(0)
        Cc.append(v / v.norm().clamp(min=1e-8))
        held.append((qw, qb))
        if (i + 1) % 200 == 0:
            print(f"  {i+1:,}/{P:,}", flush=True)
    Cc, Cw, Cb = torch.stack(Cc), torch.stack(Cw), torch.stack(Cb)

    nq = min(args.queries, P)
    qidx = rng.choice(P, nq, replace=False)
    n = args.per_colour
    rows = []
    for per in range(1, n + 1):
        res = {k: 0 for k in ("comb", "split", "isect")}
        for count, i in enumerate(qidx):
            qw, qb = held[i]
            # combined: the same TOTAL games, but one bundle capped at 5 slots
            mixed = []
            for j in range(per):
                mixed += [(qw[j], 0), (qb[j], 1)]
            qc = embed(mixed[:5])
            qw_e = embed([(x, 0) for x in qw[:per]])
            qb_e = embed([(x, 1) for x in qb[:per]])
            s_comb = qc @ Cc.T
            s_split = (qw_e @ Cw.T) + (qb_e @ Cb.T)
            res["comb"] += int(i in torch.topk(s_comb, 10).indices.tolist())
            res["split"] += int(i in torch.topk(s_split, 10).indices.tolist())
            tw = set(torch.topk(qw_e @ Cw.T, 100).indices.tolist())
            tb = set(torch.topk(qb_e @ Cb.T, 100).indices.tolist())
            inter = tw & tb
            if inter:
                order = sorted(inter, key=lambda j: -float(s_split[j]))[:10]
                res["isect"] += int(i in order)
        rows.append((per, res["comb"] / nq, res["split"] / nq, res["isect"] / nq))
        print(f"  {2*per} games done", flush=True)

    print(f"\ngallery {P:,} players | {nq} queries | top-10 recall\n")
    print(f"{'games':>6}{'combined':>11}{'colour-split':>14}{'split+isect':>13}")
    for per, c, sp, it in rows:
        print(f"{2*per:>6}{c:>11.3f}{sp:>14.3f}{it:>13.3f}")
    print("\ncombined is capped at 5 slots, so beyond 5 games it cannot use the extra;")
    print("colour-split runs two bundles of up to 5, i.e. up to 10 games of evidence.")


if __name__ == "__main__":
    main()
