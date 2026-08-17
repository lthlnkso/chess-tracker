"""Hand-derived per-game features, and the measurement that says whether any of
them are worth shipping.

A feature is only useful for identification if it varies MORE between players
than it does between one player's own games. That ratio is the intraclass
correlation:

    ICC = var_between / (var_between + var_within)

ICC near 1 is a fingerprint. ICC near 0 is noise that happens to have a
different mean for each player because we only measured a few games. The naive
thing -- "knight-move percentage differs from the population, therefore it
identifies you" -- ignores the denominator entirely, and the denominator is
usually what kills these features.

Two more things have to be true before an ICC is believable:

  the estimate must be unbiased    with k games per player, the raw spread of
                                   player means is inflated by within-player
                                   noise. We use the one-way random-effects
                                   estimator, which subtracts it off.
  it must survive Elo              think time and piece preference both track
                                   rating hard. A feature that is really just
                                   a rating estimate adds nothing, because the
                                   trunk already has a rating head.

    python handfeat.py --shard data/2026-06-big --players 4000
"""

from __future__ import annotations

import argparse

import hashlib

import chess
import numpy as np

from bitboards import decode_move

# Order matters: this is the on-disk feature layout.
NAMES = [
    "pawn_frac", "knight_frac", "bishop_frac", "rook_frac", "queen_frac",
    "king_frac", "capture_frac", "check_frac", "castled", "promo_frac",
    "n_moves", "mean_think", "std_think", "fast_frac", "slow_frac",
    "mean_from_rank", "mean_to_rank", "centre_frac",
]
N_FEATS = len(NAMES)

_PIECE_SLOT = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
               chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
_CENTRE = {chess.D4, chess.E4, chess.D5, chess.E5,
           chess.C4, chess.F4, chess.C5, chess.F5}


def game_features(codes, clk, seat, tc_base=60, tc_inc=0):
    """Features for ONE side of one game. `seat` 0 = white, 1 = black.

    Only the player's own moves count. Their opponent's habits are not theirs,
    and mixing the two would blur exactly the signal we are looking for.
    """
    f = np.zeros(N_FEATS, np.float64)
    b = chess.Board()
    own = 0
    caps = checks = promos = 0
    piece = np.zeros(6, np.float64)
    from_rank = to_rank = centre = 0.0
    castled = 0.0
    think = []
    prev_own_clk = float(tc_base)

    for t, c in enumerate(codes):
        mv = decode_move(int(c))
        mine = (t % 2) == seat
        if mine:
            p = b.piece_at(mv.from_square)
            if p is not None:
                piece[_PIECE_SLOT[p.piece_type]] += 1
                if p.piece_type == chess.KING and \
                        abs(chess.square_file(mv.to_square)
                            - chess.square_file(mv.from_square)) > 1:
                    castled = 1.0
            if b.is_capture(mv):
                caps += 1
            if mv.promotion:
                promos += 1
            if b.gives_check(mv):
                checks += 1
            # POV ranks: rank 0 is always the player's own back rank, so white
            # and black are directly comparable.
            fr = chess.square_rank(mv.from_square)
            tr = chess.square_rank(mv.to_square)
            if seat == 1:
                fr, tr = 7 - fr, 7 - tr
            from_rank += fr
            to_rank += tr
            if mv.to_square in _CENTRE:
                centre += 1
            own += 1
            cs = int(clk[t])
            if cs != 0xFFFF:
                dt = prev_own_clk - cs / 100.0 + tc_inc
                if 0.0 <= dt <= float(tc_base):
                    think.append(dt)
                prev_own_clk = cs / 100.0
        b.push(mv)

    if own == 0:
        return None
    f[0:6] = piece / own
    f[6] = caps / own
    f[7] = checks / own
    f[8] = castled
    f[9] = promos / own
    f[10] = own
    if think:
        th = np.asarray(think)
        f[11] = th.mean()
        f[12] = th.std()
        f[13] = float((th < 0.6).mean())     # pre-move / instant reply rate
        f[14] = float((th > 3.0).mean())     # long-think rate
    f[15] = from_rank / own
    f[16] = to_rank / own
    f[17] = centre / own
    return f


def opening_key(codes, seat, plies=6):
    """Identity of the opening as this player steers it."""
    if len(codes) < plies:
        return None
    return (seat, bytes(np.asarray(codes[:plies], np.uint16).tobytes()))


def opening_hash(codes, seat, plies=6):
    """Deterministic 64-bit key. Must NOT use hash(): it is salted per process,
    so workers would disagree and every run would produce a different table."""
    k = opening_key(codes, seat, plies)
    if k is None:
        return 0
    h = hashlib.blake2b(bytes([k[0]]) + k[1], digest_size=8).digest()
    return int.from_bytes(h, "little") | 1        # 0 reserved for "missing"


def icc_oneway(groups):
    """One-way random-effects ICC from unequal group sizes.

    `groups` is a list of (k_i, F) arrays. Returns (icc, var_between,
    var_within). The correction for unequal k is the standard k0.
    """
    ks = np.array([len(g) for g in groups], np.float64)
    N = ks.sum()
    grand = np.concatenate(groups, 0).mean(0)
    means = np.stack([g.mean(0) for g in groups])
    ssb = (ks[:, None] * (means - grand) ** 2).sum(0)
    ssw = np.stack([((g - g.mean(0)) ** 2).sum(0) for g in groups]).sum(0)
    a = len(groups)
    msb = ssb / max(a - 1, 1)
    msw = ssw / max(N - a, 1)
    k0 = (N - (ks ** 2).sum() / N) / max(a - 1, 1)
    vb = np.maximum((msb - msw) / max(k0, 1e-9), 0.0)
    return vb / (vb + msw + 1e-12), vb, msw


def load_player_games(shard, min_games, max_games, max_players, seed=0):
    """Group clocked games by (player, seat-side) across a shard."""
    meta = np.load(f"{shard}/meta.npy", mmap_mode="r")
    moves = np.memmap(f"{shard}/moves.u16", dtype=np.uint16, mode="r")
    clocks = np.memmap(f"{shard}/clocks.u16", dtype=np.uint16, mode="r")
    off = np.asarray(meta["offset"], np.int64)
    npl = np.asarray(meta["nply"], np.int64)
    ok = np.asarray(clocks[off]) != 0xFFFF

    pid = np.concatenate([np.asarray(meta["white_pid"]),
                          np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    seat = np.concatenate([np.zeros(len(meta), np.int8),
                           np.ones(len(meta), np.int8)])
    elo = np.concatenate([np.asarray(meta["white_elo"], np.float64),
                          np.asarray(meta["black_elo"], np.float64)])
    keep = np.concatenate([ok & (npl >= 16)] * 2)
    pid, gid, seat, elo = pid[keep], gid[keep], seat[keep], elo[keep]
    o = np.argsort(pid, kind="stable")
    pid, gid, seat, elo = pid[o], gid[o], seat[o], elo[o]
    bnd = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bnd) - 1)
    out = []
    for i in order:
        if len(out) >= max_players:
            break
        s = slice(bnd[i], bnd[i + 1])
        if bnd[i + 1] - bnd[i] < min_games:
            continue
        g_, s_, e_ = gid[s], seat[s], elo[s]
        sel = rng.permutation(len(g_))[:max_games]
        out.append((int(pid[s][0]), [(int(g_[x]), int(s_[x])) for x in sel],
                    float(np.mean(e_))))
    return out, meta, moves, clocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--players", type=int, default=4000)
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--max-games", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    players, meta, moves, clocks = load_player_games(
        args.shard, args.min_games, args.max_games, args.players, args.seed)
    print(f"{len(players):,} players with >= {args.min_games} clocked games",
          flush=True)

    groups, elos, open_keys = [], [], []
    for n, (p, picks, e) in enumerate(players):
        rows, oks = [], []
        for gi, st in picks:
            row = meta[gi]
            o, k = int(row["offset"]), int(row["nply"])
            c = np.asarray(moves[o:o + k])
            f = game_features(c, np.asarray(clocks[o:o + k]), st,
                              int(row["tc_base"]), int(row["tc_inc"]))
            if f is not None:
                rows.append(f)
                ok_ = opening_key(c, st)
                if ok_ is not None:
                    oks.append(ok_)
        if len(rows) >= args.min_games:
            groups.append(np.stack(rows)); elos.append(e); open_keys.append(oks)
        if (n + 1) % 500 == 0:
            print(f"  {n+1}/{len(players)}", flush=True)

    print(f"\n{len(groups):,} players kept | "
          f"{sum(len(g) for g in groups):,} game-sides\n", flush=True)

    icc, vb, vw = icc_oneway(groups)
    means = np.stack([g.mean(0) for g in groups])
    el = np.asarray(elos)
    print(f"{'feature':<16} {'ICC':>7} {'r(Elo)':>8} {'ICC|Elo':>8}  "
          f"{'pop mean':>9}")
    print("-" * 54)
    order = np.argsort(-icc)
    resid = []
    for j in range(N_FEATS):
        # Residual ICC: regress the player mean on Elo and re-derive how much
        # between-player variance survives. If a feature is only a rating proxy
        # this collapses, and the trunk's Elo head already covers it.
        x = np.c_[el, np.ones(len(el))]
        beta, *_ = np.linalg.lstsq(x, means[:, j], rcond=None)
        r = means[:, j] - x @ beta
        vb_res = max(r.var() - vw[j] / np.mean([len(g) for g in groups]), 0.0)
        resid.append(vb_res / (vb_res + vw[j] + 1e-12))
    resid = np.asarray(resid)
    for j in order:
        rr = np.corrcoef(el, means[:, j])[0, 1]
        print(f"{NAMES[j]:<16} {icc[j]:7.3f} {rr:8.3f} {resid[j]:8.3f}  "
              f"{np.concatenate(groups)[:, j].mean():9.3f}")

    # Opening repertoire: measure the likelihood ratio directly rather than an
    # ICC, because the variable is categorical.
    same = tot_same = 0
    rng = np.random.default_rng(args.seed)
    for oks in open_keys:
        for a in range(len(oks)):
            for b in range(a + 1, len(oks)):
                tot_same += 1
                same += oks[a] == oks[b]
    diff = tot_diff = 0
    flat = [(i, k) for i, oks in enumerate(open_keys) for k in oks]
    for _ in range(200_000):
        i = rng.integers(len(flat)); j = rng.integers(len(flat))
        if flat[i][0] == flat[j][0]:
            continue
        tot_diff += 1
        diff += flat[i][1] == flat[j][1]
    ps = same / max(tot_same, 1); pd = diff / max(tot_diff, 1)
    print(f"\nopening (first 6 plies, same seat)")
    print(f"  P(match | same player)      {ps:.4f}  ({same:,}/{tot_same:,})")
    print(f"  P(match | different player) {pd:.4f}  ({diff:,}/{tot_diff:,})")
    if pd > 0:
        print(f"  likelihood ratio on a match {ps/pd:>8.1f}x  "
              f"= {np.log(ps/pd):.2f} nats of evidence")

    if args.out:
        np.savez_compressed(args.out, icc=icc, icc_resid=resid,
                            var_between=vb, var_within=vw,
                            names=np.array(NAMES), open_same=ps, open_diff=pd)
        print(f"\nwrote {args.out}")
    print("HANDFEAT_DONE")


if __name__ == "__main__":
    main()
