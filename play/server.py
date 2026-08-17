"""Play the pre-trained model in a browser.

The model does not choose moves — it scores candidate *next positions*. So a move
is picked by generating every legal successor, encoding each one, and taking the
model's highest-scoring (or sampling by its probabilities).

Inference reuses the project's own `board_to_planes8` and `SuccessorScorer`
rather than reimplementing the encoding in JavaScript; a subtle mismatch there
would silently produce a weaker opponent and nothing would ever flag it.

    python play/server.py            # then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import http.cookies
import json
import os
import sys
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
import torch
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics                                     # noqa: E402
from model import (MultiTaskModel, Config, N_ELO_BINS, ELO_CENTRES,  # noqa: E402
                   elo_to_bin)
from timefeat import time_features, N_TIME_FEATS, N_TIME_BINS  # noqa: E402
from bitboards import decode_move, board_to_planes8, N_PLANES13  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = {}


def _build(ck):
    cfg = Config(**ck["cfg"])
    m = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                       d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                       n_elo_bins=N_ELO_BINS,
                       n_game_slots=ck.get("n_game_slots", 1),
                       elo_cond=bool(ck.get("elo_cond")))
    m.load_state_dict(ck["model"])
    m.eval()
    return m, cfg


BASE_MS = 60_000.0          # 1+0 bullet, matching the clock the page runs

# Measured top-1 / top-10 recall by number of query games, on the deployed
# 558,735-player gallery (depth_probe.py, n=300, ctx10_ft @ step 320000).
# The panel's percentages come from THIS, not from a softmax invented to look
# confident: the visitor is told what the system actually achieves.
#
# Re-measured 2026-08-17 when the gallery moved to ctx10. This table MUST be
# re-run whenever --id-ckpt or --gallery changes, or the panel quotes one
# model's accuracy while a different one answers.
#
# Against the previous ctx5_ft2 row, on the same 300 players: worse below four
# games, even at five, better at eight and ten (r@1 0.457 -> 0.567 at ten). The
# model gains where its extra context is actually filled and pays for it in the
# short-query regime, which is the expected shape -- bundle size is drawn
# uniform 1..10 in training, so each small-k case gets half the exposure it got
# under ctx5's 1..5. See docs/BRANCHES.md #3.
# 1..10 from depth_probe.py (n=300). 15/20/30 from a separate run (n=120) that
# reproduces the chunk-and-sum this server does past one bundle -- the two
# populations agree where they overlap (k=5: 0.657 vs 0.658, k=10: 0.790 vs
# 0.783), which is why they can share a table.
#
# The curve does NOT flatten at ten games, which is the whole reason the query
# cap is 30 and not TARGET_GAMES: r@10 goes 0.790 -> 0.867 between 10 and 30.
RECALL_BY_GAMES = {
    1: (0.030, 0.110), 2: (0.110, 0.297), 3: (0.253, 0.487),
    5: (0.413, 0.657), 8: (0.513, 0.753), 10: (0.567, 0.790),
    15: (0.692, 0.808), 20: (0.750, 0.842), 30: (0.708, 0.867),
}

# Who is findable at all. The gallery is players with >= 13 clocked 60+0 games
# in data/mt/2026-01..06 (union_gallery.py --min-games 13), so a visitor who
# has never played 1+0 on lichess, or who only started after June 2026, cannot
# be found no matter how well the model works. The UI has to say this: without
# it, "we couldn't find you" reads as a broken model rather than an
# out-of-scope query, and the confidence percentages -- which are measured on
# players who ARE in the gallery -- are quietly meaningless for everyone else.
GALLERY_BLURB = ("Tracking 558,735 lichess players who played 13+ bullet (1+0) "
                 "games between January and June 2026.")


def recall_for(n: int):
    """Interpolate the measured curve, forced monotone.

    r@1 dips at 10 games (0.495 vs 0.510 at 8) which is sampling noise at
    n=200. Showing confidence going DOWN because a visitor played another game
    would be a worse lie than the smoothing.
    """
    ks = sorted(RECALL_BY_GAMES)
    best1 = best10 = 0.0
    out = {}
    for k in ks:
        a, b = RECALL_BY_GAMES[k]
        best1, best10 = max(best1, a), max(best10, b)
        out[k] = (best1, best10)
    n = max(1, int(n))
    if n in out:
        return out[n]
    if n > ks[-1]:
        return out[ks[-1]]
    lo = max([k for k in ks if k <= n], default=ks[0])
    hi = min([k for k in ks if k >= n], default=ks[-1])
    if lo == hi:
        return out[lo]
    t = (n - lo) / (hi - lo)
    return (out[lo][0] + t * (out[hi][0] - out[lo][0]),
            out[lo][1] + t * (out[hi][1] - out[lo][1]))
# How many k-game bundles to fuse. Measured on a 1,200-player gallery, r@1 by
# bundle count: 1 -> 0.910, 2 -> 0.960, 3 -> 0.977, 4 -> 0.970. The gain is
# almost all in the first extra bundle and has turned over by the fourth, so
# three is the cap -- more forward passes for nothing.
MAX_BUNDLES = 3

# Empirical per-ply think times from real 60+0 games, bucketed by ply index
# (people are fast in the opening, slowest in the middlegame, fast again in time
# trouble). Built by the block at the bottom of this file's docstring; the table
# is ~2.6 KB and ships next to the server so runtime needs no shard.
_THINK = None


def sample_think_ms(ply: int) -> float:
    """A plausible think time for the bot's ply, drawn from real games.

    Falls back to a flat 1s -- the modal value across every bucket -- if the
    table is missing, rather than reintroducing the zero that caused the
    problem.
    """
    global _THINK
    if _THINK is None:
        path = os.path.join(HERE, "think_times.json")
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)["buckets"]
            _THINK = {}
            for b, hist in raw.items():
                vals = np.array([float(k) for k in hist], dtype=np.float64)
                cnt = np.array([float(v) for v in hist.values()], dtype=np.float64)
                _THINK[int(b)] = (vals, cnt / cnt.sum())
        except Exception as e:                       # noqa: BLE001
            print(f"think_times.json unusable ({e}); using 1s", file=sys.stderr)
            _THINK = {}
    b = min(max(ply, 0) // 10, 6)
    ent = _THINK.get(b)
    if not ent:
        return 1000.0
    vals, p = ent
    return float(np.random.choice(vals, p=p))


def load_verifier(ckpt: str, pack: str):
    """Optional second-stage re-ranker over the cosine shortlist.

    Needs both halves: the model, and the candidates' actual games. The gallery
    spans six months of shards, so the games ship as a compact pack (4 games x
    60 plies per player) rather than 39 GB of raw shard.
    """
    if not (ckpt and os.path.isfile(ckpt) and pack and os.path.isfile(pack)):
        print("verifier: not enabled (model or pack missing)", file=sys.stderr)
        return
    from verify2 import DualVerifier
    vk = torch.load(ckpt, map_location="cpu", weights_only=False)
    vcfg = Config(**vk["cfg"])
    trunk = MultiTaskModel(vcfg, n_planes=vk["n_planes"], n_extra=vk["n_extra"],
                           d_embed=vk["d_embed"], n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS, n_game_slots=vk["n_game_slots"],
                           elo_cond=bool(vk.get("elo_cond")))
    ver = DualVerifier(trunk, vcfg.d_model)
    ver.load_state_dict(vk["model"]); ver.eval()
    p = np.load(pack, allow_pickle=True)
    MODEL.update(ver=ver, ver_k=vk["k"], ver_mlpg=vk.get("max_len_per_game", 60),
                 pack={k: p[k] for k in ("moves", "clocks", "nply", "seat",
                                         "tc_base", "tc_inc", "have")})
    print(f"verifier {os.path.basename(ckpt)}: step {vk.get('step')} | pack "
          f"{int((p['have'] > 0).sum()):,} players x {int(p['games'])} games "
          f"x {int(p['plies'])} plies", file=sys.stderr)


def load_bayes(path: str):
    """Platt coefficients that put cosine and the verifier on one scale.

    Without this the two are combined by z-scoring each inside the shortlist and
    adding, which weights them equally by construction. The file also carries
    `recommended`, chosen by held-out rank in calibrate_bayes.py -- if the
    calibrated combination did not actually beat the heuristic, that field says
    "zsum" and nothing changes. Absent or unreadable file means the heuristic.
    """
    if not (path and os.path.isfile(path)):
        return
    try:
        cal = json.load(open(path))
        for k in ("cos_a", "cos_b", "ver_a", "ver_b"):
            cal[k] = float(cal[k])
    except (ValueError, KeyError, OSError) as e:
        print(f"bayes calib: ignored ({e})", file=sys.stderr)
        return
    MODEL["bayes"] = cal
    hp = cal.get("heldout", {})
    print(f"bayes calib {os.path.basename(path)}: fusion={cal.get('recommended')}"
          + (f" | held-out r@10 {hp.get('bayes (per-game LLR)', {}).get('r10', float('nan')):.3f} "
             f"vs {hp.get('z-score sum (deployed)', {}).get('r10', float('nan')):.3f} heuristic"
             if hp else ""), file=sys.stderr)


def load(move_ckpt: str, id_ckpt: str, gallery: str):
    """Two models, deliberately.

    Move prediction uses the PRE-TRAINED trunk: its candidate encoder was
    trained jointly with those weights, and the identification fine-tune moves
    the trunk without ever touching `cand_enc` (no gradient reaches it through
    `embed`), so the fine-tuned pair is mismatched and plays worse.
    Identification uses the fine-tuned model, which is what it is for.
    """
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    mk = torch.load(move_ckpt, map_location="cpu", weights_only=False)
    mv, mcfg = _build(mk)
    MODEL.update(move=mv, cfg=mcfg, n_planes=mk["n_planes"],
                 with_rights=mk["n_planes"] == N_PLANES13,
                 mlpg=mk.get("max_len_per_game", mcfg.max_len),
                 mv_slots=mk.get("n_game_slots", 1),
                 step=mk.get("step"), val=(mk.get("val") or {}).get("move_acc"))
    print(f"move model {os.path.basename(move_ckpt)}: step {mk.get('step')}, "
          f"val move_acc {MODEL['val']}", file=sys.stderr)

    if id_ckpt and os.path.isfile(id_ckpt):
        ik = torch.load(id_ckpt, map_location="cpu", weights_only=False)
        idm, icfg = _build(ik)
        MODEL.update(ident=idm, id_cfg=icfg, id_slots=ik.get("n_game_slots", 1),
                     id_mlpg=ik.get("max_len_per_game", icfg.max_len),
                     id_planes=ik["n_planes"], id_step=ik.get("step"))
        print(f"id model {os.path.basename(id_ckpt)}: step {ik.get('step')}, "
              f"{MODEL['id_slots']} slots", file=sys.stderr)

    if gallery and os.path.isfile(gallery):
        g = np.load(gallery, allow_pickle=True)
        names = list(g["names"])
        MODEL.update(cent=torch.from_numpy(g["centroids"].astype(np.float32)),
                     names=names, gal_k=int(g["k"]), gal_n=len(g["pids"]),
                     # Lowercased lookup for the test-only rank probe. Built once
                     # here rather than scanned per request: at 558k names a
                     # linear scan per identify call is pure waste.
                     name2idx={str(n).lower(): i for i, n in enumerate(names)})
        print(f"gallery {os.path.basename(gallery)}: {MODEL['gal_n']:,} players, "
              f"k={MODEL['gal_k']}", file=sys.stderr)
        # Optional colour banks. A player's white and black repertoires are
        # different objects and one centroid averages them into something that
        # is neither; with the challenge alternating colours we also get two
        # bundles of up to 5 games out of a 5-slot model, i.e. twice the
        # evidence. Absent from older gallery files, hence the guard.
        if "centroids_w" in g.files and "centroids_b" in g.files:
            MODEL.update(cent_w=torch.from_numpy(g["centroids_w"].astype(np.float32)),
                         cent_b=torch.from_numpy(g["centroids_b"].astype(np.float32)),
                         cov=torch.from_numpy((np.asarray(g["n_white"]) > 0) &
                                              (np.asarray(g["n_black"]) > 0)))
            print(f"  colour banks: {int(MODEL['cov'].sum()):,} players have both",
                  file=sys.stderr)


@torch.no_grad()
def think(history_moves: list[str], temperature: float, times=None, elo=None):
    """Score every legal successor of the current position.

    `elo` asks the model to play like a player of that rating. It only does
    anything for a rating-conditioned checkpoint; measured on 900 real
    positions, the top move differs 15.2% of the time between a requested 1000
    and 2200, and agreement with the move actually played peaks when the
    requested rating matches the real mover. Passing None uses the "unknown"
    embedding, which is trained (via --elo-drop) rather than an unused path.
    """
    m = MODEL["move"]
    npl, wr, mlpg = MODEL["n_planes"], MODEL["with_rights"], MODEL["mlpg"]

    board = chess.Board()
    states = []
    for u in history_moves:
        states.append(board.copy())
        board.push(chess.Move.from_uci(u))
    states.append(board.copy())

    legal = list(board.legal_moves)
    if not legal:
        return None, []

    pov = board.turn                      # encode from the mover's seat
    T = min(len(states), mlpg)
    tail = states[-T:]

    planes = np.zeros((1, T, npl, 8, 8), dtype=np.uint8)
    for i, b in enumerate(tail):
        board_to_planes8(b, pov, planes[0, i], wr)
    cands = np.zeros((1, 1, len(legal), npl, 8, 8), dtype=np.uint8)
    for j, mv in enumerate(legal):
        board.push(mv)
        board_to_planes8(board, pov, cands[0, 0, j], wr)
        board.pop()

    # The last tail position is the mover's turn, and turns alternate backwards
    # from it -- so my_turn cannot be derived from ply parity when the tail is
    # truncated to an odd offset.
    my_turn = np.array([[(T - 1 - i) % 2 == 0 for i in range(T)]], dtype=bool)
    fe = features_from_times((times or [])[:len(states)])[-T:]
    if len(fe) < T:
        fe = np.vstack([np.zeros((T - len(fe), N_TIME_FEATS), np.float32), fe])
    eb = None
    if getattr(m, "elo_cond", None) is not None and elo:
        eb = elo_to_bin(torch.tensor([int(elo)]))
    out = m(torch.from_numpy(planes),
            torch.from_numpy(fe)[None],
            torch.from_numpy(cands),
            torch.tensor([[T - 1]]),
            torch.zeros((1, T), dtype=torch.bool),
            torch.from_numpy(my_turn),
            torch.zeros((1, T), dtype=torch.long),      # single game -> slot 0
            torch.arange(T)[None],
            elo_bin=eb)
    logits = out[0][0, 0, :len(legal)]
    probs = torch.softmax(logits if temperature <= 0 else logits / temperature, -1)

    order = torch.argsort(probs, descending=True)
    ranked = [{"uci": legal[i].uci(), "san": board.san(legal[i]),
               "p": float(probs[i])} for i in order.tolist()]
    choice = ranked[0]["uci"] if temperature <= 0 else legal[int(torch.multinomial(probs, 1))].uci()
    return choice, ranked


def features_from_times(ms_per_ply, base_s=60, inc_s=0):
    """Clock features from wall-clock move times, matching timefeat.time_features.

    The model reads think time at every ply and leans on it hard: with these
    features zeroed, top-10 identification on a 4,000-player gallery collapses
    from 25/25 to 4/25. The browser measures the real thing, so there is no
    reason to feed zeros.

    Alignment is the part that matters. Feature[t] describes move t-1, never
    move t -- position t is *before* move t, so move t's duration is not
    causally available there, and leaking it would train the time head to cheat.
    """
    n = len(ms_per_ply)
    feats = np.zeros((n, N_TIME_FEATS), dtype=np.float32)
    if n == 0:
        return feats
    used = np.asarray(ms_per_ply, dtype=np.float64) / 1000.0
    before = np.zeros(n)
    rem = [float(base_s), float(base_s)]
    for t in range(n):
        side = t % 2
        before[t] = rem[side]
        rem[side] = rem[side] - used[t] + inc_s
    log_s = np.log1p(np.clip(used, 0, None))
    frac = np.clip(used / np.maximum(before, 1e-6), 0.0, 1.0)
    feats[1:, 0] = log_s[:-1]
    feats[1:, 1] = frac[:-1]
    return feats


DEV = False                      # set by --dev; gates the save button only
MAX_CLAIMS = 3                   # people legitimately have alt accounts
_CLAIMS: dict[str, list[str]] = {}


def _log_path(name):
    return os.path.join(HERE, name)


def _append_jsonl(name, rec):
    """Append one record. Never let logging break a visitor's session."""
    try:
        with open(_log_path(name), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return True
    except OSError as e:
        print(f"could not write {name}: {e}", file=sys.stderr)
        return False


def load_claims():
    """Rebuild per-visitor claims from the append-only log.

    The log is the record of truth, not a cache of it: every claim and declaim
    is an event, and current state is the replay. That way a restart cannot
    silently drop what people told us, and the history stays auditable -- a
    visitor who claims, declaims and re-claims is visible as exactly that.
    """
    fp = _log_path("claims.jsonl")
    if not os.path.isfile(fp):
        return
    n = 0
    with open(fp, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                     # a torn last line must not be fatal
            vid, nm = r.get("visitor"), r.get("name")
            if not vid or not nm:
                continue
            cur = _CLAIMS.setdefault(vid, [])
            if r.get("claimed"):
                if nm not in cur and len(cur) < MAX_CLAIMS:
                    cur.append(nm)
            elif nm in cur:
                cur.remove(nm)
            n += 1
    print(f"claims: replayed {n} events for {len(_CLAIMS):,} visitors", file=sys.stderr)


_CLOCK_WARNED = set()


def check_clock_track(ms_per_ply, human_white):
    """Shout if one side's clock never got recorded.

    The model reads think time at EVERY ply, not just the visitor's, so a client
    that logs only the human's moves hands it a half-zeroed feature vector. It
    fails silently: the board, the panel and the candidate list all look normal
    while identification collapses.

    Measured 2026-08-17 on three real games -- zeroing only the opponent's plies
    moved the true player from rank 1 to rank 2,750 of 558,735. That defect sat
    in the demo for weeks and read as "the model just isn't good enough", which
    is why this is a loud warning and not a silent repair: guessing a plausible
    number here would hide the same bug the next time it appears.
    """
    n = len(ms_per_ply)
    if n < 8:
        return
    opp = list(ms_per_ply[1 if human_white else 0::2])
    mine = list(ms_per_ply[0 if human_white else 1::2])
    for who, side in (("opponent", opp), ("visitor", mine)):
        if side and not any(side):
            key = f"{who}-all-zero"
            if key not in _CLOCK_WARNED:      # once per process, not per game
                _CLOCK_WARNED.add(key)
                print(f"WARNING: every {who} think time is zero over {len(side)} "
                      f"plies. The client is not recording that side's clock; "
                      f"identification will be far worse than it should be.",
                      file=sys.stderr)


def pack_blocks(blocks, npl, slots):
    """Concatenate game blocks into ONE sequence: slot i per game, ply restarts."""
    T = sum(b[0].shape[0] for b in blocks)
    planes = np.zeros((1, T, npl, 8, 8), np.uint8)
    extra = np.zeros((1, T, N_TIME_FEATS), np.float32)
    slot = np.zeros((1, T), np.int64)
    ppos = np.zeros((1, T), np.int64)
    at = 0
    for si, (pl, fe, mt) in enumerate(blocks):
        t = pl.shape[0]
        planes[0, at:at + t] = pl
        extra[0, at:at + t] = fe
        slot[0, at:at + t] = min(si, slots - 1)
        ppos[0, at:at + t] = np.arange(t)
        at += t
    return (torch.from_numpy(planes), torch.from_numpy(extra),
            torch.zeros((1, T), dtype=torch.bool), torch.from_numpy(slot),
            torch.from_numpy(ppos))


def pack_blocks_each(blocks, npl, slots):
    """One sequence PER block, padded to the longest -- a batch of single games."""
    B = len(blocks)
    T = max(b[0].shape[0] for b in blocks)
    planes = np.zeros((B, T, npl, 8, 8), np.uint8)
    extra = np.zeros((B, T, N_TIME_FEATS), np.float32)
    pad = np.ones((B, T), bool)
    slot = np.zeros((B, T), np.int64)
    ppos = np.zeros((B, T), np.int64)
    for i, (pl, fe, mt) in enumerate(blocks):
        t = pl.shape[0]
        planes[i, :t] = pl
        extra[i, :t] = fe
        pad[i, :t] = False
        ppos[i, :t] = np.arange(t)
    return (torch.from_numpy(planes), torch.from_numpy(extra),
            torch.from_numpy(pad), torch.from_numpy(slot), torch.from_numpy(ppos))


@torch.no_grad()
def verifier_scores(q_blocks, rows, per_game=False):
    """Verifier logits for each candidate's packed games.

    `per_game=True` returns the raw list per candidate, which is what Bayesian
    aggregation needs -- one likelihood term per game. The mean is only the
    right summary if you intend to weight every candidate equally regardless of
    how much evidence they carry.

    Evidence compounds across a candidate's games -- their per-game scores are
    near-uncorrelated (measured -0.27), so averaging is not just smoothing, it
    is accumulating independent tests.
    """
    ver, pack = MODEL["ver"], MODEL["pack"]
    K, mlpg = MODEL["ver_k"], MODEL["ver_mlpg"]
    npl, wr = MODEL["id_planes"], MODEL["with_rights"]

    qp, qe, qpad, qs, qpp = pack_blocks(list(q_blocks)[-(K - 1):], npl,
                                        MODEL["id_slots"])
    with torch.no_grad():
        qv = ver.encode(qp, qe, qpad, qs, qpp, ver.proj_q)

    blocks, owner = [], []
    for r in rows:
        for j in range(int(pack["have"][r])):
            n = int(pack["nply"][r, j])
            if n < 4:
                continue
            codes = pack["moves"][r, j, :n]
            clk = pack["clocks"][r, j, :n]
            seat = int(pack["seat"][r, j])
            b = chess.Board()
            T = min(n, mlpg)
            pl = np.zeros((T, npl, 8, 8), np.uint8)
            okg = True
            for t in range(T):
                board_to_planes8(b, chess.WHITE if seat == 0 else chess.BLACK, pl[t], wr)
                try:
                    b.push(decode_move(int(codes[t])))
                except (AssertionError, ValueError):   # truncated or odd game
                    okg = False
                    break
            if not okg:
                continue
            fe, _, _ = time_features(np.asarray(clk[:T]), int(pack["tc_base"][r, j]),
                                     int(pack["tc_inc"][r, j]))
            mt = np.zeros(T, bool); mt[seat::2] = True
            blocks.append((pl, fe[:T], mt)); owner.append(r)
    if not blocks:
        return {}

    out = {}
    B = 64
    for i in range(0, len(blocks), B):
        chunk = blocks[i:i + B]
        cp, ce, cpad, cs, cpp = pack_blocks_each(chunk, npl, MODEL["id_slots"])
        cv = ver.encode(cp, ce, cpad, cs, cpp, ver.proj_c)
        lo = ver.pair_logits(qv.expand(cv.shape[0], -1), cv).diagonal()
        for k, v in zip(owner[i:i + B], lo.tolist()):
            out.setdefault(k, []).append(v)
    if per_game:
        return {k: [float(x) for x in v] for k, v in out.items()}
    return {k: float(np.mean(v)) for k, v in out.items()}


@torch.no_grad()
def identify(games: list[dict], topn: int = 10, target: str | None = None,
             verify_depth: int = 0):
    """Top-N candidate players for a set of the visitor's finished games.

    Packs up to `n_game_slots` games into ONE sequence with per-game slot and
    ply embeddings -- the joint protocol, which measured +12% over embedding
    each game separately and averaging.
    """
    if "ident" not in MODEL or "cent" not in MODEL:
        return None
    m, slots, mlpg = MODEL["ident"], MODEL["id_slots"], MODEL["id_mlpg"]
    npl, wr = MODEL["id_planes"], MODEL["with_rights"]

    use = [g for g in games if len(g.get("history", [])) >= 2][-slots * MAX_BUNDLES:]
    if not use:
        return None

    def embed_bundle(sel, model=None, n_slots=None):
        """One joint embedding over the most recent `slots` games of `sel`.

        `model` is overridable because the RATING is read from the pre-trained
        trunk, not the identification model. The elo head was supervised during
        pre-training and only drifts during the contrastive fine-tune, which has
        no rating signal at all. Measured on 80 real rated games: the pre-trained
        trunk gets MAE 156 / r 0.837, the fine-tuned one 175 / 0.776, and the
        fine-tuned model's own uncertainty is badly inflated (sd 434 vs 227).
        """
        m = model or MODEL["ident"]
        slots = n_slots or MODEL["id_slots"]
        blocks = build_blocks(sel[-slots:])
        if not blocks:
            return None
        T = sum(b[0].shape[0] for b in blocks)
        planes = np.zeros((1, T, npl, 8, 8), np.uint8)
        extra = np.zeros((1, T, N_TIME_FEATS), np.float32)
        mine = np.zeros((1, T), bool)
        slot = np.zeros((1, T), np.int64)
        ppos = np.zeros((1, T), np.int64)
        at = 0
        for si, (pl, fe, mt) in enumerate(blocks):
            t = pl.shape[0]
            planes[0, at:at + t] = pl
            extra[0, at:at + t] = fe
            mine[0, at:at + t] = mt
            slot[0, at:at + t] = min(si, slots - 1)
            ppos[0, at:at + t] = np.arange(t)      # ply index restarts per game
            at += t
        e, elo_logits = m.embed(torch.from_numpy(planes),
                                torch.from_numpy(extra),
                                torch.zeros((1, T), dtype=torch.bool),
                                torch.from_numpy(mine),
                                torch.from_numpy(slot),
                                torch.from_numpy(ppos))
        return torch.nn.functional.normalize(e.float(), dim=-1), elo_logits

    def build_blocks(sel):
        blocks = []
        for g in sel:
            hist = g["history"]
            human_white = bool(g.get("human_white", True))
            pov = chess.WHITE if human_white else chess.BLACK
            b = chess.Board()
            pos = []
            for u in hist[:mlpg]:
                pos.append(b.copy())
                b.push(chess.Move.from_uci(u))
            T = len(pos)
            if T == 0:
                continue
            pl = np.zeros((T, npl, 8, 8), dtype=np.uint8)
            for i, bb in enumerate(pos):
                board_to_planes8(bb, pov, pl[i], wr)
            mt = np.zeros(T, dtype=bool)
            mt[0 if human_white else 1::2] = True     # the human's own turns
            raw = (g.get("times") or [])[:T]
            check_clock_track(raw, human_white)
            fe = features_from_times(raw)
            if len(fe) < T:                           # missing times -> pad, but warn
                fe = np.vstack([fe, np.zeros((T - len(fe), N_TIME_FEATS), np.float32)])
            blocks.append((pl, fe, mt))
        return blocks

    # Colour-matched scoring is OFF by default because it was measured worse:
    # on a 168,891-player gallery with 4,212 queries, combined scored r@1 0.8977
    # / r@10 0.9791 from five games, and colour-split score fusion scored 0.8846
    # / 0.9558 from TEN -- two bundles of five against a five-slot model. Halving
    # the games behind each centroid costs more than colour-matching gains, which
    # matches the centroid-richness curve (+11% top-10 going 12 -> 64 games).
    # Kept behind a flag rather than deleted so the arm can be re-measured if
    # centroids ever get rich enough that halving them is cheap.
    def rating(logits):
        """Expected Elo and the model's own spread, from the ordinal head.

        The head predicts a distribution over 100-point bins, so the spread is
        available for free and is worth showing: quoting a single number implies
        a precision the model does not have, and the distribution is genuinely
        wide when it has only seen a game or two.
        """
        if logits is None:
            return None
        p = torch.softmax(logits.float(), dim=-1)[0]
        c = ELO_CENTRES.to(p.device)
        mean = float((p * c).sum())
        sd = float(((p * (c - mean) ** 2).sum()).sqrt())
        return {"elo": int(round(mean / 10) * 10), "sd": int(round(sd / 10) * 10)}

    blocks_for_ver = build_blocks(use[-(MODEL.get("ver_k", 5) - 1):]) \
        if "ver" in MODEL else None

    white = [g for g in use if bool(g.get("human_white", True))]
    black = [g for g in use if not bool(g.get("human_white", True))]
    qw = qb = None
    elo = None
    if os.environ.get("COLOUR_FUSION") == "1" and "cent_w" in MODEL and white and black:
        ow, ob = embed_bundle(white), embed_bundle(black)
        if ow is not None and ob is not None:
            qw, lw = ow
            qb, _ = ob
            elo = rating(lw)

    if qw is not None and qb is not None:
        sim = (qw @ MODEL["cent_w"].T + qb @ MODEL["cent_b"].T)[0]
        # A player with no black games in the gallery has a zero row in that
        # bank, and a zero vector has cosine 0 with every query -- which would
        # rank them ABOVE genuinely negative matches. Score those against the
        # combined centroid using the same two bundles, so both arms are a sum
        # of two cosines and remain comparable.
        both = (qw @ MODEL["cent"].T + qb @ MODEL["cent"].T)[0]
        sim = torch.where(MODEL["cov"], sim, both)
        mode = "colour-fused"
        used = len(white[-slots:]) + len(black[-slots:])
    else:
        # The model has `slots` game slots, so one bundle caps at that many
        # games -- but nothing stops us scoring SEVERAL bundles against the same
        # full centroid and summing. Measured on a 1,500-player gallery: one
        # 5-game bundle gets r@1 0.8825, two bundles (10 games) 0.9425. The
        # colour-split arm lost because it halved each centroid, not because
        # fusing bundles is a bad idea; here the centroid is untouched.
        chunks = [use[i:i + slots] for i in range(0, len(use), slots)][:MAX_BUNDLES]
        sims, used = [], 0
        for ch in chunks:
            out = embed_bundle(ch)
            if out is None:
                continue
            q, _ = out
            sims.append((q @ MODEL["cent"].T)[0])
            used += len(ch)
        if not sims:
            return None
        sim = torch.stack(sims).sum(0)
        mode = "combined" if len(sims) == 1 else f"fused-{len(sims)}"

    # Rating from the pre-trained trunk, always -- see embed_bundle's docstring.
    # build_blocks encodes with the IDENT model's plane count and ply cap, so the
    # tensors are only valid for the move model when those agree. They do for the
    # current pair (both 13 planes, 160 plies) because one was fine-tuned from
    # the other, but a future mismatch would silently produce a garbage rating
    # rather than an error.
    if ("move" in MODEL and MODEL["n_planes"] == MODEL["id_planes"]
            and MODEL["mlpg"] == MODEL["id_mlpg"]):
        mo = embed_bundle(use, MODEL["move"], MODEL.get("mv_slots", 1))
        if mo is not None:
            elo = rating(mo[1])

    # Second stage, only when asked for. `verify_depth` candidates from the
    # cosine shortlist get their packed games scored, and the two signals are
    # z-scored before adding: cosine and a logit are on different scales, and
    # summing raw values would silently weight one of them to nothing.
    verified = None
    # "none" means the calibration measured every rerank as worse than the
    # cosine ordering it would replace, so the second stage is skipped even
    # when the caller asks for it -- the alternative is knowingly serving a
    # worse ranking because the machinery exists.
    if (MODEL.get("bayes") or {}).get("recommended") == "none":
        verify_depth = 0
    if verify_depth and "ver" in MODEL and blocks_for_ver:
        d = min(int(verify_depth), len(sim))
        rows = torch.topk(sim, d).indices.tolist()
        # Always ask for per-game scores and average here when needed, so the
        # two fusions cannot disagree about what a score is.
        vs = verifier_scores(blocks_for_ver, rows, per_game=True)
        if vs:
            cs = sim[rows].numpy()
            cal = MODEL.get("bayes")
            mode_v = (cal or {}).get("recommended", "zsum")
            comb = None
            if cal and mode_v != "zsum":
                # Calibrated combination. Platt-scaling each signal turns it
                # into log-odds, and log-odds from independent evidence ADD --
                # so a candidate with four agreeing games outranks one with a
                # single lucky game, which the z-score sum cannot express.
                zc = (cs - cs.mean()) / (cs.std() + 1e-9)
                comb = cal["cos_a"] * zc + cal["cos_b"]
                for i, r in enumerate(rows):
                    sc = vs.get(r) or ()
                    if mode_v == "bayes_mean" and sc:
                        sc = (float(np.mean(sc)),)
                    comb[i] += sum(cal["ver_a"] * s + cal["ver_b"] for s in sc)
                seen = np.array([bool(vs.get(r)) for r in rows])
            else:
                vv = np.array([np.mean(vs[r]) if vs.get(r) else np.nan
                               for r in rows], dtype=np.float64)
                seen = ~np.isnan(vv)
                if seen.sum() < 2:
                    comb = None
                else:
                    zc = (cs - cs.mean()) / (cs.std() + 1e-9)
                    zv = np.zeros_like(vv)
                    zv[seen] = (vv[seen] - vv[seen].mean()) / (vv[seen].std() + 1e-9)
                    comb = zc + zv
            if comb is not None and seen.sum() >= 2:
                new = sim.clone()
                # Lift the shortlist above everything else, reordered by the
                # combined score, so candidates we could not score keep their
                # cosine order below rather than being silently discarded.
                #
                # Shift by the MINIMUM first. Both fusions produce scores
                # centred near zero, so adding them raw drops roughly half the
                # shortlist below `base` -- and a cosine-rank-3 candidate the
                # verifier merely disliked would land beneath 500,000 players it
                # was never compared against. Subtracting the min keeps the
                # shortlist's internal order while guaranteeing every one of
                # them outranks the unscored remainder.
                base = float(sim.max()) + 1.0
                lo = float(np.min(comb))
                for i, r in enumerate(rows):
                    new[r] = base + float(comb[i]) - lo
                sim = new
                verified = {"depth": d, "scored": int(seen.sum()),
                            "fusion": mode_v}

    # Test-only probe: where does one named player actually sit? Top-10 alone
    # cannot distinguish "rank 11" from "rank 400,000", which is exactly the
    # question when checking whether the demo's games carry any signal at all.
    probe = None
    if target:
        j = MODEL.get("name2idx", {}).get(str(target).strip().lower())
        if j is not None:
            n = len(sim)
            rank = int((sim > sim[j]).sum().item()) + 1
            probe = {"name": MODEL["names"][j], "rank": rank, "of": n,
                     "top_pct": round(rank / n * 100, 4),
                     "score": round(float(sim[j]), 4)}
        else:
            probe = {"name": str(target), "missing": True}

    top = torch.topk(sim, min(topn, len(sim)))

    # Turn scores into probabilities the visitor can read. Position 1 carries the
    # measured chance that the top guess is right; the measured chance the answer
    # is somewhere in the list, minus that, is shared among positions 2..10 in
    # proportion to their scores. The column therefore sums to our real top-10
    # recall -- with one game it visibly does not add up to much, which is true.
    vals = top.values.float()
    p1, p10 = recall_for(used)
    rest = max(0.0, p10 - p1)
    tail = vals[1:]
    if len(tail):
        w = torch.softmax((tail - tail.max()) / max(float(tail.std()), 1e-3), dim=0)
        pcts = [p1] + [float(x) * rest for x in w]
    else:
        pcts = [p1]

    return {"games_used": used, "gallery": MODEL["gal_n"], "mode": mode, "elo": elo,
            "probe": probe, "in_top10": p10, "verified": verified,
            "blurb": GALLERY_BLURB,
            "top": [{"name": MODEL["names"][i], "score": round(float(v), 4),
                     "pct": round(pc, 4)}
                    for pc, v, i in zip(pcts, top.values.tolist(), top.indices.tolist())]}


def state(history_moves: list[str], ranked=None, last=None) -> dict:
    board = chess.Board()
    for u in history_moves:
        board.push(chess.Move.from_uci(u))
    over = board.is_game_over()
    reason = ""
    if over:
        if board.is_checkmate():
            reason = "checkmate"
        elif board.is_stalemate():
            reason = "stalemate"
        elif board.is_insufficient_material():
            reason = "insufficient material"
        else:
            reason = "draw"
    return {
        "fen": board.fen(),
        "history": history_moves,
        "turn": "w" if board.turn else "b",
        "legal": [m.uci() for m in board.legal_moves],
        "check": board.is_check(),
        "over": over,
        "reason": reason,
        "result": board.result() if over else "*",
        "top": (ranked or [])[:6],
        "last": last,
        "ply": len(history_moves),
        "info": {"step": MODEL.get("step"), "val": MODEL.get("val"),
                 "gallery": MODEL.get("gal_n", 0),
                 "id_slots": MODEL.get("id_slots", 0)},
    }


VISITOR_COOKIE = "cp_vid"


def human_is_white(n_plies: int, human_moving: bool) -> bool:
    """Which colour the human has, from whose turn it is.

    Ply parity gives the side to move. If the human is the one moving now they
    own that side; if the model is moving, the human owns the other one — which
    is how "let the model open" games end up recorded as the human playing
    black rather than silently mislabelled.
    """
    white_to_move = n_plies % 2 == 0
    return white_to_move if human_moving else not white_to_move


def outcome_for_human(result: str, human_white: bool) -> str:
    """Map a PGN result to win/loss/draw *from the human's point of view*.

    Recording one side consistently matters: an inverted row is invisible in
    aggregate and would quietly turn a losing model into a winning one.
    """
    if result == "1-0":
        return "win" if human_white else "loss"
    if result == "0-1":
        return "loss" if human_white else "win"
    return "draw"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def visitor_id(self):
        """Read the visitor cookie, or mint one for this response."""
        raw = self.headers.get("Cookie", "")
        if raw:
            jar = http.cookies.SimpleCookie()
            try:
                jar.load(raw)
            except http.cookies.CookieError:
                jar = {}
            if VISITOR_COOKIE in jar:
                return jar[VISITOR_COOKIE].value, False
        return metrics.new_visitor_id(), True

    def _send(self, code, body, ctype="application/json", set_cookie=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if set_cookie:
            self.send_header(
                "Set-Cookie",
                f"{VISITOR_COOKIE}={set_cookie}; Max-Age=63072000; Path=/; "
                "HttpOnly; SameSite=Lax",
            )
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            vid, is_new = self.visitor_id()
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                html = f.read()
            # The dev tools are gated SERVER-side, not by a URL parameter, so a
            # production process cannot be talked into exposing them by anyone
            # who guesses "?dev=1". Without --dev the markup never reaches the
            # browser at all.
            if DEV:
                html = html.replace(b"</head>",
                                    b"<script>window.DEV=true;</script></head>", 1)
            return self._send(200, html, "text/html; charset=utf-8",
                              set_cookie=vid if is_new else None)
        # Serve saved game sets so a visitor's history can be restored after a
        # browser wipe -- localStorage is the only copy the demo keeps, and
        # testing the Clear button has already destroyed a real session once.
        if self.path.startswith("/saved/"):
            name = os.path.basename(self.path)
            fp = os.path.join(HERE, "saved", name)
            if not name.endswith(".json") or not os.path.isfile(fp):
                return self._send(404, {"error": "not found"})
            with open(fp, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/pieces/"):
            name = os.path.basename(self.path)
            if not name.endswith(".svg") or "/" in name.replace(os.sep, "/")[:-4]:
                return self._send(404, {"error": "not found"})
            fp = os.path.join(HERE, "pieces", name)
            if not os.path.isfile(fp):
                return self._send(404, {"error": "not found"})
            with open(fp, "rb") as f:
                return self._send(200, f.read(), "image/svg+xml")
        if self.path.startswith("/api/new"):
            return self._send(200, state([]))
        self._send(404, {"error": "not found"})

    def _send_state(self, payload, vid, human_white):
        """Send a game state, recording the outcome if it is a terminal one.

        Every path out of /api/move funnels through here, so a game can't end
        via a branch that forgot to record it.
        """
        if payload.get("over"):
            metrics.record("chess.game_ended", {
                "visitor_id": vid,
                "result": outcome_for_human(payload.get("result", "*"), human_white),
                "reason": payload.get("reason") or "unknown",
                "plies": payload.get("ply"),
                "human_colour": "white" if human_white else "black",
            })
        return self._send(200, payload)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        hist = list(req.get("history", []))
        temp = float(req.get("temperature", 0.0))

        if self.path == "/api/move":
            uci = req.get("uci")
            vid, _ = self.visitor_id()
            human_white = human_is_white(len(hist), human_moving=bool(uci))

            # An empty history means this request opens a new game. The server
            # is stateless, so ply count is the only signal there is.
            if not hist:
                metrics.record("chess.game_started", {
                    "visitor_id": vid,
                    "human_colour": "white" if human_white else "black",
                })

            if uci:
                board = chess.Board()
                for u in hist:
                    board.push(chess.Move.from_uci(u))
                try:
                    mv = chess.Move.from_uci(uci)
                except ValueError:
                    return self._send(400, {"error": f"bad move {uci}"})
                if mv not in board.legal_moves:      # try promotion default
                    mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
                    if mv not in board.legal_moves:
                        return self._send(400, {"error": f"illegal move {uci}"})
                hist.append(mv.uci())

            board = chess.Board()
            for u in hist:
                board.push(chess.Move.from_uci(u))
            if board.is_game_over():
                return self._send_state(state(hist), vid, human_white)

            # The position AFTER the human's move and BEFORE the reply. The
            # client shows this while the opponent "thinks", so the board is not
            # frozen on the pre-move position for the duration -- and so the
            # think time we record is a delay the visitor actually experienced
            # rather than a number attached to an instant reply.
            mid = state(hist)
            choice, ranked = think(hist, temp, req.get("times"), req.get("bot_elo"))
            if choice is None:
                return self._send_state(state(hist), vid, human_white)
            hist.append(choice)

            # Give the reply a plausible human think time and a real clock.
            # Recording 0 ms for every opponent ply -- which is what the demo
            # used to do -- is a distribution the model has never seen: in
            # training every ply carries a genuine clock reading. Measured on a
            # visitor's five games, sampling this from real 1+0 data instead of
            # zero halved their rank in the gallery (21,302 -> ~10,000 of
            # 558,735). The server is stateless, so the bot's remaining time is
            # reconstructed from the times array the client posts back.
            prev = req.get("times") or []
            bot_first = 1 if human_white else 0
            spent = sum(float(t) for i, t in enumerate(prev)
                        if i % 2 == bot_first and t)
            remaining = max(0.0, BASE_MS - spent)
            bot_ms = sample_think_ms(len(hist) - 1)
            flagged = bot_ms >= remaining
            if flagged:
                bot_ms = remaining
            st = state(hist, ranked, last=choice)
            st["mid"] = mid
            st["bot_ms"] = round(bot_ms)
            st["bot_clock_ms"] = round(max(0.0, remaining - bot_ms))
            if flagged and not st["over"]:
                st["over"] = True
                st["reason"] = "time forfeit"
                st["result"] = "1-0" if human_white else "0-1"
            return self._send_state(st, vid, human_white)

        if self.path == "/api/save":
            # Persist a visitor's games server-side. localStorage is otherwise
            # the only copy, and it has already been lost twice in this project
            # -- once to testing the Clear button, once to a stray test run.
            #
            # DEV-gated at the ENDPOINT, not just by hiding the button. The
            # markup ships in index.html either way, so anyone with devtools
            # could unhide it; a hidden control is not an access control. This
            # writes attacker-chosen filenames into play/saved/, so the gate
            # belongs here. /api/giveup stays open by design -- it is how real
            # visitors report a miss -- and names its own files.
            if not DEV:
                return self._send(404, {"error": "not found"})
            try:
                games = req.get("games") or []
                name = str(req.get("name") or "current")
                safe = "".join(c for c in name if c.isalnum() or c in "-_")[:40] or "current"
                os.makedirs(os.path.join(HERE, "saved"), exist_ok=True)
                fp = os.path.join(HERE, "saved", f"{safe}.json")
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(games, f)
                return self._send(200, {"saved": len(games), "file": f"{safe}.json"})
            except Exception as e:                       # noqa: BLE001
                return self._send(500, {"error": str(e)})

        if self.path == "/api/identify":
            # The server is stateless, so the browser keeps the visitor's
            # finished games and posts them back -- same contract as history.
            try:
                res = identify(req.get("games") or [], target=req.get("target"),
                               verify_depth=int(req.get("verify_depth") or 0))
            except Exception as e:                      # never break the game
                print(f"identify failed: {e}", file=sys.stderr)
                res = None
            return self._send(200, res or {"top": [], "games_used": 0,
                                           "gallery": MODEL.get("gal_n", 0)})

        if self.path == "/api/claim":
            # A visitor telling us which account is theirs. This is the only
            # ground truth the demo can ever collect: everywhere else we are
            # guessing, here somebody states the answer.
            vid, _ = self.visitor_id()
            name = str(req.get("name") or "").strip().lower()[:64]
            claimed = bool(req.get("claimed"))
            if not name:
                return self._send(400, {"ok": False, "error": "no name"})
            cur = _CLAIMS.setdefault(vid, [])
            if claimed and name not in cur and len(cur) >= MAX_CLAIMS:
                return self._send(200, {"ok": False, "error": "limit",
                                        "max": MAX_CLAIMS, "claims": cur})
            # Compute the claimed account's rank HERE rather than trusting the
            # client's `rank`, which is only populated when the visitor happened
            # to type their username into the probe box -- so it was null on
            # nearly every real claim. A claim whose rank we know is a labelled
            # eval case; a claim without one is just a name. Recomputing costs a
            # single identify() call and is what makes claims usable as a test
            # set later.
            games = req.get("games") or []
            rank = of = None
            in_top = None
            if claimed and games:
                try:
                    res = identify(games, target=name)
                    probe = (res or {}).get("probe") or {}
                    if not probe.get("missing"):
                        rank, of = probe.get("rank"), probe.get("of")
                    in_top = name in [str(t["name"]).lower()
                                      for t in (res or {}).get("top") or []]
                except Exception as e:                # never block the claim
                    print(f"claim: could not rank {name}: {e}", file=sys.stderr)
            rec = {"ts": time.time(), "visitor": vid, "name": name,
                   "claimed": claimed, "games": len(games),
                   "rank": rank, "of": of, "in_top10": in_top,
                   "client_rank": req.get("rank")}
            _append_jsonl("claims.jsonl", rec)
            if claimed:
                if name not in cur:
                    cur.append(name)
            elif name in cur:
                cur.remove(name)
            print(f"claim: {vid[:8]} {'+' if claimed else '-'}{name} "
                  f"(rank {rank if rank is not None else '?'} of {of or '?'}, "
                  f"{rec['games']} games)", file=sys.stderr)
            return self._send(200, {"ok": True, "claims": cur, "max": MAX_CLAIMS})

        if self.path == "/api/giveup":
            # The most valuable record this demo produces: a labelled MISS. We
            # have the visitor's games and the username we failed to surface,
            # which is a test case no shard-derived eval can manufacture.
            # Games are written in the same shape as play/saved/*.json so the
            # existing replay tooling reads them with no conversion.
            vid, _ = self.visitor_id()
            name = str(req.get("username") or "").strip().lower()[:64]
            games = req.get("games") or []
            if not name:
                return self._send(400, {"ok": False, "error": "no username"})
            safe = "".join(c for c in name if c.isalnum() or c in "-_")[:40] or "anon"
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            fn = f"giveup_{stamp}_{safe}.json"
            saved = False
            if games:
                try:
                    os.makedirs(os.path.join(HERE, "saved"), exist_ok=True)
                    with open(os.path.join(HERE, "saved", fn), "w",
                              encoding="utf-8") as f:
                        json.dump(games, f)
                    saved = True
                except OSError as e:
                    print(f"giveup: could not save games: {e}", file=sys.stderr)
            # Where the true player ACTUALLY ranked is the whole value of a
            # give-up: missing at rank 11 and missing at rank 400,000 are
            # different failures and want different fixes. `missing` means the
            # account is not in the gallery at all, which is a third failure --
            # nothing was ever findable, and no model change would help.
            rank = of = None
            in_gallery = None
            if games:
                try:
                    res = identify(games, target=name)
                    probe = (res or {}).get("probe") or {}
                    in_gallery = not probe.get("missing", False)
                    if in_gallery:
                        rank, of = probe.get("rank"), probe.get("of")
                except Exception as e:
                    print(f"giveup: could not rank {name}: {e}", file=sys.stderr)
            _append_jsonl("giveups.jsonl", {
                "ts": time.time(), "visitor": vid, "username": name,
                "n_games": len(games), "file": fn if saved else None,
                "rank": rank, "of": of, "in_gallery": in_gallery,
                "client_rank": req.get("rank"), "top": req.get("top") or []})
            where = ("not in the gallery at all" if in_gallery is False
                     else f"true rank {rank} of {of}" if rank is not None
                     else "rank unknown")
            print(f"GIVE UP: {name} — missed over {len(games)} games, {where}; "
                  f"games -> saved/{fn if saved else 'NOT SAVED'}", file=sys.stderr)
            return self._send(200, {"ok": True, "saved": saved, "file": fn})

        if self.path == "/api/hint":
            _, ranked = think(hist, 0.0, req.get("times"))
            return self._send(200, state(hist, ranked))

        self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(HERE)
    ap.add_argument("--ckpt", default=os.path.join(root, "ckpt", "final", "ctx5_pre.pt"),
                    help="move model: the PRE-TRAINED trunk, whose candidate "
                         "encoder matches its own weights")
    # The identifier and the gallery are a MATCHED PAIR and must be changed
    # together. Centroids are only comparable to query embeddings produced by
    # the same weights; mixing a ctx5 gallery with a ctx10 encoder does not
    # degrade gracefully, it collapses to noise (measured: r@1 0.0000, median
    # rank 90,036 of 558,735 when probe_dim.py was pointed at the wrong ckpt).
    ap.add_argument("--id-ckpt", default=os.path.join(root, "ckpt", "final", "ctx10_ft.pt"),
                    help="identification model: the contrastive fine-tune. Must "
                         "be the checkpoint that built --gallery")
    ap.add_argument("--gallery", default=os.path.join(HERE, "gallery_ctx10.npz"))
    ap.add_argument("--verifier", default=os.path.join(root, "ckpt", "final",
                                                       "verifier2_best.pt"),
                    help="second-stage re-ranker; ignored unless --pack exists too")
    ap.add_argument("--pack", default=os.path.join(HERE, "verifier_pack.npz"),
                    help="candidates' games, built by build_verifier_pack.py")
    ap.add_argument("--bayes", default=os.path.join(HERE, "bayes_calib.json"),
                    help="Platt coefficients from calibrate_bayes.py; optional")
    ap.add_argument("--port", type=int, default=8000)
    # Default stays LOOPBACK. Binding 0.0.0.0 has to be a deliberate act,
    # because this process has no authentication, no rate limiting and no TLS,
    # and every request costs a torch forward pass over a 137 MB gallery -- it
    # is trivially easy to knock over. Put a reverse proxy in front of it before
    # exposing it, and see docs/prod_deployed.md.
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; use 0.0.0.0 to accept external "
                         "connections (put a proxy in front first)")
    ap.add_argument("--dev", action="store_true",
                    help="expose the dev-only save-games button in the UI")
    args = ap.parse_args()
    global DEV
    DEV = args.dev
    load(args.ckpt, args.id_ckpt, args.gallery)
    load_verifier(args.verifier, args.pack)
    load_bayes(args.bayes)
    load_claims()
    if DEV:
        print("DEV MODE: save-games button exposed", file=sys.stderr)
    metrics.start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.host not in ("127.0.0.1", "localhost"):
        print(f"WARNING: bound to {args.host} — reachable from the network. "
              f"There is no auth, no rate limiting and no TLS here.",
              file=sys.stderr)
    print(f"serving on http://{args.host}:{args.port}", file=sys.stderr)
    srv.serve_forever()


if __name__ == "__main__":
    main()
