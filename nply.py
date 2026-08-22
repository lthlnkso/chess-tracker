"""Candidate board states N plies into the future.

The pre-training task today asks which of ~32 legal SUCCESSORS actually
happened. This generalises that to depth N: which of K positions reachable in N
plies is the one the game really reached.

The distractor policy is the whole design, not a detail
--------------------------------------------------------
At depth 1 the distractors come for free and are perfectly calibrated: they are
the other legal moves, all plausible, all reachable, differing from the truth by
exactly one decision. Nothing about that survives to depth N automatically.

Sample N random legal continuations and the negatives stop being hard. Random
play produces positions no human game reaches -- hanging queens, shuffled rooks
-- so the model learns "which of these looks like a real game", which it can do
without knowing anything about the player. The task gets EASIER as depth grows,
which is the opposite of the intent.

So the default policy is `perturb`: replay the true line, diverge at ONE ply,
then continue. A distractor is then a position the game genuinely could have
reached, identical to the truth except for one decision, and the difficulty is
tunable -- a divergence at the last ply leaves no random tail at all and is the
hardest negative available; an early divergence is easier. `late_bias` controls
that distribution.

`random` is kept for comparison, because the difference between the two is
measurable and worth measuring.

Terminal positions
------------------
A game can end inside the window. Depth is clamped to what the line actually
offers and `reached` reports the depth used, so a caller can drop short samples
or train on them deliberately rather than silently comparing states at different
depths.
"""

from __future__ import annotations

import chess
import numpy as np

from fastboard import N_BB, rights_bb, snapshot_bb, successor_bb


def _play_random(board: chess.Board, n: int, rng) -> int:
    """Push up to `n` uniformly random legal moves. Returns how many were made."""
    made = 0
    for _ in range(n):
        lm = list(board.legal_moves)
        if not lm:
            break
        board.push(lm[rng.integers(len(lm))])
        made += 1
    return made


def _snap(board: chess.Board, pov: bool, with_rights: bool):
    return snapshot_bb(board, pov), (rights_bb(board, pov) if with_rights else None)


def n_ply_futures(board: chess.Board, true_line, depth: int, n_cand: int,
                  rng, pov: bool | None = None, policy: str = "perturb",
                  late_bias: float = 2.0, with_rights: bool = True,
                  max_tries: int = 6, own_plies_only: bool = True):
    """K candidate states `depth` plies ahead; exactly one is what really happened.

    `board` is the position before the window and is never mutated.
    `true_line` is the real continuation as chess.Move, in order.

    Returns dict with
        snaps    (K, N_BB) uint64      -- absolute frame, mirror applied later
        rights   (K, 5) int16 or None
        label    int                   -- index of the true state
        reached  int                   -- plies actually used (< depth if the
                                          game ended inside the window)
        n_valid  int                   -- distinct states produced
    """
    if pov is None:
        pov = board.turn
    if depth < 1:
        raise ValueError("depth must be >= 1")

    # Depth 1 has an exact, cheap answer already -- every legal move, no sampling
    # and no duplicates possible. Falling through to the generic path would only
    # reproduce it more slowly and less exactly.
    if depth == 1:
        return _depth_one(board, true_line, n_cand, rng, pov, with_rights)

    line = list(true_line)[:depth]
    # Replay the truth once, recording the position after every ply.
    work = board.copy(stack=False)
    truth_states = []
    for mv in line:
        if mv not in work.legal_moves:
            break
        work.push(mv)
        truth_states.append(work.copy(stack=False))
        if work.is_game_over(claim_draw=False):
            break
    reached = len(truth_states)
    if reached == 0:
        return None
    true_board = truth_states[-1]

    seen = set()
    snaps, rights = [], []

    def add(b: chess.Board) -> bool:
        st, r = _snap(b, pov, with_rights)
        key = (st, r)
        if key in seen:
            return False
        seen.add(key)
        snaps.append(st)
        rights.append(r)
        return True

    add(true_board)                      # the truth is always slot 0 pre-shuffle

    # Prefix boards and the legal alternatives at each divergence ply are shared
    # across every distractor that diverges there. The naive version rebuilt both
    # per candidate, which cost 1.4M generate_legal_moves calls per 120 samples
    # -- 133x the depth-1 task. Build each once.
    prefix = [board.copy(stack=False)]
    for mv in line[:reached - 1]:
        nb = prefix[-1].copy(stack=False)
        nb.push(mv)
        prefix.append(nb)
    alts_cache: dict[int, list] = {}
    # which plies in the window the target player actually moved on
    div_plies = [i for i in range(reached) if prefix[i].turn == pov]

    want = n_cand - 1
    tries = 0
    while len(snaps) - 1 < want and tries < want * max_tries:
        tries += 1
        if policy == "random":
            cand = board.copy(stack=False)
            if _play_random(cand, reached, rng) < reached:
                continue
        else:
            # Diverge at ONE ply, biased late. A divergence at the final ply
            # follows the real game exactly until then and leaves NO random tail
            # -- simultaneously the hardest negative available and the cheapest
            # to build, so difficulty and speed pull the same way here.
            u = rng.random() ** (1.0 / max(late_bias, 1e-6))
            d = min(reached - 1, int(u * reached))
            if own_plies_only and div_plies:
                # Diverge only on plies the TARGET PLAYER moved.
                #
                # A late divergence differs from the truth by exactly one move,
                # and half of those moves belong to the opponent. Telling those
                # apart teaches the model about a stranger, not about the player
                # it is supposed to identify -- the depth-1 task never had this
                # problem because the only decision in it was the player's own.
                d = div_plies[min(int(u * len(div_plies)), len(div_plies) - 1)]
            alts = alts_cache.get(d)
            if alts is None:
                alts = [m for m in prefix[d].legal_moves if m != line[d]]
                alts_cache[d] = alts
            if not alts:
                continue
            alt = alts[rng.integers(len(alts))]
            tail = reached - d - 1
            if tail == 0:
                # Last-ply divergence: the candidate IS a successor of a position
                # we already hold, so skip the board copy, the push and the
                # snapshot entirely and use the same bitboard path the depth-1
                # task uses. High late_bias sends most candidates down here,
                # which is why difficulty and speed pull the same way.
                cs, cr = successor_bb(prefix[d], alt, pov)
                if (cs, cr if with_rights else None) in seen:
                    continue
                seen.add((cs, cr if with_rights else None))
                snaps.append(cs)
                rights.append(cr if with_rights else None)
                continue
            cand = prefix[d].copy(stack=False)
            cand.push(alt)
            if _play_random(cand, tail, rng) < tail:
                continue
        add(cand)

    K = len(snaps)
    order = rng.permutation(K)
    out_s = np.array([snaps[i] for i in order], dtype=np.uint64)
    out_r = (np.array([rights[i] for i in order], dtype=np.int16)
             if with_rights else None)
    return {"snaps": out_s, "rights": out_r,
            "label": int(np.flatnonzero(order == 0)[0]),
            "reached": reached, "n_valid": K}


def _depth_one(board, true_line, n_cand, rng, pov, with_rights):
    """Exact depth-1 set: the true move plus other legal moves, no sampling loop."""
    line = list(true_line)
    if not line or line[0] not in board.legal_moves:
        return None
    true_mv = line[0]
    others = [m for m in board.legal_moves if m != true_mv]
    if len(others) > n_cand - 1:
        idx = rng.choice(len(others), n_cand - 1, replace=False)
        others = [others[i] for i in idx]
    sel = [true_mv] + others
    order = rng.permutation(len(sel))
    snaps, rights = [], []
    for i in order:
        cs, cr = successor_bb(board, sel[i], pov)
        snaps.append(cs)
        rights.append(cr)
    return {"snaps": np.array(snaps, dtype=np.uint64),
            "rights": np.array(rights, dtype=np.int16) if with_rights else None,
            "label": int(np.flatnonzero(order == 0)[0]),
            "reached": 1, "n_valid": len(sel)}
