"""Read the CPL corpus and turn engine evals into a trainable target.

`cpl_label.py` writes, for every labelled ply, the absolute Stockfish eval of
every legal move. This joins that to whatever candidates the loader drew and
converts centipawns into WIN PROBABILITY, which is the space the loss works in.

Why win probability and not centipawns. The loss is `sum_i p_i (y_i - y_j)^2`
over candidates i with the human's move j. In raw centipawns that term is
dominated by the tail: candidates span ~550cp, a third are >=300cp blunders, and
(700-40)^2 swamps everything happening among the near-best moves. Worse, cp is
not linear in importance -- dropping 100cp from +50 decides a game, from +900 it
is noise. A logistic maps both problems away: the target is bounded in [0,1] and
its gradient concentrates where the position is actually in the balance.

    wp(cp) = 1 / (1 + exp(-K * cp))      K = 0.00368208, the lichess constant
             cp    0 -> 0.500
             cp +300 -> 0.753
             cp +1000 -> 0.976   (saturated: +1000 and +1400 are both "winning")

Corpus is keyed by (game index, ply) WITHIN ONE SHARD. Game indices repeat
across shards, so a corpus built on 2026-01 is only valid against 2026-01 --
`assert_shard()` exists so that mismatch fails loudly instead of silently
joining evals to the wrong positions.
"""

from __future__ import annotations

import json
import os

import numpy as np

WP_K = 0.00368208          # lichess centipawn -> win probability
NO_EVAL = np.float32(np.nan)


def win_prob(cp):
    """Centipawns -> win probability in [0, 1], mover's point of view."""
    return 1.0 / (1.0 + np.exp(-WP_K * np.asarray(cp, np.float32)))


class CplCorpus:
    """(game, ply) -> {move code: eval cp}, as flat arrays with a sorted index."""

    def __init__(self, path):
        self.path = path
        self.ply_game = np.load(f"{path}/ply_game.npy")          # (N,)
        self.ply_idx = np.load(f"{path}/ply_idx.npy")            # (N,)
        self.offsets = np.load(f"{path}/offsets.npy")            # (N+1,)
        self.moves = np.memmap(f"{path}/moves.u16", np.uint16, "r")
        self.evals = np.memmap(f"{path}/evals.i16", np.int16, "r")
        with open(f"{path}/manifest.json") as f:
            self.manifest = json.load(f)

        # One int64 key per labelled ply, sorted, so a lookup is a binary search
        # rather than a dict of millions of tuples.
        key = (self.ply_game.astype(np.int64) << 16) | self.ply_idx.astype(np.int64)
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.row = order.astype(np.int64)

    def __len__(self):
        return len(self.key)

    def assert_shard(self, shard):
        """A corpus is bound to the shard it was built from; game indices repeat."""
        want = os.path.basename(str(self.manifest.get("shard", "")).rstrip("/"))
        got = os.path.basename(str(shard).rstrip("/"))
        if want and got and want != got:
            raise ValueError(
                f"CPL corpus was built on shard '{want}' but training on '{got}'. "
                f"Game indices are per shard, so this would join evals to the "
                f"wrong positions. Rebuild the corpus or train on '{want}'.")

    def lookup(self, game, ply, cand_codes):
        """Eval in cp for each candidate move code; NaN where unknown.

        Returns (evals, ok). `ok` is False when the ply is not in the corpus at
        all, which is the common case for an unlabelled game and must disable
        the CPL term for that ply rather than contribute zeros.
        """
        out = np.full(len(cand_codes), NO_EVAL, np.float32)
        k = (np.int64(game) << 16) | np.int64(ply)
        i = np.searchsorted(self.key, k)
        if i >= len(self.key) or self.key[i] != k:
            return out, False
        r = self.row[i]
        a, b = int(self.offsets[r]), int(self.offsets[r + 1])
        if b <= a:
            return out, False
        mc = np.asarray(self.moves[a:b])
        ev = np.asarray(self.evals[a:b], np.float32)
        o = np.argsort(mc, kind="stable")
        mc, ev = mc[o], ev[o]
        j = np.searchsorted(mc, cand_codes)
        j = np.clip(j, 0, len(mc) - 1)
        hit = mc[j] == cand_codes
        out[hit] = ev[j][hit]
        return out, True


def cpl_targets(cand_eval):
    """Win probability per candidate, with NaN preserved as NaN."""
    wp = win_prob(np.nan_to_num(cand_eval, nan=0.0))
    return np.where(np.isnan(cand_eval), np.float32(np.nan), wp.astype(np.float32))
