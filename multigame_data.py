"""Multi-game context: several of one player's games in a single sequence.

Today a sample is ONE game and multi-game queries are handled by averaging
separate embeddings. Averaging is a fixed, unlearned aggregator applied after
the fact. Concatenating games instead changes what pre-training *is*:

    predict the next move, given this game AND this player's earlier games

To do well at that the trunk has to extract whatever is stable about the player
across games -- which is exactly the quantity identification needs. The pretext
task stops being "learn chess" and becomes "learn chess conditioned on who is
playing".

Three details that matter:

- **Position encoding.** Ply index resets per game and a separate game-slot
  embedding marks which game a position belongs to. Using one absolute index
  across the concatenation would make "ply 3 of game 2" collide with "ply 69",
  destroying the ply semantics the model already learned.
- **Variable game count.** Samples carry 1..G games, drawn at random, so the
  deployed model handles a visitor who plays one game as gracefully as three.
  Training only on 3 would make single-game queries out-of-distribution.
- **Causality.** Attention is causal across the whole concatenation, so later
  games see earlier ones. Within a game that is temporal order; across games the
  order is arbitrary, which is fine -- it reads as "context I already have".
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import chess
from bitboards import board_to_planes8, decode_move, n_planes_compact
from bitboards import encode_move
from fastboard import N_BB, snapshot_bb, rights_bb, successor_bb, encode_batch
from nply import n_ply_futures
from timefeat import time_features, N_TIME_FEATS


class _Groups:
    """Per-player slices of three flat arrays.

    A list of (shard, gidx, seat) tuples per player would mean ~3 million small
    numpy objects across six shards. These are views into flat arrays instead,
    so slicing is O(1) and costs no allocation.
    """

    __slots__ = ("sh", "gi", "se", "bounds")

    def __init__(self, sh, gi, se, bounds):
        self.sh, self.gi, self.se, self.bounds = sh, gi, se, bounds

    def __len__(self):
        return len(self.bounds) - 1

    def __getitem__(self, i):
        a, b = self.bounds[i], self.bounds[i + 1]
        return self.sh[a:b], self.gi[a:b], self.se[a:b]


class MultiGameDataset(Dataset):
    """One sample = up to `max_games` game-sides from the SAME player.

    `shard` may be one path or several. With several, players are joined on the
    LOWERCASED USERNAME, because `pid` is per shard and the same integer means
    different people in different months. Joining on pid would silently blend
    strangers into one identity.

    Multi-shard matters most for the identification fine-tune: the deployed
    gallery centroids span six months, so a model trained only on same-month
    bundles has never been shown that a player in January and the same player in
    June are one person.
    """

    def __init__(self, shard, max_games: int = 3, max_len_per_game: int = 80,
                 plies_per_game: int = 8, n_cand: int = 12, with_rights: bool = True,
                 cand_depth: int = 1, late_bias: float = 8.0,
                 min_games: int = 3, vary_games: bool = True, seed: int = 0,
                 same_colour: bool = False, cpl=None, cpl_only: bool = False):
        shards = [shard] if isinstance(shard, (str, bytes, os.PathLike)) else list(shard)
        self.shard_paths = [str(x) for x in shards]
        self.meta, self.moves, self.clocks = [], [], []
        names_per_shard = []
        for sp in self.shard_paths:
            self.meta.append(np.load(os.path.join(sp, "meta.npy"), mmap_mode="r"))
            self.moves.append(np.memmap(os.path.join(sp, "moves.u16"),
                                        dtype=np.uint16, mode="r"))
            cp = os.path.join(sp, "clocks.u16")
            if not os.path.exists(cp):
                raise FileNotFoundError(f"{cp} missing -- re-ingest capturing [%clk]")
            self.clocks.append(np.memmap(cp, dtype=np.uint16, mode="r"))
            with open(os.path.join(sp, "players.txt"), encoding="utf-8") as f:
                names_per_shard.append(f.read().split("\n"))

        self.max_games = max_games
        self.max_len_per_game = max_len_per_game
        self.plies_per_game = plies_per_game
        self.n_cand = n_cand
        # 1 keeps the original task bit-for-bit: every legal successor, no
        # sampling. >1 asks which state the game reached `cand_depth` plies out,
        # with distractors that follow the true line and diverge once -- see
        # nply.py for why random continuations would make the task easier, not
        # harder.
        self.cand_depth = cand_depth
        self.late_bias = late_bias
        self.with_rights = with_rights
        self.n_planes = n_planes_compact(with_rights)
        self.vary_games = vary_games
        self.same_colour = same_colour
        # Optional CplCorpus. When present every supervised ply also carries the
        # engine eval of each candidate, so the loss can grade a wrong answer by
        # HOW wrong it was instead of treating all 31 alternatives alike.
        self.cpl = cpl
        if cpl is not None:
            for sp in self.shard_paths:
                cpl.assert_shard(sp)
        self.max_len = max_games * max_len_per_game

        # One global id per username. Built once here so the per-shard pid can be
        # remapped with a single fancy-index instead of a dict lookup per game.
        self.players, name2g = [], {}
        pid2g = []
        for names in names_per_shard:
            m = np.empty(len(names), np.int64)
            for p, nm in enumerate(names):
                key = nm.lower()
                g = name2g.get(key)
                if g is None:
                    g = len(self.players)
                    name2g[key] = g
                    self.players.append(nm)
                m[p] = g
            pid2g.append(m)

        key_l, sh_l, gi_l, se_l, ok_l = [], [], [], [], []
        for si, md in enumerate(self.meta):
            n = len(md)
            k = np.concatenate([pid2g[si][np.asarray(md["white_pid"])],
                                pid2g[si][np.asarray(md["black_pid"])]])
            key_l.append(k)
            sh_l.append(np.full(2 * n, si, np.uint8))
            gi_l.append(np.concatenate([np.arange(n, dtype=np.int64)] * 2))
            se_l.append(np.concatenate([np.zeros(n, np.int8), np.ones(n, np.int8)]))
            clocked = np.asarray(
                self.clocks[si][np.asarray(md["offset"], dtype=np.int64)]) != 0xFFFF
            ok_l.append(np.concatenate([clocked, clocked]))
        key = np.concatenate(key_l); shd = np.concatenate(sh_l)
        gidx = np.concatenate(gi_l); seat = np.concatenate(se_l)
        ok = np.concatenate(ok_l)
        del key_l, sh_l, gi_l, se_l, ok_l

        keep = ok
        key, shd, gidx, seat = key[keep], shd[keep], gidx[keep], seat[keep]
        order = np.argsort(key, kind="stable")
        key, shd, gidx, seat = key[order], shd[order], gidx[order], seat[order]
        b = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
        sizes = np.diff(b)
        big = sizes >= min_games
        starts, ends = b[:-1][big], b[1:][big]
        take = np.concatenate([np.arange(a, z) for a, z in zip(starts, ends)]) \
            if big.any() else np.zeros(0, np.int64)
        self.gpid = key[starts] if big.any() else np.zeros(0, np.int64)
        bounds = np.r_[0, np.cumsum(sizes[big])]
        self.groups = _Groups(np.ascontiguousarray(shd[take]),
                              np.ascontiguousarray(gidx[take]),
                              np.ascontiguousarray(seat[take]), bounds)

        if cpl is not None and cpl_only:
            # The corpus covers a small SUBSET of the shard (45,759 of 24.2M
            # games). Training on everything would fire the CPL term on ~0.2% of
            # plies, so the arms would be statistically identical and the
            # experiment would measure nothing. Restrict to labelled games, and
            # drop players left with too few -- both arms then see exactly the
            # same data and differ only in w_cpl.
            lab = np.unique(np.asarray(cpl.ply_game))
            keep_g = np.isin(self.groups.gi, lab)
            b0, keep_rows, new_bounds = self.groups.bounds, [], [0]
            gp = []
            for i in range(len(self.groups)):
                a, z = int(b0[i]), int(b0[i + 1])
                idx = np.flatnonzero(keep_g[a:z]) + a
                if len(idx) >= min_games:
                    keep_rows.append(idx)
                    new_bounds.append(new_bounds[-1] + len(idx))
                    gp.append(self.gpid[i])
            if keep_rows:
                sel = np.concatenate(keep_rows)
                self.groups = _Groups(self.groups.sh[sel], self.groups.gi[sel],
                                      self.groups.se[sel],
                                      np.asarray(new_bounds, np.int64))
                self.gpid = np.asarray(gp)
            else:
                self.groups = _Groups(np.zeros(0, np.uint8), np.zeros(0, np.int64),
                                      np.zeros(0, np.int8), np.zeros(1, np.int64))
                self.gpid = np.zeros(0, np.int64)

    def __len__(self) -> int:
        return len(self.groups)

    def _one_game(self, si: int, gi: int, seat: int, rng):
        """Replay one game, snapshotting bitboards instead of encoding planes.

        Three costs removed versus the original, which are the same three that
        successor_data.py shed for a 5.9x loader speedup:

          - `board_to_planes8` per position, which round-trips every occupied
            square through a Python `Piece` object;
          - `board.copy()` per ply, kept only so candidates could be generated
            in a second pass -- they are generated inline now, so the copies go;
          - push/pop per candidate, replaced by `successor_bb`.

        Nothing is encoded here. Snapshots come back as raw bitboards and
        __getitem__ encodes every position of every game in the sample, plus all
        their candidates, in one vectorised call.

        The RNG is drawn in exactly the order the previous implementation used
        (per game: the supervised plies, then per ply the truncation choice and
        the shuffle) so the two produce identical samples from identical seeds.
        """
        row = self.meta[si][gi]
        o, n = int(row["offset"]), int(row["nply"])
        codes = np.asarray(self.moves[si][o:o + n])
        clk = np.asarray(self.clocks[si][o:o + n])
        T = min(len(codes), self.max_len_per_game)
        pov = chess.WHITE if seat == 0 else chess.BLACK
        C = self.n_cand

        n_sup = min(self.plies_per_game, T)
        chosen = np.sort(rng.choice(T, size=n_sup, replace=False))
        slot = {int(t): k for k, t in enumerate(chosen)}

        pos = np.zeros((T, N_BB), dtype=np.uint64)
        cnd = np.zeros((n_sup * C, N_BB), dtype=np.uint64)
        pos_r = np.zeros((T, 5), dtype=np.int16)
        cnd_r = np.zeros((n_sup * C, 5), dtype=np.int16)
        pos_r[:, 4] = -1                       # -1 = no en-passant square; 0 is a1
        cnd_r[:, 4] = -1
        labels = np.zeros(n_sup, dtype=np.int64)
        nsel = np.zeros(n_sup, dtype=np.int64)
        cnd_mv = np.zeros(n_sup * C, dtype=np.uint16)   # move code per slot

        board = chess.Board()
        for t in range(T):
            pos[t] = snapshot_bb(board, pov)
            if self.with_rights:
                pos_r[t] = rights_bb(board, pov)
            true_mv = decode_move(int(codes[t]))
            k = slot.get(t)
            if k is not None:
                base = k * C
                if self.cand_depth <= 1:
                    others = [m for m in board.legal_moves if m != true_mv]
                    if len(others) > C - 1:
                        others = [others[x] for x in rng.choice(len(others), C - 1, replace=False)]
                    sel = [true_mv] + others
                    order = rng.permutation(len(sel))
                    labels[k] = int(np.flatnonzero(order == 0)[0])
                    nsel[k] = len(sel)
                    for x, pi in enumerate(order):
                        cs, cr = successor_bb(board, sel[pi], pov)
                        cnd[base + x] = cs
                        cnd_mv[base + x] = encode_move(sel[pi])
                        if self.with_rights:
                            cnd_r[base + x] = cr
                else:
                    # N-ply futures. cnd_mv stays zero: past depth 1 a candidate
                    # is a position, not a move, so there is nothing for the CPL
                    # corpus to join on -- cpl_ok below therefore stays False and
                    # the graded term switches itself off, which is the correct
                    # behaviour rather than a silent mismatch.
                    fut = n_ply_futures(board, [decode_move(int(c)) for c in
                                                codes[t:t + self.cand_depth]],
                                        self.cand_depth, C, rng, pov=pov,
                                        late_bias=self.late_bias,
                                        with_rights=self.with_rights)
                    if fut is None:
                        nsel[k] = 0
                    else:
                        m = fut["n_valid"]
                        cnd[base:base + m] = fut["snaps"]
                        if self.with_rights and fut["rights"] is not None:
                            cnd_r[base:base + m] = fut["rights"]
                        labels[k] = fut["label"]
                        nsel[k] = m
            board.push(true_mv)

        # Each game in a sample can be a different colour, so the POV mirror is
        # applied per game here rather than once over the whole batch.
        if pov == chess.BLACK:
            pos = pos.byteswap()
            cnd = cnd.byteswap()
            for arr in (pos_r, cnd_r):
                e = arr[:, 4]
                arr[:, 4] = np.where(e >= 0, e ^ 56, e)

        # Join candidates to engine evals. Done here, where the game index and
        # ply are still in scope, and per ply because the corpus is sparse --
        # an unlabelled ply must switch the CPL term OFF rather than feed zeros.
        cnd_ev = np.full(n_sup * C, np.float32("nan"), np.float32)
        cpl_ok = np.zeros(n_sup, dtype=bool)
        if self.cpl is not None:
            for kk in range(n_sup):
                lo = kk * C
                ev, ok = self.cpl.lookup(gi, int(chosen[kk]),
                                         cnd_mv[lo:lo + int(nsel[kk])])
                if ok:
                    cnd_ev[lo:lo + int(nsel[kk])] = ev
                    cpl_ok[kk] = True

        feats, targets, valid = time_features(clk, int(row["tc_base"]), int(row["tc_inc"]))
        my_turn = np.zeros(T, dtype=bool)
        my_turn[seat::2] = True
        elo = int(row["white_elo"] if seat == 0 else row["black_elo"])
        # Outcome from THIS player's side, not White's: 0 loss, 1 draw, 2 win.
        # meta stores +1 White, -1 Black, 0 draw, so Black's view is a sign flip.
        r = int(row["result"])
        if seat == 1:
            r = -r
        result = {-1: 0, 0: 1, 1: 2}[r]
        return (pos, pos_r, cnd, cnd_r, chosen, labels, nsel, T,
                feats[:T], targets[:T], valid[:T], my_turn, elo, cnd_ev, cpl_ok,
                result)

    def __getitem__(self, i: int):
        rng = np.random.default_rng()
        shds, gidx, seats = self.groups[i]
        k = rng.integers(1, self.max_games + 1) if self.vary_games else self.max_games
        k = int(min(k, len(gidx)))
        if self.same_colour:
            # Deployment scores a colour-matched query against a colour-specific
            # centroid, so every bundle it ever sees is one colour. Uniform
            # sampling makes that a ~6% case in training ((1/2)^k each way), and
            # a colour-split eval of the mixed-trained model lost 2.3 points of
            # top-10 DESPITE twice the query games -- far more than the measured
            # cost of halving a centroid (32->64 games is worth only +0.6%), so
            # the mismatch is the likelier culprit. Train on what we deploy.
            #
            # Falls back to mixed when a player lacks k of the chosen colour,
            # rather than dropping them: excluding those players would quietly
            # bias the set toward the colour-balanced and toward the very active.
            want = int(rng.integers(0, 2))
            same = np.flatnonzero(seats == want)
            if len(same) < k:
                other = np.flatnonzero(seats == 1 - want)
                same = other if len(other) >= k else np.arange(len(gidx))
            pick = rng.choice(same, size=min(k, len(same)), replace=False)
        else:
            pick = rng.choice(len(gidx), size=k, replace=False)

        P, C = self.n_planes, self.n_cand
        pos_l, posr_l, cnd_l, cndr_l = [], [], [], []
        extra_l, ttgt_l, tval_l, mine_l, slot_l = [], [], [], [], []
        lab_l, ply_l, nsel_l, elos = [], [], [], []
        res_l = []
        cev_l, cok_l = [], []
        offset = 0
        for slot, j in enumerate(pick):
            gi, seat, si = int(gidx[j]), int(seats[j]), int(shds[j])
            (pos, posr, cnd, cndr, chosen, labels, nsel, T,
             fe, tt, tv, mt, elo, cev, cok, res) = self._one_game(si, gi, seat, rng)
            cev_l.append(cev.reshape(-1, self.n_cand)); cok_l.append(cok)
            pos_l.append(pos); posr_l.append(posr)
            cnd_l.append(cnd); cndr_l.append(cndr)
            extra_l.append(fe); ttgt_l.append(tt); tval_l.append(tv)
            mine_l.append(mt)
            slot_l.append(np.full(T, slot, dtype=np.int64))
            lab_l.append(labels); nsel_l.append(nsel)
            ply_l.append(chosen + offset)
            elos.append(elo)
            # one label per PLY: the head is a value function, predicting the
            # eventual result from each position. The trunk is causal, so ply t
            # genuinely cannot see the moves that decided the game.
            res_l.append(np.full(T, res, dtype=np.int64))
            offset += T

        # One encode for the entire sample: every position of every game, plus
        # every candidate. Each game was already put in its own POV frame inside
        # _one_game, because games in one sample can be different colours.
        total_T = offset
        snaps = np.concatenate(pos_l + cnd_l)
        rgts = np.concatenate(posr_l + cndr_l) if self.with_rights else None
        enc = encode_batch(snaps, P, rgts)
        planes = enc[:total_T]
        cands = enc[total_T:].reshape(-1, C, P, 8, 8)
        nsel = np.concatenate(nsel_l)

        return {
            "planes": torch.from_numpy(planes),
            "extra": torch.from_numpy(np.concatenate(extra_l)),
            "time_target": torch.from_numpy(np.concatenate(ttgt_l)),
            "time_valid": torch.from_numpy(np.concatenate(tval_l)),
            "my_turn": torch.from_numpy(np.concatenate(mine_l)),
            "game_slot": torch.from_numpy(np.concatenate(slot_l)),
            "cands": torch.from_numpy(cands) if len(cands) else torch.zeros(0),
            # Per-candidate validity. The previous collate marked all C slots
            # valid for every supervised ply, so positions with fewer than C
            # legal moves had all-zero boards scored as real candidates.
            "cand_valid": torch.from_numpy(
                np.arange(C)[None, :] < nsel[:, None]),
            "label": torch.from_numpy(np.concatenate(lab_l)),
            # Engine eval per candidate in centipawns, NaN where unknown, plus a
            # per-ply switch. Both are all-NaN / all-False when no corpus is
            # attached, so the baseline path is untouched.
            "cand_eval": torch.from_numpy(np.concatenate(cev_l)),
            "cpl_ok": torch.from_numpy(np.concatenate(cok_l)),
            "ply_idx": torch.from_numpy(np.concatenate(ply_l)),
            "player_id": int(self.gpid[i]),
            "elo": int(np.median(elos)),
            "result": torch.from_numpy(np.concatenate(res_l)),
            "n_games": k,
        }


def collate_multigame(batch: list[dict]) -> dict:
    B = len(batch)
    T = max(b["planes"].shape[0] for b in batch)
    S = max(b["cands"].shape[0] for b in batch)
    NP = batch[0]["planes"].shape[1]
    C = batch[0]["cands"].shape[1]
    NE = batch[0]["extra"].shape[1]

    planes = torch.zeros((B, T, NP, 8, 8), dtype=torch.uint8)
    extra = torch.zeros((B, T, NE), dtype=torch.float32)
    pad = torch.ones((B, T), dtype=torch.bool)
    ttgt = torch.zeros((B, T), dtype=torch.long)
    tval = torch.zeros((B, T), dtype=torch.bool)
    mine = torch.zeros((B, T), dtype=torch.bool)
    slot = torch.zeros((B, T), dtype=torch.long)
    cands = torch.zeros((B, S, C, NP, 8, 8), dtype=torch.uint8)
    cmask = torch.zeros((B, S, C), dtype=torch.bool)
    ply = torch.zeros((B, S), dtype=torch.long)
    lab = torch.zeros((B, S), dtype=torch.long)
    res = torch.zeros((B, T), dtype=torch.long)
    pmask = torch.zeros((B, S), dtype=torch.bool)
    cev = torch.full((B, S, C), float('nan'), dtype=torch.float32)
    cok = torch.zeros((B, S), dtype=torch.bool)

    for i, b in enumerate(batch):
        t = b["planes"].shape[0]; s = b["cands"].shape[0]
        planes[i, :t] = b["planes"]; extra[i, :t] = b["extra"]
        pad[i, :t] = False; ttgt[i, :t] = b["time_target"]
        tval[i, :t] = b["time_valid"]; mine[i, :t] = b["my_turn"]
        res[i, :t] = b["result"]
        slot[i, :t] = b["game_slot"]
        if s:
            cands[i, :s] = b["cands"]; cmask[i, :s] = b["cand_valid"]
            ply[i, :s] = b["ply_idx"]; lab[i, :s] = b["label"]; pmask[i, :s] = True
            cev[i, :s] = b["cand_eval"]; cok[i, :s] = b["cpl_ok"]
    return {"planes": planes, "extra": extra, "pad_mask": pad, "time_target": ttgt,
            "time_valid": tval, "my_turn": mine, "game_slot": slot, "cands": cands,
            "cand_mask": cmask, "ply_idx": ply, "label": lab, "ply_mask": pmask,
            "cand_eval": cev, "cpl_ok": cok,
            "player_id": torch.tensor([b["player_id"] for b in batch], dtype=torch.long),
            "elo": torch.tensor([b["elo"] for b in batch], dtype=torch.long),
            "result": res,
            "n_games": torch.tensor([b["n_games"] for b in batch], dtype=torch.long)}
