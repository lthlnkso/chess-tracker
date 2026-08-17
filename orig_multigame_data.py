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
from timefeat import time_features, N_TIME_FEATS


class MultiGameDataset(Dataset):
    """One sample = up to `max_games` game-sides from the SAME player."""

    def __init__(self, shard: str, max_games: int = 3, max_len_per_game: int = 80,
                 plies_per_game: int = 8, n_cand: int = 12, with_rights: bool = True,
                 min_games: int = 3, vary_games: bool = True, seed: int = 0):
        self.meta = np.load(os.path.join(shard, "meta.npy"), mmap_mode="r")
        self.moves = np.memmap(os.path.join(shard, "moves.u16"), dtype=np.uint16, mode="r")
        cp = os.path.join(shard, "clocks.u16")
        if not os.path.exists(cp):
            raise FileNotFoundError(f"{cp} missing -- re-ingest capturing [%clk]")
        self.clocks = np.memmap(cp, dtype=np.uint16, mode="r")
        with open(os.path.join(shard, "players.txt"), encoding="utf-8") as f:
            self.players = f.read().split("\n")

        self.max_games = max_games
        self.max_len_per_game = max_len_per_game
        self.plies_per_game = plies_per_game
        self.n_cand = n_cand
        self.with_rights = with_rights
        self.n_planes = n_planes_compact(with_rights)
        self.vary_games = vary_games
        self.max_len = max_games * max_len_per_game

        # group game-sides by player; only players with enough games qualify
        pid = np.concatenate([np.asarray(self.meta["white_pid"]),
                              np.asarray(self.meta["black_pid"])])
        gidx = np.concatenate([np.arange(len(self.meta)), np.arange(len(self.meta))])
        seat = np.concatenate([np.zeros(len(self.meta), np.int8),
                               np.ones(len(self.meta), np.int8)])
        first = np.asarray(self.meta["offset"], dtype=np.int64)
        clocked = np.asarray(self.clocks[first]) != 0xFFFF
        ok = np.concatenate([clocked, clocked])

        order = np.argsort(pid, kind="stable")
        pid, gidx, seat, ok = pid[order], gidx[order], seat[order], ok[order]
        bounds = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
        self.groups, self.gpid = [], []
        for i in range(len(bounds) - 1):
            sl = slice(bounds[i], bounds[i + 1])
            keep = ok[sl]
            if keep.sum() < min_games:
                continue
            self.groups.append((gidx[sl][keep], seat[sl][keep]))
            self.gpid.append(int(pid[sl][0]))
        self.gpid = np.asarray(self.gpid)

    def __len__(self) -> int:
        return len(self.groups)

    def _one_game(self, gi: int, seat: int, rng):
        row = self.meta[gi]
        o, n = int(row["offset"]), int(row["nply"])
        codes = np.asarray(self.moves[o:o + n])
        clk = np.asarray(self.clocks[o:o + n])
        T = min(len(codes), self.max_len_per_game)
        pov = chess.WHITE if seat == 0 else chess.BLACK

        planes = np.zeros((T, self.n_planes, 8, 8), dtype=np.uint8)
        board = chess.Board()
        boards = []
        for t in range(T):
            board_to_planes8(board, pov, planes[t], self.with_rights)
            boards.append(board.copy())
            board.push(decode_move(int(codes[t])))

        feats, targets, valid = time_features(clk, int(row["tc_base"]), int(row["tc_inc"]))
        my_turn = np.zeros(T, dtype=bool)
        my_turn[seat::2] = True
        elo = int(row["white_elo"] if seat == 0 else row["black_elo"])
        return planes, feats[:T], targets[:T], valid[:T], my_turn, boards, codes[:T], elo

    def __getitem__(self, i: int):
        rng = np.random.default_rng()
        gidx, seats = self.groups[i]
        k = rng.integers(1, self.max_games + 1) if self.vary_games else self.max_games
        k = int(min(k, len(gidx)))
        pick = rng.choice(len(gidx), size=k, replace=False)

        P, C = self.n_planes, self.n_cand
        planes_l, extra_l, ttgt_l, tval_l, mine_l, slot_l = [], [], [], [], [], []
        cand_l, lab_l = [], []
        ply_idx_l = []
        elos = []
        offset = 0
        for slot, j in enumerate(pick):
            gi, seat = int(gidx[j]), int(seats[j])
            pl, fe, tt, tv, mt, boards, codes, elo = self._one_game(gi, seat, rng)
            T = pl.shape[0]
            planes_l.append(pl); extra_l.append(fe); ttgt_l.append(tt)
            tval_l.append(tv); mine_l.append(mt)
            slot_l.append(np.full(T, slot, dtype=np.int64))
            elos.append(elo)

            n_sup = min(self.plies_per_game, T)
            chosen = np.sort(rng.choice(T, size=n_sup, replace=False))
            for t in chosen:
                b = boards[t]
                true_mv = decode_move(int(codes[t]))
                others = [m for m in b.legal_moves if m != true_mv]
                if len(others) > C - 1:
                    others = [others[x] for x in rng.choice(len(others), C - 1, replace=False)]
                sel = [true_mv] + others
                order = rng.permutation(len(sel))
                cc = np.zeros((C, P, 8, 8), dtype=np.uint8)
                pov = chess.WHITE if seat == 0 else chess.BLACK
                for x, pi in enumerate(order):
                    b.push(sel[pi]); board_to_planes8(b, pov, cc[x], self.with_rights); b.pop()
                cand_l.append(cc)
                lab_l.append(int(np.flatnonzero(order == 0)[0]))
                ply_idx_l.append(offset + int(t))
            offset += T

        return {
            "planes": torch.from_numpy(np.concatenate(planes_l)),
            "extra": torch.from_numpy(np.concatenate(extra_l)),
            "time_target": torch.from_numpy(np.concatenate(ttgt_l)),
            "time_valid": torch.from_numpy(np.concatenate(tval_l)),
            "my_turn": torch.from_numpy(np.concatenate(mine_l)),
            "game_slot": torch.from_numpy(np.concatenate(slot_l)),
            "cands": torch.from_numpy(np.stack(cand_l)) if cand_l else torch.zeros(0),
            "label": torch.tensor(lab_l, dtype=torch.long),
            "ply_idx": torch.tensor(ply_idx_l, dtype=torch.long),
            "player_id": int(self.gpid[i]),
            "elo": int(np.median(elos)),
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
    pmask = torch.zeros((B, S), dtype=torch.bool)

    for i, b in enumerate(batch):
        t = b["planes"].shape[0]; s = b["cands"].shape[0]
        planes[i, :t] = b["planes"]; extra[i, :t] = b["extra"]
        pad[i, :t] = False; ttgt[i, :t] = b["time_target"]
        tval[i, :t] = b["time_valid"]; mine[i, :t] = b["my_turn"]
        slot[i, :t] = b["game_slot"]
        if s:
            cands[i, :s] = b["cands"]; cmask[i, :s] = True
            ply[i, :s] = b["ply_idx"]; lab[i, :s] = b["label"]; pmask[i, :s] = True
    return {"planes": planes, "extra": extra, "pad_mask": pad, "time_target": ttgt,
            "time_valid": tval, "my_turn": mine, "game_slot": slot, "cands": cands,
            "cand_mask": cmask, "ply_idx": ply, "label": lab, "ply_mask": pmask,
            "player_id": torch.tensor([b["player_id"] for b in batch], dtype=torch.long),
            "elo": torch.tensor([b["elo"] for b in batch], dtype=torch.long),
            "n_games": torch.tensor([b["n_games"] for b in batch], dtype=torch.long)}
