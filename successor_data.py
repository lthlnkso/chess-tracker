"""Dataset for successor-state scoring.

A sample is one (game, seat). It carries the game's board sequence in that
player's frame, plus, at a sampled set of plies, the set of candidate successor
states -- the true one and a sample of the other legal ones, shuffled so position
in the list carries no signal.

Candidates come from real move generation, so every candidate is legal and the
model never has to learn legality.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

import chess
from bitboards import board_to_planes8, decode_move, n_planes_compact
from fastboard import (N_BB, snapshot_bb, rights_bb, successor_bb,
                       encode_batch)

_ZERO8 = (0,) * N_BB
_NO_RIGHTS = (0, 0, 0, 0, -1)   # -1 = no en-passant square; 0 would mean a1


class SuccessorDataset(Dataset):
    def __init__(
        self,
        shard: str,
        max_len: int = 160,
        plies_per_game: int = 12,
        n_cand: int = 16,
        only_my_turn: bool = False,
        with_rights: bool = True,
    ):
        # mmap: six months of shards is ~460M rows (~13 GB) of metadata, which
        # would not survive being loaded eagerly in every dataloader worker.
        self.meta = np.load(os.path.join(shard, "meta.npy"), mmap_mode="r")
        self.moves = np.memmap(os.path.join(shard, "moves.u16"), dtype=np.uint16, mode="r")
        with open(os.path.join(shard, "players.txt"), encoding="utf-8") as f:
            self.players = f.read().split("\n")
        self.max_len = max_len
        self.plies_per_game = plies_per_game
        self.n_cand = n_cand
        self.only_my_turn = only_my_turn
        self.with_rights = with_rights
        self.n_planes = n_planes_compact(with_rights)

        self.index = np.stack([
            np.repeat(np.arange(len(self.meta)), 2),
            np.tile([0, 1], len(self.meta)),
        ], axis=1)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        gi, seat = (int(x) for x in self.index[i])
        row = self.meta[gi]
        o, n = int(row["offset"]), int(row["nply"])
        codes = np.asarray(self.moves[o:o + n])
        out = self._build(codes, seat)
        out["player_id"] = int(row["white_pid"] if seat == 0 else row["black_pid"])
        out["game_idx"] = gi
        return out

    def _build(self, codes: np.ndarray, seat: int) -> dict:
        pov = chess.WHITE if seat == 0 else chess.BLACK

        T = min(len(codes), self.max_len)
        C = self.n_cand
        rng = np.random.default_rng()

        eligible = np.arange(seat, T, 2) if self.only_my_turn else np.arange(T)
        n_sup = min(self.plies_per_game, len(eligible))
        chosen = np.sort(rng.choice(eligible, size=n_sup, replace=False))
        slot = {int(t): k for k, t in enumerate(chosen)}

        P = self.n_planes
        cand_mask = np.zeros((n_sup, C), dtype=bool)
        labels = np.zeros(n_sup, dtype=np.int64)

        # Snapshot bitboards during the replay and encode all of them in one
        # vectorised pass at the end. Encoding position-by-position through
        # board_to_planes8 was ~75% of dataloader time: it round-trips every
        # occupied square through a Python `Piece` object (613k piece_at calls
        # per 24k positions) only to write bits that python-chess already holds
        # as bitboards. Move generation, by contrast, was ~4% -- so this is the
        # part worth replacing, and legality still comes from python-chess.
        n_snap = T + n_sup * C
        # Python lists, converted once. Writing uint64s into a numpy array one
        # element at a time cost ~8 scalar stores per position (270k per 100
        # samples); one np.array() over the finished list is a single bulk
        # conversion. The POV mirror is deferred to encode_batch for the same
        # reason -- as a numpy byteswap it is free, as int.to_bytes round-trips
        # it was 239k Python calls.
        snaps_l = [_ZERO8] * n_snap
        rgt_l = [_NO_RIGHTS] * n_snap if self.with_rights else None

        board = chess.Board()
        for t in range(T):
            snaps_l[t] = snapshot_bb(board, pov)
            if self.with_rights:
                rgt_l[t] = rights_bb(board, pov)
            true_move = decode_move(int(codes[t]))
            k = slot.get(t)
            if k is not None:
                others = [m for m in board.legal_moves if m != true_move]
                if len(others) > C - 1:
                    others = [others[j] for j in rng.choice(len(others), C - 1, replace=False)]
                sel = [true_move] + others
                order = rng.permutation(len(sel))
                labels[k] = int(np.flatnonzero(order == 0)[0])
                base = T + k * C
                if self.with_rights:
                    for j, pick in enumerate(order):
                        snaps_l[base + j], rgt_l[base + j] = successor_bb(
                            board, sel[pick], pov)
                else:
                    for j, pick in enumerate(order):
                        snaps_l[base + j], _ = successor_bb(board, sel[pick], pov)
                cand_mask[k, :len(sel)] = True
            board.push(true_move)

        snaps = np.array(snaps_l, dtype=np.uint64)
        rgt = np.array(rgt_l, dtype=np.int16) if self.with_rights else None
        enc = encode_batch(snaps, P, rgt, flip=(pov == chess.BLACK))
        planes = enc[:T]
        cands = enc[T:].reshape(n_sup, C, P, 8, 8)

        return {
            "planes": torch.from_numpy(planes),
            "ply_idx": torch.from_numpy(chosen.astype(np.int64)),
            "cands": torch.from_numpy(cands),
            "cand_mask": torch.from_numpy(cand_mask),
            "label": torch.from_numpy(labels),
        }


def collate(batch: list[dict]) -> dict:
    """Right-pad the board sequence (T) and the supervised-ply count (P)."""
    B = len(batch)
    T = max(b["planes"].shape[0] for b in batch)
    P = max(b["ply_idx"].shape[0] for b in batch)
    C = batch[0]["cands"].shape[1]
    NP = batch[0]["planes"].shape[1]

    planes = torch.zeros((B, T, NP, 8, 8), dtype=torch.uint8)
    pad_mask = torch.ones((B, T), dtype=torch.bool)          # True = padding
    cands = torch.zeros((B, P, C, NP, 8, 8), dtype=torch.uint8)
    cand_mask = torch.zeros((B, P, C), dtype=torch.bool)
    ply_idx = torch.zeros((B, P), dtype=torch.long)
    label = torch.zeros((B, P), dtype=torch.long)
    ply_mask = torch.zeros((B, P), dtype=torch.bool)         # True = real supervised ply

    for i, b in enumerate(batch):
        t, p = b["planes"].shape[0], b["ply_idx"].shape[0]
        planes[i, :t] = b["planes"]
        pad_mask[i, :t] = False
        cands[i, :p] = b["cands"]
        cand_mask[i, :p] = b["cand_mask"]
        ply_idx[i, :p] = b["ply_idx"]
        label[i, :p] = b["label"]
        ply_mask[i, :p] = True

    return {
        "planes": planes, "pad_mask": pad_mask, "cands": cands,
        "cand_mask": cand_mask, "ply_idx": ply_idx, "label": label,
        "ply_mask": ply_mask,
        "player_id": torch.tensor([b["player_id"] for b in batch], dtype=torch.long),
        "game_idx": torch.tensor([b["game_idx"] for b in batch], dtype=torch.long),
    }


class MultiShardSuccessorDataset(SuccessorDataset):
    """Successor dataset over a combine.py index spanning several month shards.

    Reuses the parent's per-sample logic; only the lookup of which game to read
    changes, so the encoding and candidate sampling stay identical.
    """

    def __init__(self, combined: str, **kw):
        import json
        with open(os.path.join(combined, "shards.json")) as f:
            spec = json.load(f)
        self.shard_paths = spec["shards"]
        self.metas = [np.load(os.path.join(s, "meta.npy"), mmap_mode="r")
                      for s in self.shard_paths]
        self.moves_all = [np.memmap(os.path.join(s, "moves.u16"), dtype=np.uint16, mode="r")
                          for s in self.shard_paths]
        self.idx = np.load(os.path.join(combined, "index.npy"), mmap_mode="r")
        with open(os.path.join(combined, "players.txt"), encoding="utf-8") as f:
            self.players = f.read().split("\n")

        self.max_len = kw.get("max_len", 160)
        self.plies_per_game = kw.get("plies_per_game", 12)
        self.n_cand = kw.get("n_cand", 16)
        self.only_my_turn = kw.get("only_my_turn", False)
        self.with_rights = kw.get("with_rights", True)
        self.n_planes = n_planes_compact(self.with_rights)

    def __len__(self) -> int:
        return len(self.idx)

    def _lookup(self, i: int):
        row = self.idx[i]
        si = int(row["shard"])
        return (self.metas[si][int(row["game"])], self.moves_all[si],
                int(row["seat"]), int(row["gpid"]), int(row["game"]))

    def __getitem__(self, i: int) -> dict:
        meta_row, moves, seat, gpid, gi = self._lookup(i)
        o, n = int(meta_row["offset"]), int(meta_row["nply"])
        codes = np.asarray(moves[o:o + n])
        out = self._build(codes, seat)
        out["player_id"] = gpid
        out["game_idx"] = gi
        return out


class MultiTaskDataset(SuccessorDataset):
    """SuccessorDataset plus clock features, time targets and Elo.

    Only games carrying a full clock trace are kept -- a partial trace cannot be
    aligned to plies, and silently zero-filling it would teach the time head that
    missing means instant.
    """

    def __init__(self, shard: str, **kw):
        super().__init__(shard, **kw)
        import os as _os
        from timefeat import CLK_UNKNOWN
        cp = _os.path.join(shard, "clocks.u16")
        if not _os.path.exists(cp):
            raise FileNotFoundError(
                f"{cp} missing -- re-ingest with a build that captures [%clk]")
        self.clocks = np.memmap(cp, dtype=np.uint16, mode="r")
        assert len(self.clocks) == len(self.moves), "clocks/moves length mismatch"

        first = np.asarray(self.meta["offset"], dtype=np.int64)
        has = np.asarray(self.clocks[first]) != CLK_UNKNOWN
        keep_games = np.flatnonzero(has)
        self.index = np.stack([
            np.repeat(keep_games, 2), np.tile([0, 1], len(keep_games)),
        ], axis=1)
        self.n_clocked_games = int(len(keep_games))

    def __getitem__(self, i: int) -> dict:
        from timefeat import time_features
        gi, seat = (int(x) for x in self.index[i])
        row = self.meta[gi]
        o, n = int(row["offset"]), int(row["nply"])
        codes = np.asarray(self.moves[o:o + n])
        clk = np.asarray(self.clocks[o:o + n])

        out = self._build(codes, seat)
        T = out["planes"].shape[0]

        feats, targets, valid = time_features(
            clk, int(row["tc_base"]), int(row["tc_inc"]))
        out["extra"] = torch.from_numpy(np.ascontiguousarray(feats[:T]))
        out["time_target"] = torch.from_numpy(np.ascontiguousarray(targets[:T]))
        out["time_valid"] = torch.from_numpy(np.ascontiguousarray(valid[:T]))

        my_turn = np.zeros(T, dtype=bool)
        my_turn[seat::2] = True
        out["my_turn"] = torch.from_numpy(my_turn)
        out["player_id"] = int(row["white_pid"] if seat == 0 else row["black_pid"])
        out["elo"] = int(row["white_elo"] if seat == 0 else row["black_elo"])
        out["game_idx"] = gi
        return out


def collate_multitask(batch: list[dict]) -> dict:
    """Pads the successor fields plus the per-ply time channels."""
    base = collate(batch)
    B = len(batch)
    T = base["planes"].shape[1]
    NE = batch[0]["extra"].shape[1]

    extra = torch.zeros((B, T, NE), dtype=torch.float32)
    ttgt = torch.zeros((B, T), dtype=torch.long)   # bucket index, not a value
    tval = torch.zeros((B, T), dtype=torch.bool)
    myturn = torch.zeros((B, T), dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["extra"].shape[0]
        extra[i, :t] = b["extra"]
        ttgt[i, :t] = b["time_target"]
        tval[i, :t] = b["time_valid"]
        myturn[i, :t] = b["my_turn"]
    base.update(extra=extra, time_target=ttgt, time_valid=tval, my_turn=myturn,
                elo=torch.tensor([b["elo"] for b in batch], dtype=torch.long))
    return base


class MultiShardMultiTaskDataset(MultiShardSuccessorDataset):
    """Multi-task data over a combine.py index spanning several month shards.

    Same clock/Elo handling as MultiTaskDataset, but the game lookup goes
    through the cross-month index so the identification gallery can span all
    six months. Rows whose game lacks a clock trace are dropped, for the same
    reason: a partial trace cannot be aligned to plies.
    """

    def __init__(self, combined: str, **kw):
        super().__init__(combined, **kw)
        import os as _os
        from timefeat import CLK_UNKNOWN
        self.clocks_all = []
        for s in self.shard_paths:
            cp = _os.path.join(s, "clocks.u16")
            if not _os.path.exists(cp):
                raise FileNotFoundError(f"{cp} missing -- re-ingest capturing [%clk]")
            self.clocks_all.append(np.memmap(cp, dtype=np.uint16, mode="r"))

        keep = np.ones(len(self.idx), dtype=bool)
        for si in range(len(self.shard_paths)):
            sel = np.flatnonzero(np.asarray(self.idx["shard"]) == si)
            if not len(sel):
                continue
            g = np.asarray(self.idx["game"])[sel]
            first = np.asarray(self.metas[si]["offset"], dtype=np.int64)[g]
            keep[sel] = np.asarray(self.clocks_all[si][first]) != CLK_UNKNOWN
        self.idx = np.asarray(self.idx[keep])
        self.n_dropped = int((~keep).sum())

    def __getitem__(self, i: int) -> dict:
        from timefeat import time_features
        row = self.idx[i]
        si, seat, gpid, gi = (int(row["shard"]), int(row["seat"]),
                              int(row["gpid"]), int(row["game"]))
        meta_row = self.metas[si][gi]
        o, n = int(meta_row["offset"]), int(meta_row["nply"])
        codes = np.asarray(self.moves_all[si][o:o + n])
        clk = np.asarray(self.clocks_all[si][o:o + n])

        out = self._build(codes, seat)
        T = out["planes"].shape[0]
        feats, targets, valid = time_features(
            clk, int(meta_row["tc_base"]), int(meta_row["tc_inc"]))
        out["extra"] = torch.from_numpy(np.ascontiguousarray(feats[:T]))
        out["time_target"] = torch.from_numpy(np.ascontiguousarray(targets[:T]))
        out["time_valid"] = torch.from_numpy(np.ascontiguousarray(valid[:T]))
        my_turn = np.zeros(T, dtype=bool)
        my_turn[seat::2] = True
        out["my_turn"] = torch.from_numpy(my_turn)
        out["player_id"] = gpid
        out["elo"] = int(meta_row["white_elo"] if seat == 0 else meta_row["black_elo"])
        out["game_idx"] = gi
        return out
