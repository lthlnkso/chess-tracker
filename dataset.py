"""Torch Dataset over an ingested shard: one sample per (game, seat).

Each sample is the game as the attributed player saw it — POV bitboards plus the
move played from each position. Bitboards are expanded here rather than read off
disk; see README.md.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

import chess
from bitboards import game_to_bitboards, N_PLANES

PAD_MOVE = 0xFFFF


class PlayerGameDataset(Dataset):
    def __init__(self, shard: str, max_len: int = 160, min_games_per_player: int = 0):
        # mmap: six months of shards is ~460M rows (~13 GB) of metadata, which
        # would not survive being loaded eagerly in every dataloader worker.
        self.meta = np.load(os.path.join(shard, "meta.npy"), mmap_mode="r")
        self.moves = np.memmap(os.path.join(shard, "moves.u16"), dtype=np.uint16, mode="r")
        with open(os.path.join(shard, "players.txt"), encoding="utf-8") as f:
            self.players = f.read().split("\n")
        self.max_len = max_len

        # index: (game_idx, seat) where seat 0 = White, 1 = Black
        seats = np.stack([
            np.repeat(np.arange(len(self.meta)), 2),
            np.tile([0, 1], len(self.meta)),
        ], axis=1)
        if min_games_per_player > 0:
            pids = np.concatenate([self.meta["white_pid"], self.meta["black_pid"]])
            counts = np.bincount(pids, minlength=len(self.players))
            keep = counts[self._pid_of(seats)] >= min_games_per_player
            seats = seats[keep]
        self.index = seats

    def _pid_of(self, seats: np.ndarray) -> np.ndarray:
        g, s = seats[:, 0], seats[:, 1]
        return np.where(s == 0, self.meta["white_pid"][g], self.meta["black_pid"][g])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        gi, seat = (int(x) for x in self.index[i])
        row = self.meta[gi]
        o, n = int(row["offset"]), int(row["nply"])
        codes = np.asarray(self.moves[o:o + n])

        pov = chess.WHITE if seat == 0 else chess.BLACK
        planes, mv = game_to_bitboards(codes, pov, include_final=False)
        planes, mv = planes[: self.max_len], mv[: self.max_len]

        pid = int(row["white_pid"] if seat == 0 else row["black_pid"])
        # Plies this player actually moved: even indices for White, odd for Black.
        my_turn = np.zeros(len(mv), dtype=bool)
        my_turn[seat::2] = True

        return {
            "planes": torch.from_numpy(np.ascontiguousarray(planes)),
            "moves": torch.from_numpy(mv.astype(np.int64)),
            "my_turn": torch.from_numpy(my_turn),
            "player_id": pid,
            "elo": int(row["white_elo"] if seat == 0 else row["black_elo"]),
            "game_idx": gi,
        }


def collate(batch: list[dict]) -> dict:
    """Right-pad to the longest sequence in the batch."""
    t = max(b["planes"].shape[0] for b in batch)
    n = len(batch)
    planes = torch.zeros((n, t, N_PLANES, 8, 8), dtype=torch.uint8)
    moves = torch.full((n, t), PAD_MOVE, dtype=torch.int64)
    my_turn = torch.zeros((n, t), dtype=torch.bool)
    pad_mask = torch.ones((n, t), dtype=torch.bool)  # True = padding
    for i, b in enumerate(batch):
        L = b["planes"].shape[0]
        planes[i, :L] = b["planes"]
        moves[i, :L] = b["moves"]
        my_turn[i, :L] = b["my_turn"]
        pad_mask[i, :L] = False
    return {
        "planes": planes,
        "moves": moves,
        "my_turn": my_turn,
        "pad_mask": pad_mask,
        "player_id": torch.tensor([b["player_id"] for b in batch], dtype=torch.long),
        "elo": torch.tensor([b["elo"] for b in batch], dtype=torch.long),
        "game_idx": torch.tensor([b["game_idx"] for b in batch], dtype=torch.long),
    }
