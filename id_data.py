"""Data plumbing for player identification.

Three pieces:

`split_players`  -- deterministic 80/20 split over players, then the game-side
                    assignment rule: a game-side trains if its own player is a
                    train player; a game is a test game only when *both* seats
                    belong to test players.

`EmbedDataset`   -- board sequences only. Identification never needs candidate
                    successors, and generating them was the dataloader
                    bottleneck during pre-training, so this is ~2x cheaper.

`PKSampler`      -- batches of P players x K games, which is what batch-hard
                    triplet mining requires: every anchor needs a positive and a
                    negative present in its own batch.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

import chess
from bitboards import board_to_planes8, decode_move, n_planes_compact


# --- splitting -------------------------------------------------------------

def _hash_u64(x: np.ndarray, seed: int) -> np.ndarray:
    h = (x.astype(np.uint64) + np.uint64(seed)) * np.uint64(0x9E3779B97F4A7C15)
    h ^= h >> np.uint64(31)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(29)
    return h


def split_players(idx: np.ndarray, n_players: int, test_frac: float = 0.2,
                  seed: int = 0, strict: bool = False):
    """Return (train_rows, test_rows, is_test_player).

    train_rows -- indices into `idx` used for metric learning
    test_rows  -- indices into `idx` for games where both seats are test players

    With `strict`, any game touching a test player is withheld from training.
    The default (the looser rule) still never trains on a test player's own
    game-side, but a test player's *games* can appear in training labelled as
    their opponent. That is the standard practice and the leakage is indirect,
    but `strict` exists so the difference can be measured rather than assumed.
    """
    is_test_player = (_hash_u64(np.arange(n_players, dtype=np.uint64), seed)
                      % np.uint64(1000)) < np.uint64(int(test_frac * 1000))

    gpid = np.asarray(idx["gpid"])
    row_is_test = is_test_player[gpid]

    # A game is identified by (shard, game); both seats present means both
    # players survived the 100+ games filter.
    key = (np.asarray(idx["shard"], dtype=np.uint64) << np.uint64(32)) | \
        np.asarray(idx["game"], dtype=np.uint64)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)

    # per-game: how many of its present seats are test players
    test_per_game = np.bincount(inv, weights=row_is_test.astype(np.int64),
                                minlength=len(uniq)).astype(np.int64)
    both_test = (counts == 2) & (test_per_game == 2)
    touches_test = test_per_game > 0

    test_rows = np.flatnonzero(both_test[inv])
    if strict:
        train_rows = np.flatnonzero(~row_is_test & ~touches_test[inv])
    else:
        train_rows = np.flatnonzero(~row_is_test)
    return train_rows, test_rows, is_test_player


# --- dataset ---------------------------------------------------------------

class EmbedDataset(Dataset):
    """Board sequences in the attributed player's frame, no candidates."""

    def __init__(self, combined: str, rows: np.ndarray | None = None,
                 max_len: int = 160, with_rights: bool = True):
        with open(os.path.join(combined, "shards.json")) as f:
            spec = json.load(f)
        self.shard_paths = spec["shards"]
        self.metas = [np.load(os.path.join(s, "meta.npy"), mmap_mode="r")
                      for s in self.shard_paths]
        self.moves_all = [np.memmap(os.path.join(s, "moves.u16"), dtype=np.uint16, mode="r")
                          for s in self.shard_paths]
        full = np.load(os.path.join(combined, "index.npy"), mmap_mode="r")
        self.idx = full if rows is None else np.asarray(full[rows])
        self.max_len = max_len
        self.with_rights = with_rights
        self.n_planes = n_planes_compact(with_rights)

    def __len__(self) -> int:
        return len(self.idx)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self.idx["gpid"])

    def __getitem__(self, i: int) -> dict:
        row = self.idx[i]
        si, seat = int(row["shard"]), int(row["seat"])
        meta_row = self.metas[si][int(row["game"])]
        o, n = int(meta_row["offset"]), int(meta_row["nply"])
        codes = np.asarray(self.moves_all[si][o:o + n])

        pov = chess.WHITE if seat == 0 else chess.BLACK
        T = min(len(codes), self.max_len)
        planes = np.zeros((T, self.n_planes, 8, 8), dtype=np.uint8)

        board = chess.Board()
        for t in range(T):
            board_to_planes8(board, pov, planes[t], self.with_rights)
            board.push(decode_move(int(codes[t])))

        my_turn = np.zeros(T, dtype=bool)
        my_turn[seat::2] = True          # the plies this player actually chose
        return {
            "planes": torch.from_numpy(planes),
            "my_turn": torch.from_numpy(my_turn),
            "player_id": int(row["gpid"]),
            "row": i,
        }


def collate(batch: list[dict]) -> dict:
    B = len(batch)
    T = max(b["planes"].shape[0] for b in batch)
    NP = batch[0]["planes"].shape[1]
    planes = torch.zeros((B, T, NP, 8, 8), dtype=torch.uint8)
    pad_mask = torch.ones((B, T), dtype=torch.bool)
    my_turn = torch.zeros((B, T), dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["planes"].shape[0]
        planes[i, :t] = b["planes"]
        pad_mask[i, :t] = False
        my_turn[i, :t] = b["my_turn"]
    return {
        "planes": planes, "pad_mask": pad_mask, "my_turn": my_turn,
        "player_id": torch.tensor([b["player_id"] for b in batch], dtype=torch.long),
        "row": torch.tensor([b["row"] for b in batch], dtype=torch.long),
    }


# --- sampling --------------------------------------------------------------

class PKSampler(Sampler):
    """Yield batches of P players x K game-sides.

    Batch-hard mining is only defined when each anchor has a same-player
    positive in the batch, so a plain shuffle will not do.
    """

    def __init__(self, labels: np.ndarray, p: int = 32, k: int = 4,
                 batches_per_epoch: int = 100_000, seed: int = 0):
        self.p, self.k = p, k
        self.batches_per_epoch = batches_per_epoch
        self.rng = np.random.default_rng(seed)

        order = np.argsort(labels, kind="stable")
        sorted_lbl = labels[order]
        bounds = np.flatnonzero(np.r_[True, sorted_lbl[1:] != sorted_lbl[:-1], True])
        self.groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
        # A player with fewer than K rows cannot fill its slot.
        self.groups = [g for g in self.groups if len(g) >= k]
        if not self.groups:
            raise ValueError(f"no player has >= {k} game-sides")

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        n = len(self.groups)
        for _ in range(self.batches_per_epoch):
            picks = self.rng.choice(n, size=min(self.p, n), replace=False)
            batch = []
            for gi in picks:
                g = self.groups[gi]
                batch.extend(self.rng.choice(g, size=self.k, replace=False).tolist())
            yield batch
