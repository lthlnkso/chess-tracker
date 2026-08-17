"""The pre-optimisation `_build`, kept only so speedups can be measured, not claimed.

Every "Nx faster" number in this project compares against this. Reconstructing it
from git is unreliable -- the working tree had uncommitted changes before the
optimisation started -- so the original is preserved here verbatim: the version
that encoded each position through `board_to_planes8` and reached candidate
successors by push/pop.

This is also the reference the equivalence tests compare against, so if it ever
drifts from what was actually replaced, those tests fail rather than the
benchmark quietly flattering itself.
"""

from __future__ import annotations

import numpy as np
import torch
import chess

from bitboards import board_to_planes8, decode_move
from successor_data import MultiTaskDataset


class OriginalBuildMixin:
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
        planes = np.zeros((T, P, 8, 8), dtype=np.uint8)
        cands = np.zeros((n_sup, C, P, 8, 8), dtype=np.uint8)
        cand_mask = np.zeros((n_sup, C), dtype=bool)
        labels = np.zeros(n_sup, dtype=np.int64)

        board = chess.Board()
        for t in range(T):
            board_to_planes8(board, pov, planes[t], self.with_rights)
            true_move = decode_move(int(codes[t]))
            k = slot.get(t)
            if k is not None:
                others = [m for m in board.legal_moves if m != true_move]
                if len(others) > C - 1:
                    others = [others[j] for j in rng.choice(len(others), C - 1, replace=False)]
                sel = [true_move] + others
                order = rng.permutation(len(sel))
                labels[k] = int(np.flatnonzero(order == 0)[0])
                for j, pick in enumerate(order):
                    board.push(sel[pick])
                    board_to_planes8(board, pov, cands[k, j], self.with_rights)
                    board.pop()
                cand_mask[k, :len(sel)] = True
            board.push(true_move)

        return {
            "planes": torch.from_numpy(planes),
            "ply_idx": torch.from_numpy(chosen.astype(np.int64)),
            "cands": torch.from_numpy(cands),
            "cand_mask": torch.from_numpy(cand_mask),
            "label": torch.from_numpy(labels),
        }


class OriginalMultiTaskDataset(OriginalBuildMixin, MultiTaskDataset):
    """MultiTaskDataset with the original per-position encoder."""
