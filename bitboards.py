"""Bitboard encoding for chess positions, from a chosen player's point of view.

Plane layout, in absolute (white-at-bottom) terms, shape (18, 8, 8) indexed
[plane, rank, file] where rank 0 is White's back rank:

     0-5   white  P N B R Q K
     6-11  black  P N B R Q K
    12     side to move (all ones if White to move)
    13     white kingside castling right
    14     white queenside castling right
    15     black kingside castling right
    16     black queenside castling right
    17     en-passant target square

`to_pov` rewrites that into "me vs them": my pieces always land in planes 0-5
and my back rank is always rank 0, so the model never has to learn the board
twice.
"""

from __future__ import annotations

import numpy as np
import chess

N_PLANES = 18

# Piece type (1..6 in python-chess) -> plane offset within a colour's block.
_PIECE_OFFSET = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_planes(board: chess.Board, out: np.ndarray | None = None) -> np.ndarray:
    """Encode `board` as an absolute (white-at-bottom) uint8 (18, 8, 8) array."""
    if out is None:
        out = np.zeros((N_PLANES, 8, 8), dtype=np.uint8)
    else:
        out.fill(0)

    for square, piece in board.piece_map().items():
        plane = _PIECE_OFFSET[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
        out[plane, square >> 3, square & 7] = 1

    if board.turn == chess.WHITE:
        out[12] = 1

    if board.has_kingside_castling_rights(chess.WHITE):
        out[13] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        out[14] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        out[15] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        out[16] = 1

    if board.ep_square is not None:
        out[17, board.ep_square >> 3, board.ep_square & 7] = 1

    return out


# Plane permutation that swaps the two colours' blocks.
_COLOUR_SWAP = np.array(
    [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 15, 16, 13, 14, 17], dtype=np.intp
)


def to_pov(planes: np.ndarray, pov: bool) -> np.ndarray:
    """Reorient absolute planes so `pov` (chess.WHITE/BLACK) is the "me" side.

    Accepts (18, 8, 8) or a batched (T, 18, 8, 8). White POV is a no-op.
    """
    if pov == chess.WHITE:
        return planes
    flipped = planes[..., _COLOUR_SWAP, :, :][..., ::-1, :]
    # Plane 12 means "White to move" absolutely; from Black's seat it means "my turn".
    flipped = flipped.copy()
    flipped[..., 12, :, :] = 1 - flipped[..., 12, :, :]
    return flipped


# --- move (de)coding -------------------------------------------------------
# uint16: bits 0-5 from-square, 6-11 to-square, 12-14 promotion piece type
# (0 = none, otherwise the python-chess piece type 2..5).

def encode_move(move: chess.Move) -> int:
    return move.from_square | (move.to_square << 6) | ((move.promotion or 0) << 12)


def decode_move(code: int) -> chess.Move:
    promo = (code >> 12) & 0x7
    return chess.Move(code & 0x3F, (code >> 6) & 0x3F, promotion=promo or None)


def game_to_bitboards(
    move_codes: np.ndarray,
    pov: bool,
    include_final: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay a packed move sequence into a POV bitboard sequence.

    Returns
        planes: uint8 (T, 18, 8, 8) — the position before each move, plus the
                final position if `include_final`.
        moves:  uint16 (T,) — the move played from each position, POV-flipped to
                match the planes. The final position (if included) gets 0xFFFF.
    """
    n = len(move_codes)
    t = n + 1 if include_final else n
    planes = np.zeros((t, N_PLANES, 8, 8), dtype=np.uint8)
    out_moves = np.full(t, 0xFFFF, dtype=np.uint16)

    board = chess.Board()
    for i, code in enumerate(move_codes):
        board_to_planes(board, planes[i])
        out_moves[i] = code
        board.push(decode_move(int(code)))
    if include_final:
        board_to_planes(board, planes[n])

    planes = to_pov(planes, pov)
    if pov == chess.BLACK:
        out_moves = _flip_move_codes(out_moves)
    return planes, out_moves


# --- compact POV encoding --------------------------------------------------
# 0-5  piece type (pawn..king), colour-agnostic
# 6    my pieces
# 7    their pieces
# --- with rights (13 planes) ---
# 8    my kingside castling right
# 9    my queenside castling right
# 10   their kingside castling right
# 11   their queenside castling right
# 12   en-passant target square
#
# Written straight into the POV frame, so planes 6/7 already mean me/them and the
# castling planes need no colour swap. Only the en-passant square needs mirroring.
#
# The 8-plane variant cannot distinguish positions differing only in castling
# rights or an available en-passant capture. Legality is unaffected either way
# (candidates come from the move generator), but the rights are real state the
# model was previously blind to. `N_PLANES_COMPACT` selects between them.

N_PLANES8 = 8
N_PLANES13 = 13


def n_planes_compact(with_rights: bool) -> int:
    return N_PLANES13 if with_rights else N_PLANES8


def board_to_planes8(
    board: chess.Board,
    pov: bool,
    out: np.ndarray | None = None,
    with_rights: bool = False,
) -> np.ndarray:
    """Encode `board` as uint8 (8 or 13, 8, 8), already in `pov`'s frame."""
    n = n_planes_compact(with_rights)
    if out is None:
        out = np.zeros((n, 8, 8), dtype=np.uint8)
    else:
        out.fill(0)

    flip = pov == chess.BLACK
    for square, piece in board.piece_map().items():
        sq = square ^ 56 if flip else square
        out[_PIECE_OFFSET[piece.piece_type], sq >> 3, sq & 7] = 1
        out[6 if piece.color == pov else 7, sq >> 3, sq & 7] = 1

    if with_rights:
        them = not pov
        if board.has_kingside_castling_rights(pov):
            out[8] = 1
        if board.has_queenside_castling_rights(pov):
            out[9] = 1
        if board.has_kingside_castling_rights(them):
            out[10] = 1
        if board.has_queenside_castling_rights(them):
            out[11] = 1
        if board.ep_square is not None:
            sq = board.ep_square ^ 56 if flip else board.ep_square
            out[12, sq >> 3, sq & 7] = 1

    return out


def _flip_move_codes(codes: np.ndarray) -> np.ndarray:
    """Vertically mirror the squares inside packed move codes (0xFFFF passes through)."""
    codes = codes.astype(np.uint32)
    valid = codes != 0xFFFF
    frm = codes & 0x3F
    to = (codes >> 6) & 0x3F
    promo = (codes >> 12) & 0x7
    frm = frm ^ 56  # square ^ 56 mirrors the rank, keeping the file
    to = to ^ 56
    out = (frm | (to << 6) | (promo << 12)).astype(np.uint16)
    return np.where(valid, out, np.uint16(0xFFFF))
