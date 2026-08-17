"""Vectorised board -> plane encoding, replacing the per-square Python path.

`board_to_planes8` costs ~75% of dataloader time, and profiling shows why: it
calls `board.piece_map()`, which allocates a Python dict of `Piece` objects by
calling `piece_at` once per occupied square (613,816 calls for 24,638 positions),
and then writes numpy elements one at a time. Every one of those pieces already
exists inside python-chess as a bit in a bitboard. The conversion to objects and
back is pure overhead.

So: read the bitboards directly, and defer the bit-twiddling to one vectorised
call over the whole sample.

    snap = snapshot(board, pov)        # 8 uint64, cheap attribute reads
    ...                                # collect T + P*C of them
    planes = encode_batch(snaps, ...)  # one unpackbits for all of them

Two facts make this exact rather than approximate:

  - `np.unpackbits(..., bitorder="little")` over the little-endian bytes of a
    uint64 puts bit `s` at index `s`, and python-chess numbers squares
    rank-major from A1 -- which is precisely the [rank, file] layout the plane
    format wants. No permutation needed.
  - The POV flip is `square ^ 56`, a vertical mirror, which on a bitboard is
    exactly a byte swap. `int.to_bytes(8, "big")` does it for free.

Plane layout matches bitboards.board_to_planes8 exactly (0-5 piece type, 6 mine,
7 theirs, 8-11 castling, 12 en passant) and is verified against it in
test_fastboard.py over real games.
"""

from __future__ import annotations

import numpy as np
import chess

N_BB = 8                       # 6 piece types + mine + theirs
_EMPTY = np.zeros(N_BB, dtype=np.uint64)


def snapshot_bb(board: chess.Board, pov: bool) -> tuple:
    """The 8 bitboards for `board`, mine/theirs ordered by `pov`.

    Returned as a plain tuple in the ABSOLUTE frame. The vertical mirror that
    puts black at the bottom is deferred to `encode_batch(..., flip=True)`,
    where numpy byte-swaps the whole array at once -- doing it here cost two
    Python int method calls per bitboard, 239k of them per 100 samples.
    """
    return (board.pawns, board.knights, board.bishops, board.rooks,
            board.queens, board.kings,
            board.occupied_co[pov], board.occupied_co[not pov])


def snapshot(board: chess.Board, pov: bool, out: np.ndarray | None = None) -> np.ndarray:
    """Array-writing wrapper over `snapshot_bb`, in `pov`'s frame. Test-facing."""
    if out is None:
        out = np.empty(N_BB, dtype=np.uint64)
    bbs = snapshot_bb(board, pov)
    if pov == chess.WHITE:
        for i, b in enumerate(bbs):
            out[i] = b
    else:
        for i, b in enumerate(bbs):
            out[i] = int.from_bytes(b.to_bytes(8, "little"), "big")
    return out


def rights(board: chess.Board, pov: bool) -> tuple:
    """(my_kingside, my_queenside, their_kingside, their_queenside, ep_square).

    ep_square is returned already POV-flipped, or -1 when there is none.
    """
    them = not pov
    ep = board.ep_square
    if ep is not None and pov == chess.BLACK:
        ep ^= 56
    return (board.has_kingside_castling_rights(pov),
            board.has_queenside_castling_rights(pov),
            board.has_kingside_castling_rights(them),
            board.has_queenside_castling_rights(them),
            -1 if ep is None else ep)


def rights_bb(board: chess.Board, pov: bool) -> tuple:
    """Like `rights`, but leaves ep_square absolute for `encode_batch` to mirror."""
    them = not pov
    ep = board.ep_square
    return (board.has_kingside_castling_rights(pov),
            board.has_queenside_castling_rights(pov),
            board.has_kingside_castling_rights(them),
            board.has_queenside_castling_rights(them),
            -1 if ep is None else ep)


BB_RANK_1 = 0xFF
BB_RANK_8 = 0xFF << 56
_KS_ROOK = {chess.WHITE: (7, 5, 6), chess.BLACK: (63, 61, 62)}    # from, to, king_to
_QS_ROOK = {chess.WHITE: (0, 3, 2), chess.BLACK: (56, 59, 58)}


def successor_bb(board: chess.Board, move: chess.Move, pov: bool):
    """Bitboards and rights AFTER `move`, without touching `board`.

    Replaces push/snapshot/pop, which was the largest single cost in the loader:
    `Board.push` allocates a `_BoardState`, appends to the move stack, and
    maintains halfmove clocks, promoted masks and fullmove counters -- none of
    which the encoder reads. All that is needed is eight integers and five
    flags, and those are a dozen bit operations away.

    Standard chess only (the corpus is lichess `60+0` standard), which is what
    lets the castling-right booleans be read straight off the rights bitboard
    instead of through python-chess's Chess960-aware scan.
    """
    color = board.turn
    frm, to = move.from_square, move.to_square
    frm_bb, to_bb = 1 << frm, 1 << to

    p, n, b = board.pawns, board.knights, board.bishops
    r, q, k = board.rooks, board.queens, board.kings
    occ_w, occ_b = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]
    pt = board.piece_type_at(frm)
    theirs = occ_b if color == chess.WHITE else occ_w

    cap_bb = 0
    if theirs & to_bb:
        cap_bb = to_bb
    elif pt == chess.PAWN and board.ep_square is not None and to == board.ep_square \
            and (frm & 7) != (to & 7):
        cap_bb = 1 << (to - 8 if color == chess.WHITE else to + 8)
    if cap_bb:
        inv = ~cap_bb
        p &= inv; n &= inv; b &= inv; r &= inv; q &= inv
        if color == chess.WHITE:
            occ_b &= inv
        else:
            occ_w &= inv

    castling = board.is_castling(move)
    if castling:
        rf, rt, king_to = (_KS_ROOK if board.is_kingside_castling(move) else _QS_ROOK)[color]
        rfb, rtb, ktb = 1 << rf, 1 << rt, 1 << king_to
        k = (k & ~frm_bb) | ktb
        r = (r & ~rfb) | rtb
        delta = (frm_bb | rfb, ktb | rtb)
    else:
        inv = ~frm_bb
        if pt == chess.PAWN:
            p &= inv
        elif pt == chess.KNIGHT:
            n &= inv
        elif pt == chess.BISHOP:
            b &= inv
        elif pt == chess.ROOK:
            r &= inv
        elif pt == chess.QUEEN:
            q &= inv
        else:
            k &= inv
        dest = move.promotion or pt
        if dest == chess.PAWN:
            p |= to_bb
        elif dest == chess.KNIGHT:
            n |= to_bb
        elif dest == chess.BISHOP:
            b |= to_bb
        elif dest == chess.ROOK:
            r |= to_bb
        elif dest == chess.QUEEN:
            q |= to_bb
        else:
            k |= to_bb
        delta = (frm_bb, to_bb)

    clear, add = delta
    if color == chess.WHITE:
        occ_w = (occ_w & ~clear) | add
    else:
        occ_b = (occ_b & ~clear) | add

    # Same rule python-chess applies: a move from or to a corner kills that
    # rook's right, and any king move kills both of its side's rights.
    cr = board.castling_rights & ~to_bb & ~frm_bb
    if pt == chess.KING:
        cr &= ~(BB_RANK_1 if color == chess.WHITE else BB_RANK_8)

    ep = -1
    if pt == chess.PAWN and (to - frm) in (16, -16):
        ep = (frm + to) >> 1

    mine, theirs = (occ_w, occ_b) if pov == chess.WHITE else (occ_b, occ_w)
    if pov == chess.WHITE:
        rr = (cr >> 7 & 1, cr & 1, cr >> 63 & 1, cr >> 56 & 1)
    else:
        rr = (cr >> 63 & 1, cr >> 56 & 1, cr >> 7 & 1, cr & 1)
    # ep stays absolute; encode_batch mirrors it alongside the bitboards.
    return (p, n, b, r, q, k, mine, theirs), (rr[0], rr[1], rr[2], rr[3], ep)


def successor(board: chess.Board, move: chess.Move, pov: bool,
              out: np.ndarray | None = None):
    """Array-writing wrapper in `pov`'s frame. Test-facing."""
    if out is None:
        out = np.empty(N_BB, dtype=np.uint64)
    bbs, rr = successor_bb(board, move, pov)
    ep = rr[4]
    if pov == chess.WHITE:
        for i, x in enumerate(bbs):
            out[i] = x
    else:
        for i, x in enumerate(bbs):
            out[i] = int.from_bytes(x.to_bytes(8, "little"), "big")
        if ep >= 0:
            ep ^= 56
    return out, (bool(rr[0]), bool(rr[1]), bool(rr[2]), bool(rr[3]), ep)


def encode_batch(snaps: np.ndarray, n_planes: int = 13,
                 rights_arr: np.ndarray | None = None,
                 flip: bool = False) -> np.ndarray:
    """(N, 8) uint64 -> (N, n_planes, 8, 8) uint8.

    `rights_arr` is (N, 5) int16 of the tuples from `rights()`, required when
    n_planes is 13 and ignored when it is 8.
    """
    n = len(snaps)
    if flip:
        # The whole POV mirror, for every position at once: ^56 on each square
        # is a byte reversal of the bitboard.
        snaps = snaps.byteswap()
    # One unpackbits for every position and every bitboard at once. Viewing the
    # uint64 array as uint8 is a reinterpret, not a copy, and is correct only on
    # a little-endian host -- asserted rather than assumed.
    flat = np.unpackbits(snaps.astype("<u8").view(np.uint8).reshape(n, N_BB * 8),
                         axis=1, bitorder="little")
    out = np.zeros((n, n_planes, 8, 8), dtype=np.uint8)
    out[:, :N_BB] = flat.reshape(n, N_BB, 8, 8)

    if n_planes == 13:
        if rights_arr is None:
            raise ValueError("13-plane encoding needs rights_arr")
        r = np.asarray(rights_arr)
        for i in range(4):
            out[:, 8 + i] = r[:, i].astype(np.uint8)[:, None, None]
        ep = r[:, 4]
        if flip:
            ep = np.where(ep >= 0, ep ^ 56, ep)
        has = ep >= 0
        if has.any():
            idx = np.flatnonzero(has)
            sq = ep[idx]
            out[idx, 12, sq >> 3, sq & 7] = 1
    return out


assert np.little_endian, "encode_batch reinterprets uint64 as uint8 in place"
