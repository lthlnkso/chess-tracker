"""Stream a lichess monthly .pgn.zst into compact, mmap-able shards.

The archive is decompressed on the fly and never lands on disk. Each accepted
game becomes a packed uint16 move sequence plus a fixed-width metadata row;
`bitboards.game_to_bitboards` replays those moves into POV bitboard tensors at
training time. See README.md for why the bitboards are not materialised here.

    python ingest.py --url https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst \
                     --out /workspace/data/2013-01
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import chess
import zstandard as zstd

from bitboards import encode_move

META_DTYPE = np.dtype([
    ("offset", np.uint64),     # index into the flat moves.u16 array
    ("nply", np.uint16),
    ("white_pid", np.uint32),
    ("black_pid", np.uint32),
    ("white_elo", np.uint16),
    ("black_elo", np.uint16),
    ("result", np.int8),       # +1 white win, 0 draw, -1 black win
    ("termination", np.uint8), # index into TERMINATIONS
    ("tc_base", np.uint16),    # initial seconds, 0 for correspondence
    ("tc_inc", np.uint16),     # increment seconds
    ("date", np.uint32),       # YYYYMMDD
])

TERMINATIONS = ["Normal", "Time forfeit", "Abandoned", "Rules infraction", "Unterminated", "?"]
_TERM_IDX = {t: i for i, t in enumerate(TERMINATIONS)}

RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

_HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]\s*$')
_COMMENT_RE = re.compile(r"\{[^}]*\}")
_NAG_RE = re.compile(r"\$\d+")
_MOVENUM_RE = re.compile(r"\b\d+\.(\.\.)?")
_RESULT_TOKEN_RE = re.compile(r"(1-0|0-1|1/2-1/2|\*)\s*$")
# lichess writes "{ [%clk 0:00:59 ] }" after each ply: time REMAINING for the
# player who just moved. Present on ~99% of modern games, absent before ~2017.
_CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
CLK_UNKNOWN = 0xFFFF          # sentinel; real values clamp to 65534 cs (~10.9 min)


# --- parsing ---------------------------------------------------------------

def parse_game(raw: str, min_plies: int, keep_tc: frozenset[str] | None = None):
    """Parse one PGN game. Returns (headers, move_codes) or None if rejected.

    Every rejection that can be decided from headers is decided before the SAN
    loop, which is ~90% of the cost. Filtering to one time control therefore
    makes a full-corpus pass dramatically cheaper, not just smaller.
    """
    headers = {}
    movetext_lines = []
    in_movetext = False
    for line in raw.splitlines():
        if not in_movetext:
            m = _HEADER_RE.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
                continue
            if not line.strip():
                continue
            in_movetext = True
        movetext_lines.append(line)

    if "FEN" in headers or headers.get("Variant", "Standard") != "Standard":
        return None
    result = RESULTS.get(headers.get("Result", "*"))
    if result is None:
        return None
    if headers.get("Termination") in ("Abandoned", "Rules infraction"):
        return None
    if keep_tc is not None and headers.get("TimeControl", "") not in keep_tc:
        return None

    text = " ".join(movetext_lines)

    # Pull clocks out in order BEFORE comments are stripped. They are only
    # trustworthy if there is exactly one per ply -- a partial list cannot be
    # aligned to moves, so the whole game is marked clockless rather than
    # silently misaligned.
    clk_raw = _CLK_RE.findall(text)

    text = _COMMENT_RE.sub(" ", text)
    text = _NAG_RE.sub(" ", text)
    text = _RESULT_TOKEN_RE.sub(" ", text)
    text = _MOVENUM_RE.sub(" ", text)

    board = chess.Board()
    codes = []
    try:
        for san in text.split():
            codes.append(encode_move(board.push_san(san)))
    except (ValueError, AssertionError):
        return None  # malformed / illegal movetext: drop the whole game

    if len(codes) < min_plies:
        return None

    if len(clk_raw) == len(codes):
        clocks = []
        for h, m, sec in clk_raw:
            cs = int((int(h) * 3600 + int(m) * 60 + float(sec)) * 100)
            clocks.append(min(max(cs, 0), CLK_UNKNOWN - 1))
    else:
        clocks = [CLK_UNKNOWN] * len(codes)
    return headers, codes, clocks


def _parse_tc(tc: str) -> tuple[int, int]:
    if not tc or tc == "-" or "+" not in tc:
        return 0, 0
    base, _, inc = tc.partition("+")
    try:
        return min(int(base), 65535), min(int(inc), 65535)
    except ValueError:
        return 0, 0


def _parse_elo(v: str) -> int:
    try:
        return max(0, min(int(v), 65535))
    except (ValueError, TypeError):
        return 0


def _parse_date(v: str) -> int:
    try:
        y, m, d = v.split(".")
        return int(y) * 10000 + int(m) * 100 + int(d)
    except (ValueError, AttributeError):
        return 0


_MIN_PLIES = 0     # set per worker via initializer
_KEEP_TC = None


def _init_worker(min_plies: int, keep_tc=None):
    global _MIN_PLIES, _KEEP_TC
    _MIN_PLIES = min_plies
    _KEEP_TC = frozenset(keep_tc) if keep_tc else None


def parse_batch(raw_games: list[str]):
    """Worker entry point: parse a batch, return lightweight tuples."""
    out = []
    for raw in raw_games:
        parsed = parse_game(raw, _MIN_PLIES, _KEEP_TC)
        if parsed is None:
            continue
        h, codes, clocks = parsed
        out.append((
            h.get("White", "?"),
            h.get("Black", "?"),
            np.asarray(codes, dtype=np.uint16).tobytes(),
            np.asarray(clocks, dtype=np.uint16).tobytes(),
            len(codes),
            _parse_elo(h.get("WhiteElo")),
            _parse_elo(h.get("BlackElo")),
            RESULTS[h["Result"]],
            _TERM_IDX.get(h.get("Termination", "?"), _TERM_IDX["?"]),
            *_parse_tc(h.get("TimeControl", "")),
            _parse_date(h.get("UTCDate", "")),
        ))
    return out


# --- streaming -------------------------------------------------------------

def open_stream(url: str | None, path: str | None):
    """Yield a text stream of decompressed PGN, without staging it on disk."""
    if path:
        fh = open(path, "rb")
        raw = zstd.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(raw, encoding="utf-8", errors="replace"), None
    proc = subprocess.Popen(
        ["curl", "-fsSL", "--retry", "5", "--retry-delay", "5", url],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 22,
    )
    raw = zstd.ZstdDecompressor().stream_reader(proc.stdout)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace"), proc


def iter_games(stream):
    """Split the PGN text stream into per-game chunks."""
    buf: list[str] = []
    seen_movetext = False
    for line in stream:
        if line.startswith("[Event ") and seen_movetext:
            yield "".join(buf)
            buf = []
            seen_movetext = False
        buf.append(line)
        if buf and not line.startswith("[") and line.strip():
            seen_movetext = True
    if seen_movetext:
        yield "".join(buf)


def batched(it, n):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


# --- driver ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="lichess .pgn.zst URL, streamed directly")
    src.add_argument("--path", help="local .pgn.zst file")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--min-plies", type=int, default=10)
    ap.add_argument("--time-controls", default="",
                    help="comma-separated TimeControl values to keep, e.g. '600+0,180+0'. "
                         "Applied before movetext parsing.")
    ap.add_argument("--limit", type=int, default=0, help="stop after N accepted games (0 = all)")
    # os.cpu_count() reports the *host's* cores inside a RunPod container (128 on
    # a box where we are allocated 21), so cap the default rather than trust it.
    ap.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--batch", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    moves_path = os.path.join(args.out, "moves.u16")
    clocks_path = os.path.join(args.out, "clocks.u16")

    players: dict[str, int] = {}
    metas: list[tuple] = []
    offset = 0
    seen = 0
    n_clocked = 0
    truncated = False
    t0 = time.time()

    stream, proc = open_stream(args.url, args.path)
    keep_tc = [t.strip() for t in args.time_controls.split(",") if t.strip()] or None
    if keep_tc:
        print(f"time-control filter: {keep_tc}", file=sys.stderr)
    pool = Pool(args.workers, initializer=_init_worker,
                initargs=(args.min_plies, keep_tc))
    try:
        with open(moves_path, "wb", buffering=1 << 22) as mf, \
                open(clocks_path, "wb", buffering=1 << 22) as cf:
            batches = batched(iter_games(stream), args.batch)
            for results in pool.imap(parse_batch, batches, chunksize=1):
                seen += args.batch
                for (w, b, blob, cblob, nply, welo, belo, res, term,
                     tcb, tci, date) in results:
                    wp = players.setdefault(w, len(players))
                    bp = players.setdefault(b, len(players))
                    mf.write(blob)
                    cf.write(cblob)
                    n_clocked += cblob[:2] != b"\xff\xff"
                    metas.append((offset, nply, wp, bp, welo, belo, res, term, tcb, tci, date))
                    offset += nply
                dt = time.time() - t0
                print(
                    f"\r{len(metas):>10,} games kept | {len(players):>9,} players | "
                    f"{offset:>12,} plies | {len(metas)/max(dt,1e-9):>8,.0f} g/s | {dt:>6.0f}s",
                    end="", file=sys.stderr, flush=True,
                )
                if args.limit and len(metas) >= args.limit:
                    truncated = True
                    break
    finally:
        pool.terminate()
        pool.join()
        try:
            stream.close()
        except Exception:
            pass
    print(file=sys.stderr)

    # A partial download would silently drop games, so make the transfer prove
    # it finished. Only meaningful for a full pass; --limit stops us on purpose.
    if proc is not None:
        if truncated:
            proc.kill()
            proc.wait()
        else:
            err = proc.stderr.read().decode("utf-8", "replace").strip()
            if proc.wait() != 0:
                raise SystemExit(f"download failed (curl exit {proc.returncode}): {err}")

    meta = np.array(metas, dtype=META_DTYPE)
    np.save(os.path.join(args.out, "meta.npy"), meta)

    names = [None] * len(players)
    for name, pid in players.items():
        names[pid] = name
    with open(os.path.join(args.out, "players.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({
            "source": args.url or args.path,
            "complete": not truncated,  # False when --limit cut the pass short
            "games": int(len(meta)),
            "plies": int(offset),
            "players": len(players),
            "min_plies": args.min_plies,
            "time_controls": keep_tc,
            "moves_dtype": "uint16",
            "clocks_dtype": "uint16",
            "games_with_clocks": int(n_clocked),
            "meta_dtype": [(n, META_DTYPE[n].str) for n in META_DTYPE.names],
            "terminations": TERMINATIONS,
            "elapsed_sec": round(time.time() - t0, 1),
        }, f, indent=2)

    print(f"wrote {len(meta):,} games / {offset:,} plies / {len(players):,} players to {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
