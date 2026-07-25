"""Measure what a modern lichess month costs, by streaming a prefix of it.

Recent PGNs carry [%eval] and [%clk] annotations that 2013 did not, so both the
games-per-compressed-byte ratio and the parse cost differ. Extrapolating 2013
numbers to 2026 would be wrong.

    python measure_month.py --month 2026-06 --gb 3 --workers 19
"""

from __future__ import annotations

import argparse
import io
import subprocess
import time

import zstandard as zstd

from ingest import iter_games, batched, parse_batch, _init_worker
from multiprocessing import Pool

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-06")
    ap.add_argument("--gb", type=float, default=3.0, help="compressed GB to pull")
    ap.add_argument("--workers", type=int, default=19)
    ap.add_argument("--total-gb", type=float, required=True, help="full archive size, GB")
    args = ap.parse_args()

    url = URL.format(args.month)
    nbytes = int(args.gb * 1e9)
    proc = subprocess.Popen(
        ["curl", "-sL", "-r", f"0-{nbytes-1}", url],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 22,
    )
    stream = io.TextIOWrapper(
        zstd.ZstdDecompressor().stream_reader(proc.stdout), encoding="utf-8", errors="replace"
    )

    t0 = time.time()
    seen = kept = plies = 0
    pool = Pool(args.workers, initializer=_init_worker, initargs=(10,))
    try:
        for results in pool.imap(parse_batch, batched(iter_games(stream), 2000), chunksize=1):
            seen += 2000
            kept += len(results)
            plies += sum(r[3] for r in results)
    except Exception as e:                       # truncated tail of the prefix
        print(f"(stopped: {type(e).__name__})")
    finally:
        pool.terminate(); pool.join()
        proc.kill()

    dt = time.time() - t0
    scale = args.total_gb / args.gb
    print(f"\n--- {args.month}, first {args.gb:g} GB of {args.total_gb:g} GB ---")
    print(f"  games seen           {seen:,}")
    print(f"  games kept           {kept:,}  ({100*kept/max(seen,1):.1f}%)")
    print(f"  plies                {plies:,}  ({plies/max(kept,1):.1f}/game)")
    print(f"  wall                 {dt:.0f}s  ->  {kept/dt:,.0f} kept-games/s")
    print(f"  shard bytes so far   {plies*2/1e6:,.0f} MB moves + {kept*28/1e6:,.0f} MB meta")
    print(f"\n--- extrapolated to the full month (x{scale:.1f}) ---")
    print(f"  games                {kept*scale/1e6:,.1f} M")
    print(f"  ingest wall          {dt*scale/3600:.2f} h")
    print(f"  shard size           {(plies*2 + kept*28)*scale/1e9:,.1f} GB")


if __name__ == "__main__":
    main()
