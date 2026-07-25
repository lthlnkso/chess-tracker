"""Header-only survey of a lichess month: which time controls dominate?

Deliberately does not parse movetext. Turning SAN into moves is ~90% of ingest
cost, and picking a time-control filter needs only the headers, so this runs
roughly an order of magnitude faster than a full pass.

    python survey.py --month 2026-06 --gb 2
"""

from __future__ import annotations

import argparse
import io
import subprocess
import time
from collections import Counter

import zstandard as zstd

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"


def bucket(tc: str) -> str:
    """lichess speed classes, by estimated duration = base + 40*increment."""
    if not tc or tc == "-":
        return "correspondence"
    try:
        base, inc = tc.split("+")
        est = int(base) + 40 * int(inc)
    except ValueError:
        return "unknown"
    if est < 29:
        return "ultraBullet"
    if est < 179:
        return "bullet"
    if est < 479:
        return "blitz"
    if est < 1499:
        return "rapid"
    return "classical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-06")
    ap.add_argument("--gb", type=float, default=2.0)
    args = ap.parse_args()

    nbytes = int(args.gb * 1e9)
    proc = subprocess.Popen(
        ["curl", "-sL", "-r", f"0-{nbytes-1}", URL.format(args.month)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 22,
    )
    stream = io.TextIOWrapper(
        zstd.ZstdDecompressor().stream_reader(proc.stdout), encoding="utf-8", errors="replace"
    )

    tcs: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    n = 0
    t0 = time.time()
    try:
        for line in stream:
            if line.startswith('[TimeControl "'):
                tc = line[14:line.index('"', 14)]
                tcs[tc] += 1
                buckets[bucket(tc)] += 1
                n += 1
    except Exception as e:
        print(f"(stopped: {type(e).__name__})")
    finally:
        proc.kill()

    dt = time.time() - t0
    print(f"\n{args.month}: {n:,} games surveyed from {args.gb:g} GB in {dt:.0f}s "
          f"({n/dt:,.0f} games/s, headers only)\n")

    print("by speed class:")
    for b, c in buckets.most_common():
        print(f"  {b:<16} {c:>10,}  {100*c/n:5.1f}%")

    print("\ntop individual time controls:")
    cum = 0
    for tc, c in tcs.most_common(12):
        cum += c
        print(f"  {tc:<12} {c:>10,}  {100*c/n:5.1f}%   (cumulative {100*cum/n:5.1f}%)")


if __name__ == "__main__":
    main()
