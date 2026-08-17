"""Hand-feature profile for every player in the gallery.

Reads `play/verifier_pack.npz` -- which already ships 4 games x 60 plies for all
558,735 gallery players -- and reduces each player to a fixed-size profile:

    feats   (P, N_FEATS) float32   mean feature vector over their games
    opens   (P, 4)       uint64    hash of each game's first 6 plies + seat
    n       (P,)         uint8     games contributing

40 MB, versus 247 MB for the games themselves and 39 GB for the raw shards.
That size difference is the whole argument for hand features: they are the only
part of a player's history cheap enough to keep for everyone.

    python build_handfeat_pack.py --out play/handfeat_pack.npz
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from handfeat import N_FEATS, game_features, opening_hash

_G = {}


def _init(d, P):
    _G["moves"] = np.memmap(f"{d}/moves.u16", np.uint16, "r").reshape(P, 4, 60)
    _G["clocks"] = np.memmap(f"{d}/clocks.u16", np.uint16, "r").reshape(P, 4, 60)
    _G["nply"] = np.memmap(f"{d}/nply.u16", np.uint16, "r").reshape(P, 4)
    _G["seat"] = np.memmap(f"{d}/seat.u8", np.uint8, "r").reshape(P, 4)
    _G["tcb"] = np.memmap(f"{d}/tcb.u16", np.uint16, "r").reshape(P, 4)
    _G["tci"] = np.memmap(f"{d}/tci.u16", np.uint16, "r").reshape(P, 4)
    _G["have"] = np.memmap(f"{d}/have.u8", np.uint8, "r")


def _chunk(rng):
    lo, hi = rng
    n = hi - lo
    F = np.zeros((n, N_FEATS), np.float32)
    O = np.zeros((n, 4), np.uint64)
    C = np.zeros(n, np.uint8)
    for i in range(lo, hi):
        rows = []
        for g in range(int(_G["have"][i])):
            k = int(_G["nply"][i, g])
            if k < 16:
                continue
            st = int(_G["seat"][i, g])
            codes = _G["moves"][i, g, :k]
            f = game_features(codes, _G["clocks"][i, g, :k], st,
                              int(_G["tcb"][i, g]), int(_G["tci"][i, g]))
            if f is None:
                continue
            rows.append(f)
            O[i - lo, len(rows) - 1] = np.uint64(opening_hash(codes, st))
        if rows:
            F[i - lo] = np.stack(rows).mean(0)
            C[i - lo] = len(rows)
    return lo, F, O, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="play/verifier_pack.npz")
    ap.add_argument("--out", default="play/handfeat_pack.npz")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    ap.add_argument("--scratch", default=os.environ.get(
        "SCRATCH", "/tmp/handfeat_scratch"))
    args = ap.parse_args()

    z = np.load(args.pack, allow_pickle=True)
    P = len(z["have"])
    d = args.scratch
    os.makedirs(d, exist_ok=True)
    # Decompress to flat files so workers can memmap instead of inheriting or
    # pickling ~540 MB of arrays.
    for key, src, dt in (("moves", "moves", np.uint16),
                         ("clocks", "clocks", np.uint16),
                         ("nply", "nply", np.uint16),
                         ("seat", "seat", np.uint8),
                         ("tcb", "tc_base", np.uint16),
                         ("tci", "tc_inc", np.uint16),
                         ("have", "have", np.uint8)):
        p = f"{d}/{key}.u{16 if dt is np.uint16 else 8}"
        if not os.path.exists(p):
            np.asarray(z[src], dt).tofile(p)
    print(f"{P:,} players | {args.workers} workers", flush=True)

    step = 2000
    chunks = [(i, min(i + step, P)) for i in range(0, P, step)]
    F = np.zeros((P, N_FEATS), np.float32)
    O = np.zeros((P, 4), np.uint64)
    C = np.zeros(P, np.uint8)
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(d, P)) as pool:
        for n, (lo, f, o, c) in enumerate(pool.imap_unordered(_chunk, chunks)):
            F[lo:lo + len(f)] = f
            O[lo:lo + len(o)] = o
            C[lo:lo + len(c)] = c
            if (n + 1) % 25 == 0:
                done = (n + 1) * step
                el = time.time() - t0
                print(f"  {done:,}/{P:,}  {el:.0f}s  "
                      f"eta {el * (len(chunks) / (n + 1) - 1):.0f}s", flush=True)

    np.savez_compressed(args.out, feats=F, opens=O, n=C, names=z["names"],
                        feat_names=np.array(
                            __import__("handfeat").NAMES))
    print(f"\nwrote {args.out}: {os.path.getsize(args.out)/1e6:.0f} MB | "
          f"{int((C > 0).sum()):,}/{P:,} players with a profile")
    print("HANDPACK_DONE")


if __name__ == "__main__":
    main()
