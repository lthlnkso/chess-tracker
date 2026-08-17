"""Label positions with Stockfish evals for every legal move -- the CPL corpus.

The loss we are building wants, for each supervised ply, how good every move the
model can choose actually was. Cross-entropy calls all 31 non-played candidates
equally wrong; this makes the target graded, so a move of the same QUALITY as the
human's is nearly a right answer and a blunder is catastrophically wrong.

Three measured decisions, from profile_cpl{,2,3}.py:

  depth 6         extra depth only sharpens the near-best band (0-30cp), and
                  label noise there is 106cp even at depth 8 -- that band is
                  noise-dominated at any affordable depth. Depth 6 keeps the
                  coarse structure (candidates span ~550cp) at 3x less cost.
  multipv 32      ONE search returns every root move's eval from a shared tree.
                  Per-candidate searches would be 6.2G searches on a shard.
                  MultiPV width is the dominant cost term, not depth.
  1 thread/engine measured 12.4 pos/s/core at Threads=1 vs 4.0 at Threads=4.
                  SMP scales badly for throughput work.

We store ABSOLUTE eval per move, not centipawn loss. Win probability is
non-linear in eval -- dropping 100cp from +50 is a real error and from +900 is
nothing -- so the training-time transform needs the absolute value. CPL is
recoverable as max(eval) - eval; the reverse is not.

Every LEGAL move is stored, not just the 32 the loader happens to draw. It is
~34 moves versus 32, and it decouples the corpus from candidate sampling: change
how candidates are drawn later and these labels still apply.

    python cpl_label.py --shard data/mt/2026-01 --out /data/cpl --players 20000
"""

from __future__ import annotations

import argparse
import atexit
import os
import time
from multiprocessing import Pool

import chess
import chess.engine
import numpy as np

from bitboards import decode_move, encode_move

MATE_CP = 2000       # mate clipped; unbounded values would swamp the loss


from cpuquota import cpu_quota  # noqa: E402  (shared; see cpuquota.py)


_ENG = {}


def _init(engine_path, depth, multipv, hash_mb, shard, outdir, _unused=None):
    """One engine per worker, plus the worker's OWN memmap and output shard.

    Nothing large crosses a process boundary. The previous version shipped a
    move array in and ~132 small numpy arrays per game back, and left 91 of 94
    engines idle -- workers block on the result queue when the parent cannot
    drain it, which looks identical to starvation. Workers now read the memmap
    themselves and append to their own files, so the pool carries only ints.
    """
    import os as _os, random as _rnd, time as _t
    # 94 simultaneous UCI handshakes time out at python-chess's 10s default --
    # measured: only ~3 of 94 engines survived, and 3 x 36 plies/s is exactly the
    # 114 plies/s the stalled run produced. Stagger the spawns, give the
    # handshake real headroom, and retry rather than leaving a dead worker that
    # silently does nothing for the rest of the run.
    _t.sleep(_rnd.random() * 3)
    err = None
    for attempt in range(5):
        try:
            e = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=120)
            e.configure({"Threads": 1, "Hash": hash_mb})
            _ENG["e"] = e
            err = None
            break
        except Exception as ex:                       # noqa: BLE001
            err = ex
            _t.sleep(3 + attempt * 5)
    if err is not None:
        raise RuntimeError(f"engine init failed after 5 tries: {err}")
    _ENG["depth"], _ENG["multipv"] = depth, multipv
    _ENG["mv"] = np.memmap(f"{shard}/moves.u16", dtype=np.uint16, mode="r")
    wid = _os.getpid()
    _ENG["wid"] = wid
    _ENG["f_mc"] = open(f"{outdir}/w{wid}.moves.u16", "wb")
    _ENG["f_ev"] = open(f"{outdir}/w{wid}.evals.i16", "wb")
    _ENG["f_ix"] = open(f"{outdir}/w{wid}.index.i64", "wb")   # gi, ply, count
    # Flushed per game, not at exit: the time cap kills workers with
    # pool.terminate(), which runs no atexit handlers, so anything still in a
    # buffer would be lost. Per-game flush costs a few KB of syscall and makes
    # an interrupted run keep everything it actually finished.
    atexit.register(_flush)


def _flush():
    for k in ("f_mc", "f_ev", "f_ix"):
        try:
            _ENG[k].flush(); _ENG[k].close()
        except Exception:
            pass


def _label_game(task):
    """All plies of one game, written straight to this worker's own files.

    Returns only a count, so the parent's result loop is trivial and can never
    become the thing that stalls the engines.

    Plies are walked in order and share one `game` token so Stockfish keeps its
    transposition table warm across them -- measured 1.35x on consecutive plies.
    """
    gi, off, npl = task
    codes = np.asarray(_ENG["mv"][off:off + npl])
    eng, depth, mpv = _ENG["e"], _ENG["depth"], _ENG["multipv"]
    tok = (gi,)
    b = chess.Board()
    out = []
    for t, c in enumerate(codes):
        mv = decode_move(int(c))
        if b.is_game_over() or mv not in b.legal_moves:
            break
        n_legal = b.legal_moves.count()
        if n_legal < 2:                      # forced move: nothing to choose
            out.append((t, np.zeros(0, np.uint16), np.zeros(0, np.int16)))
            b.push(mv)
            continue
        try:
            info = eng.analyse(b, chess.engine.Limit(depth=depth),
                               multipv=min(mpv, n_legal), game=tok)
        except chess.engine.EngineError:
            break
        mc, ev = [], []
        for e in info:
            pv = e.get("pv")
            if not pv:
                continue
            sc = e["score"].pov(b.turn).score(mate_score=MATE_CP)
            mc.append(encode_move(pv[0]))
            ev.append(int(np.clip(sc, -MATE_CP, MATE_CP)))
        if mc:
            out.append((t, np.array(mc, np.uint16), np.array(ev, np.int16)))
        b.push(mv)
    n = 0
    for t, mc, ev in out:
        if len(mc) == 0:
            continue
        _ENG["f_mc"].write(mc.tobytes())
        _ENG["f_ev"].write(ev.tobytes())
        _ENG["f_ix"].write(np.array([gi, t, len(mc)], np.int64).tobytes())
        n += len(mc)
    for k in ("f_mc", "f_ev", "f_ix"):
        _ENG[k].flush()
    return n


def pick_games(shard, n_players, min_games, seed, per_player=20):
    """Whole games of a player subset, so any ply the loader samples is labelled.

    Labelling a scatter of plies would leave the loader picking unlabelled ones;
    labelling every ply of chosen players' games means the CPL arm can sample as
    freely as the baseline does.
    """
    meta = np.load(f"{shard}/meta.npy", mmap_mode="r")
    ck = np.memmap(f"{shard}/clocks.u16", dtype=np.uint16, mode="r")
    off = np.asarray(meta["offset"], np.int64)
    ok = np.asarray(ck[off]) != 0xFFFF
    pid = np.concatenate([np.asarray(meta["white_pid"]),
                          np.asarray(meta["black_pid"])])
    gid = np.concatenate([np.arange(len(meta))] * 2)
    keep = np.concatenate([ok] * 2)
    pid, gid = pid[keep], gid[keep]
    o = np.argsort(pid, kind="stable")
    pid, gid = pid[o], gid[o]
    b = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    sizes = np.diff(b)
    elig = np.flatnonzero(sizes >= min_games)
    rng = np.random.default_rng(seed)
    rng.shuffle(elig)
    chosen = elig[:n_players]
    # Cap per player. The full shard has ~154 games for an eligible player, so
    # 30k players is 4.6M games -- far beyond any time cap. Taking a slice of
    # that in game order would give every player a thin random sample; capping
    # per player gives a bounded corpus with EVEN coverage, which is what lets
    # the loader form bundles from a player's labelled games.
    rng2 = np.random.default_rng(seed + 1)
    picks = []
    for i in chosen:
        g = gid[b[i]:b[i + 1]]
        picks.append(g if len(g) <= per_player
                     else rng2.choice(g, per_player, replace=False))
    games = np.unique(np.concatenate(picks)) if picks else np.zeros(0, np.int64)
    return games, len(chosen)


def _merge(outdir, out):
    """Fold the per-worker shards into one corpus, then drop the shards."""
    import glob
    idx_files = sorted(glob.glob(f"{outdir}/w*.index.i64"))
    g_l, p_l, c_l = [], [], []
    with open(f"{out}/moves.u16", "wb") as fm, open(f"{out}/evals.i16", "wb") as fe:
        for ix in idx_files:
            w = ix[:-len(".index.i64")]
            tri = np.fromfile(ix, np.int64).reshape(-1, 3)
            if not len(tri):
                continue
            g_l.append(tri[:, 0]); p_l.append(tri[:, 1]); c_l.append(tri[:, 2])
            with open(f"{w}.moves.u16", "rb") as f:
                fm.write(f.read())
            with open(f"{w}.evals.i16", "rb") as f:
                fe.write(f.read())
    if not g_l:
        return 0, 0
    counts = np.concatenate(c_l).astype(np.int32)
    np.save(f"{out}/ply_game.npy", np.concatenate(g_l).astype(np.int64))
    np.save(f"{out}/ply_idx.npy", np.concatenate(p_l).astype(np.int32))
    np.save(f"{out}/offsets.npy", np.r_[0, np.cumsum(counts)].astype(np.int64))
    for f in glob.glob(f"{outdir}/w*"):
        os.remove(f)
    return len(counts), int(counts.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--players", type=int, default=20000)
    ap.add_argument("--min-games", type=int, default=7)
    ap.add_argument("--games-per-player", type=int, default=20,
                    help="cap per player; even coverage beats a thin slice of a "
                         "corpus the time cap cannot finish")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--multipv", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = the cgroup CPU quota, NOT nproc")
    ap.add_argument("--hash-mb", type=int, default=16,
                    help="per engine. 94 x 64MB all allocating during "
                         "the UCI handshake is what tipped it over")
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--bench", type=int, default=0,
                    help="if >0, time this many games at 1 worker and at "
                         "--workers, then exit. Says whether a throughput "
                         "shortfall is the hardware or the architecture.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.workers <= 0:
        args.workers = max(1, cpu_quota())
        print(f"workers = {args.workers} (cgroup quota; nproc says "
              f"{os.cpu_count()})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    scratch = os.path.join(args.out, "_w")
    os.makedirs(scratch, exist_ok=True)
    meta = np.load(f"{args.shard}/meta.npy", mmap_mode="r")
    games, n_pl = pick_games(args.shard, args.players, args.min_games,
                             args.seed, args.games_per_player)
    off = np.asarray(meta["offset"], np.int64)
    npl = np.asarray(meta["nply"], np.int64)
    tasks = [(int(g), int(off[g]), int(npl[g])) for g in games]
    print(f"{n_pl:,} players -> {len(tasks):,} games "
          f"(~{len(tasks)*66/1e6:.1f}M plies) | depth {args.depth} "
          f"multipv {args.multipv} | {args.workers} engines", flush=True)

    init = (args.engine, args.depth, args.multipv, args.hash_mb,
            args.shard, scratch, None)

    if args.bench:
        # One worker, then many, on the SAME games. If per-worker throughput
        # collapses at scale the machine is the limit (memory bandwidth, shared
        # vCPU); if it holds, the earlier stall was the result pipe.
        sample = tasks[:args.bench]
        for w in (1, args.workers):
            for f in os.listdir(scratch):
                os.remove(os.path.join(scratch, f))
            t0 = time.time()
            with Pool(w, initializer=_init, initargs=init) as pool:
                n = sum(pool.map(_label_game, sample, chunksize=1))
            dt = time.time() - t0
            print(f"  bench {w:>3} workers: {len(sample)} games, {dt:.1f}s | "
                  f"{n/dt:.0f} evals/s | {len(sample)/dt:.2f} games/s | "
                  f"{len(sample)/dt/w:.3f} games/s/worker", flush=True)
        print("CPL_BENCH_DONE")
        return

    done = 0
    t0 = time.time()
    deadline = t0 + args.max_hours * 3600
    with Pool(args.workers, initializer=_init, initargs=init) as pool:
        for _ in pool.imap_unordered(_label_game, tasks, chunksize=8):
            done += 1
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"  {done:,}/{len(tasks):,} games | {done/el:.1f} games/s "
                      f"| {el/60:.1f} min | eta {(len(tasks)-done)/max(done/el,1e-9)/60:.0f} min",
                      flush=True)
            if time.time() > deadline:
                print("  max-hours reached; keeping what finished", flush=True)
                pool.terminate()
                break
    # Workers only flush on clean exit; terminate() skips that, so the tail of
    # an interrupted worker's buffer is lost. That is a few games, not a corpus.
    time.sleep(2)

    n_plies, n_evals = _merge(scratch, args.out)
    if not n_plies:
        print("CPL_LABEL_EMPTY"); return
    import json
    with open(f"{args.out}/manifest.json", "w") as f:
        json.dump({"shard": args.shard, "depth": args.depth,
                   "multipv": args.multipv, "plies": n_plies,
                   "move_evals": n_evals, "games_done": done,
                   "players": n_pl, "mate_cp": MATE_CP,
                   "stores": "absolute eval, mover POV"}, f, indent=2)
    os.rmdir(scratch) if not os.listdir(scratch) else None
    mb = sum(os.path.getsize(f"{args.out}/{f}") for f in os.listdir(args.out)
             if os.path.isfile(f"{args.out}/{f}")) / 1e6
    print(f"\nwrote {args.out}: {n_plies:,} plies, {n_evals:,} move-evals, {mb:.0f} MB")
    print("CPL_LABEL_DONE")


if __name__ == "__main__":
    main()
