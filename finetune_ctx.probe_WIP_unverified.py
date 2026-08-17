"""SupCon fine-tune for multi-game-context models.

`finetune_mt.py` cannot do this job: it feeds the single-game `MultiTaskDataset`
and never passes `game_slot`/`ply_pos`, so a context model fine-tuned with it
would silently lose the per-game position encoding it was pre-trained with.

Two things differ from the single-game fine-tune:

- **Positives come for free.** A sample is "a random subset of player P's games",
  so drawing the same player index twice yields two *different* game subsets of
  the same player -- a genuine positive pair, not the same row twice.
- **The split is by player and is written into the checkpoint.** Pre-training
  and eval previously each re-derived the held-out set from a shared seed, which
  only stays correct while every filtering step upstream matches exactly. Here
  the held-out player ids are persisted, so the eval cannot drift onto players
  the contrastive head has already memorised.
- **Stopping keys on identification, not on validation loss.** The loss is a
  proxy and it saturates hours before the metric it proxies for: the ctx5 run
  stopped at 2.6h with a flat val curve while recall was still climbing
  +0.0041/hr at hour 9. A periodic probe runs identify_eval_ctx.py's own
  joint/matched protocol, at the same k and the same gallery size, on a frozen
  slice of the held-out players. Val loss is still computed and logged.

  Three things about that decision are deliberate and were each a bug first:

  - It reads **MRR**, not recall@1. recall@1 is a mean of Bernoullis, so a query
    flipping between rank 1 and 2 moves it by a full 1/N for an embedding change
    of nothing. recall@1 is still the headline number in every log line.
  - Both sides of the comparison are **trailing means** over `--probe-trail`
    probes. Against a running per-probe maximum the bar climbs on its own -- at a
    true plateau the max of n draws sits sigma*sqrt(2 ln n) above the mean, which
    at sigma=0.007 is +0.008 by probe 5 and +0.013 by probe 20 -- so patience got
    harder to satisfy the longer the run went.
  - **last.pt is the last step, never the best one.** A patience stop happens by
    construction >= patience probes *after* the argmax, so the argmax is an
    earlier, less-trained model chosen on the same noise used to declare it a
    maximum. The peak is recorded in history.json as a diagnostic and nothing
    else.

    python finetune_ctx.py --ckpt ckpt/ctx3_pre/last.pt --shard data/mt/2026-01 \
        --out ckpt/ctx3_ft --max-hours 1.0
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from fastboard import encode_batch
from multigame_data import MultiGameDataset, collate_multigame
from model import MultiTaskModel, Config, supcon_loss, N_ELO_BINS
from contrastive import ALL_LOSSES, make_loss, needs_proxies, default_pk
from timefeat import N_TIME_FEATS, N_TIME_BINS
from train_multigame import ply_positions, to_dev


class PKPlayers(Sampler):
    """Each batch is P players x K draws, every draw a different game subset."""

    def __init__(self, pool, p=24, k=4, batches=10**9, seed=0):
        self.pool = np.asarray(pool)
        self.p, self.k, self.batches = p, k, batches
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.batches):
            pick = self.rng.choice(len(self.pool), size=self.p, replace=False)
            yield [int(self.pool[i]) for i in pick for _ in range(self.k)]

    def __len__(self):
        return self.batches


@contextlib.contextmanager
def fp32_matmul():
    """Turn TF32 off for the duration, because the final eval never turns it on.

    Training wants TF32; identify_eval_ctx.py is a separate process that leaves
    the default alone. A TF32 matmul keeps 10 mantissa bits, so every gallery dot
    product carries ~1e-3 of relative error -- the same order as the +0.002 per
    probe the probe exists to resolve, and enough to make the probe measure a
    slightly different model from the one that gets scored at the end.
    """
    was = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = was


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.05 * base + 0.95 * base * 0.5 * (1 + math.cos(math.pi * p))


class ProbeGames(Dataset):
    """One (game, seat) -> the raw bitboards the probe needs.

    Exists only so the one-time probe corpus can be built by worker processes;
    replaying ~10k games through python-chess in-process costs a minute, and
    doing it inside the training loop would contend with the loader's workers
    for the rest of the run. Bitboards, not planes: 83 B/ply against 832, and
    encode_batch runs per probe batch instead.

    Size it before raising --probe-players. Measured on data/2026-06-big at
    --max-len-per-game 160: 5.5 KB per game average (13 KB for a game that hits
    the 160-ply cap; the mean game is ~65 plies), 15 games per player at the
    default G=12/k=5, so 83 KB per probe player -- 332 MB at 4000 players and
    ~1.25 GB at the default 15000. That is resident in the parent process for
    the whole run, on top of the model and the loader's workers.
    """

    def __init__(self, ds, picks):
        self.ds, self.picks = ds, picks

    def __len__(self):
        return len(self.picks)

    def __getitem__(self, i):
        gi, seat = self.picks[i]
        # _one_game's rng drives only the supervised-ply and candidate draws,
        # both of which the probe discards, so the fields kept here are a
        # deterministic function of (gi, seat) whatever generator is passed --
        # which is what makes a frozen probe set actually frozen.
        (pos, pos_r, _c, _cr, _ch, _l, _n, _T, fe, _tt, _tv, mt, _e) = \
            self.ds._one_game(int(gi), int(seat), np.random.default_rng(0))
        return pos, pos_r, np.ascontiguousarray(fe), mt


def probe_pack(rows, n_planes, n_slots, with_rights, device):
    """Right-pad game bundles into one batch, in identify_eval_ctx's layout.

    Batching is exact rather than approximate: attention is causal with no
    key-padding mask (model.Block), so a real position at t only ever attends to
    <= t, all real. Everything pooling touches is masked -- pad_mask marks the
    pad, my_turn is False there -- and game_slot/ply_pos are 0 in the pad region
    because their embeddings are averaged out by that same mask.
    """
    B = len(rows)
    T = max(sum(g[0].shape[0] for g in r) for r in rows)
    snaps = np.concatenate([g[0] for r in rows for g in r])
    rgts = np.concatenate([g[1] for r in rows for g in r]) if with_rights else None
    enc = encode_batch(snaps, n_planes, rgts)
    planes = np.zeros((B, T, n_planes, 8, 8), np.uint8)
    extra = np.zeros((B, T, N_TIME_FEATS), np.float32)
    pad = np.ones((B, T), bool)
    mine = np.zeros((B, T), bool)
    slot = np.zeros((B, T), np.int64)
    ppos = np.zeros((B, T), np.int64)
    at = 0
    for i, r in enumerate(rows):
        o = 0
        for s, (pos, _pr, fe, mt) in enumerate(r):
            t = pos.shape[0]
            planes[i, o:o + t] = enc[at:at + t]
            extra[i, o:o + t] = fe
            mine[i, o:o + t] = mt
            slot[i, o:o + t] = min(s, n_slots - 1)
            # Ply index restarts per game. Without this the concatenation runs
            # past cfg.max_len and encode() clamps almost every position onto
            # the last position embedding.
            ppos[i, o:o + t] = np.arange(t)
            at += t
            o += t
        pad[i, :o] = False
    dev = lambda x: torch.as_tensor(x).to(device)
    return dev(planes), dev(extra), dev(pad), dev(mine), dev(slot), dev(ppos)


def probe_metrics(Q, C, QL, CL):
    """Score exactly as identify_eval_ctx.py does, plus two smooth statistics."""
    N = len(CL)
    # An all-False row makes argmax return 0, so a query whose own centroid is
    # missing would be ranked against an arbitrary other player's.
    assert bool(torch.isin(QL, CL).all()), "query player missing from the gallery"
    sim = Q @ C.T
    top = sim.topk(min(100, N), 1).indices
    m = CL[top] == QL[:, None]
    tc = (CL[None, :] == QL[:, None]).float().argmax(1)
    rank = (sim > sim.gather(1, tc[:, None])).sum(1).float()
    # recall@j is identically 1 once j >= N, so only report what this gallery
    # can actually resolve -- a probe gallery is far smaller than the eval's.
    r = {f"recall@{j}": float(m[:, :j].any(1).float().mean())
         for j in (1, 10, 100) if j < N}
    # recall@1 is a mean of Bernoullis and its SE at 15k queries (~0.004) is still
    # the same order as the +0.002/probe slope being looked for; the fixed gallery
    # pairs the probes so deltas are tighter than that, but mean normalised rank
    # and MRR use every query's whole ranking and are the statistics to read for
    # trend -- mrr is what --probe-patience actually keys on. They are also the
    # only ones comparable across gallery sizes.
    r.update(median_rank=float(rank.median()),
             mean_norm_rank=float((rank / N).mean()),
             mrr=float((1.0 / (rank + 1)).mean()),
             gallery=N, queries=int(len(QL)))
    r["chance@1"] = 1.0 / N
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--from-scratch", action="store_true",
                    help="random init instead of a pre-trained trunk -- the "
                         "control for whether pre-training earns its compute")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-embed", type=int, default=128)
    ap.add_argument("--max-games", type=int, default=0,
                    help="from-scratch only; 1 = single-game")
    ap.add_argument("--max-len-per-game", type=int, default=160)
    ap.add_argument("--eval-every", type=int, default=5000,
                    help="steps between held-out SupCon evaluations")
    ap.add_argument("--eval-batches", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4,
                    help="accepted for CLI compatibility and recorded, but val "
                         "loss no longer stops the run -- see --probe-patience")
    ap.add_argument("--probe-every-hours", type=float, default=0.5,
                    help="identification probe cadence, in hours of TRAINING "
                         "time (probe time is excluded); 0 disables the probe "
                         "and with it the only stopping signal")
    ap.add_argument("--probe-players", type=int, default=15000,
                    help="probe gallery size. 1500 was budgeted against a "
                         "2-4 MINUTE probe; the probe measures 2-4 SECONDS, so "
                         "the old default bought nothing and its sampling noise "
                         "was the dominant term in the stopping decision. 15000 "
                         "is also within a factor of ~1.3 of the 20k-player final "
                         "eval, so the level is roughly comparable, not just the "
                         "shape. Costs ~1.25 GB of resident corpus and a probe "
                         "that scales linearly in this number -- see ProbeGames")
    ap.add_argument("--probe-centroid-games", type=int, default=12,
                    help="gallery games per player; 12 = identify_eval_ctx's own "
                         "--gallery-games default, which is what makes the probe "
                         "and the eval the same estimator. --gallery-mode matched "
                         "cuts them into non-overlapping k-game chunks and drops "
                         "the remainder, so 12 with --probe-k 5 gives 2 chunks and "
                         "drops 2 games -- identical to what the eval does at k=5")
    ap.add_argument("--probe-k", type=int, default=5,
                    help="games per joint query; clamped to the slot count. 5 so "
                         "every game_emb slot of a ctx5 model is exercised -- "
                         "5-game context is the premise of the branch, and a k=3 "
                         "probe never touches slots 3 and 4")
    ap.add_argument("--probe-min-games", type=int, default=17,
                    help="minimum clocked games for a probe player. 17 = the "
                         "eval's --gallery-games 12 + max(--ks) 5, so the probe "
                         "scores the same population the eval will; filtering at "
                         "the probe's own G+k instead would compare two different "
                         "populations. Raise it if the eval will use a larger k")
    ap.add_argument("--probe-patience", type=int, default=6,
                    help="stop after this many probes with no gain on the "
                         "trailing-mean MRR (see --probe-trail)")
    ap.add_argument("--probe-trail", type=int, default=3,
                    help="probes averaged on BOTH sides of the patience test. A "
                         "running per-probe maximum ratchets its own bar up by "
                         "sigma*sqrt(2 ln n) at a flat plateau; two means do not")
    ap.add_argument("--probe-min-delta", type=float, default=0.001,
                    help="gain required on the trailing-mean MRR, not on recall@1")
    ap.add_argument("--probe-min-hours", type=float, default=0.0,
                    help="floor, in training hours, before patience may fire; "
                         "the prior run was still gaining +0.0041/hr at hour 9. "
                         "The counter is RESET below the floor, not merely held: "
                         "deferring it turns the floor into a fixed wall-clock cut")
    ap.add_argument("--probe-batch", type=int, default=64)
    ap.add_argument("--probe-seed", type=int, default=4242)
    ap.add_argument("--probe-set", default="",
                    help="npz holding the frozen probe set; written on first use "
                         "and reloaded after, so a watchdog restart measures the "
                         "same players (default <out>/probe_set.npz)")
    ap.add_argument("--probe-holdout", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="drop probe players from test_pids (default ON). "
                         "--no-probe-holdout keeps them, which shrinks nothing but "
                         "selects the stop step on players that stay in the final "
                         "eval -- the eval then reports a number the stopping rule "
                         "already optimised against. probe_pids is written into "
                         "every checkpoint either way, so an eval can exclude them "
                         "after the fact")
    ap.add_argument("--probe-selftest", action="store_true",
                    help="run the acceptance checks (determinism, batching, "
                         "positive control) before training")
    ap.add_argument("--collapse-step", type=int, default=2000,
                    help="from random init the loss legitimately sits near the "
                         "ceiling for a while; check later than a fine-tune would")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=100_000_000)
    ap.add_argument("--p", type=int, default=24)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--min-games", type=int, default=0,
                    help="0 = use the checkpoint's game count")
    ap.add_argument("--loss", default="ms", choices=ALL_LOSSES,
                    help="ms by default, not supcon: measured +52%% on an "
                         "identical trunk and budget, replicated across two seeds")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast; carries most of the fine-tune speedup "
                         "but is the one change that is not numerically exact")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # trail[-0:] is the whole list, so a 0 here would silently average every probe
    # ever taken instead of the last few, and patience would never fire.
    if args.probe_trail < 1:
        raise SystemExit("--probe-trail must be >= 1")

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.from_scratch:
        n_slots = max(1, args.max_games)
        mlpg = args.max_len_per_game
        cfg = Config(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                     d_ff=args.d_model * 4, max_len=mlpg + 8, d_embed=args.d_embed)
        ck = None
        print(f"FROM SCRATCH: {n_slots} game slots, {mlpg} plies/game", flush=True)
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt is required unless --from-scratch")
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = Config(**ck["cfg"])
        n_slots = ck.get("n_game_slots", 1)
        mlpg = ck.get("max_len_per_game", cfg.max_len - 8)
        print(f"ckpt step {ck['step']}, {n_slots} game slots, {mlpg} plies/game", flush=True)

    # Enough games to build a query AND a disjoint gallery at eval time.
    need = args.min_games or (n_slots + 2)
    ds = MultiGameDataset(args.shard, max_games=n_slots, max_len_per_game=mlpg,
                          plies_per_game=4, n_cand=4, min_games=need, seed=args.seed)
    print(f"{len(ds):,} players with >= {need} clocked games", flush=True)

    rng = np.random.default_rng(args.seed)
    n_test = max(1, int(len(ds) * args.test_frac))
    test_idx = rng.choice(len(ds), n_test, replace=False)
    is_test = np.zeros(len(ds), bool)
    is_test[test_idx] = True
    train_pool = np.flatnonzero(~is_test)
    test_pids = ds.gpid[is_test]
    print(f"train {len(train_pool):,} players | held out {int(is_test.sum()):,}", flush=True)

    n_planes = ds.n_planes if ck is None else ck["n_planes"]
    n_extra = N_TIME_FEATS if ck is None else ck["n_extra"]
    d_embed = args.d_embed if ck is None else ck["d_embed"]
    model = MultiTaskModel(cfg, n_planes=n_planes, n_extra=n_extra,
                           d_embed=d_embed, n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS, n_game_slots=n_slots).to(device)
    if ck is not None:
        model.load_state_dict(ck["model"])
    print(f"model {sum(q.numel() for q in model.parameters())/1e6:.2f}M params",
          flush=True)
    model.train()

    pk = PKPlayers(train_pool, p=args.p, k=args.k, batches=args.steps, seed=args.seed)
    dl = DataLoader(ds, batch_sampler=pk, num_workers=args.workers,
                    collate_fn=collate_multigame, pin_memory=device == "cuda")
    # Held-out players. The sample is NOT actually fixed across evaluations, as
    # this comment used to claim: PKPlayers.__iter__ resumes its generator, and
    # MultiGameDataset.__getitem__ reseeds from OS entropy and redraws both the
    # game count and the games. So the val curve carries sample noise on top of
    # model change -- part of why it looked flat. Logging only now; the probe
    # below is the one that had to be built without that defect.
    val_pool = np.flatnonzero(is_test)
    val_pk = PKPlayers(val_pool, p=args.p, k=args.k,
                       batches=args.eval_batches, seed=12345)
    val_dl = DataLoader(ds, batch_sampler=val_pk, num_workers=max(4, args.workers // 4),
                        collate_fn=collate_multigame, pin_memory=device == "cuda")

    @torch.no_grad()
    def validate():
        model.eval()
        tot, n = 0.0, 0
        for vb in val_dl:
            vb = to_dev(vb, device)
            vpp = ply_positions(vb["game_slot"], vb["pad_mask"])
            ve, _ = model.embed(vb["planes"], vb["extra"], vb["pad_mask"],
                                vb["my_turn"], vb["game_slot"], vpp)
            vl, _ = loss_fn(ve.float(), vb["player_id"])
            tot += float(vl); n += 1
        model.train()
        return tot / max(n, 1)
    loss_fn = make_loss(args.loss).to(device)
    if needs_proxies(args.loss):
        raise SystemExit("proxy losses need a class bank keyed on train players; "
                         "they also collapsed in every configuration tested here")
    # ---- identification probe -------------------------------------------
    # The stopping signal. Built here, once, and frozen: fixing the players, the
    # gallery/query split and the game picks turns the gallery's sampling error
    # into a constant bias instead of per-probe noise, which is what lets a
    # cheap probe resolve a slope of +0.002 per half hour at all.
    #
    # Captured before --compile rebinds model.embed: probe shapes are a third
    # shape family (after train and validate) and dynamo drops a code object to
    # eager for good after 8 recompiles. The probe is a rounding error of the
    # compute, so run it uncompiled and decouple it from the training graph.
    probe_embed = model.embed
    run_probe, probe_fp, probe_pids, probe_proto = None, "", None, None
    probe_every = args.probe_every_hours * 3600
    if probe_every > 0:
        G, PK = args.probe_centroid_games, min(args.probe_k, n_slots)
        if PK != args.probe_k:
            # Past n_slots two games share the last slot embedding, ply_positions
            # then reads them as one run and numbers plies straight through, and
            # encode() clamps everything over cfg.max_len onto one position.
            # identify_eval_ctx skips joint for k > n_slots for the same reason.
            print(f"probe k {args.probe_k} > {n_slots} slots; using k={PK}", flush=True)
        if ds.n_planes != n_planes:
            raise SystemExit(f"probe needs matching plane counts: dataset "
                             f"{ds.n_planes}, checkpoint {n_planes}")
        # Its own generator, never the module-level `rng`: that one has already
        # been advanced by the split draw, so sharing it would make the probe set
        # silently depend on every earlier RNG call.
        prng = np.random.default_rng(args.probe_seed)
        pset = args.probe_set or os.path.join(args.out, "probe_set.npz")
        # np.savez appends .npz unconditionally. Without this, --probe-set foo
        # writes foo.npz and then tests os.path.exists("foo") forever, so every
        # watchdog restart silently rebuilt a *different* frozen set -- which is
        # the one thing the frozen set exists to prevent.
        if not pset.endswith(".npz"):
            pset += ".npz"
        # The eval's population, not the probe's own. identify_eval_ctx keeps
        # players with --gallery-games + max(--ks) games, 12 + 5 = 17 in
        # ctx5_run.sh; filtering here at G+k would admit players the eval never
        # sees, and how many games a player has is itself correlated with how
        # identifiable they are.
        need_probe = max(G + PK, args.probe_min_games)

        if os.path.exists(pset):
            # A watchdog restart re-derives the split from --seed, which only
            # reproduces while len(ds) is unchanged. Reloading the frozen set
            # makes the curve comparable across restarts *and* across runs.
            z = np.load(pset)
            if int(z["G"]) != G or int(z["k"]) != PK:
                raise SystemExit(f"{pset} holds G={int(z['G'])} k={int(z['k'])}, "
                                 f"asked for G={G} k={PK}")
            # The picks are (game_id, seat) into this shard's meta; against a
            # different shard they address unrelated games.
            if str(z["shard"]) != args.shard:
                raise SystemExit(f"{pset} was built on shard {z['shard']}")
            pids = [int(x) for x in z["pids"]]
            gal_picks = [[(int(a), int(b)) for a, b in zip(g, s)]
                         for g, s in zip(z["gal_gid"], z["gal_seat"])]
            qry_picks = [[(int(a), int(b)) for a, b in zip(g, s)]
                         for g, s in zip(z["qry_gid"], z["qry_seat"])]
            print(f"probe set reloaded from {pset}", flush=True)
        else:
            # Held-out dataset indices only, and by index rather than by
            # re-deriving the grouping from meta -- a MS fine-tune memorises its
            # train players, so one train player in the gallery would produce a
            # fast, clean, entirely fake improvement curve.
            cand = [int(i) for i in val_pool if len(ds.groups[i][0]) >= need_probe]
            if len(cand) < 32:
                raise SystemExit(f"only {len(cand)} held-out players with >= "
                                 f"{need_probe} clocked games; lower "
                                 f"--probe-min-games/--probe-centroid-games/"
                                 f"--probe-k or pass --probe-every-hours 0")
            take = prng.choice(len(cand), min(args.probe_players, len(cand)),
                               replace=False)
            pids, gal_picks, qry_picks = [], [], []
            for i in sorted(int(cand[j]) for j in take):
                g, s = ds.groups[i]
                # One permutation, split in two: gallery and query games cannot
                # overlap by construction.
                perm = prng.permutation(len(g))
                pids.append(int(ds.gpid[i]))
                gal_picks.append([(int(g[j]), int(s[j])) for j in perm[:G]])
                qry_picks.append([(int(g[j]), int(s[j])) for j in perm[G:G + PK]])
            np.savez(pset, pids=np.asarray(pids, np.int64),
                     gal_gid=np.asarray(gal_picks, np.int64)[:, :, 0],
                     gal_seat=np.asarray(gal_picks, np.int64)[:, :, 1],
                     qry_gid=np.asarray(qry_picks, np.int64)[:, :, 0],
                     qry_seat=np.asarray(qry_picks, np.int64)[:, :, 1],
                     G=G, k=PK, mlpg=mlpg, seed=args.probe_seed, shard=args.shard)

        probe_pids = np.asarray(pids, np.int64)
        train_pid = set(int(x) for x in ds.gpid[~is_test])
        n_leak = len(set(pids) & train_pid)
        n_dup = sum(len(set(a) & set(b)) for a, b in zip(gal_picks, qry_picks))
        print(f"probe: {len(pids):,} players (>= {need_probe} clocked games, the "
              f"eval's own filter) | train overlap {n_leak} | "
              f"gallery/query overlap {n_dup}", flush=True)
        assert n_leak == 0, "probe players leaked from the train pool"
        assert n_dup == 0, "a player's centroid and query games overlap"
        assert len(set(pids)) == len(pids), "duplicate probe players"
        probe_fp = hashlib.sha1(np.concatenate([
            probe_pids, np.asarray(gal_picks, np.int64).ravel(),
            np.asarray(qry_picks, np.int64).ravel()]).tobytes()).hexdigest()[:12]

        if args.probe_holdout:
            test_pids = test_pids[~np.isin(test_pids, probe_pids)]
            print(f"  probe players carved out of test_pids -> "
                  f"{len(test_pids):,} eval players", flush=True)
            if len(test_pids) < len(probe_pids):
                print(f"  !! only {len(test_pids):,} players left for the final "
                      f"eval after carving out {len(probe_pids):,} probe players "
                      f"-- lower --probe-players or raise --test-frac", flush=True)
        else:
            print("  NOTE: --no-probe-holdout -- probe players stay inside "
                  "test_pids, so the stop step is chosen on part of the final "
                  "eval set. probe_pids is written to every checkpoint; the eval "
                  "MUST exclude them or it is reporting its own selection.",
                  flush=True)

        # --gallery-mode matched: non-overlapping k-game chunks, remainder
        # dropped, exactly as identify_eval_ctx.build_gallery does it. A joint
        # k-game query scored against single-game centroids is the artifact that
        # invalidated the earlier "averaging beats joint" result.
        #
        # G=12, k=5 -> chunks at 0 and 5, 2 chunks, games 10 and 11 dropped. That
        # remainder is not waste to be tuned away: it is what the eval does, and
        # the point is that the probe averages the same NUMBER of chunks the eval
        # averages. The old G=5,k=3 default produced exactly one chunk, so the
        # .mean(1) below was a no-op and the probe measured a single 3-game
        # embedding while the eval measured a mean of four.
        chunks = [(j, j + PK) for j in range(0, G - PK + 1, PK)] or [(0, G)]
        used = chunks[-1][1]
        want = []
        for gp, qp in zip(gal_picks, qry_picks):
            want.extend(gp[:used])
            want.extend(qp)
        tb = time.time()
        cl = DataLoader(ProbeGames(ds, want), batch_size=64, shuffle=False,
                        num_workers=min(args.workers, 16), collate_fn=list)
        blocks = [b for part in cl for b in part]
        del cl
        cache = dict(zip(want, blocks))
        mb = sum(sum(a.nbytes for a in b) for b in blocks) / 1e6
        gal_rows = [[cache[p] for p in gp[a:b]] for gp in gal_picks for a, b in chunks]
        qry_rows = [[cache[p] for p in qp] for qp in qry_picks]
        # Cap the probe batch at the training batch's token count so a probe can
        # never need memory the steady state has not already allocated -- an OOM
        # from fragmentation would otherwise land at hour 6, not hour 0.
        probe_batch = max(1, min(args.probe_batch,
                                 (args.p * args.k * n_slots) // max(PK, 1)))
        print(f"probe corpus: {len(want):,} games, {mb:.0f} MB, built in "
              f"{time.time() - tb:.0f}s | {len(chunks)} x {PK}-game centroid "
              f"chunks of {G} | batch {probe_batch} | fp {probe_fp}", flush=True)
        probe_proto = {"players": len(pids), "k": PK, "gallery_games": G,
                       "centroid_chunks": len(chunks), "gallery_mode": "matched",
                       "mode": "joint", "mlpg": mlpg, "fingerprint": probe_fp,
                       "seed": args.probe_seed, "set_file": pset,
                       "min_games": need_probe, "holdout": bool(args.probe_holdout),
                       "corpus_mb": round(mb, 1), "corpus_games": len(want),
                       "stop_statistic": "mrr", "stop_trail": args.probe_trail,
                       "probe_pids": [int(x) for x in probe_pids]}

        CL = torch.tensor(pids, device=device)

        def embed_rows(rows):
            out = []
            for s in range(0, len(rows), probe_batch):
                p = probe_pack(rows[s:s + probe_batch], n_planes, n_slots,
                               ds.with_rights, device)
                out.append(probe_embed(*p)[0].float())
            return torch.cat(out)

        @torch.no_grad()
        def run_probe(qrows=None):
            assert probe_fp == hashlib.sha1(np.concatenate([
                probe_pids, np.asarray(gal_picks, np.int64).ravel(),
                np.asarray(qry_picks, np.int64).ravel()]).tobytes()).hexdigest()[:12]
            was = model.training
            cpu_rng = torch.get_rng_state()
            gpu_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            model.eval()
            try:
                assert not model.training
                # fp32, matching identify_eval_ctx, which has no autocast: bf16
                # quantises the 20-bin Elo softmax that is concatenated straight
                # into the embedding input. Explicit rather than incidental, so
                # the probe stays fp32 if it is ever moved inside the autocast.
                # fp32_matmul because "no autocast" is not enough -- TF32 is a
                # global backend flag this file sets to True for training.
                with fp32_matmul(), torch.autocast("cuda", enabled=False):
                    v = embed_rows(gal_rows).view(len(pids), len(chunks), -1).mean(1)
                    C = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    q = embed_rows(qry_rows if qrows is None else qrows)
                    Q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                return probe_metrics(Q, C, CL, CL)
            finally:
                # In `finally` because a probe that raises must not leave the
                # remaining 20 hours training with dropout disabled -- the loss
                # would drop and the log would look better, not worse.
                if was:
                    model.train()
                # eval() fires no dropout so nothing should have been consumed;
                # restoring anyway keeps the training stream bit-identical to a
                # run with no probe.
                torch.set_rng_state(cpu_rng)
                if gpu_rng is not None:
                    torch.cuda.set_rng_state_all(gpu_rng)

        def probe_selftest():
            p = probe_pack(qry_rows[:4], n_planes, n_slots, ds.with_rights, device)
            # Cheapest possible proof that the probe is on the joint protocol:
            # the hand-built per-game ply index equals the trainer's run-length
            # one. They diverge exactly when k > n_slots.
            assert torch.equal(ply_positions(p[4], p[2]), p[5])
            was = model.training
            model.eval()
            try:
                with torch.no_grad(), fp32_matmul(), torch.autocast("cuda", enabled=False):
                    eb = probe_embed(*p)[0].float()
                    es = torch.cat([probe_embed(*probe_pack(
                        [r], n_planes, n_slots, ds.with_rights, device))[0].float()
                        for r in qry_rows[:4]])
            finally:
                if was:
                    model.train()
            # Right-padding must not change an embedding; if it does, pad_mask is
            # wrong and padding is being pooled into the player vector.
            d = float((eb - es).abs().max())
            assert d < 1e-4, f"batched != single by {d}"
            r1, r2 = run_probe(), run_probe()
            # Two probes, zero steps between them. Under eval() + no_grad() + a
            # frozen gallery the probe is a deterministic function of the
            # weights, so any difference means the gallery is drifting or
            # dropout is still on.
            assert r1 == r2, (r1, r2)
            # Positive control: query with the gallery's own first chunk. If the
            # pipeline is wired up but scoring the wrong rows, nothing else here
            # catches it.
            #
            # The old threshold of recall@1 > 0.99 was satisfiable only with ONE
            # chunk, where the query vector *is* the centroid. With C chunks the
            # centroid is a mean of C vectors of which the query is one, the
            # self-term is diluted to ~1/C, and recall@1 legitimately falls -- at
            # the new default of 2 chunks a fixed 0.99 aborts every run before it
            # starts. So gate it on the chunk count.
            #
            # recall@1 also falls with gallery size, which makes it a poor gate on
            # its own: measured on a random-init trunk (the weakest model this
            # check ever sees) the control scored 0.65 at N=300, 0.51 at N=1200,
            # 0.45 at N=2200 -- about -0.10 per e-fold, extrapolating to ~0.26 at
            # the default N=15000. Hence a floor of 0.10, still 1500x chance
            # there: a spurious abort costs a whole pod run, and the detection
            # power is in the next assert anyway. mean_norm_rank over that same
            # sweep was 0.009-0.014, i.e. flat in N -- it is the size-invariant
            # statistic, and it is what keeps this check non-vacuous against a
            # wrong-rows null of 0.5 (held-out queries on that trunk scored 0.23).
            pc = run_probe([[cache[x] for x in gp[:PK]] for gp in gal_picks])
            lo = 0.99 if len(chunks) == 1 else 0.10
            assert pc["recall@1"] > lo, (len(chunks), lo, pc)
            assert pc["mean_norm_rank"] < 0.05, (len(chunks), pc)
            print(f"  selftest ok: ply_pos, batched==single ({d:.2e}), "
                  f"positive control ({len(chunks)} chunk(s), r@1 floor {lo:.2f}) "
                  f"r@1 {pc['recall@1']:.4f} norm_rank {pc['mean_norm_rank']:.4f} "
                  f"vs held-out {r1['mean_norm_rank']:.4f} "
                  f"({r1['mean_norm_rank'] / max(pc['mean_norm_rank'], 1e-9):.0f}x "
                  f"separation), back-to-back probes identical", flush=True)

        if args.probe_selftest:
            probe_selftest()
    else:
        print("WARNING: --probe-every-hours 0 -- nothing stops this run but "
              "--max-hours", flush=True)

    # `core` is the uncompiled module; a compiled handle's state_dict() carries
    # an "_orig_mod." prefix that no loader in this repo matches.
    core = model
    if args.compile:
        model.embed = torch.compile(model.embed, dynamic=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))

    curve, step, t0 = [], 0, time.time()
    budget = args.max_hours * 3600
    ema = None
    val_hist, best_val, last_val = [], float("inf"), None
    probe_hist, probe_seconds, next_probe = [], 0.0, probe_every
    # `trail` holds the deciding statistic (MRR) for probes taken while
    # fine-tuning; the step-0 baseline is kept out of it because a pre-fine-tune
    # dip would otherwise spend the patience budget on the trunk's own head start.
    # best_recall/best_step are diagnostics for history.json and select nothing:
    # the argmax of a noisy probe is by construction an earlier, less-trained
    # model, and shipping it is the bug this replaced.
    trail, best_trail, base_probe, never_beat = [], None, None, None
    best_recall, best_step, bad_probes, stopped_early = -1.0, 0, 0, False
    rate, rate_pre, rate_check_at, mark = 0.0, None, None, (0, t0)

    def save(path, extra=None):
        d = {"model": core.state_dict(), "cfg": cfg.__dict__,
             "n_planes": n_planes, "n_extra": n_extra, "d_embed": d_embed,
             "n_time_bins": N_TIME_BINS, "n_elo_bins": N_ELO_BINS,
             "n_game_slots": n_slots, "max_len_per_game": mlpg,
             "step": step, "test_pids": test_pids, "supcon_ema": ema,
             # probe_pids so an eval can exclude the players the stopping rule
             # was computed on, whether or not --probe-holdout already did.
             "loss": args.loss, "max_games": n_slots, "probe_pids": probe_pids}
        if extra:
            d.update(extra)
        torch.save(d, os.path.join(args.out, path))

    def dump_history():
        """Written at every probe, not just at a clean exit: a preempted 20h pod
        used to lose every curve it had produced."""
        d = {"args": vars(args), "curve": curve, "final_ema": ema,
             "val_history": val_hist, "best_val": best_val,
             "probe_history": probe_hist, "probe_protocol": probe_proto,
             # Diagnostics. best_step is where recall@1 happened to peak; last.pt
             # is NOT that step and is not meant to be.
             "best_recall": best_recall, "best_step": best_step,
             "best_trail_mrr": best_trail, "baseline_probe": base_probe,
             "never_beat_baseline": never_beat,
             "probe_minutes": round(probe_seconds / 60, 1),
             "stopped_on_probe_patience": stopped_early,
             "minutes": round((time.time() - t0) / 60, 1)}
        tmp = os.path.join(args.out, "history.json.tmp")
        with open(tmp, "w") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, os.path.join(args.out, "history.json"))   # never truncated

    if run_probe is not None:
        # Baseline: the pre-fine-tune trunk is the curve's zero point, and
        # without it there is no way to tell a fine-tune that improved recall
        # from one that only failed to wreck it.
        tp = time.time()
        r = run_probe()
        probe_seconds += time.time() - tp
        r.update(step=0, hours=0.0, wall_hours=0.0, lr=0.0, baseline=True,
                 probe_seconds=round(time.time() - tp, 1), fingerprint=probe_fp)
        base_probe = r
        probe_hist.append(r)
        print(f"  >> probe @ 0.0h step 0 (baseline): r@1 {r['recall@1']:.4f} "
              f"norm_rank {r['mean_norm_rank']:.5f} mrr {r['mrr']:.4f} | "
              f"{r['probe_seconds']:.0f}s gallery {r['gallery']:,} "
              f"chance {r['chance@1']:.2e}", flush=True)
        dump_history()
        mark = (0, time.time())

    for b in dl:
        if step >= args.steps or (time.time() - t0) >= budget:
            break
        # Net of probe time, so the LR at a given step is what the same step
        # would have seen in a run with no probe -- otherwise the probed run is
        # not the same experiment as the one it is being compared against. The
        # break above stays on wall clock because that is what the pod bills;
        # the cost is that the schedule lands at ~6% of base rather than 5%.
        frac = min(1.0, (time.time() - t0 - probe_seconds) / budget)
        for g in opt.param_groups:
            g["lr"] = lr_at(max(int(frac * 20_000), min(step, args.warmup)),
                            20_000, args.lr, args.warmup)

        b = to_dev(b, device)
        pp = ply_positions(b["game_slot"], b["pad_mask"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            emb, _ = model.embed(b["planes"], b["extra"], b["pad_mask"], b["my_turn"],
                                 b["game_slot"], pp)
            loss, st = loss_fn(emb.float(), b["player_id"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        lv = float(loss.detach())
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv

        if step % 100 == 0:
            dt = time.time() - t0
            rate = (step - mark[0]) / max(time.time() - mark[1], 1e-9)
            mark = (step, time.time())
            # it/s net of probe time, so the number still reads as training
            # throughput and a probe does not look like a hardware problem.
            print(f"step {step:>6} | supcon {lv:6.3f} (ema {ema:6.3f}) | "
                  f"{step/max(dt-probe_seconds, 1e-9):5.2f} it/s | {dt/60:6.1f} min",
                  flush=True)
            curve.append({"step": step, "loss": lv, "ema": ema})
            # Exceeding dynamo's recompile limit drops the training graph to
            # eager permanently -- one buried WARNING and then ~1.9x slower for
            # the rest of the run, which would confound the very comparison the
            # probe exists to settle.
            if rate_check_at and step >= rate_check_at:
                if rate_pre and rate < 0.9 * rate_pre:
                    print(f"  !! throughput {rate:.2f} it/s vs {rate_pre:.2f} "
                          f"before the probe -- check for a dynamo recompile "
                          f"or eager fallback", flush=True)
                rate_check_at = None

        # Collapse is the failure mode that killed the triplet run; SupCon's
        # ceiling is ln(B-1), so an EMA parked there means no signal at all.
        if step == args.collapse_step and ema > 0.97 * math.log(args.p * args.k - 1):
            raise SystemExit(f"collapsed: ema {ema:.3f} ~ ln(B-1) "
                             f"{math.log(args.p*args.k-1):.3f}")

        if step % args.eval_every == 0:
            # Logging only. This is the signal that stopped the ctx5 run at 2.6h
            # while recall was still climbing, and it is measured on a drifting
            # sample besides; it stops nothing now.
            v = validate()
            last_val, best_val = v, min(best_val, v)
            val_hist.append({"step": step, "val_supcon": v})
            print(f"  >> val @ {step}: supcon {v:.4f} (best {best_val:.4f}, "
                  f"not a stopping signal)", flush=True)

        # Scheduled in training time, the units the +0.0041/hr slope is quoted
        # in -- which also means a probe cannot push its own clock forward. The
        # max() covers every other kind of stall: without it, one interval spent
        # anywhere leaves next_probe in the past and probes fire every step.
        if run_probe is not None and (time.time() - t0 - probe_seconds) >= next_probe:
            rate_pre, rate_check_at = rate, step + 300
            tp = time.time()
            r = run_probe()
            dtp = time.time() - tp
            probe_seconds += dtp
            now = time.time() - t0 - probe_seconds
            next_probe = max(next_probe + probe_every, now + 0.5 * probe_every)
            r.update(step=step, hours=round(now / 3600, 3),
                     wall_hours=round((time.time() - t0) / 3600, 3),
                     probe_seconds=round(dtp, 1), fingerprint=probe_fp,
                     lr=opt.param_groups[0]["lr"], val_supcon=last_val)
            # Decide on the trailing mean of MRR; report recall@1. Two separate
            # reasons, both of which used to be wrong here:
            #
            # MRR, because recall@1 is the noisiest statistic the probe produces
            # -- a mean of Bernoullis, where a query flipping between rank 1 and
            # rank 2 costs a full 1/N for an embedding change of essentially zero.
            # MRR moves by 1/2 - 1/3 for the same flip and reads every query's
            # whole ranking.
            #
            # Trailing means on BOTH sides, because comparing a single draw
            # against a running maximum ratchets the bar upward on its own: at a
            # flat plateau the max of n draws sits sigma*sqrt(2 ln n) above the
            # mean, so "no gain" gets easier to hit the longer the run lasts,
            # independent of the model.
            trail.append(r["mrr"])
            cur = (float(np.mean(trail[-args.probe_trail:]))
                   if len(trail) >= args.probe_trail else None)
            r["trail_mrr"] = cur
            better = cur is not None and (best_trail is None
                                          or cur > best_trail + args.probe_min_delta)
            if better:
                best_trail, bad_probes = cur, 0
            elif cur is not None:
                bad_probes += 1
            # RESET below the floor, not just refuse to fire. Merely deferring it
            # leaves the counter already at patience when the floor passes, so
            # the run stops at exactly --probe-min-hours every time -- a fixed
            # wall-clock cut wearing a patience rule's clothes. Resetting means
            # the floor buys a genuine patience window after it.
            floored = now < args.probe_min_hours * 3600
            if floored:
                bad_probes = 0
            if r["recall@1"] > best_recall:
                best_recall, best_step = r["recall@1"], step   # diagnostic only
            probe_hist.append(r)
            tag = (f"warm-up {len(trail)}/{args.probe_trail}" if cur is None else
                   "best" if better else
                   f"no gain, under the {args.probe_min_hours}h floor" if floored
                   else f"no gain ({bad_probes}/{args.probe_patience})")
            print(f"  >> probe @ {r['hours']}h step {step}: "
                  f"r@1 {r['recall@1']:.4f} r@10 {r.get('recall@10', -1):.4f} "
                  f"norm_rank {r['mean_norm_rank']:.5f} mrr {r['mrr']:.4f} "
                  f"trail {'n/a' if cur is None else f'{cur:.5f}'} | "
                  f"{dtp:.0f}s gallery {r['gallery']:,} ({tag})", flush=True)
            dump_history()
            mark = (step, time.time())
            # No wall-clock term here: the reset above is what enforces the floor.
            if bad_probes >= args.probe_patience:
                stopped_early = True
                lr_now = opt.param_groups[0]["lr"]
                # --lr 0 is the frozen-weights control for this stopping rule
                # (every probe identical, so patience must fire); it must not
                # take the reporting line down with a division by zero.
                lr_frac = lr_now / args.lr if args.lr else 0.0
                # --max-hours is the cosine horizon, not a cost cap, so a
                # patience stop leaves the model mid-decay and never annealed.
                # Logged rather than fixed: read it before concluding the
                # stopped model is what the trajectory was worth.
                print(f"early stop: {args.probe_patience} probes without a "
                      f"+{args.probe_min_delta} gain on the {args.probe_trail}-probe "
                      f"trailing mean of MRR -- MRR is the statistic that decided, "
                      f"not recall@1 (best trailing MRR {best_trail:.5f}, now "
                      f"{cur:.5f}). Headline recall@1 {r['recall@1']:.4f}, peaked "
                      f"{best_recall:.4f} at step {best_step} -- last.pt is THIS "
                      f"step, not that one, because the peak is an argmax over the "
                      f"same noise used to call it a peak. lr {lr_now:.2e} = "
                      f"{lr_frac:.0%} of base, {frac:.0%} through the "
                      f"decay horizon -- un-annealed", flush=True)
                break

        if step % 1000 == 0 or (time.time() - t0) >= budget:
            save("last.pt")

    # last.pt is the last step. There is deliberately no best.pt: a patience stop
    # lands >= --probe-patience probes after the argmax, so "best" is always an
    # earlier, less-trained model, picked out by the same probe noise that
    # declared it the maximum. The peak lives in history.json as a diagnostic.
    save("last.pt")
    stale = os.path.join(args.out, "best.pt")
    if os.path.exists(stale):
        print(f"  !! {stale} is left over from an older run of this script and is "
              f"NOT this run's model -- delete it, and eval last.pt", flush=True)

    # A fine-tune that degraded the trunk plateaus exactly like a healthy one:
    # patience counts probes without a gain, and there are no gains to be had
    # either way. The step-0 baseline is the only thing that separates them.
    if base_probe is not None and len(probe_hist) > 1:
        peak = max(p["mrr"] for p in probe_hist[1:])
        peak_r1 = max(p["recall@1"] for p in probe_hist[1:])
        never_beat = peak <= base_probe["mrr"]
        if never_beat:
            print("!" * 78, flush=True)
            print(f"!! THIS FINE-TUNE NEVER BEAT ITS OWN STARTING POINT. Baseline "
                  f"(step 0) mrr {base_probe['mrr']:.5f} r@1 "
                  f"{base_probe['recall@1']:.4f}; best of the {len(probe_hist)-1} "
                  f"probes since, mrr {peak:.5f} r@1 {peak_r1:.4f}. Whatever "
                  f"stopped this run, it was not a plateau at a better model -- "
                  f"last.pt is worse than the trunk it started from. Evaluate "
                  f"--ckpt before believing last.pt.", flush=True)
            print("!" * 78, flush=True)
        else:
            print(f"baseline -> best: mrr {base_probe['mrr']:.5f} -> {peak:.5f} "
                  f"(+{peak - base_probe['mrr']:.5f}), r@1 "
                  f"{base_probe['recall@1']:.4f} -> {peak_r1:.4f}", flush=True)
    dump_history()
    print("CTX_FINETUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
