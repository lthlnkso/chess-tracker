# Training the chess transformer, end to end

This is the runbook for producing a model that can be dropped into the product.
It is written for someone who wants to try a **variation** — a bigger trunk,
longer context, a different loss — and needs to know what to run, in what order,
and how to tell whether the variation is actually better.

Measured results for every artifact we have shipped or rejected are in
[`model_registry.csv`](model_registry.csv). Add a row when you finish a run.
Read the **Known traps** section before you start; most of it was learned by
losing money.

---

## 1. What the product needs

Three artifacts go into the deployed product:

| artifact | job | produced by |
|---|---|---|
| **conditioned trunk** | plays the bot's moves at a requested rating, and predicts the visitor's rating | stage 1 + stage 3 |
| **identifier** | turns a set of games into a 128-d vector | stage 2 |
| **gallery** | one centroid per lichess player | stage 4 |

A fourth, the **verifier**, is a candidate rather than shipped — see stage 5.

The product question is one number, and everything below exists to move it:

> Given 1–10 games a visitor has just played against our bot, what is the
> probability the true player is in the **top 10** of a ~2M-player gallery?

---

## 2. Data

Shards live on the RunPod network volume and are reachable over S3 without a pod
(`runpod/s3io.py`, `runpod/fetch_final.sh`). One month is ~6.5 GB.

```
data/mt/2026-01 .. 2026-06     ingested, 1+0 bullet ("60+0") only, rated standard
  meta.npy      per-game: offset, nply, white_pid, black_pid, elos, tc, result
  moves.u16     flat move stream, indexed by meta.offset
  clocks.u16    flat clock stream, centiseconds, 0xFFFF = unknown
  players.txt   newline-joined usernames; index == pid
```

Two facts that bite:

- **`pid` is per shard.** The same username has a different integer in each
  month. Join on the lowercased username, never on `pid`. `union_gallery.py`
  does this; anything else you write must too.
- **Clocks are 1-second granularity.** `[%clk]` ticks once per second in these
  games, so every real think time is a whole number of seconds. Feeding the
  model fractional seconds is off-distribution.

To ingest a new month: `ingest.py` (see `runpod/ingest_worker.sh`), ~36 min per
month.

---

## 3. Stage 1 — pre-train the trunk

Predicts the **next position**, not the next move: every legal successor is
encoded and scored, and the model picks the highest. Multi-task — move, think
time, and rating heads share the trunk.

```bash
python train_multigame.py --shard data/mt/2026-01 --out ckpt/pre \
    --max-games 5 --max-len-per-game 160 \
    --lr 1.5e-4 --warmup 1000 --batch 48 \
    --d-model 256 --layers 8 --heads 8 \
    --plies-per-game 8 --n-cand 32 --workers 24 \
    --eval-every 8000 --eval-batches 25 --balance-elo \
    --patience 10 --min-delta 0.0002 --amp --compile \
    --max-hours 20
```

- `--balance-elo` samples across the rating range instead of the crowded middle.
  Keep it; the elo head and any rating conditioning depend on it.
- **Watch `move_acc`.** Reference: 0.4964 at 816k steps.
- To vary the model, change `--d-model/--layers/--heads`; to vary context,
  `--max-games` and `--max-len-per-game` (this is the one that unlocks a
  verifier with more than 4 query games — see stage 5).

---

## 4. Stage 2 — contrastive fine-tune (the identifier)

Turns the trunk into an embedding where the same player's games land together.
**Multi-Similarity loss over pairs inside a P×K batch** — 24 players × 4 bundles.
No centroids are involved in training; centroids are an inference-time
construction.

```bash
python finetune_ctx.py --shard data/mt/2026-01 --ckpt ckpt/pre/last.pt \
    --out ckpt/ft --loss ms \
    --p 24 --k 4 --lr 3e-5 --warmup 200 --workers 24 \
    --eval-every 8000 --eval-batches 25 --patience 1000000 \
    --amp --compile --max-hours 20
```

- **MS loss won a sweep** against SupCon, Circle, Triplet, ProxyAnchor and
  ArcFace (+52/53% over the next best across two seeds). The proxy-based losses
  collapsed. Don't re-run that sweep without a reason.
- **Do not use `--same-colour`.** It exists, it is tested, and it lost twice.
- `--patience 1000000` disables early stopping. Use a real patience (8) only if
  you are deliberately looking for saturation.

---

## 5. Stage 3 — rating conditioning (optional but shipped)

Adds a rating embedding to the trunk input so the bot can be asked to play like
a specific Elo. **Zero-initialised**, so a checkpoint that has never trained it
is bit-identical to before.

```bash
python train_multigame.py --shard data/mt/2026-01 --out ckpt/pre_elo \
    --init ckpt/pre/last.pt --elo-cond --elo-drop 0.1 \
    --lr 8e-5 --warmup 500 --batch 48 --balance-elo \
    --eval-every 8000 --patience 8 --amp --compile --max-hours 10
```

`--elo-drop 0.1` trains the "unknown rating" slot so callers that supply no
rating still hit a trained path.

**Verify it actually learned** — a zero-init embedding can train to nothing while
the loss curve looks healthy, because move accuracy is dominated by the trunk:

```bash
python elo_probe.py --ckpt ckpt/pre_elo/last.pt --shard data/mt/2026-01
```

Reference: top move differs 15.2% between requested 1000 and 2200, and agreement
with the move actually played **peaks when the requested rating matches the real
mover** (0.441 vs 0.422–0.437). If those rows are all equal, the embedding is dead.

---

## 6. Stage 4 — build the gallery

One centroid per player, pooled across every month.

```bash
python union_gallery.py --ckpt ckpt/ft/last.pt \
    --shards data/mt/2026-0{1,2,3,4,5,6} \
    --out play/gallery_2026.npz \
    --k 5 --gallery-games 64 --min-games 13
```

- `--gallery-games` is a **cap, not a requirement**: a player with 20 games gets
  a 20-game centroid. Requiring N would quietly restrict the gallery to
  hyper-active players and flatter every number you then measure.
- Centroid richness matters: 12 → 64 games is worth **+11% top-10**. The curve
  flattens after ~32 (12→32 is +4.8%, 32→64 is +0.6%).

Optionally rebuild `play/elo_table.npz` (`build_elo_table.py`) — per-player
ratings, needed only if you are experimenting with rating-aware re-ranking.

---

## 7. Stage 5 — the verifier (candidate, not shipped)

Reads your games and **one** candidate game and answers "same player?". Exists
because cosine puts the right player at rank 1 for 51.6% of ten-game queries but
inside the top 1000 for 94.4% — the shortlist nearly always contains the answer
and the ordering is what fails.

Two architectures, both trained from the stage-2 identifier:

- `verify.py` — **cross-encoder**. One fused sequence, 4 query games + 1
  candidate. The trunk is causal, so the candidate attends to your games and not
  the reverse. AUC 0.828 on hard negatives.
- `verify2.py` — **dual encoder**, ~80× the training throughput (measured), by
  scoring all B×B pairs in a batch instead of B. Gives up cross-attention.

```bash
python verify2.py train --shard data/mt/2026-01 --ckpt ckpt/ft/last.pt \
    --out ckpt/verifier --k 5 --mlpg 60 \
    --batch 48 --neighbours 64 --workers 16 --lr 6e-5 --max-hours 1
```

Negatives must be **hard**. At inference every candidate has already been ranked
into a shortlist by cosine, so easy negatives are a distribution the model will
never meet. `verify2.py` builds each batch from one neighbourhood, which makes
every off-diagonal pair a hard negative for free.

---

## 7b. Hand-derived features (measured, cut)

Piece-type move fractions, timing summaries, checks per game, opening keys --
computed with `handfeat.py`, profiled over the whole gallery by
`build_handfeat_pack.py`, scored by `bayesfeat.py`. **They do not help.** The
result is recorded here because the idea is an obvious one to have twice.

**Intraclass correlation first.** A feature is useful only if it varies more
between players than between one player's own games. `python handfeat.py` prints
ICC for all 18, measured on 45,568 game-sides:

| feature | ICC | ICC after removing Elo |
|---|---|---|
| mean_think | 0.382 | 0.248 |
| fast_frac | 0.367 | 0.293 |
| castled | 0.146 | 0.138 |
| pawn_frac | 0.097 | 0.106 |
| knight_frac | 0.051 | 0.053 |
| queen_frac | 0.042 | 0.049 |

**The piece fractions are the weakest thing on the list.** ICC 0.05 means ~95% of
the variance in "what fraction of your moves were knight moves" is variance
between your own games. People do have piece preferences; a bullet game is too
small a sample to see them, and the opponent dictates most of it. The fitted LLR
weight for `knight_frac` is **-0.007**. The timing features are the strongest,
and they are exactly the ones the model already gets -- `time_features` feeds
every ply and there is a think-time head.

**Openings are the only interesting one, and cosine already has them.** A shared
6-ply opening is 26x evidence against a *random* player. But:

| | vs random | vs cosine top-100 |
|---|---|---|
| P(shares an opening) | 0.0158 | 0.0892 |

Cosine neighbours are **5.64x** enriched for shared openings, so most of that
26x is signal the shortlist has already spent. Working the residual through at
the best depth (4 plies) with 64-game profiles leaves **~0.6 nats per 5-game
query**, down from 5.0 against random -- and 64-game opening profiles do not
exist; the shipped pack has 4 games, worth ~0.8 nats before redundancy.

**End-to-end, on the real 558,735-player gallery, 5 games, held-out queries:**

| scorer | r@1 | r@10 | r@100 | median rank |
|---|---|---|---|---|
| cosine | 0.3400 | 0.5800 | 0.7850 | 6 |
| features alone | 0.0050 | 0.0050 | 0.0150 | 71,802 |
| cosine + w*features | best w = **0.00** | | | |

The combination weight was swept on a dev half and reported on a held-out half.
Every positive weight is monotonically worse. Features alone beat chance (median
71,802 of 558,735) -- the signal is real, it is just signal cosine already has,
delivered with extra noise.

**Traps if you retry this anyway.** Two alignment bugs will silently produce
garbage rather than an error: lichess `[%clk]` ticks in whole seconds so every
gallery think time is an integer, while the browser measures milliseconds
(`profile_from_uci` quantises); and `hash()` is salted per process, so a
multiprocess build of an opening index without a fixed digest disagrees between
workers and across runs (`opening_hash` uses blake2b).

---

## 8. Evaluation

### 8.1 The product metric

Top-10 recall against the full gallery, as a function of games played. This is
the number that decides whether a variation is better.

```bash
python gallery_ctx.py --ckpt ckpt/ft/last.pt --shard data/mt/2026-01 \
    --out results/curve.json \
    --ks 1,2,3,4,5 --gallery-games 64 \
    --gallery-players 200000 --query-players 5000 \
    --sizes 1000,10000,50000,100000,200000 --workers 22
```

Two things make this honest:

- **Queries must be held out.** The script refuses to run if the checkpoint
  carries no `test_pids`; distractors need not be held out, because they are
  only ever wrong answers, and the deployed gallery is everyone.
- **You cannot extrapolate UP to 2M.** `recall_at_size()` is a hypergeometric
  over which distractors survive a *subsample*, so it is only defined for
  `N <= M`, the gallery actually embedded. Ask for a larger N and it returns
  `nan`; `gallery_ctx.py` silently drops those sizes (`sizes = [s for s in sizes
  if s <= M]`), so a `--sizes` list containing 2000000 will quietly produce
  nothing rather than an error.

  **To report the product metric at 2M you must embed a ~2M gallery.** That is
  the whole point of `--gallery-players`: set it to the real roster size and pay
  for the embedding pass. A 558,735-player gallery took ~50 min on one A5000.
  Anything smaller is a proxy, and must be labelled with the size it was
  measured at.

Reference, `ctx5_ft2` on a real 200,000-player gallery, 5,000 held-out queries:

| games | r@1 | **r@10** | r@100 |
|---|---|---|---|
| 1 | 7.1% | 19.1% | 39.9% |
| 3 | 46.8% | 70.3% | 86.4% |
| 5 | 68.5% | **85.7%** | 94.9% |

Deeper, on the 558,735-player production gallery:

| games | r@1 | r@10 | r@100 | r@1000 |
|---|---|---|---|---|
| 5 | 44.0% | 65.6% | 80.4% | 92.4% |
| 10 | 51.6% | 74.0% | 87.6% | 94.4% |

### 8.2 End to end, through the product

The offline metric measures a player's **real lichess games**. The product feeds
it **games played against our bot**, and the gap is enormous — this is the single
biggest thing this project learned:

```
same player, same gallery, same model
  real lichess games      rank 1
  games vs our bot        rank ~8,000-21,000
```

So an offline improvement is necessary but not sufficient. To test end to end:

1. Point the demo at the new artifacts (`.claude/launch.json`: `--ckpt`
   conditioned trunk, `--gallery` new gallery; identifier defaults to
   `ckpt/final/ctx5_ft2.pt` in `play/server.py`).
2. Play 10 games. Enter a username in the demo's box — it reports exact rank and
   percentile out of the whole gallery, which is far more informative than
   "not in the top 10" (that covers rank 11 to rank 550,000).
3. Compare against the reference chain in section 9.

`verify_eval.py` does the shortlist-level version offline: builds a real cosine
shortlist, scores every candidate's games with a verifier, combines, and reports
r@1/r@10 for cosine alone vs combined.

### 8.3 Intermediate metrics — what to watch during a run

| stage | metric | good | dead |
|---|---|---|---|
| pre-train | `move_acc` | ≥0.49 by 800k steps | flat at ~0.2 |
| pre-train | `elo_mae` | <130 | >300 |
| contrastive | `val_loss` (the MS loss; older runs call it `val_supcon`) | <0.39 | stuck ≥0.6 |
| rating cond. | `elo_probe` disagreement | ≥10% | 0% (embedding dead) |
| verifier | AUC on **hard** negatives | >0.75 | ~0.5 |
| verifier | loss | falling below ln(B) | pinned at exactly ln(B) |

That last row is not hypothetical: a diverged run sat at exactly ln(48) = 3.8712
for 90 minutes. `verify2.py` now aborts on divergence.

---

## 9. Reference results to beat

```
identification, 200k gallery, 5 real games        85.7% top-10
identification, 558k gallery, 10 real games       74.0% top-10
demo, 6 games vs the bot (before fixes)           rank 21,302 / 558,735
demo, 6 games vs the bot (after fixes)            rank 29 / 558,735
elo prediction, real games                        MAE 156, r 0.837
bot move accuracy                                 0.5096 next-state
```

The demo's leap from rank ~21,000 to 29 came from **making the games look like
real games**, not from better matching:

1. bot think times sampled from real 1+0 data (was a hard 0 — a value no ply in
   the training data ever takes)
2. human times rounded to whole seconds, matching `[%clk]` granularity
3. a rating-matched opponent
4. 10 games instead of 5, fused as bundles (r@1 0.910 → 0.960)

---

## 10. Known traps

**We ship `last.pt`, not the best-val checkpoint.** `ctx5_ft2`'s best val was
0.3578 but the shipped weights scored 0.3888. Nobody chose that. If you are
chasing a small gain, save and evaluate `best.pt`.

**Colour splitting loses.** Separate white/black centroids with colour-matched
queries lost on the mixed-trained model (97.9% → 95.6%) and lost *by more* on a
model trained exclusively on single-colour bundles (96.8% → 92.7%). The cause is
halving the games behind each centroid, not the fusion — fusing multiple bundles
against the *full* centroid gains +5 points.

**Elo is already in the embedding.** `model.embed()` concatenates the predicted
rating distribution before the projection head. Adding rating as a re-ranking
feature is redundant; it measured as near-useless.

**Don't add two accelerators at once.** A verifier run diverged with a learnable
logit scale under bf16 plus `torch.compile`; isolating them cost a wasted run.
Add one, confirm the loss moves, then add the next.

**Data loading is not the bottleneck.** Profiled: 2.75 ms/item to replay boards
against ~100 ms/item of model. With 16 workers the loader supplies ~5,600
items/s while the GPU consumes ~660. Optimise the model, not the pipeline.

**The verifier ignores most of each game.** AUC is flat from 160 plies per game
down to 40 (0.8297 vs 0.8270) while cost falls 3×. Identity lives in the opening
and early middlegame.

**Infrastructure**, all of which has cost money here:

- Pods take 15–20 min to become reachable while a 16.3 GB image pulls. Prefer a
  slim image plus a self-contained venv; `runpod/novol_run.sh` boots in ~1 min.
- Network volumes are DC-locked and have failed to mount for hours at a time.
  Pull inputs over S3 instead and you are free to run in any datacenter.
- `python:3.12-slim` has **no C compiler**, which `--compile` needs, and **no
  curl**, which a naive self-terminate script needs. Both are handled in
  `runpod/novol_run.sh` and `runpod/novol_stopper.sh`.
- Always arm a self-terminating stopper. Orphaned pods have cost ~$2.30 here.
- Cheapest throughput measured: **A5000 at $0.27/hr → 135k steps/$**, versus a
  4090's 73k. Pick on steps-per-dollar, not speed.

---

## 11. Checklist for a variation

1. Add a row to `model_registry.csv` with the hypothesis before you run.
2. Stage 1 → check `move_acc`.
3. Stage 2 → check `val_loss` (the MS loss; pre-2026-08-16 runs print it as `supcon`).
4. Stage 4 with the **same** gallery settings as the baseline, or the comparison
   is meaningless.
5. Section 8.1 with **identical** `--sizes` and `--query-players`.
6. Only if 8.1 improves: rebuild the production gallery and test end to end (8.2).
7. Record the product metric, not just the loss. Two of the ideas in this repo
   improved a training metric and lost on the product one.
