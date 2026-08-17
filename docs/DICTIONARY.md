# Dictionary

What every name in this project means, where its log is, and why the run existed.

**Rule for this file: only things with exact evidence.** A number here should be
traceable to a checkpoint, a registry row, or a log line someone has actually
read. Anything remembered-but-unverified goes in "Unknown" at the bottom, not in
the tables. Half the wasted money in this project came from confident
comparisons between things that were not comparable.

Measured metrics live in [`model_registry.csv`](model_registry.csv); the runbook
lives in [`TRAINING.md`](TRAINING.md); the forks we chose and could still
un-choose live in [`BRANCHES.md`](BRANCHES.md). This file is the index that says
which name is which.

---

## 1. Naming, and the traps in it

| convention | means |
|---|---|
| `ctx5_*` | 5 game slots, 160 plies per game — the shipped context shape |
| `*_pre` | stage 1, move/time/rating pre-training (`train_multigame.py`) |
| `*_ft*` | stage 2, contrastive identification fine-tune (`finetune_ctx.py`) |
| `*ftc*` | the **c** is `--same-colour`. Rejected line — see TRAINING.md traps |
| `*_partial.pt` | rolling 30-minute upload from a live pod, not a final artifact |
| `*_best.pt` | lowest val at some eval. **Selected on a noisy metric — see §7** |
| `*_last.pt` | final weights. This is what has historically shipped |

**Traps that have actually cost us:**

- **`novol_ft2.sh` and `novol_ft3.sh` are off by one from what they produce.**
  `novol_ft2.sh` continues `ctx5_ftc` → `ctx5_ftc2`. `novol_ft3.sh` continues
  `ctx5_ft2` → `ctx5_ft3`. The script number does not match the model number.
- **`novol_ft3.sh`'s original header described the colour run.** It was pasted
  from `novol_ft2.sh` and never updated; its `--colour-split` eval came from the
  same paste. Corrected in place 2026-08-14, but assume other headers are stale.
- **`mt_run.log` is NOT any `ctx5_*` run.** It is `train_multitask.py`, single
  game, `--n-cand 16`, `--batch 128`. Chance accuracy there is 1/16 vs 1/32 for
  every `ctx5_*` model, so its `move_acc` numbers are **not comparable to
  anything else in this repo.** They look better and they are not.
- **`s3io.py down` skips when the local file matches on SIZE.** Two checkpoints
  of the same architecture are the same size, so it will silently keep a stale
  one. `rm` the local file to force a real download.
- **`novol_run.sh` forwarded only `PY` and `FT_H` to the pod** until the
  `EXTRA_ENV` passthrough was added 2026-08-14. Before that, any other knob in a
  work script silently fell back to its default.
- **NEVER size a population on `data/2026-06-big`.** It is a 45 MB PARTIAL
  month (160,075 games, 46,836 players). The real shard `mt/2026-01` has
  **24,244,206 games and 522,735 players**. On the partial everyone looks
  inactive, so any "how many players have >= N games" question answered there is
  wrong by a wide margin: it said the >=10 filter costs 51% of players and 37% of
  games; on the full shard it costs **2% of the games** (263,823 players, 98% of
  games). Measure population questions on the pod, against the real shard.
- **Time caps are runaway backstops, not stopping points.** A stage should end on
  `--patience`, i.e. when the metric stops improving. Both `ctx5_pre` (816k steps)
  and `ctx5_ft2` (775k) ended because a **wall clock expired mid-climb**, and
  nobody ever learned where their curves went. Setting a cap by working backwards
  from a previous run's compute reproduces that error and answers a question
  nobody asked — we are betting on a parameter set, not matching a prior run's
  budget.
- **`nproc` LIES inside a RunPod container.** On an A5000 pod, `nproc`,
  `os.cpu_count()` AND `os.sched_getaffinity()` all reported **96**, while
  `/sys/fs/cgroup/cpu.max` said `765000 100000` = **7.65 cores**. Spawning 94
  Stockfish engines onto 7.65 cores did not just fail to help -- context
  switching made it *worse* than 7, and the symptom (engines parked in S state,
  ~3 in R) is indistinguishable from task starvation. Always size worker pools
  from the cgroup quota: `cpu_quota()`, now shared in `cpuquota.py`.
  **Second instance, 2026-08-17:** an eval pod reported nproc 48 against a ~8-core
  quota, and `gallery_ctx.py --workers 24` starved the GPU to **0% utilisation**.
  The k=10 gallery built at **28 bundles/s against k=5's 503/s** -- an 18x
  slowdown that reads like a model or memory problem and is pure CPU
  oversubscription. k=10 costs twice the decode of k=5 (ten games per bundle,
  not five), so it is the first place a starved loader shows up. Eval paths in
  eval_only.sh / ft_only.sh / big_run.sh now size workers from cpu_quota().
- **Timing `engine.analyse()` does NOT measure the engine.** It includes
  python-chess parsing every multipv `InfoDict`. At depth 6 with multipv 32 the
  search is ~30ms and the PARSING is the larger half: measured on a saturated
  pod, python workers burned 407% CPU against Stockfish's 358%. A local profile
  that lumps them together will report "engine 99.7%, python 0.3%" and send you
  hunting the wrong bottleneck.
- **Anything labelled `supcon` from BEFORE 2026-08-16 is actually the MS loss.**
  `finetune_ctx.py` builds ONE `loss_fn = make_loss(args.loss)` and uses it for
  both training and validation; `--loss` has defaulted to `ms` throughout. The
  old prints, the `val_supcon` / `supcon_ema` checkpoint keys and the registry's
  `val_supcon_best` were all **Multi-Similarity** values under a name left over
  from when SupCon was the default. Historical comparisons are still sound --
  every contrastive row in the registry was `--loss ms` -- only the word was
  wrong.
  **Fixed 2026-08-16:** the trainer now prints the actual loss name and writes
  `val_loss` / `loss_ema` (with `loss` recording which), and the registry says
  `val_ms_best`. But **checkpoints and logs produced before that date still
  carry the old keys** -- including `ctx5_ft2` and the in-flight ctx10 runs,
  whose pods took the code as it was at launch. Read either name.
- **`NOVOL_ALL_DONE` is what deletes the pod.** `novol_stopper.sh` greps the work
  log for it. A work script that prints it on a failure path BEFORE uploading its
  log destroys its own diagnostic: the first `ctx10-bet` died ~20 min in, said
  ALL_DONE, and the pod took the log with it. Every exit path must upload first
  -- `big_run.sh` now routes all of them through `finish()`.

---

## 2. Trunk lineage — plays moves, predicts rating

| name | from | steps | GPU-h | key facts | verdict |
|---|---|---|---|---|---|
| `ctx5_pre.pt` | scratch | 816,000 | 17.4 | val move_acc **0.4964**, time_acc 0.4956, elo_mae **125.37** | shipped |
| `ctx5_pre_elo.pt` | `ctx5_pre.pt` | 400,000 | 8.2 | val move_acc 0.5096 **while being told the rating** — not comparable to 0.4964, see below | shipped (bot only) |

`ctx5_pre.pt` config, read from the checkpoint itself:
`d_model 256, n_layers 8, n_heads 8, d_ff 1024, max_len 168, dropout 0.1,
d_embed 128`, 5 game slots, 160 plies/game, 7.92M params.

**The bot and the identifier FORK off `ctx5_pre`; they are not a chain.**
`ctx5_pre_elo` (conditioned) drives the demo's opponent, `ctx5_ft2` (contrastive)
is the identifier, and neither descends from the other. So adding a conditioning
stage cannot affect identification — and chaining them would be risky, because
`embed()` concatenates the predicted rating into the projection head while
`--elo-drop 0.1` hands the trunk the true rating 90% of the time, leaving the
"unknown rating" path (the one identification actually uses) the least trained.

**CAUTION on the 0.5096 vs 0.4964 comparison.** `train_multigame.py:85` feeds the
TRUE rating at validation when `--elo-cond` is on. The conditioned model was told
the answer; the unconditioned one was not. Whether conditioning improves the
trunk is **unmeasured**.

`ctx5_pre_elo.pt` adds a zero-initialised rating embedding to the trunk input,
so an untrained one is bit-identical to its parent. Requested-rating 1000 vs
2200 changes the top move **15.2%** of the time, and agreement with the move
actually played peaks when the requested rating matches the real mover
(0.441 vs 0.422–0.437). This is the model the demo's bot plays from.

**`ctx5_pre.pt`'s early training curve does not exist.** Only the final `val`
dict survived in the checkpoint; `history.json` was never uploaded. So "was the
big model ahead of the small one at step N" is unanswerable from what we have.

---

## 3. Identifier lineage — turns games into the 128-d vector

This is the core of the product. Every gallery is built with one of these.

| name | from | steps | GPU-h | val_ms | top10_200k_5games | verdict |
|---|---|---|---|---|---|---|
| `ctx5_ft.pt` | `ctx5_pre.pt` | 76,000 | 2.6 | 0.4218 | — | superseded |
| `ctx5_ft2.pt` | `ctx5_pre.pt` | 775,351 | 6.0 | 0.3578 best / **0.3888 shipped** | **0.8570** | **shipped** |
| `ctx5_ft3.pt` | `ctx5_ft2.pt` | 85,300 | 2.0 | 0.3829 | — | abandoned |
| `ctx5_ft4` | `ctx5_ft3.pt` | 72,000 | ~1.4 | 0.3777 best @ step 8,000 | **0.8346** | **cut** |
| `ctx5_ftc.pt` | `ctx5_pre.pt` | 347,036 | 6.0 | 0.4817 | 0.7932 | rejected |
| `ctx5_ftc2.pt` | `ctx5_ftc.pt` | 72,000 | 1.7 | 0.4851 | 0.7932 | rejected |

**Why the ft2 → ft3 → ft4 chain exists.** `ctx5_ft2` trained with
`--patience 1000000`, i.e. early stopping switched off, and stopped because a
**6-hour wall clock expired** — not because it converged. Nobody knew whether
there was more to gain. `ctx5_ft3` resumed it with `--patience 8` to find out and
was itself killed at 2.0 GPU-h to fund the Elo run. `ctx5_ft4` resumed that and
is the first of the three to stop on **convergence**: patience fired at step
72,000 with no improvement after step 8,000.

So the chain answers one question — "is there more in this model?" — and the
answer is **no: continuing made the product metric WORSE.** On the same protocol
r@10 fell 0.8570 (shipped ft2) → 0.8346 (ft4) while `val_ms` "improved"
0.3888 → 0.3777. The val metric moved in the **opposite** direction from the
thing we ship. `ctx5_ft2` remains the identifier and this lineage is closed.

That run also settled best-vs-last for good: **`last.pt` 0.8346 beats `best.pt`
0.8276** on the product metric. Shipping `last.pt` was always correct and
best-val selection is actively worse — exactly what §7's noise analysis
predicted.

---

## 4. Verifier lineage — "are these two players the same person?"

Second stage over the cosine shortlist. Exists because cosine puts the right
player at rank 1 for 51.6% of ten-game queries but inside the top 1000 for
94.4%: the shortlist nearly always contains the answer and the ordering fails.

| name | arch | steps | val AUC | r@10 gain vs cosine | verdict |
|---|---|---|---|---|---|
| `verifier_best.pt` | v1 cross-encoder (`verify.py`) | 398,000 | 0.8281 | +0.044 (~1.4σ) | candidate |
| `verifier2_sat.pt` | v2 dual encoder (`verify2.py`) | 88,800 | **0.8607** | **−0.007** | **cut** |
| `verify3.py` | v3, `--shortlist --extra 96` | — | — | — | in flight (yours) |

**The v2 result is the important one, and it is a negative.** It saturated
properly (12 flat evals, not time-capped) at AUC 0.8607 — but that AUC is
measured against 47 in-batch neighbours drawn from a 40k pool. Against the
**deployed** top-100 of 558,735 the per-game AUC is only **0.6058**, and
reranking *loses* to plain cosine (r@10 0.780 vs 0.787) under every fusion tried,
including the calibrated Bayesian LLR. `play/bayes_calib.json` therefore carries
`recommended=none` and the server skips the second stage.

The diagnosis in the registry: **the fix is harder negatives from the real
shortlist, not more steps.** That is what `verify3.py --shortlist` is for.

Local files: `ckpt/final/verifier2_best.pt` is the saturated step-88,800 model;
`verifier2_best_1h.pt` is the earlier 1-hour, step-39,200, AUC-0.840 checkpoint
kept for comparison.

---

## 5. Galleries, packs and tables

| file | what | size | built from |
|---|---|---|---|
| `play/gallery_2026.npz` | **the** gallery: 558,735 centroids × 128, float16 | 137 MB | `ctx5_ft2.pt`, shards 2026-01..06 |
| `play/gallery_deploy.npz` | older 200k single-month gallery | 144 MB | `ctx5_ft2.pt`, 2026-01 |
| `play/verifier_pack.npz` | 4 games × 60 plies for **all** 558,735 players | 247 MB | 6 shards |
| `play/handfeat_pack.npz` | 18 hand features + opening keys per player | 44 MB | `verifier_pack.npz` |
| `play/bayes_calib.json` | Platt coefficients; currently `recommended=none` | — | held-out scores |

`gallery_2026.npz` keys are `centroids, names, pids, k, centroid_games, ckpt,
shards` — note it is **`centroids`**, not `cent`. Mean 46.3 games per centroid;
`--gallery-games 64` is a cap, not a requirement.

Row order is shared: `verifier_pack`, `handfeat_pack` and `gallery_2026` are all
aligned to the same 558,735 names, so a row index means the same player in all
three.

**Data shards.** `data/mt/2026-01 .. 2026-06` live on the volume (~6.5 GB each,
reachable over S3 without a pod). `data/2026-06-big` is the local partial month:
160,075 games, 46,836 players. `pid` is **per shard** — join on lowercased
username, never on `pid`.

---

## 6. Where the logs are

S3 gateway to the network volume — bucket **`shusq6ritt`**, endpoint
`https://s3api-eu-cz-1.runpod.io`, region `EU-CZ-1`, credentials
`RUNPOD_S3_ACCESS_KEY` / `_SECRET_KEY` in `.env`. Everything final lands under
`final/`.

| log | run |
|---|---|
| `final/verify_run.log` | verifier v1 |
| `final/verify2_run.log` | verifier v2 saturation |
| `final/verify3_run.log` | verifier v3 (yours) |
| `final/ft2_run.log` | `ctx5_ftc` → `ctx5_ftc2` (**not** ft2 — see §1) |
| `final/ft4_run.log` | `ctx5_ft3` → `ctx5_ft4` |
| `final/elo_run.log` | `ctx5_pre_elo` |
| `final/big_probe_run.log` | 10× params scaling probe |
| `final/dim_sweep_run.log` | embedding-dimension sweep |
| `final/small_ctrl_run.log` | small-model control for the scaling probe |
| `final/union_run.log` | `gallery_2026.npz` build |
| `final/pack_run.log` | `verifier_pack.npz` build |
| `mt_run.log` (bucket root) | ancestor `train_multitask.py` run — **incomparable, see §1** |

On a live pod the working log is `/data/<name>.log`; `runpod/novol_stopper.sh`
watches it and deletes the pod when it sees `NOVOL_ALL_DONE`.

---

## 7. Metrics — and how much of each is noise

| metric | what it measures | trustworthy? |
|---|---|---|
| `top10_200k_5games` | P(true player in top 10 of a 200k gallery from 5 games) | **the product metric.** 5,000 queries — the one number to optimise |
| `val_move_acc` | next-position accuracy among `--n-cand` candidates | comparable **only** at equal `n_cand` |
| `val_loss` (was `val_supcon`) | contrastive val loss — **the MS loss**, lower better | **noise-dominated at this scale — see below** |
| `val_auc_hard_neg` | verifier AUC vs in-batch neighbours | optimistic; the deployed equivalent was 0.6058 vs 0.8607 |

**`val_ms` cannot rank checkpoints this close together.** Measured on
`ctx5_ft4`'s nine evals: mean 0.4061, **sd 0.0158**. The entire gap from ft2's
shipped 0.3888 to ft4's best 0.3777 is **0.0111 — smaller than one sd.**

Worse, "best val" selects the luckiest eval, not the best model. ft4's best sits
1.79 sd below its own mean; the expected minimum of 9 pure-noise draws is 1.48
sd. `ctx5_ft2`'s celebrated best-val 0.3578 came from ~102 evals, where pure
noise alone would produce ≈0.3662.

**Consequence: the "we ship last.pt and nobody chose that" trap is probably not
a bug at all** — best.pt was mostly a lucky reading. Root cause is
`--eval-batches 25`; raise it before trusting checkpoint selection again.

---

## 7b. Embedding dimension -- measured, 64 is free

Four arms at an identical 1.25h fine-tune budget from ctx5_pre.pt, each scored on
top10_200k_5games. Compare arms to the **128 control (0.7726)**, never to the
shipped 0.8570 -- that gap is training budget, not architecture.

| d_embed | r@1 | r@10 | r@100 | gallery | verdict |
|---|---|---|---|---|---|
| 128 | 0.5540 | 0.7726 | 0.9006 | 143 MB | control |
| **64** | 0.5426 | **0.7706** | **0.9058** | **72 MB** | **free** |
| 32 | 0.4874 | 0.7288 | 0.8894 | 36 MB | -0.044, real |
| 16 | 0.3056 | 0.5782 | 0.8004 | 18 MB | collapses |

64 costs 0.002 r@10 -- inside noise -- and is slightly BETTER at r@100, for half
the gallery and half the search matmul. Take it when deployment cost matters.
**It is not an accuracy lever:** fewer dimensions cannot carry more information,
so the best case was always "free", never "better".

---

## 7c. `cpl` — centipawn-loss training (IN DEVELOPMENT, not yet run)

A graded move-prediction target instead of a one-hot one. Cross-entropy calls all
31 non-played candidates equally wrong; CPL says a move of the same QUALITY as
the human's is nearly a right answer and a blunder is catastrophic. The intent is
to teach the trunk a player's ERROR PROFILE rather than their exact moves --
motivated by the elo head being our strongest signal.

Loss: `sum_i p_i * (WP_i - WP_j)^2`, where WP is **win probability**, not raw
centipawns. Centipawns are not linear in importance (300cp lost from a won
position is nothing, from an equal one is decisive) and raw cp-squared is
dominated by the tail, which the model already avoids.

**Deliberately small model for this arm**: d_model 256 / 8 layers / 5 game slots,
i.e. the shipped `ctx5_pre` shape. CPL is the only variable; confounding it with
the ctx10 params/context bet would make a result uninterpretable.

**Files.** `cpl_label.py` (corpus builder), `cplcorpus.py` (reader + win-prob
transform), `runpod/cpl_run.sh` (pod runner), `profile_cpl{,2,3}.py` (the
profiling that set the parameters). Loss lives in `model.py:cpl_loss`, wired
through `multitask_loss(..., w_cpl=)`; loader join is in
`multigame_data.py._one_game`.

**How to run it.** `--cpl-dir <corpus> --w-cpl 1.0` on `train_multigame.py`.
**`--w-cpl 0` makes the CPL code fully inert** (the `cpl` key does not even
appear in the stats dict), so the control arm is the existing recipe rather than
a re-implementation of it -- run both from the same command with one flag
changed.

**Verified end to end** on a smoke corpus: 8/8 supervised plies of a labelled
game joined with 0 unresolved candidates, and 6 training steps drove the term
0.03317 -> 0.02627 with gradients flowing.

**Two correctness details that are easy to get wrong.** A corpus is bound to the
shard it was built from -- game indices repeat across shards, so
`CplCorpus.assert_shard()` refuses a mismatch instead of silently joining evals
to the wrong positions. And an unlabelled ply is DROPPED from the term, not
zero-filled: a ply with no engine data carries no information about move quality
and must not vote for uniform probabilities.

**Corpus format** (`--out` dir): `ply_game.npy`, `ply_idx.npy`, `offsets.npy`,
`moves.u16`, `evals.i16`, `manifest.json`. Stores **absolute eval per legal move,
mover POV**, clipped +-2000cp. CPL is `max(eval) - eval`; the reverse is not
recoverable, which is why eval is what gets stored. EVERY legal move is stored,
not just the 32 the loader draws, so changing candidate sampling later does not
invalidate the corpus.

**Measured parameters** (`profile_cpl{,2,3}.py`, on real mid-game positions):

| lever | finding |
|---|---|
| depth | 6. Deeper only sharpens the 0-30cp band, where label noise is 106cp even at depth 8 -- noise-dominated at any affordable depth |
| multipv | 32, and it is the DOMINANT cost: 6.1ms at mpv1 -> 70.3ms at mpv32, roughly linear past 4. Depth is the secondary term |
| threads | 1 per engine. 12.4 pos/s/core at Threads=1 vs 4.0 at Threads=4 |
| TT reuse | 1.35x on consecutive plies of one game (free if walked in game order) |
| label quality | vs depth 12: rank rho 0.754, blunder agreement 82.5%, RMS 126cp |
| candidate spread | median CPL 144cp, 90th pct 537cp; only 11% within 30cp of best; 34% are >=300cp blunders |

**STATUS: corpus generating.** `cpl-label` is running 7 engines (the real cgroup
quota) at 2.4 games/s over 61,161 games, ~$1.35 for the 5h cap. Four launches
were needed; the first three are recorded because each failure mode is one that
looks like something else:

1. First attempt materialised a 4.6M-element task list up front, leaving 94
   engines idle at 13% CPU while one core built a list. Fixed by streaming tasks
   from a generator.
2. Second ran at 114 plies/s with 3 of 94 engines in R state. Looked like a
   blocked result pipe, so workers were rewritten to read the memmap and write
   their own output shards -- nothing large crosses a process boundary now.
   That was a real improvement but NOT the cause.
3. Third hit `TimeoutError` in `protocol.initialize()`: 94 simultaneous UCI
   handshakes blow python-chess's 10s default. Fixed with staggered spawns, a
   120s timeout, retries, and 16MB hash. Also real, also not the cause.
4. The actual cause was the cgroup quota (see traps): 94 engines on 7.65 cores.
   At 7 workers the quota saturates and throughput is 4x the single-worker rate.

Worth knowing for a v2: Python UCI parsing now outweighs the search (407% vs
358% CPU). A minimal UCI client that regexes out only `multipv/score cp/pv` and
ignores the rest should roughly double throughput -- python-chess builds a full
`InfoDict` per PV, 32 per call, for a 30ms search.

---

## 8. Measured facts worth not re-deriving

- Bullet games average **33.2 own moves** (~66 plies) against the 160-ply cap —
  so raising `--max-len-per-game` buys padding, not context. The real context
  axis is `--max-games`.
- Hand-crafted features are **100% redundant with cosine**: best fusion weight
  measured on held-out queries is 0.00. Piece-type fractions have ICC ≈ 0.05.
- A shared 6-ply opening is 26× evidence vs a random player, but cosine
  neighbours are already **5.64×** enriched for shared openings.
- PCA-truncating the shipped gallery: r@10 0.620 (128-d) → 0.570 (64) → 0.463
  (32) → 0.235 (16). Participation ratio **35.9 effective dims** of 128.
- The 10× model (79.76M params, `d_model 512 / 24 layers / 8 heads`, 10 slots)
  runs at **4.69 steps/s at batch 24** on an A6000 = 0.18M position-forwards/s,
  vs `ctx5_pre`'s 0.50M/s. Equal-work pre-train ≈ **48 GPU-h**. Needs **≥32 GB**
  VRAM at batch 24 (27.9 GB used), so it does not fit a $0.27 A5000 at that batch.

---

## 9. Unknown / not measured

Listed so nobody assumes otherwise:

- `ctx5_pre`'s learning curve before step 816,000. Never uploaded.
- `ctx5_ft2`'s `best.pt` weights. Written on the pod, never uploaded, pod gone.
- Whether the 10× model overtakes the 7.9M model at any step count.
- ~~Whether `best.pt` beats `last.pt` on the product metric.~~ **RESOLVED
  2026-08-14: `last.pt` wins, 0.8346 vs 0.8276.**
- Why the first `ctx10-bet` died at ~20 min. OOM is the leading hypothesis
  (20.5 of 24.5 GB with sequence length varying by game length), but the log was
  destroyed by the `NOVOL_ALL_DONE` bug in §1, so this is inference, not
  evidence.
- RunPod balance: `/v1/billing/balance` returns HTTP 400, so spend has been
  estimated from `costPerHr` × runtime, never read directly.
