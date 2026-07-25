# Findings

Where every number in the README came from, plus the failures worth recording.
This is the working record of the project, kept deliberately including the
negative results and the mistakes.

## Why the bitboards are not written to disk

The natural reading of "convert games to sequences of bitboards" is to
materialise the tensors. That does not survive contact with the scale in the
mission statement:

| representation | per game | Jan 2013 (119k games) | a 2025-size month (~100M games) |
|---|---|---|---|
| uint8 planes, both POVs | ~160 KB | ~19 GB | ~16 TB |
| bit-packed planes, both POVs | ~20 KB | ~2.4 GB | ~2.0 TB |
| **packed moves + metadata (this repo)** | **~169 B** | **20 MB (measured)** | **~17 GB** |

A position is 18 planes x 64 squares, a game averages 68 plies, and every game is
needed from both seats — that is ~1000x inflation over the move list that
generates it. So the shard stores moves, and `bitboards.game_to_bitboards`
replays them into POV tensors in the dataloader.

Measured on the pod: 16 loader workers sustain **437k positions/s** (6.2k
game-sides/s at `max_len=160`). A small transformer will not consume batches
anywhere near that fast, so expansion is free — the GPU stays the bottleneck,
which is where we want it.

If a training run later does turn out to be dataloader-bound, the fix is to cache
expanded tensors for the *sampled subset* being trained on, not for the corpus.

## Point of view

The mission needs boards "from the perspective of the player the game is
attributed to". `to_pov` mirrors ranks and swaps the colour blocks, so the
attributed player's pieces are always planes 0-5 starting on rank 0, and plane 12
means "my turn" rather than "White to move". Each game therefore yields two
training samples, one per seat.

This is checked against python-chess's own `board.mirror()` on random positions
in `verify.py`, so the flip is not just self-consistent.

## Layout of a shard

`ingest.py` writes a directory per month:

- `moves.u16` — every game's moves concatenated, `uint16` each:
  bits 0-5 from-square, 6-11 to-square, 12-14 promotion piece type.
- `meta.npy` — one fixed-width row per game: `offset` into `moves.u16`, `nply`,
  `white_pid`, `black_pid`, both Elos, `result`, `termination`, time control
  (`tc_base`/`tc_inc`), `date`.
- `players.txt` — username per line; line number is the `pid`.
- `manifest.json` — source URL, counts, and `complete` (false if `--limit` cut
  the pass short).

`moves.u16` is memory-mapped, so random access during training does not need the
corpus in RAM.

## Running it

The archive is streamed and decompressed in flight — it never lands on disk,
which is what makes the multi-GB recent months tractable. Note lichess ships
**zstd** (`.pgn.zst`), not gzip.

```bash
/workspace/venv/bin/python ingest.py \
  --url https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst \
  --out /workspace/data/2013-01
/workspace/venv/bin/python verify.py /workspace/data/2013-01
```

Games are dropped if they are non-standard (a `FEN`/`Variant` header), have no
decisive/drawn result, were abandoned or ended in a rules infraction, contain
movetext we cannot parse, or are shorter than `--min-plies` (default 10 — very
short games carry no style signal).

## The corpus

Six months of lichess (2026-01 .. 2026-06), filtered to the most common time
control, **`60+0` bullet** (27.4% of all games, confirmed by a header-only survey):

| | |
|---|---|
| games ingested | 134,721,048 |
| players seen | 1,286,522 |
| players with >= 100 games | **269,407** |
| game-sides kept | 255,616,946 |
| median games per kept player | 414 |
| on disk | ~21 GB |

Ingest ran ~30 min per month at 30 workers. Jan 2013 (118,734 games, 4,856
players) remains the fast fixture used for benchmarks and correctness checks.

## Infrastructure

Final run: RunPod **RTX 3090** (community, $0.22-0.50/hr) in **EU-CZ-1**, 32 vCPU /
125 GB, with a 120 GB network volume at `/workspace`. Chosen on measured
cost-per-sample, not list price — see Step 3. Total spend for the whole project was
about $12. The pod is terminated; the volume holds the shards and checkpoints.

Six things about this setup are not obvious and each cost real time:

- The `runpod/pytorch` images do not start `sshd`. The pod needs an explicit
  `dockerStartCmd` that installs `$PUBLIC_KEY` and execs `sshd -D`, or the
  container exits and SSH reports *"container is not running"*.
- `ssh.runpod.io` (the proxy) demands a PTY, so it needs `ssh -tt` and cannot carry
  `rsync`/`scp`. Use the pod's direct host/port; both change with every new pod.
- `/workspace` is MooseFS-backed and refuses `chown` **and `chmod`**, so `rsync -a`
  fails (hence `--no-o --no-g`) and file permissions there are advisory.
- `os.cpu_count()` reports the *host's* cores (128-256), not the allocation (21-32).
  Defaults derived from it oversubscribe several-fold.
- **Network volumes are locked to one datacenter, and that dictates GPU choice.**
  US-IL-1 has only 4090s; EU-CZ-1 has 3090/4090/5090. Pick the datacenter from the
  GPU list *before* ingesting data into it.
- `pkill -f <pattern>` matches **your own SSH session** when the pattern appears in
  its command line. It killed a live connection and a driver script here. Resolve
  PIDs first, then kill by PID.

## Step 2: the model

Primary architecture — **successor-state scoring**. History is a sequence of
board states `(T, 8, 8, 8)` in the attributed player's frame; a candidate
successor `(8, 8, 8)` is scored against it to a single logit, higher = more
likely to be what actually came next.

    history (B,T,8,8,8) --> causal transformer --> h_t  (B,D)
    candidate (B,P,C,8,8,8) --> shared board MLP --> e   (B,P,C,D)
    logit = <h_t, e> / sqrt(D)

Candidates come from real move generation, so the model never spends capacity
learning legality. The dot product means C candidates cost one matmul rather than
C transformer passes. Training samples 16 candidates per supervised ply (true
successor plus up to 15 other legal ones, shuffled so list position carries no
signal) and takes cross-entropy over that set.

The 8 planes are 6 colour-agnostic piece-type planes plus "my pieces" / "their
pieces". Because the frame is already POV, those last two *are* the me/them
distinction. **What does not fit in 8 planes: castling rights and the
en-passant square.** Legality is unaffected (the generator handles it), but two
positions differing only in those rights are identical to the model — worth
revisiting if it plateaus.

### Results (Jan 2013, 7.4M params, 6000 steps, 10.7 min on the 4090)

Scored against **every** legal successor, not the 16 seen in training:

| metric | value |
|---|---|
| top-1 | **0.372** (uniform-over-legal chance 0.058 — 6.4x) |
| top-3 | 0.625 |
| median rank of the move actually played | 1 |
| mean percentile of the played move | 86.1 |
| legal moves per ply | mean 30.4, median 32, max 71 |

Reported training accuracy (0.479) is against ~15 sampled distractors and is the
flattering number; 0.372 is the honest one. `eval_successor.py` produces it.

### Baseline: 4096-way policy head

`model.ChessTransformer` + `train.py` are the first thing I built — an 18-plane
encoder with a `from x to` softmax over all 4096 square pairs. It reaches val
top-1 0.352 / top-5 0.716 (7.7M params, 5.4 min).

Do not read 0.372 vs 0.352 as a clean win: the policy model is scored
**unmasked** over 4096 mostly-illegal pairs, so the two are not on the same
footing. The defensible claim is that the successor model matches or beats it
while using fewer planes and less state. Kept as a baseline, not deleted.

## Step 3: scaling to six months

### Which GPU

Benchmarked 3090 / 4090 / 5090 on the real training loop, each pod stream-ingesting
Jan 2013 to its own disk so no shared volume was needed. Two numbers per card:
`gpu_only` replays one resident batch (pure device throughput); `end2end` includes
the dataloader.

| GPU | $/hr paid | gpu_only | end2end | $/M game-sides |
|---|---|---|---|---|
| **RTX 3090** | 0.22 | 23.3 it/s | 8.4 it/s | **0.057** |
| RTX 5090 | 0.99 | 63.8 it/s | 14.5 it/s | 0.148 |
| RTX 4090 | 0.69 | 15.2 it/s | 5.6 it/s | 0.266 |

The 3090 wins by 2.6x over the 5090 and it is not close. Three caveats that the
figure states and that matter more than the ranking:

- **Every run was 63-77% dataloader-bound** at the benchmark's 16 workers, against
  hosts with 96-256 vCPU. The `end2end` column therefore measures my worker cap as
  much as the GPU. The 3090 still wins on `gpu_only` cost, so the conclusion holds,
  but the margins are not the real margins.
- **Prices are what RunPod actually charged**, and the tiers differ: the 3090 was
  only obtainable on community cloud, the 4090/5090 only on secure. That is a real
  cost difference, not a like-for-like GPU comparison.
- **The 4090 landed on a slow host.** At 7.8M parameters the kernels are tiny and
  launch overhead dominates, so host CPU leaks into `gpu_only`. A 4090 being slower
  than a 3090 on-device is not credible as a general claim about the silicon.

### Encoding for the production run: 8 planes

The six-month run uses the **8-plane** encoding (`--no-rights`). Adding castling
rights and en passant (13 planes) was measured on Jan 2013 and made no
difference — full-legal top-1 0.370 vs 0.368, and the 13-plane run actually
finished with *higher* val loss (1.630 vs 1.615).

The reason is structural: a causal model reading the whole game can infer
castling rights from whether the king or rook has already moved, so the explicit
planes are redundant with the history. En passant is live for one ply in a small
fraction of games. 38% fewer input features for the same accuracy.

Both encodings remain selectable, and the plane count is stored in the
checkpoint so downstream loaders match it rather than assume it.

### The corpus

Six months (2026-01 .. 2026-06), streamed and filtered to the single most common
time control, **`60+0` bullet**, chosen from a header-only survey (`survey.py`).
Filtering on headers means movetext is never parsed for games we discard, which is
where nearly all the ingest cost lives — ~30 min per month at 30 workers rather
than the ~1.7 h an unfiltered pass measured.

`combine.py` then maps per-month usernames into one global id space, counts each
player's games across all six months, and writes an index of the
`(shard, game, seat)` triples worth training on. It copies no move data — shards
stay put and get memory-mapped, and the player filter is a numpy pass over
metadata rather than a second parse.

## Step 4: player identification

Metric learning on top of the pre-trained trunk. `PlayerEncoder` reuses the
transformer (its modules are named identically, so the checkpoint loads with
`strict=False`), drops the candidate scorer, and attaches a fresh 128-dim
projection head. Pooling is over the player's **own turns** — their choices are
the identity signal, the opponent's are not.

**Split.** Players are hashed 80/20. A game-side trains iff its own player is a
train player; a game is a *test* game only when both seats are test players.
Two things worth stating plainly:

- No train sample is ever a test player's own game-side — verified by assertion,
  not assumed.
- The default rule still lets a test player's *games* appear in training labelled
  as their **opponent**. Their label never appears, so this is indirect, but it is
  not nothing. `--strict-split` withholds any game touching a test player (~21%
  fewer training rows in simulation) so the difference can be measured.

**Loss.** Batch-hard triplet (Hermans et al.) over P=32 players x K=4 games per
batch, soft margin. Random triplets go slack once the easy ones are solved, which
is why the `PKSampler` exists — every anchor needs a same-player positive and a
different-player negative present in its own batch.

**Evaluation** (`identify_eval.py`), test players only. Each contributes 80% of
their game-sides to a centroid and 20% to a query pool; every query is matched
against every test player's centroid by cosine similarity, reporting recall@1/10/100
plus median rank. Then repeated with queries pooled at 1 / 10 / half / all — how
much evidence does identification actually need?

The recall computation was checked against random embeddings and lands exactly on
chance (0.0025 measured vs 0.0020 theoretical at 500 players), so the metric is
calibrated rather than accidentally flattering.

Identification never needs candidate successors, so `EmbedDataset` skips move
generation entirely — roughly 2x cheaper per sample than the pre-training loader,
which was the bottleneck.

### Loss: SupCon, not triplet

Batch-hard triplet **collapsed within 40 steps**, twice: every embedding mapped to
one point, `d_pos == d_neg`, loss pinned at `softplus(0) = ln 2 = 0.693`. That is
not a tuning failure — it is triplet's *global minimum*, and a network reaches it
by the trivial route of ignoring its input and emitting a constant.

Supervised contrastive loss inverts the incentive. Under collapse every similarity
is identical, the softmax is uniform, and the loss is its **maximum** `ln(B-1)`.
Verified numerically: on a collapsed batch triplet scores 0.6934 (its floor) while
SupCon scores 4.8442 = exactly `ln(127)` (its ceiling).

Two caveats found while checking this, both worth knowing:

- SupCon's collapse point is a **saddle, not an attractor**. Gradient descent will
  not fall into it, but from *perfect* symmetry it also cannot climb out — the
  gradient is zero by symmetry. Irrelevant in practice (a fresh projection head is
  asymmetric) but it means "SupCon can't collapse" is too strong a claim.
- On perfectly separated clusters SupCon bottoms out at `log(K-1)`, not 0, because
  the softmax must spread mass over the K-1 positives. With K=4 that floor is
  1.0986 — worth knowing before reading a loss of 3.7 as failure.

`finetune_id.py` aborts at step 2000 if the pos/neg cosine gap is still under 0.02,
so a collapsed run dies in minutes rather than burning a night.

### Results

20,000 held-out players, none seen in training. Chance for top-1 is 0.005%.

| query | top-1 | top-10 | top-100 |
|---|---|---|---|
| **1 game** | **0.110** | 0.282 | 0.563 |
| 10 games pooled | 0.883 | 0.976 | 0.997 |
| half their games | 0.874 | 0.960 | 0.990 |
| all their games | **0.959** | 0.993 | 0.999 |

A single 1+0 bullet game identifies its author out of 20,000 candidates **2,192x
above chance**. Mean percentile of the true player is 96.8 even for single-game
queries — when the model is wrong it is usually still close.

`half` scoring below a fixed 10 games is an artifact of the definition, not of
pooling: test players average ~53 query games but the distribution is skewed, and
for anyone holding 10-19 games "half" is 5-9 games, i.e. *fewer* than the fixed-10
condition.

Trained by a 4-hour SupCon fine-tune (424,226 steps, 51.5M game-sides, ~21% of one
epoch) on top of the 2.5-hour pre-trained trunk, all on one RTX 3090.

## Files

- `ingest.py` — stream a `.pgn.zst` into a shard, parallel across workers.
- `bitboards.py` — 18- and 8-plane encodings, POV flip, move packing, replay.
- `verify.py` — correctness checks plus corpus stats.
- `setup_pod.sh` / `sync.sh` — pod prep and code push.

Successor scorer (pre-training):
- `successor_data.py` — history + sampled legal-successor candidates; the
  multi-shard variant reads a `combine.py` index.
- `model.py: SuccessorScorer` — encoder, candidate encoder, scoring.
- `train_successor.py`, `eval_successor.py`.

Scaling:
- `survey.py` — header-only time-control survey (~4x faster than a full parse).
- `measure_month.py` — what a modern month costs, measured from a prefix.
- `combine.py` — global player ids + the 100+ games filter, no data copied.
- `bench_gpu.py`, `bench_one.sh`, `make_pod.py` — GPU benchmarking and pod setup.

Figures (`plots/`, light + dark, PNG + SVG, all regenerable from `plots/data/`):
- `plot_pretrain.py`, `plot_accuracy.py`, `plot_gpu_cost.py`,
  `plot_identification.py`, `plot_architecture.py`.

Identification:
- `id_data.py` — player split, `EmbedDataset`, `PKSampler`.
- `model.py: PlayerEncoder, batch_hard_triplet`.
- `finetune_id.py`, `identify_eval.py`, `phase_id.sh`.

Policy baseline:
- `dataset.py`, `model.py: ChessTransformer`, `train.py`.

## Not captured yet

Move *timing* (lichess `[%clk]` comments) is probably a strong identity signal
and is currently discarded along with the rest of the PGN comments. Adding it
means one more array per game; worth doing before concluding anything about how
identifiable players are from board states alone.
