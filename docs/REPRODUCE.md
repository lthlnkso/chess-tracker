# Reproducing the results

Three tiers, by how much they cost you.

| tier | needs | time | cost | reproduces |
|---|---|---|---|---|
| 1 | laptop | ~2 min | free | the data pipeline and its correctness checks |
| 2 | any GPU | ~20 min | ~$0.15 | pre-training end to end on one small month |
| 3 | GPU + disk | ~12 h | ~$12 | the headline identification numbers |

---

## Tier 1 — data pipeline (free)

```bash
bash scripts/reproduce_quickstart.sh
```

Equivalent to:

```bash
python ingest.py \
  --url https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst \
  --out data/2013-01
python verify.py data/2013-01
```

Expected: `118,734 games / 8,110,498 plies / 4,856 players`, 20 MB on disk, and
four passing correctness checks. These are exact — the pipeline is deterministic
given the same archive.

`verify.py` checks the POV flip against python-chess's own `board.mirror()` on
random positions, replays every sampled game's packed moves for legality, and
confirms the "my turn" plane has the right parity from each seat.

## Tier 2 — pre-training on one month (~20 min on a GPU)

```bash
python train_successor.py --shard data/2013-01 --out ckpt/demo \
    --steps 6000 --workers 16 --no-rights
python eval_successor.py --ckpt ckpt/demo/best.pt --shard data/2013-01
```

Expected, scored against *every* legal successor rather than the 16 sampled in
training:

```
top-1   ~0.37   (uniform-over-legal chance ~0.058)
top-3   ~0.62
median rank of the move actually played   1
```

Training-time accuracy reads ~0.48 because it is scored against ~15 sampled
distractors. **0.37 is the honest number**; `eval_successor.py` produces it.

Drop `--no-rights` for the 13-plane encoding with castling and en passant. It
makes no measurable difference — that is a finding, not an oversight.

## Tier 3 — the full six months

Sizes and timings measured on an RTX 3090 (32 vCPU, 125 GB RAM), which won on
cost-per-sample; see `docs/FINDINGS.md`.

### 3a. Pick a time control

```bash
python runpod/measure_month.py --month 2026-06 --gb 1 --total-gb 28.2
python survey.py --month 2026-06 --gb 2
```

`survey.py` reads headers only — about 4x faster than a full parse — and reports
`60+0` at 27.4% of games, the single most common control.

### 3b. Ingest (~30 min per month, 32 vCPU)

```bash
for M in 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06; do
  python ingest.py \
    --url https://database.lichess.org/standard/lichess_db_standard_rated_$M.pgn.zst \
    --out data/$M --workers 30 --time-controls '60+0'
done
```

~3.5 GB per month. The `--time-controls` filter is applied to headers *before*
movetext is parsed, which is what keeps this to 30 minutes rather than ~1.7 hours.

### 3c. Build the filtered cross-month index (~5 min)

```bash
python combine.py --shards data/2026-* --out data/combined --min-games 100
```

Maps per-month usernames into one global id space and keeps players with 100+
games across the whole period. Copies no move data — shards stay put and are
memory-mapped.

Expected: `1,286,522 players total, 269,407 with >= 100 games`,
`255,616,946 game-sides`, median 414 games per kept player.

### 3d. Pre-train (2.5 h)

```bash
python train_successor.py --combined data/combined --out ckpt/prod \
    --max-hours 2.5 --steps 100000000 --lr 1.5e-4 --warmup 1000 \
    --workers 30 --batch 128 --eval-every 5000 --no-rights
```

Expected final held-out next-state accuracy **~0.554**.

**Do not raise `--max-hours` without lowering `--lr`.** With `--max-hours` the
cosine is driven by elapsed time, so a longer budget holds the LR near peak for
tens of thousands of steps. At 5 h with `lr 3e-4` the run peaked at step 30k and
then degraded on both train and val.

### 3e. Identification fine-tune (4 h)

```bash
python finetune_id.py --combined data/combined \
    --pretrained ckpt/prod/best.pt --out ckpt/id \
    --loss supcon --max-hours 4 --p 32 --k 4 --workers 30 \
    --collapse-after 2000
```

Watch the `gap` column (positive-pair minus negative-pair cosine). It should climb
past 0.1 within the first few thousand steps; ours ended at 0.162. If it sits near
zero the embedding has collapsed and the run aborts at step 2000 by design.

`--loss triplet` is kept for comparison and **will** collapse. That is the point.

### 3f. Evaluate

```bash
python identify_eval.py --ckpt ckpt/id/last.pt --combined data/combined \
    --out ckpt/id/eval.json --max-test-players 20000 --workers 30
python plots/plot_identification.py --data ckpt/id/eval.json
```

Expected against a 20,000-player gallery (chance 0.005%):

| query | top-1 | top-10 | top-100 |
|---|---|---|---|
| 1 game | 0.110 | 0.282 | 0.563 |
| 10 pooled | 0.883 | 0.976 | 0.997 |
| all | 0.959 | 0.993 | 0.999 |

All pooling rows are scored on the **same** players (those holding at least 10
query games). Without that, "10 pooled" silently drops players with the least
evidence and looks better than "all" for the wrong reason.

---

## Notes on exactness

- Tier 1 is exact.
- Tiers 2 and 3 are stochastic: dataloader worker ordering is not deterministic, so
  expect ±0.01 on accuracies. Two runs of the same pre-training config differed by
  0.008 at step 5k and converged to within 0.002 by step 10k.
- Lichess archives are immutable, so the corpus itself is stable.
- Wall-clock times assume ~30 dataloader workers. The pipeline is CPU-bound on
  candidate generation, so fewer cores costs throughput roughly linearly.

## Renting the same hardware

`runpod/` holds the pod lifecycle scripts actually used: `make_pod.py` (create and
wait for sshd), `bench_gpu.py` + `runpod/bench_one.sh` (cost-per-sample
benchmarking), and `watchdog.sh` (terminate on completion or stall so an
unattended failure cannot bill overnight).

Set `RUNPOD_API_KEY` in `.env`. That file is gitignored — do not commit it.
