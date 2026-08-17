# Branches not taken

Every fork where the other path is still live. When something plateaus and the
obvious moves are exhausted, this is the list to come back to.

Each entry records **what we chose, why, what the alternative costs, and what
evidence would flip it.** A branch is only listed if the alternative is still
viable — settled negatives (colour splitting, hand features, XGBoost re-ranking)
live in [`TRAINING.md`](TRAINING.md) and should not be reopened without new
information.

Ordered roughly by *expected value if revisited*, not by date.

---

## 1. Stage C measures a 200k proxy, not the production gallery

**Chose:** `top10_200k_5games` — a 200,000-player gallery from ONE shard, because
it is the only metric comparable to every historical row in the registry.

**Alternative:** build the real gallery with `union_gallery.py` (558,735 players,
six months) and measure r@10 against it, with a temporal split — gallery from
2026-01..05, queries from 2026-06 — which is both leakage-free and exactly how
deployment works.

**Why this is first on the list:** the two numbers are far apart. The shipped
model scores **0.8570** on the proxy and roughly **0.58–0.62** against the real
558k gallery, and even that is leakage-inflated. We have been optimising a number
that is ~0.25 higher than the product's.

**Cost:** one `union_gallery` pass (~$1) plus an `eval_union.py` that does not
exist yet. Also needs `ctx5_ft2` re-measured on the same protocol or the
comparison is apples-to-oranges again.

**Flip when:** always. This should be done regardless of how ctx10 lands.

---

## 2. P=16/K=4 vs P=24/K=3 in the contrastive batch

**Chose (2026-08-16):** let the ladder pick, which lands on **P=16, K=4**.
Rationale: at rank 13 of 550k we are tuning, not fixing, and the conservative
branch changes one variable instead of two. K=4 is the value the loss sweep ran
at.

**Alternative:** pin **P=24, K=3** (72 bundles, fits the same memory as 16×4=64).
Keeps all 24 distinct identities per batch — ~2,500 negative pairs vs ~1,900 —
at the cost of halving positives per player (3 vs 6).

**Why it matters:** P is how many people the model must separate per step, and
our failure mode is ordering *within* a shortlist that already contains the
answer. That is a discrimination problem, which is P's job.

**Cost:** one contrastive re-run, ~$3–6. The trunk is unaffected, so stage A does
not repeat.

**Flip when:** stage C comes back at or below 0.8570 and we suspect the recipe
rather than the model. This is the first thing to try in that case.

---

## 3. Only ~10% of training uses the full 10-game context

**Chose:** `vary_games=True` default — bundle size drawn uniform 1..10, mean 5.5,
so **only 10% of samples exercise all ten slots** (vs 20% for ctx5's five).
Deployment does not know how many games a visitor will play, so the model must
handle every length.

**Alternative:** bias the sampler toward larger k (e.g. uniform 5..10), so the
long context we paid for gets the majority of the training signal.

**Cost:** a one-line change in `multigame_data.py:178` plus a contrastive re-run.

**Flip when:** the k=10 number in stage C disappoints while k=5 looks fine. That
pattern would say the long-context path is undertrained rather than useless.

---

## SETTLED (negative): gallery centroid depth

**Tested 2026-08-17, dead.** `--gallery-games 64` truncates to whole k=10
bundles, so centroids use exactly 60 games and 59% of the gallery piles up
there. The obvious question was whether a deeper centroid is a cleaner target.

Two arms, same shard, same 50,000 players, same 2,000 held-out queries:

| --gallery-games | mean centroid | r@1 | r@10 |
|---|---|---|---|
| 60 | 44.6 games | 0.9250 | 0.9830 |
| 128 | 73.1 games | 0.9245 | 0.9820 |

**1.64x the centroid depth bought nothing** — deltas of −0.0005 (0.08 sigma) and
−0.0010 (0.35 sigma). The centroid saturates well before 60 games, so the
inherited cap is not costing us anything and a full rebuild at 128 (~10 h,
~$2.50) would be wasted money.

Note this is the GALLERY side only. Query-side depth is a completely different
story and still the biggest lever we have: r@10 goes 0.790 at ten query games to
0.867 at thirty.

---

## 4. d_embed 128 vs 64

**Chose:** 128, deferred the change.

**Alternative:** 64. **Measured free** — r@10 0.7706 vs 0.7726 at matched budget,
0.34 SE apart, and r@100 slightly *better*. Halves the gallery (143 → 72 MB) and
the search matmul.

**Why deferred:** it buys nothing perceptible today (local demo, one user), and
taking it costs a fine-tune plus a gallery rebuild. It is a deployment-cost
lever, never an accuracy one — fewer dimensions cannot carry more information.

**Flip when:** the gallery goes to ~2M players (512 MB → 256 MB, the difference
between fitting a cheap VPS and not), or whenever we are rebuilding the identifier
anyway — the decision is free at that moment.

---

## 5. Elo conditioning (stage A2) not run for ctx10

**Chose:** identifier-only. ctx10 will have an elo *head* (predicts rating) but
no elo *conditioning* (accepts a requested rating and plays like it), so it
cannot drive the bot.

**Alternative:** run stage A2, `--init ctx10_pre.pt --elo-cond --elo-drop 0.1`,
mirroring `ctx5_pre` → `ctx5_pre_elo`.

**Why deferred:** ~17 GPU-h (~$4.30) on a trunk we would abandon if ctx10 loses.
It forks off `ctx10_pre.pt`, which stage A already uploads, so it can start any
time without disturbing anything.

**Flip when:** ctx10 wins stage C and we want one coherent model family. Until
then `ctx5_pre_elo` drives the bot perfectly well.

---

## 6. Multi-task weights `w_time` / `w_elo` both 0.3

**Chose:** the inherited defaults. Never swept.

**Alternative:** raise them. The trunk's gradient is
`∇L_move + 0.3·∇L_time + 0.3·∇L_elo`, and the two 0.3-weighted heads are the
*player-level* ones — the ones that resemble identification. Raising them tilts
the representation toward player structure at some cost to move accuracy.

**The tension:** the bot wants the opposite, and it forks off the same trunk. A
trunk tuned hard for identification makes a worse opponent.

**Flip when:** stage C is close but short. This is a cheap knob that points
directly at the product task, and the observed asymmetry (elo −23%, time +0.006,
move +0.013 with 2x context) is evidence the player-level heads respond to
exactly the changes identification cares about.

---

## 7. `min_games` coupled to slot count

**Chose:** left `--min-games` at the slot count (10), so eligibility is "≥10
clocked games in the month". Measured cost on the real shard: **2% of games**
(263,823 players, 98% of games) — and it matches the product, whose gallery
targets active accounts.

**Alternative:** decouple. `--min-games` now exists on `train_multigame.py`
(added 2026-08-16, default 0 = use max_games), and `vary_games` already clamps
bundle size to what a player has, so a lower threshold costs nothing structural.

**Flip when:** we want to identify *casual* players. The current training pool
skews to the active half by construction, so a visitor with 12 games a year is
out of distribution.

---

## 8. Six months of data, not more

**Chose:** `data/mt/2026-01..06` — 39.8 GB, 145M games, 608,504 eligible players.

**Alternative:** ingest more months. Lichess publishes monthly dumps going back
years; `ingest.py` costs ~36 min per month.

**Why not yet:** we are not data-limited for *pre-training* — 3.20 G plies per
shard against a 24M-parameter model is ~133 plies per parameter, roughly 7x what
a model this size can absorb.

**But:** the contrastive stage is different. It cycles the *player* set every
~21 minutes, so the same 608k identities are seen hundreds of times. If stage B
overfits, more months means more distinct people — the opposite conclusion from
stage A, and the one that actually matters for identification.

---

## 9. Verifier: more steps vs harder negatives

**Chose:** cut v2. It saturated at AUC 0.8607 against in-batch neighbours but
scores only **0.6058** against the deployed top-100 of 558,735, and reranking
*loses* to plain cosine (r@10 0.780 vs 0.787) under every fusion tried including
calibrated Bayesian LLR. `play/bayes_calib.json` says `recommended=none`.

**Alternative:** `verify3.py --shortlist`, training on negatives drawn from the
real cosine shortlist rather than from neighbourhood batches. That is the
diagnosis in the registry — the fix is harder negatives, not more steps.

**Status:** yours, already run once (2026-08-14). Not folded into the demo.

---

## 10. Demo caps fusion at 15 games

**Chose:** `MAX_BUNDLES = 3` (`play/server.py:93`), so a visitor's games are
encoded as up to three 5-game bundles whose similarities are summed — everything
past 15 games is discarded.

**Alternative:** raise the cap; or, with ctx10's ten slots, encode ten games
*jointly* in one bundle rather than as two fused fives. Joint encoding measured
**+12%** over separate-and-average.

**Flip when:** ctx10 ships. This is the mechanism by which its extra context is
supposed to pay off in the product, and it needs a server change to realise —
the model alone does not do it.
