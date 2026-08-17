# The deployed demo — exact configuration

The version played on 2026-08-17 that identified a real account at **rank 1 of
558,735 after three games**. This document is the recipe for reproducing it
byte-for-byte.

Read the [Golden test](#golden-test) section first. It is the only way to know a
deployment is the real thing, and it takes ten seconds.

---

## Launch

```bash
.venv/bin/python play/server.py \
  --ckpt      ckpt/final/ctx5_pre.pt \
  --id-ckpt   ckpt/final/ctx10_ft.pt \
  --gallery   play/gallery_ctx10.npz \
  --verifier "" --bayes "" \
  --port 8010
```

**Every flag is explicit on purpose.** The original session relied on
`--ckpt`, `--id-ckpt` and `--gallery` defaults, and those defaults have since
been edited in `server.py`. A future edit to a default must not silently change
what ships. Pass all five.

Startup must print exactly this. It is the cheapest fingerprint available:

```
move model ctx5_pre.pt: step 816000, val move_acc 0.4963562798500061
id model ctx10_ft.pt: step 320000, 10 slots
gallery gallery_ctx10.npz: 558,735 players, k=10
verifier: not enabled (model or pack missing)
serving on http://localhost:8010
```

---

## Artifacts

| file | bytes | sha256 |
|---|---|---|
| `ckpt/final/ctx5_pre.pt` | 31,726,363 | `5416f7a7eab46935aadfe74ddbe2fe97d10477fb293b8b922fa34a6dec2f9379` |
| `ckpt/final/ctx10_ft.pt` | 97,241,235 | `b230917f935d33b12a038257f1b9fdaf85316f9b1b467e371047dce58070b7a8` |
| `play/gallery_ctx10.npz` | 137,162,027 | `24908fbd6598011a8fcda3c211ff9907b6c2e269875c5b27b342482ba5b737b3` |
| `play/index.html` | — | `aba005a9c026b668a260887ae99296aa5694896d83bf6b77ba830ca90a5e9e72` |
| `play/server.py` | — | `179b605a848c56f55f1239ce96bfe1780381012a6d9e692cd0513f1f7b67cd00` |

```bash
shasum -a 256 ckpt/final/ctx5_pre.pt ckpt/final/ctx10_ft.pt play/gallery_ctx10.npz play/index.html play/server.py
```

### Off-machine backups

The two model files and the gallery are untracked by git, so S3 is the only copy
that is not on one laptop. All three are verified present:

| S3 key | local path |
|---|---|
| `final/ctx10_ft_last.pt` | `ckpt/final/ctx10_ft.pt` |
| `final/ctx10_pre.pt` | `ckpt/final/ctx10_pre.pt` |
| `final/gallery_ctx10.npz` | `play/gallery_ctx10.npz` |

The gallery upload **failed** during the original build (the script did not
source `.s3env`), so it lived only on this machine until it was pushed by hand on
2026-08-17 and verified by round-tripping the sha256 — not by trusting the byte
count. Rebuilding it would cost ~4.8 h and ~$1.20.

Bucket `shusq6ritt`, endpoint `https://s3api-eu-cz-1.runpod.io`, region `EU-CZ-1`.

Environment: python 3.14.4, torch 2.13.0, numpy 2.5.1, python-chess 1.11.2.
Repo at `31e5be7` **plus uncommitted working-tree changes** — the checkpoints and
the gallery are untracked, so a clean clone of that commit does **not** reproduce
this. The sha256 table above is the real manifest.

---

## The two models do different jobs

This is the single most common misreading of this system, so it is stated first.

| flag | role | file here |
|---|---|---|
| `--ckpt` | **the bot.** Chooses moves. Plays against the visitor. | `ctx5_pre.pt` |
| `--id-ckpt` | **the identifier.** Never plays. Scores finished games against the gallery. | `ctx10_ft.pt` |

**The ctx10 model has never played a move.** It only embeds completed games. The
bot is the older 5-slot trunk, and that is correct, not an oversight — see
[Deliberate omissions](#deliberate-omissions).

### `ctx5_pre.pt` — the bot

```
step 816000 | slots 5 | d_embed 128 | max_len_per_game 160 | elo_cond False
n_planes 13 | n_extra 2
cfg: d_model 256, n_layers 8, n_heads 8, d_ff 1024, max_len 168, dropout 0.1
val move_acc 0.4964
```

`elo_cond=False` means **the elo slider in the UI does nothing in this build.**
`ctx5_pre_elo.pt` (step 400000, `elo_cond=True`, move_acc 0.5096) supports it and
is a strictly better move model, but it is *not* what was played and swapping it
changes the opponent. Treat that as an untested change, not a free upgrade.

### `ctx10_ft.pt` — the identifier

```
step 320000 | slots 10 | d_embed 128 | max_len_per_game 160 | elo_cond False
n_planes 13 | n_extra 2
cfg: d_model 384, n_layers 12, n_heads 8, d_ff 1536, max_len 168, dropout 0.1
```

Provenance: S3 `final/ctx10_ft_last.pt` → saved locally as `ckpt/final/ctx10_ft.pt`.

Produced by stage B of the ctx10 run:

```bash
python finetune_ctx.py \
  --shard data/mt/2026-01 ... data/mt/2026-06 \
  --ckpt <ctx10 pre-trained trunk> --out ctx10_ft \
  --loss ms --p 16 --k 4 --lr 3e-5 --warmup 200 --workers 24 \
  --steps 100000000 --eval-every 8000 --eval-batches 100 \
  --patience 12 --amp --compile --max-hours 24
```

Ended on early stopping (12 evals without improvement) at step 320,000. Best
`val_ms` was **0.2160 at step 224,000**.

**`last.pt` was shipped, not `best.pt`** — decided on the measured number, not on
the folklore that last usually wins: r@10 at k=10 was **0.9664 (last)** vs
**0.9628 (best)** on the 200k eval gallery.

---

## Gallery

`play/gallery_ctx10.npz` — 558,735 centroids × 128 dims, float16.

```
ckpt = last.pt | k = 10 | shards = 2026-01 .. 2026-06
centroid_games: mean 45.3, median 60, min 10, max 60
L2 norms: mean 1.0000 (min 0.9999, max 1.0001) | 0 non-finite, 0 all-zero rows
```

Built on a RunPod A5000 by `runpod/union_ctx10.sh`, which runs:

```bash
python union_gallery.py \
  --ckpt <ctx10_ft last.pt> \
  --shards data/mt/2026-01 data/mt/2026-02 data/mt/2026-03 \
           data/mt/2026-04 data/mt/2026-05 data/mt/2026-06 \
  --out gallery_ctx10.npz \
  --k 10 --gallery-games 64 --min-games 13 --batch 192 --workers 24
```

~4.8 h at 149 bundles/s over 2,529,268 bundles (~$1.20).

`--min-games 13` and `--gallery-games 64` are inherited from the previous shipped
gallery so the roster is **identical** to `gallery_2026.npz` — same 558,735
players, same order. That is what makes the two galleries a controlled swap.

### The identifier and the gallery are a matched pair

Centroids are only comparable to query embeddings from **the same weights**.
Mixing them does not degrade gracefully, it collapses: pointing a mismatched
checkpoint at a gallery measured r@1 **0.0000**, median rank 90,036 of 558,735.

Never change one without the other. The gallery records the checkpoint that built
it in its `ckpt` field — check it.

---

## Client requirement — the clock track

`play/index.html` **must record think times for both sides.** The line that does
it:

```js
// index.html:611
if(S.history.length > times.length) times.push(S.bot_ms ?? 1000);
```

`server.py` is stateless; the browser posts the whole game history back, so this
array *is* the timing evidence. The model reads think time at every ply, not just
the visitor's.

**If one side's times are missing, identification collapses silently.** Measured
on the three real games below — nothing else changed, only the opponent's clock
zeroed:

| after game | intact | opponent times zeroed |
|---|---|---|
| 1 | 79 | 6,837 |
| 2 | 20 | 7,570 |
| 3 | **1** | **2,750** |

There is no error, no visual difference, and the candidate list still looks
plausible. This defect was live for weeks and read as "the model isn't good
enough." `server.py:check_clock_track()` now prints a one-time warning:

```
WARNING: every opponent think time is zero over N plies. The client is not
recording that side's clock; identification will be far worse than it should be.
```

It warns rather than repairs on purpose — substituting a plausible number would
hide the same bug next time.

`index.html` is re-read from disk on **every** request (`server.py:815`), so a
client fix takes effect without restarting the server.

---

## Deliberate omissions

Things that are off, and why. None are accidents.

**Verifier / re-ranker — `--verifier "" --bayes ""`.** `load_verifier` returns
early when either path is missing, so these disable the second stage. It was
trained in the ctx5 embedding space and scores AUC **0.6058** against the deployed
shortlist; re-ranking *loses* to plain cosine (r@10 0.780 vs 0.787) under every
fusion tried. `play/bayes_calib.json` itself records `recommended=none`.

**Elo conditioning.** Stage A2 was never run for ctx10 (`docs/BRANCHES.md` #5), so
no ctx10 checkpoint can drive the elo slider. The bot here is `ctx5_pre.pt`, which
also lacks it. Cost to add: ~17 GPU-h, ~$4.30.

**The ctx10 trunk as bot.** Available but not shipped. `ckpt/final/ctx10_pre.pt`
(step 504,000, sha256 `9efdbf3518cf210d554015bd3d8fcbb3a35babfe2dc6c3f7070acafe3d84ac7f`)
is the trunk stage B fine-tuned from, and it is a complete playable model —
`cand_enc`, `time_head` and `elo_head` all present, verified answering 1.e4 with
1...e5. It has the best move accuracy of the three candidates:

| bot | step | move_acc | elo_cond |
|---|---|---|---|
| `ctx5_pre.pt` | 816,000 | 0.4964 | False | **shipped** |
| `ctx5_pre_elo.pt` | 400,000 | 0.5096 | True |
| `ctx10_pre.pt` | 504,000 | **0.5114** | False |

Switching costs nothing that currently works — the shipped bot already has
`elo_cond=False`, so the slider is dead either way. It is held back only because
it **changes the opponent**, and therefore the games a visitor plays, which is
untested. The golden test cannot catch that: it replays fixed games, so it passes
regardless of the bot.

It was very nearly lost. Stage A was interrupted during the pivot to the full
six-month dataset, so `big_run.sh:148` never uploaded `final/ctx10_pre.pt`; only
the 20-minute rolling keeper had it, under the misleading name
`ctx10_pre_partial.pt`. `ft_only.sh:116-122` fell back to that partial and logged
the fact, which is the only reason we can prove what the identifier descends from.
It is now archived under both keys in S3.

Stage A therefore **stopped by interruption, not convergence**, at step 504,000,
and no `ctx10_pre_history.json` was ever uploaded — so there is no curve, and how
much pre-training was left on the table is unknown.

**Colour-split centroids.** `gallery_ctx10.npz` has no `centroids_w` /
`centroids_b`; the server guards on their presence and falls back to combined
scoring. Measured worse: 0.8846 / 0.9558 split-fused from ten games vs 0.8977 /
0.9791 combined from five.

**Hand-crafted features.** Best fusion weight measured 0.00 — fully redundant with
cosine. Not wired in.

---

## Measured accuracy

`depth_probe.py`, n=300, against the full 558,735 gallery. These are the numbers
in `server.py:RECALL_BY_GAMES`, which drive the confidence percentages the visitor
sees:

| games | r@1 | r@10 |
|---|---|---|
| 1 | 0.030 | 0.110 |
| 2 | 0.110 | 0.297 |
| 3 | 0.253 | 0.487 |
| 5 | 0.413 | 0.657 |
| 8 | 0.513 | 0.753 |
| 10 | **0.567** | **0.790** |

```bash
python depth_probe.py --ckpt ckpt/final/ctx10_ft.pt \
  --gallery play/gallery_ctx10.npz --queries 300 --ks 1,2,3,5,8,10
```

**Re-run this whenever `--id-ckpt` or `--gallery` changes**, or the panel quotes
one model's accuracy while a different one answers.

Two honest caveats:

- **Run-to-run spread is ±5 points** at n=200–300 from player-pool sampling alone.
  Do not read these to finer resolution.
- **Every number is from human-vs-human lichess games.** The product feeds games
  played against our own bot. That distribution shift is untested at scale; the
  only evidence is the single account below.

Not to be confused with the **0.9664** from stage C — that is a 200k gallery whose
centroids come from the *same month* as the query. Same queries re-derived at
matched size give 0.869 at 200k and 0.830 at 558k, so most of the difference is
protocol, not roster size. Stage C was the right yardstick for choosing between
models; it was never the product number.

---

## Feedback capture

Three controls added 2026-08-17. None of them touch scoring — the golden test is
unchanged by all of it — but they are the only channel through which this system
ever learns what the right answer was.

### "It's me" — claims

A button on every candidate row. Clicking opens a confirm modal; confirming POSTs
to `/api/claim` and highlights the row. Clicking a claimed row un-claims it
immediately, with no modal — a correction should not need ceremony.

Up to **3 claims per visitor** (`MAX_CLAIMS`), because alt accounts are normal.
The cap is enforced server-side against the visitor cookie, not just in the UI.

`play/claims.jsonl`, one event per line:

```json
{"ts": …, "visitor": "XJzY-…", "name": "someplayer", "claimed": true,
 "games": 3, "rank": 1, "of": 558735, "in_top10": true, "client_rank": null}
```

**`rank` is computed server-side**, by re-running `identify()` against the
claimed name — not taken from the client. The client only knows a rank when the
visitor happened to type their username into the probe box, so it was `null` on
nearly every real claim, which made claims useless as eval data. `client_rank`
keeps whatever the browser thought, purely so the two can be compared.

A claim with a rank is a labelled eval case. A claim without one is just a name.

**Append-only events, not current state.** `load_claims()` replays the file at
startup to rebuild per-visitor claims, so a restart cannot silently drop what
people told us, and claim → declaim → re-claim stays visible as exactly that.
A torn final line is skipped rather than fatal.

`rank` is populated only when the visitor also typed their username into the
probe field; otherwise the server never learned it and logs `null`.

### "None of these are me" — give-ups

Appears under the candidate list once there is at least one game. Asks for a
username, then POSTs to `/api/giveup`.

**This is the most valuable record the demo produces**: the visitor's games plus
the username we failed to surface — a labelled miss, which no shard-derived eval
can manufacture. The games are written to
`play/saved/giveup_<UTC-stamp>_<username>.json` in **exactly** the same shape as
the other saved sets, so existing replay tooling reads them with no conversion.

`play/giveups.jsonl` records the metadata, including the top-10 we wrongly
offered:

```json
{"ts": …, "visitor": "…", "username": "tuxedo_cake", "n_games": 3,
 "file": "giveup_20260817-054424_tuxedo_cake.json", "rank": 2, "of": 558735,
 "in_gallery": true, "client_rank": null, "top": ["someplayer", "tuxedo_cake", …]}
```

`rank` is likewise recomputed server-side, and it is the whole point of the
record: missing at rank 11 and missing at rank 400,000 are different failures
wanting different fixes. `in_gallery: false` is a third kind — the account was
never in the gallery, so no model change would have found them.

Deliberately **not** dev-gated: real visitors are the whole point.

### Dev save button — `--dev` only

```bash
… play/server.py … --dev
```

Adds a bar under the candidate list with a filename box and a Save button,
writing to `play/saved/<name>.json`. Startup prints `DEV MODE: save-games button
exposed`.

Gated in **two** places, and the endpoint gate is the one that matters:

- `window.DEV` is injected into `index.html` server-side only under `--dev`, so
  the bar stays hidden otherwise;
- `/api/save` returns **404** unless `DEV`. The markup ships either way, so
  anyone with devtools could unhide the button — a hidden control is not an
  access control, and this endpoint writes caller-named files into `play/saved/`.

Verified: without `--dev`, `/api/save` → 404 while `/api/giveup` and `/api/claim`
→ 200.

---

## Golden test

The only check that proves a deployment is this exact build. `MAX_BUNDLES = 3`
with 10 slots, so up to 30 games are fused; three games take one bundle.

```bash
python - <<'PY'
import json, urllib.request
games = json.load(open("play/saved/someplayer_live3.json"))
def ask(gs):
    body = json.dumps({"games": gs, "target": "someplayer"}).encode()
    req = urllib.request.Request("http://localhost:8010/api/identify", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return (json.load(r).get("probe") or {}).get("rank")
got = [ask(games[:n]) for n in (1, 2, 3)]
print(got, "MATCH" if got == [79, 20, 1] else "*** MISMATCH ***")
PY
```

Must print `[79, 20, 1] MATCH`.

`play/saved/someplayer_live3.json` holds the three real games from the 2026-08-17
session — 84, 59 and 54 plies; white, black, white. Ranks are out of 558,735.
Keep that file: it is the regression suite.

A mismatch means the identifier, the gallery, or the clock handling changed. The
ranks are exquisitely sensitive — the same games through the *previous* stack
(`ctx5_ft2` + `gallery_2026.npz`) give **74, 9, 1**, close at the ends and clearly
different in the middle.

---

## Known-good deltas, and what they cost

| change | effect | status |
|---|---|---|
| `--ckpt ctx5_pre_elo.pt` | elo slider works; move_acc 0.4964 → 0.5096 | **changes the opponent.** Not what was played. Re-run the golden test — it should still pass, since the bot does not affect scoring of games already played |
| `--id-ckpt ctx5_ft2.pt` + `--gallery gallery_2026.npz` | the previous stack | golden test gives 74, 9, 1 |
| Re-enable verifier | measured worse than plain cosine | do not |

Anything that touches `--id-ckpt` or `--gallery` requires re-running **both** the
golden test and `depth_probe.py`, and updating `RECALL_BY_GAMES`.
