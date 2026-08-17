#!/usr/bin/env bash
# The cpl experiment: does grading a wrong move by HOW wrong it was beat
# treating every wrong move alike?
#
# Arms differ in exactly one number, --w-cpl. Everything else -- seed, data,
# model, steps -- is identical, and w_cpl=0 makes the CPL code fully inert (the
# `cpl` key does not even appear in the stats), so the control is the ordinary
# cross-entropy recipe rather than a re-implementation that might differ.
#
# SMALL model, 5 game slots: the shipped ctx5_pre shape. The ctx10 params/context
# bet is unproven, and confounding two changes would make either result
# uninterpretable.
#
# --cpl-only is on for EVERY arm including the control. The corpus covers ~0.2%
# of the shard, so without it the term fires on 1 ply in 500 and the arms are
# statistically identical. With it, 99% of supervised plies carry engine labels.
# The control must see the same restricted data or the comparison is worthless.
#
# WEIGHTS. The CPL term's natural scale is ~0.03 against cross-entropy's ~2.3,
# so w_cpl=1 contributes about 1% of the loss -- close to no experiment at all.
# The sweep spans "barely present" to "dominant":
#     0    control
#     3    ~4% of total loss
#    10    ~12%
#    30    ~28%
#   100    ~57%
#
# This is a SCREEN, not the verdict. It ranks arms on val move_acc for a fixed
# short budget; the number that decides anything is top10_200k after a
# contrastive stage, which is a second run on whichever weights survive.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/cpl_sweep.log
SHARD=$D/shard
CORPUS=$D/cpl
MONTH=${MONTH:-2026-01}
WEIGHTS=${WEIGHTS:-"0 3 10 30 100"}
ARM_H=${ARM_H:-1.0}
DM=${DM:-256}
LAYERS=${LAYERS:-8}
HEADS=${HEADS:-8}
SLOTS=${SLOTS:-5}
MLPG=${MLPG:-160}
BATCH=${BATCH:-48}

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi
cd /root/code || exit 1

finish() {
    cp -f "$LOG" "$D/cpl_sweep_run.log" 2>/dev/null
    $PY runpod/s3io.py up "$D/cpl_sweep_run.log" final/cpl_sweep_run.log || true
    echo "$1"; echo NOVOL_ALL_DONE; exit "${2:-0}"
}

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

$PY runpod/s3io.py down "data/mt/$MONTH" "$SHARD" || finish NOVOL_UPLOAD_FAILED 1
$PY runpod/s3io.py down final/cpl_corpus.tar.gz "$D/cpl_corpus.tar.gz" || finish CPL_NO_CORPUS 1
mkdir -p "$CORPUS" && tar -xzf "$D/cpl_corpus.tar.gz" -C "$D" 2>/dev/null \
    || tar -xzf "$D/cpl_corpus.tar.gz" -C "$CORPUS" --strip-components=1
[ -s "$CORPUS/offsets.npy" ] || finish CPL_CORPUS_UNREADABLE 1

# The uploaded manifest records the pod-local path it was BUILT with, which is
# not what assert_shard compares against. Point it at the logical month.
$PY - "$CORPUS/manifest.json" "$MONTH" <<'MANIFEST_FIX'
import json, sys
p, month = sys.argv[1], sys.argv[2]
d = json.load(open(p)); d["shard"] = f"data/mt/{month}"
json.dump(d, open(p, "w"), indent=2)
print(f"  corpus: {d['plies']:,} plies, {d['move_evals']:,} move-evals, "
      f"shard -> {d['shard']}")
MANIFEST_FIX

for W in $WEIGHTS; do
    OUT=$D/cpl_w$W
    echo "=== $(date -u +%H:%M:%S) arm w_cpl=$W (cap ${ARM_H}h) ==="
    # Same --seed for every arm so the player split, the batch order and the
    # supervised-ply draws are identical. The only difference is w_cpl.
    if $PY train_multigame.py --shard "$SHARD" --out "$OUT" \
            --cpl-dir "$CORPUS" --cpl-only --w-cpl "$W" \
            --max-games "$SLOTS" --max-len-per-game "$MLPG" \
            --d-model "$DM" --layers "$LAYERS" --heads "$HEADS" \
            --batch "$BATCH" --lr 1.5e-4 --warmup 500 \
            --plies-per-game 8 --n-cand 32 --workers 8 \
            --steps 100000000 --eval-every 2000 --eval-batches 40 \
            --balance-elo --patience 1000000 --seed 0 \
            --amp --max-hours "$ARM_H"; then
        echo "ARM_${W}_DONE"
    else
        echo "ARM_${W}_FAILED"
    fi
    [ -s "$OUT/last.pt" ] && $PY runpod/s3io.py up "$OUT/last.pt" "final/cpl_w${W}_last.pt"
    [ -s "$OUT/history.json" ] && $PY runpod/s3io.py up "$OUT/history.json" "final/cpl_w${W}_history.json"
    cp -f "$LOG" "$D/cpl_sweep_run.log" 2>/dev/null
    $PY runpod/s3io.py up "$D/cpl_sweep_run.log" final/cpl_sweep_run.log || true
done

echo "=== $(date -u +%H:%M:%S) summary ==="
$PY - "$D" "$WEIGHTS" <<'SUMMARY'
import json, os, sys
d, weights = sys.argv[1], sys.argv[2].split()
print(f"{'w_cpl':>7} {'best move_acc':>14} {'final move_acc':>15} {'steps':>9}")
for w in weights:
    h = f"{d}/cpl_w{w}/history.json"
    if not os.path.isfile(h):
        print(f"{w:>7} {'(no history)':>14}"); continue
    hist = json.load(open(h)).get("history") or []
    accs = [e.get("move_acc", 0.0) for e in hist if "move_acc" in e]
    step = hist[-1].get("step", 0) if hist else 0
    if accs:
        print(f"{w:>7} {max(accs):>14.4f} {accs[-1]:>15.4f} {step:>9,}")
print("\nA SCREEN, not a verdict: this ranks arms on val move_acc at a fixed")
print("budget. Whether CPL helps IDENTIFICATION needs a contrastive stage and")
print("top10_200k, which is a second run on whichever weights survive here.")
SUMMARY

finish NOVOL_UPLOAD_OK 0
} >> "$LOG" 2>&1
