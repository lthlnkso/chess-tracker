#!/usr/bin/env bash
# Finish the identifier fine-tune that was cut short to fund the Elo run.
#
# History: ctx5_ft2 trained 775k steps and ships. ctx5_ft3 resumed it and moved
# val 0.3888 -> 0.3829 in 85k more steps, then stopped at 2.0 GPU-hours -- killed
# by budget, not by its own patience-8 saturation test. It was still improving.
# This resumes from those weights and lets the patience stop decide when to end.
#
# It also closes a hole that has been open since ft2. `finetune_ctx.py` writes
# best.pt, but novol_ft3.sh only ever uploaded last.pt, so every best-val
# checkpoint we have trained died with its pod. ft2's best val was 0.3578 against
# the shipped 0.3888 and NOBODY EVER MEASURED IT on the product metric. This run
# uploads both and evaluates both, so "we ship last.pt" stops being an accident
# and becomes a measured choice.
#
# No --colour-split in the eval: that question is settled and closed (it lost
# twice), and running it here would only spend GPU time re-losing it.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/ft4.log
SHARD=$D/shard
START_CKPT=$D/ctx5_ft3_start.pt
OUT=$D/ctx5_ft4
FT_H=${FT_H:-8}
GAL_PLAYERS=${GAL_PLAYERS:-200000}

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) | $(nproc) vCPU ==="

$PY runpod/s3io.py down final/ctx5_ft3.pt "$START_CKPT" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }
$PY runpod/s3io.py down data/mt/2026-01 "$SHARD" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }

# Push BOTH rolling checkpoints every 30 min. best.pt is the one that has been
# getting lost, so it goes first.
( while :; do sleep 1800
    [ -s "$OUT/best.pt" ] && $PY runpod/s3io.py up "$OUT/best.pt" final/ctx5_ft4_best_partial.pt >/dev/null 2>&1
    [ -s "$OUT/last.pt" ] && $PY runpod/s3io.py up "$OUT/last.pt" final/ctx5_ft4_partial.pt >/dev/null 2>&1
  done ) &
KEEPER=$!

echo "=== $(date -u +%H:%M:%S) resume ft3 (860k steps in), train to saturation, cap ${FT_H}h ==="
# lr unchanged at 3e-5: it is the proven setting for this stage, and changing it
# at the same time as extending the run would leave us unable to say which one
# moved the number.
if $PY finetune_ctx.py --shard "$SHARD" --ckpt "$START_CKPT" --out "$OUT" \
        --loss ms --max-hours "$FT_H" --steps 100000000 \
        --p 24 --k 4 --lr 3e-5 --warmup 200 --workers 24 \
        --eval-every 8000 --eval-batches 25 --patience 8 \
        --amp --compile; then
    echo FT4_TRAIN_DONE
else
    echo FT4_TRAIN_FAILED
fi
kill "$KEEPER" 2>/dev/null

for f in best.pt last.pt history.json; do
    [ -s "$OUT/$f" ] && $PY runpod/s3io.py up "$OUT/$f" "final/ctx5_ft4_$f"
done

# Evaluate BOTH on the product metric, same protocol as the registry's
# top10_200k_5games, so the numbers land on a scale we already have.
for W in best last; do
    [ -s "$OUT/$W.pt" ] || continue
    echo "=== $(date -u +%H:%M:%S) eval $W.pt | top-10 @ ${GAL_PLAYERS} ==="
    if $PY gallery_ctx.py --ckpt "$OUT/$W.pt" --shard "$SHARD" \
            --out "$D/ft4_${W}_eval.json" --ks 5 \
            --gallery-games 64 --min-gallery-games 8 \
            --gallery-players "$GAL_PLAYERS" --query-players 5000 \
            --sizes 10000,50000,200000 --batch 192 --workers 24; then
        echo "FT4_EVAL_${W}_DONE"
        $PY runpod/s3io.py up "$D/ft4_${W}_eval.json" "final/ft4_${W}_eval.json" || true
    else
        echo "FT4_EVAL_${W}_FAILED"
    fi
done

echo "=== $(date -u +%H:%M:%S) uploading ==="
cp -f "$LOG" "$D/ft4_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/ft4_run.log" final/ft4_run.log || true
if [ -s "$OUT/last.pt" ]; then echo NOVOL_UPLOAD_OK; else echo NOVOL_UPLOAD_FAILED; fi
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
