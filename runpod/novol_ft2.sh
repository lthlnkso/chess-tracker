#!/usr/bin/env bash
# Continue the colour-homogeneous fine-tune to saturation, then measure the one
# number that matters: top-10 recall on the full gallery.
#
# The previous run stopped at 347k steps because a 6-hour wall clock ran out on
# hardware ~2.2x slower than ft2's, not because it converged -- its val was still
# falling. Comparing that checkpoint to a 775k-step model said more about the
# budget than about colour. This resumes from those weights and runs until the
# validation metric genuinely stops improving.
#
# `--patience 8` at `--eval-every 8000` means 64k steps of no gain before
# stopping. That is a saturation test, not the aggressive early stop that cut a
# previous run short; it exists so a plateau does not keep billing.
#
# Both retrieval arms are measured at the end, because for the PRODUCT the
# question is simply which scoring gives the best top-10 -- not which one wins a
# controlled comparison.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/ft2.log
SHARD=$D/shard
START_CKPT=$D/ctx5_ftc.pt
OUT=$D/ctx5_ftc2
FT_H=${FT_H:-10}

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="

echo "=== $(date -u +%H:%M:%S) fetching ==="
$PY runpod/s3io.py down final/ctx5_ftc.pt "$START_CKPT" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }
$PY runpod/s3io.py down data/mt/2026-01 "$SHARD" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }

# A ten-hour run must not be all-or-nothing. Push the rolling checkpoint every
# 30 minutes so a crash, a preemption or an exhausted balance costs at most half
# an hour instead of the whole run.
(
  while :; do
    sleep 1800
    [ -s "$OUT/last.pt" ] && $PY runpod/s3io.py up "$OUT/last.pt" final/ctx5_ftc2_partial.pt >/dev/null 2>&1
  done
) &
KEEPER=$!

echo "=== $(date -u +%H:%M:%S) resume from 347k, train to saturation (cap ${FT_H}h) ==="
if $PY finetune_ctx.py --shard "$SHARD" --ckpt "$START_CKPT" --out "$OUT" \
        --same-colour \
        --loss ms --max-hours "$FT_H" --steps 100000000 \
        --p 24 --k 4 --lr 3e-5 --warmup 200 --workers 24 \
        --eval-every 8000 --eval-batches 25 --patience 8 \
        --amp --compile; then
    echo FT_TRAIN_DONE
else
    echo FT_TRAIN_FAILED
fi
kill "$KEEPER" 2>/dev/null

[ -s "$OUT/last.pt" ] && $PY runpod/s3io.py up "$OUT/last.pt" final/ctx5_ftc2.pt
[ -s "$OUT/history.json" ] && $PY runpod/s3io.py up "$OUT/history.json" final/ctx5_ftc2_history.json

echo "=== $(date -u +%H:%M:%S) eval: top-10 on the full gallery, both arms ==="
if [ -s "$OUT/last.pt" ] && $PY gallery_ctx.py --ckpt "$OUT/last.pt" --shard "$SHARD" \
        --out "$D/ctx5_ftc2_colour.json" \
        --colour-split --ks 1,3,5 --gallery-games 64 \
        --gallery-players 200000 --query-players 5000 \
        --sizes 1000,10000,50000,100000,200000 --workers 22; then
    echo FT_EVAL_DONE
else
    echo FT_EVAL_FAILED
fi

echo "=== $(date -u +%H:%M:%S) uploading ==="
ok=1
[ -s "$D/ctx5_ftc2_colour.json" ] && { $PY runpod/s3io.py up "$D/ctx5_ftc2_colour.json" final/ctx5_ftc2_colour.json || ok=0; }
cp -f "$LOG" "$D/ft2_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/ft2_run.log" final/ft2_run.log || true

if [ "$ok" -eq 1 ] && [ -s "$OUT/last.pt" ]; then echo NOVOL_UPLOAD_OK; else echo NOVOL_UPLOAD_FAILED; fi
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
