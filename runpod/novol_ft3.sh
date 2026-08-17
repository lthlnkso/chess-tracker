#!/usr/bin/env bash
# SUPERSEDED by ft4.sh -- kept for provenance. See the correction below.
#
# CORRECTION: the header this file used to carry described continuing the
# COLOUR-homogeneous model ("previous run stopped at 347k steps"), which is
# ctx5_ftc's story, copy-pasted from novol_ft2.sh. This script does not do that.
# It downloads final/ctx5_ft2.pt and continues the MIXED-colour identifier, which
# is what the registry's ctx5_ft3 row records. The --colour-split eval below came
# from the same paste and re-measured a question already settled twice.
#
# What it actually did: ctx5_ft2 trained with --patience 1000000 (early stopping
# off) and stopped when a 6-hour wall clock expired, not because it converged.
# This resumed those weights with --patience 8 to find where the curve ends. It
# moved val 0.3888 -> 0.3829 in 85k steps, then was itself killed at 2.0 GPU-hours
# to fund the Elo run -- so it never reached saturation either.
#
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/ft3.log
SHARD=$D/shard
START_CKPT=$D/ctx5_ft2_start.pt
OUT=$D/ctx5_ft3
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
$PY runpod/s3io.py down final/ctx5_ft2.pt "$START_CKPT" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }
$PY runpod/s3io.py down data/mt/2026-01 "$SHARD" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }

# A ten-hour run must not be all-or-nothing. Push the rolling checkpoint every
# 30 minutes so a crash, a preemption or an exhausted balance costs at most half
# an hour instead of the whole run.
(
  while :; do
    sleep 1800
    [ -s "$OUT/last.pt" ] && $PY runpod/s3io.py up "$OUT/last.pt" final/ctx5_ft3_partial.pt >/dev/null 2>&1
  done
) &
KEEPER=$!

echo "=== $(date -u +%H:%M:%S) resume ft2 from 775k, train to saturation (cap ${FT_H}h) ==="
if $PY finetune_ctx.py --shard "$SHARD" --ckpt "$START_CKPT" --out "$OUT" \
        --loss ms --max-hours "$FT_H" --steps 100000000 \
        --p 24 --k 4 --lr 3e-5 --warmup 200 --workers 24 \
        --eval-every 8000 --eval-batches 25 --patience 8 \
        --amp --compile; then
    echo FT_TRAIN_DONE
else
    echo FT_TRAIN_FAILED
fi
kill "$KEEPER" 2>/dev/null

[ -s "$OUT/last.pt" ] && $PY runpod/s3io.py up "$OUT/last.pt" final/ctx5_ft3.pt
[ -s "$OUT/history.json" ] && $PY runpod/s3io.py up "$OUT/history.json" final/ctx5_ft3_history.json

echo "=== $(date -u +%H:%M:%S) eval: top-10 on the full gallery, both arms ==="
if [ -s "$OUT/last.pt" ] && $PY gallery_ctx.py --ckpt "$OUT/last.pt" --shard "$SHARD" \
        --out "$D/ctx5_ft3_colour.json" \
        --colour-split --ks 1,3,5 --gallery-games 64 \
        --gallery-players 200000 --query-players 5000 \
        --sizes 1000,10000,50000,100000,200000 --workers 22; then
    echo FT_EVAL_DONE
else
    echo FT_EVAL_FAILED
fi

echo "=== $(date -u +%H:%M:%S) uploading ==="
ok=1
[ -s "$D/ctx5_ft3_colour.json" ] && { $PY runpod/s3io.py up "$D/ctx5_ft3_colour.json" final/ctx5_ft3_eval.json || ok=0; }
cp -f "$LOG" "$D/ft3_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/ft3_run.log" final/ft3_run.log || true

if [ "$ok" -eq 1 ] && [ -s "$OUT/last.pt" ]; then echo NOVOL_UPLOAD_OK; else echo NOVOL_UPLOAD_FAILED; fi
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
