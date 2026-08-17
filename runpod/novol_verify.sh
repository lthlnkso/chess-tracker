#!/usr/bin/env bash
# Train the verifier: given four of a player's games and one more, same person?
#
# Why this and not more re-ranking: measured on the real 558,735-player gallery,
# cosine puts the right player at rank 1 for 51.6% of ten-game queries but inside
# the top 1000 for 94.4%. The shortlist nearly always contains the answer; the
# ordering is what fails. An XGBoost re-ranker over that shortlist bought +1.8
# points, inside its own noise, because it was re-ranking two 128-d summaries.
# This reads the games themselves.
#
# The budget is small ($3.65 total), so this is a hypothesis test, not a product
# model: does a verifier beat cosine at ordering candidates that already look
# alike? AUC on hard negatives is the number to watch -- accuracy on random
# negatives would be near 1.0 and mean nothing.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/verify.log
SHARD=$D/shard
TRUNK=$D/ctx5_ft2.pt
OUT=$D/verifier
FT_H=${FT_H:-8}

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) | $(nproc) vCPU ==="

$PY runpod/s3io.py down final/ctx5_ft2.pt "$TRUNK" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }
$PY runpod/s3io.py down data/mt/2026-01 "$SHARD" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }

( while :; do sleep 1800
    [ -s "$OUT/best.pt" ] && $PY runpod/s3io.py up "$OUT/best.pt" final/verifier_partial.pt >/dev/null 2>&1
  done ) &
KEEPER=$!

echo "=== $(date -u +%H:%M:%S) verifier fine-tune, cap ${FT_H}h ==="
# 80% hard negatives: at inference every candidate has already been ranked into
# the top 1000 by cosine, so easy negatives are not the distribution this model
# will ever see. The remaining 20% keeps it calibrated on ordinary players.
if $PY verify.py train --shard "$SHARD" --ckpt "$TRUNK" --out "$OUT" \
        --k 5 --players 40000 --min-games 6 \
        --neighbours 50 --hard-frac 0.8 \
        --items-per-epoch 20000 --eval-items 4000 \
        --batch 48 --workers 16 --lr 6e-5 --max-hours "$FT_H"; then
    echo VERIFY_TRAIN_DONE
else
    echo VERIFY_TRAIN_FAILED
fi
kill "$KEEPER" 2>/dev/null

for f in best.pt last.pt history.json; do
    [ -s "$OUT/$f" ] && $PY runpod/s3io.py up "$OUT/$f" "final/verifier_$f"
done

echo "=== $(date -u +%H:%M:%S) uploading ==="
cp -f "$LOG" "$D/verify_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/verify_run.log" final/verify_run.log || true
if [ -s "$OUT/best.pt" ] || [ -s "$OUT/last.pt" ]; then echo NOVOL_UPLOAD_OK; else echo NOVOL_UPLOAD_FAILED; fi
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
