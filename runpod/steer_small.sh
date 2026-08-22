#!/usr/bin/env bash
# Small-model validation of the skip-bad-step guard, before committing GPU hours
# to the 3x run again.
#
# Same code path and same two new flags as the big run, on the 7.9M trunk. It is
# ~5x faster per step, so an hour here covers more steps than all three failed
# big runs combined -- which is the point: the failure is a rare NaN gradient,
# and the only way to know the guard holds is to run past several of them.
#
# Success is not "no skips". Skips are EXPECTED. Success is skips appearing in
# the log while the loss stays flat and the weights stay finite.
set -u

W=/workspace
PY=${PY:-/workspace/venv/bin/python}
SHARD=$W/data/mt/2026-01
INIT=$W/final/ctx5_pre.pt
OUT=$W/ckpt/ctx5_steer_val
LOG=$W/ctx5_steer_val.log
HRS=${HRS:-1.5}

mkdir -p "$OUT"
cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="
for p in "$SHARD/meta.npy" "$INIT"; do
    [ -e "$p" ] || { echo "MISSING $p"; echo STEER_FAILED; exit 1; }
done
CPUS=$($PY cpuquota.py 2>/dev/null || echo 8)
echo "cpu quota: $CPUS"

$PY train_multigame.py --shard "$SHARD" --out "$OUT" --init "$INIT" \
    --max-games 5 --max-len-per-game 160 \
    --d-model 256 --layers 8 --heads 8 --d-embed 128 \
    --plies-per-game 8 --n-cand 32 --batch 48 \
    --lr 4e-5 --warmup 500 \
    --balance-gap --elo-steer \
    --workers "$CPUS" --eval-every 5000 --eval-batches 25 \
    --patience 1000000 --amp --compile --max-hours "$HRS" \
  && echo STEER_DONE || echo STEER_FAILED

echo "=== $(date -u +%H:%M:%S) end ==="
} >> "$LOG" 2>&1
