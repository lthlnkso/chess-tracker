#!/usr/bin/env bash
# 3x trunk, rating self-steering, gap-balanced sampling.
#
# CONTINUATION, not a fresh pre-train. A from-scratch 24M/10-slot run is ~50
# GPU-h and the budget is ~18, which would have bought ~150k steps against
# ctx10_pre's 504k -- a worse model than the one we already have. Initialising
# from ctx10_pre instead spends the whole budget on the thing being tested.
# elo_steer and elo_cond are zero-init, so step 0 here is bit-identical to
# ctx10_pre and every gradient after it goes into the new pathway.
#
# NOT --elo-cond. train_multigame.py feeds the TRUE rating into the trunk when
# that flag is on, so elo_head can predict the rating by reading its own input --
# which is why ctx5_pre_elo's val_move_acc was never comparable to ctx5_pre's.
# Here it would be worse than incomparable: elo_steer reads elo_head, so the
# "self-estimate" steering the move would just be the supplied label echoed back,
# and at --elo-drop 0.1 that is 90% of training. The point of this run is that
# the model INFERS the rating, so nothing may hand it one.
#
# Everything lives on the network volume, so a pod that dies does not take the
# checkpoints with it.
set -u

W=/workspace
PY=${PY:-/root/venv/bin/python}
SHARD=$W/data/mt/2026-01
INIT=$W/final/ctx10_pre.pt
OUT=$W/ckpt/ctx10_steer
LOG=$W/ctx10_steer.log
HRS=${HRS:-17}

mkdir -p "$OUT"
cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="
for p in "$SHARD/meta.npy" "$INIT"; do
    [ -e "$p" ] || { echo "MISSING $p"; echo STEER_FAILED; exit 1; }
done
# nproc reports the HOST's cores inside a container, not the allocation
CPUS=$($PY cpuquota.py 2>/dev/null || echo 8)
echo "cpu quota: $CPUS"

$PY train_multigame.py --shard "$SHARD" --out "$OUT" --init "$INIT" \
    --max-games 10 --max-len-per-game 160 \
    --d-model 384 --layers 12 --heads 8 --d-embed 128 \
    --plies-per-game 8 --n-cand 32 --batch 32 \
    --lr 6e-5 --warmup 800 \
    --balance-gap --elo-steer \
    --workers "$CPUS" --eval-every 4000 --eval-batches 25 \
    --patience 1000000 --amp --compile --max-hours "$HRS" \
  && echo STEER_DONE || echo STEER_FAILED

echo "=== $(date -u +%H:%M:%S) end ==="
ls -la "$OUT"
} >> "$LOG" 2>&1
