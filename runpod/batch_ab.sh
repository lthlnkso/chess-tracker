#!/usr/bin/env bash
# Does a bigger contrastive batch actually train a better model, or only a faster one?
#
# Batch 512 sees +36% samples/min but takes ~3x FEWER optimiser steps, and each
# anchor sees 508 in-batch negatives instead of 124. Those pull in opposite
# directions and throughput cannot settle it, so this compares on recall.
#
# Three arms, not two. Larger batches conventionally want a larger learning rate,
# so running 512 only at the incumbent LR would handicap it and the result would
# say "batch size does not help" when it meant "1e-4 is wrong for batch 512":
#
#   A  batch 128, lr 1e-4   the incumbent
#   B  batch 512, lr 1e-4   batch size alone
#   C  batch 512, lr 2e-4   batch size with the usual sqrt scaling
#
# Sequential, never concurrent: the step is GPU-bound, so two arms sharing the
# card would halve each and confound the comparison with contention.
#
# Compared on the probe curve, not on the loss. Loss values are not comparable
# across batch sizes at all -- the number of negatives in the denominator changes.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code
M=/workspace/data/mt/2026-01
PRE=/workspace/ckpt/ws1_pre/last.pt
HOURS=${HOURS:-0.6}

if [ ! -f $PRE ]; then echo "AB_FAILED: no trunk at $PRE"; exit 1; fi

run () {   # name, p, lr
    $PY finetune_mt.py --shard $M --pretrained $PRE \
        --out /workspace/ckpt/ab_$1 --loss ms \
        --max-hours $HOURS --p $2 --k 4 --lr $3 --warmup 200 \
        --workers 24 --min-games 8 --balance-elo \
        --amp --compile --static-len 160 \
        --probe-every-hours 0.2 --probe-players 2000 \
        --eval-players 2000 --collapse-after 100000 \
        > /workspace/ckpt/ab_$1.log 2>&1
    echo "AB_ARM_DONE $1"
}

run b128_lr1  32  1e-4
run b512_lr1 128  1e-4
run b512_lr2 128  2e-4

for A in b128_lr1 b512_lr1 b512_lr2; do
    if [ -f /workspace/ckpt/ab_$A/eval.json ]; then echo "AB_OK $A"; else echo "AB_FAILED $A"; tail -15 /workspace/ckpt/ab_$A.log; fi
done
echo AB_ALL_DONE
