#!/usr/bin/env bash
# Re-run workstream 3's identification fine-tunes at equal STEPS.
#
# The first pass budgeted the fine-tune in wall clock (--max-hours 0.5), and the
# six arms finished their pre-training at different times, so they hit the
# fine-tune stage under different amounts of CPU contention: 4,832 steps for the
# C=16 arm against 7,582 for C=64. A 57% spread in training length is larger than
# any effect being measured, which makes that pass useless for ranking -- C=64's
# apparent +11.7% is indistinguishable from "trained half again as long".
#
# Fixing --steps and setting --max-hours 0 hands the LR schedule to the step
# count, so contention now only moves wall clock, which nothing is measured in.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code
M=/workspace/data/cand/2026-01
STEPS=${STEPS:-5000}

ft () {
    $PY finetune_mt.py --shard $M --pretrained /workspace/ckpt/c_$1/last.pt \
        --out /workspace/ckpt/c_$1_ideq --loss supcon \
        --max-hours 0 --steps $STEPS --p 32 --k 4 --workers 5 --min-games 8 \
        --eval-players 5000 --balance-elo --collapse-after 100000 \
        > /workspace/ckpt/c_$1.fteq.log 2>&1
}

ft c4 & ft c8 & ft c16 &
wait
ft c32 & ft c64 & ft curr &
wait

for A in c4 c8 c16 c32 c64 curr; do
    [ -f /workspace/ckpt/c_${A}_ideq/eval.json ] && echo "WS3EQ_OK $A" || echo "WS3EQ_FAILED $A"
done
echo WS3EQ_ALL_DONE
