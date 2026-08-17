#!/usr/bin/env bash
# Workstream 1: is the best config actually saturated?
#
# Same 7.9M arm-C config as the run that produced ckpt/mt_pre.pt + ckpt/mt_id.pt,
# same month, same Elo balancing -- only the clock changes: 8h pre-train instead
# of 3h, 7h fine-tune instead of 4h. Both stages of the original stopped on the
# --max-hours wall with their metrics still moving (pre-train val move_acc
# 0.455 -> 0.549 and still climbing; SupCon pos/neg gap 0.194 -> 0.307 and still
# climbing), so neither number was a convergence result.
#
# --probe-every-hours is the point of the exercise. The loss curve cannot tell
# you whether *recall* has flattened, and recall is the deliverable. A 2000-
# player gallery probe every hour costs a couple of minutes and turns "we ran
# longer" into a saturation curve.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code
M=/workspace/data/mt/2026-01

$PY train_multitask.py --shard $M --out /workspace/ckpt/ws1_pre \
    --max-hours 8 --steps 100000000 \
    --lr 1.5e-4 --warmup 1000 --batch 128 --d-model 256 --layers 8 --heads 8 \
    --plies-per-game 12 --n-cand 16 --workers 24 \
    --eval-every 4000 --eval-batches 30 --balance-elo
if [ ! -f /workspace/ckpt/ws1_pre/last.pt ]; then
    echo "WS1_PRETRAIN_FAILED"; exit 1
fi
echo WS1_PRETRAIN_DONE

$PY finetune_mt.py --shard $M \
    --pretrained /workspace/ckpt/ws1_pre/last.pt --out /workspace/ckpt/ws1_id \
    --max-hours 7 --p 32 --k 4 --workers 24 --min-games 8 \
    --eval-players 20000 --balance-elo --loss supcon \
    --probe-every-hours 1 --probe-players 2000
if [ -f /workspace/ckpt/ws1_id/eval.json ]; then
    echo WS1_ALL_DONE
else
    echo "WS1_FINETUNE_FAILED"; exit 1
fi
