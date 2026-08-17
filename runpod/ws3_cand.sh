#!/usr/bin/env bash
# Workstream 3: how many candidate successors should pre-training score against?
#
# Six arms: C = 4, 8, 16 (the incumbent), 32, 64, and a curriculum that starts at
# 2 and doubles each time the model gets good enough at the current difficulty.
#
# Two measurement decisions that matter more than the arms themselves:
#
#  1. Equal STEPS, not equal wall clock. C=64 generates 16x the candidate boards
#     per sample that C=4 does, on the same CPU workers, so equal wall clock
#     would confound "harder task" with "fewer samples". it/s is recorded per arm
#     so the equal-budget view can be recovered afterwards.
#  2. Validation is pinned to C=16 for every arm (--eval-n-cand). Raw move_acc is
#     not comparable across arms because chance is 1/C -- 0.50 at C=2 is nothing
#     and 0.50 at C=64 is enormous. A fixed validation task puts them all on one
#     axis.
#
# The number that actually decides it is neither: it is identification recall
# after an identical short SupCon fine-tune, which is what the trunk is for.
#
# Small scale on purpose, on a community pod with a self-ingested slice of the
# same month -- the comparison is internal to the six arms, so it does not need
# the full 24M-game shard, and community is 2.3x cheaper than the volume DC.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code
M=/workspace/data/cand/2026-01
STEPS=${STEPS:-12000}
FT_HOURS=${FT_HOURS:-0.5}
WORKERS=${WORKERS:-9}

if [ ! -f $M/manifest.json ]; then
    $PY ingest.py \
        --url https://database.lichess.org/standard/lichess_db_standard_rated_2026-01.pgn.zst \
        --out $M --workers 28 --time-controls '60+0' --limit 4000000 \
        2>&1 | tr '\r' '\n' | tail -2
fi
echo INGEST_DONE
$PY -c "import json;m=json.load(open('$M/manifest.json'));print(m['games'],'games',m['players'],'players')"

# Three at a time: nine workers each keeps the GPU fed without the arms starving
# one another so badly that the it/s numbers stop meaning anything.
run_arm () {   # $1 = arm name, $2.. = extra train flags
    local name=$1; shift
    $PY train_multitask.py --shard $M --out /workspace/ckpt/c_$name \
        --steps $STEPS --lr 3e-4 --warmup 500 --batch 64 \
        --d-model 256 --layers 8 --heads 8 --plies-per-game 12 \
        --workers $WORKERS --eval-every 2000 --eval-batches 20 \
        --eval-n-cand 16 --balance-elo "$@" \
        > /workspace/ckpt/c_$name.pre.log 2>&1
    $PY finetune_mt.py --shard $M --pretrained /workspace/ckpt/c_$name/last.pt \
        --out /workspace/ckpt/c_${name}_id --loss supcon \
        --max-hours $FT_HOURS --p 32 --k 4 --workers $WORKERS --min-games 8 \
        --eval-players 5000 --balance-elo --collapse-after 100000 \
        > /workspace/ckpt/c_$name.ft.log 2>&1
}

run_arm c4  --n-cand 4  &
run_arm c8  --n-cand 8  &
run_arm c16 --n-cand 16 &
wait
run_arm c32 --n-cand 32 &
run_arm c64 --n-cand 64 &
run_arm curr --n-cand 64 --cand-curriculum --cand-start 2 --cand-max 64 \
             --cand-min-steps 1000 --cand-promote 0.45 --cand-patience 3000 &
wait

for A in c4 c8 c16 c32 c64 curr; do
    if [ -f /workspace/ckpt/c_${A}_id/eval.json ]; then
        echo "WS3_ARM_OK $A"
    else
        echo "WS3_ARM_FAILED $A"; tail -20 /workspace/ckpt/c_$A.pre.log /workspace/ckpt/c_$A.ft.log
    fi
done
echo WS3_ALL_DONE
