#!/usr/bin/env bash
# Workstream 2: which metric-learning objective builds the better player vector?
#
# Four arms, one per family (see contrastive.py for why these four). Every arm
# starts from the SAME pre-trained trunk -- ckpt/mt_pre.pt, the one that produced
# the published 13.9% result -- and gets the same month, the same Elo balancing,
# the same P*K game-sides per step and the same wall clock. The objective is the
# only thing that differs, which is what makes the ranking mean anything.
#
# Deliberately NOT waiting for workstream 1's longer trunk: holding the trunk
# fixed at the known checkpoint makes these results directly comparable to the
# existing ckpt/mt_id.pt baseline, and the winning loss can be re-run on a better
# trunk afterwards for a fraction of the cost.
#
# All four run concurrently on one GPU. The model is 7.9M parameters and peaks
# near 3 GB, so four fit in 24 GB with room to spare; the contended resource is
# CPU workers for python-chess move generation, hence WORKERS below is the pod's
# core count divided four ways.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code
M=/workspace/data/mt/2026-01
PRE=/workspace/ckpt/mt_pre/last.pt
HOURS=${HOURS:-3}
WORKERS=${WORKERS:-7}

if [ ! -f $PRE ]; then echo "WS2_FAILED: no pretrained trunk at $PRE"; exit 1; fi

for L in supcon ms circle proxyanchor; do
    $PY finetune_mt.py --shard $M --pretrained $PRE \
        --out /workspace/ckpt/ws2_$L --loss $L \
        --max-hours $HOURS --p 32 --k 4 --workers $WORKERS --min-games 8 \
        --eval-players 20000 --balance-elo \
        --probe-every-hours 1 --probe-players 2000 \
        > /workspace/ckpt/ws2_$L.log 2>&1 &
done
wait

for L in supcon ms circle proxyanchor; do
    if [ -f /workspace/ckpt/ws2_$L/eval.json ]; then
        echo "WS2_ARM_OK $L"
    else
        echo "WS2_ARM_FAILED $L"; tail -20 /workspace/ckpt/ws2_$L.log
    fi
done
echo WS2_ALL_DONE
