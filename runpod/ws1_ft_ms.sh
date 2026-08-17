#!/usr/bin/env bash
# Fine-tune workstream 1's long trunk with Multi-Similarity instead of SupCon.
#
# ws1_long.sh was written before the loss sweep returned; its SupCon stage is
# superseded (MS scored +52% on an identical trunk and budget). Rather than edit
# ws1_long.sh while bash is mid-read of it -- bash reads scripts incrementally,
# and editing a running one corrupts execution -- the wrapper is killed and the
# pre-training python, which is a separate process, is left to finish orphaned.
#
# Waits on history.json, not last.pt: last.pt is rewritten every --eval-every
# steps, so its presence says nothing about the run being over. history.json is
# written once, at the end.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code
M=/workspace/data/mt/2026-01

until [ -f /workspace/ckpt/ws1_pre/history.json ]; do sleep 120; done
while pgrep -f "train_multitask[.]py" > /dev/null; do sleep 60; done
sleep 30
echo WS1_PRETRAIN_DONE

$PY -c "import json;h=json.load(open('/workspace/ckpt/ws1_pre/history.json'));\
print('pretrain:',h['steps'],'steps',h['minutes'],'min | final val',h['history'][-1])"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY finetune_mt.py --shard $M \
    --pretrained /workspace/ckpt/ws1_pre/last.pt --out /workspace/ckpt/ws1_id_ms \
    --max-hours 10 --p 32 --k 4 --workers 24 --min-games 8 \
    --eval-players 20000 --balance-elo --loss ms \
    --probe-every-hours 1 --probe-players 2000

[ -f /workspace/ckpt/ws1_id_ms/eval.json ] && echo WS1_ALL_DONE || echo WS1_FINETUNE_FAILED
