#!/usr/bin/env bash
# Phase 4: player identification. Waits for pre-training to finish, smoke-tests
# the fine-tune path, then runs it for real and evaluates.
#
# Waits on the TRAIN_DONE marker rather than on process presence -- pgrep also
# matches the shells that merely *mention* the script name, so it fires early.
set -ex
PY=/workspace/venv/bin/python
cd /workspace/code

while ! grep -q 'TRAIN_DONE' /workspace/after_ingest.log 2>/dev/null; do
    sleep 120
done
# Prefer best-by-val over the final step: a run whose tail degraded should not
# hand a worse encoder to the fine-tune.
if [ -f /workspace/ckpt/prod/best.pt ]; then
    PRE=/workspace/ckpt/prod/best.pt
elif [ -f /workspace/ckpt/prod/last.pt ]; then
    PRE=/workspace/ckpt/prod/last.pt
else
    echo NO_PRETRAIN_CKPT; exit 1
fi
echo "PRETRAIN_READY $PRE"

# Cheap gate: fail here rather than three hours in.
$PY finetune_id.py --combined /workspace/data/combined \
    --pretrained $PRE --out /workspace/ckpt/id_smoke \
    --max-hours 0.05 --p 16 --k 4 --workers 24 --log-every 20 --ckpt-every 100
echo SMOKE_OK

$PY finetune_id.py --combined /workspace/data/combined \
    --pretrained $PRE --out /workspace/ckpt/id \
    --max-hours 3 --p 32 --k 4 --workers 30 --log-every 100 --ckpt-every 2000
echo FINETUNE_DONE

$PY identify_eval.py --ckpt /workspace/ckpt/id/last.pt \
    --combined /workspace/data/combined --out /workspace/ckpt/id/eval.json \
    --max-test-players 20000 --workers 30
echo IDENTIFY_DONE
