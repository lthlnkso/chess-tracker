#!/usr/bin/env bash
# Pre-training only; ingest and combine are already done.
#
# Second attempt. The first run peaked at val acc 0.517 (step 30k) and then
# degraded on BOTH train and val -- and since only 2.5% of an epoch had been
# consumed, every batch was still fresh data, so that was optimisation
# instability, not overfitting. Cause: stretching the cosine over 5 hours parks
# the LR near its 3e-4 peak for tens of thousands of steps.
#
# Fix: halve the peak LR, lengthen warmup, and shorten the budget so the cosine
# actually anneals. 2.5h at ~11.9 it/s is ~5.5% of an epoch (a full epoch is
# 45.6 hours, so duration is not the binding constraint -- stability is).
set -x
PY=/workspace/venv/bin/python
cd /workspace/code

$PY train_successor.py --combined /workspace/data/combined \
    --out /workspace/ckpt/prod --max-hours 2.5 --steps 100000000 \
    --lr 1.5e-4 --warmup 1000 \
    --workers 30 --batch 128 --eval-every 5000 --eval-batches 30 --no-rights
echo TRAIN_DONE
