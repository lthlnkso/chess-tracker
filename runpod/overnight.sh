#!/usr/bin/env bash
# Overnight: identification fine-tune + full evaluation.
# Pre-training is already done (best.pt, val acc 0.554).
#
# SupCon, not triplet. Triplet's minimum is the constant-output solution, and a
# network reaches it trivially -- it collapsed within 40 steps, twice. SupCon
# makes that same solution its *maximum*. A collapse detector aborts at step 2000
# if the pos/neg cosine gap is still flat, so a bad run fails in minutes.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code

$PY finetune_id.py --combined /workspace/data/combined \
    --pretrained /workspace/ckpt/prod/best.pt --out /workspace/ckpt/id \
    --loss supcon --max-hours 4 --p 32 --k 4 --workers 30 \
    --log-every 200 --ckpt-every 2000 --collapse-after 2000
echo FINETUNE_DONE

$PY identify_eval.py --ckpt /workspace/ckpt/id/last.pt \
    --combined /workspace/data/combined --out /workspace/ckpt/id/eval.json \
    --max-test-players 20000 --min-games 10 --workers 30
echo IDENTIFY_DONE
