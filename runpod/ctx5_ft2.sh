#!/usr/bin/env bash
# Re-run of the ctx5 fine-tune, long, with early stopping DISABLED.
#
# The first attempt stopped at 2.6h / 76,000 steps because --patience keyed on
# validation contrastive loss, which saturates hours before recall does. That
# made the context-vs-single-game comparison unfair to the context model. The
# recall-probe replacement is still being verified, so rather than launch
# unproven code overnight this run removes the stopping criterion entirely:
# --patience 1000000 means only the clock ends it.
#
# 6h at ~15 it/s is ~320k steps, 4.2x the aborted run. --max-hours doubles as the
# cosine LR horizon, so the schedule anneals properly inside it.
#
# Runs the build already on the pod -- the one that measured 14.83 it/s -- plus
# the vectorised ply_positions in train_multigame.py. Its reported it/s against
# that 14.83 baseline IS the A/B for that fix, measured for free.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code
M=/workspace/data/mt/2026-01
K=/workspace/ckpt
FT_H=${FT_H:-6}

$PY finetune_ctx.py --shard $M --ckpt $K/ctx5_pre/last.pt --out $K/ctx5_ft2 \
    --loss ms --max-hours $FT_H --steps 100000000 \
    --p 24 --k 4 --lr 3e-5 --warmup 200 --workers 24 \
    --eval-every 8000 --eval-batches 25 --patience 1000000 \
    --amp --compile
[ -f $K/ctx5_ft2/last.pt ] || { echo "FT2_FAILED"; exit 1; }
echo FT2_DONE

$PY identify_eval_ctx.py --ckpt $K/ctx5_ft2/last.pt --shard $M \
    --out $K/ctx5_eval2.json --ks 1,2,3,4,5 \
    --gallery-games 12 --eval-players 20000 --gallery-mode matched
[ -f $K/ctx5_eval2.json ] && echo FT2_ALL_DONE || echo "FT2_EVAL_FAILED"
