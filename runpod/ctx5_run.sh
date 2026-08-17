#!/usr/bin/env bash
# Full-context run: 5 games of context, end to end, trained until it stops improving.
#
# Everything learned this session, applied at once:
#
#   5-game context      each sample is up to 5 games of one player, packed into
#                       one sequence with per-game slot and ply embeddings
#   no truncation       160 plies/game cuts 0.09% of games (the old 80 cut 25%
#                       of every player's evidence)
#   MS loss             +52% over SupCon on an identical trunk, twice replicated
#   fast loader         6.2-7.6x on the multi-game path, bit-for-bit identical
#   bf16 + compile      ~1.9x on the fine-tune
#   trained to a stop   --patience on both stages instead of a --max-hours wall;
#                       every previous run in this project was cut off mid-climb
#
# Context is used at fine-tune AND eval time, not just pre-training -- the
# earlier evidence that averaging beats joint encoding turned out to compare
# joint k-game queries against single-game centroids, and the corrected run was
# never finished. --gallery-mode matched is what makes that comparison fair.
#
# On --max-hours. It is NOT a cost cap here -- it is the cosine LR decay horizon.
# Drop it and the schedule falls back to --steps (1e8), leaving the LR essentially
# constant, and a model whose LR never comes down plateaus for the wrong reason.
# So it is set far beyond where either stage is expected to land: patience decides,
# and the horizon exists only to shape the decay.
#
# Patience is deliberately slack. At ~11 it/s, evals every 8000 steps and a
# patience of 10 means a stage must sit within 0.0002 of its best for ~2 hours
# before it stops. Every previous run in this project was cut off mid-climb;
# stopping early is the failure mode to avoid, not slowness.
#
# Budget: 20h + 20h + eval at $0.52/hr is ~$22 of the $26.6 balance. Fine-tuning
# gets a reserved half because a trunk with no identification head is unusable --
# spending the whole budget on pre-training would leave nothing to show for it.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code
M=/workspace/data/mt/2026-01
K=/workspace/ckpt
PRE_H=${PRE_H:-20}
FT_H=${FT_H:-20}

$PY train_multigame.py --shard $M --out $K/ctx5_pre \
    --max-games 5 --max-len-per-game 160 \
    --max-hours $PRE_H --steps 100000000 \
    --lr 1.5e-4 --warmup 1000 --batch 48 \
    --d-model 256 --layers 8 --heads 8 \
    --plies-per-game 8 --n-cand 32 --workers 24 \
    --eval-every 8000 --eval-batches 25 --balance-elo \
    --patience 10 --min-delta 0.0002 --amp --compile
[ -f $K/ctx5_pre/last.pt ] || { echo "CTX5_PRETRAIN_FAILED"; exit 1; }
echo CTX5_PRETRAIN_DONE

$PY finetune_ctx.py --shard $M --ckpt $K/ctx5_pre/last.pt --out $K/ctx5_ft \
    --loss ms --max-hours $FT_H --steps 100000000 \
    --p 24 --k 4 --lr 3e-5 --warmup 200 --workers 24 \
    --eval-every 4000 --eval-batches 25 --patience 10 \
    --amp --compile
[ -f $K/ctx5_ft/last.pt ] || { echo "CTX5_FINETUNE_FAILED"; exit 1; }
echo CTX5_FINETUNE_DONE

# The checkpoint stores its own test_pids, so the eval cannot accidentally score
# players the fine-tune trained on.
$PY identify_eval_ctx.py --ckpt $K/ctx5_ft/last.pt --shard $M \
    --out $K/ctx5_eval.json --ks 1,2,3,4,5 \
    --gallery-games 12 --eval-players 20000 --gallery-mode matched
[ -f $K/ctx5_eval.json ] && echo CTX5_ALL_DONE || echo "CTX5_EVAL_FAILED"
