#!/usr/bin/env bash
# The two things left undone, in priority order because the balance is ~$3.
#
#   1. pooled recall for ws1_id_ms -- how many games it takes to identify someone.
#      finetune_mt.py's built-in eval only ever scores ONE game per query, so this
#      curve does not exist for the model yet.
#   2. the ensemble, now including that model.
#
# Defaults on identify_eval_mt.py (seed 0, min_games 8, test_frac 0.2,
# eval_players 20000) reproduce exactly the split the fine-tune held out, which
# is also the split behind the published eval_pooled.json -- so the new curve is
# directly comparable to the old 88.3%-at-ten-games number rather than merely
# similar.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code
M=/workspace/data/mt/2026-01
K=/workspace/ckpt

$PY identify_eval_mt.py --ckpt $K/ws1_id_ms/last.pt --shard $M \
    --out $K/ws1_pooled.json --pools 1,2,3,5,10 --workers 22
echo FINAL_POOLED_DONE

# Capacity diversity was worth 4x objective diversity last night, so the new
# model is paired with big92M first.
$PY ensemble_sweep.py --shard $M --out $K/ens_final_big.json \
    --ckpts $K/ws1_id_ms/last.pt,$K/big_id/last.pt,$K/mt_id/last.pt \
    --names ws1_ms,big92M,published \
    --extra-pids $K/common_test_pids.npy \
    --gallery-players 50000 --query-players 5000 --workers 22 || true
echo FINAL_ENS_BIG_DONE

$PY ensemble_sweep.py --shard $M --out $K/ens_final_same.json \
    --ckpts $K/ws1_id_ms/last.pt,$K/ws2_ms/last.pt,$K/mt_id/last.pt \
    --names ws1_ms,short_ms,published \
    --extra-pids $K/mt_test_pids.npy \
    --gallery-players 50000 --query-players 5000 --workers 22 || true
echo FINAL_ALL_DONE
