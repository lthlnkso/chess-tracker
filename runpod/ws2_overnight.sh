#!/usr/bin/env bash
# podB overnight chain. Sized to fill the night so the pod is never idle-billing.
#
#   1. wait out the accidental second ws2 run, keep its evals as run 2
#   2. ensemble the three objectives (shared trunk) + the published model
#   3. ensemble again with big_id, which differs in capacity, not objective
#   4. seed-1 replicate of ms vs supcon -- the variance estimate the headline
#      +52% currently lacks
#
# Every stage is `|| true`: a stage that fails must not cost the ones after it,
# because nobody is awake to restart them.
#
# On --extra-pids. ensemble_sweep needs each member's held-out players, and none
# of these checkpoints store them, so they are supplied. mt_test_pids.npy is the
# seed-0 / min_games 8 / test_frac 0.2 split, and the ws2 fine-tune logs print
# "56,497 held-out players", which is exactly its length -- so it is the right
# set for every member of stage 2. big_id was fine-tuned by a different script,
# so stage 3 falls back to common_test_pids.npy, the pre-computed intersection.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code
M=/workspace/data/mt/2026-01
K=/workspace/ckpt

until [ -f $K/ws2_supcon/eval.json ] && [ -f $K/ws2_ms/eval.json ] \
   && [ -f $K/ws2_circle/eval.json ]; do sleep 120; done
while pgrep -f "finetune_mt[.]py" > /dev/null; do sleep 60; done
sleep 30
echo OVERNIGHT_RUN2_DONE

for L in supcon ms circle; do cp $K/ws2_$L/eval.json $K/ws2_$L/eval_run2.json; done
$PY arm_report.py --title "contrastive loss - run 2 (replication)" --baseline supcon \
    --arm supcon=$K/ws2_supcon --arm ms=$K/ws2_ms --arm circle=$K/ws2_circle \
    --out $K/ws2_report_run2.json || true
echo OVERNIGHT_REPLICATION_DONE

$PY ensemble_sweep.py --shard $M --out $K/ens_objectives.json \
    --ckpts $K/ws2_ms/last.pt,$K/ws2_circle/last.pt,$K/ws2_supcon/last.pt,$K/mt_id/last.pt \
    --names ms,circle,supcon,published \
    --extra-pids $K/mt_test_pids.npy \
    --gallery-players 50000 --query-players 5000 --workers 22 || true
echo OVERNIGHT_ENS_OBJECTIVES_DONE

$PY ensemble_sweep.py --shard $M --out $K/ens_with_big.json \
    --ckpts $K/ws2_ms/last.pt,$K/ws2_circle/last.pt,$K/mt_id/last.pt,$K/big_id/last.pt \
    --names ms,circle,published,big92M \
    --extra-pids $K/common_test_pids.npy \
    --gallery-players 50000 --query-players 5000 --workers 22 || true
echo OVERNIGHT_ENS_BIG_DONE

# Seed 1 changes the split, the PK order and the balance sampler, so these two
# cannot join the ensembles above -- they exist only to say whether the ms/supcon
# gap survives a different draw.
for L in ms supcon; do
    $PY finetune_mt.py --shard $M --pretrained $K/mt_pre/last.pt \
        --out $K/ws2seed1_$L --loss $L --seed 1 \
        --max-hours 3 --p 32 --k 4 --workers 11 --min-games 8 \
        --eval-players 20000 --balance-elo > $K/ws2seed1_$L.log 2>&1 &
done
wait
$PY arm_report.py --title "contrastive loss - seed 1" --baseline supcon \
    --arm supcon=$K/ws2seed1_supcon --arm ms=$K/ws2seed1_ms \
    --out $K/ws2_report_seed1.json || true
echo OVERNIGHT_ALL_DONE
