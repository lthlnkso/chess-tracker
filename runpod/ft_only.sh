#!/usr/bin/env bash
# Stage B + C only, from a trunk that is still pre-training on another pod.
#
# Why split it out: measured on ctx10's own validation curve, continued stage A
# is now trading the WRONG WAY for this product. Over the last ~230k steps
# move_acc trends +0.0053/100k (real -- 9.3 noise-sd end to end) while elo_mae
# trends +1.25/100k WORSE and time_acc is flat. w_move is 1.0 against 0.3 for the
# two auxiliary heads, so as the move objective keeps improving it dominates and
# the player-level heads drift. Identification is a player-level task, so more
# pre-training is mildly anti-correlated with the thing we ship.
#
# The trunk therefore comes from the rolling partial rather than waiting for
# stage A to converge. If stage A does end up somewhere better, rerunning this
# script picks up whatever ctx10_pre.pt holds at that point.
#
# P=16/K=4 is expected (the ladder settles there); that was chosen deliberately
# over P=24/K=3 -- see docs/BRANCHES.md #2 for the road not taken.
#
#   d_model 384 / 12 layers / 8 heads = 24.0M params (3.0x the shipped 7.9M)
#   10 game slots x 160 plies = 1600 positions (2x the shipped context)
#   n_cand 32 and d_embed 128 UNCHANGED, deliberately.
#
# Why those two are unchanged. Cutting n_cand to 16 buys 5.7% of forward FLOPs
# at this size and makes every move_acc number incomparable to our history --
# the exact trap mt_run.log set for us. And d_embed cannot raise accuracy: fewer
# dimensions cannot carry more information, so 32-d is a deployment-cost lever,
# not a quality one.
#
# Why 3x and not 10x: 24.0M at batch 24 needs ~8 GB against the 10x model's
# measured 27.9 GB, so this fits a $0.27 A5000 and the 10x never could. A probe
# also showed the 10x model sitting 0.034 behind the 7.9M one at equal data with
# the gap NOT closing, so 10x was not buying anything we could see.
#
# This is a bet, not an experiment: no arms, no controls. It runs all three
# stages and ends with top10_200k, the number that decides whether it ships.
#
# Stage-level idempotency: if final/ctx10_pre.pt already exists, stage A is
# skipped and the run resumes at the fine-tune. Relaunching after a dead pod
# therefore costs the current stage, not the whole pipeline.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/ft_only.log
SHARDS_DIR=$D/shards
MONTHS=${MONTHS:-"2026-01 2026-02 2026-03 2026-04 2026-05 2026-06"}
PRE=$D/ctx10_pre
FT=$D/ctx10_ft
PRE_CKPT=$D/ctx10_pre_start.pt
DM=${DM:-384}
LAYERS=${LAYERS:-12}
HEADS=${HEADS:-8}
SLOTS=${SLOTS:-10}
MLPG=${MLPG:-160}
# These are RUNAWAY BACKSTOPS, not planned stopping points. The stage is meant
# to end on --patience, i.e. when the metric stops improving. Setting a clock
# that binds first is how ctx5_pre and ctx5_ft2 both ended up truncated mid-climb
# with nobody knowing where their curves went -- twice, and it is in the registry
# twice. We are betting on this parameter set, not matching a previous run's
# compute, so the model gets to train until it is done.
PRE_H=${PRE_H:-48}
FT_H=${FT_H:-24}
PRE_LADDER=${PRE_LADDER:-"32 24 16 12"}
# Start at 16, NOT 24. p=24 passed a 30-step smoke at 23.0 of 23.55 GiB and then
# OOM'd an hour later (2026-08-16) when a batch of long games needed 100 MiB more
# -- sequence length varies ~10x per sample, so a short smoke never samples the
# tail. 16 is also the branch we chose deliberately: see docs/BRANCHES.md #2.
FT_LADDER=${FT_LADDER:-"16 12 10 8"}       # P in the PxK batch, K fixed at 4

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd /root/code || exit 1
# nproc LIES inside a RunPod container: it reports the host's cores, not the
# cgroup quota. Measured 2026-08-17 on the eval pod -- nproc said 48, the real
# quota was ~8, and 25 dataloader workers starved the GPU to 0%. The k=10
# gallery build ran at 28 bundles/s against k=5's 503/s. Size loader pools from
# cpuquota.py, never from nproc.
WORKERS=${WORKERS:-$($PY -c "from cpuquota import cpu_quota; print(cpu_quota())" 2>/dev/null || echo 8)}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Every exit must upload the log BEFORE printing NOVOL_ALL_DONE. That token is
# what novol_stopper.sh watches to delete the pod, so printing it first means a
# crash deletes its own diagnostic. That is exactly how the 2026-08-14 run was
# lost: it died ~20 min in, said ALL_DONE, and took the log with it.
finish() {
    cp -f "$LOG" "$D/ft_only_run.log" 2>/dev/null
    $PY runpod/s3io.py up "$D/ft_only_run.log" final/ft_only_run.log || true
    echo "$1"
    echo NOVOL_ALL_DONE
    exit "${2:-0}"
}

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) | $(nproc) vCPU ==="
echo "config: d_model $DM | layers $LAYERS | heads $HEADS | slots $SLOTS | ${MLPG} plies"

# All six months, not one. The deployed gallery centroids span 2026-01..06, so
# a model trained on single-month bundles is never shown that January-you and
# June-you are the same person -- a train/deploy mismatch, not just less data.
# Players are joined on lowercased USERNAME (pid is per shard).
SHARD=""
for M in $MONTHS; do
    $PY runpod/s3io.py down "data/mt/$M" "$SHARDS_DIR/$M" || finish NOVOL_UPLOAD_FAILED 1
    SHARD="$SHARD $SHARDS_DIR/$M"
done
echo "=== $(date -u +%H:%M:%S) shards: $SHARD ==="
df -h "$D" | tail -1

########################### trunk: no training here ##########################
# Prefer the finished trunk if stage A has already uploaded one; otherwise take
# the rolling partial. Either way this script never pre-trains.
$PY runpod/s3io.py down final/ctx10_pre.pt "$PRE_CKPT" 2>/dev/null
if [ -s "$PRE_CKPT" ]; then
    echo "=== $(date -u +%H:%M:%S) trunk: final ctx10_pre.pt ==="
else
    $PY runpod/s3io.py down final/ctx10_pre_partial.pt "$PRE_CKPT" || finish FT_ONLY_NO_TRUNK 1
    [ -s "$PRE_CKPT" ] || finish FT_ONLY_NO_TRUNK 1
    echo "=== $(date -u +%H:%M:%S) trunk: rolling partial ctx10_pre_partial.pt ==="
fi
$PY -c "import torch,sys; k=torch.load(sys.argv[1],map_location='cpu',weights_only=False); \
print(f'  trunk step {k.get(\"step\")} | slots {k.get(\"n_game_slots\")} | \
val {k.get(\"val\",{})}')" "$PRE_CKPT" || true

########################## stage B: contrastive ##############################
# The PxK batch is the memory risk here: 24x4 = 96 bundles x 1600 positions is
# TWICE the pre-train batch, so the proven p24 may not fit even though the
# pre-train did. Ladder down P rather than discover this eight hours in.
FP=0
for P in $FT_LADDER; do
    echo "=== $(date -u +%H:%M:%S) B: smoke p=$P k=4 ==="
    if $PY finetune_ctx.py --shard $SHARD --ckpt "$PRE_CKPT" --out "$D/smokeB" \
            --loss ms --p "$P" --k 4 --lr 3e-5 --warmup 20 --workers 12 \
            --steps 400 --eval-every 1000000 --patience 1000000 \
            --amp --max-hours 0.25 >"$D/smokeB_$P.log" 2>&1; then
        FP=$P; echo "  p=$P fits"; break
    fi
    echo "  p=$P failed:"; tail -3 "$D/smokeB_$P.log" | sed 's/^/    /'
    rm -rf "$D/smokeB"
done
[ "$FP" -eq 0 ] && finish BIG_RUN_NO_FT_BATCH 1

( while :; do sleep 1800
    [ -s "$FT/best.pt" ] && $PY runpod/s3io.py up "$FT/best.pt" final/ctx10_ft_best_partial.pt >/dev/null 2>&1
    [ -s "$FT/last.pt" ] && $PY runpod/s3io.py up "$FT/last.pt" final/ctx10_ft_partial.pt >/dev/null 2>&1
  done ) & KEEPER=$!

echo "=== $(date -u +%H:%M:%S) B: contrastive, p=$FP k=4, cap ${FT_H}h ==="
# --eval-batches 100, not the usual 25. Measured on ctx5_ft4, val_supcon at 25
# batches has sd 0.0158 while the gaps we are ranking are ~0.011 -- so "best"
# was selecting the luckiest eval rather than the best model. 4x the batches
# roughly halves that.
$PY finetune_ctx.py --shard $SHARD --ckpt "$PRE_CKPT" --out "$FT" \
        --loss ms --p "$FP" --k 4 --lr 3e-5 --warmup 200 --workers 24 \
        --steps 100000000 --eval-every 8000 --eval-batches 100 \
        --patience 12 --amp --compile --max-hours "$FT_H" \
    && echo B_FINETUNE_DONE || echo B_FINETUNE_FAILED
kill "$KEEPER" 2>/dev/null

for f in best.pt last.pt history.json; do
    [ -s "$FT/$f" ] && $PY runpod/s3io.py up "$FT/$f" "final/ctx10_ft_$f"
done

############################# stage C: evaluate ##############################
# --ks 5,10 on purpose. k=5 is directly comparable to the shipped 0.8570; k=10
# is what this model was actually built for, and is the number that says whether
# the context bet paid.
for W in best last; do
    [ -s "$FT/$W.pt" ] || continue
    echo "=== $(date -u +%H:%M:%S) C: eval $W.pt | top-10 @ 200k, k=5 and k=10 ==="
    # ONE shard here on purpose: top10_200k is defined on mt/2026-01 and the
    # registry's 0.8570 was measured there. gallery_ctx.py also takes a single
    # --shard. Training spans six months; the yardstick must not move.
    if $PY gallery_ctx.py --ckpt "$FT/$W.pt" --shard "$SHARDS_DIR/2026-01" \
            --out "$D/ctx10_${W}_eval.json" --ks 5,10 \
            --gallery-games 64 --min-gallery-games 8 \
            --gallery-players 200000 --query-players 5000 \
            --sizes 10000,50000,200000 --batch 192 --workers "$WORKERS"; then
        echo "C_EVAL_${W}_DONE"
        $PY runpod/s3io.py up "$D/ctx10_${W}_eval.json" "final/ctx10_${W}_eval.json" || true
    else
        echo "C_EVAL_${W}_FAILED"
    fi
    cp -f "$LOG" "$D/big_run_run.log" 2>/dev/null
    $PY runpod/s3io.py up "$D/big_run_run.log" final/big_run_run.log || true
done

echo "=== $(date -u +%H:%M:%S) uploading ==="
finish NOVOL_UPLOAD_OK 0
} >> "$LOG" 2>&1
