#!/usr/bin/env bash
# The bet: 3x params, 10 game slots, full pipeline to a usable identifier.
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
LOG=$D/big_run.log
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
FT_LADDER=${FT_LADDER:-"24 20 16 12"}      # P in the PxK batch, K fixed at 4

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
    cp -f "$LOG" "$D/big_run_run.log" 2>/dev/null
    $PY runpod/s3io.py up "$D/big_run_run.log" final/big_run_run.log || true
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

########################### stage A: pre-train ###############################
$PY runpod/s3io.py down final/ctx10_pre.pt "$PRE_CKPT" 2>/dev/null
if [ -s "$PRE_CKPT" ]; then
    echo "=== $(date -u +%H:%M:%S) stage A SKIPPED, ctx10_pre.pt already in S3 ==="
else
    # Warm start from the rolling partial if one exists, so relaunching to fix a
    # cap costs the optimizer state, not the weights. --init loads weights only
    # (fresh AdamW, warmup restarts) which at 80k+ steps is a rounding error
    # against redoing the hours.
    RESUME=""
    $PY runpod/s3io.py down final/ctx10_pre_partial.pt "$D/ctx10_resume.pt" 2>/dev/null
    if [ -s "$D/ctx10_resume.pt" ]; then
        RESUME="--init $D/ctx10_resume.pt"
        echo "=== $(date -u +%H:%M:%S) warm start from ctx10_pre_partial.pt ==="
    fi
    PB=0
    for B in $PRE_LADDER; do
        echo "=== $(date -u +%H:%M:%S) A: smoke batch $B ==="
        if $PY train_multigame.py --shard $SHARD --out "$D/smokeA" \
                --max-games "$SLOTS" --max-len-per-game "$MLPG" \
                --d-model "$DM" --layers "$LAYERS" --heads "$HEADS" \
                --batch "$B" --lr 1.5e-4 --warmup 50 \
                --plies-per-game 8 --n-cand 32 --workers 12 \
                --steps 30 --eval-every 1000000 --amp --balance-elo \
                --max-hours 0.15 >"$D/smokeA_$B.log" 2>&1; then
            PB=$B; echo "  batch $B fits"; break
        fi
        echo "  batch $B failed:"; tail -3 "$D/smokeA_$B.log" | sed 's/^/    /'
        rm -rf "$D/smokeA"
    done
    [ "$PB" -eq 0 ] && finish BIG_RUN_NO_PRE_BATCH 1

    ( sleep 600
      while :; do
        [ -s "$PRE/last.pt" ] && $PY runpod/s3io.py up "$PRE/last.pt" final/ctx10_pre_partial.pt >/dev/null 2>&1
        sleep 1200
      done ) & KEEPER=$!

    echo "=== $(date -u +%H:%M:%S) A: pre-train, batch $PB, cap ${PRE_H}h ==="
    # patience 10 / min-delta 2e-4 is ctx5_pre's own recipe. ctx5_pre stopped on
    # a WALL CLOCK with early stopping effectively off, so nobody knows where its
    # curve ended. This one is allowed to stop when it converges.
    $PY train_multigame.py --shard $SHARD --out "$PRE" \
            --max-games "$SLOTS" --max-len-per-game "$MLPG" \
            --d-model "$DM" --layers "$LAYERS" --heads "$HEADS" \
            --batch "$PB" --lr 1.5e-4 --warmup 1000 \
            --plies-per-game 8 --n-cand 32 --workers 24 \
            --steps 100000000 --eval-every 8000 --eval-batches 25 \
            --balance-elo --patience 10 --min-delta 0.0002 $RESUME \
            --amp --compile --max-hours "$PRE_H" \
        && echo A_PRETRAIN_DONE || { echo A_PRETRAIN_FAILED; kill "$KEEPER" 2>/dev/null; }
    kill "$KEEPER" 2>/dev/null

    [ -s "$PRE/last.pt" ] || finish BIG_RUN_NO_TRUNK 1
    $PY runpod/s3io.py up "$PRE/last.pt" final/ctx10_pre.pt
    [ -s "$PRE/history.json" ] && $PY runpod/s3io.py up "$PRE/history.json" final/ctx10_pre_history.json
    cp -f "$PRE/last.pt" "$PRE_CKPT"
fi

########################## stage B: contrastive ##############################
# The PxK batch is the memory risk here: 24x4 = 96 bundles x 1600 positions is
# TWICE the pre-train batch, so the proven p24 may not fit even though the
# pre-train did. Ladder down P rather than discover this eight hours in.
FP=0
for P in $FT_LADDER; do
    echo "=== $(date -u +%H:%M:%S) B: smoke p=$P k=4 ==="
    if $PY finetune_ctx.py --shard $SHARD --ckpt "$PRE_CKPT" --out "$D/smokeB" \
            --loss ms --p "$P" --k 4 --lr 3e-5 --warmup 20 --workers 12 \
            --steps 30 --eval-every 1000000 --patience 1000000 \
            --amp --max-hours 0.15 >"$D/smokeB_$P.log" 2>&1; then
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
