#!/usr/bin/env bash
# How many embedding dimensions does identification need?
#
# The shipped identifier emits 128. Nothing chose that number -- it is the
# default in model.py. Free evidence first: PCA-truncating the shipped 558,735
# gallery costs r@10 0.620 -> 0.570 at 64 dims, 0.463 at 32, 0.235 at 16. But
# PCA optimises VARIANCE, not separability, so it is a lower bound on what a
# model trained at that width would reach. This run measures the real thing.
#
# One control that cannot be skipped: the registry's 128-d number (top10_200k
# 0.8570) comes from a 775k-step, 6-hour run. Comparing a 75-minute 32-d arm
# against it would measure training budget, not width. Every arm here gets the
# SAME budget, including a fresh 128-d arm, and only arms are compared.
#
# Expect throughput to be flat. embed_head is 0.3% of the parameters (7.90M at
# d_embed 16 vs 7.92M at 128) and the cost is the trunk running 5 games x 160
# plies through 8 layers. What shrinks is the gallery: 143 MB -> 18 MB at 16-d,
# and the search matmul with it.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/dim_sweep.log
SHARD=$D/shard
TRUNK=$D/ctx5_pre.pt
DIMS=${DIMS:-"128 64 32 16"}
FT_H=${FT_H:-1.25}
GAL_PLAYERS=${GAL_PLAYERS:-200000}

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) | $(nproc) vCPU ==="

$PY runpod/s3io.py down final/ctx5_pre.pt "$TRUNK" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }
$PY runpod/s3io.py down data/mt/2026-01 "$SHARD" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }

for K in $DIMS; do
    OUT=$D/ft_d$K
    echo "=== $(date -u +%H:%M:%S) d_embed=$K | fine-tune, cap ${FT_H}h ==="
    # --new-embed re-initialises embed_head.2 only; the trunk loads strictly, so
    # every arm starts from an identical trunk and differs solely in width.
    if $PY finetune_ctx.py --shard "$SHARD" --ckpt "$TRUNK" --out "$OUT" \
            --loss ms --d-embed "$K" --new-embed \
            --p 24 --k 4 --lr 3e-5 --warmup 200 --workers 24 \
            --eval-every 8000 --eval-batches 25 --patience 1000000 \
            --amp --compile --max-hours "$FT_H"; then
        echo "DIM_${K}_TRAIN_DONE"
    else
        echo "DIM_${K}_TRAIN_FAILED"
        continue
    fi

    # Same gallery protocol as the registry's top10_200k_5games, so these land
    # on a scale we already have numbers on.
    echo "=== $(date -u +%H:%M:%S) d_embed=$K | gallery + eval ==="
    if $PY gallery_ctx.py --ckpt "$OUT/last.pt" --shard "$SHARD" \
            --out "$D/gal_d$K.npz" --ks 5 \
            --gallery-games 64 --min-gallery-games 8 \
            --gallery-players "$GAL_PLAYERS" --query-players 5000 \
            --sizes 10000,50000,200000 \
            --batch 192 --workers 24; then
        echo "DIM_${K}_EVAL_DONE"
    else
        echo "DIM_${K}_EVAL_FAILED"
    fi

    for f in last.pt history.json; do
        [ -s "$OUT/$f" ] && $PY runpod/s3io.py up "$OUT/$f" "final/dim${K}_$f"
    done
    cp -f "$LOG" "$D/dim_sweep_run.log" 2>/dev/null
    $PY runpod/s3io.py up "$D/dim_sweep_run.log" final/dim_sweep_run.log || true
done

echo "=== $(date -u +%H:%M:%S) uploading ==="
cp -f "$LOG" "$D/dim_sweep_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/dim_sweep_run.log" final/dim_sweep_run.log && echo NOVOL_UPLOAD_OK || echo NOVOL_UPLOAD_FAILED
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
