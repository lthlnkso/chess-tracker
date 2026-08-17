#!/usr/bin/env bash
# Evaluate ONE checkpoint at the largest gallery a single shard supports.
#
# Why not 558,735: that number comes from union_gallery.py merging six months by
# USERNAME. gallery_ctx.py takes one shard, and mt/2026-01 holds 282,485 players
# with >=8 clocked games. So this is half production scale, with a size curve
# (10k -> 282k) at both k=5 and k=10 so the slope toward 558k is visible.
#
# Reads a checkpoint straight from S3, so it can evaluate a rolling partial from
# a run that is still training on another pod.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/eval_only.log
SHARD=$D/shard
CKPT=$D/eval_ckpt.pt
CKPT_KEY=${CKPT_KEY:-final/ctx10_ft_best_partial.pt}
MONTH=${MONTH:-2026-01}
GAL_PLAYERS=${GAL_PLAYERS:-282000}
KS=${KS:-5,10}
SIZES=${SIZES:-10000,50000,100000,200000,282000}
TAG=${TAG:-ctx10_best}

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

finish() {
    cp -f "$LOG" "$D/${TAG}_eval.log" 2>/dev/null
    $PY runpod/s3io.py up "$D/${TAG}_eval.log" "final/${TAG}_eval.log" || true
    echo "$1"; echo NOVOL_ALL_DONE; exit "${2:-0}"
}

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "ckpt $CKPT_KEY | shard $MONTH | gallery $GAL_PLAYERS | ks $KS"

$PY runpod/s3io.py down "$CKPT_KEY" "$CKPT" || finish EVAL_NO_CKPT 1
[ -s "$CKPT" ] || finish EVAL_NO_CKPT 1
$PY runpod/s3io.py down "data/mt/$MONTH" "$SHARD" || finish EVAL_NO_SHARD 1

$PY -c "import torch,sys; k=torch.load(sys.argv[1],map_location='cpu',weights_only=False); \
print(f'  ckpt step {k.get(\"step\")} | slots {k.get(\"n_game_slots\")} | \
d_embed {k.get(\"d_embed\")} | loss {k.get(\"loss\")} | \
val {k.get(\"val_loss\", k.get(\"val_supcon\"))}')" "$CKPT" || true

echo "=== $(date -u +%H:%M:%S) gallery + eval ==="
if $PY gallery_ctx.py --ckpt "$CKPT" --shard "$SHARD" \
        --out "$D/${TAG}_eval.json" --ks "$KS" \
        --gallery-games 64 --min-gallery-games 8 \
        --gallery-players "$GAL_PLAYERS" --query-players 5000 \
        --sizes "$SIZES" --batch 192 --workers "$WORKERS"; then
    echo EVAL_DONE
    $PY runpod/s3io.py up "$D/${TAG}_eval.json" "final/${TAG}_eval.json" || true
else
    echo EVAL_FAILED
fi
finish NOVOL_UPLOAD_OK 0
} >> "$LOG" 2>&1
