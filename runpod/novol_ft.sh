#!/usr/bin/env bash
# Fine-tune on colour-homogeneous bundles, then measure both retrieval arms.
#
# Hypothesis: the colour-split arm lost not because colour is uninformative but
# because the encoder had never seen a single-colour bundle. Uniform sampling
# makes an all-one-colour 5-game bundle a ~5% case (measured), yet that is 100%
# of what a colour-split gallery is queried with. The measured centroid-richness
# curve says halving a 64-game centroid should cost ~0.6%, and the split arm lost
# 2.3 points of top-10 with TWICE the query games -- too big a gap for halving.
#
# Everything except the sampling matches ctx5_ft2 exactly -- same trunk, same
# shard, same loss, p, k, lr, warmup and the same 6-hour budget -- so this is a
# controlled comparison rather than another confounded one.
#
# The decisive number is INTERNAL to this model: combined vs split_fused, both
# scored from the same checkpoint. A budget or hardware difference cannot reach
# that comparison, which is what makes it worth the GPU hours.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/ft.log
SHARD=$D/shard
TRUNK=$D/ctx5_pre.pt
OUT=$D/ctx5_ftc
FT_H=${FT_H:-6}

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="

echo "=== $(date -u +%H:%M:%S) fetching inputs ==="
$PY runpod/s3io.py down final/ctx5_pre.pt "$TRUNK" || { echo FT_FETCH_FAILED; echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }
$PY runpod/s3io.py down data/mt/2026-01 "$SHARD" || { echo FT_FETCH_FAILED; echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }

echo "=== $(date -u +%H:%M:%S) fine-tune, colour-homogeneous bundles, ${FT_H}h ==="
if $PY finetune_ctx.py --shard "$SHARD" --ckpt "$TRUNK" --out "$OUT" \
        --same-colour \
        --loss ms --max-hours "$FT_H" --steps 100000000 \
        --p 24 --k 4 --lr 3e-5 --warmup 200 --workers 24 \
        --eval-every 8000 --eval-batches 25 --patience 1000000 \
        --amp --compile; then
    echo FT_TRAIN_DONE
else
    echo FT_TRAIN_FAILED
fi

# Upload the model BEFORE the eval: six hours of training must not be hostage to
# a later stage failing.
[ -s "$OUT/last.pt" ] && $PY runpod/s3io.py up "$OUT/last.pt" final/ctx5_ftc.pt
[ -s "$OUT/history.json" ] && $PY runpod/s3io.py up "$OUT/history.json" final/ctx5_ftc_history.json

echo "=== $(date -u +%H:%M:%S) eval: combined vs colour-split, same checkpoint ==="
if [ -s "$OUT/last.pt" ] && $PY gallery_ctx.py --ckpt "$OUT/last.pt" --shard "$SHARD" \
        --out "$D/ctx5_ftc_colour.json" \
        --colour-split --ks 5 --gallery-games 64 \
        --gallery-players 200000 --query-players 5000 --workers 22; then
    echo FT_EVAL_DONE
else
    echo FT_EVAL_FAILED
fi

echo "=== $(date -u +%H:%M:%S) uploading ==="
ok=1
[ -s "$D/ctx5_ftc_colour.json" ] && { $PY runpod/s3io.py up "$D/ctx5_ftc_colour.json" final/ctx5_ftc_colour.json || ok=0; }
cp -f "$LOG" "$D/ft_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/ft_run.log" final/ft_run.log || true

if [ "$ok" -eq 1 ] && [ -s "$OUT/last.pt" ]; then
    echo NOVOL_UPLOAD_OK
else
    echo NOVOL_UPLOAD_FAILED
fi
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
