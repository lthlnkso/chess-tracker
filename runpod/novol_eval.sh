#!/usr/bin/env bash
# Re-run only the k=5 eval that died, for a checkpoint that already exists.
#
# The k=5 stage is the memory-heaviest: five games of up to 160 plies per bundle,
# collated into one padded tensor. It died with 22 dataloader workers on a 9-vCPU
# pod -- a worker was killed and the parent saw a ConnectionResetError from the
# multiprocessing teardown, which looks like a network fault and is not one.
# Workers are conservative here and scaled to the machine, not guessed.
#
# k=5 also carries the colour-split arm (it runs at max(ks)), which is the whole
# point: the main arm scores MIXED-colour queries, and a colour-trained model is
# off-distribution there. The colour-matched arm is the one that tests it under
# the conditions it was trained for.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/eval.log
SHARD=$D/shard
CKPT=$D/ctx5_ftc2.pt
NW=${NW:-8}

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) | $(nproc) vCPU | $(free -g | sed -n 2p | awk '{print $2}') GB RAM ==="

$PY runpod/s3io.py down final/ctx5_ftc2.pt "$CKPT" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }
$PY runpod/s3io.py down data/mt/2026-01 "$SHARD" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }

echo "=== $(date -u +%H:%M:%S) k=5 eval, both arms, workers=$NW ==="
if $PY gallery_ctx.py --ckpt "$CKPT" --shard "$SHARD" \
        --out "$D/ctx5_ftc2_k5.json" \
        --colour-split --ks 5 --gallery-games 64 \
        --gallery-players 200000 --query-players 5000 \
        --sizes 1000,10000,50000,100000,200000 --workers "$NW"; then
    echo EVAL_DONE
else
    echo EVAL_FAILED
fi

echo "=== $(date -u +%H:%M:%S) uploading ==="
ok=1
if [ -s "$D/ctx5_ftc2_k5.json" ]; then
    $PY runpod/s3io.py up "$D/ctx5_ftc2_k5.json" final/ctx5_ftc2_k5.json || ok=0
else
    ok=0
fi
cp -f "$LOG" "$D/eval_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/eval_run.log" final/eval_run.log || true

if [ "$ok" -eq 1 ]; then echo NOVOL_UPLOAD_OK; else echo NOVOL_UPLOAD_FAILED; fi
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
