#!/usr/bin/env bash
# The whole job on a pod with NO network volume: pull inputs from S3, run, push
# results back to S3.
#
# The volume mount is what was failing (see s3io.py), so nothing here touches
# /workspace. Inputs come down to container disk and results go straight back to
# the volume's S3 gateway, which means the artifacts survive the pod and are
# fetchable with runpod/fetch_final.sh exactly as before.
#
# Credentials arrive in this process's environment from /root/.s3env, which the
# launcher writes and this script shreds before doing anything else.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/colour.log
SHARD=$D/shard
CKPT=$D/ctx5_ft2.pt
CODE=/root/code

mkdir -p "$D"

# Load and immediately destroy the on-disk copy of the S3 keys.
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd "$CODE" || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="

echo "=== $(date -u +%H:%M:%S) fetching inputs from S3 ==="
$PY runpod/s3io.py down final/ctx5_ft2.pt "$CKPT" || { echo NOVOL_FETCH_FAILED; echo NOVOL_ALL_DONE; exit 1; }
$PY runpod/s3io.py down data/mt/2026-01 "$SHARD" || { echo NOVOL_FETCH_FAILED; echo NOVOL_ALL_DONE; exit 1; }
df -h "$D" | tail -1

echo "=== $(date -u +%H:%M:%S) colour-split sweep @200k ==="
if $PY gallery_ctx.py --ckpt "$CKPT" --shard "$SHARD" \
        --out "$D/ctx5_colour.json" \
        --colour-split --ks 5 --gallery-games 64 \
        --gallery-players 200000 --query-players 5000 --workers 22; then
    echo COLOUR_SWEEP_DONE
else
    echo COLOUR_SWEEP_FAILED
fi

echo "=== $(date -u +%H:%M:%S) deployment gallery @200k (combined + white + black) ==="
if $PY play/build_gallery.py --ckpt "$CKPT" --shard "$SHARD" \
        --out "$D/gallery_deploy.npz" \
        --players 200000 --gallery-games 32 --colour \
        --batch 192 --workers 22; then
    echo GALLERY_DONE
else
    echo GALLERY_FAILED
fi

echo "=== $(date -u +%H:%M:%S) uploading results ==="
ok=1
for f in ctx5_colour.json gallery_deploy.npz; do
    if [ -s "$D/$f" ]; then
        $PY runpod/s3io.py up "$D/$f" "final/$f" || ok=0
    else
        echo "  (no $f produced)"
    fi
done
# The log is worth having even when the run failed -- it is the only record of
# why, once the pod is gone.
cp -f "$LOG" "$D/colour_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/colour_run.log" "final/colour_run.log" || true

if [ "$ok" -eq 1 ] && { [ -s "$D/ctx5_colour.json" ] || [ -s "$D/gallery_deploy.npz" ]; }; then
    echo NOVOL_UPLOAD_OK
else
    echo NOVOL_UPLOAD_FAILED
fi
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
