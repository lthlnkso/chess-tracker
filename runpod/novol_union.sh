#!/usr/bin/env bash
# Build the 2026 gallery: one centroid per player, pooled across every month.
#
# This is the product artifact. The January-only gallery was not a smaller
# version of it but a biased one -- eligibility was decided by activity in one
# arbitrary month, so an account with 274 games across the year but 3 in January
# was unfindable at any gallery size.
#
# Downloads every ingested month (~39 GB) to container disk over S3, so it needs
# neither the network volume mount nor a particular datacenter.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/union.log
CKPT=$D/ctx5_ft2.pt
MONTHS=${MONTHS:-"2026-01 2026-02 2026-03 2026-04 2026-05 2026-06"}

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="

echo "=== $(date -u +%H:%M:%S) fetching model + shards ==="
$PY runpod/s3io.py down final/ctx5_ft2.pt "$CKPT" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }
SHARDS=""
for m in $MONTHS; do
    $PY runpod/s3io.py down "data/mt/$m" "$D/$m" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }
    SHARDS="$SHARDS $D/$m"
done
df -h "$D" | tail -1

echo "=== $(date -u +%H:%M:%S) union gallery ==="
# shellcheck disable=SC2086
if $PY union_gallery.py --ckpt "$CKPT" --shards $SHARDS \
        --out "$D/gallery_2026.npz" \
        --k 5 --gallery-games 64 --min-games 13 \
        --batch 192 --workers 22; then
    echo UNION_DONE
else
    echo UNION_FAILED
fi

echo "=== $(date -u +%H:%M:%S) uploading ==="
ok=1
if [ -s "$D/gallery_2026.npz" ]; then
    $PY runpod/s3io.py up "$D/gallery_2026.npz" final/gallery_2026.npz || ok=0
else
    ok=0
fi
cp -f "$LOG" "$D/union_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/union_run.log" final/union_run.log || true

if [ "$ok" -eq 1 ]; then echo NOVOL_UPLOAD_OK; else echo NOVOL_UPLOAD_FAILED; fi
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
