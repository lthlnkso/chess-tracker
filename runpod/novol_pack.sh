#!/usr/bin/env bash
# Build the verifier pack: a few games per gallery player, small enough to ship.
set -u
D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/pack.log
MONTHS=${MONTHS:-"2026-01 2026-02 2026-03 2026-04 2026-05 2026-06"}

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi
cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start ==="
$PY runpod/s3io.py down final/gallery_2026.npz "$D/gallery_2026.npz" || \
    { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }
SHARDS=""
for m in $MONTHS; do
    $PY runpod/s3io.py down "data/mt/$m" "$D/$m" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }
    SHARDS="$SHARDS $D/$m"
done
df -h "$D" | tail -1

echo "=== $(date -u +%H:%M:%S) building pack ==="
# shellcheck disable=SC2086
if $PY build_verifier_pack.py --gallery "$D/gallery_2026.npz" --shards $SHARDS \
        --out "$D/verifier_pack.npz" --games 4 --plies 60; then
    echo PACK_OK
else
    echo PACK_FAILED
fi

echo "=== $(date -u +%H:%M:%S) uploading ==="
ok=1
[ -s "$D/verifier_pack.npz" ] && { $PY runpod/s3io.py up "$D/verifier_pack.npz" final/verifier_pack.npz || ok=0; } || ok=0
cp -f "$LOG" "$D/pack_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/pack_run.log" final/pack_run.log || true
if [ "$ok" -eq 1 ]; then echo NOVOL_UPLOAD_OK; else echo NOVOL_UPLOAD_FAILED; fi
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
