#!/usr/bin/env bash
# Runs ON the pod. Everything that needs the big shard, done in ONE boot.
#
# Boot is the dominant cost of a short job on this project: the pytorch image is
# 16.3 GB compressed and a cold host spends 15-20 minutes pulling it before sshd
# exists. That is roughly a third of the cost of a 45-minute run, and it is paid
# again on every fresh pod. So this script batches both remaining pieces of work
# -- the colour-split measurement and the deployment gallery -- and the launcher
# STOPS the pod at the end instead of terminating it, which keeps the container
# disk and makes the next start take seconds.
#
# Cap choice is not arbitrary. The sweep uses 64 so it is directly comparable to
# ctx5_curve_rich.json; the gallery build uses 32 PER COLOUR, which is the same
# 64 games per player, so the deployed centroids match what was measured.
set -u

VOL=/workspace
# Overridable: the volume venv inherits torch from the 16.3 GB pytorch image via
# --system-site-packages, so a run booted from a slim image needs a different,
# self-contained interpreter (see slim_launch.sh).
PY=${PY:-$VOL/venv/bin/python}
LOG=$VOL/colour.log
SHARD=$VOL/data/mt/2026-01
CKPT=$VOL/final/ctx5_ft2.pt

cd "$VOL/code" || exit 1
mkdir -p "$VOL/final"

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="
[ -s "$CKPT" ] || echo "MISSING CKPT $CKPT"
[ -s "$SHARD/meta.npy" ] || echo "MISSING SHARD $SHARD"

echo "=== $(date -u +%H:%M:%S) colour-split sweep @200k ==="
if $PY gallery_ctx.py --ckpt "$CKPT" --shard "$SHARD" \
        --out "$VOL/final/ctx5_colour.json" \
        --colour-split --ks 5 --gallery-games 64 \
        --gallery-players 200000 --query-players 5000 --workers 22; then
    echo COLOUR_SWEEP_DONE
else
    echo COLOUR_SWEEP_FAILED
fi

echo "=== $(date -u +%H:%M:%S) deployment gallery @200k (combined + white + black) ==="
if $PY play/build_gallery.py --ckpt "$CKPT" --shard "$SHARD" \
        --out "$VOL/final/gallery_deploy.npz" \
        --players 200000 --gallery-games 32 --colour \
        --batch 192 --workers 22; then
    echo GALLERY_DONE
else
    echo GALLERY_FAILED
fi

sync
echo "=== $(date -u +%H:%M:%S) final: ==="
ls -la "$VOL/final"
echo COLOUR_ALL_DONE
} >> "$LOG" 2>&1
