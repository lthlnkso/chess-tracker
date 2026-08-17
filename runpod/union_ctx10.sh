#!/usr/bin/env bash
# Build the PRODUCTION centroid set with the ctx10 identifier.
#
# This is the 558,735-player union across 2026-01..06 -- the roster the demo
# actually searches, not the 200k proxy stage C reports. Same --min-games 13 and
# --gallery-games 64 as the shipped gallery_2026.npz, so the roster is identical
# and the only variable is the model.
#
# k=10, not the shipped k=5. That is the design point of this trunk: stage C
# measured r@10 0.9628 at k=10 against 0.8788 at k=5 on the same 200k gallery.
#
# Chained rather than launched by hand: stage C still owns the GPU, so this
# waits for it to exit and starts the instant it does. No dead pod time, and
# rerunning it is harmless -- it skips the build if the output already exists.
set -u

D=/data
PY=/root/venv/bin/python
LOG=$D/union.log
OUT=$D/gallery_ctx10.npz
SHARDS="$D/shards/2026-01 $D/shards/2026-02 $D/shards/2026-03 $D/shards/2026-04 $D/shards/2026-05 $D/shards/2026-06"

cd /root/code || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# s3io.py reads its credentials from the environment, and every launcher that
# uses it sources .s3env first. This script did not, and the 2026-08-17 build
# finished all 2.5M bundles and then died on KeyError: 'RUNPOD_S3_ACCESS_KEY'.
# Worse, .s3env is SHREDDED by whichever launcher reads it first, so a chained
# job cannot rely on it existing -- ft_only.sh had already consumed it. Keep a
# copy at .s3env.keep for chained stages, and fall back to it here.
for f in /root/.s3env /root/.s3env.keep; do
    if [ -f "$f" ]; then set -a; . "$f"; set +a; break; fi
done
if [ -z "${RUNPOD_S3_ACCESS_KEY:-}" ]; then
    echo "WARNING: no S3 credentials; the build will run but cannot upload." >&2
fi

{
if [ -s "$OUT" ]; then
    echo "=== $(date -u +%H:%M:%S) $OUT already built, nothing to do ==="
    exit 0
fi

echo "=== $(date -u +%H:%M:%S) waiting for stage C to release the GPU ==="
while pgrep -f "gallery_ctx[.]py" >/dev/null; do sleep 60; done
echo "=== $(date -u +%H:%M:%S) GPU free ==="

# Pick best.pt vs last.pt on the MEASURED number, not on the folklore that last
# always wins. Falls back to last.pt only if neither eval produced a readable
# r@10 -- never silently builds against a checkpoint we cannot justify.
$PY - <<'PYEOF' > "$D/ckpt_choice.txt"
import json
def r10(p):
    try:
        return json.load(open(p))["pools"]["10"]["direct_at_full_gallery"]["recall@10"]
    except Exception:
        return -1.0
b, l = r10("/data/ctx10_best_eval.json"), r10("/data/ctx10_last_eval.json")
import sys
print(f"best.pt r@10(k=10) = {b:.4f}", file=sys.stderr)
print(f"last.pt r@10(k=10) = {l:.4f}", file=sys.stderr)
print("/data/ctx10_ft/best.pt" if b >= l else "/data/ctx10_ft/last.pt")
PYEOF
CKPT=$(cat "$D/ckpt_choice.txt")
echo "=== $(date -u +%H:%M:%S) building with $CKPT ==="

# Token discipline, the same as finish() in big_run.sh: upload BEFORE printing
# NOVOL_ALL_DONE, because that token is what the stopper watches to delete the
# pod. NOVOL_UPLOAD_FAILED is the stopper's own veto -- printing it on any
# failure path leaves the pod up for rescue instead of destroying a 3-hour
# build, which is exactly how a run was lost on 2026-08-14.
if $PY union_gallery.py --ckpt "$CKPT" --shards $SHARDS --out "$OUT" \
        --k 10 --gallery-games 64 --min-games 13 --batch 192 --workers 24; then
    echo UNION_BUILD_DONE
    if $PY runpod/s3io.py up "$OUT" final/gallery_ctx10.npz; then
        echo UNION_UPLOAD_OK
    else
        echo NOVOL_UPLOAD_FAILED
    fi
else
    echo UNION_BUILD_FAILED
    echo NOVOL_UPLOAD_FAILED
fi
echo "=== $(date -u +%H:%M:%S) done ==="
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
