#!/usr/bin/env bash
# Verifier v3: retrain with negatives mined from the DEPLOYED gallery shortlist.
#
# v2 saturated at val AUC 0.861 and then ranked a real visitor 50th of 100 --
# a coin flip -- because its negatives came from a 40,000-player training shard
# while serving ranks against 558,735. It got very good at a question the
# product never asks.
#
# The shortlist is built HERE rather than shipped, because it maps this shard's
# pids to gallery rows: a shortlist built against a different shard would pair
# every player with a stranger's negatives and still train perfectly happily.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/verify3.log
SHARD=$D/shard
TRUNK=$D/ctx5_ft2.pt
GAL=$D/gallery_2026.npz
PACK=$D/verifier_pack.npz
SHORT=$D/shortlist.npz
OUT=$D/verifier3
FT_H=${FT_H:-8}
POOL=${POOL:-120000}
EXTRA=${EXTRA:-96}

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) | $(nproc) vCPU ==="

for spec in "final/ctx5_ft2.pt $TRUNK" "final/gallery_2026.npz $GAL" \
            "final/verifier_pack.npz $PACK" "data/mt/2026-01 $SHARD"; do
    set -- $spec
    $PY runpod/s3io.py down "$1" "$2" || { echo VERIFY3_FETCH_FAILED; echo NOVOL_ALL_DONE; exit 1; }
done

# Idempotent: a relaunch onto a pod that already built this skips straight to
# training rather than paying for the embed pass twice.
if [ ! -s "$SHORT" ]; then
    echo "=== $(date -u +%H:%M:%S) building gallery shortlists ==="
    $PY build_shortlist.py --ckpt "$TRUNK" --gallery "$GAL" --shard "$SHARD" \
        --min-games 6 --players "$POOL" --topn 128 --batch 64 --workers 16 \
        --out "$SHORT" || { echo SHORTLIST_FAILED; echo NOVOL_ALL_DONE; exit 1; }
fi

( while :; do sleep 1800
    [ -s "$OUT/best.pt" ] && $PY runpod/s3io.py up "$OUT/best.pt" final/verifier3_partial.pt >/dev/null 2>&1
  done ) &
KEEPER=$!

echo "=== $(date -u +%H:%M:%S) verifier v3 fine-tune, cap ${FT_H}h ==="
if $PY verify3.py train --shard "$SHARD" --ckpt "$TRUNK" --out "$OUT" \
        --shortlist "$SHORT" --pack "$PACK" \
        --k 5 --min-games 6 --neighbours 64 --mlpg 60 \
        --batch 48 --extra "$EXTRA" --batches-per-epoch 600 --eval-batches 40 \
        --eval-every 800 --patience 12 \
        --workers 16 --lr 6e-5 --max-hours "$FT_H"; then
    echo VERIFY3_TRAIN_DONE
else
    echo VERIFY3_TRAIN_FAILED
fi
kill "$KEEPER" 2>/dev/null

for f in best.pt last.pt history.json; do
    [ -s "$OUT/$f" ] && $PY runpod/s3io.py up "$OUT/$f" "final/verifier3_$f"
done
[ -s "$SHORT" ] && $PY runpod/s3io.py up "$SHORT" final/shortlist.npz

echo "=== $(date -u +%H:%M:%S) uploading ==="
cp -f "$LOG" "$D/verify3_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/verify3_run.log" final/verify3_run.log || true
if [ -s "$OUT/best.pt" ] || [ -s "$OUT/last.pt" ]; then echo NOVOL_UPLOAD_OK; else echo NOVOL_UPLOAD_FAILED; fi
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
