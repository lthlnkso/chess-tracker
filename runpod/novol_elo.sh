#!/usr/bin/env bash
# Teach the trunk to play at a requested rating.
#
# score_candidates() is a dot product of the trunk state with candidate
# encodings, so move choice cannot depend on rating unless rating enters the
# trunk -- the elo head only ever fed embed_head. A zero-initialised embedding
# added to the trunk input fixes that, and starts as an exact no-op so this is a
# fine-tune of a good model rather than a fresh 25-hour pre-train.
#
# Why it matters beyond being a nice knob: the demo's opponent currently plays
# one blend averaged over every rating in the data, matching no human in the
# gallery, and the identifier reads the opponent's plies as half of every game.
# A visitor's real games have opponents near their own rating.
#
# --elo-drop keeps the unconditioned path trained, so a caller that supplies no
# rating still gets sensible play instead of an untrained code path.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/elo.log
SHARD=$D/shard
TRUNK=$D/ctx5_pre.pt
OUT=$D/ctx5_pre_elo
FT_H=${FT_H:-10}

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

( while :; do sleep 1800
    [ -s "$OUT/last.pt" ] && $PY runpod/s3io.py up "$OUT/last.pt" final/ctx5_pre_elo_partial.pt >/dev/null 2>&1
  done ) &
KEEPER=$!

echo "=== $(date -u +%H:%M:%S) rating-conditioned fine-tune, cap ${FT_H}h ==="
# lr well below the 1.5e-4 the trunk was pre-trained with: the trunk is already
# good and the job here is to learn one new embedding without damaging it.
# --balance-elo matters more than usual now -- the conditioning has to be
# trained across the whole rating range, not just the crowded middle.
if $PY train_multigame.py --shard "$SHARD" --out "$OUT" \
        --init "$TRUNK" --elo-cond --elo-drop 0.1 \
        --max-games 5 --max-len-per-game 160 \
        --max-hours "$FT_H" --steps 100000000 \
        --lr 8e-5 --warmup 500 --batch 48 \
        --d-model 256 --layers 8 --heads 8 \
        --plies-per-game 8 --n-cand 32 --workers 24 \
        --eval-every 8000 --eval-batches 25 --balance-elo \
        --patience 8 --min-delta 0.0002 --amp --compile; then
    echo ELO_TRAIN_DONE
else
    echo ELO_TRAIN_FAILED
fi
kill "$KEEPER" 2>/dev/null

[ -s "$OUT/last.pt" ] && $PY runpod/s3io.py up "$OUT/last.pt" final/ctx5_pre_elo.pt
[ -s "$OUT/history.json" ] && $PY runpod/s3io.py up "$OUT/history.json" final/ctx5_pre_elo_history.json

# Behavioural check. A conditioning embedding can train to nothing while the
# loss curve looks healthy, because move accuracy is dominated by the trunk.
echo "=== $(date -u +%H:%M:%S) probe: does the requested rating change play? ==="
if [ -s "$OUT/last.pt" ]; then
    $PY elo_probe.py --ckpt "$OUT/last.pt" --shard "$SHARD" \
        --games 150 --bands 1000,1400,1800,2200 2>&1 | tail -20
fi

echo "=== $(date -u +%H:%M:%S) uploading ==="
cp -f "$LOG" "$D/elo_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/elo_run.log" final/elo_run.log || true
if [ -s "$OUT/last.pt" ]; then echo NOVOL_UPLOAD_OK; else echo NOVOL_UPLOAD_FAILED; fi
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
