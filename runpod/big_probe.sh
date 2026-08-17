#!/usr/bin/env bash
# How long would the 10x model actually take to pre-train?
#
# Config: d_model 512 / 24 layers / 8 heads = 79.8M params (10.1x the shipped
# 7.9M trunk), and 10 game slots instead of 5.
#
# "2x context" is 10 GAMES, not 320 plies. Measured mean own-moves per bullet
# game is 33.2, so a game is ~66 plies against the existing 160-ply cap --
# doubling the cap would buy padding. Doubling the slot count doubles real
# content, and it is the axis the product wants: the demo currently discards
# everything past your five most recent games.
#
# This is a THROUGHPUT PROBE, not a training run. The arithmetic before it says
# a full 816k-step pre-train at this size costs ~351 GPU-hours (~$95) as an
# optimistic floor, against ctx5_pre's measured 17.4. That is not fundable at
# this balance, so the point of spending ~$0.30 here is to replace the floor with
# a measurement and let the real decision be made on a real number.
#
# The batch ladder exists because OOM three minutes into a long run has cost us
# real money twice. Each rung is a 40-step smoke test; the first that survives
# becomes the batch for the timed probe.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/${TAG:-big_probe}.log
SHARD=$D/shard
OUT=$D/big
DM=${DM:-512}
LAYERS=${LAYERS:-24}
HEADS=${HEADS:-8}
SLOTS=${SLOTS:-10}
MLPG=${MLPG:-160}
PROBE_H=${PROBE_H:-1.0}
LADDER=${LADDER:-"24 16 12 8 6 4 2"}
TAG=${TAG:-big_probe}   # artifact prefix; set it for control arms

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd /root/code || exit 1

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) | $(nproc) vCPU ==="
echo "config: d_model $DM | layers $LAYERS | heads $HEADS | slots $SLOTS | ${MLPG} plies"

$PY runpod/s3io.py down data/mt/2026-01 "$SHARD" || { echo NOVOL_UPLOAD_FAILED; echo NOVOL_ALL_DONE; exit 1; }

BATCH=0
for B in $LADDER; do
    echo "=== $(date -u +%H:%M:%S) smoke test batch $B ==="
    # No --compile here: a 24-layer graph takes minutes to compile and we only
    # need to know whether the activations fit.
    if $PY train_multigame.py --shard "$SHARD" --out "$D/smoke" \
            --max-games "$SLOTS" --max-len-per-game "$MLPG" \
            --d-model "$DM" --layers "$LAYERS" --heads "$HEADS" \
            --batch "$B" --lr 1.5e-4 --warmup 50 \
            --plies-per-game 8 --n-cand 32 --workers 12 \
            --steps 40 --eval-every 1000000 --amp --balance-elo \
            --max-hours 0.15 >"$D/smoke_$B.log" 2>&1; then
        BATCH=$B
        echo "batch $B fits"
        break
    fi
    echo "batch $B failed:"; tail -4 "$D/smoke_$B.log" | sed 's/^/    /'
    rm -rf "$D/smoke"
done

if [ "$BATCH" -eq 0 ]; then
    echo "BIG_PROBE_NO_BATCH_FITS"
    cp -f "$LOG" "$D/${TAG}_run.log"; $PY runpod/s3io.py up "$D/${TAG}_run.log" final/${TAG}_run.log || true
    echo NOVOL_ALL_DONE; exit 1
fi

echo "=== $(date -u +%H:%M:%S) timed probe: batch $BATCH, ${PROBE_H}h, with --compile ==="
$PY train_multigame.py --shard "$SHARD" --out "$OUT" \
        --max-games "$SLOTS" --max-len-per-game "$MLPG" \
        --d-model "$DM" --layers "$LAYERS" --heads "$HEADS" \
        --batch "$BATCH" --lr 1.5e-4 --warmup 1000 \
        --plies-per-game 8 --n-cand 32 --workers 24 \
        --steps 100000000 --eval-every 4000 --eval-batches 15 \
        --balance-elo --patience 1000000 --amp --compile \
        --max-hours "$PROBE_H" && echo BIG_PROBE_TRAIN_DONE || echo BIG_PROBE_TRAIN_FAILED

# Extrapolate on POSITIONS, not steps. The batch here is far smaller than
# ctx5_pre's 48, so steps/s is not comparable across the two runs -- what
# transfers is how many position-forwards the GPU retires per second.
echo "=== $(date -u +%H:%M:%S) extrapolation ==="
$PY - "$OUT" "$BATCH" "$SLOTS" "$MLPG" <<'PYEOF'
import json, os, sys
out, batch, slots, mlpg = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
h = os.path.join(out, "history.json")
if not os.path.isfile(h):
    print("no history.json; read steps/s from the log above"); raise SystemExit
c = json.load(open(h)).get("curve") or []
if len(c) < 4:
    print(f"only {len(c)} curve points; read steps/s from the log"); raise SystemExit
# Measure over a LATE window. torch.compile spends minutes building a 24-layer
# graph before step 1, and averaging that into a one-hour probe would understate
# steady-state throughput by a lot.
a, b = c[len(c) // 2], c[-1]
dstep, dmin = b["step"] - a["step"], b["minutes"] - a["minutes"]
if dstep <= 0 or dmin <= 0:
    print("degenerate window; read steps/s from the log"); raise SystemExit
sps = dstep / (dmin * 60)
pos = sps * batch * slots * mlpg
BASE_POS = 816_000 * 48 * 5 * 160      # ctx5_pre's total position-forwards
print(f"steady state (steps {a['step']}->{b['step']}): {sps:.2f} steps/s "
      f"at batch {batch}")
print(f"  = {pos/1e6:.2f}M position-forwards/s")
print(f"ctx5_pre did {BASE_POS/1e9:.1f}G position-forwards in 17.4 GPU-h "
      f"({BASE_POS/17.4/3600/1e6:.2f}M/s)")
hrs = BASE_POS / pos / 3600
print(f"\nequal-work pre-train at this size: {hrs:.0f} GPU-h "
      f"= ${hrs*0.27:.0f} at $0.27/hr  ({hrs/17.4:.1f}x ctx5_pre)")
print("NOTE: equal WORK is not equal QUALITY. A 10x model generally needs more")
print("tokens than the small one, not the same, so treat this as a floor.")
PYEOF

for f in last.pt history.json; do
    [ -s "$OUT/$f" ] && $PY runpod/s3io.py up "$OUT/$f" "final/${TAG}_$f"
done
cp -f "$LOG" "$D/${TAG}_run.log" 2>/dev/null
$PY runpod/s3io.py up "$D/${TAG}_run.log" final/${TAG}_run.log && echo NOVOL_UPLOAD_OK || echo NOVOL_UPLOAD_FAILED
echo NOVOL_ALL_DONE
} >> "$LOG" 2>&1
