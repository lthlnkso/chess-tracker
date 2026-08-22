#!/usr/bin/env bash
# A/B for the win/draw/loss head on the small trunk.
#
# The control already exists and cost nothing extra: stage 1 of the steer-auto
# run was this identical recipe WITHOUT the head -- 83,700 steps, move_acc 0.501,
# elo_mae ~120. So this run changes exactly one thing, --w-result, and is
# compared at matched steps.
#
# Same seed, same shard, same lr, same everything else. Slightly longer wall
# clock because the extra head costs a little throughput, so that it comfortably
# passes the control's 83,700 steps rather than stopping short of them.
set -u
W=/workspace
CODE=$W/code3
PY=$W/venv/bin/python
LOG=$W/result_ab.log
POD=${RUNPOD_POD_ID:-}

term() {
    echo "=== $(date -u +%H:%M:%S) terminating pod ${POD:-unknown} ===" >> "$LOG"
    if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "$POD" ]; then
        for i in 1 2 3 4 5; do
            c=$(curl -s -o /dev/null -w '%{http_code}' -m 25 -X DELETE \
                -H "Authorization: Bearer $RUNPOD_API_KEY" \
                "https://rest.runpod.io/v1/pods/$POD")
            echo "  terminate attempt $i -> $c" >> "$LOG"
            case "$c" in 200|204) break;; esac
            sleep 20
        done
    else
        echo "  NO API KEY OR POD ID -- cannot self-terminate" >> "$LOG"
    fi
}
trap term EXIT

{
echo "=== $(date -u +%H:%M:%S) boot | pod=$POD ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1
cd "$CODE" || { echo "NO CODE AT $CODE"; exit 1; }
CPUS=$($PY cpuquota.py 2>/dev/null || echo 8)
echo "cpu quota: $CPUS"
echo "=== TREATMENT: small trunk + --w-result 0.3 ==="
$PY train_multigame.py --shard $W/data/mt/2026-01 --out $W/ckpt/ctx5_result_ab \
    --init $W/final/ctx5_pre.pt \
    --max-games 5 --max-len-per-game 160 \
    --d-model 256 --layers 8 --heads 8 --d-embed 128 \
    --plies-per-game 8 --n-cand 32 --batch 48 --lr 4e-5 --warmup 500 \
    --balance-gap --elo-steer --w-result 0.3 \
    --workers "$CPUS" --eval-every 5000 --eval-batches 25 \
    --patience 1000000 --amp --compile --max-hours "${HRS:-1.5}"
echo "exit=$?"
echo "=== $(date -u +%H:%M:%S) AB_DONE ==="
} >> "$LOG" 2>&1
