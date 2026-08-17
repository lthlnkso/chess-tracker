#!/usr/bin/env bash
# Keep podA earning its $0.50/hr overnight.
#
# The failure that actually costs money here is silent: the fine-tune dies at
# 04:00, nobody is awake, and the pod bills eight hours for an idle GPU. This
# checks every 15 minutes for three states -- work running, work finished, or
# neither -- and only acts on the third.
#
# Two consecutive misses, not one, because the hand-off from pre-training has a
# legitimate gap: the waiter sleeps 120s, polls for the trainer to exit, sleeps
# 30s, then spends minutes memory-mapping a 38M-row index before the process
# name shows up as busy.
set -u
K=/workspace/ckpt
PY=/workspace/venv/bin/python
MISSES=0

while true; do
    sleep 900
    if [ -f $K/ws1_id_ms/eval.json ]; then
        echo "$(date -u +%H:%M) finished cleanly, watchdog exiting"
        exit 0
    fi
    if pgrep -f "train_multitask[.]py" > /dev/null \
    || pgrep -f "finetune_mt[.]py" > /dev/null \
    || pgrep -f "ws1_ft_[m]s.sh" > /dev/null; then
        MISSES=0
        continue
    fi
    MISSES=$((MISSES + 1))
    echo "$(date -u +%H:%M) nothing running (miss $MISSES)"
    [ $MISSES -lt 2 ] && continue

    if [ ! -f $K/ws1_pre/last.pt ]; then
        echo "no pre-trained trunk to fall back on; cannot restart"
        exit 1
    fi
    echo "$(date -u +%H:%M) RESTARTING fine-tune from $K/ws1_pre/last.pt"
    cd /workspace/code
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup $PY finetune_mt.py --shard /workspace/data/mt/2026-01 \
        --pretrained $K/ws1_pre/last.pt --out $K/ws1_id_ms \
        --max-hours 6 --p 32 --k 4 --workers 24 --min-games 8 \
        --eval-players 20000 --balance-elo --loss ms \
        --probe-every-hours 1 --probe-players 2000 \
        > /workspace/ws1_ft_ms_restart.log 2>&1 < /dev/null &
    MISSES=0
done
