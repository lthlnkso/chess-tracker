#!/usr/bin/env bash
# Embedding-width sweep on a second host, then self-terminate.
#
#   bash sweep_embed.sh <pod_id> <minutes_per_dim> <dims...>
#
# Every arm starts from the SAME pre-trained trunk and differs only in d_embed,
# so the comparison is clean. Data and the trunk are copied to local container
# disk first: the shared network volume is already feeding the main training
# pod's 30 dataloader workers, and two pods hammering MooseFS would slow both
# and confound the timing.
set -x
POD_ID=$1; MINUTES=$2; shift 2
DIMS="$@"
PY=/workspace/venv/bin/python
LOCAL=/root/sweepdata
KEYFILE=/workspace/.rp_key

mkdir -p $LOCAL
if [ ! -f $LOCAL/2026-01/manifest.json ]; then
    mkdir -p $LOCAL/2026-01
    cp /workspace/data/mt/2026-01/* $LOCAL/2026-01/
fi
cp /workspace/ckpt/mt_pre/last.pt $LOCAL/trunk.pt
ls -la $LOCAL/2026-01/ $LOCAL/trunk.pt
echo DATA_READY

for D in $DIMS; do
    OUT=/workspace/ckpt/sweep_d$D
    if [ -f $OUT/eval.json ]; then echo "SKIP d$D"; continue; fi
    HOURS=$($PY -c "print($MINUTES/60)")
    $PY /workspace/code/finetune_mt.py --shard $LOCAL/2026-01 \
        --pretrained $LOCAL/trunk.pt --out $OUT \
        --d-embed $D --max-hours $HOURS --p 32 --k 4 --workers 28 \
        --min-games 8 --eval-players 5000 --eval-batch 192 --balance-elo \
        --collapse-after 2000
    if [ -f $OUT/eval.json ]; then echo "SWEEP_DONE d$D"; else echo "SWEEP_FAILED d$D"; fi
done

$PY /workspace/code/sweep_report.py --glob '/workspace/ckpt/sweep_d*/eval.json' \
    --out /workspace/ckpt/sweep_summary.json
echo SWEEP_ALL_DONE

# Release the host. Results are on the network volume, which outlives the pod.
curl -s -X DELETE -H "Authorization: Bearer $(cat $KEYFILE)" \
    "https://rest.runpod.io/v1/pods/${POD_ID}"
echo TERMINATE_SENT
