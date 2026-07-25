#!/usr/bin/env bash
# Drive one already-running benchmark pod: push code, build env, ingest a month
# onto its own container disk, benchmark, pull the JSON back.
#   ./bench_one.sh <host> <port> "<GPU label>" <price_per_hr> <outdir>
set -euo pipefail

HOST=$1; PORT=$2; LABEL=$3; PRICE=$4; OUTDIR=$5
TAG=$(echo "$LABEL" | awk '{print $NF}')
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
R="ssh $SSH_OPTS -p $PORT root@$HOST"
URL=https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst

echo "[$TAG] pushing code"
$R "mkdir -p /root/code /root/data"
rsync -az --no-o --no-g -e "ssh $SSH_OPTS -p $PORT" \
    --include='*.py' --exclude='*' ./ "root@$HOST:/root/code/"

echo "[$TAG] building venv"
$R "python3 -m venv --system-site-packages /root/venv && \
    /root/venv/bin/pip install -q python-chess zstandard numpy" >/dev/null

echo "[$TAG] ingesting Jan 2013"
$R "cd /root/code && /root/venv/bin/python ingest.py --url $URL \
    --out /root/data/2013-01 --workers 12 2>&1 | tr '\r' '\n' | tail -1"

echo "[$TAG] benchmarking"
$R "cd /root/code && /root/venv/bin/python bench_gpu.py --shard /root/data/2013-01 \
    --gpu '$LABEL' --price $PRICE --steps 150 --e2e-steps 200 --out /root/bench.json" \
    | tail -25

mkdir -p "$OUTDIR"
rsync -az -e "ssh $SSH_OPTS -p $PORT" "root@$HOST:/root/bench.json" "$OUTDIR/$TAG.json"
echo "[$TAG] DONE -> $OUTDIR/$TAG.json"
