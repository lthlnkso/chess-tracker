#!/usr/bin/env bash
# One ingest worker: parse the months it is given into the shared volume.
#
# Deliberately a *GPU* pod despite doing no GPU work. RunPod CPU-only pods bill
# $1.99/hr for 32 vCPU; a secure RTX 3090 bills $0.50/hr for the same 32 vCPU.
# Ingest is pure CPU (zstd + SAN parsing), so the cheapest CPU throughput on the
# platform happens to come with a GPU attached.
#
#   bash ingest_worker.sh 2026-02 2026-03
set -x
PY=/workspace/venv/bin/python
cd /workspace/code

for M in "$@"; do
    if [ -f /workspace/data/mt/$M/manifest.json ]; then
        echo "SKIP $M (already complete)"
        continue
    fi
    # Write to a private staging dir, then move into place. A half-written shard
    # that carries a manifest would be indistinguishable from a good one.
    rm -rf /workspace/data/mt/.staging_$M
    $PY ingest.py \
        --url https://database.lichess.org/standard/lichess_db_standard_rated_$M.pgn.zst \
        --out /workspace/data/mt/.staging_$M --workers 30 --time-controls '60+0' \
        2>&1 | tr '\r' '\n' | tail -2
    if [ -f /workspace/data/mt/.staging_$M/manifest.json ]; then
        mkdir -p /workspace/data/mt
        mv /workspace/data/mt/.staging_$M /workspace/data/mt/$M
        echo "DONE $M"
    else
        echo "FAILED $M"
    fi
done
echo WORKER_DONE
