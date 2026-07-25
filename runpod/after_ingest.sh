#!/usr/bin/env bash
# Phase 3 driver: wait out the month-by-month ingest, build the filtered
# cross-month index, then pre-train for five hours.
#
# 8-plane encoding (--no-rights). On Jan 2013 the 13-plane variant with castling
# rights and en passant was indistinguishable (full-legal top-1 0.370 vs 0.368)
# and finished with slightly *higher* val loss, because a causal model reading
# the whole game can infer castling rights from whether the king or rook has
# moved. Simpler encoding, 38% fewer input features, same accuracy.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code

while pgrep -f 'ingest.py --url' > /dev/null; do sleep 60; done
echo INGEST_FINISHED
du -sh /workspace/data/*

$PY combine.py --shards /workspace/data/2026-01 /workspace/data/2026-02 \
    /workspace/data/2026-03 /workspace/data/2026-04 /workspace/data/2026-05 \
    /workspace/data/2026-06 --out /workspace/data/combined --min-games 100
echo COMBINE_DONE

$PY train_successor.py --combined /workspace/data/combined \
    --out /workspace/ckpt/prod --max-hours 5 --steps 100000000 \
    --workers 30 --batch 128 --eval-every 5000 --eval-batches 30 --no-rights
echo TRAIN_DONE
