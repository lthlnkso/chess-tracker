#!/usr/bin/env bash
# Overnight: arm C end-to-end on one real month.
#
#   time as INPUT + time-bucket head + Elo head, Elo-balanced sampling,
#   then SupCon identification fine-tune and evaluation.
#
# One month, not six: 2026-01 alone is 24.2M games / 522k players (~93
# game-sides each), and the six-month run consumed under 1% of its available
# supervision. Re-ingesting all six would spend 2.5h of the night to relieve a
# constraint that was never binding.
#
# The existing shards cannot be reused -- they were ingested by a build that
# stripped {...} comments, which is where [%clk] lives.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code
M=2026-01

if [ ! -f /workspace/data/mt/$M/manifest.json ]; then
    $PY ingest.py \
        --url https://database.lichess.org/standard/lichess_db_standard_rated_$M.pgn.zst \
        --out /workspace/data/mt/$M --workers 30 --time-controls '60+0' \
        2>&1 | tr '\r' '\n' | tail -2
fi
echo INGEST_DONE
$PY -c "import json;m=json.load(open('/workspace/data/mt/$M/manifest.json'));print(m['games'],'games',m['games_with_clocks'],'with clocks')"
df -h /workspace | tail -1

$PY train_multitask.py --shard /workspace/data/mt/$M \
    --out /workspace/ckpt/mt_pre --max-hours 3 --steps 100000000 \
    --lr 1.5e-4 --warmup 1000 --batch 128 --d-model 256 --layers 8 --heads 8 \
    --plies-per-game 12 --n-cand 16 --workers 30 \
    --eval-every 4000 --eval-batches 30 --balance-elo
if [ ! -f /workspace/ckpt/mt_pre/last.pt ]; then
    echo "PRETRAIN_FAILED: no checkpoint written"; exit 1
fi
echo PRETRAIN_DONE

$PY finetune_mt.py --shard /workspace/data/mt/$M \
    --pretrained /workspace/ckpt/mt_pre/last.pt --out /workspace/ckpt/mt_id \
    --max-hours 4 --p 32 --k 4 --workers 30 --min-games 8 \
    --eval-players 20000 --balance-elo
if [ -f /workspace/ckpt/mt_id/eval.json ]; then
    echo ALL_DONE
else
    echo "FINETUNE_FAILED: no eval.json"; exit 1
fi
