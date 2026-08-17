#!/usr/bin/env bash
# Run the ArcFace arm once the three pair arms have vacated the GPU.
#
# It has to run alone: the full 128 x 226k logit matrix plus its arccos
# intermediates OOM'd a 24 GB card already hosting three arms (partial-FC now
# caps that, but the card was genuinely full). Running alone also means ~4x the
# CPU workers, so matching the others on WALL CLOCK would silently hand this arm
# four times the training. It is matched on STEPS instead -- read off supcon's
# completed run -- and --max-hours 0 hands the LR schedule to the step count.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code

until [ -f /workspace/ckpt/ws2_supcon/eval.json ] \
   && [ -f /workspace/ckpt/ws2_ms/eval.json ] \
   && [ -f /workspace/ckpt/ws2_circle/eval.json ]; do
    sleep 120
done
# Let the evaluation processes fully release their CUDA contexts.
while pgrep -f "finetune_mt[.]py" > /dev/null; do sleep 60; done
sleep 30

STEPS=$($PY -c "import json;print(json.load(open('/workspace/ckpt/ws2_supcon/eval.json'))['steps'])")
echo "matching supcon at $STEPS steps"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY finetune_mt.py --shard /workspace/data/mt/2026-01 \
    --pretrained /workspace/ckpt/mt_pre/last.pt \
    --out /workspace/ckpt/ws2_arcface --loss arcface \
    --max-hours 0 --steps "$STEPS" --p 32 --k 4 --workers 22 --min-games 8 \
    --eval-players 20000 --balance-elo \
    --probe-every-hours 0.5 --probe-players 2000 --proxy-warmup 1500 \
    > /workspace/ckpt/ws2_arcface.log 2>&1

[ -f /workspace/ckpt/ws2_arcface/eval.json ] && echo WS2_ARCFACE_OK || echo WS2_ARCFACE_FAILED
