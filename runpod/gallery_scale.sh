#!/usr/bin/env bash
# The product metric: top-10 recall from 1-5 games against the whole player base.
#
# Everything measured so far tops out at a 50,000-player gallery, and the
# challenge screen ranks against every eligible lichess player. sweep_gallery.py
# derives smaller galleries from a large one exactly (hypergeometric over which
# distractors survive the subsample), but it cannot go the other way, so the
# gallery actually embedded has to be as large as the corpus allows.
#
# 2026-01 holds 522,735 players. Distractors are deliberately not restricted to
# held-out players -- only queries must be held out -- so the reachable gallery
# is the whole month, not the 20% test split. That is also what deployment looks
# like: the gallery is everyone.
#
# The remaining gap to a 6-month gallery (~1.29M players ever seen in 60+0) is
# 2.6x, which is a far shorter extrapolation than the 40x that quoting a 2M
# number off the 50k data would have required.
set -x
PY=/workspace/venv/bin/python
cd /workspace/code

# --test-pids-file is mandatory here, not optional: finetune_mt.py does not store
# test_pids in the checkpoint, and without the split the script (correctly)
# refuses to run rather than quietly scoring queries the model trained on.
# mt_test_pids.npy is the seed-0 / min_games 8 / test_frac 0.2 split, which is
# exactly what this fine-tune held out -- confirmed by its pooled eval landing on
# the same 10,675 matched players as the published run.
$PY sweep_gallery.py --ckpt /workspace/ckpt/ws1_id_ms/last.pt \
    --shard /workspace/data/mt/2026-01 \
    --out /workspace/ckpt/ws1_gallery_scale.json \
    --test-pids-file /workspace/ckpt/mt_test_pids.npy \
    --gallery-players 500000 --query-players 5000 \
    --pools 1,2,3,4,5 \
    --sizes 1000,10000,50000,100000,200000,300000,400000,500000 \
    --workers 22
echo GALLERY_SCALE_DONE
