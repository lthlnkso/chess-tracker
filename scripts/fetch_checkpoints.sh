#!/usr/bin/env bash
# Model weights are attached to the GitHub Release rather than committed, so a
# clone stays small.
set -euo pipefail
REPO="${REPO:-lthlnkso/chess-tracker}"
TAG="${TAG:-v1.0}"
mkdir -p ckpt
for f in pretrain_best.pt id_supcon.pt; do
    if [ -f "ckpt/$f" ]; then echo "have ckpt/$f"; continue; fi
    echo "fetching $f ..."
    curl -fL --progress-bar \
        "https://github.com/${REPO}/releases/download/${TAG}/${f}" -o "ckpt/$f"
done
ls -la ckpt/
