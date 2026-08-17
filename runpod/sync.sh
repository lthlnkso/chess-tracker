#!/usr/bin/env bash
# Push local code to the pod. Uses the `chesspod` alias in ~/.ssh/config, which
# points at the pod's direct SSH port -- the ssh.runpod.io proxy needs a PTY and
# cannot carry rsync.
set -euo pipefail

HOST="${1:-chesspod}"
# --no-o --no-g: the MooseFS-backed /workspace refuses chown, which -a implies.
# --include='*/' has to come first: without it the trailing --exclude='*' stops
# rsync descending into subdirectories at all, so runpod/ and scripts/ silently
# never arrive and only top-level files sync.
rsync -az --no-o --no-g --delete \
    --include='*/' \
    --include='*.py' --include='*.sh' --include='*.md' \
    --exclude='*' \
    --prune-empty-dirs \
    ./ "$HOST:/workspace/code/"
echo "synced -> $HOST:/workspace/code/"
