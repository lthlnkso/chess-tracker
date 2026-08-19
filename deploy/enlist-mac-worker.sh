#!/usr/bin/env bash
# Enlist this Mac as an identify worker when demand is high. The production box
# stands alone without it -- one local identify worker covers normal traffic --
# so this is surge capacity, not a dependency.
#
#   ./deploy/enlist-mac-worker.sh 3      # start 3 identify workers
#   ./deploy/enlist-mac-worker.sh 0      # stand down
#
# Identify only, deliberately. A remote MOVE worker measured WORSE than none:
# move p50 167 -> 195 ms, p95 478 -> 642 ms, because a Cloudflare round trip
# each way costs more than the 30 ms of compute it offloads. Identify runs 190 ms
# behind a spinner nobody is watching, which is the work worth shipping out.
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-3}"
API="${API:-https://chess.lthlnkso.com}"
LOGDIR="${TMPDIR:-/tmp}/chess-workers"; mkdir -p "$LOGDIR"

pkill -f "[w]orker.py --api" 2>/dev/null || true
sleep 2
if [ "$N" -eq 0 ]; then echo "stood down; 0 workers"; exit 0; fi

set -a; . ~/.chess-worker.env; set +a
for i in $(seq 1 "$N"); do
  DEVICE=cpu OMP_NUM_THREADS=2 nohup .venv/bin/python worker.py \
    --api "$API" --kinds identify --batch 3 \
    --idle-sleep 0.05 --max-idle-sleep 0.25 \
    > "$LOGDIR/identify$i.log" 2>&1 &
done
# Loading the gallery takes ~20 s; report only once they are actually claiming.
for _ in $(seq 1 40); do
  up=$(grep -l "worker up" "$LOGDIR"/identify*.log 2>/dev/null | wc -l | tr -d ' ')
  [ "$up" -ge "$N" ] && break
  sleep 2
done
echo "$up/$N identify workers up against $API   (logs: $LOGDIR)"
