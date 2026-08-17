#!/usr/bin/env bash
# Backstop for the ONE window a pod cannot cover itself: between creation and
# stopper.sh starting on it.
#
# Once stopper.sh is running the pod is self-sufficient -- it stops on its own
# when the work finishes and again on a hard deadline if the work hangs. But if
# the launcher dies first (laptop sleeps mid-rsync, ssh never comes up), nothing
# on the pod knows it should ever stop, and a $0.50/hr pod against a $25 balance
# runs for two days. This project has already lost ~$2.30 to exactly that shape
# of orphan, so the backstop is cheap insurance.
#
# It only ever STOPS, never terminates: a stop preserves the container disk and
# whatever the run had already written to the network volume.
#
#   ./runpod/local_guard.sh <pod-id> <launcher-log>
set -u
cd "$(dirname "$0")/.." || exit 1

KEY=$(grep RUNPOD_API_KEY .env | cut -d= -f2)
POD=${1:?usage: local_guard.sh <pod-id> [launcher-log]}
LLOG=${2:-}
MAX_HOURS=${MAX_HOURS:-5}
LAUNCH_GRACE=${LAUNCH_GRACE:-45}          # minutes to reach LAUNCH_OK
START=$(date +%s)
DEADLINE=$(( START + MAX_HOURS * 3600 ))
GRACE=$(( START + LAUNCH_GRACE * 60 ))

stop_pod() {
    echo "$(date -u +%H:%M) GUARD STOPPING $POD: $1"
    curl -s -X POST -H "Authorization: Bearer $KEY" \
        "https://rest.runpod.io/v1/pods/$POD/stop" -w "\nHTTP %{http_code}\n"
    exit 0
}

echo "$(date -u +%H:%M) guarding $POD (grace ${LAUNCH_GRACE}m, cap ${MAX_HOURS}h)"
while :; do
    S=$(curl -s -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$POD" \
        | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('desiredStatus',''))
except Exception: print('')")
    case "$S" in
        EXITED)  echo "$(date -u +%H:%M) pod EXITED -- stopper did its job"; exit 0 ;;
        "")      echo "$(date -u +%H:%M) pod not found -- gone"; exit 0 ;;
    esac

    now=$(date +%s)
    launched=0
    [ -n "$LLOG" ] && grep -q LAUNCH_OK "$LLOG" 2>/dev/null && launched=1

    # Past the grace window with no successful handoff, the pod is an orphan:
    # nothing on it will ever stop it.
    [ "$launched" -eq 0 ] && [ "$now" -ge "$GRACE" ] && \
        stop_pod "no LAUNCH_OK after ${LAUNCH_GRACE}m"
    [ "$now" -ge "$DEADLINE" ] && stop_pod "hard cap ${MAX_HOURS}h"

    sleep 180
done
