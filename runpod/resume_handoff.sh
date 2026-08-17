#!/usr/bin/env bash
# Finish the handoff to an already-created pod: wait for sshd, sync code, launch
# the work and the stopper.
#
# Split out from colour_run.sh because pod creation and handoff fail for
# different reasons and on different timescales. Creation either works or hits
# capacity; the handoff waits on a 16.3 GB image pull that has been observed to
# take well over 20 minutes, and a launcher that gives up at 20 leaves a running
# pod with nothing on it that knows to stop. This retries the endpoint lookup
# every cycle -- the ports move while the container is still coming up, so a
# cached ip:port from two minutes ago is not a reliable target.
#
#   ./runpod/resume_handoff.sh <pod-id> [minutes]
set -u
cd "$(dirname "$0")/.." || exit 1

KEY=$(grep RUNPOD_API_KEY .env | cut -d= -f2)
POD=${1:?usage: resume_handoff.sh <pod-id> [minutes]}
MINUTES=${2:-70}
DEADLINE=$(( $(date +%s) + MINUTES * 60 ))
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

endpoint() {
    curl -s -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"query { pod(input:{podId:\\\"$POD\\\"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }\"}" \
        | python3 -c "
import json,sys
try:
    rt = json.load(sys.stdin)['data']['pod']['runtime'] or {}
    for p in rt.get('ports') or []:
        if p['isIpPublic'] and p['privatePort'] == 22: print(p['ip'], p['publicPort'])
except Exception: pass"
}

H=""; P=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    EP=$(endpoint)
    H=$(echo "$EP" | awk '{print $1}'); P=$(echo "$EP" | awk '{print $2}')
    if [ -n "$P" ]; then
        # shellcheck disable=SC2086
        if ssh -n $SSHOPT -o ConnectTimeout=10 -p "$P" "root@$H" 'echo UP' 2>/dev/null | grep -q UP; then
            echo "$(date -u +%H:%M) sshd up at $H:$P"
            break
        fi
        echo "$(date -u +%H:%M) endpoint $H:$P but sshd not answering yet"
    else
        echo "$(date -u +%H:%M) no endpoint yet (container still starting)"
    fi
    H=""; P=""
    sleep 45
done
[ -n "$P" ] || { echo "GAVE UP after ${MINUTES}m; pod $POD still has no sshd"; exit 1; }

# shellcheck disable=SC2086
rsync -az --no-o --no-g --include='*/' --include='*.py' --include='*.sh' --include='*.md' \
    --exclude='*' --prune-empty-dirs \
    -e "ssh $SSHOPT -p $P" ./ "root@$H:/workspace/code/" || { echo "rsync failed"; exit 1; }
echo "code synced"

# shellcheck disable=SC2086
printf '%s' "$KEY" | ssh $SSHOPT -p "$P" "root@$H" 'cat > /root/.rpkey && chmod 600 /root/.rpkey'

# shellcheck disable=SC2086
ssh -n $SSHOPT -p "$P" "root@$H" \
    'cd /workspace/code && chmod +x runpod/colour_work.sh runpod/stopper.sh &&
     setsid nohup bash runpod/colour_work.sh > /workspace/colour_boot.log 2>&1 < /dev/null &
     sleep 1
     setsid nohup env MAX_HOURS=6 bash runpod/stopper.sh > /workspace/stopper_boot.log 2>&1 < /dev/null &
     sleep 3; echo LAUNCHED; pgrep -af "colour_work|stopper" | head'
echo "POD=$POD HOST=$H PORT=$P"
echo LAUNCH_OK
