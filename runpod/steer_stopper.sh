#!/usr/bin/env bash
# Stop the pod when the steer run finishes, or at a hard deadline.
#
# This exists because an unattended pod that outlives its job is how a balance
# disappears: $0.74/hr is $17.76 a day, against a balance of $24.66. The
# training loop has its own --max-hours, but a crashed or wedged trainer would
# leave the GPU billing forever, so the deadline here is independent of it.
#
# Runs ON the pod, so it does not depend on the laptop staying awake.
# The API key is read from /root/.rpkey and shredded immediately.
set -u

VOL=/workspace
LOG=$VOL/steer_stopper.log
WORKLOG=$VOL/ctx10_steer.log
CKPT=$VOL/ckpt/ctx10_steer/last.pt
POD_ID=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
DEADLINE=$(( $(date +%s) + ${MAX_HOURS:-18} * 3600 ))

log() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }

APIKEY=$(cat /root/.rpkey 2>/dev/null)
shred -u /root/.rpkey 2>/dev/null || rm -f /root/.rpkey
[ -n "$APIKEY" ] || { log "no API key; cannot stop"; exit 1; }
[ -n "$POD_ID" ] || { log "could not read RUNPOD_POD_ID"; exit 1; }
log "armed for pod $POD_ID, deadline +${MAX_HOURS:-18}h"

while :; do
    grep -qE "STEER_DONE|STEER_FAILED" "$WORKLOG" 2>/dev/null && { log "work reported done"; break; }
    [ "$(date +%s)" -ge "$DEADLINE" ] && { log "hard deadline reached"; break; }
    sleep 60
done

# bracket one character so this does not match the stopper's own command line
for _ in $(seq 1 20); do
    pgrep -f "train_multigame[.]py" > /dev/null || break
    log "trainer still running, waiting"
    sleep 60
done
sync
[ -s "$CKPT" ] && log "checkpoint present: $(stat -c%s "$CKPT" 2>/dev/null) bytes" \
               || log "WARNING: no checkpoint at $CKPT"

log "stopping pod $POD_ID"
code=$(curl -s -o "$VOL/steer_stop.out" -w '%{http_code}' -X POST \
        -H "Authorization: Bearer $APIKEY" \
        "https://rest.runpod.io/v1/pods/$POD_ID/stop")
log "STOP returned $code"
