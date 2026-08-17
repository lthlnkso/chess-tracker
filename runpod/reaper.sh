#!/usr/bin/env bash
# Self-terminate the pod once the run is finished AND its artifacts are verified
# on the network volume.
#
# Runs ON the pod so it does not depend on the laptop being awake. The previous
# arrangement kept the terminate in a local background task, which dies with the
# session and has already cost this project ~$2.15 of idle billing once.
#
# Ordering is the whole point: verify, then destroy. It will refuse to terminate
# while the model is missing, because a pod costing $0.99/hr is far cheaper than
# a lost training run. The only case where it terminates empty-handed is the hard
# deadline, where there is by definition nothing to preserve.
#
# The API key is read from /root/.rpkey and that file is shredded immediately, so
# the credential lives in this process's memory and nowhere on disk. Deliberately
# no `set -x`: it would echo the key into the log.
set -u

VOL=/workspace
K=$VOL/ckpt
FINAL=$VOL/final
LOG=$VOL/reaper.log
POD_ID=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
DEADLINE=$(( $(date +%s) + ${MAX_HOURS:-11} * 3600 ))

log() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }

APIKEY=$(cat /root/.rpkey 2>/dev/null)
shred -u /root/.rpkey 2>/dev/null || rm -f /root/.rpkey
[ -n "$APIKEY" ] || { log "no API key; reaper cannot terminate"; exit 1; }
[ -n "$POD_ID" ] || { log "could not read RUNPOD_POD_ID"; exit 1; }
log "armed for pod $POD_ID, deadline $(date -u -d @$DEADLINE +%H:%M 2>/dev/null || echo "+${MAX_HOURS:-11}h")"

timed_out=0
while :; do
    grep -q FT2_ALL_DONE "$VOL/ft2.log" 2>/dev/null && { log "run reported FT2_ALL_DONE"; break; }
    grep -qE "FT2_FAILED|FT2_EVAL_FAILED" "$VOL/ft2.log" 2>/dev/null && { log "run reported FAILURE"; break; }
    [ "$(date +%s)" -ge "$DEADLINE" ] && { timed_out=1; log "hard deadline reached"; break; }
    sleep 120
done

# Belt and braces: markers are echoed after the python exits, but a partially
# flushed checkpoint would be worse than a few extra minutes of billing.
for _ in $(seq 1 30); do
    pgrep -f "finetune_ctx[.]py|identify_eval_ctx[.]py" > /dev/null || break
    log "work still running, waiting"
    sleep 60
done

mkdir -p "$FINAL"
save() {  # src dst
    if [ -s "$1" ]; then
        cp -f "$1" "$FINAL/$2" && log "saved $2 ($(stat -c%s "$1") bytes)"
    else
        log "MISSING $1"
    fi
}
save "$K/ctx5_ft2/last.pt"        ctx5_ft2.pt
save "$K/ctx5_ft2/history.json"   ctx5_ft2_history.json
save "$K/ctx5_eval2.json"         ctx5_eval2.json
save "$K/ctx5_pre/last.pt"        ctx5_pre.pt
save "$VOL/ft2.log"               ft2.log
sync
log "final contents: $(ls -la "$FINAL" | tail -n +2 | wc -l) files"

if [ ! -s "$FINAL/ctx5_ft2.pt" ] && [ "$timed_out" -eq 0 ]; then
    log "REFUSING TO TERMINATE: run finished but no model landed in $FINAL"
    exit 1
fi

log "terminating pod $POD_ID"
code=$(curl -s -o "$VOL/reaper_delete.out" -w '%{http_code}' -X DELETE \
        -H "Authorization: Bearer $APIKEY" \
        "https://rest.runpod.io/v1/pods/$POD_ID")
log "DELETE returned $code"
