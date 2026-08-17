#!/usr/bin/env bash
# Stop -- not terminate -- the pod once the run is done and its artifacts are on
# the network volume.
#
# reaper.sh DELETEs, which is right when a project is finished for the day. This
# is the cheaper default for a project still in flight: a stopped pod keeps its
# container disk and its pulled image, so the next start is seconds instead of
# the 15-20 minutes it takes a cold host to pull the 16.3 GB pytorch image. The
# tradeoff is that a stopped pod still bills for disk (cents per hour), so it
# needs the same discipline as a running one -- just an order of magnitude
# cheaper to forget about.
#
# Runs ON the pod so it does not depend on the laptop being awake. Verify, then
# stop: it will not stop while the expected outputs are missing, because a pod at
# ~$0.50/hr is far cheaper than re-running an hour of embedding.
#
# The API key is read from /root/.rpkey and shredded immediately, so it lives in
# this process's memory and nowhere on disk. Deliberately no `set -x`.
set -u

VOL=/workspace
LOG=$VOL/stopper.log
WORKLOG=$VOL/colour.log
POD_ID=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
DEADLINE=$(( $(date +%s) + ${MAX_HOURS:-6} * 3600 ))

log() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }

APIKEY=$(cat /root/.rpkey 2>/dev/null)
shred -u /root/.rpkey 2>/dev/null || rm -f /root/.rpkey
[ -n "$APIKEY" ] || { log "no API key; cannot stop"; exit 1; }
[ -n "$POD_ID" ] || { log "could not read RUNPOD_POD_ID"; exit 1; }
log "armed for pod $POD_ID, deadline +${MAX_HOURS:-6}h"

timed_out=0
while :; do
    grep -q COLOUR_ALL_DONE "$WORKLOG" 2>/dev/null && { log "work reported ALL_DONE"; break; }
    [ "$(date +%s)" -ge "$DEADLINE" ] && { timed_out=1; log "hard deadline reached"; break; }
    sleep 60
done

for _ in $(seq 1 30); do
    pgrep -f "gallery_ctx[.]py|build_gallery[.]py" > /dev/null || break
    log "work still running, waiting"
    sleep 60
done
sync

have=0
[ -s "$VOL/final/ctx5_colour.json" ] && { have=$((have+1)); log "have ctx5_colour.json"; }
[ -s "$VOL/final/gallery_deploy.npz" ] && { have=$((have+1)); log "have gallery_deploy.npz"; }
cp -f "$WORKLOG" "$VOL/final/colour.log" 2>/dev/null

if [ "$have" -eq 0 ] && [ "$timed_out" -eq 0 ]; then
    log "REFUSING TO STOP: work finished but produced nothing"
    exit 1
fi

log "stopping pod $POD_ID ($have/2 artifacts)"
code=$(curl -s -o "$VOL/stopper_stop.out" -w '%{http_code}' -X POST \
        -H "Authorization: Bearer $APIKEY" \
        "https://rest.runpod.io/v1/pods/$POD_ID/stop")
log "STOP returned $code"
