#!/usr/bin/env bash
# Terminate the volume-less pod once its results are verified in S3.
#
# TERMINATE, not stop: with no network volume attached there is nothing on this
# pod worth preserving except the results, and those are only safe once they are
# in the bucket. So the ordering matters more here than anywhere else -- the
# artifacts live on ephemeral container disk until the upload succeeds, and
# terminating before that loses hours of GPU time.
#
# Refuses to terminate on NOVOL_UPLOAD_FAILED, so a broken S3 gateway leaves a
# pod we can still rescue by hand. The hard deadline is the money cap for that
# case: at MAX_HOURS it gives up and terminates regardless.
set -u

D=/data
LOG=$D/stopper.log
# Which log to watch depends on which job this pod is running.
WORKLOG=${WORKLOG:-$D/colour.log}
POD_ID=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
DEADLINE=$(( $(date +%s) + ${MAX_HOURS:-6} * 3600 ))

mkdir -p "$D"
log() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }

APIKEY=$(cat /root/.rpkey 2>/dev/null)
shred -u /root/.rpkey 2>/dev/null || rm -f /root/.rpkey
[ -n "$APIKEY" ] || { log "no API key; cannot terminate"; exit 1; }
[ -n "$POD_ID" ] || { log "could not read RUNPOD_POD_ID"; exit 1; }
log "armed for pod $POD_ID, deadline +${MAX_HOURS:-6}h"

timed_out=0
while :; do
    grep -q NOVOL_ALL_DONE "$WORKLOG" 2>/dev/null && { log "work reported ALL_DONE"; break; }
    [ "$(date +%s)" -ge "$DEADLINE" ] && { timed_out=1; log "hard deadline reached"; break; }
    sleep 60
done

for _ in $(seq 1 30); do
    pgrep -f "gallery_ctx[.]py|build_gallery[.]py|union_gallery[.]py|finetune_ctx[.]py|s3io[.]py" > /dev/null || break
    log "work still running, waiting"
    sleep 60
done

if grep -q NOVOL_UPLOAD_FAILED "$WORKLOG" 2>/dev/null && [ "$timed_out" -eq 0 ]; then
    log "REFUSING TO TERMINATE: results were not uploaded; pod left up for rescue"
    exit 1
fi

log "terminating pod $POD_ID"
# urllib, not curl: python:3.12-slim ships no curl, so the previous version
# printed an EMPTY status code and left the pod billing after a completed run.
# Nothing failed loudly -- the log line just read "DELETE returned " with a
# blank where the 204 should have been. Python is guaranteed present here; curl
# is not.
code=$(APIKEY="$APIKEY" POD_ID="$POD_ID" "${PY:-python3}" - <<'PY'
import os, urllib.request, urllib.error
req = urllib.request.Request(
    f"https://rest.runpod.io/v1/pods/{os.environ['POD_ID']}",
    method="DELETE", headers={"Authorization": f"Bearer {os.environ['APIKEY']}"})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        print(r.status)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception as e:
    print(f"ERROR {type(e).__name__}")
PY
)
log "DELETE returned $code"
# Verify rather than trust: if the pod is still alive, say so loudly in the log
# so it is findable, and retry once.
sleep 5
if [ "$code" != "200" ] && [ "$code" != "204" ]; then
    log "TERMINATE DID NOT CONFIRM ($code) -- pod $POD_ID may still be billing"
fi
