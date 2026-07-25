#!/usr/bin/env bash
# Overnight guard. Terminates the pod once the pipeline is finished OR has been
# idle long enough to mean it died.
#
# The point is the failure case: if phase_id.sh crashes at 2am, an unattended pod
# bills $0.52/hr until someone notices. Success is easy; this is insurance.
#
#   bash watchdog.sh <pod_id> <idle_minutes>
set -u
POD_ID=$1
IDLE_LIMIT=${2:-25}
KEYFILE=/workspace/.rp_key
LOG=/workspace/watchdog.log

idle=0
while true; do
    sleep 60

    if grep -q 'IDENTIFY_DONE' /workspace/phase_id.log 2>/dev/null; then
        echo "$(date -u +%H:%M) pipeline finished cleanly -> terminating" >> $LOG
        break
    fi

    # Any real GPU work in flight?
    if ps -eo args --no-headers | grep -qE '[t]rain_successor\.py|[f]inetune_id\.py|[i]dentify_eval\.py'; then
        idle=0
    else
        idle=$((idle + 1))
        echo "$(date -u +%H:%M) no GPU job running (${idle}/${IDLE_LIMIT} min)" >> $LOG
    fi

    if [ $idle -ge $IDLE_LIMIT ]; then
        echo "$(date -u +%H:%M) idle ${IDLE_LIMIT}m -- pipeline died -> terminating" >> $LOG
        break
    fi
done

# Data lives on the network volume, so terminating the pod loses nothing.
curl -s -X DELETE -H "Authorization: Bearer $(cat $KEYFILE)" \
    "https://rest.runpod.io/v1/pods/${POD_ID}" >> $LOG 2>&1
echo "$(date -u +%H:%M) terminate request sent" >> $LOG
