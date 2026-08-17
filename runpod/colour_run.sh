#!/usr/bin/env bash
# Launch the one remaining pod session: colour-split measurement + deployment
# gallery, then STOP the pod.
#
# Cost shape of a short job here is boot-dominated -- 15-20 minutes of pulling a
# 16.3 GB image before sshd exists, against ~60-90 minutes of actual work. Three
# things follow, and this script does all three:
#   * ask for many GPU types at once, so a full datacenter costs a retry rather
#     than an hour of polling (the /v1/pods API takes gpuTypeIds as an ARRAY)
#   * do every piece of work that needs the big shard in this single boot
#   * stop rather than terminate at the end, so a follow-up starts in seconds
#
#   ./runpod/colour_run.sh
set -u
cd "$(dirname "$0")/.." || exit 1

KEY=$(grep RUNPOD_API_KEY .env | cut -d= -f2)
VOLUME=${VOLUME:-shusq6ritt}
DC=${DC:-EU-CZ-1}
MINUTES=${MINUTES:-90}
GAP=${GAP:-120}
DEADLINE=$(( $(date +%s) + MINUTES * 60 ))

# The work is CPU-bound on board encoding as much as it is GPU-bound, so any of
# these is fine and availability matters far more than the card.
GPUS='["NVIDIA RTX A5000","NVIDIA RTX A4500","NVIDIA RTX A4000","NVIDIA GeForce RTX 4090","NVIDIA GeForce RTX 3090","NVIDIA RTX 4000 Ada Generation","NVIDIA RTX A6000","NVIDIA L4","NVIDIA L40S","NVIDIA GeForce RTX 5090"]'

START='set -e; command -v sshd >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server); mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; ssh-keygen -A; exec /usr/sbin/sshd -D -e -o PermitRootLogin=prohibit-password'

POD=""
n=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    n=$((n + 1))
    RESP=$(curl -s -X POST https://rest.runpod.io/v1/pods \
        -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "$(python3 - <<PY
import json
print(json.dumps({"name":"colour","imageName":"runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404",
 "gpuTypeIds":json.loads('''$GPUS'''),"gpuCount":1,"containerDiskInGb":40,"ports":["22/tcp"],
 "supportPublicIp":True,"cloudType":"SECURE","dataCenterIds":["$DC"],
 "networkVolumeId":"$VOLUME","volumeMountPath":"/workspace",
 "dockerEntrypoint":["/bin/bash","-c"],"dockerStartCmd":['''$(python3 -c "import json;print(json.dumps('''$START'''))")''']}))
PY
)")
    POD=$(printf '%s' "$RESP" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('id','') or '')
except Exception: print('')")
    [ -n "$POD" ] && { echo "created $POD (attempt $n)"; break; }
    echo "$(date -u +%H:%M) attempt $n: $(printf '%s' "$RESP" | head -c 160)"
    sleep "$GAP"
done
# A silently-failed ID parse once left pods running and billing while the log
# said "no capacity", so confirm against the pod list rather than trusting it.
if [ -z "$POD" ]; then
    echo "no pod id parsed; checking /v1/pods for orphans"
    curl -s -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods \
        | python3 -c "import json,sys;print([(p['name'],p['id']) for p in json.load(sys.stdin)])"
    exit 1
fi

for _ in $(seq 1 60); do
    EP=$(curl -s -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"query { pod(input:{podId:\\\"$POD\\\"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }\"}" \
        | python3 -c "
import json,sys
try:
    rt = json.load(sys.stdin)['data']['pod']['runtime'] or {}
    for p in rt.get('ports') or []:
        if p['isIpPublic'] and p['privatePort'] == 22: print(p['ip'], p['publicPort'])
except Exception: pass")
    [ -n "$EP" ] && break
    sleep 30
done
H=$(echo "$EP" | awk '{print $1}'); P=$(echo "$EP" | awk '{print $2}')
[ -n "$P" ] || { echo "no endpoint after 30m; pod $POD LEFT RUNNING"; exit 1; }
echo "endpoint $H:$P"

for _ in $(seq 1 60); do
    ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
        -o ConnectTimeout=12 -p "$P" "root@$H" 'echo UP' 2>/dev/null | grep -q UP && break
    sleep 20
done

rsync -az --no-o --no-g --include='*/' --include='*.py' --include='*.sh' --include='*.md' \
    --exclude='*' --prune-empty-dirs \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $P" \
    ./ "root@$H:/workspace/code/" || { echo "rsync failed; pod $POD LEFT RUNNING"; exit 1; }
echo "code synced"

# The key is on the pod only so the stopper can stop the pod; stopper.sh shreds
# it the moment it starts.
printf '%s' "$KEY" | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR -p "$P" "root@$H" 'cat > /root/.rpkey && chmod 600 /root/.rpkey'

ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    -p "$P" "root@$H" \
    'cd /workspace/code && chmod +x runpod/colour_work.sh runpod/stopper.sh &&
     setsid nohup bash runpod/colour_work.sh > /workspace/colour_boot.log 2>&1 < /dev/null &
     sleep 1
     setsid nohup env MAX_HOURS=6 bash runpod/stopper.sh > /workspace/stopper_boot.log 2>&1 < /dev/null &
     sleep 2; echo LAUNCHED; pgrep -af "colour_work|stopper" | head'
echo "POD=$POD HOST=$H PORT=$P"
echo LAUNCH_OK
