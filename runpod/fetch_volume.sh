#!/usr/bin/env bash
# Get files off the EU-CZ-1 network volume when that datacenter has no capacity.
#
# A network volume can only be read from a pod in its own datacenter, so when
# EU-CZ-1 SECURE is full there is no way to reach the artifacts at all. This
# retries until something frees up, copies what it finds, and destroys the pod
# only after the files are verified present locally.
#
# The one real trick: /v1/pods accepts gpuTypeIds as an ARRAY, so a single
# request can say "any of these, whichever is free" instead of polling each type
# in turn. That both raises the hit rate and cuts the request count by ~10x.
#
#   ./fetch_volume.sh            # defaults below
#   MINUTES=600 ./fetch_volume.sh
set -u
cd "$(dirname "$0")/.." || exit 1

KEY=$(grep RUNPOD_API_KEY .env | cut -d= -f2)
VOLUME=${VOLUME:-shusq6ritt}
DC=${DC:-EU-CZ-1}
MINUTES=${MINUTES:-600}
GAP=${GAP:-150}
DEADLINE=$(( $(date +%s) + MINUTES * 60 ))

# Cheapest first: this pod does nothing but copy files, so the GPU is irrelevant
# and only availability matters.
GPUS='["NVIDIA RTX A4000","NVIDIA RTX A4500","NVIDIA RTX A5000","NVIDIA RTX 4000 Ada Generation","NVIDIA GeForce RTX 3090","NVIDIA GeForce RTX 3090 Ti","NVIDIA GeForce RTX 4080","NVIDIA RTX A6000","NVIDIA GeForce RTX 4090","NVIDIA L4","NVIDIA GeForce RTX 5090","NVIDIA RTX 5000 Ada Generation","NVIDIA L40","NVIDIA L40S"]'

START='set -e; command -v sshd >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server); mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; ssh-keygen -A; exec /usr/sbin/sshd -D -e -o PermitRootLogin=prohibit-password'

POD=""
n=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    n=$((n + 1))
    POD=$(curl -s -X POST https://rest.runpod.io/v1/pods \
        -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "$(python3 - <<PY
import json
print(json.dumps({"name":"fetch-vol","imageName":"runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404",
 "gpuTypeIds":json.loads('''$GPUS'''),"gpuCount":1,"containerDiskInGb":40,"ports":["22/tcp"],
 "supportPublicIp":True,"cloudType":"SECURE","dataCenterIds":["$DC"],
 "networkVolumeId":"$VOLUME","volumeMountPath":"/workspace",
 "dockerEntrypoint":["/bin/bash","-c"],"dockerStartCmd":['''$(python3 -c "import json;print(json.dumps('''$START'''))")''']}))
PY
)" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('id','') or '')
except Exception: print('')")
    [ -n "$POD" ] && { echo "created $POD (attempt $n)"; break; }
    echo "$(date -u +%H:%M) attempt $n: no capacity in $DC"
    sleep "$GAP"
done
[ -n "$POD" ] || { echo "gave up after ${MINUTES}m with no capacity"; exit 1; }

for _ in $(seq 1 45); do
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
    sleep 40
done
H=$(echo "$EP" | awk '{print $1}'); P=$(echo "$EP" | awk '{print $2}')
# Never build an ssh command with an empty port -- that silently produces a
# malformed option string and a confusing failure.
[ -n "$P" ] || { echo "no endpoint after 30m; pod $POD LEFT RUNNING"; exit 1; }
echo "endpoint $H:$P"

S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
for _ in $(seq 1 40); do
    ssh -n $S -o ConnectTimeout=12 -p "$P" "root@$H" 'echo UP' 2>/dev/null | grep -q UP && break
    sleep 40
done

mkdir -p ckpt/final plots/data/final/ctx5
ssh -n $S -p "$P" "root@$H" 'ls -la /workspace/final/ 2>/dev/null; echo "--- reaper:"; cat /workspace/reaper.log 2>/dev/null; echo "--- run:"; grep -E "FT2_" /workspace/ft2.log 2>/dev/null | tail -4'
scp $S -P "$P" "root@$H:/workspace/final/ctx5_eval2.json"        plots/data/final/ 2>/dev/null && echo OK_eval
scp $S -P "$P" "root@$H:/workspace/final/ctx5_ft2_history.json"  plots/data/final/ 2>/dev/null && echo OK_hist
scp $S -P "$P" "root@$H:/workspace/final/ft2.log"                plots/data/final/ctx5/ 2>/dev/null && echo OK_log
scp $S -P "$P" "root@$H:/workspace/final/ctx5_ft2.pt"            ckpt/final/ 2>/dev/null && echo OK_model

# Terminate only once the files are verifiably here. A pod costs $0.25-0.50/hr;
# a second capacity wait could cost hours.
if [ -s plots/data/final/ctx5_eval2.json ] && [ -s ckpt/final/ctx5_ft2.pt ]; then
    curl -s -X DELETE -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$POD" \
        -w "TERMINATE HTTP %{http_code}\n"
    echo FETCH_OK
else
    echo "FETCH INCOMPLETE -- pod $POD LEFT RUNNING at $H:$P"
fi
