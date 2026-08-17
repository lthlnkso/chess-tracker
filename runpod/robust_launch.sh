#!/usr/bin/env bash
# Create a pod, and if it does not come up, kill it and try a different host.
#
# The failure this exists for: a pod can sit in RUNNING with uptimeInSeconds 0
# and no ports for 98 minutes and never start. Waiting is the wrong response --
# the image pull is per-host, so the same 16.3 GB that hangs on one machine
# usually lands in 15 minutes on the next. Cutting a bad host loose after 20
# minutes and retrying elsewhere costs less than waiting on it once.
#
# Every exit path either hands off to a pod that will stop itself, or terminates
# the pod. An unattended launcher that can leave something billing is worse than
# no launcher: this project has already lost ~$2.30 to orphans and ~$0.82 to the
# stuck pod that prompted this script.
#
#   ./runpod/robust_launch.sh
set -u
cd "$(dirname "$0")/.." || exit 1

KEY=$(grep RUNPOD_API_KEY .env | cut -d= -f2)
VOLUME=${VOLUME:-shusq6ritt}
DC=${DC:-EU-CZ-1}
WAIT_MIN=${WAIT_MIN:-20}
ATTEMPTS=${ATTEMPTS:-4}
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# One type per attempt: a different type is a different host pool, which is the
# whole point of retrying.
GPUS=("NVIDIA RTX A5000" "NVIDIA GeForce RTX 4090" "NVIDIA RTX A4500" "NVIDIA L40S")

START='set -e; command -v sshd >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server); mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; ssh-keygen -A; exec /usr/sbin/sshd -D -e -o PermitRootLogin=prohibit-password'

kill_pod() {
    echo "  terminating $1"
    curl -s -X DELETE -H "Authorization: Bearer $KEY" \
        "https://rest.runpod.io/v1/pods/$1" -w " HTTP %{http_code}\n"
}

endpoint() {
    curl -s -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"query { pod(input:{podId:\\\"$1\\\"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }\"}" \
        | python3 -c "
import json,sys
try:
    rt = json.load(sys.stdin)['data']['pod']['runtime'] or {}
    for p in rt.get('ports') or []:
        if p['isIpPublic'] and p['privatePort'] == 22: print(p['ip'], p['publicPort'])
except Exception: pass"
}

for a in $(seq 1 "$ATTEMPTS"); do
    G=${GPUS[$(( (a - 1) % ${#GPUS[@]} ))]}
    echo "$(date -u +%H:%M) attempt $a/$ATTEMPTS on '$G'"
    POD=$(curl -s -X POST https://rest.runpod.io/v1/pods \
        -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "$(python3 - <<PY
import json
print(json.dumps({"name":"colour","imageName":"runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404",
 "gpuTypeIds":["$G"],"gpuCount":1,"containerDiskInGb":40,"ports":["22/tcp"],
 "supportPublicIp":True,"cloudType":"SECURE","dataCenterIds":["$DC"],
 "networkVolumeId":"$VOLUME","volumeMountPath":"/workspace",
 "dockerEntrypoint":["/bin/bash","-c"],"dockerStartCmd":['''$(python3 -c "import json;print(json.dumps('''$START'''))")''']}))
PY
)" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('id','') or '')
except Exception: print('')")

    if [ -z "$POD" ]; then
        echo "  no capacity for '$G'"
        sleep 20
        continue
    fi
    echo "  created $POD"

    H=""; P=""
    END=$(( $(date +%s) + WAIT_MIN * 60 ))
    while [ "$(date +%s)" -lt "$END" ]; do
        EP=$(endpoint "$POD")
        H=$(echo "$EP" | awk '{print $1}'); P=$(echo "$EP" | awk '{print $2}')
        if [ -n "$P" ]; then
            # shellcheck disable=SC2086
            ssh -n $SSHOPT -o ConnectTimeout=10 -p "$P" "root@$H" 'echo UP' 2>/dev/null \
                | grep -q UP && { echo "  sshd up at $H:$P"; break; }
        fi
        H=""; P=""
        sleep 45
    done

    if [ -z "$P" ]; then
        echo "  no sshd after ${WAIT_MIN}m -- bad host, moving on"
        kill_pod "$POD"
        continue
    fi

    # From here any failure must terminate the pod, not leave it billing.
    # shellcheck disable=SC2086
    if ! rsync -az --no-o --no-g --include='*/' --include='*.py' --include='*.sh' \
            --include='*.md' --exclude='*' --prune-empty-dirs \
            -e "ssh $SSHOPT -p $P" ./ "root@$H:/workspace/code/"; then
        echo "  rsync failed"; kill_pod "$POD"; continue
    fi
    # shellcheck disable=SC2086
    printf '%s' "$KEY" | ssh $SSHOPT -p "$P" "root@$H" \
        'cat > /root/.rpkey && chmod 600 /root/.rpkey' || { kill_pod "$POD"; continue; }
    # shellcheck disable=SC2086
    if ! ssh -n $SSHOPT -p "$P" "root@$H" \
        'cd /workspace/code && chmod +x runpod/colour_work.sh runpod/stopper.sh &&
         setsid nohup bash runpod/colour_work.sh > /workspace/colour_boot.log 2>&1 < /dev/null &
         sleep 1
         setsid nohup env MAX_HOURS=6 bash runpod/stopper.sh > /workspace/stopper_boot.log 2>&1 < /dev/null &
         sleep 3; echo LAUNCHED; pgrep -af "colour_work|stopper" | head'; then
        echo "  launch failed"; kill_pod "$POD"; continue
    fi

    echo "POD=$POD HOST=$H PORT=$P"
    echo LAUNCH_OK
    exit 0
done

echo "ALL $ATTEMPTS ATTEMPTS FAILED"
echo "pods still alive (should be none):"
curl -s -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods \
    | python3 -c "import json,sys;print([(p['name'],p['id'],p['desiredStatus']) for p in json.load(sys.stdin)])"
exit 1
