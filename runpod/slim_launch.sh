#!/usr/bin/env bash
# Boot from a ~50 MB image and build a self-contained venv on the volume, instead
# of pulling the 16.3 GB pytorch image on every pod.
#
# Written because EU-CZ-1 stopped being able to start that image: two pods in a
# row sat in RUNNING with uptimeInSeconds 0 and no ports, one for 98 minutes,
# across two different GPU types. The pull is the thing that is failing, so the
# fix is to stop needing it. python:3.12-slim is ~50 MB and a `pip install torch`
# is ~3 GB -- a fifth of the image, fetched from a different network path.
#
# The venv is built ON the volume at /workspace/venv2, so it survives the pod and
# every future run can boot from the slim image in seconds. That makes this a
# one-time cost, not a per-run one.
#
# /workspace/venv cannot be reused: it was created with --system-site-packages
# and carries only chess, numpy, zstandard and pip -- torch comes from the image.
#
#   ./runpod/slim_launch.sh
set -u
cd "$(dirname "$0")/.." || exit 1

KEY=$(grep RUNPOD_API_KEY .env | cut -d= -f2)
VOLUME=${VOLUME:-shusq6ritt}
DC=${DC:-EU-CZ-1}
IMAGE=${IMAGE:-python:3.12-slim}
WAIT_MIN=${WAIT_MIN:-12}
ATTEMPTS=${ATTEMPTS:-6}
TORCH=${TORCH:-2.9.1}
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Ask for everything at once: with the datacenter this constrained, availability
# matters far more than which card we land on.
GPUS='["NVIDIA GeForce RTX 4090","NVIDIA RTX A5000","NVIDIA RTX A4500","NVIDIA RTX A4000","NVIDIA L40S","NVIDIA L40","NVIDIA RTX A6000","NVIDIA GeForce RTX 3090","NVIDIA GeForce RTX 5090","NVIDIA RTX 4000 Ada Generation","NVIDIA L4"]'

START='set -e; command -v sshd >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server rsync); mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; ssh-keygen -A; exec /usr/sbin/sshd -D -e -o PermitRootLogin=prohibit-password'

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
    echo "$(date -u +%H:%M) attempt $a/$ATTEMPTS with $IMAGE"
    POD=$(curl -s -X POST https://rest.runpod.io/v1/pods \
        -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "$(python3 - <<PY
import json
print(json.dumps({"name":"colour-slim","imageName":"$IMAGE",
 "gpuTypeIds":json.loads('''$GPUS'''),"gpuCount":1,"containerDiskInGb":40,"ports":["22/tcp"],
 "supportPublicIp":True,"cloudType":"SECURE","dataCenterIds":["$DC"],
 "networkVolumeId":"$VOLUME","volumeMountPath":"/workspace",
 "dockerEntrypoint":["/bin/bash","-c"],"dockerStartCmd":['''$(python3 -c "import json;print(json.dumps('''$START'''))")''']}))
PY
)" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('id','') or '')
except Exception: print('')")

    [ -z "$POD" ] && { echo "  no capacity"; sleep 90; continue; }
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
        sleep 30
    done
    [ -n "$P" ] || { echo "  no sshd after ${WAIT_MIN}m"; kill_pod "$POD"; continue; }

    # Build the venv on the VOLUME so this cost is paid once, ever. Idempotent:
    # a venv2 left by an earlier pod is reused and the pip install is skipped.
    # shellcheck disable=SC2086
    if ! ssh -n $SSHOPT -o ConnectTimeout=20 -p "$P" "root@$H" "
        set -e
        echo '--- gpu ---'; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo 'no nvidia-smi'
        if [ ! -x /workspace/venv2/bin/python ]; then
            echo '--- creating /workspace/venv2 ---'
            python3 -m venv /workspace/venv2
            /workspace/venv2/bin/pip install -q --upgrade pip
            /workspace/venv2/bin/pip install -q torch==$TORCH numpy chess zstandard
        else
            echo '--- reusing /workspace/venv2 ---'
        fi
        /workspace/venv2/bin/python -c 'import torch,numpy,chess; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")'
    "; then
        echo "  venv build/verify FAILED"; kill_pod "$POD"; continue
    fi

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
         setsid nohup env PY=/workspace/venv2/bin/python bash runpod/colour_work.sh > /workspace/colour_boot.log 2>&1 < /dev/null &
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
curl -s -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods \
    | python3 -c "import json,sys;print('surviving pods:', [(p['name'],p['id'],p['desiredStatus']) for p in json.load(sys.stdin)])"
exit 1
