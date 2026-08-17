#!/usr/bin/env bash
# Launch the colour run on a pod with NO network volume.
#
# Tonight's evidence, in order: a pod mounting shusq6ritt with the 16.3 GB
# pytorch image never started in 98 minutes; the same with a 50 MB image failed
# three times in a row at 12 minutes each; an otherwise identical pod with no
# volume was answering ssh in 45 SECONDS. The mount is the fault, so this path
# does not use it -- inputs come from the volume's S3 gateway, which stayed
# healthy for reads and writes the whole time.
#
# Boot is no longer the expensive part (50 MB image, ~1 min), so this retries
# quickly rather than waiting on any one host.
#
#   ./runpod/novol_run.sh
set -u
cd "$(dirname "$0")/.." || exit 1

KEY=$(grep RUNPOD_API_KEY .env | cut -d= -f2)
# Empty DC means ANY datacenter. Mounting the volume is what pinned us to
# EU-CZ-1; pulling inputs over S3 instead removes that constraint entirely, and
# on 2026-08-11 EU-CZ-1 could not start pods at all -- seven in a row stuck at
# uptime 0 across two images, two disk sizes, and with and without the volume.
# Cross-region egress of a 7.2 GB shard costs a few minutes; being stuck in a
# broken datacenter costs the whole night.
DC=${DC-EU-CZ-1}
IMAGE=${IMAGE:-python:3.12-slim}
DISK=${DISK:-100}
WAIT_MIN=${WAIT_MIN:-8}
ATTEMPTS=${ATTEMPTS:-6}
TORCH=${TORCH:-2.9.1}
# Which job this pod runs, and the log its stopper must watch.
WORK=${WORK:-novol_work.sh}
WORKLOG=${WORKLOG:-/data/colour.log}
MAXH=${MAXH:-6}
FT_H=${FT_H:-6}
# Anything else a work script needs, as "K=V K=V". Only PY and FT_H were
# forwarded before, so a work script's other knobs silently fell back to their
# defaults -- which turns a control arm into a rerun of the thing it controls.
EXTRA_ENV=${EXTRA_ENV:-}
PODNAME=${PODNAME:-colour-novol}
export PODNAME
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Overridable: a long run must be pinned to cheap cards. At $0.99/hr a
# 16-hour job costs more than the whole balance and would be killed
# mid-run when funds ran out; the same job on an A5000 is ~$4.30.
GPUS=${GPUS:-'["NVIDIA GeForce RTX 4090","NVIDIA RTX A5000","NVIDIA RTX A4500","NVIDIA RTX A4000","NVIDIA L40S","NVIDIA L40","NVIDIA RTX A6000","NVIDIA GeForce RTX 3090","NVIDIA GeForce RTX 5090","NVIDIA RTX 4000 Ada Generation","NVIDIA L4"]'}

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
    echo "$(date -u +%H:%M) attempt $a/$ATTEMPTS ($IMAGE, ${DISK}GB, no volume)"
    # Build the body in ONE pass, with the start command carried through the
    # environment. The previous version ran json.dumps over $START and then
    # embedded that already-quoted result inside another json.dumps, so the pod
    # received ["\"set -e; ...\""] -- one literal-quoted word. `bash -c` then
    # looked for a command named `set -e; command -v sshd ...`, found none, and
    # the container exited immediately. That presents as uptime 0 with no ports,
    # which is indistinguishable from a slow image pull, and it cost this project
    # an entire night and ~$2 of pods that never ran a line of our code.
    export START IMAGE GPUS DISK DC
    BODY=$(python3 - <<'PY'
import json, os
body = {"name": os.environ.get("PODNAME", "colour-novol"),
        "imageName": os.environ["IMAGE"],
        "gpuTypeIds": json.loads(os.environ["GPUS"]),
        "gpuCount": 1,
        "containerDiskInGb": int(os.environ["DISK"]),
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "cloudType": "SECURE",
        "dockerEntrypoint": ["/bin/bash", "-c"],
        "dockerStartCmd": [os.environ["START"]]}
if os.environ.get("DC"):
    body["dataCenterIds"] = [os.environ["DC"]]
print(json.dumps(body))
PY
)
    POD=$(curl -s -X POST https://rest.runpod.io/v1/pods \
        -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "$BODY" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('id','') or '')
except Exception: print('')")

    [ -z "$POD" ] && { echo "  no capacity"; sleep 60; continue; }
    echo "  created $POD"

    # Read back what the API actually stored. A start command that is not a bare
    # shell line means the container will exit before it does anything, and
    # waiting on it is pure waste -- fail loudly here instead of at minute 6.
    if ! curl -s -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$POD" \
        | python3 -c "
import json,sys
cmd = (json.load(sys.stdin).get('dockerStartCmd') or [''])[0]
print('  start cmd:', (cmd[:60] + '...') if len(cmd) > 60 else cmd)
sys.exit(1 if (not cmd or cmd.lstrip().startswith(('\"', \"'\"))) else 0)"; then
        echo "  MALFORMED dockerStartCmd -- container cannot start"
        kill_pod "$POD"
        exit 2
    fi

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
        sleep 20
    done
    [ -n "$P" ] || { echo "  no sshd after ${WAIT_MIN}m"; kill_pod "$POD"; continue; }

    # shellcheck disable=SC2086
    if ! ssh -n $SSHOPT -o ConnectTimeout=30 -p "$P" "root@$H" "
        set -e
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo 'no nvidia-smi'
        # python:3.12-slim has no C compiler, and torch.compile's triton backend
        # builds a C extension at first use -- without this, --compile dies with
        # 'Failed to find C compiler' three minutes into a six-hour run.
        apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential >/dev/null
        python3 -m venv /root/venv
        /root/venv/bin/pip install -q --upgrade pip
        /root/venv/bin/pip install -q torch==$TORCH numpy chess zstandard boto3
        /root/venv/bin/python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO GPU\")'
    "; then
        echo "  venv build/verify FAILED"; kill_pod "$POD"; continue
    fi

    # shellcheck disable=SC2086
    if ! rsync -az --no-o --no-g --include='*/' --include='*.py' --include='*.sh' \
            --include='*.md' --exclude='*' --prune-empty-dirs \
            -e "ssh $SSHOPT -p $P" ./ "root@$H:/root/code/"; then
        echo "  rsync failed"; kill_pod "$POD"; continue
    fi

    # Credentials go over stdin, never on a command line: an ssh argv is visible
    # in the process list on both ends. Both files are shredded by the scripts
    # that read them.
    # shellcheck disable=SC2086
    printf '%s' "$KEY" | ssh $SSHOPT -p "$P" "root@$H" \
        'cat > /root/.rpkey && chmod 600 /root/.rpkey' || { kill_pod "$POD"; continue; }
    # shellcheck disable=SC2086
    grep -E '^RUNPOD_S3_(ACCESS|SECRET)_KEY=' .env | ssh $SSHOPT -p "$P" "root@$H" \
        'cat > /root/.s3env && chmod 600 /root/.s3env' || { kill_pod "$POD"; continue; }

    # shellcheck disable=SC2086
    if ! ssh -n $SSHOPT -p "$P" "root@$H" \
        'chmod +x /root/code/runpod/*.sh
         setsid nohup env PY=/root/venv/bin/python FT_H='$FT_H' '"$EXTRA_ENV"' bash /root/code/runpod/'$WORK' > /root/work_boot.log 2>&1 < /dev/null &
         sleep 1
         setsid nohup env MAX_HOURS='$MAXH' WORKLOG='$WORKLOG' bash /root/code/runpod/novol_stopper.sh > /root/stopper_boot.log 2>&1 < /dev/null &
         sleep 3; echo LAUNCHED; ps -eo pid,args | grep -E "novol_" | grep -v grep'; then
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
