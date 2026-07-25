"""Create a pod and wait until sshd answers; print the SSH endpoint.

Exists because the RunPod REST/GraphQL endpoints intermittently 403/500 and the
images do not start sshd on their own -- both of which have to be handled every
single time a pod is made.

    python make_pod.py --gpu "NVIDIA GeForce RTX 3090" --name train --dc EU-CZ-1 \
        --volume shusq6ritt --disk 60
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request

API = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"

START_CMD = (
    "set -e; "
    "command -v sshd >/dev/null || (apt-get update -qq && "
    "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server); "
    "mkdir -p /run/sshd /root/.ssh; "
    'printf "%s\\n" "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; '
    "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; "
    "ssh-keygen -A; "
    "exec /usr/sbin/sshd -D -e -o PermitRootLogin=prohibit-password"
)

KEY = ""


def req(url, method="GET", body=None, tries=5):
    last = None
    for i in range(tries):
        try:
            r = urllib.request.Request(
                url, method=method,
                data=json.dumps(body).encode() if body else None,
                headers={"Authorization": f"Bearer {KEY}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(r) as resp:
                raw = resp.read().decode()
            return json.loads(raw, strict=False) if raw.strip() else {}
        except Exception as e:
            last = e
            time.sleep(4 * (i + 1))
    raise last


def endpoint(pod_id):
    try:
        d = req(GQL, "POST", {"query": f'query {{ pod(input:{{podId:"{pod_id}"}}) '
                                       f'{{ runtime {{ ports {{ ip isIpPublic privatePort publicPort }} }} }} }}'},
                tries=3)
        rt = (d.get("data", {}).get("pod") or {}).get("runtime") or {}
        for p in rt.get("ports") or []:
            if p["privatePort"] == 22 and p["isIpPublic"]:
                return p["ip"], p["publicPort"]
    except Exception:
        pass
    return None


def main():
    global KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dc", default="EU-CZ-1")
    ap.add_argument("--volume", default="")
    ap.add_argument("--disk", type=int, default=60)
    ap.add_argument("--wait", type=int, default=45, help="minutes to wait for sshd")
    args = ap.parse_args()

    for line in open(".env"):
        if line.startswith("RUNPOD_API_KEY="):
            KEY = line.split("=", 1)[1].strip()

    base = {
        "name": args.name,
        "imageName": "runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404",
        "gpuTypeIds": [args.gpu], "gpuCount": 1,
        "containerDiskInGb": args.disk,
        "ports": ["22/tcp"], "supportPublicIp": True,
        "dockerEntrypoint": ["/bin/bash", "-c"],
        "dockerStartCmd": [START_CMD],
    }
    # A network volume needs SECURE cloud; without one, COMMUNITY is much cheaper.
    if args.volume:
        attempts = [{"cloudType": "SECURE", "dataCenterIds": [args.dc],
                     "networkVolumeId": args.volume, "volumeMountPath": "/workspace"}]
    else:
        attempts = [{"cloudType": "COMMUNITY", "dataCenterIds": [args.dc]},
                    {"cloudType": "SECURE", "dataCenterIds": [args.dc]},
                    {"cloudType": "COMMUNITY"}]

    pod = None
    for extra in attempts:
        try:
            p = req(f"{API}/pods", "POST", {**base, **extra}, tries=2)
            if p.get("id"):
                pod = p
                print(f"created {p['id']} via {extra['cloudType']} at ${p.get('costPerHr')}/hr",
                      file=sys.stderr)
                break
        except Exception as e:
            print(f"  {extra['cloudType']}: {type(e).__name__}", file=sys.stderr)
    if not pod:
        sys.exit("could not create pod")

    deadline = time.time() + args.wait * 60
    while time.time() < deadline:
        time.sleep(20)
        ep = endpoint(pod["id"])
        if not ep:
            continue
        host, port = ep
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15",
             "-p", str(port), f"root@{host}", "echo UP"],
            capture_output=True, text=True)
        if "UP" in r.stdout:
            print(json.dumps({"pod_id": pod["id"], "host": host, "port": port,
                              "price": pod.get("costPerHr")}))
            return
        print(f"  {host}:{port} not answering yet", file=sys.stderr)
    sys.exit(f"pod {pod['id']} never became reachable")


if __name__ == "__main__":
    main()
