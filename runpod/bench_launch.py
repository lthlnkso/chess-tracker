"""Spin up one pod per GPU, benchmark, collect JSON, tear down.

Runs the GPUs in parallel because the image pull dominates wall time (~15 min).
Each pod is self-contained: it stream-ingests Jan 2013 onto its own container
disk, so no network volume is needed and the pods can live in any datacenter.

    python bench_launch.py --dc EU-CZ-1 --out plots/data/bench
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import urllib.request

API = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
KEY = ""

LICHESS = ("https://database.lichess.org/standard/"
           "lichess_db_standard_rated_2013-01.pgn.zst")

# sshd is not started by the runpod images on their own; see README.
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

GPUS = [
    ("NVIDIA GeForce RTX 3090", "RTX 3090"),
    ("NVIDIA GeForce RTX 4090", "RTX 4090"),
    ("NVIDIA GeForce RTX 5090", "RTX 5090"),
]

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20",
]

lock = threading.Lock()


def log(tag: str, msg: str) -> None:
    with lock:
        print(f"[{tag:>9}] {msg}", flush=True)


def _request(req, tries: int = 4):
    """RunPod intermittently returns 403/500; a single blip must not kill a run."""
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(req) as r:
                raw = r.read().decode()
            return json.loads(raw, strict=False) if raw.strip() else {}
        except Exception as e:
            last = e
            time.sleep(3 * (i + 1))
    raise last


def api(method: str, path: str, body: dict | None = None, tries: int = 4):
    return _request(urllib.request.Request(
        f"{API}/{path}", method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    ), tries)


def gql(query: str, tries: int = 4):
    return _request(urllib.request.Request(
        GQL, method="POST", data=json.dumps({"query": query}).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    ), tries)


def ssh_endpoint(pod_id: str):
    """GraphQL runtime.ports is the reliable source; REST often omits it."""
    try:
        d = gql(f'query {{ pod(input:{{podId:"{pod_id}"}}) {{ runtime '
                f'{{ ports {{ ip isIpPublic privatePort publicPort }} }} }} }}')
        rt = (d.get("data", {}).get("pod") or {}).get("runtime") or {}
        for p in rt.get("ports") or []:
            if p["privatePort"] == 22 and p["isIpPublic"]:
                return p["ip"], p["publicPort"]
    except Exception:
        pass
    try:
        p = api("GET", f"pods/{pod_id}", tries=2)
        for m in p.get("portMappings") or []:
            if str(m.get("privatePort")) == "22" and p.get("publicIp"):
                return p["publicIp"], m["publicPort"]
    except Exception:
        pass
    return None


def create_pod(name: str, gpu_id: str, dc: str, tag: str):
    """COMMUNITY is cheaper but not every GPU is offered there; fall back."""
    base = {
        "name": name,
        "imageName": "runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404",
        "gpuTypeIds": [gpu_id], "gpuCount": 1,
        "containerDiskInGb": 40,
        "ports": ["22/tcp"], "supportPublicIp": True,
        "dockerEntrypoint": ["/bin/bash", "-c"],
        "dockerStartCmd": [START_CMD],
    }
    attempts = [
        {"cloudType": "COMMUNITY", "dataCenterIds": [dc]},
        {"cloudType": "SECURE", "dataCenterIds": [dc]},
        {"cloudType": "COMMUNITY"},
        {"cloudType": "SECURE"},
    ]
    for extra in attempts:
        try:
            pod = api("POST", "pods", {**base, **extra}, tries=2)
            if pod.get("id"):
                where = extra.get("dataCenterIds", ["any-dc"])[0]
                log(tag, f"created via {extra['cloudType']}/{where}")
                return pod
        except Exception as e:
            log(tag, f"{extra['cloudType']}/{extra.get('dataCenterIds',['any'])[0]}: "
                     f"{type(e).__name__}")
    return None


def sh(host: str, port: int, cmd: str, timeout: int = 3600):
    return subprocess.run(
        ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def run_one(gpu_id: str, label: str, dc: str, outdir: str, steps: int, e2e: int):
    tag = label.split()[-1]
    pod = None
    try:
        pod = create_pod(f"bench-{tag}", gpu_id, dc, tag)
        if not pod:
            log(tag, "CREATE FAILED on every cloud/DC combination")
            return None
        price = pod.get("costPerHr")
        log(tag, f"pod {pod['id']} created at ${price}/hr, waiting for sshd "
                 f"(image pull ~15 min)")

        host = port = None
        for _ in range(120):                       # up to 40 min
            time.sleep(20)
            ep = ssh_endpoint(pod["id"])
            if ep:
                host, port = ep
                r = sh(host, port, "echo UP", timeout=40)
                if "UP" in r.stdout:
                    break
        if not host:
            log(tag, "never became reachable")
            return None
        log(tag, f"reachable at {host}:{port}, setting up")

        r = sh(host, port, "mkdir -p /root/code /root/data && nvidia-smi "
                           "--query-gpu=name --format=csv,noheader", timeout=120)
        log(tag, f"device: {r.stdout.strip()}")

        subprocess.run(
            ["rsync", "-az", "--no-o", "--no-g",
             "-e", f"ssh {' '.join(SSH_OPTS)} -p {port}",
             "--include=*.py", "--exclude=*", "./", f"root@{host}:/root/code/"],
            check=True, capture_output=True, timeout=300,
        )

        r = sh(host, port,
               "python3 -m venv --system-site-packages /root/venv && "
               "/root/venv/bin/pip install -q python-chess zstandard numpy 2>&1 | tail -2",
               timeout=900)
        log(tag, "venv ready" if r.returncode == 0 else f"venv FAILED {r.stderr[-300:]}")

        r = sh(host, port,
               f"cd /root/code && /root/venv/bin/python ingest.py --url {LICHESS} "
               f"--out /root/data/2013-01 --workers 12 2>&1 | tail -1", timeout=1800)
        log(tag, f"ingest: {r.stdout.strip()[-90:]}")

        r = sh(host, port,
               f"cd /root/code && /root/venv/bin/python bench_gpu.py "
               f"--shard /root/data/2013-01 --gpu '{label}' --price {price} "
               f"--steps {steps} --e2e-steps {e2e} --out /root/bench.json 2>&1",
               timeout=2400)
        if r.returncode != 0:
            log(tag, f"BENCH FAILED: {(r.stdout + r.stderr)[-400:]}")
            return None

        os.makedirs(outdir, exist_ok=True)
        dest = os.path.join(outdir, f"{tag}.json")
        subprocess.run(
            ["rsync", "-az", "-e", f"ssh {' '.join(SSH_OPTS)} -p {port}",
             f"root@{host}:/root/bench.json", dest], check=True, timeout=120)
        res = json.load(open(dest))
        log(tag, f"gpu_only {res['gpu_only_it_s']} it/s | end2end {res['end2end_it_s']} it/s "
                 f"| ${res['usd_per_m_gamesides_e2e']}/M game-sides")
        return res
    except Exception as e:
        log(tag, f"ERROR {type(e).__name__}: {str(e)[:250]}")
        return None
    finally:
        if pod and pod.get("id"):
            try:
                api("DELETE", f"pods/{pod['id']}")
                log(tag, "pod terminated")
            except Exception as e:
                log(tag, f"TERMINATE FAILED ({e}) -- CHECK CONSOLE, pod {pod['id']}")


def main():
    global KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc", default="EU-CZ-1")
    ap.add_argument("--out", default="plots/data/bench")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--e2e-steps", type=int, default=200)
    args = ap.parse_args()

    for line in open(".env"):
        if line.startswith("RUNPOD_API_KEY="):
            KEY = line.split("=", 1)[1].strip()
    if not KEY:
        sys.exit("no RUNPOD_API_KEY in .env")

    threads, out = [], {}
    for gpu_id, label in GPUS:
        t = threading.Thread(target=lambda g=gpu_id, l=label: out.update(
            {l: run_one(g, l, args.dc, args.out, args.steps, args.e2e_steps)}))
        t.start()
        threads.append(t)
        time.sleep(4)                              # stagger pod creation
    for t in threads:
        t.join()

    print("\n=== results ===")
    for label, r in out.items():
        print(f"{label}: {'FAILED' if not r else r['usd_per_m_gamesides_e2e']}")


if __name__ == "__main__":
    main()
