"""Move files between a pod and the RunPod network volume over the S3 API.

Exists because the volume's MOUNT is not always usable. On 2026-08-10 every pod
that mounted shusq6ritt sat in RUNNING with uptimeInSeconds 0 and never started
-- one for 98 minutes -- across two images (16.3 GB and 50 MB) and four GPU
types, while an otherwise identical pod with no volume booted in 45 seconds. The
S3 gateway to the same volume kept working for both reads and writes throughout,
so this is the way in and out when the mount is the thing that is broken.

Credentials come from the environment (RUNPOD_S3_ACCESS_KEY / _SECRET_KEY) and
are never written to disk by this script.

    python s3io.py down data/mt/2026-01 /data/shard
    python s3io.py up   /data/out.json  final/out.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

BUCKET = os.environ.get("RP_BUCKET", "shusq6ritt")
ENDPOINT = os.environ.get("RP_S3_ENDPOINT", "https://s3api-eu-cz-1.runpod.io")
REGION = os.environ.get("RP_S3_REGION", "EU-CZ-1")

# Multi-GB shards over a gateway that has already proven flaky today: wide
# parallelism for throughput, generous retries so one stalled part does not lose
# a 7 GB download.
XFER = TransferConfig(multipart_threshold=64 * 1024 * 1024,
                      multipart_chunksize=64 * 1024 * 1024,
                      max_concurrency=16, use_threads=True)


def client():
    return boto3.client(
        "s3", endpoint_url=ENDPOINT, region_name=REGION,
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_KEY"],
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                      read_timeout=120, connect_timeout=30))


def human(n):
    return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.1f} MB"


def down(prefix, dest, attempts=4):
    s3 = client()
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            if not o["Key"].endswith("/"):
                keys.append((o["Key"], o["Size"]))
    if not keys:
        sys.exit(f"nothing under {prefix!r}")
    total = sum(s for _, s in keys)
    print(f"{len(keys)} object(s), {human(total)} -> {dest}", flush=True)

    single = len(keys) == 1 and keys[0][0] == prefix
    for key, size in keys:
        out = dest if single else os.path.join(dest, os.path.relpath(key, prefix))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) == size:
            print(f"  = {os.path.basename(out)} ({human(size)}, already here)", flush=True)
            continue
        for a in range(1, attempts + 1):
            t0 = time.time()
            try:
                s3.download_file(BUCKET, key, out, Config=XFER)
                dt = max(time.time() - t0, 1e-9)
                print(f"  + {os.path.basename(out)} {human(size)} in {dt:.0f}s "
                      f"({size/dt/1e6:.0f} MB/s)", flush=True)
                break
            except Exception as e:
                print(f"  ! {key} attempt {a}/{attempts}: {type(e).__name__} {e}", flush=True)
                if a == attempts:
                    sys.exit(1)
                time.sleep(5 * a)


def up(path, key, attempts=6):
    s3 = client()
    size = os.path.getsize(path)
    for a in range(1, attempts + 1):
        t0 = time.time()
        try:
            s3.upload_file(path, BUCKET, key, Config=XFER)
            # Verify rather than trust: a silent short write here would be
            # discovered only after the pod is gone.
            got = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
            if got != size:
                raise IOError(f"size mismatch: local {size}, remote {got}")
            print(f"  ^ {key} {human(size)} in {max(time.time()-t0,1e-9):.0f}s", flush=True)
            return
        except Exception as e:
            print(f"  ! upload {key} attempt {a}/{attempts}: {type(e).__name__} {e}", flush=True)
            if a == attempts:
                sys.exit(1)
            time.sleep(5 * a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("down", "up"))
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    (down if a.mode == "down" else up)(a.src, a.dst)
