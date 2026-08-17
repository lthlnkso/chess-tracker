#!/usr/bin/env bash
# Pull /workspace/final off the network volume over the S3 API.
#
# The volume is itself an S3 bucket (bucket name == volume id), so this needs no
# pod at all. That matters: the old path was "create a pod in EU-CZ-1, wait 15-20
# minutes for a 16.3 GB image pull, scp, terminate", which cost real money and
# failed outright whenever the datacenter was full. This takes seconds and works
# while the pod is stopped or gone.
#
#   ./runpod/fetch_final.sh                 # everything under final/
#   ./runpod/fetch_final.sh ctx5_colour     # only keys matching a substring
set -u
cd "$(dirname "$0")/.." || exit 1

PAT=${1:-}
DEST=${DEST:-plots/data/final}
mkdir -p "$DEST" ckpt/final play

.venv/bin/python - "$PAT" "$DEST" <<'PY'
import os, sys, boto3

pat, dest = sys.argv[1], sys.argv[2]
env = dict(l.strip().split("=", 1) for l in open(".env")
           if "=" in l and not l.startswith("#"))
s3 = boto3.client("s3", endpoint_url="https://s3api-eu-cz-1.runpod.io",
                  aws_access_key_id=env["RUNPOD_S3_ACCESS_KEY"],
                  aws_secret_access_key=env["RUNPOD_S3_SECRET_KEY"],
                  region_name="EU-CZ-1")
B = "shusq6ritt"

# Big binaries belong next to the other checkpoints/galleries, not in the plots
# tree, so route by extension rather than dumping everything in one place.
def target(key):
    name = os.path.basename(key)
    if name.endswith(".pt"):
        return os.path.join("ckpt/final", name)
    if name.endswith(".npz"):
        return os.path.join("play", name)
    return os.path.join(dest, name)

found = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix="final/"):
    for o in page.get("Contents", []):
        key, size = o["Key"], o["Size"]
        if key.endswith("/") or (pat and pat not in key):
            continue
        out = target(key)
        # Skip only when the local copy is byte-identical in size; a partial or
        # superseded file must be re-fetched, not silently kept.
        if os.path.exists(out) and os.path.getsize(out) == size:
            print(f"  = {out} ({size/1e6:.1f} MB, unchanged)")
            found += 1
            continue
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        s3.download_file(B, key, out)
        print(f"  + {out} ({size/1e6:.1f} MB)")
        found += 1
print(f"{found} file(s)" + (f" matching {pat!r}" if pat else ""))
PY
