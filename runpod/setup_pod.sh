#!/usr/bin/env bash
# Prepare the pod. Everything lands on /workspace (the network volume), so it
# survives the pod being terminated and recreated.
set -euo pipefail

mkdir -p /workspace/{code,data,ckpt}

if [ ! -d /workspace/venv ]; then
    # --system-site-packages so we inherit the image's CUDA-enabled torch
    # instead of pulling a second multi-GB copy onto the volume.
    python3 -m venv --system-site-packages /workspace/venv
fi

/workspace/venv/bin/pip install -q --upgrade pip
/workspace/venv/bin/pip install -q python-chess zstandard numpy

/workspace/venv/bin/python - <<'PY'
import torch, chess, zstandard, numpy
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
print(f"python-chess {chess.__version__}  numpy {numpy.__version__}  zstandard {zstandard.__version__}")
PY

echo "--- disk ---"
df -h /workspace | tail -1
echo "--- cpu/mem ---"
echo "$(nproc) vCPU, $(free -g | awk '/^Mem:/{print $2}') GB RAM"
