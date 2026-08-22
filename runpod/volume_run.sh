#!/usr/bin/env bash
# Self-contained run. No sshd, no interactive shell, nothing installed at boot.
#
# Three pods in a row sat RUNNING with the ssh port mapped and nothing listening,
# because the start command installs openssh-server with apt BEFORE exec'ing
# sshd -- so a flaky mirror leaves an unreachable pod billing by the hour. The
# fix is to stop needing a shell: the code and the logs both live on the network
# volume, which is readable AND writable over the S3 gateway with no pod at all.
#
# So this script is the whole job. It validates on the small trunk first, only
# promotes to the 3x run if that survives, and terminates the pod itself at the
# end -- including on failure, which is the part that matters when nobody is
# watching.
set -u

W=/workspace
CODE=$W/code2
PY=$W/venv/bin/python
LOG=$W/steer_auto.log
POD=${RUNPOD_POD_ID:-}

term() {
    echo "=== $(date -u +%H:%M:%S) terminating pod ${POD:-unknown} ===" >> "$LOG"
    if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "$POD" ]; then
        for i in 1 2 3 4 5; do
            code=$(curl -s -o /dev/null -w '%{http_code}' -m 25 -X DELETE \
                   -H "Authorization: Bearer $RUNPOD_API_KEY" \
                   "https://rest.runpod.io/v1/pods/$POD")
            echo "  terminate attempt $i -> $code" >> "$LOG"
            case "$code" in 200|204) break;; esac
            sleep 20
        done
    else
        echo "  NO API KEY OR POD ID -- cannot self-terminate" >> "$LOG"
    fi
}
# terminate on ANY exit path, including an unhandled error or a kill
trap term EXIT

{
echo "=== $(date -u +%H:%M:%S) boot | pod=$POD ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1
cd "$CODE" || { echo "NO CODE AT $CODE"; exit 1; }
(
  command -v sshd >/dev/null || (apt-get update -qq && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server)
  mkdir -p /run/sshd /root/.ssh
  printf "%s\n" "${PUBLIC_KEY:-}" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys
  /usr/sbin/sshd -D -e -o PermitRootLogin=prohibit-password
) >/tmp/sshd.log 2>&1 &
echo "sshd: attempted in background (best effort, nothing waits on it)"
CPUS=$($PY cpuquota.py 2>/dev/null || echo 8)
echo "$CPUS" > /tmp/cpus          # the training blocks run in their own subshells
echo "cpu quota: $CPUS"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
} >> "$LOG" 2>&1

# ---- stage 1: small trunk, proves the skip guard holds -------------------
{
echo "=== $(date -u +%H:%M:%S) STAGE 1: small trunk validation (${VAL_HRS:-1.2}h) ==="
$PY train_multigame.py --shard $W/data/mt/2026-01 --out $W/ckpt/ctx5_steer_val \
    --init $W/final/ctx5_pre.pt \
    --max-games 5 --max-len-per-game 160 \
    --d-model 256 --layers 8 --heads 8 --d-embed 128 \
    --plies-per-game 8 --n-cand 32 --batch 48 --lr 4e-5 --warmup 500 \
    --balance-gap --elo-steer \
    --workers "$(cat /tmp/cpus 2>/dev/null || echo 8)" \
    --eval-every 5000 --eval-batches 25 --patience 1000000 --amp --compile \
    --max-hours "${VAL_HRS:-1.2}"
echo "stage1 exit=$?"
} >> "$LOG" 2>&1

# ---- gate: did the small run survive with finite weights? ----------------
OK=$($PY - <<'PYEOF' 2>/dev/null
import torch, json, sys
try:
    ck = torch.load("/workspace/ckpt/ctx5_steer_val/last.pt", map_location="cpu",
                    weights_only=False)
    sd = ck["model"]
    finite = all(torch.isfinite(v).all() for v in sd.values() if v.dtype.is_floating_point)
    step = int(ck.get("step") or 0)
    # a run that dies at 2k proves nothing; all three big failures were past 11k
    print("PASS" if (finite and step > 40000) else f"FAIL finite={finite} step={step}")
except Exception as e:
    print(f"FAIL {e}")
PYEOF
)
echo "=== $(date -u +%H:%M:%S) STAGE 1 VERDICT: $OK ===" >> "$LOG"

case "$OK" in
  PASS*)
    {
    echo "=== $(date -u +%H:%M:%S) STAGE 2: 3x trunk (${BIG_HRS:-11}h) ==="
    $PY train_multigame.py --shard $W/data/mt/2026-01 --out $W/ckpt/ctx10_steer \
        --init $W/final/ctx10_pre.pt \
        --max-games 10 --max-len-per-game 160 \
        --d-model 384 --layers 12 --heads 8 --d-embed 128 \
        --plies-per-game 8 --n-cand 32 --batch 32 --lr 4e-5 --warmup 800 \
        --balance-gap --elo-steer \
        --workers "$(cat /tmp/cpus 2>/dev/null || echo 8)" \
        --eval-every 4000 --eval-batches 25 --patience 1000000 --amp --compile \
        --max-hours "${BIG_HRS:-11}"
    echo "stage2 exit=$?"
    } >> "$LOG" 2>&1
    ;;
  *)
    echo "  small run did not pass; NOT promoting to the 3x run" >> "$LOG"
    ;;
esac

echo "=== $(date -u +%H:%M:%S) ALL_DONE ===" >> "$LOG"
