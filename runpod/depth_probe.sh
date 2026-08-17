#!/usr/bin/env bash
# Does a DEEPER gallery centroid make a player easier to find?
#
# Every eval we have ever run had the target sitting at the cap, because
# --gallery-games 64 truncated to whole k=10 bundles gives exactly 60 games and
# 59% of the gallery piles up there. So gallery-side depth has never been
# varied, and we do not know whether it matters at all. Query-side depth we know
# helps a lot (r@10 0.790 -> 0.867 from 10 to 30 query games).
#
# One shard, two arms, identical players and identical queries -- the only thing
# that changes is how many of each player's games go into their centroid.
# gallery_ctx.py already reserves k games per player for the query, so this is
# leak-free without any extra machinery.
#
# 64 was never chosen for this model: at the old k=5 it also yielded 60 games
# (64//5*5), so the number was inherited and its effect at k=10 is a coincidence.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/depth.log
SHARD=$D/shard
MONTH=${MONTH:-2026-01}
CKPT=$D/ctx10_ft.pt
PLAYERS=${PLAYERS:-50000}
QUERIES=${QUERIES:-2000}
ARMS=${ARMS:-"60 128"}

mkdir -p "$D"
for f in /root/.s3env /root/.s3env.keep; do
    if [ -f "$f" ]; then set -a; . "$f"; set +a; break; fi
done

cd /root/code || exit 1
WORKERS=${WORKERS:-$($PY -c "from cpuquota import cpu_quota; print(cpu_quota())" 2>/dev/null || echo 8)}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

finish() {
    cp -f "$LOG" "$D/depth_run.log" 2>/dev/null
    $PY runpod/s3io.py up "$D/depth_run.log" final/depth_run.log || true
    echo "$1"
    echo NOVOL_ALL_DONE
    exit "${2:-0}"
}

{
echo "=== $(date -u +%H:%M:%S) start | $(nvidia-smi --query-gpu=name --format=csv,noheader) | ${WORKERS} workers ==="

$PY runpod/s3io.py down "data/mt/$MONTH" "$SHARD" || finish NOVOL_UPLOAD_FAILED 1
$PY runpod/s3io.py down final/ctx10_ft_last.pt "$CKPT" || finish NOVOL_UPLOAD_FAILED 1
echo "=== $(date -u +%H:%M:%S) shard + ckpt down ==="

for G in $ARMS; do
    OUT=$D/depth_${G}.json
    if [ -s "$OUT" ]; then echo "arm $G already done"; continue; fi
    echo "=== $(date -u +%H:%M:%S) arm: --gallery-games $G ==="
    # --min-gallery-games 8 and --ks 10 held fixed; ONLY the depth changes.
    if $PY gallery_ctx.py --ckpt "$CKPT" --shard "$SHARD" --out "$OUT" \
            --ks 10 --gallery-games "$G" --min-gallery-games 8 \
            --gallery-players "$PLAYERS" --query-players "$QUERIES" \
            --sizes 10000,50000 --batch 192 --workers "$WORKERS"; then
        echo "ARM_${G}_DONE"
        $PY runpod/s3io.py up "$OUT" "final/depth_${G}.json" || true
    else
        echo "ARM_${G}_FAILED"
    fi
    cp -f "$LOG" "$D/depth_run.log" 2>/dev/null
    $PY runpod/s3io.py up "$D/depth_run.log" final/depth_run.log || true
done

echo "=== $(date -u +%H:%M:%S) comparison ==="
$PY - "$D" $ARMS <<'PYEOF'
import json, os, sys
d, arms = sys.argv[1], sys.argv[2:]
rows = []
for a in arms:
    p = os.path.join(d, f"depth_{a}.json")
    if not os.path.isfile(p):
        print(f"  arm {a}: missing"); continue
    r = json.load(open(p))
    # gallery_ctx.py writes centroid_games_stats at the TOP level, keyed by k.
    rows.append((a, r.get("gallery_built"),
                 r["centroid_games_stats"]["10"],
                 r["pools"]["10"]["direct_at_full_gallery"]))
print(f"{'gallery-games':>14} {'players':>9} {'mean cent':>10} {'r@1':>8} {'r@10':>8} {'r@100':>8}")
for a, n, cs, f in rows:
    mean = (cs or {}).get("mean", float("nan"))
    print(f"{a:>14} {n:>9,} {mean:>10.1f} {f['recall@1']:>8.4f} "
          f"{f['recall@10']:>8.4f} {f['recall@100']:>8.4f}")
if len(rows) == 2:
    d10 = rows[1][3]["recall@10"] - rows[0][3]["recall@10"]
    print(f"\ndelta r@10 ({rows[1][0]} vs {rows[0][0]}): {d10:+.4f}")
    print("n=2000 queries -> SE on a proportion near 0.9 is about 0.007, so a")
    print("real effect should clear roughly +/-0.015 to be worth a rebuild.")
PYEOF

finish NOVOL_UPLOAD_OK 0
} >> "$LOG" 2>&1
