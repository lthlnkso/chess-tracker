#!/usr/bin/env bash
# Build the CPL corpus: Stockfish evals for every legal move, for every ply of a
# player subset. This is the input to the `cpl` experiment.
#
# CPU-ONLY. The GPU on this pod is dead weight -- pick the cheapest card with the
# most vCPU, because throughput here is one single-threaded engine per core
# (measured 12.4 pos/s/core at Threads=1 vs 4.0 at Threads=4).
#
# Settings are all measured, in profile_cpl{,2,3}.py:
#   depth 6      deeper only sharpens the 0-30cp band, where label noise is
#                106cp even at depth 8. That band is noise-dominated at any
#                affordable depth; the coarse structure (candidates span ~550cp)
#                is what survives, and depth 6 keeps it at 3x less cost.
#   multipv 32   one search returns every root move from a shared tree. The
#                per-candidate alternative is 6.2G searches on a shard. MultiPV
#                WIDTH is the dominant cost term -- 6.1ms at mpv1, 70ms at mpv32.
#   whole games  every ply of chosen players' games, so the loader can sample
#                plies as freely as the baseline does instead of being pinned to
#                a labelled scatter.
#
# The labeller writes whatever it finished when --max-hours expires, so the cap
# bounds cost without risking an empty result.
set -u

D=/data
PY=${PY:-/root/venv/bin/python}
LOG=$D/cpl.log
SHARD=$D/shard
OUT=$D/cpl
MONTH=${MONTH:-2026-01}
PLAYERS=${PLAYERS:-30000}
MIN_GAMES=${MIN_GAMES:-7}     # 5 game slots + 2, matching finetune_ctx
PER_PLAYER=${PER_PLAYER:-20}  # cap per player -> even coverage, bounded corpus
DEPTH=${DEPTH:-6}
MULTIPV=${MULTIPV:-32}
LABEL_H=${LABEL_H:-5}

mkdir -p "$D"
if [ -f /root/.s3env ]; then
    set -a; . /root/.s3env; set +a
    shred -u /root/.s3env 2>/dev/null || rm -f /root/.s3env
fi

cd /root/code || exit 1

finish() {
    cp -f "$LOG" "$D/cpl_run.log" 2>/dev/null
    $PY runpod/s3io.py up "$D/cpl_run.log" final/cpl_run.log || true
    echo "$1"; echo NOVOL_ALL_DONE; exit "${2:-0}"
}

{
echo "=== $(date -u +%H:%M:%S) start | $(nproc) vCPU | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="

echo "=== $(date -u +%H:%M:%S) installing stockfish ==="
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq stockfish >/dev/null 2>&1
SF=$(command -v stockfish || echo /usr/games/stockfish)
[ -x "$SF" ] || finish CPL_NO_STOCKFISH 1
echo "  $SF"; "$SF" quit </dev/null 2>&1 | head -1

$PY runpod/s3io.py down "data/mt/$MONTH" "$SHARD" || finish NOVOL_UPLOAD_FAILED 1

# nproc lies inside a RunPod container -- it reports the HOST. cpl_label.py
# reads the cgroup quota itself when --workers is 0.
WORKERS=0
QUOTA=$(awk '{if ($1=="max") print 0; else printf "%d", $1/$2}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
echo "  nproc=$(nproc)  cgroup quota=${QUOTA:-unknown} cores"
echo "=== $(date -u +%H:%M:%S) labelling: $PLAYERS players, depth $DEPTH, "\
"multipv $MULTIPV, $WORKERS engines, cap ${LABEL_H}h ==="

# Bench first: 1 worker vs all workers on the same games. The previous attempt
# ran 94 engines at 114 plies/s with 3 in R state, and "engines idle" has two
# very different causes -- a blocked result pipe, or hardware that cannot feed
# this many engines. This says which, for about two minutes of pod time.
echo "=== $(date -u +%H:%M:%S) scaling bench ==="
$PY cpl_label.py --shard "$SHARD" --out "$D/bench" --engine "$SF" \
    --players 400 --min-games "$MIN_GAMES" --games-per-player "$PER_PLAYER" \
    --depth "$DEPTH" --multipv "$MULTIPV" --workers "$WORKERS" --bench 60 || true
rm -rf "$D/bench"

if $PY cpl_label.py --shard "$SHARD" --out "$OUT" --engine "$SF" \
        --players "$PLAYERS" --min-games "$MIN_GAMES" \
        --games-per-player "$PER_PLAYER" \
        --depth "$DEPTH" --multipv "$MULTIPV" \
        --workers "$WORKERS" --max-hours "$LABEL_H"; then
    echo CPL_LABEL_OK
else
    echo CPL_LABEL_FAILED
fi

[ -s "$OUT/offsets.npy" ] || finish CPL_NO_CORPUS 1

echo "=== $(date -u +%H:%M:%S) packing + uploading ==="
tar -C "$D" -cf - cpl | zstd -3 -T0 -o "$D/cpl_corpus.tar.zst" 2>/dev/null \
    || tar -C "$D" -czf "$D/cpl_corpus.tar.gz" cpl
for f in cpl_corpus.tar.zst cpl_corpus.tar.gz; do
    [ -s "$D/$f" ] && $PY runpod/s3io.py up "$D/$f" "final/$f"
done
# Record the LOGICAL shard, not the pod-local path. cpl_label.py writes
# whatever --shard it was given ("/data/shard"), and CplCorpus.assert_shard
# compares basenames -- so an uncorrected manifest makes a perfectly good corpus
# refuse to train against the month it actually came from.
$PY - "$OUT/manifest.json" "$MONTH" <<'MANIFEST_FIX'
import json, sys
p, month = sys.argv[1], sys.argv[2]
d = json.load(open(p)); d["shard"] = f"data/mt/{month}"
json.dump(d, open(p, "w"), indent=2)
print(f"  manifest shard -> {d['shard']}")
MANIFEST_FIX
$PY runpod/s3io.py up "$OUT/manifest.json" final/cpl_manifest.json || true
ls -la "$D"/cpl_corpus.* 2>/dev/null

finish NOVOL_UPLOAD_OK 0
} >> "$LOG" 2>&1
