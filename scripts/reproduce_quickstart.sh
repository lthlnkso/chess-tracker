#!/usr/bin/env bash
# The fast, free part of the pipeline: ingest one month, verify the encoding,
# print corpus stats. ~2 minutes, no GPU, ~20 MB on disk.
set -euo pipefail
PY="${PY:-python}"
OUT="${OUT:-data/2013-01}"
URL=https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst

$PY ingest.py --url "$URL" --out "$OUT" --workers "${WORKERS:-8}"
$PY verify.py "$OUT"
