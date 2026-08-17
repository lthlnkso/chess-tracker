#!/usr/bin/env bash
# Fetch the three files the demo needs but git does not carry.
#
#   ckpt/final/ctx5_pre.pt      32 MB   the bot (chooses moves)
#   ckpt/final/ctx10_ft.pt      97 MB   the identifier (scores finished games)
#   play/gallery_ctx10.npz     137 MB   558,735 centroids
#
# The identifier and the gallery are a MATCHED PAIR -- centroids are only
# comparable to query embeddings from the same weights, and mixing them does not
# degrade gracefully, it collapses (measured r@1 0.0000). Both are pinned to the
# same source and verified by sha256 below.
#
#   ARTIFACT_BASE=https://…            any static host serving the three names
#   ARTIFACT_BASE=s3://bucket/prefix   uses awscli if you would rather not
#                                      publish weights
set -euo pipefail

REPO="${REPO:-lthlnkso/chess-tracker}"
TAG="${TAG:-artifacts-ctx10}"
BASE="${ARTIFACT_BASE:-https://github.com/${REPO}/releases/download/${TAG}}"

# sha256 of exactly what docs/prod_deployed.md documents as deployed. A
# corrupted or substituted gallery is not a loud failure -- it identifies the
# wrong people with full confidence -- so this is verified, not assumed.
read -r -d '' WANT <<'EOF' || true
ckpt/final/ctx5_pre.pt 5416f7a7eab46935aadfe74ddbe2fe97d10477fb293b8b922fa34a6dec2f9379
ckpt/final/ctx10_ft.pt b230917f935d33b12a038257f1b9fdaf85316f9b1b467e371047dce58070b7a8
play/gallery_ctx10.npz 24908fbd6598011a8fcda3c211ff9907b6c2e269875c5b27b342482ba5b737b3
EOF

sha_of() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

mkdir -p ckpt/final play

echo "$WANT" | while read -r path want; do
    [ -n "$path" ] || continue
    name=$(basename "$path")
    if [ -f "$path" ] && [ "$(sha_of "$path")" = "$want" ]; then
        echo "have  $path"
        continue
    fi
    echo "fetch $path"
    case "$BASE" in
        s3://*) aws s3 cp "${BASE}/${name}" "$path" ;;
        *)      curl -fL --retry 3 --progress-bar "${BASE}/${name}" -o "$path" ;;
    esac
    got=$(sha_of "$path")
    if [ "$got" != "$want" ]; then
        echo "SHA256 MISMATCH for $path" >&2
        echo "  want $want" >&2
        echo "  got  $got" >&2
        rm -f "$path"
        exit 1
    fi
    echo "  verified $name"
done

ls -la ckpt/final/*.pt play/gallery_ctx10.npz
