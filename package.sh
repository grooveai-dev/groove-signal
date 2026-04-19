#!/bin/bash
# Build a clean distributable zip of Groove for contributors.
# Usage: bash package.sh [--output filename.zip]
#
# Excludes: venv, __pycache__, .groove, agent logs, dev docs, .DS_Store

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=$(date +%Y%m%d)
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --output|-o) OUTPUT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--output filename.zip]"
            echo "Creates a clean zip for distribution (no venv, no dev files)."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$SCRIPT_DIR/groove-deploy-${VERSION}.zip"
fi

STAGING=$(mktemp -d)
DEST="$STAGING/groove-deploy"
mkdir -p "$DEST"

echo "Packaging groove-deploy..."

cp "$SCRIPT_DIR/setup.sh" "$DEST/"
cp "$SCRIPT_DIR/requirements.txt" "$DEST/"
cp "$SCRIPT_DIR/README.txt" "$DEST/"
cp "$SCRIPT_DIR/QUICKSTART.md" "$DEST/"

cp -r "$SCRIPT_DIR/src" "$DEST/src"
cp -r "$SCRIPT_DIR/scripts" "$DEST/scripts"

mkdir -p "$DEST/tests"
cp "$SCRIPT_DIR/tests/__init__.py" "$DEST/tests/"
cp "$SCRIPT_DIR/tests"/test_*.py "$DEST/tests/"

find "$DEST" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$DEST" -name ".DS_Store" -delete 2>/dev/null || true
find "$DEST" -name "*.pyc" -delete 2>/dev/null || true

rm -f "$OUTPUT"
(cd "$STAGING" && zip -r "$OUTPUT" groove-deploy -x "*.DS_Store" "*.pyc" "*__pycache__*")

rm -rf "$STAGING"

SIZE=$(du -h "$OUTPUT" | cut -f1)
FILE_COUNT=$(unzip -l "$OUTPUT" | grep -c "groove-deploy/")
echo ""
echo "Done: $OUTPUT ($SIZE, $FILE_COUNT files)"
