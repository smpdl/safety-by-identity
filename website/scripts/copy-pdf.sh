#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/Safety By Identity.pdf"
DEST="$(cd "$(dirname "$0")/.." && pwd)/safety-by-identity.pdf"

if [[ ! -f "$SRC" ]]; then
  echo "Paper PDF not found: $SRC" >&2
  exit 1
fi

cp "$SRC" "$DEST"
