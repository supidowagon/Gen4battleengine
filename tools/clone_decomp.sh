#!/usr/bin/env bash
# Clone the pret/pokeplatinum decompilation into external/pokeplatinum at a
# pinned commit. All extraction tools and docs/decomp_paths.md refer to this
# exact commit; bump PINNED_COMMIT deliberately and re-verify the paths doc.
set -euo pipefail

REPO_URL="https://github.com/pret/pokeplatinum"
PINNED_COMMIT="579d2d5c65e736a1a4268fb7acbfd70712177c6c"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/pokeplatinum"

if [ ! -d "$DEST" ]; then
    mkdir -p "$ROOT/external"
    git clone "$REPO_URL" "$DEST"
    git -C "$DEST" -c advice.detachedHead=false checkout "$PINNED_COMMIT"
fi

ACTUAL="$(git -C "$DEST" rev-parse HEAD)"
echo "$ACTUAL"
if [ "$ACTUAL" != "$PINNED_COMMIT" ]; then
    echo "warning: external/pokeplatinum is at $ACTUAL, expected pinned $PINNED_COMMIT" >&2
    exit 1
fi
