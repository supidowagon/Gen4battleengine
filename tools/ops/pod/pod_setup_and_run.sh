#!/usr/bin/env bash
# Run ON THE POD (Ubuntu image, root) after this folder (pod/) has been copied to /root/pod.
# Sets up the repo at COMMIT, unpacks the built tables, smoke-tests, then starts the
# parallel fast tier detached. Progress: bash /root/pod/progress.sh /root/out
set -euo pipefail
COMMIT=4676fd41f00c33d77c5e0545021a48737d2a5294
DECOMP=579d2d5c65e736a1a4268fb7acbfd70712177c6c
BUNDLE_SHA=566f17c36e3f0f439a4c6a762f6024d4c27436b470bc199ba1e13ddc9b4913d5
cd /root
echo "$BUNDLE_SHA  pod/tables_4676fd4.tar.xz" | sha256sum -c -
export DEBIAN_FRONTEND=noninteractive
command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq git curl time xz-utils ca-certificates >/dev/null; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d repo ] || git clone -q "${NUZLINE_REPO_URL:?set NUZLINE_REPO_URL}" repo
git -C repo -c advice.detachedHead=false checkout -q "$COMMIT"
mkdir -p repo/external
[ -d repo/external/pokeplatinum ] || git clone -q https://github.com/pret/pokeplatinum repo/external/pokeplatinum
git -C repo/external/pokeplatinum -c advice.detachedHead=false checkout -q "$DECOMP"
tar -xJf pod/tables_4676fd4.tar.xz -C repo
cd repo
uv sync --frozen
git status --short   # expect clean (tables are git-ignored; vanilla manifest identical)
# smoke test: three quick files
uv run --frozen pytest -m "not slow" -q -p no:cacheprovider \
  tests/data/test_game_constants.py tests/prep/test_variants.py tests/tools/test_backlog.py
# Parts: collect the fast tier HERE (node ids hold OS paths, so a Windows collection does not
# match) and deal the node ids round-robin into NPARTS files. In the 2026-11-17 trial the
# static time-estimated split left 6 of 10 parts at 35-60% after 130 min, while the same
# tail dealt round-robin over 10 processes finished in 8-16 min each.
NPARTS="${NPARTS:-10}"
rm -rf /root/pod/parts_rr && mkdir -p /root/pod/parts_rr
uv run --frozen pytest --collect-only -q -m "not slow" -p no:cacheprovider 2>/dev/null \
  | grep '::' > /root/pod/collected.txt
awk -v n="$NPARTS" '{print > sprintf("/root/pod/parts_rr/part_%02d.txt", (NR-1)%n)}' /root/pod/collected.txt
wc -l < /root/pod/collected.txt
nohup bash /root/pod/run_parts.sh /root/repo /root/pod/parts_rr /root/out > /root/out_runner.log 2>&1 &
echo "runner started"
