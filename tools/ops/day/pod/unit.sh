#!/usr/bin/env bash
# POD SIDE, started by runner.sh. Usage: unit.sh <unit>. One pytest process over out/<unit>.ids.
# Env: DUR (pytest --durations, default 40); MARK comes from meta.env (pack.sh).
export PATH="/root/.local/bin:$PATH" UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"; u="$1"; . "$D/meta.env"; cd "$D/repo"
mapfile -t args < "$O/$u.ids"; ma=(-m "${MARK:-not slow}"); [ "$(cat "$O/$u.m" 2>/dev/null)" = A ] && ma=()
date +%s > "$O/$u.start"
/usr/bin/time -v -o "$O/$u.time" uv run --frozen pytest "${ma[@]}" -q -rfEs -p no:cacheprovider \
  --durations="${DUR:-40}" "${args[@]}" > "$O/$u.log" 2>&1
rc=$?; date +%s > "$O/$u.end"; echo $rc > "$O/$u.rc"
