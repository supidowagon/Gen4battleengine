#!/usr/bin/env bash
# POD SIDE, started by runner.sh. Usage: unit.sh <unit> | unit.sh --warmup <node id>.
# <unit>: one pytest process over out/<unit>.ids. --warmup: one pytest process over <node id>
# without a -m filter, its files in out/warmup/ (not a unit: agg.sh reports it apart and never
# counts its tests); runner.sh runs it before the first unit to fill the compile cache.
# Env: DUR (pytest --durations, default 40); MARK comes from meta.env (pack.sh).
# NUZLINE_CACHE_DIR (default /work/_jaxcache) is pod-wide: every unit of every run on the pod
# shares it. tests/conftest.py keeps XLA's compile cache in <it>/jax-tests/<jax version>, whose
# entries JAX keys by program, so a later run (same or other commit) loads every unchanged program.
export PATH="/root/.local/bin:$PATH" UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
export NUZLINE_CACHE_DIR="${NUZLINE_CACHE_DIR:-/work/_jaxcache}"
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"; . "$D/meta.env"; cd "$D/repo"
if [ "$1" = --warmup ]; then mkdir -p "$O/warmup"; P="$O/warmup/warmup"; args=("$2"); ma=()
else u="$1"; P="$O/$u"; mapfile -t args < "$O/$u.ids"
  ma=(-m "${MARK:-not slow and not fullset}"); [ "$(cat "$O/$u.m" 2>/dev/null)" = A ] && ma=(); fi
date +%s > "$P.start"
/usr/bin/time -v -o "$P.time" uv run --frozen pytest "${ma[@]}" -q -rfEs -p no:cacheprovider \
  --durations="${DUR:-40}" "${args[@]}" > "$P.log" 2>&1
rc=$?; date +%s > "$P.end"; echo $rc > "$P.rc"
