#!/usr/bin/env bash
# POD SIDE, started by runner.sh. Usage: unit.sh <unit> | unit.sh --warmup <node id>.
# <unit>: one pytest process over out/<unit>.ids. --warmup: one pytest process over <node id>
# without a -m filter, its files in out/warmup/ (not a unit: agg.sh reports it apart and never
# counts its tests); runner.sh runs it before the first unit to fill the compile cache.
# Env: DUR (pytest --durations, default 40); MARK comes from meta.env (pack.sh).
# NUZLINE_TEST_COMPILE_CACHE (default /work/_jaxcache/jax-tests) is pod-wide: tests/conftest.py
# keeps XLA's compile cache in <it>/<16-hex hash of the jax and jaxlib versions>, whose entries JAX
# keys by program, so a later run (same or other commit) loads every unchanged program. Every other
# cache (cache_root(): the trace exports, the CLI's jax/ folders, each pruned to one code version)
# stays in this run's repo/.cache, so runs of other commits never prune each other's.
# Threads: threads.env (setup.sh) holds THREADS=none|eigen|taskset. eigen appends
# --xla_cpu_multi_thread_eigen=false to XLA_FLAGS and sets OMP/MKL/OPENBLAS_NUM_THREADS=1;
# taskset pins the process to a free slot of cores/MAXP cores (pod-wide flock on /work/_slots/<i>;
# unpinned when every slot is taken). XLA_FLAGS is part of JAX's compile-cache key: one setting per
# pod keeps the cache warm. Every 5 s the process group's threads, alive and running (state R),
# go to <unit>.thr.
export PATH="/root/.local/bin:$PATH" UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
export NUZLINE_TEST_COMPILE_CACHE="${NUZLINE_TEST_COMPILE_CACHE:-/work/_jaxcache/jax-tests}"
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"; . "$D/meta.env"; cd "$D/repo"
if [ "$1" = --warmup ]; then mkdir -p "$O/warmup"; P="$O/warmup/warmup"; args=("$2"); ma=()
else u="$1"; P="$O/$u"; mapfile -t args < "$O/$u.ids"
  ma=(-m "${MARK:-not slow and not fullset}"); [ "$(cat "$O/$u.m" 2>/dev/null)" = A ] && ma=(); fi
TH=$(sed -n 's/^THREADS=//p' "$D/threads.env" 2>/dev/null); TS=()
case "$TH" in
  eigen) export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=false" \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1;;
  taskset) NC=$(nproc); K=$(( NC / ${MAXP:-10} )); [ "$K" -ge 1 ] || K=1; mkdir -p /work/_slots
    for s in $(seq 0 $(( NC / K - 1 ))); do exec 8> "/work/_slots/$s"
      flock -n 8 && { TS=(taskset -c "$((s * K))-$((s * K + K - 1))"); break; }; exec 8>&-; done;;
esac
G=$(ps -o pgid= -p $$ | tr -d ' ')
( while [ ! -f "$P.rc" ]; do ps -L -eo pgid=,stat=,comm= | awk -v g="$G" '$1 == g && $3 != "ps" && $3 != "awk" {a++; if ($2 ~ /^R/) r++}
    END {print a + 0, r + 0}' >> "$P.thr"; sleep 5; done ) &
date +%s > "$P.start"
/usr/bin/time -v -o "$P.time" "${TS[@]}" uv run --frozen pytest "${ma[@]}" -q -rfEs -p no:cacheprovider \
  --durations="${DUR:-40}" "${args[@]}" > "$P.log" 2>&1
rc=$?; date +%s > "$P.end"; echo $rc > "$P.rc"
