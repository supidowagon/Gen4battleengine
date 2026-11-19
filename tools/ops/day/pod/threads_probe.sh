#!/usr/bin/env bash
# POD SIDE. Usage: threads_probe.sh. Run by setup.sh before the runner starts when the exec-pod body
# sets THREAD_PROBE=1 (run it on a pod with no other run active: it measures one process at a time).
# Runs the same pytest process (THREAD_PROBE_IDS, default tests/engine/test_step_basic.py, -m "not
# slow") once per thread setting, each with an empty compile cache of its own (every setting
# compiles step cold, the part that uses most threads), and samples the process group every second:
#   none     no thread setting (the runs before this probe)
#   eigen    XLA_FLAGS --xla_cpu_multi_thread_eigen=false; OMP/MKL/OPENBLAS_NUM_THREADS=1
#   taskset  taskset -c 0..k-1, k = cores / MAXP (one runner slot's share), no flags
# Writes out/threads_probe.txt: per setting the most threads alive, the most and the mean running
# (state R) at once, CPU-s (user + sys), wall s, CPU-s / wall, peak RSS MB and rc; then the verdict
# THREADS=<eigen|taskset|none> into out/threads_probe.env and /work/_threads.env (the pod's default
# for later runs, THREADS=auto in unit.sh): the first of eigen, taskset whose rc is 0, whose most
# running threads are at most 4, whose CPU-s are at most 5% above none's and whose equality run
# passed; else none.
# Equality run: the threading settings can change XLA's float summation order, and the player tests
# assert exact float equality (tests/search/test_player_eval.py, test_player_learn.py). The files of
# THREAD_PROBE_EQ (default those two and tests/search/test_player.py) run without a -m filter
# (their float-comparing tests are marked slow; fullset included) once per setting, the three at
# once, each with its own empty caches. Expected count: a --collect-only of the same files. A
# setting's equality run is ok only with rc 0 and exactly that many passed (fails closed: a skip,
# a deselect or a short run is a fail). A setting whose run is not ok is never the verdict, and a
# fail under none (the files themselves fail) leaves the verdict at none.
# Lines "eq <setting> ok|fail rc <rc> passed <n> of <expected> <pytest summary>".
set -u
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"; . "$D/meta.env"; R="$D/repo"; PB="$D/probe"
export PATH="/root/.local/bin:$PATH" UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
read -r -a ids <<< "${THREAD_PROBE_IDS:-tests/engine/test_step_basic.py}"
read -r -a eqids <<< "${THREAD_PROBE_EQ:-tests/search/test_player_eval.py tests/search/test_player_learn.py tests/search/test_player.py}"
NC=$(nproc); K=$(( NC / ${MAXP:-10} )); [ "$K" -ge 1 ] || K=1
rm -rf "$PB"; mkdir -p "$PB"
# sample <pgid> <file>: one line per second, "<threads alive> <threads running>" over the group.
sample() { local g="$1" f="$2" p a r st
  while ps -eo pgid= | grep -qx " *$g"; do a=0; r=0
    for p in $(ps -eo pid=,pgid= | awk -v g="$g" '$2 == g {print $1}'); do
      for st in /proc/"$p"/task/*/stat; do [ -r "$st" ] || continue; a=$((a + 1))
        [ "$(sed 's/^.*) //' "$st" 2>/dev/null | cut -c1)" = R ] && r=$((r + 1)); done; done
    echo "$a $r" >> "$f"; sleep 1; done; }
run() { local c="$1"; shift; local P="$PB/$c"; mkdir -p "$P/cache"; : > "$P/samples"
  ( cd "$R" && export NUZLINE_CACHE_DIR="$P/cache" NUZLINE_TEST_COMPILE_CACHE="$P/cache/jax-tests" && "$@" setsid /usr/bin/time -v -o "$P/time" \
      uv run --frozen pytest -m "not slow" -q -p no:cacheprovider "${ids[@]}" \
      > "$P/log" 2>&1; echo $? > "$P/rc" ) &
  local w=$! g=""; for _ in $(seq 1 50); do g=$(pgrep -n -f "^/usr/bin/time -v -o $P/time" || true); [ -n "$g" ] && break; sleep 0.2; done
  [ -n "$g" ] && sample "$(ps -o pgid= -p "$g" | tr -d ' ')" "$P/samples"; wait "$w"
  awk -v c="$c" -v rc="$(cat "$P/rc")" -v f="$P/time" '
    {if ($1 > a) a = $1; if ($2 > r) r = $2; s += $2; n++}
    END {while ((getline l < f) > 0) {
           if (l ~ /User time \(seconds\)|System time \(seconds\)/) {split(l, x, ": "); cpu += x[2]}
           if (l ~ /Elapsed \(wall clock\)/) {k = split(l, x, ": "); m = split(x[k], t, ":"); w = 0; for (i = 1; i <= m; i++) w = w * 60 + t[i]}
           if (l ~ /Maximum resident/) {split(l, x, ": "); rss = int(x[2] / 1024)}}
         printf "%s %d %d %.1f %.0f %.0f %.2f %d %s\n", c, a, r, (n ? s / n : 0), cpu, w, (w ? cpu / w : 0), rss, rc}' "$P/samples"; }
{ echo "# threads_probe: ${ids[*]}; mark not slow; $NC cores, taskset slice $K; other pytest processes on the pod at start: $(pgrep -c -f 'pytest' || true)"
  echo "setting threads_max running_max running_mean cpu_s wall_s cores rss_MB rc"
  EIGEN=(env XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=false" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
  TASKSET=(taskset -c "0-$((K - 1))")
  run none env
  run eigen "${EIGEN[@]}"
  run taskset "${TASKSET[@]}"
  NEQ=$(cd "$R" && uv run --frozen pytest --collect-only -q -p no:cacheprovider "${eqids[@]}" 2>/dev/null | grep -c '::')
  echo "# equality run: ${eqids[*]}; no -m filter (slow tests included); expected $NEQ passed"
  eq() { local c="$1"; shift; local P="$PB/eq_$c"; mkdir -p "$P/cache"
    ( cd "$R" && export NUZLINE_CACHE_DIR="$P/cache" NUZLINE_TEST_COMPILE_CACHE="$P/cache/jax-tests" &&
      "$@" uv run --frozen pytest -q -p no:cacheprovider "${eqids[@]}" > "$P/log" 2>&1; echo $? > "$P/rc" ) & }
  eq none env; eq eigen "${EIGEN[@]}"; eq taskset "${TASKSET[@]}"; wait
  for c in none eigen taskset; do
    s=$(grep -E ' in [0-9.]+s' "$PB/eq_$c/log" | tail -1); rc=$(cat "$PB/eq_$c/rc")
    n=$(echo "$s" | grep -oE '[0-9]+ passed' | cut -d' ' -f1); n=${n:-0}
    ok=fail; [ "$rc" = 0 ] && [ "$NEQ" -gt 0 ] && [ "$n" = "$NEQ" ] && ok=ok
    echo "eq $c $ok rc $rc passed $n of $NEQ $s"; done
} > "$O/threads_probe.txt"
V=$(awk '$1 == "eq" {eq[$2] = $3; next} NF == 9 && $1 ~ /^(none|eigen|taskset)$/ {c[$1] = $5; r[$1] = $3; rc[$1] = $9}
  END {v = "none"; split("eigen taskset", o, " ");
       for (i = 1; i <= 2; i++) {s = o[i]; if (rc[s] == "0" && rc["none"] == "0" && eq[s] == "ok" && eq["none"] == "ok" && r[s] <= 4 && c[s] <= c["none"] * 1.05) {v = s; break}}
       print v}' "$O/threads_probe.txt")
echo "THREADS=$V" | tee "$O/threads_probe.env" > /work/_threads.env
echo "verdict THREADS=$V" >> "$O/threads_probe.txt"
cat "$O/threads_probe.txt"
