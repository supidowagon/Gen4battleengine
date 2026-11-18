#!/usr/bin/env bash
# Run the fast tier split into parts, all in parallel, on the pod.
# Usage: run_parts.sh <repo dir> <parts dir> <out dir>
# parts dir holds part_NN.txt, one test file or node id per line (ids may contain spaces).
# Each part: <out>/part_NN.log (pytest output), part_NN.time (GNU time -v: peak RSS),
# part_NN.start / part_NN.end (epoch seconds), part_NN.rc (exit code).
# <out>/DONE is written when every part has exited.
# Note: background jobs of a non-interactive shell ignore SIGINT, so pytest here cannot be
# interrupted into printing its summary; stop a part with SIGTERM (to the pytest process,
# the child of `uv`) and read its outcomes by position from the -q progress dots.
set -u
export PATH="$HOME/.local/bin:$PATH"
REPO="$1"; PARTS="$2"; OUT="$3"
mkdir -p "$OUT"
cd "$REPO"
date +%s > "$OUT/ALL.start"
pids=()
for f in "$PARTS"/part_*.txt; do
  n="$(basename "$f" .txt)"
  (
    mapfile -t args < "$f"
    date +%s > "$OUT/$n.start"
    /usr/bin/time -v -o "$OUT/$n.time" \
      uv run --frozen pytest -m "not slow" -q -rfEs -p no:cacheprovider --durations=40 \
      "${args[@]}" > "$OUT/$n.log" 2>&1
    echo $? > "$OUT/$n.rc"
    date +%s > "$OUT/$n.end"
  ) &
  pids+=($!)
done
# sample the container's cgroup memory (MB) every 30 s while parts run
( while [ ! -f "$OUT/DONE" ]; do
    echo "$(date +%s) $(( $(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0) / 1048576 ))" >> "$OUT/mem_used_mb.txt"; sleep 30
  done ) &
for p in "${pids[@]}"; do wait "$p"; done
date +%s > "$OUT/ALL.end"
touch "$OUT/DONE"
