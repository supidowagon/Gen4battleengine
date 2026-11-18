#!/usr/bin/env bash
# One-line-per-part progress: last pytest percentage, exit code if done; then free memory.
OUT="${1:-/root/out}"
for l in "$OUT"/part_*.log; do
  n="$(basename "$l" .log)"
  pct="$(grep -o '\[ *[0-9]*%\]' "$l" | tail -1)"
  rc="$(cat "$OUT/$n.rc" 2>/dev/null)"
  echo "$n ${pct:-[0%]} ${rc:+rc=$rc}"
done
echo "cgroup mem $(( $(cat /sys/fs/cgroup/memory.current) / 1073741824 ))G of $(( $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0) / 1073741824 ))G, peak $(( $(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0) / 1073741824 ))G"
[ -f "$OUT/DONE" ] && echo DONE
