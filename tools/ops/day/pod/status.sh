#!/usr/bin/env bash
# POD SIDE. Usage: status.sh [wait seconds]. Setup state, collection failures, unit counts,
# memory, killed and crashed units, running units with F/E marks so far, every failing finished
# unit with its FAILED/ERROR lines, and the other runs still active on the pod.
[ -n "${1:-}" ] && sleep "$1"
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"
grep -E 'SETUP_|units|RUNNER|rror|code' "$D/setup.log" 2>/dev/null | tail -4 | sed 's/^/setup: /'
grep -vE 'collect_rc=(0|5)$' "$D/collected.txt" 2>/dev/null | sed 's/^/COLLECT FAILED: /'
active=""; for d in /work/*/; do r=${d%/}; [ "$r" = "$D" ] && continue; case "$(basename "$r")" in _*) continue;; esac
  if [ -f "$r/out/queue.txt" ] && [ ! -f "$r/out/DONE" ]; then active="$active $(basename "$r")"
  elif [ -f "$r/setup.log" ] && [ ! -f "$r/out/queue.txt" ] && ! grep -q 'SETUP_END\|exit\|rror' "$r/setup.log"; then active="$active $(basename "$r")(setup)"; fi; done
echo "other active runs:${active:- none}"
[ -f "$O/queue.txt" ] || exit 0
cd "$O"
echo "now $(date +%s) start $(cat ALL.start 2>/dev/null) units $(grep -c . queue.txt) done $(ls *.rc 2>/dev/null | wc -l) running $(ls run 2>/dev/null | wc -l) pending $(grep -c . pending.txt 2>/dev/null) killed $(cat killed.txt 2>/dev/null | wc -l) crashed $(cat crashed.txt 2>/dev/null | wc -l) DONE=$([ -f DONE ] && echo y || echo n)"
echo "mem now $(( $(cat /sys/fs/cgroup/memory.current) / 1048576 )) MB peak $(( $(cat /sys/fs/cgroup/memory.peak) / 1048576 )) MB load $(cut -d' ' -f1 /proc/loadavg)"
for u in $(ls run 2>/dev/null); do n=$(head -c 20000 "$u.log" 2>/dev/null | grep -E '^[.sFExX]+' | tr -cd 'FE' | wc -c); [ "$n" -gt 0 ] && echo "running $u has $n F/E"; done
for r in *.rc; do [ -e "$r" ] || continue; c=$(cat "$r"); u=$(basename "$r" .rc)
  case $c in 0) ;; *) echo "BAD $u rc=$c $(grep "^$u " queue.txt | cut -d' ' -f2)"; grep -E '^(FAILED|ERROR) ' "$u.log" 2>/dev/null | cut -c1-240 | head -20;; esac; done
tail -2 "$D/runner.log"
