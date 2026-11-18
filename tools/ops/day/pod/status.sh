#!/usr/bin/env bash
# POD SIDE. Usage: bash /work/_bin/status.sh [run name] [log lines, default 15]
# Without a run name: one line per run plus free memory.
N="${2:-15}"
if [ -z "${1:-}" ]; then
  for d in /work/*/; do n=$(basename "$d"); [ "${n#_}" = "$n" ] && echo "$n: $(cat "$d/STATE" 2>/dev/null)"; done
  echo "free: $(bash /work/_bin/freemem.sh) GB of $(( $(cat /sys/fs/cgroup/memory.max) / 1073741824 )) GB"
  exit 0
fi
D="/work/$1"; [ -d "$D" ] || { echo "no run $1"; exit 1; }
echo "state: $(cat "$D/STATE")"
if [ -f "$D/start" ]; then
  E=$( [ -f "$D/end" ] && cat "$D/end" || date +%s ); echo "wall: $(( E - $(cat "$D/start") )) s"
fi
[ -f "$D/rc" ] && echo "exit code: $(cat "$D/rc")"
[ -f "$D/time.txt" ] && grep -E "Maximum resident|Elapsed" "$D/time.txt" | sed 's/^\s*//'
echo "--- launch steps"; grep -E "^[[0-9:]+]|rror" "$D/launch.log" 2>/dev/null | tail -n 6
echo "--- pytest.log (last $N)"; tail -n "$N" "$D/pytest.log" 2>/dev/null; echo
echo "free: $(bash /work/_bin/freemem.sh) GB"
