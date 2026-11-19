#!/usr/bin/env bash
# POD SIDE. Usage: status.sh [wait seconds]. Setup state, this run's setup= and runner= state
# (always printed), collection failures, unit counts, memory, killed and crashed units, running
# units with F/E marks so far, every failing finished unit with its FAILED/ERROR lines, and the
# other runs still active on the pod.
# setup=alive|done|gone: setup.pid's process alive | dead with runner.pid written | dead (or no
# setup.pid) without runner.pid, a failed setup whether or not the log says SETUP_FAILED.
# runner=alive|gone|starting|none: runner.pid's process alive | dead | not alive yet while setup
# is alive | no runner.pid and setup not alive.
[ -n "${1:-}" ] && sleep "$1"
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"
grep -E 'SETUP_|units|RUNNER|rror|code' "$D/setup.log" 2>/dev/null | tail -4 | sed 's/^/setup: /'
grep -vE 'collect_rc=(0|5)$' "$D/collected.txt" 2>/dev/null | sed 's/^/COLLECT FAILED: /'
# A run is active only while its setup or runner process (setup.pid, runner.pid) is alive, runs
# setup.sh or runner.sh and has its working folder inside that run; log text alone never counts.
alive() { local p; p=$(cat "$1" 2>/dev/null) || return 1; case "$p" in ''|*[!0-9]*) return 1;; esac
  kill -0 "$p" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE 'runner\.sh|setup\.sh' || return 1
  case "$(readlink "/proc/$p/cwd" 2>/dev/null)" in "$2"|"$2"/*) return 0;; esac; return 1; }
if alive "$D/setup.pid" "$D"; then SETUP=alive; elif [ -f "$D/runner.pid" ]; then SETUP=done; else SETUP=gone; fi
if [ -f "$D/runner.pid" ]; then
  if alive "$D/runner.pid" "$D"; then RUNNER=alive; elif [ "$SETUP" = alive ]; then RUNNER=starting; else RUNNER=gone; fi
elif [ "$SETUP" = alive ]; then RUNNER=starting; else RUNNER=none; fi
echo "setup=$SETUP runner=$RUNNER DONE=$([ -f "$O/DONE" ] && echo y || echo n)"
active=""; for d in /work/*/; do r=${d%/}; [ "$r" = "$D" ] && continue; case "$(basename "$r")" in _*) continue;; esac
  if alive "$r/runner.pid" "$r"; then active="$active $(basename "$r")"
  elif alive "$r/setup.pid" "$r"; then active="$active $(basename "$r")(setup)"; fi; done
echo "other active runs:${active:- none}"
[ -f "$O/queue.txt" ] || exit 0
cd "$O"
echo "now $(date +%s) start $(cat ALL.start 2>/dev/null) units $(grep -c . queue.txt) done $(ls *.rc 2>/dev/null | wc -l) running $(ls run 2>/dev/null | wc -l) pending $(grep -c . pending.txt 2>/dev/null) killed $(cat killed.txt 2>/dev/null | wc -l) crashed $(cat crashed.txt 2>/dev/null | wc -l)"
pk=n/a; [ -r /sys/fs/cgroup/memory.peak ] && pk="$(( $(cat /sys/fs/cgroup/memory.peak) / 1048576 )) MB"
echo "mem now $(( $(cat /sys/fs/cgroup/memory.current) / 1048576 )) MB peak $pk load $(cut -d' ' -f1 /proc/loadavg)"
for u in $(ls run 2>/dev/null); do n=$(head -c 20000 "$u.log" 2>/dev/null | grep -E '^[.sFExX]+' | tr -cd 'FE' | wc -c); [ "$n" -gt 0 ] && echo "running $u has $n F/E"; done
for r in *.rc; do [ -e "$r" ] || continue; c=$(cat "$r"); u=$(basename "$r" .rc)
  case $c in 0) ;; *) echo "BAD $u rc=$c $(grep "^$u " queue.txt | cut -d' ' -f2)"; grep -E '^(FAILED|ERROR) ' "$u.log" 2>/dev/null | cut -c1-240 | head -20;; esac; done
tail -2 "$D/runner.log"
