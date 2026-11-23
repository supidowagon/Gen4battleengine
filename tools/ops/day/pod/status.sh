#!/usr/bin/env bash
# POD SIDE. Usage: status.sh [wait seconds]. Setup state, this run's setup= and runner= state
# (always printed), collection failures, unit counts, memory, killed and crashed units, running
# units with F/E marks so far, every failing finished unit with its FAILED/ERROR lines, and the
# other runs still active on the pod (/work/*/ and /work/*/*/) plus the count of live pytest
# processes outside this run; "other active runs: none" only when both are empty. A run whose
# bin/notify.sh (notify.pid) is alive and whose results are not taken yet (no notify/taken_d) is
# listed as <name>(beacon): its results are still on offer. A run whose soundness sweep driver
# (sweep.pid, bin/sweep.sh; tools/ops/day/sweep_body.sh) is alive is listed as <name>(sweep): its
# jobs run `.venv/bin/python -m pytest`, which the pytest count below does not match. A run whose
# driver died while a job or replay group named in sweep/out/run/* lives (gate.sh sweeplive) is
# listed the same way.
# setup=alive|done|gone: setup.pid's process alive | dead with runner.pid written | dead (or no
# setup.pid) without runner.pid, a failed setup whether or not the log says SETUP_FAILED.
# runner=alive|gone|starting|none: runner.pid's process alive | dead | not alive yet while setup
# is alive | no runner.pid and setup not alive.
[ -n "${1:-}" ] && sleep "$1"
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"; . "$D/bin/gate.sh"   # cgmem: cgroup v2 or v1
grep -E 'SETUP_|units|RUNNER|rror|code' "$D/setup.log" 2>/dev/null | tail -4 | sed 's/^/setup: /'
grep -vE 'collect_rc=(0|5)$' "$D/collected.txt" 2>/dev/null | sed 's/^/COLLECT FAILED: /'
# A run is active only while its setup or runner process (setup.pid, runner.pid) is alive, runs
# a runner*.sh or setup*.sh script and has its working folder inside that run; log text alone
# never counts. A zombie has an empty cmdline, so it fails the script test.
# alive <pid file> <run folder> [<cmdline pattern>, default the runner or setup script]
alive() { local p; p=$(cat "$1" 2>/dev/null) || return 1; case "$p" in ''|*[!0-9]*) return 1;; esac
  kill -0 "$p" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE "${3:-runner[0-9]*\.sh|setup[_a-z0-9]*\.sh}" || return 1
  case "$(readlink "/proc/$p/cwd" 2>/dev/null)" in "$2"|"$2"/*) return 0;; esac; return 1; }
if alive "$D/setup.pid" "$D"; then SETUP=alive; elif [ -f "$D/runner.pid" ]; then SETUP=done; else SETUP=gone; fi
if [ -f "$D/runner.pid" ]; then
  if alive "$D/runner.pid" "$D"; then RUNNER=alive; elif [ "$SETUP" = alive ]; then RUNNER=starting; else RUNNER=gone; fi
elif [ "$SETUP" = alive ]; then RUNNER=starting; else RUNNER=none; fi
echo "setup=$SETUP runner=$RUNNER DONE=$([ -f "$O/DONE" ] && echo y || echo n)"
active=""; for d in /work/*/ /work/*/*/; do r=${d%/}
  case "$r" in "$D"|"$D"/*|/work/_*) continue;; esac; n=${r#/work/}
  if alive "$r/runner.pid" "$r"; then active="$active $n"
  elif alive "$r/setup.pid" "$r"; then active="$active $n(setup)"
  elif alive "$r/quick.pid" "$r" 'quick\.sh'; then active="$active $n(quick)"
  elif alive "$r/sweep.pid" "$r" 'sweep\.sh'; then active="$active $n(sweep)"
  elif sweeplive "$r"; then active="$active $n(sweep)"   # the driver died, its job groups live (setsid)
  elif [ ! -f "$r/notify/taken_d" ] && alive "$r/notify.pid" "$r" 'notify\.sh'; then active="$active $n(beacon)"; fi; done
# Live pytest processes whose working folder is outside this run (pgrep skips zombies: no cmdline).
np=0; for p in $(pgrep -f '/\.venv/bin/pytest'); do
  case "$(readlink "/proc/$p/cwd" 2>/dev/null)" in "$D"|"$D"/*) ;; *) np=$((np + 1));; esac; done
[ "$np" -gt 0 ] && active="$active pytest_outside_this_run=$np"
echo "other active runs:${active:- none}"
[ -f "$O/queue.txt" ] || exit 0
cd "$O"
NRUN=0; for r in run/*; do [ -e "$r" ] && [ ! -f "$(basename "$r").rc" ] && NRUN=$((NRUN + 1)); done   # with an .rc: done
echo "now $(date +%s) start $(cat ALL.start 2>/dev/null) units $(grep -c . queue.txt) done $(ls *.rc 2>/dev/null | wc -l) running $NRUN pending $(grep -c . pending.txt 2>/dev/null) killed $(cat killed.txt 2>/dev/null | wc -l) crashed $(cat crashed.txt 2>/dev/null | wc -l)"
pk=n/a; b=$(cgmem peak) && pk="$(( b / 1048576 )) MB"; mn=n/a; b=$(cgmem current) && mn="$(( b / 1048576 )) MB"
echo "mem now $mn peak $pk load $(cut -d' ' -f1 /proc/loadavg)"
for u in $(ls run 2>/dev/null); do n=$(head -c 20000 "$u.log" 2>/dev/null | grep -E '^[.sFExX]+' | tr -cd 'FE' | wc -c); [ "$n" -gt 0 ] && echo "running $u has $n F/E"; done
for r in *.rc; do [ -e "$r" ] || continue; c=$(cat "$r"); u=$(basename "$r" .rc)
  case $c in 0) ;; *) echo "BAD $u rc=$c $(grep "^$u " queue.txt | cut -d' ' -f2)"; grep -E '^(FAILED|ERROR) ' "$u.log" 2>/dev/null | cut -c1-240 | head -20;; esac; done
tail -2 "$D/runner.log"
