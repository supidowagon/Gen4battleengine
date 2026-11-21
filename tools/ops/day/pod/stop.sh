#!/usr/bin/env bash
# POD SIDE. Usage: stop.sh <run> [<root>]   (root default /work). Stops every process of the run
# <root>/<run> for good: tools/ops/day/stop_body.sh prints the exec-pod body that writes this
# script to /work/_podcheck/ and runs it (DAYBOX.md, Housekeeping, "Stop a run"). Works on runs of
# any script version. Targets, each with all its descendants (ps ppid):
#   the pids in setup.pid, runner.pid and notify.pid (setup.sh and notify.sh run under relative
#   paths, so a /work/<run> pattern on the command line misses them; a killed setup.sh can no
#   longer start the runner);
#   every process group in out/run/* (the units and their pytest processes);
#   every process whose working folder is <root>/<run> or below (/proc/<pid>/cwd): the runner's
#   memory-log loop, also when the runner died before it, and anything else the run left.
# Never PID 1, never this script or its parent. Rounds: TERM, then KILL for what is left, each
# round re-reading the targets (a process started meanwhile is caught); zombies count as gone
# (PID 1 of the pod never reaps them). Then it deletes the run's claims (<root>/.claims/<run>.*),
# writes <root>/<run>/STOPPED (epoch) and prints STOP_OK <run>, or STOP_LEFT <pids> (exit 1).
set -u
RUN="${1:-}"; R="${2:-/work}"
case "$RUN" in ''|_*|*[!a-z0-9_-]*) echo "STOP_REFUSED: run name must be [a-z0-9_-], not starting with _"; exit 2;; esac
D="$R/$RUN"; [ -d "$D" ] || { echo "STOP_REFUSED: no folder $D"; exit 2; }
me=$$; par=$PPID
ok() { case "$1" in ''|*[!0-9]*|0|1) return 1;; esac; [ "$1" != "$me" ] && [ "$1" != "$par" ]; }
# live <pid>...: those that exist and are not zombies.
live() { [ $# -gt 0 ] || return 0; ps -eo pid=,stat= | awk -v l=" $* " 'index(l, " " $1 " ") && $2 !~ /^Z/ {print $1}'; }
# desc <pid>: its descendants.
desc() { ps -eo pid=,ppid= | awk -v r="$1" '{p[$1] = $2} END {s[r] = 1
  do { c = 0; for (x in p) if (!(x in s) && (p[x] in s)) { s[x] = 1; c = 1 } } while (c)
  for (x in s) if (x != r) print x }'; }
targets() { local f p g c
  for f in setup.pid runner.pid notify.pid; do p=$(cat "$D/$f" 2>/dev/null); ok "$p" || continue; echo "$p"; desc "$p"; done
  for f in "$D"/out/run/*; do [ -f "$f" ] || continue; g=$(cat "$f" 2>/dev/null); ok "$g" || continue
    ps -eo pid=,pgid= | awk -v g="$g" '$1 == g || $2 == g {print $1}'; done
  for p in /proc/[0-9]*; do p=${p#/proc/}; c=$(readlink "/proc/$p/cwd" 2>/dev/null) || continue
    case "$c" in "$D"|"$D"/*) echo "$p";; esac; done; }
pick() { local p out=""; for p in $(targets | sort -un); do ok "$p" && out="$out $p"; done
  # shellcheck disable=SC2086
  live $out | tr '\n' ' '; }
for sig in TERM KILL KILL; do
  t=$(pick); [ -n "${t// /}" ] || break
  echo "kill -$sig $t"
  # shellcheck disable=SC2086
  kill -"$sig" $t 2>/dev/null; sleep 3
done
rm -f "$R/.claims/$RUN".*
t=$(pick)
if [ -n "${t// /}" ]; then echo "STOP_LEFT $t"; exit 1; fi
date +%s > "$D/STOPPED"; echo "STOP_OK $RUN"
