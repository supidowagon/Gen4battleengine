#!/usr/bin/env bash
# POD SIDE. Usage: stop.sh <run> [<root>]   (root default /work). Stops every process of the run
# <root>/<run> for good: tools/ops/day/stop_body.sh prints the exec-pod body that writes this
# script to /work/_podcheck/ and runs it (DAYBOX.md, Housekeeping, "Stop a run"). Works on runs of
# any script version. Targets:
#   the pids in setup.pid, runner.pid, notify.pid and quick.pid (a quick lane run: the exec-pod
#   body's shell, whose child is bin/quick.sh), each with all its descendants (ps ppid)
#   (setup.sh and notify.sh run under relative paths, so a /work/<run> pattern on the command line
#   misses them; a killed setup.sh can no longer start the runner);
#   the members of every process group in out/run/* (the units and their pytest processes);
# A pid-file pid and a group member count only when they belong to this run (belongs: their
# /proc/<pid>/cwd is <root>/<run> or below, as setup.sh and notify.sh (cd "$D"), the runner
# (cd "$D/out") and unit.sh (cd "$D/repo") are, or their cmdline holds "<root>/<run>/bin/"; for a
# quick lane run q-<stream>-*: also a cwd in <root>/_lane/<stream>/repo, where its unit q runs): a
# live run's setup.pid names a dead setup.sh whose pid the kernel may have given to another run's
# process. One that does not belong is skipped and printed ("skip <pid> <source>: not this run's").
#   every process whose working folder is <root>/<run> or below (/proc/<pid>/cwd): the runner's
#   memory-log loop, also when the runner died before it, and anything else the run left.
# Never PID 1, never this script or its parent. Rounds: TERM, then KILL for what is left, each
# round re-reading the targets (a process started meanwhile is caught); zombies count as gone
# (PID 1 of the pod never reaps them). Then it deletes the run's claims (<root>/.claims/<run>.*),
# writes <root>/<run>/STOPPED (epoch) and prints STOP_OK <run>, or STOP_LEFT <pids> (exit 1).
# Runs nested under <root>/<run>/ (folders <root>/<run>/<x>/, which podcheck.sh and status.sh list
# as runs) are stopped with it, by the working-folder rule; a nested run cannot be named here.
set -u
RUN="${1:-}"; R="${2:-/work}"
case "$RUN" in ''|_*|*[!a-z0-9_-]*) echo "STOP_REFUSED: run name must be [a-z0-9_-], not starting with _"; exit 2;; esac
D="$R/$RUN"; [ -d "$D" ] || { echo "STOP_REFUSED: no folder $D"; exit 2; }
me=$$; par=$PPID
# A quick lane run's unit runs in its stream's lane repo (bin/quick.sh; one quick run per stream at a time).
LANE=""; case "$RUN" in q-*) s=${RUN#q-}; s=${s%%-*}; [ -n "$s" ] && LANE="$R/_lane/$s/repo";; esac
ok() { case "$1" in ''|*[!0-9]*|0|1) return 1;; esac; [ "$1" != "$me" ] && [ "$1" != "$par" ]; }
# live <pid>...: those that exist and are not zombies.
live() { [ $# -gt 0 ] || return 0; ps -eo pid=,stat= | awk -v l=" $* " 'index(l, " " $1 " ") && $2 !~ /^Z/ {print $1}'; }
# desc <pid>: its descendants.
desc() { ps -eo pid=,ppid= | awk -v r="$1" '{p[$1] = $2} END {s[r] = 1
  do { c = 0; for (x in p) if (!(x in s) && (p[x] in s)) { s[x] = 1; c = 1 } } while (c)
  for (x in s) if (x != r) print x }'; }
# belongs <pid>: its working folder is the run's folder or below, or its cmdline names "$D/bin/".
belongs() { local c; c=$(readlink "/proc/$1/cwd" 2>/dev/null) && case "$c" in "$D"|"$D"/*) return 0;; esac
  [ -n "$LANE" ] && [ -n "$c" ] && case "$c" in "$LANE"|"$LANE"/*) return 0;; esac
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -qF "$D/bin/"; }
targets() { local f p g c
  for f in setup.pid runner.pid notify.pid quick.pid; do p=$(cat "$D/$f" 2>/dev/null); ok "$p" && [ -n "$(live "$p")" ] || continue
    if belongs "$p"; then echo "$p"; desc "$p"; else echo "skip $p $f: not this run's" >&2; fi; done
  for f in "$D"/out/run/*; do [ -f "$f" ] || continue; g=$(cat "$f" 2>/dev/null); ok "$g" || continue
    for p in $(ps -eo pid=,pgid= | awk -v g="$g" '$1 == g || $2 == g {print $1}'); do
      if belongs "$p"; then echo "$p"; else echo "skip $p out/run/${f##*/}: not this run's" >&2; fi; done; done
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
