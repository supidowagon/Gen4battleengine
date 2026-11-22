#!/usr/bin/env bash
# POD SIDE. Usage: sweep.sh <jobs file> [<out dir>]   (out dir default <run>/sweep_out; relative
# paths are taken from the caller's folder)
# The nightly soundness sweep's driver (tools/ops/nightly_brief.md, "Soundness sweep"). Runs inside
# a run folder setup.sh has prepared: <run>/repo (checkout, venv, built tables), <run>/bin (this
# script and gate.sh); the run folder lies in WORK. tools/ops/day/sweep_body.sh prints the
# exec-pod body that starts it detached.
# The jobs file has one job per line, `<pytest node id> <short name> <seed>` (tools/ops/sweep/seeds.py
# prints it; '#' lines and blank lines are skipped). Each job is one pytest process, alone:
#   NUZLINE_HYPOTHESIS_PROFILE=explore <repo>/.venv/bin/python -m pytest -p no:cacheprovider -q -x #     <node id> --hypothesis-seed=<seed> -m ""
# under nice (SWEEP_NICE) and `timeout --foreground -k 60 SWEEP_JOB_S` (rc 124, or 137 after the
# KILL, is a failed job; the sweep goes on).
# Each job is a gate unit, as runner.sh's units and quick.sh's q: it runs in its own process group
# (setsid), <run>/sweep/out/run/<short>_<seed> holds that group's id and <run>/sweep/out/<short>_<seed>.w
# its weight SWEEP_W, so every podgate over WORK/*/*/out/run/* (each runner's start gate, quick.sh,
# and this driver) reserves the gap between SWEEP_W and the job's RSS while it grows. The run file
# goes when the job's group is gone. A runner's KILL_GB backstop kills only its own run's units.
# A job starts only under the pod-wide gate lock RUNNER_LOCK (flock, as runner.sh; jobs start with
# it closed), while fewer than SWEEP_P jobs of this sweep run, no gate-run claim exists (gate.sh
# claim_state over CLAIMS: any live claim, a runner waiting for room, holds every start) and
# container memory (gate.sh cgmem current; no readable file: the driver stops, exit 1) plus the
# reserve of the units live on the pod (gate.sh podgate over WORK/*/out/run/* and
# WORK/*/*/out/run/*, this sweep's own jobs included) plus SWEEP_W GB stays at or under the cap.
# The cap is the pod shape's GATE_GB (docs/process.md section 5) from the container limit (gate.sh
# cgmem max): limit minus 40 GB, at most 88, under 200 GB or with no limit (`max`, a value of 2^60
# or more); limit minus 56 GB, at most 200, from 200 GB. SWEEP_CAP set replaces it. Refused at the
# start (ERROR line, exit 1): an unreadable limit, SWEEP_P under 1, SWEEP_P x SWEEP_W over the cap
# (the sweep alone would not fit), an out dir that already holds a sweep (jobs.done), a live job of
# another sweep of this run in <run>/sweep/out/run/.
# Stop: `kill -TERM -- -<pid>` with the pid of /work/<run>/sweep.pid (sweep_body.sh starts the
# driver with setsid, so the pid is its process group; refuse an empty, non-numeric, 0 or 1 pid).
# TERM, INT or HUP to the driver alone also stops its jobs (their process groups, as recorded in
# the run files, and their process trees), writes <out>/SWEEP_STOPPED and exits 143/130/129,
# without SWEEP_DONE.
# Files in <out>: jobs.txt (the job lines this sweep was given: collect.py's expected count),
# jobs.done (the jobs started, in order), <short>_<seed>.log, .time (`wall=<s> rc=<n>`, and
# `peak_kb=<n>` when /usr/bin/time exists), driver.log, summary.txt (node, seed, result, wall_s, rc,
# peak_GB, tab separated; result PASS when rc is 0, FAIL otherwise, LOST when a job left no .time),
# and, when a log of this sweep holds a Hypothesis falsifier (the markers of
# tools/ops/sweep/extract.py MARKERS), replay_in/ (extract.py) and replay.txt (replay.py, loads the
# engine). SWEEP_DONE is written last.
# Exit code: 0 unless the driver itself breaks (a refusal above, bad usage, no repo, no memory
# reader, a signal; then no SWEEP_DONE).
# Knobs (env): SWEEP_P (5), SWEEP_W (13, GB one job is expected to peak at), SWEEP_CAP (unset: from
# the limit, above), SWEEP_NICE (10), SWEEP_JOB_S (3600, one job's time limit), SWEEP_SPACING (20 s
# after each start), SWEEP_POLL (15 s between gate readings while waiting), SWEEP_REPLAY_S (3600,
# replay timeout), SWEEP_TOOLS (<repo>/tools/ops/sweep), and as runner.sh: RUNNER_LOCK
# (/work/.runner.lock), CLAIMS (/work/.claims), CLAIM_MAX_S (3600), DEFAULT_WEIGHT_GB (12), WORK
# (/work), RESERVE_TOP, RESERVE_FULL_GB, RESERVE_PCT (gate.sh defaults), CGROOT.
set -u
D=$(cd "$(dirname "$0")/.." && pwd)
JOBS="${1:-}"; OUT="${2:-$D/sweep_out}"; REPO="$D/repo"; U="$D/sweep/out"
# The driver works from $REPO: relative paths are the caller's.
case "$JOBS" in ''|/*) ;; *) JOBS="$PWD/$JOBS";; esac
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT";; esac
P="${SWEEP_P:-5}"; W="${SWEEP_W:-13}"; CAP="${SWEEP_CAP:-}"; NICE="${SWEEP_NICE:-10}"; SPACING="${SWEEP_SPACING:-20}"; POLL="${SWEEP_POLL:-15}"
JOB_S="${SWEEP_JOB_S:-3600}"; REPLAY_S="${SWEEP_REPLAY_S:-3600}"; TOOLS="${SWEEP_TOOLS:-$REPO/tools/ops/sweep}"
LOCK="${RUNNER_LOCK:-/work/.runner.lock}"; CLAIMS="${CLAIMS:-/work/.claims}"; NO_W_FILE_GB="${DEFAULT_WEIGHT_GB:-12}"; WORK="${WORK:-/work}"
[ -n "$JOBS" ] && [ -f "$JOBS" ] || { echo "usage: sweep.sh <jobs file> [<out dir>]" >&2; exit 1; }
for n in P W NICE JOB_S; do case "${!n}" in ''|*[!0-9]*) echo "ERROR: SWEEP_$n '${!n}' is not a whole number" >&2; exit 1;; esac; done
[ "$P" -ge 1 ] || { echo "ERROR: SWEEP_P '$P' is under 1" >&2; exit 1; }
case "$CAP" in *[!0-9]*) echo "ERROR: SWEEP_CAP '$CAP' is not a whole number" >&2; exit 1;; esac
[ -x "$REPO/.venv/bin/python" ] || { echo "ERROR: no $REPO/.venv/bin/python" >&2; exit 1; }
[ "${D%/*}" = "${WORK%/}" ] || { echo "ERROR: run folder $D is not in WORK $WORK (its jobs would not count in the pod's gate)" >&2; exit 1; }
[ -f "$OUT/jobs.done" ] && { echo "ERROR: $OUT already holds a sweep (jobs.done); give a new out dir" >&2; exit 1; }
mkdir -p "$OUT" "$U/run" || exit 1
rm -f "$OUT/SWEEP_DONE"
LOG="$OUT/driver.log"; say() { echo "$(date +%s) $*" >> "$LOG"; }
. "$D/bin/gate.sh"   # cgmem, podgate, claim_state (pidlive, runnerpid)
fail() { say "ERROR: $*; driver stops"; echo "ERROR: $*" >&2; exit 1; }
# grouplive <pgid>: the process group has a member that is not a zombie (podgate's rule).
grouplive() { case "$1" in ''|*[!0-9]*|0|1) return 1;; esac
  ps -eo pgid=,stat= | awk -v g="$1" '$1 == g && $2 !~ /^Z/ {f = 1} END {exit !f}'; }
for f in "$U"/run/*; do [ -f "$f" ] || continue
  grouplive "$(cat "$f" 2>/dev/null)" && fail "a job of another sweep of this run is live ($f)"
  rm -f "$f"; done
reserve_knobs || fail "bad RESERVE_ knob"
b=$(cgmem current) || fail "no cgroup memory file"
# The cap: the shape's GATE_GB from the container limit (docs/process.md section 5).
lim=$(cgmem max) || lim=
case "$lim" in
  max) lgb=max; ccap=88;;
  ''|*[!0-9]*) fail "container memory limit unreadable (gate.sh cgmem max: '$lim')";;
  *) if [ "${#lim}" -ge 19 ] || [ "$lim" -ge 1152921504606846976 ]; then lgb=max; ccap=88
     else lgb=$(( lim / 1000000000 ))
       if [ "$lim" -ge 200000000000 ]; then ccap=$(( lgb - 56 )); [ "$ccap" -le 200 ] || ccap=200
       else ccap=$(( lgb - 40 )); [ "$ccap" -le 88 ] || ccap=88; fi; fi;;
esac
CAPSRC="limit $lgb"; [ "$lgb" = max ] || CAPSRC="limit ${lgb}GB"
[ -n "$CAP" ] && CAPSRC="SWEEP_CAP" || CAP=$ccap
[ "$CAP" -ge "$W" ] || fail "cap ${CAP}GB ($CAPSRC) is under one job's weight SWEEP_W ${W}GB"
[ $(( P * W )) -le "$CAP" ] || fail "SWEEP_P ${P} x SWEEP_W ${W}GB = $(( P * W ))GB is over the cap ${CAP}GB ($CAPSRC)"
tr -d '\r' < "$JOBS" | grep -v '^[[:space:]]*\(#\|$\)' > "$OUT/jobs.txt"   # collect.py's expected count
cd "$REPO" || exit 1
say "DRIVER_START repo=$REPO P=$P W=$W CAP=$CAP ($CAPSRC) nice=$NICE job_s=$JOB_S jobs=$(grep -c . "$OUT/jobs.txt")"
TIMEBIN=; [ -x /usr/bin/time ] && TIMEBIN=/usr/bin/time
TO=; command -v timeout > /dev/null && TO=timeout
# One job, run as `setsid bash -c "$JOB" sweep-job <args>` (its own process group). timeout runs
# with --foreground so that it stays in that group (without it, timeout makes a group of its own,
# which podgate would not sum and stop() would not reach).
JOB='O=$1 j=$2 node=$3 s=$4 repo=$5 nice=$6 tb=$7 to=$8 js=$9; t0=$(date +%s)
tm=(); [ -n "$tb" ] && tm=("$tb" -v -o "$O/$j.rss")
tl=(); [ -n "$to" ] && tl=("$to" --foreground -k 60 "$js")
NUZLINE_HYPOTHESIS_PROFILE=explore nice -n "$nice" "${tm[@]}" "${tl[@]}" "$repo/.venv/bin/python" -m pytest -p no:cacheprovider -q -x "$node" --hypothesis-seed="$s" -m "" > "$O/$j.log" 2>&1 < /dev/null
rc=$?; t1=$(date +%s)
kb=$(sed -n "s/.*Maximum resident set size (kbytes): *//p" "$O/$j.rss" 2>/dev/null | head -1)
echo "wall=$((t1-t0)) rc=$rc${kb:+ peak_kb=$kb}" > "$O/$j.time"
echo "$(date +%s) END $j rc=$rc wall=$((t1-t0))" >> "$O/driver.log"'
# reap [final]: a job's run file goes once its process group is gone. A group whose leader ended
# but whose other members live on (a test's child) gets TERM, and with "final" up to 5 s later KILL.
reap() { local f g k
  for f in "$U"/run/*; do [ -f "$f" ] || continue; g=$(cat "$f" 2>/dev/null)
    case "$g" in ''|*[!0-9]*|0|1) rm -f "$f"; continue;; esac
    pidlive "$g" && continue
    if grouplive "$g"; then say "LEFTOVER ${f##*/}: group $g outlived its job; TERM"; kill -TERM -- "-$g" 2>/dev/null
      [ "${1:-}" = final ] || continue
      for k in 1 2 3 4 5 6 7 8 9 10; do grouplive "$g" || break; sleep 0.5; done
      kill -KILL -- "-$g" 2>/dev/null; fi
    grouplive "$g" || rm -f "$f"; done; }
# tree <pid>: the pid and its descendants, deepest first.
tree() { local c; for c in $(ps -eo pid=,ppid= | awk -v p="$1" '$2 == p {print $1}'); do tree "$c"; done; echo "$1"; }
stop() { trap - TERM INT HUP; local p f g all="" gs=""
  for f in "$U"/run/*; do [ -f "$f" ] || continue; g=$(cat "$f" 2>/dev/null)
    case "$g" in ''|*[!0-9]*|0|1) ;; *) gs="$gs $g"; kill -TERM -- "-$g" 2>/dev/null;; esac; done
  for p in $(jobs -p); do all="$all $(tree "$p")"; done   # a job whose run file is not written yet
  [ -n "$all" ] && kill -TERM $all 2>/dev/null
  for p in 1 2 3 4 5 6 7 8 9 10; do f=; for g in $gs; do grouplive "$g" && f=y; done; [ -z "$f" ] && break; sleep 0.5; done
  for g in $gs; do kill -KILL -- "-$g" 2>/dev/null; done
  rm -f "$U"/run/*
  say "STOPPED by $1: jobs stopped, no SWEEP_DONE"; date +%s > "$OUT/SWEEP_STOPPED"; exit "$2"; }
rm -f "$OUT/SWEEP_STOPPED"
trap 'stop TERM 143' TERM; trap 'stop INT 130' INT; trap 'stop HUP 129' HUP
mkdir -p "${LOCK%/*}" 2>/dev/null; exec 8>> "$LOCK" || fail "cannot open the gate lock $LOCK"
: > "$OUT/jobs.done"   # the jobs this driver started, in order: <node> <short> <seed>
held=
while read -r node short s rest; do
  case "$node" in ''|'#'*) continue;; esac
  case "$s" in ''|*[!0-9]*) say "SKIP bad job line: $node $short $s"; continue;; esac
  # Check and start under the pod lock, as runner.sh: no runner passes its gate on the same reading.
  while :; do
    reap
    n=$(jobs -rp | wc -l)
    if [ "$n" -lt "$P" ]; then
      if flock -w 120 8; then
        b=$(cgmem current) || b=
        case "$b" in ''|*[!0-9]*) flock -u 8; say "ERROR: memory unreadable; driver stops"; stop ERROR 1;; esac
        m=$(( b / 1000000000 ))
        read -r plive pres <<< "$(podgate "$WORK"/*/out/run/* "$WORK"/*/*/out/run/*)"
        case "$plive$pres" in ''|*[!0-9]*) flock -u 8; say "ERROR: the gate reserve is not a number; driver stops"; stop ERROR 1;; esac
        read -r cs cn _ <<< "$(claim_state - 2>> "$LOG")"
        if [ "$cs" = none ] && [ $((m + pres + W)) -le "$CAP" ]; then break; fi
        flock -u 8
        why="mem ${m}GB + reserve ${pres}GB (${plive} units) + ${W}GB over cap ${CAP}GB"; key=cap
        [ "$cs" = none ] || { why="claim $cn"; key=$why; }
        [ "$held" = "$key" ] || { held=$key; say "HOLD $why"; }   # once per cause
      else say "gate lock $LOCK still held after 120 s"; fi
    fi
    sleep "$POLL"
  done
  held=
  j="${short}_${s}"; echo "$node $short $s" >> "$OUT/jobs.done"; echo "$W" > "$U/$j.w"
  setsid bash -c "$JOB" sweep-job "$OUT" "$j" "$node" "$s" "$REPO" "$NICE" "$TIMEBIN" "$TO" "$JOB_S" 8>&- &
  g=$!; echo "$g" > "$U/run/.$j.tmp" && mv -f "$U/run/.$j.tmp" "$U/run/$j"   # a unit in every podgate from here
  flock -u 8
  say "START $j pgid=$g mem=${m}GB reserve=${pres}GB running=$((n+1))"
  sleep "$SPACING"
done < "$JOBS"
wait
reap final
say "ALL_JOBS_DONE"
# summary.txt, in job order
{ printf 'node\tseed\tresult\twall_s\trc\tpeak_GB\n'
  while read -r node short s; do
    t=$(cat "$OUT/${short}_${s}.time" 2>/dev/null) || { printf '%s\t%s\tLOST\t-\t-\t-\n' "$short" "$s"; continue; }
    wall=$(echo "$t" | sed -n 's/.*wall=\([0-9]*\).*/\1/p'); rc=$(echo "$t" | sed -n 's/.*rc=\([0-9]*\).*/\1/p')
    kb=$(echo "$t" | sed -n 's/.*peak_kb=\([0-9]*\).*/\1/p'); res=FAIL; [ "$rc" = 0 ] && res=PASS
    gb=-; [ -n "$kb" ] && gb=$(awk -v k="$kb" 'BEGIN { printf "%.2f", k / 1048576 }')
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$short" "$s" "$res" "${wall:--}" "${rc:--}" "$gb"
  done < "$OUT/jobs.done"; } > "$OUT/summary.txt"
# replay of the falsifiers (the engine loads here; the jobs are over, so the pod has the memory).
# Only this sweep's logs (jobs.done); MARKS are extract.py's MARKERS (tests/tools/test_sweep.py
# checks that they match).
MARKS='Failing test case|Falsifying example'
fails=0
while read -r node short s; do
  grep -qE "$MARKS" "$OUT/${short}_${s}.log" 2>/dev/null && fails=$((fails + 1))
done < "$OUT/jobs.done"
if [ "$fails" -gt 0 ]; then
  say "REPLAY_START logs=$fails"
  rm -rf "$OUT/replay_in"; mkdir -p "$OUT/replay_in"
  RUNNER=; [ -n "$TO" ] && RUNNER="timeout -k 30 $REPLAY_S"
  "$REPO/.venv/bin/python" "$TOOLS/extract.py" "$OUT" "$OUT/replay_in" > "$OUT/extract.txt" 2>&1
  $RUNNER "$REPO/.venv/bin/python" "$TOOLS/replay.py" "$REPO" "$OUT/replay_in" > "$OUT/replay.txt" 2>&1
  say "REPLAY_END rc=$?"
fi
date +%s > "$OUT/SWEEP_DONE"
say "SWEEP_DONE"
exit 0
