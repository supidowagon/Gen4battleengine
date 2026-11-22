#!/usr/bin/env bash
# POD SIDE. Usage: sweep.sh <jobs file> [<out dir>]   (out dir default <run>/sweep_out)
# The nightly soundness sweep's driver (tools/ops/nightly_brief.md, "Soundness sweep"). Runs inside
# a run folder setup.sh has prepared: <run>/repo (checkout, venv, built tables), <run>/bin (this
# script and gate.sh). tools/ops/day/sweep_body.sh prints the exec-pod body that starts it detached.
# The jobs file has one job per line, `<pytest node id> <short name> <seed>` (tools/ops/sweep/seeds.py
# prints it; '#' lines and blank lines are skipped). Each job is one pytest process, alone:
#   NUZLINE_HYPOTHESIS_PROFILE=explore <repo>/.venv/bin/python -m pytest -p no:cacheprovider -q -x \
#     <node id> --hypothesis-seed=<seed> -m ""
# A job starts only under the pod-wide gate lock RUNNER_LOCK (flock, as runner.sh; jobs start with
# it closed), while fewer than SWEEP_P jobs of this sweep run, no gate-run claim exists (gate.sh
# claim_state over CLAIMS: any live claim, a runner waiting for room, holds every start) and
# container memory (gate.sh cgmem current; no readable file: the driver stops, exit 1) plus the
# reserve of the gate-run units live on the pod (gate.sh podgate over WORK/*/out/run/* and
# WORK/*/*/out/run/*, as runner.sh) plus SWEEP_W GB stays at or under the cap. The cap is the pod shape's GATE_GB (docs/process.md section 5)
# from the container limit (gate.sh cgmem max): limit minus 40 GB, at most 88, under 200 GB or with
# no limit (`max`, a value of 2^60 or more); limit minus 56 GB, at most 200, from 200 GB. SWEEP_CAP
# set replaces it. An unreadable limit, or a cap under SWEEP_W, refuses the start (ERROR line,
# exit 1). Jobs run under nice (SWEEP_NICE). Nothing is killed: a job that fails is a result.
# The jobs are not runner units: a gate run's runner sees their memory but holds no reserve for
# their growth (about 12 GB each over minutes), so a gate run that starts during the sweep starts
# fewer units until the jobs end; its KILL_GB backstop still applies to its own units.
# Stop: `kill -- -$(cat /work/<run>/sweep.pid)` (sweep_body.sh starts the driver with setsid, so
# the pid is its process group). TERM, INT or HUP to the driver alone also stops its jobs (their
# process trees), writes <out>/SWEEP_STOPPED and exits 143/130/129, without SWEEP_DONE.
# Files in <out>: <short>_<seed>.log, .time (`wall=<s> rc=<n>`, and `peak_kb=<n>` when
# /usr/bin/time exists), driver.log, summary.txt (node, seed, result, wall_s, rc, peak_GB, tab
# separated; result PASS when rc is 0, FAIL otherwise, LOST when a job left no .time), and, when a
# log holds a Hypothesis falsifier, replay_in/ (tools/ops/sweep/extract.py) and replay.txt
# (tools/ops/sweep/replay.py, loads the engine). SWEEP_DONE is written last.
# Exit code: 0 unless the driver itself breaks (bad usage, no repo, no memory reader, no limit or
# too small a cap, a signal; then no SWEEP_DONE).
# Knobs (env): SWEEP_P (5), SWEEP_W (13, GB one job is expected to peak at), SWEEP_CAP (unset: from
# the limit, above), SWEEP_NICE (10), SWEEP_SPACING (20 s after each start, so the next reading sees
# the job's growth), SWEEP_POLL (15 s between gate readings while waiting), SWEEP_REPLAY_S (3600,
# replay timeout), SWEEP_TOOLS (<repo>/tools/ops/sweep), and as runner.sh: RUNNER_LOCK
# (/work/.runner.lock), CLAIMS (/work/.claims), CLAIM_MAX_S (3600), DEFAULT_WEIGHT_GB (12), WORK
# (/work), RESERVE_TOP, RESERVE_FULL_GB, RESERVE_PCT (gate.sh defaults), CGROOT.
set -u
D=$(cd "$(dirname "$0")/.." && pwd)
JOBS="${1:-}"; OUT="${2:-$D/sweep_out}"; REPO="$D/repo"
P="${SWEEP_P:-5}"; W="${SWEEP_W:-13}"; CAP="${SWEEP_CAP:-}"; NICE="${SWEEP_NICE:-10}"; SPACING="${SWEEP_SPACING:-20}"; POLL="${SWEEP_POLL:-15}"
REPLAY_S="${SWEEP_REPLAY_S:-3600}"; TOOLS="${SWEEP_TOOLS:-$REPO/tools/ops/sweep}"
LOCK="${RUNNER_LOCK:-/work/.runner.lock}"; CLAIMS="${CLAIMS:-/work/.claims}"; NO_W_FILE_GB="${DEFAULT_WEIGHT_GB:-12}"; WORK="${WORK:-/work}"
[ -f "$JOBS" ] || { echo "usage: sweep.sh <jobs file> [<out dir>]" >&2; exit 1; }
for n in P W NICE; do case "${!n}" in ''|*[!0-9]*) echo "ERROR: SWEEP_$n '${!n}' is not a whole number" >&2; exit 1;; esac; done
case "$CAP" in *[!0-9]*) echo "ERROR: SWEEP_CAP '$CAP' is not a whole number" >&2; exit 1;; esac
[ -x "$REPO/.venv/bin/python" ] || { echo "ERROR: no $REPO/.venv/bin/python" >&2; exit 1; }
mkdir -p "$OUT" || exit 1
rm -f "$OUT/SWEEP_DONE"
LOG="$OUT/driver.log"; say() { echo "$(date +%s) $*" >> "$LOG"; }
. "$D/bin/gate.sh"   # cgmem, podgate, claim_state (pidlive, runnerpid)
fail() { say "ERROR: $*; driver stops"; echo "ERROR: $*" >&2; exit 1; }
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
cd "$REPO" || exit 1
say "DRIVER_START repo=$REPO P=$P W=$W CAP=$CAP ($CAPSRC) nice=$NICE"
TIMEBIN=; [ -x /usr/bin/time ] && TIMEBIN=/usr/bin/time
# tree <pid>: the pid and its descendants, deepest first.
tree() { local c; for c in $(ps -eo pid=,ppid= | awk -v p="$1" '$2 == p {print $1}'); do tree "$c"; done; echo "$1"; }
stop() { trap - TERM INT HUP; local p all=""
  for p in $(jobs -p); do all="$all $(tree "$p")"; done
  [ -n "$all" ] && kill -TERM $all 2>/dev/null
  say "STOPPED by $1: jobs stopped, no SWEEP_DONE"; date +%s > "$OUT/SWEEP_STOPPED"; exit "$2"; }
rm -f "$OUT/SWEEP_STOPPED"
trap 'stop TERM 143' TERM; trap 'stop INT 130' INT; trap 'stop HUP 129' HUP
mkdir -p "${LOCK%/*}" 2>/dev/null; exec 8>> "$LOCK" || fail "cannot open the gate lock $LOCK"
: > "$OUT/jobs.done"   # the jobs this driver started, in order: <short>_<seed>
held=
while read -r node short s rest; do
  case "$node" in ''|'#'*) continue;; esac
  case "$s" in ''|*[!0-9]*) say "SKIP bad job line: $node $short $s"; continue;; esac
  # Check and start under the pod lock, as runner.sh: no runner passes its gate on the same reading.
  while :; do
    n=$(jobs -rp | wc -l)
    if [ "$n" -lt "$P" ]; then
      if flock -w 120 8; then
        b=$(cgmem current) || b=
        case "$b" in ''|*[!0-9]*) flock -u 8; say "ERROR: memory unreadable; driver stops"; wait; exit 1;; esac
        m=$(( b / 1000000000 ))
        read -r plive pres <<< "$(podgate "$WORK"/*/out/run/* "$WORK"/*/*/out/run/*)"
        read -r cs cn _ <<< "$(claim_state - 2>> "$LOG")"
        if [ "$cs" = none ] && [ $((m + pres + W)) -le "$CAP" ]; then break; fi
        flock -u 8
        why="mem ${m}GB + reserve ${pres}GB (${plive} gate units) + ${W}GB over cap ${CAP}GB"; key=cap
        [ "$cs" = none ] || { why="claim $cn"; key=$why; }
        [ "$held" = "$key" ] || { held=$key; say "HOLD $why"; }   # once per cause
      else say "gate lock $LOCK still held after 120 s"; fi
    fi
    sleep "$POLL"
  done
  held=
  j="${short}_${s}"; echo "$node $short $s" >> "$OUT/jobs.done"
  (
    t0=$(date +%s)
    if [ -n "$TIMEBIN" ]; then
      NUZLINE_HYPOTHESIS_PROFILE=explore nice -n "$NICE" "$TIMEBIN" -v -o "$OUT/$j.rss" "$REPO/.venv/bin/python" -m pytest -p no:cacheprovider -q -x "$node" --hypothesis-seed="$s" -m "" > "$OUT/$j.log" 2>&1 < /dev/null
    else
      NUZLINE_HYPOTHESIS_PROFILE=explore nice -n "$NICE" "$REPO/.venv/bin/python" -m pytest -p no:cacheprovider -q -x "$node" --hypothesis-seed="$s" -m "" > "$OUT/$j.log" 2>&1 < /dev/null
    fi
    rc=$?; t1=$(date +%s)
    kb=$(sed -n 's/.*Maximum resident set size (kbytes): *//p' "$OUT/$j.rss" 2>/dev/null | head -1)
    echo "wall=$((t1-t0)) rc=$rc${kb:+ peak_kb=$kb}" > "$OUT/$j.time"
    echo "$(date +%s) END $j rc=$rc wall=$((t1-t0))" >> "$LOG"
  ) 8>&- &
  flock -u 8
  say "START $j mem=${m}GB reserve=${pres}GB running=$((n+1))"
  sleep "$SPACING"
done < "$JOBS"
wait
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
# replay of the falsifiers (the engine loads here; the jobs are over, so the pod has the memory)
fails=$(grep -l "Failing test case" "$OUT"/*.log 2>/dev/null | grep -v '/driver\.log$' | wc -l)
if [ "$fails" -gt 0 ]; then
  say "REPLAY_START logs=$fails"
  rm -rf "$OUT/replay_in"; mkdir -p "$OUT/replay_in"
  RUNNER=; command -v timeout > /dev/null && RUNNER="timeout -k 30 $REPLAY_S"
  "$REPO/.venv/bin/python" "$TOOLS/extract.py" "$OUT" "$OUT/replay_in" > "$OUT/extract.txt" 2>&1
  $RUNNER "$REPO/.venv/bin/python" "$TOOLS/replay.py" "$REPO" "$OUT/replay_in" > "$OUT/replay.txt" 2>&1
  say "REPLAY_END rc=$?"
fi
date +%s > "$OUT/SWEEP_DONE"
say "SWEEP_DONE"
exit 0
