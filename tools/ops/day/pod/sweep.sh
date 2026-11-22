#!/usr/bin/env bash
# POD SIDE. Usage: sweep.sh <jobs file> [<out dir>]   (out dir default <run>/sweep_out)
# The nightly soundness sweep's driver (tools/ops/nightly_brief.md, "Soundness sweep"). Runs inside
# a run folder setup.sh has prepared: <run>/repo (checkout, venv, built tables), <run>/bin (this
# script and gate.sh). tools/ops/day/sweep_body.sh prints the exec-pod body that starts it detached.
# The jobs file has one job per line, `<pytest node id> <short name> <seed>` (tools/ops/sweep/seeds.py
# prints it; '#' lines and blank lines are skipped). Each job is one pytest process, alone:
#   NUZLINE_HYPOTHESIS_PROFILE=explore <repo>/.venv/bin/python -m pytest -p no:cacheprovider -q -x \
#     <node id> --hypothesis-seed=<seed> -m ""
# A job starts while fewer than SWEEP_P jobs of this sweep run and container memory (gate.sh cgmem,
# the same reader runner.sh uses; no readable file: the driver stops, exit 1) plus SWEEP_W GB stays
# at or under SWEEP_CAP GB. Nothing else is gated, nothing is killed: a job that fails is a result.
# Files in <out>: <short>_<seed>.log, .time (`wall=<s> rc=<n>`, and `peak_kb=<n>` when
# /usr/bin/time exists), driver.log, summary.txt (node, seed, result, wall_s, rc, peak_GB, tab
# separated; result PASS when rc is 0, FAIL otherwise, LOST when a job left no .time), and, when a
# log holds a Hypothesis falsifier, replay_in/ (tools/ops/sweep/extract.py) and replay.txt
# (tools/ops/sweep/replay.py, loads the engine). SWEEP_DONE is written last.
# Exit code: 0 unless the driver itself breaks (bad usage, no repo, no memory reader; then no SWEEP_DONE).
# Knobs (env): SWEEP_P (5), SWEEP_W (13, GB one job is expected to peak at), SWEEP_CAP (190, GB),
# SWEEP_SPACING (20 s after each start, so the next reading sees the job's growth), SWEEP_POLL (15 s
# between gate readings while waiting), SWEEP_REPLAY_S (3600, replay timeout), SWEEP_TOOLS
# (<repo>/tools/ops/sweep), CGROOT (gate.sh).
set -u
D=$(cd "$(dirname "$0")/.." && pwd)
JOBS="${1:-}"; OUT="${2:-$D/sweep_out}"; REPO="$D/repo"
P="${SWEEP_P:-5}"; W="${SWEEP_W:-13}"; CAP="${SWEEP_CAP:-190}"; SPACING="${SWEEP_SPACING:-20}"; POLL="${SWEEP_POLL:-15}"
REPLAY_S="${SWEEP_REPLAY_S:-3600}"; TOOLS="${SWEEP_TOOLS:-$REPO/tools/ops/sweep}"
[ -f "$JOBS" ] || { echo "usage: sweep.sh <jobs file> [<out dir>]" >&2; exit 1; }
for n in P W CAP; do case "${!n}" in ''|*[!0-9]*) echo "ERROR: SWEEP_$n '${!n}' is not a whole number" >&2; exit 1;; esac; done
[ -x "$REPO/.venv/bin/python" ] || { echo "ERROR: no $REPO/.venv/bin/python" >&2; exit 1; }
mkdir -p "$OUT" || exit 1
rm -f "$OUT/SWEEP_DONE"
LOG="$OUT/driver.log"; say() { echo "$(date +%s) $*" >> "$LOG"; }
. "$D/bin/gate.sh"   # cgmem
b=$(cgmem current) || { say "ERROR: no cgroup memory file; driver stops"; echo "ERROR: no cgroup memory file" >&2; exit 1; }
cd "$REPO" || exit 1
say "DRIVER_START repo=$REPO P=$P W=$W CAP=$CAP"
TIMEBIN=; [ -x /usr/bin/time ] && TIMEBIN=/usr/bin/time
: > "$OUT/jobs.done"   # the jobs this driver started, in order: <short>_<seed>
while read -r node short s rest; do
  case "$node" in ''|'#'*) continue;; esac
  case "$s" in ''|*[!0-9]*) say "SKIP bad job line: $node $short $s"; continue;; esac
  while :; do
    n=$(jobs -rp | wc -l); b=$(cgmem current) || b=
    case "$b" in ''|*[!0-9]*) say "ERROR: memory unreadable; driver stops"; wait; exit 1;; esac
    m=$(( b / 1000000000 ))
    if [ "$n" -lt "$P" ] && [ $((m + W)) -le "$CAP" ]; then break; fi
    sleep "$POLL"
  done
  j="${short}_${s}"; echo "$node $short $s" >> "$OUT/jobs.done"
  (
    t0=$(date +%s)
    if [ -n "$TIMEBIN" ]; then
      NUZLINE_HYPOTHESIS_PROFILE=explore "$TIMEBIN" -v -o "$OUT/$j.rss" "$REPO/.venv/bin/python" -m pytest -p no:cacheprovider -q -x "$node" --hypothesis-seed="$s" -m "" > "$OUT/$j.log" 2>&1 < /dev/null
    else
      NUZLINE_HYPOTHESIS_PROFILE=explore "$REPO/.venv/bin/python" -m pytest -p no:cacheprovider -q -x "$node" --hypothesis-seed="$s" -m "" > "$OUT/$j.log" 2>&1 < /dev/null
    fi
    rc=$?; t1=$(date +%s)
    kb=$(sed -n 's/.*Maximum resident set size (kbytes): *//p' "$OUT/$j.rss" 2>/dev/null | head -1)
    echo "wall=$((t1-t0)) rc=$rc${kb:+ peak_kb=$kb}" > "$OUT/$j.time"
    echo "$(date +%s) END $j rc=$rc wall=$((t1-t0))" >> "$LOG"
  ) &
  say "START $j mem=${m}GB running=$((n+1))"
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
