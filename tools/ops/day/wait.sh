#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: wait.sh <run> [--until done|first-failure] [--timeout <hours>] [--every <s>]
# Blocks until the pod run <run> (sent by pack.sh, started by its exec-pod body) is over or, with
# --until first-failure, has its first failure. Needs no pod exec and no model turn: the pod's
# bin/notify.sh offers two beacons with `runpodctl send --code <code>`, each under its own random
# code from $RENT/out/<run>/meta.env (drawn by pack.sh): WATCH_F = firstfail.txt (status.sh text at
# the first failure), WATCH_D = results.tgz (fetch.sh --pack, once the run is over). A croc code
# ends in the relay index the sender picked, so every --every seconds (default 120) this tries
# `runpodctl receive <code>-<i>` for i = 0, 1, ..., at most 30 s each. The sweep ends at the end of
# the relay list ("relay index <i> not found") or when the relay list cannot be fetched ("could not
# get list of relays", HTTP 429 included); any other answer (room not ready, an unreachable relay)
# is logged unless it is "room not ready", and the sweep goes on to the next index.
# Each round it also reads pack.sh's $RENT/out/<run>/send.log: a failed payload send ("croc: send:"
# or an "error" line) before any beacon was taken exits 4.
# Done: results.tgz is unpacked by collect.sh --local into $RENT/out/<run>/results/. A run already
# collected (results/agg.txt with a RESULT line) is reported at once.
# Prints one line WAIT: <state>, then the status text or the summary of agg.txt.
# Exit: 0 run over, RESULT: PASS | 1 run over, RESULT: FAIL | 2 first failure (run still going;
# wait.sh <run> again for the end) | 3 timeout (default 8 h) | 4 usage, lock or transfer error.
# One wait.sh per run at a time (lock $RENT/out/<run>/watch/lock; a lock whose PID is dead is taken over).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# common.sh and check_run_name exit 2 on failure; 2 means "first failure" here, so test them in subshells.
( . "$HERE/../common.sh" ) || { echo "WAIT: ERROR tools/ops/common.sh failed"; exit 4; }
. "$HERE/../common.sh"
USAGE="usage: wait.sh <run> [--until done|first-failure] [--timeout <hours>] [--every <s>]"
RUN="${1:-}"; [ -n "$RUN" ] || { echo "WAIT: ERROR $USAGE"; exit 4; }
( check_run_name "$RUN" ) || { echo "WAIT: ERROR bad run name"; exit 4; }; shift
UNTIL=done; HOURS=8; EVERY=120
while [ $# -gt 0 ]; do
  case "$1" in --until|--timeout|--every) [ $# -ge 2 ] || { echo "WAIT: ERROR $1 needs a value"; exit 4; } ;; esac
  case "$1" in
    --until) UNTIL="$2"; shift 2 ;;
    --timeout) HOURS="$2"; shift 2 ;;
    --every) EVERY="$2"; shift 2 ;;
    *) echo "WAIT: ERROR unknown argument $1; $USAGE"; exit 4 ;; esac; done
case "$UNTIL" in done|first-failure) ;; *) echo "WAIT: ERROR --until done|first-failure"; exit 4;; esac
echo "$HOURS" | grep -qE '^[0-9]+(\.[0-9]+)?$' || { echo "WAIT: ERROR --timeout <hours>"; exit 4; }
echo "$EVERY" | grep -qE '^[1-9][0-9]*$' || { echo "WAIT: ERROR --every <seconds>"; exit 4; }
P="$RENT/out/$RUN"; A="$P/results/agg.txt"

# over: print the result of the collected run and exit 0 (PASS) or 1.
over() { local R; R=$(grep '^RESULT:' "$A" 2>/dev/null | tail -1)
  [ -n "$R" ] || R="RESULT: FAIL (agg.txt has no RESULT line)"
  echo "WAIT: RUN OVER, $R (full report: $A)"
  # The summary: per-file table, collection failures, crashed or killed units, wall and peak,
  # failures; the per-unit list, slowest tests and skips stay in the file.
  awk '/^--- per unit/ || /^--- slowest/ || /^--- skips/ {skip = 1; next} /^--- / {skip = 0} !skip' "$A" 2>/dev/null
  grep -E '^(SETUP_FAILED rc=|ERROR|RECEIVE_FAILED|UNPACK_FAILED)' "$P/results/setup.log" 2>/dev/null | tail -5 | sed 's/^/setup: /'
  case "$R" in "RESULT: PASS") exit 0;; *) exit 1;; esac; }
grep -q '^RESULT:' "$A" 2>/dev/null && over

CF=$(sed -n 's/^WATCH_F=//p' "$P/meta.env" 2>/dev/null | tr -d '\r')
CD=$(sed -n 's/^WATCH_D=//p' "$P/meta.env" 2>/dev/null | tr -d '\r')
[ -n "$CF" ] && [ -n "$CD" ] || { echo "WAIT: ERROR no WATCH_F/WATCH_D in $P/meta.env (pack the run with this pack.sh)"; exit 4; }
W="$P/watch"; mkdir -p "$W"
# Lock: the PID is written into a fresh folder that is then renamed to $LK in one step, so a lock
# never exists without its PID. A lock whose PID is dead (or a PID-less one older than 60 s, left
# by an older wait.sh) is renamed away first.
LK="$W/lock"; NL=$(mktemp -d "$W/lock.new.XXXXXX") || { echo "WAIT: ERROR cannot create $W/lock.new.*"; exit 4; }
echo $$ > "$NL/pid"
if ! mv -T "$NL" "$LK" 2>/dev/null; then
  op=$(cat "$LK/pid" 2>/dev/null)
  age=$(( $(date +%s) - $(stat -c %Y "$LK" 2>/dev/null || date +%s) ))   # a PID-less lock older than 60 s is stale
  if { [ -z "$op" ] && [ "$age" -lt 60 ]; } || { [ -n "$op" ] && kill -0 "$op" 2>/dev/null; }; then rm -rf "$NL"; echo "WAIT: ERROR wait.sh $RUN already runs (pid ${op:-?}, lock $LK)"; exit 4; fi
  mv -T "$LK" "$W/lock.stale.$$" 2>/dev/null && rm -rf "$W/lock.stale.$$"
  mv -T "$NL" "$LK" 2>/dev/null || { rm -rf "$NL"; echo "WAIT: ERROR cannot take the lock $LK"; exit 4; }
fi
[ "$(cat "$LK/pid" 2>/dev/null)" = $$ ] || { echo "WAIT: ERROR lost the lock $LK to another wait.sh"; exit 4; }
trap 'rm -rf "$LK"' EXIT; trap 'exit 4' INT TERM HUP
END=$(( $(date +%s) + $(awk -v h="$HOURS" 'BEGIN {print int(h * 3600)}') ))
log() { echo "$(date +%s) $*" >> "$W/wait.log"; }

# take <code> <file>: 0 = the beacon <file> was received and checked, left in $W/<file>;
# 1 = not taken this sweep (or the deadline passed); 2 = a results.tgz arrived broken.
take() { local c="$1" f="$2" i=0 t="$W/try" out rc left
  while [ "$i" -lt 100 ]; do
    left=$(( END - $(date +%s) )); [ "$left" -gt 0 ] || return 1
    [ "$left" -gt 30 ] && left=30
    rm -rf "$t"; mkdir "$t"
    out=$(cd "$t" && HOME="$RENT/home" USERPROFILE="$RENT/home" timeout "$left" "$RPC" receive "$c-$i" 2>&1 < /dev/null); rc=$?
    if [ "$rc" = 0 ] && [ -s "$t/$f" ]; then
      case "$f" in
        *.tgz) tar -tzf "$t/$f" > /dev/null 2>&1 || { mv -f "$t/$f" "$W/$f.broken"; rm -rf "$t"; log "$f broken"; return 2; } ;;
        *) head -1 "$t/$f" | grep -q '^FIRST_FAILURE' || { log "$f at relay $i has no FIRST_FAILURE line"; rm -rf "$t"; return 1; } ;;
      esac
      mv -f "$t/$f" "$W/$f"; rm -rf "$t"; log "took $f at relay $i"; return 0
    fi
    rm -rf "$t"
    printf '%s\n' "$out" | grep -qE 'relay index [0-9]+ not found' && return 1   # end of the relay list
    if printf '%s\n' "$out" | grep -q 'could not get list of relays'; then
      log "$f relay list: $(printf '%s' "$out" | tr '\r\n' '  ' | tail -c 200)"; return 1; fi
    printf '%s\n' "$out" | grep -q 'room not ready' \
      || log "$f relay $i rc=$rc: $(printf '%s' "$out" | sed "s/$c/<code>/g" | tr '\r\n' '  ' | tail -c 200)"
    i=$((i + 1))
  done
  return 1; }

log "wait start until=$UNTIL timeout=${HOURS}h every=${EVERY}s"
while [ "$(date +%s)" -lt "$END" ]; do
  take "$CD" results.tgz; rc=$?
  if [ "$rc" = 2 ]; then
    echo "WAIT: ERROR results.tgz from the pod is broken ($W/results.tgz.broken); fetch it by hand: DAYBOX.md Collect, fetch.sh then collect.sh $RUN <code>"; exit 4
  fi
  if [ "$rc" = 0 ]; then
    mv -f "$W/results.tgz" "$P/results.tgz"
    bash "$HERE/collect.sh" "$RUN" --local > "$W/collect.out" 2>&1 || { echo "WAIT: ERROR collect.sh failed"; cat "$W/collect.out"; exit 4; }
    over
  fi
  if [ "$UNTIL" = first-failure ] && take "$CF" firstfail.txt; then
    echo "WAIT: FIRST FAILURE, run still going ($W/firstfail.txt)"; cat "$W/firstfail.txt"; exit 2
  fi
  # A payload that never reached the pod: no notify.sh runs there, so no beacon will come.
  if [ ! -s "$W/firstfail.txt" ] && grep -qE 'croc: send:|"error"' "$P/send.log" 2>/dev/null; then
    echo "WAIT: ERROR payload not delivered: pack.sh's send failed ($P/send.log); pack again under a new run name"
    tr '\r' '\n' < "$P/send.log" | grep -E 'croc: send:|"error"' | tail -2; exit 4
  fi
  left=$(( END - $(date +%s) )); [ "$left" -gt 0 ] || break
  sleep $(( left < EVERY ? left : EVERY ))
done
echo "WAIT: TIMEOUT after ${HOURS} h, no beacon from the pod (log: $W/wait.log)"; exit 3
