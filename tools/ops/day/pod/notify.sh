#!/usr/bin/env bash
# POD SIDE, started detached by setup.sh when meta.env sets WATCH_CODE (pack.sh does). Feeds the
# PC's tools/ops/day/wait.sh, which cannot call pod exec: every 20 s it reads bin/status.sh, and
#  - at the first failure (a BAD unit, a COLLECT FAILED line, a running unit with F/E marks,
#    SETUP_FAILED, or a dead run) it writes notify/firstfail.txt (the status text) and offers it
#    with `runpodctl send --code <WATCH_CODE>f`;
#  - when the run is over (DONE=y, or the same dead state on two reads 20 s apart: setup=gone
#    runner=none, or runner=gone without DONE) it packs results.tgz with `bin/fetch.sh --pack`
#    and offers it with `runpodctl send --code <WATCH_CODE>d`.
# Each offer is repeated until a receiver takes it or WATCH_HOURS (default 24) pass; the f offer
# stops once the d offer is taken. Signals only its own send processes (timeout, and pkill of the
# pending f send by its command line, which holds this run's random code).
# Reads only this run's folder; writes notify/, notify.log, agg.txt and results.tgz in it.
set -u
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D"; . ./meta.env
[ -n "${WATCH_CODE:-}" ] || { echo "no WATCH_CODE in meta.env"; exit 0; }
RPC=/work/_bin/runpodctl; N="$D/notify"; mkdir -p "$N"
END=$(( $(date +%s) + ${WATCH_HOURS:-24} * 3600 ))
# offer <suffix> <file in $N or $D>: send it under WATCH_CODE<suffix> until taken, END, or
# (for f) the stop file. Each send attempt lasts at most 600 s.
offer() { local s="$1" f="$2"
  while [ "$(date +%s)" -lt "$END" ]; do
    [ "$s" = f ] && [ -f "$N/stop" ] && return 1
    ( cd "$(dirname "$f")" && timeout 600 "$RPC" send --code "$WATCH_CODE$s" "$(basename "$f")" ) \
      >> "$N/send_$s.log" 2>&1 < /dev/null && { date +%s > "$N/taken_$s"; echo "$(date +%s) $s taken"; return 0; }
    sleep 10
  done; echo "$(date +%s) $s not taken before WATCH_HOURS"; return 1; }
FAILED=0; DEAD=0
while :; do
  S=$(bash "$D/bin/status.sh" 2>&1); L=$(printf '%s\n' "$S" | grep -m1 '^setup=')
  over=0; dead=0
  case "$L" in
    *DONE=y*) over=1 ;;
    *"setup=gone runner=none"*|*"runner=gone"*) dead=1 ;;
  esac
  if [ "$dead" = 1 ]; then DEAD=$((DEAD + 1)); [ "$DEAD" -ge 2 ] && over=1; else DEAD=0; fi
  bad=0
  # setup.log runs under set -x: its trace of the trap also holds the text SETUP_FAILED, so only
  # the echoed line (status.sh prints it as "setup: SETUP_FAILED rc=...") counts.
  printf '%s\n' "$S" | grep -qE '^(BAD |COLLECT FAILED|running u_[0-9]+ has |setup: SETUP_FAILED rc=)' && bad=1
  [ "$over" = 1 ] && [ "$dead" = 1 ] && bad=1
  if [ "$bad" = 1 ] && [ "$FAILED" = 0 ]; then FAILED=1
    { echo "FIRST_FAILURE $(date +%s)"; printf '%s\n' "$S"; } > "$N/firstfail.txt"
    echo "$(date +%s) first failure"; offer f "$N/firstfail.txt" &
  fi
  if [ "$over" = 1 ]; then
    echo "$(date +%s) run over: $L"; bash "$D/bin/fetch.sh" --pack
    offer d "$D/results.tgz"; touch "$N/stop"
    # End a pending f offer now: only processes carrying this run's own random code match.
    pkill -TERM -f -- "send --code ${WATCH_CODE}f" 2>/dev/null; wait; exit 0
  fi
  sleep 20
done
