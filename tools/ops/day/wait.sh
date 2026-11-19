#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: wait.sh <run> [--until done|first-failure] [--timeout <hours>] [--every <s>]
# Blocks until the pod run <run> (sent by pack.sh, started by its exec-pod body) is over or, with
# --until first-failure, has its first failure. Needs no pod exec and no model turn: the pod's
# bin/notify.sh offers two beacons with `runpodctl send --code <WATCH_CODE><suffix>` (WATCH_CODE
# from $RENT/out/<run>/meta.env, drawn by pack.sh): `f` = firstfail.txt (status.sh text at the
# first failure), `d` = results.tgz (fetch.sh --pack, once the run is over). A croc code ends in
# the relay index the sender picked, so every --every seconds (default 60) this tries
# `runpodctl receive <WATCH_CODE><suffix>-<i>` for i = 0, 1, ... until the relay list ends
# ("relay index <i> not found"); a wrong index answers "room not ready" in under a second.
# Done: results.tgz is unpacked by collect.sh --local into $RENT/out/<run>/results/.
# Prints one line WAIT: <state>, then the status text or the summary of agg.txt.
# Exit: 0 run over, RESULT: PASS | 1 run over, RESULT: FAIL | 2 first failure (run still going;
# wait.sh <run> again for the end) | 3 timeout (default 8 h) | 4 usage or setup error.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"
RUN="${1:-}"; [ -n "$RUN" ] || { echo "usage: wait.sh <run> [--until done|first-failure] [--timeout <hours>] [--every <s>]"; exit 4; }
check_run_name "$RUN" || exit 4; shift
UNTIL=done; HOURS=8; EVERY=60
while [ $# -gt 0 ]; do case "$1" in
  --until) UNTIL="${2:-}"; shift 2 ;;
  --timeout) HOURS="${2:-}"; shift 2 ;;
  --every) EVERY="${2:-}"; shift 2 ;;
  *) echo "WAIT: ERROR unknown argument $1"; exit 4 ;; esac; done
case "$UNTIL" in done|first-failure) ;; *) echo "WAIT: ERROR --until done|first-failure"; exit 4;; esac
echo "$HOURS" | grep -qE '^[0-9]+(\.[0-9]+)?$' || { echo "WAIT: ERROR --timeout <hours>"; exit 4; }
echo "$EVERY" | grep -qE '^[1-9][0-9]*$' || { echo "WAIT: ERROR --every <seconds>"; exit 4; }
P="$RENT/out/$RUN"
CODE=$(sed -n 's/^WATCH_CODE=//p' "$P/meta.env" 2>/dev/null | tr -d '\r')
[ -n "$CODE" ] || { echo "WAIT: ERROR no WATCH_CODE in $P/meta.env (pack the run with this pack.sh)"; exit 4; }
W="$P/watch"; mkdir -p "$W"
END=$(( $(date +%s) + $(awk -v h="$HOURS" 'BEGIN {print int(h * 3600)}') ))

# take <suffix> <file>: 0 when the beacon <file> was received and checked; it is left in $W/<file>.
take() { local s="$1" f="$2" i=0 t="$W/try" out
  while [ "$i" -lt 100 ]; do
    rm -rf "$t"; mkdir "$t"
    out=$(cd "$t" && HOME="$RENT/home" USERPROFILE="$RENT/home" timeout 300 "$RPC" receive "$CODE$s-$i" 2>&1 < /dev/null); rc=$?
    if [ "$rc" = 0 ] && [ -s "$t/$f" ]; then
      case "$f" in
        *.tgz) tar -tzf "$t/$f" > /dev/null 2>&1 || { echo "$(date +%s) $f broken, retrying" >> "$W/wait.log"; i=$((i + 1)); continue; } ;;
        *) head -1 "$t/$f" | grep -q '^FIRST_FAILURE' || { i=$((i + 1)); continue; } ;;
      esac
      mv -f "$t/$f" "$W/$f"; rm -rf "$t"; echo "$(date +%s) took $s at relay $i" >> "$W/wait.log"; return 0
    fi
    echo "$out" | grep -qE 'relay index [0-9]+ not found' && break   # past the last relay
    i=$((i + 1))
  done
  rm -rf "$t"; return 1; }

echo "$(date +%s) wait start until=$UNTIL timeout=${HOURS}h" >> "$W/wait.log"
while [ "$(date +%s)" -lt "$END" ]; do
  if take d results.tgz; then
    mv -f "$W/results.tgz" "$P/results.tgz"
    bash "$HERE/collect.sh" "$RUN" --local > "$W/collect.out" 2>&1 || { echo "WAIT: ERROR collect.sh failed"; cat "$W/collect.out"; exit 4; }
    A="$P/results/agg.txt"; R=$(grep '^RESULT:' "$A" | tail -1)
    echo "WAIT: RUN OVER, $R (full report: $A)"
    # The summary: per-file table, collection failures, crashed or killed units, wall and peak,
    # failures; the per-unit list, slowest tests and skips stay in the file.
    awk '/^--- per unit/ || /^--- slowest/ || /^--- skips/ {skip = 1; next} /^--- / {skip = 0} !skip' "$A"
    printf '%s\n' "--- setup (last lines)"; tail -3 "$P/results/setup.log" 2>/dev/null
    case "$R" in "RESULT: PASS") exit 0;; *) exit 1;; esac
  fi
  if [ "$UNTIL" = first-failure ] && take f firstfail.txt; then
    echo "WAIT: FIRST FAILURE, run still going ($W/firstfail.txt)"; cat "$W/firstfail.txt"; exit 2
  fi
  sleep "$EVERY"
done
echo "WAIT: TIMEOUT after ${HOURS} h, no beacon from the pod (log: $W/wait.log)"; exit 3
