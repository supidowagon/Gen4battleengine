#!/usr/bin/env bash
# POD SIDE, started by setup.sh. Starts the units of out/queue.txt in order, one pytest process each.
# Start gate: container memory (/sys/fs/cgroup/memory.current, all runs on the pod) < GATE_GB
# and fewer than MAXP of this run's units running.
# Backstop: memory.current >= KILL_GB kills this run's newest unit (its process group) and puts
# it back at the front of the queue. Never signals PID 1.
# Knobs (env of setup.sh): GATE_GB (default 88), MAXP (10), SPACING s between starts (15), KILL_GB (118).
set -u
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"
GATE_GB="${GATE_GB:-88}"; MAXP="${MAXP:-10}"; SPACING="${SPACING:-15}"; KILL_GB="${KILL_GB:-118}"
cd "$O"; mkdir -p run
memgb() { echo $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )); }
[ -f pending.txt ] || cp queue.txt pending.txt
[ -f ALL.start ] || date +%s > ALL.start
( while [ ! -f DONE ]; do echo "$(date +%s) $(( $(cat /sys/fs/cgroup/memory.current) / 1048576 ))" >> mem_mb.txt; sleep 15; done ) &
while :; do
  for r in run/*; do [ -e "$r" ] || continue; u=$(basename "$r")
    [ -f "$u.rc" ] && { rm -f "$r"; echo "$(date +%s) done $u rc=$(cat "$u.rc")"; }; done
  nrun=$(ls run | wc -l); npend=$(grep -c . pending.txt || true)
  [ "$nrun" -eq 0 ] && [ "$npend" -eq 0 ] && break
  mem=$(memgb)
  if [ "$mem" -ge "$KILL_GB" ] && [ "$nrun" -gt 0 ]; then
    v=$(ls -t run | head -1); pg=$(cat "run/$v")
    case "$pg" in ''|*[!0-9]*|0|1) echo "$(date +%s) bad pgid '$pg' for $v, not killed"; sleep 30; continue;; esac
    kill -TERM -- "-$pg" 2>/dev/null; sleep 5; kill -KILL -- "-$pg" 2>/dev/null; sleep 1
    { grep "^$v " queue.txt; cat pending.txt; } > p.new; mv p.new pending.txt
    rm -f "run/$v" "$v.rc" "$v.start" "$v.end" "$v.log" "$v.time"
    echo "$(date +%s) KILLED $v at ${mem}GB, requeued"; echo "$v $mem" >> killed.txt; sleep 30; continue
  fi
  if [ "$npend" -gt 0 ] && [ "$nrun" -lt "$MAXP" ] && [ "$mem" -lt "$GATE_GB" ]; then
    read -r u f < pending.txt; sed -i 1d pending.txt
    setsid bash "$D/bin/unit.sh" "$u" < /dev/null > /dev/null 2>&1 &
    echo "$!" > "run/$u"; echo "$(date +%s) start $u $f mem ${mem} GB running $((nrun + 1))"
    sleep "$SPACING"; continue
  fi
  sleep 10
done
date +%s > ALL.end; touch DONE
