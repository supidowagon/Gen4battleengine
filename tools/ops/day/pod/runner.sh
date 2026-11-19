#!/usr/bin/env bash
# POD SIDE, started by setup.sh. Starts the units of out/queue.txt in order, one pytest process each.
# Start gate (pod-wide): container memory (/sys/fs/cgroup/memory.current) < GATE_GB and fewer
# than MAXP live units across every run on the pod (/work/*/out/run/*).
# Backstop: memory.current >= KILL_GB kills this run's newest live unit (its process group) and
# puts it back at the front of the queue, at most twice per unit; a third kill records rc 98
# and lists the unit in overkilled.txt.
# A unit whose process group is gone without an .rc is a crash: rc 97, listed in crashed.txt.
# Signals only process groups this runner started and that are alive; never PID 1.
# Knobs (env of setup.sh): GATE_GB (default 88), MAXP (10), SPACING s between starts (15), KILL_GB (118).
set -u
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"
GATE_GB="${GATE_GB:-88}"; MAXP="${MAXP:-10}"; SPACING="${SPACING:-15}"; KILL_GB="${KILL_GB:-118}"
cd "$O"; mkdir -p run
memgb() { echo $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )); }
okpg() { case "$1" in ''|*[!0-9]*|0|1) return 1;; esac; kill -0 -- "-$1" 2>/dev/null; }
podlive() {  # live units of every run on the pod
  local n=0 f
  for f in /work/*/out/run/*; do [ -f "$f" ] && okpg "$(cat "$f" 2>/dev/null)" && n=$((n + 1)); done
  echo "$n"
}
[ -f pending.txt ] || cp queue.txt pending.txt
[ -f ALL.start ] || date +%s > ALL.start
( while [ ! -f DONE ]; do echo "$(date +%s) $(( $(cat /sys/fs/cgroup/memory.current) / 1048576 ))" >> mem_mb.txt; sleep 15; done ) &
while :; do
  for r in run/*; do [ -e "$r" ] || continue; u=$(basename "$r")
    if [ -f "$u.rc" ]; then rm -f "$r"; echo "$(date +%s) done $u rc=$(cat "$u.rc")"
    elif ! okpg "$(cat "$r")"; then
      sleep 1; [ -f "$u.rc" ] && continue; okpg "$(cat "$r")" && continue   # finished or still starting
      echo 97 > "$u.rc"; date +%s > "$u.end"; echo "$u" >> crashed.txt; rm -f "$r"
      echo "$(date +%s) CRASH $u: process group gone without rc"
    fi
  done
  nrun=$(ls run | wc -l); npend=$(grep -c . pending.txt || true)
  [ "$nrun" -eq 0 ] && [ "$npend" -eq 0 ] && break
  mem=$(memgb)
  if [ "$mem" -ge "$KILL_GB" ] && [ "$nrun" -gt 0 ]; then
    v=$(ls -t run | head -1); pg=$(cat "run/$v")
    if okpg "$pg"; then
      kill -TERM -- "-$pg" 2>/dev/null; sleep 5; okpg "$pg" && kill -KILL -- "-$pg" 2>/dev/null; sleep 1
      nk=$(grep -c "^$v " killed.txt 2>/dev/null || true); echo "$v $mem" >> killed.txt
      if [ "${nk:-0}" -ge 2 ]; then   # third kill: no more requeues; rc 98, a failure in agg.sh
        echo 98 > "$v.rc"; date +%s > "$v.end"; echo "$v" >> overkilled.txt; rm -f "run/$v"
        echo "$(date +%s) KILLED $v at ${mem}GB, third kill: rc 98, not requeued"
      else
        { grep "^$v " queue.txt; cat pending.txt; } > p.new; mv p.new pending.txt
        rm -f "run/$v" "$v.rc" "$v.start" "$v.end" "$v.log" "$v.time"
        echo "$(date +%s) KILLED $v at ${mem}GB, requeued"
      fi
    fi
    sleep 30; continue
  fi
  if [ "$npend" -gt 0 ] && [ "$mem" -lt "$GATE_GB" ] && [ "$(podlive)" -lt "$MAXP" ]; then
    read -r u f < pending.txt; sed -i 1d pending.txt
    setsid bash "$D/bin/unit.sh" "$u" < /dev/null > /dev/null 2>&1 &
    echo "$!" > "run/$u"; echo "$(date +%s) start $u $f mem ${mem} GB pod units $(podlive)"
    sleep "$SPACING"; continue
  fi
  sleep 10
done
date +%s > ALL.end; touch DONE
