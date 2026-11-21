#!/usr/bin/env bash
# POD SIDE, started by setup.sh. Starts the units of out/queue.txt in order, one pytest process each.
# Start gate: container memory (gate.sh cgmem: cgroup v2 memory.current, else v1
# memory/memory.usage_in_bytes; the runner exits at once when neither is readable) < GATE_GB, fewer than MAXP live
# units across every run on the pod (/work/*/out/run/* and /work/*/*/out/run/*), fewer than
# RUNMAXP units of this run in run/, and the memory gate: memory + reserve + w < GATE_GB, where w
# is the next unit's expected peak (out/<unit>.w, written by setup.sh) and reserve covers the
# gaps (expected peak minus current RSS, process group sum, where positive) of the live units on
# the pod (every run): every gap of a unit with w >= RESERVE_FULL_GB, and for the lighter ones the
# larger of their RESERVE_TOP largest gaps and RESERVE_PCT % of their weight sum minus their RSS
# sum (gate.sh podgate). A unit without a .w file (a run packed before
# weights) counts DEFAULT_WEIGHT_GB.
# With no live unit on the pod (memory < GATE_GB still holds) a unit starts when memory + w <
# KILL_GB, so a unit heavier than GATE_GB runs alone; one with w >= KILL_GB starts alone anyway
# ("start-over-kill"), and one kept out only by memory no unit holds starts after IDLE_WAIT_S
# ("start-idle-wait"), so no unit waits forever (gate.sh admit). When the head unit does not fit,
# the first later pending unit that fits starts ("skip-ahead"), while the head's hold is younger
# than CLAIM_S. A head held CLAIM_S s gets a pod-wide claim, CLAIMS/<run>.<unit> ("<epoch> <weight>
# <runner pid>"): while the oldest live claim on the pod is another run's, this runner starts only
# a unit that fits beside it (reserve + claim weight; "claim-fit"); while it is its own, only the
# claimed head may start, as soon as it fits; the claim is removed at the start. A claim whose runner pid is dead or a zombie is deleted (gate.sh choose,
# claim_state). The check and the start happen under the pod-wide lock RUNNER_LOCK (flock), so two
# runners never pass the gate, or claim, on the same reading.
# A unit is live while its pid (run/<unit>: the pid of `setsid bash unit.sh`, which becomes the
# group id once setsid() runs; before that it sits in this runner's group) or a member of its
# process group exists and is not a zombie: PID 1 of the pod (sleep infinity) never reaps orphans,
# so finished units can linger as zombies. A background child of this non-interactive shell is no
# group leader, so setsid never forks and the pid stays the unit's.
# Backstop, checked once per loop: container memory >= KILL_GB kills this run's newest live unit
# (its process group) and puts it back at the front of the queue, at most twice per unit; a third
# kill records rc 98 and lists the unit in overkilled.txt. Each kill line names the largest of
# this run's live units and its RSS at the time of the kill.
# A unit whose process group is gone without an .rc is a crash: rc 97, listed in crashed.txt.
# DONE is written only when every unit of queue.txt has an .rc: a unit left without one goes back
# in line (lost.txt); after two requeues (kills and losses together) it gets rc 98 (overkilled.txt).
# Signals only process groups this runner started and that are alive; never PID 1.
# Knobs (env of the exec-pod body that starts setup.sh): GATE_GB (default 88), MAXP (10),
# RUNMAXP (MAXP), SPACING s between starts (4: the expected-peak reserve covers a unit's growth
# after its start, which a longer spacing used to wait out), KILL_GB (118), DEFAULT_WEIGHT_GB (12: weight of a
# unit without a .w file, a run packed before weights; those runs are mostly fast-tier chunks of
# measured files and 12 GB is the 90th percentile of 7183 measured units),
# RESERVE_TOP (2: light units reserved at once; 18 mixed units under a memory-only gate peaked at
# 185-229 GB, 0.6-0.73 x their weight sum), RESERVE_FULL_GB (40: units this heavy keep their whole
# gap reserved), RESERVE_PCT (75: the light reserve is at least this % of the light units' weight
# sum minus their RSS; 0.73 measured worst, and chunks of one file grow in step; the three
# RESERVE_ knobs must be whole numbers, else the runner stops with an ERROR line), IDLE_WAIT_S (300), RUNNER_LOCK (/work/.runner.lock), CLAIM_S (120), CLAIMS (/work/.claims),
# CLAIM_MAX_S (3600: a claim whose pid is alive but no runner, a reused pid, is deleted once
# older, "claim-expired"; a dead runner's claim goes at once, "claim-dead"; a live runner's claim never expires
# by time; this runner's own lost claim lets its head be gated again with a fresh hold timer),
# UNIT_MAX_S (21600: measured unit walls reach 10080 s on uncontended pods; a unit running longer
# is killed with rc 96, listed in timedout.txt, a failure), WORK (/work: the folder whose runs the
# pod-wide gate counts), WARMUP, WARMUP_S (900).
# Warm-up: when queue.txt has a tests/engine unit, one pytest process over the node id WARMUP
# (default tests/engine/test_step_basic.py::test_miss_deals_no_damage, which compiles step; none
# turns it off) runs alone first (unit.sh --warmup, files in out/warmup/), so step is compiled once
# into the pod-wide compile cache instead of by every early unit at once. It waits for the memory
# gate, is killed after WARMUP_S s, and its result never fails the run; a restarted runner skips it
# (warmup/started). It also runs when the cache already holds step: then it costs one cache load
# (about 17 s) and its tests, cheaper than deciding whether every program it compiles is cached.
set -u
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"
GATE_GB="${GATE_GB:-88}"; MAXP="${MAXP:-10}"; RUNMAXP="${RUNMAXP:-$MAXP}"; SPACING="${SPACING:-4}"; KILL_GB="${KILL_GB:-118}"
NO_W_FILE_GB="${DEFAULT_WEIGHT_GB:-12}"; IDLE_WAIT_S="${IDLE_WAIT_S:-300}"; LOCK="${RUNNER_LOCK:-/work/.runner.lock}"
CLAIMS="${CLAIMS:-/work/.claims}"; CLAIM_S="${CLAIM_S:-120}"; CLAIM_MAX_S="${CLAIM_MAX_S:-3600}"; RUN=$(basename "$D")
UNIT_MAX_S="${UNIT_MAX_S:-21600}"; WORK="${WORK:-/work}"; RESERVE_TOP="${RESERVE_TOP:-2}"; RESERVE_FULL_GB="${RESERVE_FULL_GB:-40}"; RESERVE_PCT="${RESERVE_PCT:-75}"
cd "$O"; mkdir -p run
. "$D/bin/gate.sh"   # cgmem, wgb, podgate, admit, choose, claims, unit_overdue
memgb() { local b; b=$(cgmem current) || return 1; echo $(( b / 1000000000 )); }
nomem() { echo "$(date +%s) ERROR: no cgroup memory file (v2 memory.current or v1 memory/memory.usage_in_bytes); runner stops"; exit 1; }
memgb > /dev/null || nomem
reserve_knobs || { echo "$(date +%s) ERROR: bad RESERVE_ knob; runner stops"; exit 1; }
for n in RESERVE_TOP RESERVE_FULL_GB RESERVE_PCT; do   # run.env gets their effective values, defaults too
  grep -q "^export $n=" "$D/run.env" 2>/dev/null || echo "export $n='${!n}'" >> "$D/run.env"; done
# Temp files this runner writes (p.new, run/.<unit>.tmp, its claim temp files) go on any exit.
trap 'rm -f p.new run/.*.tmp "$CLAIMS/.$RUN".*.tmp' EXIT; trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP
okpg() { case "$1" in ''|*[!0-9]*|0|1) return 1;; esac   # the pid (setsid not called yet) or a group
  ps -eo pid=,pgid=,stat= | awk -v g="$1" '($1 == g || $2 == g) && $3 !~ /^Z/ {f = 1} END {exit !f}'; }   # member, not a zombie
largest() {  # this run's live unit with the largest RSS (summed over its process group)
  local r g m best=none bm=-1
  for r in run/*; do [ -f "$r" ] || continue; g=$(cat "$r" 2>/dev/null); okpg "$g" || continue
    m=$(ps -eo pgid=,rss= | awk -v g="$g" '$1 == g {s += $2} END {print int(s / 1024)}')
    [ "$m" -gt "$bm" ] && { bm=$m; best="$(basename "$r") ${m} MB"; }; done
  echo "$best"
}
[ -f pending.txt ] || cp queue.txt pending.txt
[ -f ALL.start ] || date +%s > ALL.start
( while [ ! -f DONE ]; do b=$(cgmem current) && echo "$(date +%s) $(( b / 1048576 ))" >> mem_mb.txt; sleep 15; done ) &
WARMUP="${WARMUP:-tests/engine/test_step_basic.py::test_miss_deals_no_damage}"
if [ "$WARMUP" != none ] && [ ! -f warmup/started ] && grep -q ' tests/engine/' queue.txt; then
  while m=$(memgb) || nomem; [ "$m" -ge "$GATE_GB" ]; do sleep 10; done
  mkdir -p warmup; date +%s > warmup/started   # a restarted runner never starts a second one
  setsid bash "$D/bin/unit.sh" --warmup "$WARMUP" < /dev/null > /dev/null 2>&1 8>&- &
  wpg=$!; sleep 2; echo "$(date +%s) warm-up start $WARMUP"; wend=$(( $(date +%s) + ${WARMUP_S:-900} ))
  while okpg "$wpg" && [ "$(date +%s)" -lt "$wend" ]; do sleep 5; done
  if okpg "$wpg"; then kill -TERM -- "-$wpg" 2>/dev/null; sleep 5; okpg "$wpg" && kill -KILL -- "-$wpg" 2>/dev/null
    mkdir -p warmup; echo 124 > warmup/warmup.rc; echo "$(date +%s) warm-up killed after ${WARMUP_S:-900} s"
  else echo "$(date +%s) warm-up end rc=$(cat warmup/warmup.rc 2>/dev/null || echo none)"; fi
fi
# Pod-wide gate lock, opened after the background jobs above so none of them holds it; units
# start with it closed (8>&-), so a lock held by a dead runner dies with that runner.
exec 8>> "$LOCK"
while :; do
  for r in run/*; do [ -e "$r" ] || continue; u=$(basename "$r")
    if [ -f "$u.rc" ]; then rm -f "$r"; echo "$(date +%s) done $u rc=$(cat "$u.rc")"
    elif unit_overdue "$u" "$(date +%s)" && okpg "$(cat "$r")"; then
      pg=$(cat "$r"); kill -TERM -- "-$pg" 2>/dev/null; sleep 5; okpg "$pg" && kill -KILL -- "-$pg" 2>/dev/null; sleep 1
      echo 96 > "$u.rc"; date +%s > "$u.end"; echo "$u" >> timedout.txt; rm -f "$r"
      echo "$(date +%s) TIMEOUT $u: over UNIT_MAX_S ${UNIT_MAX_S} s, killed: rc 96"
    elif ! okpg "$(cat "$r")"; then
      sleep 1; [ -f "$u.rc" ] && continue; okpg "$(cat "$r")" && continue   # finished or still starting
      echo 97 > "$u.rc"; date +%s > "$u.end"; echo "$u" >> crashed.txt; rm -f "$r"
      echo "$(date +%s) CRASH $u: process group gone without rc"
    fi
  done
  nrun=$(ls run | wc -l); npend=$(grep -c . pending.txt || true)
  if [ "$nrun" -eq 0 ] && [ "$npend" -eq 0 ]; then
    # DONE only when every unit of queue.txt has an .rc. A unit without one (a lost requeue) goes
    # back in line; its requeues (kills and losses) count together, and after two it gets rc 98.
    lost=0
    for v in $(cut -d' ' -f1 queue.txt); do [ -f "$v.rc" ] && continue; lost=1
      nk=$(cat killed.txt lost.txt 2>/dev/null | grep -c "^$v " || true); echo "$v -" >> lost.txt
      if [ "${nk:-0}" -ge 2 ]; then
        echo 98 > "$v.rc"; date +%s > "$v.end"; echo "$v" >> overkilled.txt
        echo "$(date +%s) LOST $v: no result after two requeues: rc 98"
      else grep "^$v " queue.txt >> pending.txt; echo "$(date +%s) LOST $v: no result, requeued"; fi
    done
    [ "$lost" = 0 ] && break; continue
  fi
  mem=$(memgb) || nomem; [ "$mem" -ge "$GATE_GB" ] && idle0=""   # the idle wait counts only time under GATE_GB
  if [ "$mem" -ge "$KILL_GB" ] && [ "$nrun" -gt 0 ]; then
    v=$(ls -t run | head -1); pg=$(cat "run/$v")
    if okpg "$pg"; then
      big=$(largest)   # measured before the kill
      kill -TERM -- "-$pg" 2>/dev/null; sleep 5; okpg "$pg" && kill -KILL -- "-$pg" 2>/dev/null; sleep 1
      nk=$(cat killed.txt lost.txt 2>/dev/null | grep -c "^$v " || true); echo "$v $mem" >> killed.txt
      if [ "${nk:-0}" -ge 2 ]; then   # third kill: no more requeues; rc 98, a failure in agg.sh
        echo 98 > "$v.rc"; date +%s > "$v.end"; echo "$v" >> overkilled.txt; rm -f "run/$v"
        echo "$(date +%s) KILLED $v at ${mem}GB, third kill: rc 98, not requeued; largest of this run's live units then: $big"
      else
        { grep "^$v " queue.txt; cat pending.txt; } > p.new; mv p.new pending.txt
        rm -f "run/$v" "$v.rc" "$v.start" "$v.end" "$v.log" "$v.time"
        echo "$(date +%s) KILLED $v at ${mem}GB, requeued; largest of this run's live units then: $big"
      fi
    fi
    sleep 30; continue
  fi
  if [ "$npend" -gt 0 ] && [ "$mem" -lt "$GATE_GB" ] && [ "$nrun" -lt "$RUNMAXP" ]; then
    flock -w 120 8 || { echo "$(date +%s) gate lock $LOCK still held after 120 s"; continue; }
    now=$(date +%s); mem=$(memgb) || { flock -u 8; nomem; }; read -r plive pres <<< "$(podgate "$WORK"/*/out/run/* "$WORK"/*/*/out/run/*)"
    read -r u f < pending.txt; w=$(wgb "$u.w")
    [ "$plive" -eq 0 ] || idle0=""; [ "$plive" -eq 0 ] && [ -z "${idle0:-}" ] && idle0=$now
    waited=$(( now - ${idle0:-$now} ))
    [ "${h0u:-}" = "$u" ] || { h0u=$u; h0t=$now; }   # the head's hold age counts from its first hold
    read -r cs cu cw <<< "$(claim_state "$RUN")"
    [ "$cs" = mine ] && [ "$cu" != "$u" ] && { claim_drop "$RUN" "$cu"; cs=none; }   # stale: not the head
    [ -n "${claimed:-}" ] && [ "$cs" != mine ] && { claimed=""; h0t=$now; }   # claim expired: fresh hold
    if [ "$mem" -lt "$GATE_GB" ]; then read -r d su a cl <<< "$(choose "$plive" "$pres" "$mem" "$waited" $(( now - h0t )) "$cs" "${cw:-0}")"
    else d=none; a=hold-mem; cl=""; idle0=""; fi
    if [ "$d" = start ]; then
      grep -v "^$su " pending.txt > p.new; mv p.new pending.txt; sf=$(grep "^$su " queue.txt | cut -d' ' -f2); sw=$(wgb "$su.w")
      setsid bash "$D/bin/unit.sh" "$su" < /dev/null > /dev/null 2>&1 8>&- &
      echo "$!" > "run/.$su.tmp"; mv -f "run/.$su.tmp" "run/$su"   # other runners never read a half-written id
      [ "$su" = "$u" ] && { claim_drop "$RUN" "$u"; claimed=""; idle0=""; held=""; h0u=""; }
      flock -u 8
      echo "$now start $su $sf weight ${sw} GB mem ${mem} GB reserve ${pres} GB pod units $((plive + 1))$([ "$a" = start ] || echo " ($a)")"
      sleep "$SPACING"; continue
    fi
    [ "$cl" = claim ] && [ "$cs" = none ] && { claim_write "$RUN" "$u" "$w" "$now"; claimed=$u
      echo "$now claim $u: held $(( now - h0t )) s, weight ${w} GB; other new-script runners start nothing until it starts"; }
    flock -u 8
    ho=""; [ "$cs" = other ] && ho=$cu   # a new claim owner reprints the hold line
    [ "${held:-}" != "$u $a $ho" ] && { held="$u $a $ho"
      echo "$now $a $u: mem ${mem} + reserve ${pres} + weight ${w} GB; GATE_GB ${GATE_GB} KILL_GB ${KILL_GB}, ${plive} units live$([ "$cs" = other ] && echo ", claim $cu is older")"; }
  fi
  sleep 10
done
date +%s > ALL.end; touch DONE
# Network volume (only when /vol, or CACHE_VOL, is a mountpoint): publish the pod's new compile cache entries in
# the background (bin/cache_sync.sh publish, log /work/<run>/cache_sync.log), after DONE so neither
# DONE nor the beacon waits for it. Without /vol nothing runs.
if mountpoint -q "${CACHE_VOL:-/vol}" 2>/dev/null; then
  CACHE_PY="$D/repo/.venv/bin/python" setsid nohup bash "$D/bin/cache_sync.sh" publish > "$D/cache_sync.log" 2>&1 < /dev/null 8>&- &
fi
