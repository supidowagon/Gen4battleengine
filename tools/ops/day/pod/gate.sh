# POD SIDE, sourced by runner.sh: the start gate's arithmetic (tests/tools/test_pod_gate.py runs it
# with a fake ps). Reads DEFW, MAXP, GATE_GB, KILL_GB, IDLE_WAIT_S from the caller.

# wgb <file>: a unit's expected peak in GB from its .w file; DEFW when missing or not a number.
wgb() { local w; w=$(cat "$1" 2>/dev/null); case "$w" in ''|*[!0-9]*) echo "$DEFW";; *) echo "$w";; esac; }

# podgate <run files>: "<live units> <reserve GB>" over the given run/<unit> files (each holds a
# process group id; its weight is ../<unit>.w), from one ps snapshot. A unit is live while its group
# has a member that is not a zombie; group ids 0 and 1 and malformed ids never count. Its reserve is
# weight - group RSS (GB = 10^9 bytes) when positive. The reserve total is rounded up.
podgate() {
  local f
  { for f in "$@"; do [ -f "$f" ] || continue
      echo "U $(cat "$f" 2>/dev/null) $(cat "${f%/run/*}/${f##*/}.w" 2>/dev/null)"; done
    ps -eo pgid=,stat=,rss= | awk '{print "P", $1, $2, $3}'; } | awk -v d="$DEFW" '
    $1 == "P" { if ($3 !~ /^Z/) live[$2] = 1; rss[$2] += $4; next }
    $1 == "U" && $2 ~ /^[0-9]+$/ && $2 > 1 { n++; g[n] = $2; w[n] = ($3 ~ /^[0-9]+$/) ? $3 : d }
    END { for (i = 1; i <= n; i++) if (live[g[i]]) { c++; r = w[i] - rss[g[i]] * 1024 / 1e9; if (r > 0) t += r }
          printf "%d %d\n", c, (t == int(t)) ? t : int(t) + 1 }'
}

# admit <live units> <reserve GB> <memory GB> <weight GB> <s waited idle>: the decision for the
# next unit, memory already under GATE_GB.
#   start            other units live and memory + reserve + weight < GATE_GB, or the pod idle and
#                    memory + weight < KILL_GB (a unit heavier than GATE_GB runs alone)
#   start-over-kill  pod idle and weight >= KILL_GB: it can never fit, it runs alone
#   start-idle-wait  pod idle, memory + weight >= KILL_GB, waited IDLE_WAIT_S: memory that no unit
#                    holds (page cache, a snapshot job) has not fallen, so start rather than wait forever
#   wait-idle        pod idle, memory + weight >= KILL_GB, not waited long enough yet
#   hold-maxp        MAXP units live
#   hold             other units live and the unit does not fit
admit() {
  local live=$1 res=$2 mem=$3 w=$4 waited=$5
  if [ "$live" -ge "$MAXP" ]; then echo hold-maxp
  elif [ "$live" -gt 0 ]; then
    if [ $((mem + res + w)) -lt "$GATE_GB" ]; then echo start; else echo hold; fi
  elif [ $((mem + w)) -lt "$KILL_GB" ]; then echo start
  elif [ "$w" -ge "$KILL_GB" ]; then echo start-over-kill
  elif [ "$waited" -ge "$IDLE_WAIT_S" ]; then echo start-idle-wait
  else echo wait-idle; fi
}
