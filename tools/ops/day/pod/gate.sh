# POD SIDE, sourced by runner.sh, setup.sh, status.sh, agg.sh and podcheck.sh: the start gate's
# arithmetic (tests/tools/test_pod_gate.py runs it with a fake ps) and the container memory reader.
# Functions only. Reads NO_W_FILE_GB, MAXP, GATE_GB, KILL_GB, IDLE_WAIT_S, CLAIMS, CLAIM_S,
# CLAIM_MAX_S, UNIT_MAX_S from the caller.

# cgmem <current|max|peak>: container memory in bytes. cgroup v2 first (<CGROOT>/memory.current,
# memory.max, memory.peak), else cgroup v1 (<CGROOT>/memory/memory.usage_in_bytes,
# memory.limit_in_bytes, memory.max_usage_in_bytes): some rented hosts are v1. CGROOT defaults to
# /sys/fs/cgroup. v2's memory.max reads "max" without a limit, v1's limit a value near 2^63.
# Prints nothing and returns 1 when neither file is readable; callers fail closed.
cgmem() { local r="${CGROOT:-/sys/fs/cgroup}" a b
  case "$1" in current) a=memory.current; b=memory.usage_in_bytes;; max) a=memory.max; b=memory.limit_in_bytes;;
    peak) a=memory.peak; b=memory.max_usage_in_bytes;; *) return 2;; esac
  if [ -r "$r/$a" ]; then cat "$r/$a"; elif [ -r "$r/memory/$b" ]; then cat "$r/memory/$b"; else return 1; fi; }

# unit_overdue <unit> <now>: the unit started (./<unit>.start, written by unit.sh) more than
# UNIT_MAX_S s before <now>.
unit_overdue() { local s; s=$(cat "$1.start" 2>/dev/null); case "$s" in ''|*[!0-9]*) return 1;; esac
  [ $(( $2 - s )) -gt "$UNIT_MAX_S" ]; }

# wgb <file>: a unit's expected peak in GB from its .w file; NO_W_FILE_GB (runner.sh's
# DEFAULT_WEIGHT_GB, for units of runs packed before weights) when missing or not a number.
wgb() { local w; w=$(cat "$1" 2>/dev/null); case "$w" in ''|*[!0-9]*) echo "$NO_W_FILE_GB";; *) echo "$w";; esac; }

# podgate <run files>: "<live units> <reserve GB>" over the given run/<unit> files (each holds the
# pid of the unit's setsid bash, which becomes its process group id; its weight is ../<unit>.w),
# from one ps snapshot. A unit is live while that pid exists and is not a zombie (a unit that has
# not called setsid yet) or its group has a member that is not a zombie; ids 0 and 1 and malformed
# ids never count. Its reserve is weight - group RSS (GB = 10^9 bytes) when positive. The reserve
# total is rounded up.
podgate() {
  local f
  { for f in "$@"; do [ -f "$f" ] || continue
      echo "U $(cat "$f" 2>/dev/null) $(cat "${f%/run/*}/${f##*/}.w" 2>/dev/null)"; done
    ps -eo pid=,pgid=,stat=,rss= | awk '{print "P", $1, $2, $3, $4}'; } | awk -v d="$NO_W_FILE_GB" '
    $1 == "P" { if ($4 !~ /^Z/) { live[$2] = 1; live[$3] = 1 }; rss[$3] += $5; next }
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

# pidlive <pid>: the process exists and is not a zombie.
pidlive() { case "$1" in ''|*[!0-9]*|0|1) return 1;; esac
  ps -eo pid=,stat= | awk -v p="$1" '$1 == p && $2 !~ /^Z/ {f = 1} END {exit !f}'; }

# Pod-wide claims (CLAIMS dir, default /work/.claims): a unit held CLAIM_S s gets a claim file
# <run>.<unit> = "<epoch> <weight> <runner pid>"; the oldest live claim (epoch, then name) wins and
# every other runner starts nothing while it exists. Callers hold the pod lock.
# claim_write <run> <unit> <weight> <epoch>: written through a temp file, with this shell's pid.
claim_write() { mkdir -p "$CLAIMS"; local t="$CLAIMS/.$1.$2.tmp"
  echo "$4 $3 ${CLAIM_PID:-$$}" > "$t" && mv -f "$t" "$CLAIMS/$1.$2"; }
claim_drop() { rm -f "$CLAIMS/$1.$2"; }
# claim_state <run>: "none" | "mine <unit>" | "other <run>.<unit>" for the oldest live claim;
# deletes claims whose runner pid is dead (or a zombie), malformed ones, and claims older than
# CLAIM_MAX_S (logged "<epoch> claim-expired <name> age <s> s" on stderr) on the way.
claim_state() {
  local c n e w p best="" be=""
  for c in "$CLAIMS"/*; do [ -f "$c" ] || continue; n=${c##*/}
    read -r e w p < "$c" 2>/dev/null || { rm -f "$c"; continue; }
    case "$e" in ''|*[!0-9]*) rm -f "$c"; continue;; esac
    pidlive "$p" || { rm -f "$c"; continue; }
    [ $(( ${NOW:-$(date +%s)} - e )) -gt "${CLAIM_MAX_S:-3600}" ] && { rm -f "$c"
      echo "${NOW:-$(date +%s)} claim-expired $n age $(( ${NOW:-$(date +%s)} - e )) s" >&2; continue; }
    if [ -z "$best" ] || [ "$e" -lt "$be" ] || { [ "$e" -eq "$be" ] && [[ "$n" < "$best" ]]; }; then best=$n; be=$e; fi
  done
  if [ -z "$best" ]; then echo none
  elif [ "${best%%.*}" = "$1" ]; then echo "mine ${best#*.}"
  else echo "other $best"; fi
}

# choose <live> <reserve GB> <memory GB> <s waited idle> <head hold age s> <claim: none|mine|other>
# over pending.txt (lines "<unit> <file>", weights in ./<unit>.w): which unit to start.
# Prints four fields "<start|none> <unit|-> <how> <claim|->":
#   "start <unit> <admit result> -" the head fits (admit start*)
#   "start <unit> skip-ahead -"     the head does not fit, its hold is younger than CLAIM_S, no claim
#                                   is live, and <unit> is the first later unit that fits (admit start)
#   "none - <admit result of the head> <claim|->"  nothing starts; "claim": the head has been held
#                                   CLAIM_S s and no claim is live, so the caller claims it
#   "none - claim-other -"          another run's claim is the oldest live one
# With "mine" only the head (the claimed unit) may start.
choose() {
  local live=$1 res=$2 mem=$3 waited=$4 age=$5 cl=$6 u f a w
  [ "$cl" = other ] && { echo "none - claim-other -"; return; }
  read -r u f < pending.txt; w=$(wgb "$u.w")
  a=$(admit "$live" "$res" "$mem" "$w" "$waited")
  case "$a" in start*) echo "start $u $a -"; return;; esac
  [ "$cl" = mine ] && { echo "none - $a -"; return; }
  [ "$age" -ge "$CLAIM_S" ] && { echo "none - $a claim"; return; }
  [ "$a" = hold-maxp ] && { echo "none - $a -"; return; }
  while read -r u f; do
    [ "$(admit "$live" "$res" "$mem" "$(wgb "$u.w")" "$waited")" = start ] && { echo "start $u skip-ahead -"; return; }
  done < <(tail -n +2 pending.txt)
  echo "none - $a -"
}
